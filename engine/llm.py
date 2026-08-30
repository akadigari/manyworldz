"""The one door to the Anthropic API.

Every call is cached to disk (same question -> free repeat) and metered
against a hard budget. When the budget is gone, the engine stops asking.
It never quietly keeps spending.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

CACHE_DIR = config.CACHE / "llm"     # saved copies of past answers
# Running total of what we have spent this month. Overridable so two
# loops that share this engine do not share one budget: the Kalshi run
# (four times a day) and the FutureEval tournament (every 5 minutes) both
# metered into one file, so Kalshi spending ate the tournament's
# headroom, and hitting the cap stops a whole tournament cycle, which is
# a hard zero on every open question in the round. Each workflow now
# points at its own meter.
SPEND_FILE = Path(os.environ.get("MANYWORLDZ_SPEND_FILE")
                  or config.DATA / "spend.json")

# Rough price per 1 million "tokens" (roughly, chunks of a word) the model
# reads (input) or writes (output). Used only to estimate spending so we
# can enforce the budget cap, not an exact invoice.
_PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-fable-5": (10.00, 50.00),
}
_DEFAULT_PRICE = (10.00, 50.00)  # unknown model -> assume frontier prices (the safe guess)


def resolve_model(name: str | None) -> str:
    """Turn a friendly name ("haiku", "fable") into a full model ID.

    Full IDs pass straight through, so any model the API serves works,
    including ones released after this file was written.
    """
    name = name or config.ENGINE_MODEL
    return config.MODELS.get(name.lower().strip(), name)


def _extract_text(msg) -> str:
    """Pull the plain text out of an API response, safely.

    Two quirks handled here: reasoning models (like Fable) may include
    "thinking" blocks we don't want, and their safety layer can decline a
    request entirely (stop_reason "refusal"). In that case we return ""
    so the crowd counts it as one unusable answer and moves on. Nothing
    is ever invented.
    """
    if getattr(msg, "stop_reason", None) == "refusal":
        return ""
    return "".join(b.text for b in msg.content if b.type == "text")


def _call_api(prompt: str, model: str, max_tokens: int):
    """The only function that really talks to the API. Split out so tests
    can replace it. Reads ANTHROPIC_API_KEY from the environment: anyone's
    key works; nothing is hardcoded."""
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text(msg), msg.usage.input_tokens, msg.usage.output_tokens


_SPEND_LOCK = threading.Lock()


def _this_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _read_spend_state() -> dict:
    """The current month's spend window.

    The budget is a monthly allowance, not a lifetime total: a
    FutureEval season runs 300 to 500 questions over four months, and
    a lifetime cap would silence the bot partway through while every
    unanswered question scored zero. A state file from an earlier
    month reads as a fresh window.
    """
    state = {"calls": 0, "est_usd": 0.0, "month": _this_month()}
    if SPEND_FILE.exists():
        try:
            saved = json.loads(SPEND_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return state
        if saved.get("month") == state["month"]:
            saved.setdefault("calls", 0)
            saved.setdefault("est_usd", 0.0)
            return saved
    return state


def spent_usd() -> float:
    """Estimated API spend inside the current month's window, in
    dollars. 0.0 if nothing has been spent yet this month."""
    return _read_spend_state().get("est_usd", 0.0)


def _reserve_spend(model: str, est_tokens_in: int, max_tokens_out: int) -> float:
    """Atomically claim a call's worst-case cost against the budget.

    Raises the budget RuntimeError if the month's window is already at
    the cap. Returns the dollars reserved so the caller can settle the
    difference once the real token counts are known.
    """
    price_in, price_out = _PRICES.get(model.split("/")[-1],
                                      _PRICES.get(model, _DEFAULT_PRICE))
    cost = est_tokens_in * price_in / 1e6 + max_tokens_out * price_out / 1e6
    with _SPEND_LOCK:
        state = _read_spend_state()
        if state["est_usd"] >= config.ENGINE_BUDGET_USD:
            raise RuntimeError(
                f"engine budget cap hit (${config.ENGINE_BUDGET_USD:.2f}): "
                "raise ENGINE_BUDGET_USD in config.py to keep going")
        state["est_usd"] = round(state["est_usd"] + cost, 6)
        SPEND_FILE.parent.mkdir(parents=True, exist_ok=True)
        SPEND_FILE.write_text(json.dumps(state))
    return cost


def _adjust_spend(delta_usd: float, count_call: bool) -> None:
    """Settle a reservation: apply the actual-minus-reserved difference
    (negative releases money) and count the call if one happened."""
    with _SPEND_LOCK:
        state = _read_spend_state()
        state["est_usd"] = round(max(state["est_usd"] + delta_usd, 0.0), 6)
        if count_call:
            state["calls"] += 1
        SPEND_FILE.parent.mkdir(parents=True, exist_ok=True)
        SPEND_FILE.write_text(json.dumps(state))


def _openrouter_model(model: str) -> str:
    """Map a resolved model ID onto its OpenRouter slug.

    Overridable with OPENROUTER_MODEL_MAP, a JSON dict from our model
    IDs to OpenRouter slugs, because slug spellings are OpenRouter's
    to change: check openrouter.ai/models when the donated key arrives
    and set the map if the default "anthropic/<id>" guess is wrong.
    """
    override = os.environ.get("OPENROUTER_MODEL_MAP", "")
    if override:
        try:
            mapped = json.loads(override).get(model)
            if mapped:
                return mapped
        except json.JSONDecodeError:
            pass
    return model if "/" in model else f"anthropic/{model}"


def _call_openrouter(prompt: str, model: str, max_tokens: int):
    """Metaculus's donated credits arrive as an OpenRouter key, so when
    OPENROUTER_API_KEY is set, calls route through OpenRouter's
    OpenAI-compatible endpoint instead of the Anthropic API. Same
    return shape as _call_api; split out so tests can replace it."""
    import requests
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={"model": model, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    usage = data.get("usage") or {}
    return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def ask(prompt: str, model: str | None = None, max_tokens: int = 400) -> str:
    """Ask the model one question and return its plain-text answer.

    The same prompt+model combo is only ever paid for once. Repeat calls
    return the cached answer instantly and for free. Before any new call,
    this checks the running total against ENGINE_BUDGET_USD and refuses to
    call the API at all once the budget is used up.
    """
    model = resolve_model(model)
    key = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())["text"]
        except (json.JSONDecodeError, KeyError, OSError):
            pass    # a truncated cache entry is a miss, not a permanent error

    # Reserve the call's worst-case cost under the lock BEFORE calling.
    # Checking the total and calling afterwards was a race: N parallel
    # seats could all pass the check together and overshoot the budget
    # by N-1 calls. The reservation is adjusted to the real cost after
    # the call returns, and released entirely if the call raises.
    est_in = max(len(prompt) // 4, 1)
    reserved = _reserve_spend(model, est_in, max_tokens)

    try:
        if os.environ.get("OPENROUTER_API_KEY"):
            or_model = _openrouter_model(model)
            text, tokens_in, tokens_out = _call_openrouter(prompt, or_model, max_tokens)
            model = or_model
        else:
            text, tokens_in, tokens_out = _call_api(prompt, model, max_tokens)
    except Exception:
        _adjust_spend(-reserved, count_call=False)
        raise
    price_in, price_out = _PRICES.get(model.split("/")[-1],
                                      _PRICES.get(model, _DEFAULT_PRICE))
    actual = tokens_in * price_in / 1e6 + tokens_out * price_out / 1e6
    _adjust_spend(actual - reserved, count_call=True)
    # temp file + os.replace so a killed process can never leave a
    # half-written entry that poisons this prompt forever
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(json.dumps({"text": text}))
    os.replace(tmp, cache_file)
    return text
