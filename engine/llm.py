"""The one door to the Anthropic API.

Every call is cached to disk (same question -> free repeat) and metered
against a hard budget. When the budget is gone, the engine stops asking.
It never quietly keeps spending.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

CACHE_DIR = config.CACHE / "llm"     # saved copies of past answers
SPEND_FILE = config.DATA / "spend.json"   # running total of what we've spent

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


def spent_usd() -> float:
    """How much money we estimate has been spent on API calls so far.

    Reads the running total from disk and returns it in dollars. Returns
    0.0 if nothing has been spent yet.
    """
    if SPEND_FILE.exists():
        return json.loads(SPEND_FILE.read_text()).get("est_usd", 0.0)
    return 0.0


def _record_spend(model: str, tokens_in: int, tokens_out: int) -> None:
    """Add one API call's estimated cost to the running total saved on disk.

    Turns the token counts into a dollar estimate using the rough prices
    above, then updates the running total so spent_usd() can read it back
    next time.
    """
    price_in, price_out = _PRICES.get(model, _DEFAULT_PRICE)
    cost = tokens_in * price_in / 1e6 + tokens_out * price_out / 1e6
    state = {"calls": 0, "est_usd": 0.0}
    if SPEND_FILE.exists():
        state = json.loads(SPEND_FILE.read_text())
    state["calls"] += 1
    state["est_usd"] = round(state["est_usd"] + cost, 6)
    SPEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPEND_FILE.write_text(json.dumps(state))


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
        return json.loads(cache_file.read_text())["text"]

    if spent_usd() >= config.ENGINE_BUDGET_USD:
        raise RuntimeError(
            f"engine budget cap hit (${config.ENGINE_BUDGET_USD:.2f}): "
            "raise ENGINE_BUDGET_USD in config.py to keep going")

    text, tokens_in, tokens_out = _call_api(prompt, model, max_tokens)
    _record_spend(model, tokens_in, tokens_out)
    cache_file.write_text(json.dumps({"text": text}))
    return text
