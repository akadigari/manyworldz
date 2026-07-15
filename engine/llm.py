"""The one door to the Anthropic API.

Every call is cached to disk (same question -> free repeat) and metered
against a hard budget. When the budget is gone, the engine stops asking —
it never quietly keeps spending.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

CACHE_DIR = config.CACHE / "llm"
SPEND_FILE = config.DATA / "spend.json"

# rough $ per 1M tokens (input, output) — used only for the safety meter
_PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
}
_DEFAULT_PRICE = (5.00, 25.00)  # unknown model -> assume expensive (safe side)


def _call_api(prompt: str, model: str, max_tokens: int):
    """The only function that really talks to the API. Split out so tests
    can replace it."""
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    return text, msg.usage.input_tokens, msg.usage.output_tokens


def spent_usd() -> float:
    if SPEND_FILE.exists():
        return json.loads(SPEND_FILE.read_text()).get("est_usd", 0.0)
    return 0.0


def _record_spend(model: str, tokens_in: int, tokens_out: int) -> None:
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
    """Ask the model one question. Cached forever; budget-capped."""
    model = model or config.ENGINE_MODEL
    key = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())["text"]

    if spent_usd() >= config.ENGINE_BUDGET_USD:
        raise RuntimeError(
            f"engine budget cap hit (${config.ENGINE_BUDGET_USD:.2f}) — "
            "raise ENGINE_BUDGET_USD in config.py to keep going")

    text, tokens_in, tokens_out = _call_api(prompt, model, max_tokens)
    _record_spend(model, tokens_in, tokens_out)
    cache_file.write_text(json.dumps({"text": text}))
    return text
