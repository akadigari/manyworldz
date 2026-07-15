# Agamotto M1 — The Engine, Live — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Dr Strange machine, running live: a crowd of AI agents that votes on (or simulates futures for) real open Kalshi non-sports markets, answers what-if injections, and logs paper picks with CLV grading — tonight, not in October.

**Architecture:** Pure Python on top of the M0 codebase. `engine/llm.py` is the one place that talks to the Anthropic API (disk-cached, budget-capped). `adapters/kalshi_events.py` turns Kalshi's public API into simple "market cards." `engine/personas.py` builds the crowd; `engine/swarm.py` collects votes or simulated futures and folds them into a consensus; `engine/whatif.py` re-runs the crowd with a fact forced true. `ledger.py` logs paper picks and grades them (settlement + CLV). `run.py` wires one live cycle. Every LLM-touching function takes an `ask_fn` parameter so tests inject canned answers — the suite stays 100% offline and key-free.

**Tech Stack:** Python 3.11+ (existing venv), `anthropic`, `requests` (add to requirements), `pytest`. Kalshi public API (read-only, no key).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-14-agamotto-design.md` — resequenced Milestones: M1 = engine live; learning loop is M2 (NOT in this plan: no learn.py, no memory.py, no bench, no calibrate.py).
- Plain, simple English in comments/docstrings/output — high-school level, no jargon (owner rule).
- Commits as `akadigari <arkadigari@gmail.com>`; NEVER any AI co-author trailer.
- Dates/times as plain ISO strings end-to-end; determinism via `config.SEED = 14000605`.
- **Tests are offline and key-free:** no network, no `ANTHROPIC_API_KEY`, no `data/` access. LLM calls injected via `ask_fn`; HTTP parsed from fixtures.
- **Hard budget cap:** cumulative engine spend tracked in `data/spend.json`; hard stop at `ENGINE_BUDGET_USD = 10.00`. Never fabricate a vote — an unparseable/failed answer is skipped and counted.
- Default crowd model `claude-haiku-4-5` (spec's cost decision; "go harder" is config-only).
- Kalshi: read-only public API, non-sports only (`category != "Sports"`), paper picks only. MD-legal lane per spec.
- Market prices are in **cents (integers 1-99)**; the API sometimes returns dollar-strings — parse both (known venue quirk).
- README/public text: no MiroFish mentions, not framed as a betting product.

---

### Task 1: LLM client — cached, budget-capped `ask()`

**Files:**
- Create: `engine/__init__.py` (empty), `engine/llm.py`
- Modify: `config.py` (append engine knobs), `requirements.txt` (add `requests>=2.31`)
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `config.CACHE`, `config.DATA`
- Produces (later tasks rely on these exact names):
  `ask(prompt: str, model: str | None = None, max_tokens: int = 400) -> str`
  — returns the model's text; disk-cached by sha256(model+prompt); raises
  `RuntimeError` when the budget cap is hit. `spent_usd() -> float`.
  Config names: `ENGINE_MODEL`, `ENGINE_N_AGENTS`, `SIM_ROLLOUTS_K`,
  `DELIBERATION`, `MIN_EDGE_CENTS`, `FEE_BUFFER_CENTS`, `MARKETS_PER_RUN`,
  `ENGINE_BUDGET_USD`, `EXCLUDED_CATEGORIES`.

- [ ] **Step 1: Append to `config.py`**

```python
# ---- M1 engine knobs (the "go harder" dials) ----

ENGINE_MODEL = "claude-haiku-4-5"  # cheap crowd voices; raise tier here to go harder
ENGINE_N_AGENTS = 8       # agents per market
SIM_ROLLOUTS_K = 5        # futures each agent imagines in simulate mode
DELIBERATION = False      # one round of agents seeing each other's takes
MIN_EDGE_CENTS = 10       # crowd must differ from market by this much...
FEE_BUFFER_CENTS = 3      # ...plus this cushion for fees/spread, to log a pick
MARKETS_PER_RUN = 5       # markets the crowd votes on per cycle
ENGINE_BUDGET_USD = 10.00 # hard stop for cumulative engine spend
EXCLUDED_CATEGORIES = {"Sports"}  # non-sports only (MD-legal lane)
```

Add `requests>=2.31` to `requirements.txt` and run `venv/bin/pip install -r requirements.txt`.

- [ ] **Step 2: Write the failing test** — `tests/test_llm.py`

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import llm


def test_cache_returns_same_answer_without_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm")
    monkeypatch.setattr(llm, "SPEND_FILE", tmp_path / "spend.json")
    calls = []

    def fake_api(prompt, model, max_tokens):
        calls.append(prompt)
        return "hello", 10, 5  # text, input tokens, output tokens

    monkeypatch.setattr(llm, "_call_api", fake_api)
    a = llm.ask("What is 2+2?")
    b = llm.ask("What is 2+2?")
    assert a == b == "hello"
    assert len(calls) == 1  # second answer came from disk, not the API


def test_budget_cap_halts_before_calling(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm")
    monkeypatch.setattr(llm, "SPEND_FILE", tmp_path / "spend.json")
    monkeypatch.setattr(llm, "_call_api", lambda p, m, t: ("x", 1000, 1000))
    import config
    monkeypatch.setattr(config, "ENGINE_BUDGET_USD", 0.0)
    with pytest.raises(RuntimeError, match="budget"):
        llm.ask("anything new")


def test_spend_meter_accumulates(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm")
    monkeypatch.setattr(llm, "SPEND_FILE", tmp_path / "spend.json")
    monkeypatch.setattr(llm, "_call_api", lambda p, m, t: ("x", 1_000_000, 0))
    llm.ask("one")  # 1M input tokens on haiku = $1.00
    assert llm.spent_usd() == pytest.approx(1.00, abs=0.01)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_llm.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError`

- [ ] **Step 4: Write `engine/llm.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_llm.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add engine/ config.py requirements.txt tests/test_llm.py
git commit -m "m1: llm client — cached, budget-capped, test-injectable"
```

---

### Task 2: Kalshi adapter — live non-sports market cards

**Files:**
- Create: `adapters/kalshi_events.py`
- Test: `tests/test_kalshi_adapter.py`, `tests/fixtures/kalshi_events.json`

**Interfaces:**
- Consumes: `config.EXCLUDED_CATEGORIES`, `config.CACHE`
- Produces:
  `parse_events(payload: dict) -> list[dict]` — market cards, each:
  `{"ticker": str, "question": str, "category": str, "yes_bid": int,
  "yes_ask": int, "mid": int, "close_time": str, "volume": int}`
  (prices in cents; dollar-string inputs like "0.43" parsed to 43).
  `tradeable(cards: list[dict], now_iso: str) -> list[dict]` — keeps cards
  with two-sided quotes, spread <= 10 cents, volume >= 100, and closing
  more than 24h after `now_iso`; sorted by volume, biggest first.
  `fetch_open_markets() -> list[dict]` — live GET (network), parse, filter.
  `fetch_market(ticker: str) -> dict` — one live market with
  `{"ticker", "mid", "status", "result"}` for grading.

- [ ] **Step 1: Build the fixture** — `tests/fixtures/kalshi_events.json`

```json
{
  "events": [
    {"event_ticker": "ALBUM-DROP", "title": "Will the album drop this month?",
     "category": "Entertainment",
     "markets": [
       {"ticker": "ALBUM-DROP-JUL", "yes_sub_title": "in July",
        "yes_bid": 40, "yes_ask": 46, "volume": 5200,
        "close_time": "2026-07-31T23:59:00Z", "status": "active"},
       {"ticker": "ALBUM-DROP-THIN", "yes_sub_title": "thin book",
        "yes_bid": 0, "yes_ask": 99, "volume": 3,
        "close_time": "2026-07-31T23:59:00Z", "status": "active"}
     ]},
    {"event_ticker": "GAME-7", "title": "Will the home team win game 7?",
     "category": "Sports",
     "markets": [
       {"ticker": "GAME-7-YES", "yes_sub_title": "",
        "yes_bid": 55, "yes_ask": 57, "volume": 90000,
        "close_time": "2026-07-16T23:00:00Z", "status": "active"}
     ]},
    {"event_ticker": "CPI-STR", "title": "Will CPI come in above 3.0%?",
     "category": "Economics",
     "markets": [
       {"ticker": "CPI-STR-3", "yes_sub_title": "above 3.0",
        "yes_bid": "0.22", "yes_ask": "0.28", "volume": 1500,
        "close_time": "2026-07-15T18:00:00Z", "status": "active"}
     ]}
  ]
}
```

- [ ] **Step 2: Write the failing test** — `tests/test_kalshi_adapter.py`

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.kalshi_events import parse_events, tradeable

FIXTURE = Path(__file__).parent / "fixtures" / "kalshi_events.json"


def cards():
    return parse_events(json.loads(FIXTURE.read_text()))


def test_sports_are_excluded_and_prices_normalized():
    got = cards()
    tickers = {c["ticker"] for c in got}
    assert "GAME-7-YES" not in tickers          # sports excluded
    cpi = next(c for c in got if c["ticker"] == "CPI-STR-3")
    assert cpi["yes_bid"] == 22 and cpi["yes_ask"] == 28  # dollar-strings -> cents
    assert cpi["mid"] == 25
    album = next(c for c in got if c["ticker"] == "ALBUM-DROP-JUL")
    assert album["question"].startswith("Will the album drop")
    assert "in July" in album["question"]


def test_tradeable_filters_thin_wide_and_closing_soon():
    got = tradeable(cards(), now_iso="2026-07-15T00:00:00Z")
    tickers = [c["ticker"] for c in got]
    assert "ALBUM-DROP-JUL" in tickers   # two-sided, liquid, far from close
    assert "ALBUM-DROP-THIN" not in tickers  # one-sided/thin book
    assert "CPI-STR-3" not in tickers    # closes within 24h


def test_tradeable_sorts_by_volume():
    got = tradeable(cards(), now_iso="2026-07-15T00:00:00Z")
    volumes = [c["volume"] for c in got]
    assert volumes == sorted(volumes, reverse=True)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_kalshi_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Write `adapters/kalshi_events.py`**

```python
"""Read open Kalshi markets and turn them into simple 'market cards'.

Read-only public API, non-sports only, paper trading only. Known venue
quirk: prices usually arrive as cents (43) but sometimes as dollar
strings ("0.43") — _cents() accepts both.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

BASE = "https://api.elections.kalshi.com/trade-api/v2"


def _cents(value) -> int:
    """Turn 43, "43", or "0.43" into 43 cents. Unknown -> 0."""
    if value is None:
        return 0
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0
    if 0 < num < 1:          # dollar string like "0.43"
        return round(num * 100)
    return round(num)


def parse_events(payload: dict) -> list[dict]:
    """Flatten the /events response into one card per market."""
    cards = []
    for event in payload.get("events", []):
        if event.get("category") in config.EXCLUDED_CATEGORIES:
            continue
        for market in event.get("markets", []) or []:
            if market.get("status") not in (None, "active", "open"):
                continue
            bid, ask = _cents(market.get("yes_bid")), _cents(market.get("yes_ask"))
            sub = market.get("yes_sub_title") or ""
            question = event.get("title", "")
            if sub:
                question = f"{question} ({sub})"
            cards.append({
                "ticker": market.get("ticker", ""),
                "question": question,
                "category": event.get("category", ""),
                "yes_bid": bid,
                "yes_ask": ask,
                "mid": round((bid + ask) / 2) if (bid and ask) else 0,
                "close_time": market.get("close_time", ""),
                "volume": int(market.get("volume") or 0),
            })
    return cards


def tradeable(cards: list[dict], now_iso: str) -> list[dict]:
    """Keep markets a paper trader could actually act on."""
    from datetime import datetime, timedelta
    now = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
    keep = []
    for c in cards:
        if not (0 < c["yes_bid"] and c["yes_ask"] < 100):
            continue                                  # need a two-sided book
        if c["yes_ask"] - c["yes_bid"] > 10:
            continue                                  # spread too wide
        if c["volume"] < 100:
            continue                                  # too thin to matter
        try:
            close = datetime.fromisoformat(c["close_time"].replace("Z", "+00:00"))
        except ValueError:
            continue
        if close < now + timedelta(hours=24):
            continue                                  # about to settle
        keep.append(c)
    keep.sort(key=lambda c: c["volume"], reverse=True)
    return keep


def fetch_open_markets() -> list[dict]:
    """Live pull of open events with nested markets (paginated)."""
    import requests
    cards, cursor = [], None
    for _ in range(10):  # up to ~2000 markets; plenty
        params = {"status": "open", "with_nested_markets": "true", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE}/events", params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        cards += parse_events(payload)
        cursor = payload.get("cursor")
        if not cursor:
            break
    return cards


def fetch_market(ticker: str) -> dict:
    """One market's latest state, for grading picks."""
    import requests
    resp = requests.get(f"{BASE}/markets/{ticker}", timeout=30)
    resp.raise_for_status()
    m = resp.json().get("market", {})
    bid, ask = _cents(m.get("yes_bid")), _cents(m.get("yes_ask"))
    return {
        "ticker": ticker,
        "mid": round((bid + ask) / 2) if (bid and ask) else 0,
        "status": m.get("status", ""),
        "result": m.get("result", ""),
    }


if __name__ == "__main__":
    # CLI smoke test (no key needed): venv/bin/python adapters/kalshi_events.py
    from datetime import datetime, timezone
    live = tradeable(fetch_open_markets(),
                     datetime.now(timezone.utc).isoformat())
    print(f"{len(live)} tradeable non-sports markets right now; top 5 by volume:")
    for c in live[:5]:
        print(f"  {c['mid']:>3}c  {c['ticker']:<28} {c['question'][:60]}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_kalshi_adapter.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add adapters/kalshi_events.py tests/test_kalshi_adapter.py tests/fixtures/kalshi_events.json
git commit -m "m1: kalshi adapter — live non-sports market cards, offline-tested"
```

---

### Task 3: News snippets (no API key)

**Files:**
- Create: `engine/news.py`
- Test: `tests/test_news.py`, `tests/fixtures/news_rss.xml`

**Interfaces:**
- Consumes: `config.CACHE`
- Produces:
  `parse_rss(xml_text: str, limit: int = 3) -> list[str]` — headline strings.
  `headlines_for(query: str, limit: int = 3) -> list[str]` — live Google News
  RSS fetch, cached to `data/cache/news/` by day+query; returns `[]` on any
  network problem (never crashes a cycle).

- [ ] **Step 1: Build the fixture** — `tests/fixtures/news_rss.xml`

```xml
<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>query - Google News</title>
  <item><title>Album officially announced for July 25 release</title></item>
  <item><title>Producer teases final tracklist in interview</title></item>
  <item><title>Fans spot billboard campaign in three cities</title></item>
  <item><title>Fourth headline that should be cut by the limit</title></item>
</channel></rss>
```

- [ ] **Step 2: Write the failing test** — `tests/test_news.py`

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.news import parse_rss

FIXTURE = Path(__file__).parent / "fixtures" / "news_rss.xml"


def test_parses_headlines_and_respects_limit():
    heads = parse_rss(FIXTURE.read_text(), limit=3)
    assert len(heads) == 3
    assert heads[0] == "Album officially announced for July 25 release"


def test_garbage_xml_returns_empty_not_crash():
    assert parse_rss("<not really xml", limit=3) == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_news.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Write `engine/news.py`**

```python
"""Fresh headlines for a market question — free, no API key.

Google News RSS gives the crowd something real to react to. Failures
return an empty list: a news outage should never stop the cycle.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

CACHE_DIR = config.CACHE / "news"


def parse_rss(xml_text: str, limit: int = 3) -> list[str]:
    """Pull item titles out of an RSS feed. Bad XML -> empty list."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    titles = [item.findtext("title") or "" for item in root.iter("item")]
    return [t for t in titles if t][:limit]


def headlines_for(query: str, limit: int = 3) -> list[str]:
    """Today's headlines for a query, cached per day."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{date.today().isoformat()}_{quote(query)[:80]}"
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    try:
        import requests
        url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US"
        heads = parse_rss(requests.get(url, timeout=15).text, limit)
    except Exception:
        heads = []
    cache_file.write_text(json.dumps(heads))
    return heads
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_news.py -v`
Expected: 2 PASS

- [ ] **Step 6: Commit**

```bash
git add engine/news.py tests/test_news.py tests/fixtures/news_rss.xml
git commit -m "m1: keyless news headlines with daily cache"
```

---

### Task 4: Personas — the crowd roster

**Files:**
- Create: `engine/personas.py`
- Test: `tests/test_personas.py`

**Interfaces:**
- Consumes: `config.SEED`
- Produces:
  `ARCHETYPES: list[tuple[str, str]]` — six `(archetype, style)` pairs.
  `build_crowd(n: int, seed: int) -> list[dict]` — n agents, each
  `{"name": str, "archetype": str, "style": str}`; deterministic per seed;
  names unique.

- [ ] **Step 1: Write the failing test** — `tests/test_personas.py`

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.personas import ARCHETYPES, build_crowd


def test_six_archetypes_and_deterministic_crowd():
    assert len(ARCHETYPES) == 6
    a = build_crowd(8, seed=14000605)
    b = build_crowd(8, seed=14000605)
    assert a == b                       # same seed -> same crowd
    assert len({agent["name"] for agent in a}) == 8  # names unique


def test_crowd_cycles_all_archetypes():
    crowd = build_crowd(8, seed=1)
    used = {agent["archetype"] for agent in crowd}
    assert len(used) == 6  # with 8 agents every archetype shows up
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_personas.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `engine/personas.py`**

```python
"""The crowd roster: who is voting.

Six ways of thinking about an event, cycled across however many agents
config asks for. Names are seeded so any crowd can be rebuilt exactly.
"""
from __future__ import annotations

import random

ARCHETYPES: list[tuple[str, str]] = [
    ("stats nerd", "trusts base rates and numbers, distrusts stories"),
    ("narrative fan", "feels momentum and hype, follows the story"),
    ("sharp-money tracker", "cares only where informed money is moving"),
    ("oddsmaker", "tries to set a fair line others would bet into"),
    ("insider brain", "obsesses over who actually decides the outcome"),
    ("contrarian", "hunts for reasons the crowd is wrong"),
]

_FIRST = ["Ava", "Ben", "Cleo", "Dev", "Ember", "Finn", "Gia", "Hugo",
          "Iris", "Jax", "Kai", "Luna", "Mo", "Nia", "Oz", "Pia"]


def build_crowd(n: int, seed: int) -> list[dict]:
    """n agents cycling the six archetypes, with seeded unique names."""
    rng = random.Random(seed)
    names = rng.sample(_FIRST, k=min(n, len(_FIRST)))
    while len(names) < n:                      # crowds bigger than the name pool
        names.append(f"{rng.choice(_FIRST)}-{len(names)}")
    crowd = []
    for i in range(n):
        archetype, style = ARCHETYPES[i % len(ARCHETYPES)]
        crowd.append({"name": names[i], "archetype": archetype, "style": style})
    return crowd
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_personas.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add engine/personas.py tests/test_personas.py
git commit -m "m1: persona roster — six archetypes, seeded crowds"
```

---

### Task 5: Swarm — vote mode, deliberation, consensus

**Files:**
- Create: `engine/swarm.py`
- Test: `tests/test_swarm.py`

**Interfaces:**
- Consumes: `personas` agents, market cards, headlines, `llm.ask` (as the
  default `ask_fn`)
- Produces:
  `extract_json(text: str) -> dict | None` — first {...} block parsed, else None.
  `agent_vote(agent, card, headlines, ask_fn) -> dict | None` —
  `{"probability": float 0-1, "reason": str}` or None on junk.
  `deliberate(agent, card, own, others, ask_fn) -> dict | None` — same shape.
  `consensus(probs: list[float]) -> tuple[float, float]` — (trimmed mean,
  spread); with 5+ votes the single highest and lowest are dropped.
  `run_crowd(card, headlines, crowd, mode, k, deliberation, ask_fn) -> dict`
  — `{"probability": float, "spread": float, "votes": [...], "futures": [...],
  "skipped": int}` (futures filled by Task 6's simulate mode; empty in vote mode).

- [ ] **Step 1: Write the failing test** — `tests/test_swarm.py`

```python
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.personas import build_crowd
from engine.swarm import agent_vote, consensus, extract_json, run_crowd

CARD = {"ticker": "T", "question": "Will the album drop in July?", "mid": 43,
        "close_time": "2026-07-31T23:59:00Z", "yes_bid": 40, "yes_ask": 46,
        "volume": 5200, "category": "Entertainment"}


def canned_ask(answers):
    """ask_fn that replays a list of canned model answers in order."""
    stack = list(answers)
    def _ask(prompt, model=None, max_tokens=400):
        return stack.pop(0)
    return _ask


def test_extract_json_finds_object_in_noise():
    assert extract_json('sure! {"probability": 0.7, "reason": "hype"} done')["probability"] == 0.7
    assert extract_json("no json here") is None


def test_agent_vote_parses_and_clamps():
    agent = build_crowd(1, seed=1)[0]
    ask = canned_ask(['{"probability": 1.7, "reason": "too sure"}'])
    vote = agent_vote(agent, CARD, ["headline"], ask_fn=ask)
    assert vote["probability"] == 0.99  # clamped into [0.01, 0.99]


def test_agent_vote_returns_none_on_junk():
    agent = build_crowd(1, seed=1)[0]
    assert agent_vote(agent, CARD, [], ask_fn=canned_ask(["garbage"])) is None


def test_consensus_trims_extremes_with_five_plus():
    prob, spread = consensus([0.01, 0.5, 0.5, 0.5, 0.99])
    assert prob == pytest.approx(0.5)   # extremes dropped
    assert spread > 0


def test_run_crowd_vote_mode_counts_skips():
    crowd = build_crowd(4, seed=1)
    answers = ['{"probability": 0.6, "reason": "a"}',
               '{"probability": 0.7, "reason": "b"}',
               "junk",
               '{"probability": 0.5, "reason": "c"}']
    out = run_crowd(CARD, [], crowd, mode="vote", k=0,
                    deliberation=False, ask_fn=canned_ask(answers))
    assert out["skipped"] == 1
    assert len(out["votes"]) == 3
    assert 0.5 <= out["probability"] <= 0.7


def test_deliberation_second_round_updates():
    crowd = build_crowd(2, seed=1)
    answers = ['{"probability": 0.2, "reason": "low"}',
               '{"probability": 0.8, "reason": "high"}',
               # deliberation round answers:
               '{"probability": 0.4, "reason": "moved up"}',
               '{"probability": 0.6, "reason": "moved down"}']
    out = run_crowd(CARD, [], crowd, mode="vote", k=0,
                    deliberation=True, ask_fn=canned_ask(answers))
    probs = sorted(v["probability"] for v in out["votes"])
    assert probs == [0.4, 0.6]  # final numbers are the revised ones
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_swarm.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `engine/swarm.py`**

```python
"""The crowd itself: agents read a market, form a probability, and the
votes fold into one number plus a disagreement spread.

Every function takes ask_fn so tests can inject canned answers. Junk
answers are skipped and counted — never invented.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import llm

_VOTE_PROMPT = """You are {name}, a {archetype} — {style}.

A prediction market asks: "{question}"
The market price right now says YES has about a {mid}% chance.
Recent headlines: {headlines}

Think like your character and give YOUR OWN probability that this
resolves YES. Do not just repeat the market price.
Reply with ONLY JSON like {{"probability": 0.42, "reason": "one short sentence"}}"""

_DELIB_PROMPT = """You are {name}, a {archetype} — {style}.
Market: "{question}" (market price ~{mid}% YES). Your current view: {own}.

Other agents said:
{others}

After hearing them, give your FINAL probability. It is fine to keep your
number if they did not change your mind.
Reply with ONLY JSON like {{"probability": 0.42, "reason": "one short sentence"}}"""


def extract_json(text: str) -> dict | None:
    """Pull the first {...} out of a model answer. None if there isn't one."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return None


def _clean_prob(raw) -> float | None:
    try:
        p = float(raw)
    except (TypeError, ValueError):
        return None
    return min(max(p, 0.01), 0.99)  # never 0% or 100% — stay humble


def agent_vote(agent: dict, card: dict, headlines: list[str],
               ask_fn=llm.ask) -> dict | None:
    prompt = _VOTE_PROMPT.format(
        name=agent["name"], archetype=agent["archetype"], style=agent["style"],
        question=card["question"], mid=card["mid"],
        headlines="; ".join(headlines) if headlines else "(none found)")
    parsed = extract_json(ask_fn(prompt))
    if not parsed:
        return None
    prob = _clean_prob(parsed.get("probability"))
    if prob is None:
        return None
    return {"probability": prob, "reason": str(parsed.get("reason", ""))[:200]}


def deliberate(agent: dict, card: dict, own: dict, others: list[str],
               ask_fn=llm.ask) -> dict | None:
    prompt = _DELIB_PROMPT.format(
        name=agent["name"], archetype=agent["archetype"], style=agent["style"],
        question=card["question"], mid=card["mid"],
        own=f'{own["probability"]:.2f} ("{own["reason"]}")',
        others="\n".join(others))
    parsed = extract_json(ask_fn(prompt))
    if not parsed:
        return None
    prob = _clean_prob(parsed.get("probability"))
    if prob is None:
        return None
    return {"probability": prob, "reason": str(parsed.get("reason", ""))[:200]}


def consensus(probs: list[float]) -> tuple[float, float]:
    """Trimmed mean + spread. With 5+ votes, drop one high and one low."""
    if not probs:
        return 0.5, 0.0
    use = sorted(probs)
    if len(use) >= 5:
        use = use[1:-1]
    mean = sum(use) / len(use)
    spread = statistics.pstdev(probs) if len(probs) > 1 else 0.0
    return round(mean, 4), round(spread, 4)


def run_crowd(card: dict, headlines: list[str], crowd: list[dict],
              mode: str = "vote", k: int = 5, deliberation: bool = False,
              ask_fn=llm.ask) -> dict:
    """Run the whole crowd on one market and fold it into a consensus."""
    from engine import futures as _futures  # Task 6; local import avoids cycles

    votes, all_futures, skipped = [], [], 0
    for agent in crowd:
        if mode == "simulate":
            result = _futures.agent_futures(agent, card, headlines, k, ask_fn)
        else:
            result = agent_vote(agent, card, headlines, ask_fn)
        if result is None:
            skipped += 1
            continue
        result["agent"] = agent["name"]
        result["archetype"] = agent["archetype"]
        votes.append(result)
        all_futures.extend(result.get("futures", []))

    if deliberation and len(votes) >= 2:
        digest = [f'- {v["agent"]} ({v["archetype"]}): '
                  f'{v["probability"]:.2f} — {v["reason"]}' for v in votes]
        by_name = {a["name"]: a for a in crowd}
        revised = []
        for v in votes:
            others = [d for d in digest if not d.startswith(f'- {v["agent"]} ')]
            second = deliberate(by_name[v["agent"]], card, v, others, ask_fn)
            if second is not None:
                v = {**v, **second}
            revised.append(v)
        votes = revised

    prob, spread = consensus([v["probability"] for v in votes])
    return {"probability": prob, "spread": spread, "votes": votes,
            "futures": all_futures, "skipped": skipped}
```

- [ ] **Step 4: Create a stub so vote mode works before Task 6** — `engine/futures.py`

```python
"""Simulate mode lives here (filled in by the next task)."""
from __future__ import annotations


def agent_futures(agent, card, headlines, k, ask_fn):
    raise NotImplementedError("simulate mode arrives in the next task")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_swarm.py -v`
Expected: 6 PASS

- [ ] **Step 6: Commit**

```bash
git add engine/swarm.py engine/futures.py tests/test_swarm.py
git commit -m "m1: swarm — vote mode, deliberation round, trimmed consensus"
```

---

### Task 6: Simulate mode — each agent sees K futures

**Files:**
- Modify: `engine/futures.py` (replace the stub)
- Test: `tests/test_futures.py`

**Interfaces:**
- Consumes: agents, cards, headlines, `ask_fn`
- Produces:
  `agent_futures(agent, card, headlines, k, ask_fn) -> dict | None` —
  `{"probability": float, "reason": str, "futures": [{"story": str,
  "resolves": "YES"|"NO", "agent": str}]}`; probability = YES-fraction of
  the agent's own futures; None if fewer than half of k futures parse.

- [ ] **Step 1: Write the failing test** — `tests/test_futures.py`

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.futures import agent_futures
from engine.personas import build_crowd

CARD = {"ticker": "T", "question": "Will the album drop in July?", "mid": 43}
AGENT = build_crowd(1, seed=1)[0]


def ask_with(payload):
    return lambda prompt, model=None, max_tokens=400: payload


def test_probability_is_yes_fraction_of_futures():
    payload = ('{"futures": ['
               '{"story": "surprise midnight drop", "resolves": "YES"},'
               '{"story": "label delays to August", "resolves": "NO"},'
               '{"story": "single first, album July 30", "resolves": "YES"},'
               '{"story": "tour pushes it back", "resolves": "NO"},'
               '{"story": "deluxe drops July 25", "resolves": "YES"}],'
               '"reason": "momentum is real"}')
    out = agent_futures(AGENT, CARD, [], k=5, ask_fn=ask_with(payload))
    assert out["probability"] == pytest.approx(0.6)
    assert len(out["futures"]) == 5
    assert out["futures"][0]["agent"] == AGENT["name"]


def test_too_few_parsed_futures_returns_none():
    payload = '{"futures": [{"story": "only one", "resolves": "YES"}], "reason": "meh"}'
    assert agent_futures(AGENT, CARD, [], k=5, ask_fn=ask_with(payload)) is None


def test_junk_resolves_values_are_dropped():
    payload = ('{"futures": ['
               '{"story": "a", "resolves": "YES"},'
               '{"story": "b", "resolves": "maybe"},'
               '{"story": "c", "resolves": "NO"},'
               '{"story": "d", "resolves": "YES"}], "reason": "r"}')
    out = agent_futures(AGENT, CARD, [], k=4, ask_fn=ask_with(payload))
    assert len(out["futures"]) == 3     # "maybe" dropped
    assert out["probability"] == pytest.approx(2 / 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_futures.py -v`
Expected: FAIL — `NotImplementedError` or assertion failures

- [ ] **Step 3: Replace `engine/futures.py`**

```python
"""Simulate mode: an agent doesn't just vote — it imagines the event
playing out K times, and its probability is the share of its own futures
where the answer is YES. The stories feed the what-if view and the
dashboard's futures tree later.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import llm
from engine.swarm import extract_json

_SIM_PROMPT = """You are {name}, a {archetype} — {style}.

A prediction market asks: "{question}"
The market price right now says YES has about a {mid}% chance.
Recent headlines: {headlines}

Imagine {k} DIFFERENT ways this could actually play out — short, concrete,
one sentence each. Make them genuinely different, not five copies.
Reply with ONLY JSON like:
{{"futures": [{{"story": "one sentence", "resolves": "YES"}},
              {{"story": "another way it goes", "resolves": "NO"}}],
  "reason": "one sentence on your overall read"}}
Give exactly {k} futures, each with "resolves" as "YES" or "NO"."""


def agent_futures(agent: dict, card: dict, headlines: list[str], k: int,
                  ask_fn=llm.ask) -> dict | None:
    prompt = _SIM_PROMPT.format(
        name=agent["name"], archetype=agent["archetype"], style=agent["style"],
        question=card["question"], mid=card["mid"], k=k,
        headlines="; ".join(headlines) if headlines else "(none found)")
    parsed = extract_json(ask_fn(prompt, max_tokens=200 + 80 * k))
    if not parsed:
        return None

    futures = []
    for f in parsed.get("futures", []):
        if not isinstance(f, dict):
            continue
        verdict = str(f.get("resolves", "")).strip().upper()
        story = str(f.get("story", "")).strip()
        if verdict in ("YES", "NO") and story:
            futures.append({"story": story[:200], "resolves": verdict,
                            "agent": agent["name"]})
    if len(futures) < max(k // 2, 2):
        return None      # the model didn't really play along — skip, don't guess

    yes = sum(1 for f in futures if f["resolves"] == "YES")
    prob = min(max(yes / len(futures), 0.01), 0.99)
    return {"probability": round(prob, 4),
            "reason": str(parsed.get("reason", ""))[:200],
            "futures": futures}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_futures.py tests/test_swarm.py -v`
Expected: 9 PASS (futures + swarm still green)

- [ ] **Step 5: Commit**

```bash
git add engine/futures.py tests/test_futures.py
git commit -m "m1: simulate mode — K futures per agent, YES-fraction probability"
```

---

### Task 7: What-if — the god's-eye

**Files:**
- Create: `engine/whatif.py`
- Test: `tests/test_whatif.py`

**Interfaces:**
- Consumes: `run_crowd` (Task 5), cards, crowd
- Produces:
  `run_whatif(card, headlines, crowd, inject, mode, k, deliberation, ask_fn)
  -> dict` — `{"before": <run_crowd dict>, "after": <run_crowd dict>,
  "shift": float}` where `after` re-runs the crowd with the injected fact
  forced true, and `shift = after.probability - before.probability`.

- [ ] **Step 1: Write the failing test** — `tests/test_whatif.py`

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.personas import build_crowd
from engine.whatif import run_whatif

CARD = {"ticker": "T", "question": "Will the album drop in July?", "mid": 43}


def test_whatif_reruns_and_reports_shift():
    crowd = build_crowd(2, seed=1)
    seen_prompts = []

    def ask(prompt, model=None, max_tokens=400):
        seen_prompts.append(prompt)
        # crowd leans low before the injection, high after it
        if "WHAT-IF" in prompt:
            return '{"probability": 0.9, "reason": "fact changes everything"}'
        return '{"probability": 0.3, "reason": "doubtful"}'

    out = run_whatif(CARD, [], crowd, inject="the label confirms the date",
                     mode="vote", k=0, deliberation=False, ask_fn=ask)
    assert out["before"]["probability"] == pytest.approx(0.3)
    assert out["after"]["probability"] == pytest.approx(0.9)
    assert out["shift"] == pytest.approx(0.6)
    assert any("label confirms the date" in p for p in seen_prompts)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_whatif.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `engine/whatif.py`**

```python
"""The god's-eye: force a fact to be true and watch the crowd's number move.

We don't touch the agents — we edit the world they see. The injected fact
is prepended to the market question so every prompt (vote or simulate)
carries it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import llm
from engine.swarm import run_crowd


def run_whatif(card: dict, headlines: list[str], crowd: list[dict],
               inject: str, mode: str = "vote", k: int = 5,
               deliberation: bool = False, ask_fn=llm.ask) -> dict:
    before = run_crowd(card, headlines, crowd, mode, k, deliberation, ask_fn)

    twisted = dict(card)
    twisted["question"] = (
        f"WHAT-IF (treat as definitely true: {inject}) — {card['question']}")
    after = run_crowd(twisted, headlines, crowd, mode, k, deliberation, ask_fn)

    return {"before": before, "after": after,
            "shift": round(after["probability"] - before["probability"], 4)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_whatif.py -v`
Expected: 1 PASS

- [ ] **Step 5: Commit**

```bash
git add engine/whatif.py tests/test_whatif.py
git commit -m "m1: what-if god's-eye — inject a fact, measure the shift"
```

---

### Task 8: Paper ledger — picks, settlement, CLV

**Files:**
- Create: `ledger.py`
- Test: `tests/test_ledger.py`

**Interfaces:**
- Consumes: market cards, `fetch_market`-shaped dicts, `config.DATA`
- Produces:
  `LEDGER_COLUMNS: list[str]` — exactly: `logged_at, ticker, question, side,
  entry_mid, crowd_prob, edge_cents, mode, status, result, latest_mid,
  clv_cents, settled_at`.
  `log_pick(row: dict, path: Path | None = None) -> None` — appends (creates
  file with header when missing); refuses duplicate open (ticker, side).
  `load(path: Path | None = None) -> list[dict]`.
  `grade(latest_by_ticker: dict[str, dict], path: Path | None = None) -> dict`
  — updates open rows: `latest_mid` + `clv_cents` (YES pick: latest−entry;
  NO pick: entry−latest); rows whose market is settled get `status="settled"`,
  `result`, `settled_at`. Returns `{"updated": int, "settled": int}`.

- [ ] **Step 1: Write the failing test** — `tests/test_ledger.py`

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ledger


def pick(ticker="T1", side="YES"):
    return {"logged_at": "2026-07-15T12:00:00Z", "ticker": ticker,
            "question": "Will it happen?", "side": side, "entry_mid": 43,
            "crowd_prob": 0.61, "edge_cents": 18, "mode": "vote",
            "status": "open", "result": "", "latest_mid": 43,
            "clv_cents": 0, "settled_at": ""}


def test_log_and_load_roundtrip(tmp_path):
    path = tmp_path / "ledger.csv"
    ledger.log_pick(pick(), path=path)
    rows = ledger.load(path=path)
    assert len(rows) == 1 and rows[0]["ticker"] == "T1"
    assert int(rows[0]["entry_mid"]) == 43


def test_duplicate_open_pick_is_refused(tmp_path):
    path = tmp_path / "ledger.csv"
    ledger.log_pick(pick(), path=path)
    ledger.log_pick(pick(), path=path)   # same ticker+side while open
    assert len(ledger.load(path=path)) == 1


def test_grade_updates_clv_and_settles(tmp_path):
    path = tmp_path / "ledger.csv"
    ledger.log_pick(pick("T1", "YES"), path=path)
    ledger.log_pick(pick("T2", "NO"), path=path)
    latest = {
        "T1": {"ticker": "T1", "mid": 55, "status": "active", "result": ""},
        "T2": {"ticker": "T2", "mid": 30, "status": "settled", "result": "no"},
    }
    stats = ledger.grade(latest, path=path)
    rows = {r["ticker"]: r for r in ledger.load(path=path)}
    assert int(rows["T1"]["clv_cents"]) == 12          # YES: 55 - 43
    assert rows["T1"]["status"] == "open"
    assert rows["T2"]["status"] == "settled" and rows["T2"]["result"] == "no"
    assert int(rows["T2"]["clv_cents"]) == 13          # NO: 43 - 30
    assert stats == {"updated": 2, "settled": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_ledger.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `ledger.py`**

```python
"""The paper ledger: every pick the crowd makes, graded against reality.

Append-only CSV. CLV (closing line value) is the honest score: did the
market move toward our pick after we made it?
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

LEDGER_COLUMNS = ["logged_at", "ticker", "question", "side", "entry_mid",
                  "crowd_prob", "edge_cents", "mode", "status", "result",
                  "latest_mid", "clv_cents", "settled_at"]

_DEFAULT = config.DATA / "ledger.csv"


def load(path: Path | None = None) -> list[dict]:
    path = path or _DEFAULT
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_all(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def log_pick(row: dict, path: Path | None = None) -> None:
    """Append one pick. A second open pick on the same ticker+side is a
    repeat opinion, not a new position — refused."""
    path = path or _DEFAULT
    rows = load(path)
    for r in rows:
        if (r["ticker"] == row["ticker"] and r["side"] == row["side"]
                and r["status"] == "open"):
            return
    rows.append({col: row.get(col, "") for col in LEDGER_COLUMNS})
    _write_all(rows, path)


def grade(latest_by_ticker: dict[str, dict], path: Path | None = None) -> dict:
    """Refresh open picks against the latest market state."""
    path = path or _DEFAULT
    rows = load(path)
    updated = settled = 0
    for r in rows:
        if r["status"] != "open" or r["ticker"] not in latest_by_ticker:
            continue
        latest = latest_by_ticker[r["ticker"]]
        entry = int(r["entry_mid"])
        mid = int(latest.get("mid") or entry)
        r["latest_mid"] = mid
        r["clv_cents"] = (mid - entry) if r["side"] == "YES" else (entry - mid)
        updated += 1
        if latest.get("status") == "settled" and latest.get("result"):
            r["status"] = "settled"
            r["result"] = latest["result"]
            r["settled_at"] = datetime.now(timezone.utc).isoformat()
            settled += 1
    _write_all(rows, path)
    return {"updated": updated, "settled": settled}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_ledger.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add ledger.py tests/test_ledger.py
git commit -m "m1: paper ledger — picks, duplicate guard, CLV grading, settlement"
```

---

### Task 9: run.py — one live cycle

**Files:**
- Create: `run.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  `pick_side(crowd_prob: float, mid: int) -> tuple[str, int] | None` —
  ("YES"/"NO", edge_cents) when the edge clears `MIN_EDGE_CENTS +
  FEE_BUFFER_CENTS`, else None. Edge in cents: YES edge = crowd*100 − mid;
  NO edge = mid − crowd*100.
  `one_cycle(cards: list[dict] | None = None, ask_fn=None) -> dict` — grades
  open picks, runs the crowd on the top `MARKETS_PER_RUN` tradeable cards,
  logs new picks, returns `{"considered": int, "picks": int, "graded": dict}`.
  Passing `cards`/`ask_fn` keeps tests offline; `None` means live.
  CLI: `venv/bin/python run.py` (live) — prints a plain-English cycle report.

- [ ] **Step 1: Write the failing test** — `tests/test_run.py`

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import ledger
import run as runner
from adapters.kalshi_events import parse_events

FIXTURE = Path(__file__).parent / "fixtures" / "kalshi_events.json"


def test_pick_side_needs_edge_plus_buffer():
    # crowd 61% vs mid 43 -> YES edge 18c: clears 10 + 3
    assert runner.pick_side(0.61, 43) == ("YES", 18)
    # crowd 30% vs mid 43 -> NO edge 13c: clears
    assert runner.pick_side(0.30, 43) == ("NO", 13)
    # crowd 50% vs mid 43 -> 7c: does NOT clear
    assert runner.pick_side(0.50, 43) is None


def test_one_cycle_offline_logs_a_pick(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "_DEFAULT", tmp_path / "ledger.csv")
    cards = parse_events(json.loads(FIXTURE.read_text()))
    confident = '{"probability": 0.75, "reason": "sure thing"}'
    out = runner.one_cycle(cards=cards,
                           ask_fn=lambda p, model=None, max_tokens=400: confident)
    assert out["picks"] >= 1
    rows = ledger.load(tmp_path / "ledger.csv")
    assert rows and rows[0]["side"] == "YES"
    assert rows[0]["mode"] in ("vote", "simulate")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_run.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `run.py`**

```python
"""One live cycle of the engine.

Grade what's open, look at the biggest open non-sports markets, let the
crowd form its number, and log a paper pick when the crowd disagrees with
the market by more than fees could explain. A person places any real bet —
this only writes CSV rows.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import ledger
from adapters import kalshi_events as kalshi
from engine import llm, news
from engine.personas import build_crowd
from engine.swarm import run_crowd


def pick_side(crowd_prob: float, mid: int) -> tuple[str, int] | None:
    """Which side (if any) does the crowd's number justify, after costs?"""
    need = config.MIN_EDGE_CENTS + config.FEE_BUFFER_CENTS
    yes_edge = round(crowd_prob * 100) - mid
    if yes_edge >= need:
        return "YES", yes_edge
    if -yes_edge >= need:
        return "NO", -yes_edge
    return None


def one_cycle(cards: list[dict] | None = None, ask_fn=None) -> dict:
    live = cards is None
    ask = ask_fn or llm.ask
    now = datetime.now(timezone.utc).isoformat()

    # 1. Grade open picks (live only — needs per-ticker fetches).
    graded = {"updated": 0, "settled": 0}
    if live:
        open_tickers = {r["ticker"] for r in ledger.load()
                        if r["status"] == "open"}
        latest = {t: kalshi.fetch_market(t) for t in open_tickers}
        graded = ledger.grade(latest)
        cards = kalshi.fetch_open_markets()

    # 2. The crowd votes on the biggest tradeable markets.
    targets = kalshi.tradeable(cards, now)[:config.MARKETS_PER_RUN]
    crowd = build_crowd(config.ENGINE_N_AGENTS, config.SEED)
    mode = "simulate" if config.SIM_ROLLOUTS_K and config.DELIBERATION is None \
        else ("simulate" if getattr(config, "SIM_MODE", "vote") == "simulate"
              else "vote")

    picks = 0
    for card in targets:
        heads = news.headlines_for(card["question"]) if live else []
        result = run_crowd(card, heads, crowd, mode=mode,
                           k=config.SIM_ROLLOUTS_K,
                           deliberation=config.DELIBERATION, ask_fn=ask)
        verdict = pick_side(result["probability"], card["mid"])
        line = (f'{card["mid"]:>3}c market | {result["probability"]:.2f} crowd '
                f'(spread {result["spread"]:.2f}, {result["skipped"]} skipped) '
                f'| {card["question"][:60]}')
        if verdict is None:
            print(f"  pass  {line}")
            continue
        side, edge = verdict
        ledger.log_pick({
            "logged_at": now, "ticker": card["ticker"],
            "question": card["question"], "side": side,
            "entry_mid": card["mid"], "crowd_prob": result["probability"],
            "edge_cents": edge, "mode": mode, "status": "open",
            "result": "", "latest_mid": card["mid"], "clv_cents": 0,
            "settled_at": "",
        })
        picks += 1
        print(f"  PICK  {side} +{edge}c {line}")

    print(f"cycle done: {len(targets)} markets considered, {picks} picks, "
          f"{graded['settled']} settled, ${llm.spent_usd():.2f} total spend")
    return {"considered": len(targets), "picks": picks, "graded": graded}


if __name__ == "__main__":
    one_cycle()
```

**Note for the implementer:** the `mode` expression above as written in this
plan is convoluted — simplify it to read from a single new config knob
`SIM_MODE = "vote"` (add it to config.py next to `SIM_ROLLOUTS_K`, values
`"vote"` or `"simulate"`), i.e. `mode = config.SIM_MODE`. The test only
asserts mode lands in `("vote", "simulate")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_run.py -v` then the whole suite `venv/bin/pytest -q`
Expected: all green (27 from M0 + ~20 new)

- [ ] **Step 5: Commit**

```bash
git add run.py config.py tests/test_run.py
git commit -m "m1: live cycle — grade, crowd, edge rule, paper picks"
```

---

### Task 10: Live smoke run (needs no key for the adapter; key for the crowd)

**Files:**
- Created by running: `data/ledger.csv`, `data/spend.json`, cache files

- [ ] **Step 1: Adapter smoke (keyless)** — `venv/bin/python adapters/kalshi_events.py`
  Expected: a count of live tradeable non-sports markets + top 5 by volume.
  If the API shape differs from the fixture (field names, cursor), fix
  `parse_events` and add a fixture case capturing the real shape.

- [ ] **Step 2 (HUMAN-DEPENDENT): First real crowd run** — requires
  `ANTHROPIC_API_KEY` in the environment. Run `venv/bin/python run.py`.
  Expected: per-market lines (`pass` or `PICK`), a cycle summary with total
  spend well under $1, and `data/ledger.csv` gaining rows for any picks.

- [ ] **Step 3: Commit the README status update**

Update README.md's status line to:
```markdown
**Status: the engine is live — crowd votes on real open markets (paper only).
NBA evaluation lab: built, deferred until its data arrives.**
```

```bash
git add README.md
git commit -m "m1: engine live — status update"
```

- [ ] **Step 4: Report** — tell the owner: how many markets were considered,
  any picks with the crowd's reasoning, total spend, and that `run.py` can
  now be run any time (or wired to a GitHub Action hourly, like kayfabe,
  in a follow-up).

---

## Self-review notes (done at write time)

- **Spec coverage (M1 scope):** persona mode ✓ (Task 4-5), simulate mode ✓
  (Task 6), one deliberation round ✓ (Task 5), what-if ✓ (Task 7), live
  Kalshi non-sports adapter ✓ (Task 2), paper ledger + CLV ✓ (Task 8),
  live cycle ✓ (Task 9), budget cap + cache ✓ (Task 1), news ✓ (Task 3).
  Ensemble mode (different model tiers + partitioned evidence) is
  deliberately deferred to M2 with the learning loop — one crowd
  architecture goes live first; noted here so the gap is a decision,
  not an oversight.
- **Placeholders:** none — every step has complete code. The one known
  simplification (Task 9's `mode` expression) is flagged inline with the
  exact fix.
- **Type consistency:** market card keys (`ticker/question/category/yes_bid/
  yes_ask/mid/close_time/volume`) match across Tasks 2→9; vote dicts
  (`probability/reason`) match across 5→7; `ask_fn(prompt, model=None,
  max_tokens=400)` signature identical everywhere; ledger columns fixed in
  one constant.
