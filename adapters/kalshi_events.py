"""Read open Kalshi markets and turn them into simple 'market cards'.

Read-only public API, non-sports only, paper trading only. Known venue
quirk: prices usually arrive as cents (43) but sometimes as dollar
strings ("0.43") — _cents() accepts both. Live responses (verified
2026-07-15) use yes_bid_dollars/yes_ask_dollars ("0.1200" = 12 cents)
and volume_fp (a float string) instead of the plain yes_bid/yes_ask/
volume fields the old fixture used — we read the dollars/fp fields
first and fall back to the plain ones so both shapes work.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

BASE = "https://api.elections.kalshi.com/trade-api/v2"


def _cents(value) -> int:
    """Turn 43, "43", "0.43", or "1.0000" into cents. Unknown -> 0.

    Kalshi sends prices two ways: plain cents (43 or "43") or dollar
    strings ("0.43", "1.0000"). A dollar string always has a "." in it,
    so any string with a "." is always dollars, no matter how big the
    number is. That fixes a bug where "1.0000" (the top of the book,
    100 cents) was misread as 1 cent because 1.0 doesn't satisfy the old
    "0 < num < 1" dollar-string guess. Numeric (non-string) inputs keep
    the old guess: a bare 0.43 is treated as dollars, everything else as
    already-cents.
    """
    if value is None:
        return 0
    if isinstance(value, str) and "." in value:
        try:
            return round(float(value) * 100)
        except ValueError:
            return 0
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0
    if 0 < num < 1:          # bare float like 0.43
        return round(num * 100)
    return round(num)


def parse_events(payload: dict) -> list[dict]:
    """Turn Kalshi's raw /events response into a flat list of "cards" —
    one simple dict per tradeable market, with the fields we actually use.

    An "event" can bundle several related markets together (e.g. one
    event per election, one market per candidate), so this walks every
    event and pulls out each of its markets as its own card.
    """
    cards = []
    for event in payload.get("events", []):
        if event.get("category") in config.EXCLUDED_CATEGORIES:
            continue
        for market in event.get("markets", []) or []:
            if market.get("status") not in (None, "active", "open"):
                continue
            bid = _cents(market.get("yes_bid_dollars", market.get("yes_bid")))
            ask = _cents(market.get("yes_ask_dollars", market.get("yes_ask")))
            subtitle = market.get("yes_sub_title") or ""
            question = event.get("title", "")
            if subtitle:
                question = f"{question} ({subtitle})"
            cards.append({
                "ticker": market.get("ticker", ""),
                "question": question,
                "category": event.get("category", ""),
                "yes_bid": bid,      # highest price buyers are offering, in cents
                "yes_ask": ask,      # lowest price sellers are asking, in cents
                "mid": round((bid + ask) / 2) if (bid and ask) else 0,  # halfway between — the "market price"
                "close_time": market.get("close_time", ""),
                "volume": int(float(market.get("volume_fp") or market.get("volume") or 0)),
            })
    return cards


def tradeable(cards: list[dict], now_iso: str) -> list[dict]:
    """Filter down to markets a real trader could actually get into.

    Throws out anything with no real price, a spread too wide to be worth
    it, too little trading activity, or a close time too soon to react to.
    What's left is sorted by volume (busiest markets first).
    """
    from datetime import datetime, timedelta
    now = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
    keep = []
    for c in cards:
        if not (0 < c["yes_bid"] and c["yes_ask"] < 100 and c["mid"] > 0):
            continue                                  # need both a real buy price and a real sell price
        if c["yes_ask"] - c["yes_bid"] > 10:
            continue                                  # spread too wide
        if c["volume"] < 100:
            continue                                  # too thin to matter
        try:
            close = datetime.fromisoformat(c["close_time"].replace("Z", "+00:00"))
        except (ValueError, AttributeError, TypeError):
            continue                                  # missing/bad close_time — skip, don't crash
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
    bid = _cents(m.get("yes_bid_dollars", m.get("yes_bid")))
    ask = _cents(m.get("yes_ask_dollars", m.get("yes_ask")))
    # A one-sided book (no real bid, or no real ask — Kalshi shows a
    # missing ask as either 0 or 100) means we don't actually know a
    # price. Say so with None instead of making one up.
    one_sided = bid == 0 or ask in (0, 100)
    return {
        "ticker": ticker,
        "mid": None if one_sided else round((bid + ask) / 2),
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
