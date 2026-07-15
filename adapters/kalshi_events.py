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
