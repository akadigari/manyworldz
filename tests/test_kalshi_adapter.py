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
