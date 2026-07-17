import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.kalshi_events import _cents, parse_events, tradeable

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


def test_live_dollars_schema_normalized():
    # Live API shape (verified 2026-07-15): yes_bid_dollars/yes_ask_dollars
    # as dollar strings, volume_fp as a float string. No plain yes_bid/
    # yes_ask/volume on this market at all.
    got = cards()
    fed = next(c for c in got if c["ticker"] == "FED-CUT-Q3")
    assert fed["yes_bid"] == 12 and fed["yes_ask"] == 14
    assert fed["mid"] == 13
    assert fed["volume"] == 112505


def test_cents_parses_dollar_strings_regardless_of_size():
    # A string containing "." is always dollars, even at the top of the
    # book where the old "0 < num < 1" guess broke: "1.0000" used to come
    # out as 1 cent instead of 100.
    assert _cents("1.0000") == 100
    assert _cents("0.4300") == 43
    assert _cents("0.43") == 43
    assert _cents(43) == 43
    assert _cents("43") == 43
    assert _cents(None) == 0


def test_one_sided_book_at_top_of_range_is_excluded():
    # yes_bid_dollars "0.9900" / yes_ask_dollars "1.0000" -> bid 99 / ask
    # 100. An ask of 100 means "no one is actually offering to sell":
    # not a real two-sided market, so tradeable() must drop it.
    got = cards()
    near = next(c for c in got if c["ticker"] == "NEAR-CERTAIN-99")
    assert near["yes_bid"] == 99 and near["yes_ask"] == 100
    tickers = [c["ticker"] for c in tradeable(got, now_iso="2026-07-15T00:00:00Z")]
    assert "NEAR-CERTAIN-99" not in tickers


def test_tradeable_skips_card_with_null_close_time_instead_of_crashing():
    bad_card = {"ticker": "BAD-CLOSE", "question": "Bad close time?",
                "category": "Economics", "yes_bid": 40, "yes_ask": 46,
                "mid": 43, "close_time": None, "volume": 5000}
    got = tradeable(cards() + [bad_card], now_iso="2026-07-15T00:00:00Z")
    assert "BAD-CLOSE" not in [c["ticker"] for c in got]
