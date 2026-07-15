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
    # Fixed now_iso: pins the clock so fixture close_times (e.g. July 31,
    # 2026) can never drift into "closing soon" and get excluded as the
    # real calendar date moves forward. See test_one_cycle_no_quorum_*
    # below for the bug this used to hide.
    out = runner.one_cycle(cards=cards, now_iso="2026-07-15T00:00:00Z",
                           ask_fn=lambda p, model=None, max_tokens=400: confident)
    assert out["picks"] >= 1
    rows = ledger.load(tmp_path / "ledger.csv")
    assert rows and rows[0]["side"] == "YES"
    assert rows[0]["mode"] in ("vote", "simulate")


def test_one_cycle_all_junk_crowd_logs_no_pick(tmp_path, monkeypatch):
    # If every agent's answer is unusable, run_crowd's consensus() falls
    # back to a default 0.5 probability. Before the fix, one_cycle treated
    # that fake number as a real crowd opinion and could log a pick out of
    # thin air. It must instead print "no quorum" and log nothing.
    monkeypatch.setattr(ledger, "_DEFAULT", tmp_path / "ledger.csv")
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 4)
    cards = parse_events(json.loads(FIXTURE.read_text()))
    out = runner.one_cycle(cards=cards, now_iso="2026-07-15T00:00:00Z",
                           ask_fn=lambda p, model=None, max_tokens=400: "garbage")
    assert out["picks"] == 0
    assert out["considered"] >= 1          # the market was still looked at
    assert ledger.load(tmp_path / "ledger.csv") == []


def test_one_cycle_live_grading_survives_one_bad_ticker_fetch(tmp_path, monkeypatch, capsys):
    # Grading fans out one fetch_market() call per open ticker. One flaky
    # ticker used to take the whole cycle down; now it's skipped, counted,
    # and reported, and every other open pick still gets graded.
    ledger_path = tmp_path / "ledger.csv"
    monkeypatch.setattr(ledger, "_DEFAULT", ledger_path)

    def pick(ticker, side="YES"):
        return {"logged_at": "2026-07-15T00:00:00Z", "ticker": ticker,
                "question": "Will it happen?", "side": side, "entry_mid": 43,
                "crowd_prob": 0.61, "edge_cents": 18, "mode": "vote",
                "status": "open", "result": "", "latest_mid": 43,
                "clv_cents": 0, "settled_at": ""}

    ledger.log_pick(pick("T-OK"), path=ledger_path)
    ledger.log_pick(pick("T-FAIL"), path=ledger_path)

    def flaky_fetch_market(ticker):
        if ticker == "T-FAIL":
            raise RuntimeError("network blew up")
        return {"ticker": ticker, "mid": 55, "status": "active", "result": ""}

    monkeypatch.setattr(runner.kalshi, "fetch_market", flaky_fetch_market)
    monkeypatch.setattr(runner.kalshi, "fetch_open_markets", lambda: [])

    out = runner.one_cycle(
        ask_fn=lambda p, model=None, max_tokens=400: '{"probability": 0.5, "reason": "x"}')

    captured = capsys.readouterr()
    assert "could not refresh 1 open pick" in captured.out
    assert out["graded"]["updated"] == 1     # only T-OK got refreshed
    rows = {r["ticker"]: r for r in ledger.load(ledger_path)}
    assert int(rows["T-OK"]["latest_mid"]) == 55
    assert int(rows["T-FAIL"]["latest_mid"]) == 43   # untouched, kept old value
