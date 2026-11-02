import json
import sys
from pathlib import Path

import pytest

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


def test_one_cycle_saves_a_snapshot_for_the_dashboard(tmp_path, monkeypatch):
    # The website draws from latest_cycle.json: make sure a cycle writes it.
    import json
    monkeypatch.setattr(ledger, "_DEFAULT", tmp_path / "ledger.csv")
    # This test's canned answer is vote-shaped, so pin vote mode.
    monkeypatch.setattr(config, "SIM_MODE", "vote")
    cards = parse_events(json.loads(FIXTURE.read_text()))
    confident = '{"probability": 0.75, "reason": "sure thing"}'
    snap_file = tmp_path / "snap.json"
    runner.one_cycle(cards=cards,
                     ask_fn=lambda p, model=None, max_tokens=400: confident,
                     now_iso="2026-07-15T00:00:00Z",
                     snapshot_path=snap_file)
    snap = json.loads(snap_file.read_text())
    assert snap["markets"], "snapshot should hold the markets the crowd saw"
    market = snap["markets"][0]
    assert market["votes"]                      # every vote is preserved
    assert market["probability"] == 0.75
    assert market["verdict"] is not None        # pass / no_quorum / pick


def test_snapshot_records_no_quorum_verdict(tmp_path, monkeypatch):
    # When the whole crowd gives junk, the snapshot must record "no_quorum"
    # (never a fabricated pick) so the dashboard shows the honest truth.
    import json
    monkeypatch.setattr(ledger, "_DEFAULT", tmp_path / "ledger.csv")
    cards = parse_events(json.loads(FIXTURE.read_text()))
    snap_file = tmp_path / "snap.json"
    out = runner.one_cycle(cards=cards,
                           ask_fn=lambda p, model=None, max_tokens=400: "garbage",
                           now_iso="2026-07-15T00:00:00Z",
                           snapshot_path=snap_file)
    assert out["picks"] == 0
    snap = json.loads(snap_file.read_text())
    assert snap["markets"]
    assert all(m["verdict"] == "no_quorum" for m in snap["markets"])


def test_one_cycle_offline_logs_a_pick(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "_DEFAULT", tmp_path / "ledger.csv")
    # This test's canned answer is vote-shaped, so pin vote mode.
    monkeypatch.setattr(config, "SIM_MODE", "vote")
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

def test_one_cycle_default_simulate_mode_logs_a_pick(tmp_path, monkeypatch):
    # The engine's default is simulate mode: canned futures-shaped answers
    # must flow through a whole cycle and land a pick in the ledger.
    monkeypatch.setattr(ledger, "_DEFAULT", tmp_path / "ledger.csv")
    monkeypatch.setattr(config, "SIM_MODE", "simulate")
    cards = parse_events(json.loads(FIXTURE.read_text()))
    futures = ('{"futures": ['
               '{"story": "it lands cleanly", "resolves": "YES"},'
               '{"story": "a delay but it happens", "resolves": "YES"},'
               '{"story": "it happens early", "resolves": "YES"},'
               '{"story": "momentum carries it", "resolves": "YES"},'
               '{"story": "a surprise kills it", "resolves": "NO"}]}')
    out = runner.one_cycle(cards=cards, now_iso="2026-07-15T00:00:00Z",
                           ask_fn=lambda p, model=None, max_tokens=400: futures)
    assert out["picks"] >= 1
    rows = ledger.load(tmp_path / "ledger.csv")
    assert rows and rows[0]["mode"] == "simulate"


# -- an account-level API failure fails the run with a readable reason -----
#
# Found 2026-11-02. Since the Anthropic account ran out of credits on
# 2026-10-01, every Kalshi run crashed on its first model call with a
# raw traceback, and the reason was only in logs that need a GitHub
# sign-in. Now main() prints a ::error:: line, which shows on the run
# page itself, and still exits 1.

def test_main_prints_an_error_line_and_exits_1_when_api_access_is_off(
        tmp_path, monkeypatch, capsys):
    import functools

    import anthropic
    import httpx

    from engine import llm
    monkeypatch.setattr(ledger, "_DEFAULT", tmp_path / "ledger.csv")
    monkeypatch.setattr(llm, "SPEND_FILE", tmp_path / "spend.json")
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    message = ("Your credit balance is too low to access the Anthropic API. "
               "Please go to Plans & Billing to upgrade or purchase credits.")
    body = {"type": "error",
            "error": {"type": "invalid_request_error", "message": message}}
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    credits_gone = anthropic.BadRequestError(
        f"Error code: 400 - {body}",
        response=httpx.Response(400, request=request), body=body)

    def broken_api(*args, **kwargs):
        raise credits_gone
    monkeypatch.setattr(llm, "_call_api", broken_api)
    # The real cycle and the real llm.ask, fed fixture markets instead
    # of a live Kalshi fetch.
    cards = parse_events(json.loads(FIXTURE.read_text()))
    monkeypatch.setattr(runner, "one_cycle", functools.partial(
        runner.one_cycle, cards=cards, now_iso="2026-07-15T00:00:00Z",
        snapshot_path=tmp_path / "snap.json"))

    with pytest.raises(SystemExit) as excinfo:
        runner.main()
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert out.count("::error::") == 1
    line = next(l for l in out.splitlines() if l.startswith("::error::"))
    assert line.startswith("::error::Anthropic API access is off "
                           "(out of credits or bad key).")
    assert "credit balance is too low" in line       # the plain reason


def test_main_leaves_other_crashes_alone(monkeypatch):
    """Only the account-level error gets the friendly line. Any other
    crash keeps its full traceback, exactly as before."""
    def boom():
        raise ValueError("a real bug")
    monkeypatch.setattr(runner, "one_cycle", boom)
    with pytest.raises(ValueError, match="a real bug"):
        runner.main()
