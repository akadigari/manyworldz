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
