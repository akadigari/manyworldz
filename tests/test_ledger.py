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


def test_grade_keeps_old_values_when_mid_is_unknown(tmp_path):
    # A one-sided book means fetch_market() couldn't report a real price
    # and sends mid=None. grade() must leave latest_mid/clv_cents alone
    # instead of overwriting them with a made-up number.
    path = tmp_path / "ledger.csv"
    ledger.log_pick(pick("T1", "YES"), path=path)   # starts at latest_mid 43, clv 0
    latest = {"T1": {"ticker": "T1", "mid": None, "status": "active", "result": ""}}
    stats = ledger.grade(latest, path=path)
    rows = {r["ticker"]: r for r in ledger.load(path=path)}
    assert int(rows["T1"]["latest_mid"]) == 43
    assert int(rows["T1"]["clv_cents"]) == 0
    assert rows["T1"]["status"] == "open"
    assert stats["updated"] == 1
