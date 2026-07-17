import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verdict_m0 import evaluate


def test_clean_pass_is_go():
    v = evaluate(n_games=400, audit_errors=0, audit_n=50, worst_reid_rate=0.31)
    assert v["go"] is True
    assert v["demote_precutoff"] is True  # 31% leak still demotes: separate axis


def test_too_few_games_is_no_go():
    v = evaluate(n_games=349, audit_errors=0, audit_n=50, worst_reid_rate=0.0)
    assert v["go"] is False
    assert any("350" in r for r in v["reasons"])


def test_dirty_join_is_no_go():
    v = evaluate(n_games=400, audit_errors=2, audit_n=50, worst_reid_rate=0.0)
    assert v["go"] is False  # 4% error rate > 1% gate


def test_low_leak_does_not_demote():
    v = evaluate(n_games=400, audit_errors=0, audit_n=50, worst_reid_rate=0.05)
    assert v["demote_precutoff"] is False
