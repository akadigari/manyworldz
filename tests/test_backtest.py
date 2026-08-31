"""backtest.py: score the bot against questions Metaculus has already
resolved, so tuning stops being guesswork.

The whole exercise is worthless if the bot can see the answer, so the
leakage controls get as much test weight as the scoring math.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest
import config
from adapters import metaculus


def resolved_payload(**over):
    post = {
        "id": 41001,
        "title": "Will X happen by September?",
        "question": {
            "id": 700001, "type": "binary",
            "title": "Will X happen by September?",
            "description": "background", "resolution_criteria": "criteria",
            "fine_print": "", "resolution": "yes",
            "actual_resolve_time": "2026-08-20T00:00:00Z",
            "scheduled_close_time": "2026-08-19T00:00:00Z",
            "aggregations": {"recency_weighted": {"latest": {"centers": [0.7]}}},
        },
    }
    post["question"].update(over)
    return {"results": [post]}


# --- parsing --------------------------------------------------------------

def test_parse_resolved_reads_outcome_and_community():
    cards = metaculus.parse_resolved(resolved_payload())
    assert len(cards) == 1
    card = cards[0]
    assert card["qid"] == 700001
    assert card["outcome"] == 1.0            # "yes" -> 1
    assert card["community"] == 0.7
    assert card["resolved_at"].startswith("2026-08-20")


def test_parse_resolved_maps_no_to_zero():
    assert metaculus.parse_resolved(resolved_payload(resolution="no"))[0]["outcome"] == 0.0


def test_parse_resolved_skips_annulled_and_ambiguous():
    for bad in ("annulled", "ambiguous", None, ""):
        assert metaculus.parse_resolved(resolved_payload(resolution=bad)) == []


def test_parse_resolved_survives_a_missing_community_prediction():
    card = metaculus.parse_resolved(resolved_payload(aggregations={}))[0]
    assert card["community"] is None         # recorded, not invented


# --- leakage controls -----------------------------------------------------

def test_only_questions_resolved_after_the_model_cutoff_are_used():
    old = dict(qid=1, outcome=1.0, resolved_at="2025-01-01T00:00:00Z")
    new = dict(qid=2, outcome=1.0, resolved_at="2026-08-20T00:00:00Z")
    kept = backtest.after_cutoff([old, new])
    assert [c["qid"] for c in kept] == [2]


def test_cutoff_filter_uses_the_configured_date_not_a_literal():
    just_after = dict(qid=3, outcome=1.0,
                      resolved_at=config.MODEL_CUTOFF_DATE + "T00:00:01Z")
    just_before = dict(qid=4, outcome=1.0,
                       resolved_at="2025-07-31T23:59:59Z")
    assert [c["qid"] for c in backtest.after_cutoff([just_after, just_before])] == [3]


def test_a_card_with_no_resolve_time_is_dropped_not_assumed_recent():
    assert backtest.after_cutoff([dict(qid=5, outcome=1.0, resolved_at=None)]) == []


def test_the_run_never_researches(monkeypatch):
    """AskNews searches CURRENT news. On a question that resolved last
    week it hands the model the answer, so a backtest that researches
    scores beautifully and means nothing."""
    called = []
    from engine import news as news_mod
    monkeypatch.setattr(news_mod, "headlines_for",
                        lambda *a, **k: called.append(a) or ["LEAKED: X happened"])
    card = {"qid": 9, "post_id": 1, "qtype": "binary", "question": "Will X?",
            "criteria": "", "outcome": 1.0, "community": 0.6,
            "resolved_at": "2026-08-20T00:00:00Z"}
    backtest.score_one(card, ask=lambda p, model=None, max_tokens=400:
                       '{"probability": 0.8, "reason": "r"}')
    assert called == [], "backtest must not call the news layer at all"


def test_scored_rows_record_that_research_was_off():
    card = {"qid": 9, "post_id": 1, "qtype": "binary", "question": "Will X?",
            "criteria": "", "outcome": 1.0, "community": 0.6,
            "resolved_at": "2026-08-20T00:00:00Z"}
    row = backtest.score_one(card, ask=lambda p, model=None, max_tokens=400:
                             '{"probability": 0.8, "reason": "r"}')
    assert row["research"] == "disabled"


# --- scoring math ---------------------------------------------------------

def test_brier_is_squared_error():
    assert backtest.brier(0.8, 1.0) == pytest_approx(0.04)
    assert backtest.brier(0.3, 0.0) == pytest_approx(0.09)


def test_log_score_punishes_confident_and_wrong():
    confident_right = backtest.log_score(0.95, 1.0)
    confident_wrong = backtest.log_score(0.95, 0.0)
    assert confident_wrong > confident_right * 10


def test_log_score_is_finite_even_at_zero():
    assert backtest.log_score(0.0, 1.0) < 100        # clipped, never infinite


def test_summary_reports_the_gap_to_the_community():
    rows = [{"prob": 0.8, "outcome": 1.0, "community": 0.6, "qtype": "binary"},
            {"prob": 0.2, "outcome": 0.0, "community": 0.4, "qtype": "binary"}]
    s = backtest.summarize(rows)
    assert s["n"] == 2
    assert s["brier"] == pytest_approx(0.04)
    assert s["community_brier"] == pytest_approx(0.16)
    assert s["beat_community"] is True


def test_summary_handles_rows_with_no_community_number():
    rows = [{"prob": 0.8, "outcome": 1.0, "community": None, "qtype": "binary"}]
    s = backtest.summarize(rows)
    assert s["community_brier"] is None
    assert s["beat_community"] is None


def test_calibration_buckets_count_what_actually_happened():
    rows = [{"prob": 0.9, "outcome": 1.0, "community": None, "qtype": "binary"},
            {"prob": 0.9, "outcome": 0.0, "community": None, "qtype": "binary"}]
    s = backtest.summarize(rows)
    bucket = [b for b in s["calibration"] if b["bucket"] == "0.9-1.0"][0]
    assert bucket["n"] == 2 and bucket["hit_rate"] == pytest_approx(0.5)


# --- the dry run spends nothing ------------------------------------------

def test_dry_run_makes_no_model_calls():
    calls = []
    cards = [{"qid": 1, "post_id": 1, "qtype": "binary", "question": "Will X?",
              "criteria": "", "outcome": 1.0, "community": 0.5,
              "resolved_at": "2026-08-20T00:00:00Z"}]
    out = backtest.run(cards, ask=lambda *a, **k: calls.append(a) or "{}",
                       dry_run=True)
    assert calls == []
    assert out["summary"]["n"] == 0
    assert out["would_score"] == 1


def pytest_approx(x, tol=1e-9):
    class _A:
        def __eq__(self, other): return abs(other - x) < tol
        def __repr__(self): return f"~{x}"
    return _A()


def test_backtest_meters_into_its_own_spend_file(monkeypatch):
    """Same lesson as the Kalshi loop: a backtest that spends from the
    tournament's budget can silence a live round."""
    monkeypatch.delenv("MANYWORLDZ_SPEND_FILE", raising=False)
    from engine import llm as llm_mod
    backtest._use_own_spend_meter()
    assert llm_mod.SPEND_FILE.name == "spend_backtest.json"


def test_backtest_spend_file_still_honors_an_explicit_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MANYWORLDZ_SPEND_FILE", str(tmp_path / "custom.json"))
    from engine import llm as llm_mod
    backtest._use_own_spend_meter()
    assert llm_mod.SPEND_FILE == tmp_path / "custom.json"


def test_inspect_report_describes_the_shape_without_the_api():
    report = backtest.inspect_report(resolved_payload())
    assert "1 raw posts" in report
    assert "resolution: yes" in report
    assert "1 scoreable" in report
    assert "actual_resolve_time" in report      # the field the cutoff needs


def test_inspect_report_says_so_when_nothing_parses():
    report = backtest.inspect_report(resolved_payload(resolution="annulled"))
    assert "0 scoreable" in report
