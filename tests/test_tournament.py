import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import tournament


def cards(n=2):
    return [{"qid": 500000 + i, "question": f"Will thing {i} happen?",
            "close_time": "2026-12-31T00:00:00Z",
            "url": f"https://www.metaculus.com/questions/{i}/"}
           for i in range(n)]


CONFIDENT = '{"probability": 0.71, "reason": "seems likely"}'
FUTURES = ('{"futures": ['
          '{"story": "it happens on schedule", "resolves": "YES"},'
          '{"story": "a delay but it lands", "resolves": "YES"},'
          '{"story": "momentum carries it through", "resolves": "YES"},'
          '{"story": "an early surprise seals it", "resolves": "YES"},'
          '{"story": "something blocks it", "resolves": "NO"}]}')
ALL_YES_FUTURES = ('{"futures": ['
                   '{"story": "a", "resolves": "YES"},'
                   '{"story": "b", "resolves": "YES"},'
                   '{"story": "c", "resolves": "YES"},'
                   '{"story": "d", "resolves": "YES"},'
                   '{"story": "e", "resolves": "YES"}]}')
ALL_NO_FUTURES = ('{"futures": ['
                  '{"story": "a", "resolves": "NO"},'
                  '{"story": "b", "resolves": "NO"},'
                  '{"story": "c", "resolves": "NO"},'
                  '{"story": "d", "resolves": "NO"},'
                  '{"story": "e", "resolves": "NO"}]}')

NOW = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)
NOW_ISO = NOW.isoformat()


def ask_futures(p, model=None, max_tokens=400):
    return FUTURES


def near_close_card(qid, hours_to_close):
    close = (NOW + timedelta(hours=hours_to_close)).isoformat()
    return {"qid": qid, "question": f"Will thing {qid} happen?",
           "close_time": close, "url": f"https://www.metaculus.com/questions/{qid}/"}


def seed_log(log_path, qid, at_iso, prob=0.6, source="crowd"):
    """Write one prior "answered" row straight through _append_log, so
    the seeded row always matches whatever LOG_COLUMNS actually is."""
    tournament._append_log(
        {"qid": qid, "question": f"Will thing {qid} happen?",
         "raw_prob": prob, "prob": prob, "at": at_iso, "source": source},
        log_path)


def test_dry_run_makes_no_post_and_prints_preview(tmp_path, capsys):
    calls = []
    out = tournament.one_cycle(
        cards=cards(1), ask_fn=ask_futures, dry_run=True, token="tok",
        log_path=tmp_path / "log.csv",
        submit_fn=lambda qid, prob, token: calls.append((qid, prob)))
    assert calls == []                              # no POST at all
    assert out["submitted"] == 0
    assert out["answered"] == 1
    assert not (tmp_path / "log.csv").exists()       # nothing was actually submitted
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out


def test_tokenless_main_exits_politely_without_network_calls(monkeypatch, capsys):
    monkeypatch.delenv("METACULUS_TOKEN", raising=False)
    fetch_calls = []
    monkeypatch.setattr(tournament.metaculus, "fetch_open_questions",
                        lambda *a, **k: fetch_calls.append(1) or [])
    monkeypatch.setattr(sys, "argv", ["tournament.py"])
    tournament.main()             # must not raise
    captured = capsys.readouterr()
    assert "METACULUS_TOKEN is not set" in captured.out
    assert fetch_calls == []      # never even tried to hit the network


def test_one_cycle_logs_every_submission_with_expected_columns(tmp_path):
    log_path = tmp_path / "log.csv"
    posted = []
    out = tournament.one_cycle(
        cards=cards(2), ask_fn=ask_futures, token="tok", log_path=log_path,
        submit_fn=lambda qid, prob, token: posted.append((qid, prob, token)))
    assert out["submitted"] == 2
    assert len(posted) == 2
    rows = list(csv.DictReader(open(log_path, newline="")))
    assert len(rows) == 2
    # New columns from the tournament hardening: "raw_prob" (the crowd's
    # own number before the tournament clip) and "source" (which tier of
    # the fallback ladder answered this question).
    assert set(rows[0].keys()) == {"qid", "question", "raw_prob", "prob", "at", "source"}
    assert rows[0]["qid"] == "500000"
    assert float(rows[0]["prob"]) > 0
    assert rows[0]["source"] == "crowd"


def test_one_cycle_skips_questions_already_answered_this_run(tmp_path):
    log_path = tmp_path / "log.csv"
    posted = []
    submit_fn = lambda qid, prob, token: posted.append(qid)

    tournament.one_cycle(cards=cards(2), ask_fn=ask_futures, token="tok",
                         log_path=log_path, submit_fn=submit_fn)
    assert len(posted) == 2

    # Same two questions again: both already have a log row, neither
    # should get answered or submitted a second time.
    out = tournament.one_cycle(cards=cards(2), ask_fn=ask_futures, token="tok",
                               log_path=log_path, submit_fn=submit_fn)
    assert out["answered"] == 0
    assert out["submitted"] == 0
    assert len(posted) == 2                          # unchanged


def test_budget_error_propagates_and_preserves_partial_log(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 2)   # small, deterministic crowd
    log_path = tmp_path / "log.csv"
    posted = []

    calls = {"n": 0}
    def ask_then_blow_budget(p, model=None, max_tokens=400):
        calls["n"] += 1
        if calls["n"] > 2:          # first card's whole (2-agent) crowd answers fine
            raise RuntimeError("engine budget cap hit ($10.00)")
        return FUTURES

    with pytest.raises(RuntimeError, match="budget cap hit"):
        tournament.one_cycle(
            cards=cards(2), ask_fn=ask_then_blow_budget, token="tok",
            log_path=log_path,
            submit_fn=lambda qid, prob, token: posted.append(qid))

    # The first question's submission must have already landed on disk
    # before the second question's budget error blew up the cycle.
    assert posted == [500000]
    rows = list(csv.DictReader(open(log_path, newline="")))
    assert [r["qid"] for r in rows] == ["500000"]


def test_no_quorum_question_is_skipped_not_submitted(tmp_path):
    log_path = tmp_path / "log.csv"
    posted = []
    out = tournament.one_cycle(
        cards=cards(1), ask_fn=lambda p, model=None, max_tokens=400: "garbage",
        token="tok", log_path=log_path,
        submit_fn=lambda qid, prob, token: posted.append(qid))
    assert posted == []
    assert out["submitted"] == 0
    assert not log_path.exists()


def test_clamp_never_returns_zero_or_one():
    assert tournament._clamp(1.0) == 0.99
    assert tournament._clamp(0.0) == 0.01
    assert tournament._clamp(0.5) == 0.5


def test_questions_per_run_caps_how_many_get_answered(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "TOURNAMENT_QUESTIONS_PER_RUN", 1)
    posted = []
    out = tournament.one_cycle(
        cards=cards(3), ask_fn=ask_futures, token="tok",
        log_path=tmp_path / "log.csv",
        submit_fn=lambda qid, prob, token: posted.append(qid))
    assert out["considered"] == 3
    assert out["answered"] == 1
    assert len(posted) == 1


# ---- fallback ladder (never lose a question) ----

def test_crowd_failure_falls_back_to_a_single_run(tmp_path):
    """The full crowd blows up on its first call; the single-run retry
    (a plain vote, one more ask_fn call) gets a usable answer instead."""
    log_path = tmp_path / "log.csv"
    posted = []
    calls = {"n": 0}

    def ask_fn(p, model=None, max_tokens=400):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("simulated network blip")
        return CONFIDENT

    out = tournament.one_cycle(
        cards=cards(1), ask_fn=ask_fn, token="tok", log_path=log_path,
        submit_fn=lambda qid, prob, token: posted.append((qid, prob)))
    assert out["submitted"] == 1
    assert out["fallbacks"] == 0
    rows = list(csv.DictReader(open(log_path, newline="")))
    assert rows[0]["source"] == "single"
    assert float(rows[0]["raw_prob"]) == pytest.approx(0.71)


def test_crowd_and_single_both_fail_records_documented_fallback(tmp_path):
    """Every ask_fn call raises: the full crowd fails, the single-run
    retry fails too, and the ladder submits the documented 0.5 with
    source="fallback" instead of losing the question."""
    log_path = tmp_path / "log.csv"
    posted = []

    def ask_fn(p, model=None, max_tokens=400):
        raise ConnectionError("simulated total outage")

    out = tournament.one_cycle(
        cards=cards(1), ask_fn=ask_fn, token="tok", log_path=log_path,
        submit_fn=lambda qid, prob, token: posted.append((qid, prob)))
    assert out["submitted"] == 1
    assert out["fallbacks"] == 1
    assert posted == [(500000, tournament.LAST_RESORT_PROBABILITY)]
    rows = list(csv.DictReader(open(log_path, newline="")))
    assert rows[0]["source"] == "fallback"
    assert float(rows[0]["raw_prob"]) == 0.5
    assert float(rows[0]["prob"]) == 0.5


def test_budget_error_during_single_retry_still_stops_cleanly(tmp_path):
    """A budget error hit while trying the single-run retry (tier 2)
    must still propagate straight out, same as tier 1: no fallback-spam
    just because the ladder already moved to a lower tier."""
    log_path = tmp_path / "log.csv"
    posted = []
    calls = {"n": 0}

    def ask_fn(p, model=None, max_tokens=400):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("simulated blip")     # crowd tier: a real, non-budget failure
        raise RuntimeError("engine budget cap hit ($10.00)")  # single tier: the cap

    with pytest.raises(RuntimeError, match="budget cap hit"):
        tournament.one_cycle(
            cards=cards(1), ask_fn=ask_fn, token="tok", log_path=log_path,
            submit_fn=lambda qid, prob, token: posted.append(qid))
    assert posted == []
    assert not log_path.exists()


def test_no_quorum_still_skips_without_raising_anything(tmp_path):
    """A clean but unusable crowd result (every agent's answer failed
    to parse, no exception anywhere) stays a skip: there is nothing
    broken here to retry into being different, unlike a real crash."""
    log_path = tmp_path / "log.csv"
    posted = []
    out = tournament.one_cycle(
        cards=cards(1), ask_fn=lambda p, model=None, max_tokens=400: "garbage",
        token="tok", log_path=log_path,
        submit_fn=lambda qid, prob, token: posted.append(qid))
    assert posted == []
    assert out["submitted"] == 0
    assert out["fallbacks"] == 0
    assert not log_path.exists()


# ---- TOURNAMENT_CLIP (clip the extremes) ----

def test_tournament_clip_bounds_both_ends():
    import config
    assert tournament._tournament_clip(0.999) == 1 - config.TOURNAMENT_CLIP
    assert tournament._tournament_clip(0.001) == config.TOURNAMENT_CLIP
    assert tournament._tournament_clip(0.5) == 0.5


def test_extreme_high_crowd_probability_clipped_at_submit_not_at_raw(tmp_path):
    log_path = tmp_path / "log.csv"
    posted = []
    tournament.one_cycle(
        cards=cards(1), ask_fn=lambda p, model=None, max_tokens=400: ALL_YES_FUTURES,
        token="tok", log_path=log_path,
        submit_fn=lambda qid, prob, token: posted.append(prob))
    rows = list(csv.DictReader(open(log_path, newline="")))
    assert float(rows[0]["raw_prob"]) == 0.99      # the crowd's own honest number
    assert float(rows[0]["prob"]) == 0.98          # clipped for submission
    assert posted == [0.98]


def test_extreme_low_crowd_probability_clipped_at_submit_not_at_raw(tmp_path):
    log_path = tmp_path / "log.csv"
    posted = []
    tournament.one_cycle(
        cards=cards(1), ask_fn=lambda p, model=None, max_tokens=400: ALL_NO_FUTURES,
        token="tok", log_path=log_path,
        submit_fn=lambda qid, prob, token: posted.append(prob))
    rows = list(csv.DictReader(open(log_path, newline="")))
    assert float(rows[0]["raw_prob"]) == 0.01
    assert float(rows[0]["prob"]) == 0.02
    assert posted == [0.02]


def test_ordinary_answer_has_matching_raw_and_submitted_prob(tmp_path):
    """A probability that never gets near the extremes should come
    through the clip unchanged: raw_prob and prob match."""
    log_path = tmp_path / "log.csv"
    tournament.one_cycle(cards=cards(1), ask_fn=ask_futures, token="tok",
                         log_path=log_path,
                         submit_fn=lambda qid, prob, token: None)
    rows = list(csv.DictReader(open(log_path, newline="")))
    assert rows[0]["raw_prob"] == rows[0]["prob"]


# ---- refresh near close ----

def test_refresh_picks_up_a_stale_answer_on_a_soon_closing_question(tmp_path):
    log_path = tmp_path / "log.csv"
    seed_log(log_path, 700001, (NOW - timedelta(hours=10)).isoformat())
    card = near_close_card(700001, hours_to_close=5)   # well inside the 24h refresh window

    out = tournament.one_cycle(
        cards=[card], ask_fn=ask_futures, token="tok", log_path=log_path,
        now_iso=NOW_ISO, submit_fn=lambda qid, prob, token: None)
    assert out["refreshed"] == 1
    assert out["answered"] == 0             # this is a resubmit, not a fresh answer
    rows = list(csv.DictReader(open(log_path, newline="")))
    assert len(rows) == 2
    assert rows[1]["qid"] == "700001"
    assert rows[1]["at"] == NOW_ISO


def test_refresh_skips_a_question_whose_close_is_far_away(tmp_path):
    log_path = tmp_path / "log.csv"
    seed_log(log_path, 700002, (NOW - timedelta(hours=10)).isoformat())
    card = near_close_card(700002, hours_to_close=24 * 30)   # a month out

    out = tournament.one_cycle(
        cards=[card], ask_fn=ask_futures, token="tok", log_path=log_path,
        now_iso=NOW_ISO, submit_fn=lambda qid, prob, token: None)
    assert out["refreshed"] == 0
    rows = list(csv.DictReader(open(log_path, newline="")))
    assert len(rows) == 1                   # nothing new appended


def test_refresh_skips_a_still_fresh_submission(tmp_path):
    log_path = tmp_path / "log.csv"
    seed_log(log_path, 700003, (NOW - timedelta(hours=2)).isoformat())   # only 2h old
    card = near_close_card(700003, hours_to_close=5)

    out = tournament.one_cycle(
        cards=[card], ask_fn=ask_futures, token="tok", log_path=log_path,
        now_iso=NOW_ISO, submit_fn=lambda qid, prob, token: None)
    assert out["refreshed"] == 0
    rows = list(csv.DictReader(open(log_path, newline="")))
    assert len(rows) == 1


def test_refresh_respects_the_cap_and_prefers_the_soonest_close(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "TOURNAMENT_REFRESH_CAP", 2)
    log_path = tmp_path / "log.csv"
    qids = [700010, 700011, 700012, 700013]
    for qid in qids:
        seed_log(log_path, qid, (NOW - timedelta(hours=10)).isoformat())
    # Closes in 1h, 2h, 3h, 4h: the cap of 2 should keep the two soonest.
    stale_cards = [near_close_card(qid, hours_to_close=i + 1) for i, qid in enumerate(qids)]

    out = tournament.one_cycle(
        cards=stale_cards, ask_fn=ask_futures, token="tok", log_path=log_path,
        now_iso=NOW_ISO, submit_fn=lambda qid, prob, token: None)
    assert out["refreshed"] == 2
    rows = list(csv.DictReader(open(log_path, newline="")))
    refreshed_qids = {r["qid"] for r in rows if r["at"] == NOW_ISO}
    assert refreshed_qids == {"700010", "700011"}


def test_refresh_goes_through_the_fallback_ladder_too(tmp_path):
    """A refresh whose crowd run fails still gets the same ladder as a
    fresh question: it degrades to a single run instead of being lost."""
    log_path = tmp_path / "log.csv"
    seed_log(log_path, 700020, (NOW - timedelta(hours=10)).isoformat())
    card = near_close_card(700020, hours_to_close=5)
    calls = {"n": 0}

    def ask_fn(p, model=None, max_tokens=400):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("simulated blip")
        return CONFIDENT

    out = tournament.one_cycle(
        cards=[card], ask_fn=ask_fn, token="tok", log_path=log_path,
        now_iso=NOW_ISO, submit_fn=lambda qid, prob, token: None)
    assert out["refreshed"] == 1
    rows = list(csv.DictReader(open(log_path, newline="")))
    assert rows[1]["source"] == "single"


# ---- coverage report ----

def test_coverage_summary_line_reports_all_the_required_pieces(tmp_path, capsys):
    log_path = tmp_path / "log.csv"
    tournament.one_cycle(cards=cards(1), ask_fn=ask_futures, token="tok",
                         log_path=log_path,
                         submit_fn=lambda qid, prob, token: None)
    out = capsys.readouterr().out
    summary = [line for line in out.splitlines()
              if line.startswith("tournament cycle done")]
    assert len(summary) == 1
    line = summary[0]
    assert "open" in line
    assert "answered" in line
    assert "refreshed" in line
    assert "fallback" in line
    assert "answered all time" in line
    assert "spent" in line


# ---- new config knobs ----

def test_new_tournament_config_knobs_have_documented_defaults():
    import config
    assert config.TOURNAMENT_CLIP == 0.02
    assert config.TOURNAMENT_REFRESH_HOURS == 24
    assert config.TOURNAMENT_REFRESH_CAP == 10
