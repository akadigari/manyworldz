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


def test_tokenless_main_exits_quietly_before_the_arm_by_date(monkeypatch, capsys):
    """Still no network calls -- but now it exits 0 *deliberately*, not by default.

    Pinned to a far-future arm-by so this test doesn't start failing on its
    own once the real date passes.
    """
    from datetime import datetime, timezone
    monkeypatch.setattr(tournament.config, "TOURNAMENT_ARM_BY",
                        datetime(2099, 1, 1, tzinfo=timezone.utc))
    monkeypatch.delenv("METACULUS_TOKEN", raising=False)
    fetch_calls = []
    monkeypatch.setattr(tournament.metaculus, "fetch_open_questions",
                        lambda *a, **k: fetch_calls.append(1) or [])
    monkeypatch.setattr(sys, "argv", ["tournament.py"])

    with pytest.raises(SystemExit) as excinfo:
        tournament.main()

    assert excinfo.value.code == 0          # success: waiting, not broken
    captured = capsys.readouterr()
    assert "METACULUS_TOKEN is not set" in captured.out
    assert fetch_calls == []                # never even tried to hit the network


def test_tokenless_main_fails_loudly_after_the_arm_by_date(monkeypatch, capsys):
    from datetime import datetime, timezone
    monkeypatch.setattr(tournament.config, "TOURNAMENT_ARM_BY",
                        datetime(2020, 1, 1, tzinfo=timezone.utc))
    monkeypatch.delenv("METACULUS_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", ["tournament.py"])

    with pytest.raises(SystemExit) as excinfo:
        tournament.main()

    assert excinfo.value.code == 1
    assert "::error::" in capsys.readouterr().out


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
    assert set(rows[0].keys()) == {"qid", "question", "qtype", "raw_prob",
                                   "prob", "at", "source"}
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

def test_already_answered_question_is_never_resubmitted_even_near_close(tmp_path):
    """Tournament rule (resources page, 2026-08-20): bot makers should
    only submit ONE forecast per question. A stale answer on a
    soon-closing question stays as it is."""
    log_path = tmp_path / "log.csv"
    seed_log(log_path, 700001, (NOW - timedelta(hours=10)).isoformat())
    card = near_close_card(700001, hours_to_close=1)

    out = tournament.one_cycle(
        cards=[card], ask_fn=ask_futures, token="tok", log_path=log_path,
        now_iso=NOW_ISO, submit_fn=lambda qid, prob, token: None)
    assert out["answered"] == 0
    rows = list(csv.DictReader(open(log_path, newline="")))
    assert len(rows) == 1                   # nothing new appended


def test_api_side_already_forecast_flag_skips_even_with_an_empty_log(tmp_path):
    """The API's own my_forecasts record is the truth about what this
    account answered; a fresh checkout with an empty CSV must not
    re-answer a question the account already forecast."""
    log_path = tmp_path / "log.csv"
    card = dict(cards(1)[0], already_forecast=True)

    out = tournament.one_cycle(
        cards=[card], ask_fn=ask_futures, token="tok", log_path=log_path,
        submit_fn=lambda qid, prob, token: None)
    assert out["answered"] == 0
    assert not log_path.exists()


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
    assert "fallback" in line
    assert "answered all time" in line
    assert "spent" in line


# ---- new config knobs ----

def test_new_tournament_config_knobs_have_documented_defaults():
    import config
    assert config.TOURNAMENT_CLIP == 0.02
    assert not hasattr(config, "TOURNAMENT_REFRESH_HOURS")   # one forecast per question
    assert not hasattr(config, "TOURNAMENT_REFRESH_CAP")
    assert config.QUESTION_DEADLINE_S == 300


# -- the unconfigured state must not stay quiet forever --------------------
#
# tournament.py exits 0 when METACULUS_TOKEN is missing so the schedule can
# stay active before the account exists. That is right for a few days and
# wrong for a month: it ran ~2,000 green no-op cycles from 2026-07-20 while
# the Summer season closed. After the arm-by date, silence becomes a failure.

from datetime import datetime, timezone

from tournament import token_state


def _t(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_token_present_is_armed():
    state = token_state("secret", now=_t("2026-08-18"), arm_by=_t("2026-09-01"))

    assert state.status == "armed"
    assert state.exit_code == 0


def test_missing_token_before_the_deadline_waits_quietly():
    state = token_state(None, now=_t("2026-08-18"), arm_by=_t("2026-09-01"))

    assert state.status == "waiting"
    assert state.exit_code == 0
    assert "14 days" in state.message


def test_missing_token_after_the_deadline_fails_loudly():
    state = token_state(None, now=_t("2026-09-02"), arm_by=_t("2026-09-01"))

    assert state.status == "overdue"
    assert state.exit_code == 1
    assert "METACULUS_TOKEN" in state.message


def test_waiting_message_names_the_action():
    state = token_state(None, now=_t("2026-08-18"), arm_by=_t("2026-09-01"))

    assert "METACULUS_TOKEN" in state.message


# ---- criteria in the prompt ----

def test_resolution_criteria_reach_the_model_prompt(tmp_path):
    seen = {}
    def ask_fn(prompt, model=None, max_tokens=400):
        seen.setdefault("prompt", prompt)
        return FUTURES
    card = dict(cards(1)[0],
                criteria="Resolution criteria: resolves YES only if 20 states sign.")
    tournament.one_cycle(cards=[card], ask_fn=ask_fn, token="tok",
                         log_path=tmp_path / "log.csv",
                         submit_fn=lambda qid, prob, token: None)
    assert "20 states sign" in seen["prompt"]


# ---- multiple choice ----

MC_CARD = {"qid": 600001, "post_id": 31001, "qtype": "multiple_choice",
           "question": "Which country hosts the summit?",
           "options": ["France", "United Kingdom", "Other"],
           "criteria": "", "close_time": "2026-12-31T00:00:00Z",
           "url": "https://www.metaculus.com/questions/31001/",
           "already_forecast": False}

MC_REPLY = '{"France": 0.5, "United Kingdom": 0.3, "Other": 0.2}'


def test_multiple_choice_question_is_answered_and_submitted(tmp_path):
    sent = {}
    def submit_mc(qid, probs, token):
        sent["qid"], sent["probs"] = qid, probs
    out = tournament.one_cycle(
        cards=[dict(MC_CARD)], ask_fn=lambda p, model=None, max_tokens=400: MC_REPLY,
        token="tok", log_path=tmp_path / "log.csv",
        submit_fn=lambda qid, prob, token: None, submit_mc_fn=submit_mc)
    assert out["answered"] == 1
    assert sent["qid"] == 600001
    assert set(sent["probs"]) == {"France", "United Kingdom", "Other"}
    assert abs(sum(sent["probs"].values()) - 1.0) < 1e-6
    rows = list(csv.DictReader(open(tmp_path / "log.csv", newline="")))
    assert rows[0]["qtype"] == "multiple_choice"


def test_multiple_choice_garbage_reply_submits_uniform_fallback(tmp_path):
    sent = {}
    def submit_mc(qid, probs, token):
        sent["probs"] = probs
    out = tournament.one_cycle(
        cards=[dict(MC_CARD)], ask_fn=lambda p, model=None, max_tokens=400: "no json",
        token="tok", log_path=tmp_path / "log.csv",
        submit_fn=lambda qid, prob, token: None, submit_mc_fn=submit_mc)
    assert out["fallbacks"] == 1
    vals = list(sent["probs"].values())
    assert max(vals) - min(vals) < 1e-6      # uniform across the options


# ---- numeric / discrete ----

NUM_CARD = {"qid": 600002, "post_id": 31002, "qtype": "numeric",
            "question": "How many states sign?", "criteria": "",
            "close_time": "2026-12-31T00:00:00Z",
            "url": "https://www.metaculus.com/questions/31002/",
            "already_forecast": False, "unit": "states",
            "open_lower_bound": False, "open_upper_bound": True,
            "scaling": {"range_min": 0, "range_max": 60,
                        "inbound_outcome_count": 200, "continuous_range": None}}

NUM_REPLY = '{"p05": 5, "p25": 15, "p50": 30, "p75": 45, "p95": 55}'


def test_numeric_question_is_answered_with_a_valid_cdf(tmp_path):
    sent = {}
    def submit_num(qid, cdf, token):
        sent["qid"], sent["cdf"] = qid, cdf
    out = tournament.one_cycle(
        cards=[dict(NUM_CARD)], ask_fn=lambda p, model=None, max_tokens=400: NUM_REPLY,
        token="tok", log_path=tmp_path / "log.csv",
        submit_fn=lambda qid, prob, token: None, submit_numeric_fn=submit_num)
    assert out["answered"] == 1
    assert sent["qid"] == 600002
    assert len(sent["cdf"]) == 201
    assert all(a <= b for a, b in zip(sent["cdf"], sent["cdf"][1:]))
    rows = list(csv.DictReader(open(tmp_path / "log.csv", newline="")))
    assert rows[0]["qtype"] == "numeric"


def test_numeric_garbage_reply_submits_uniform_fallback(tmp_path):
    sent = {}
    def submit_num(qid, cdf, token):
        sent["cdf"] = cdf
    out = tournament.one_cycle(
        cards=[dict(NUM_CARD)], ask_fn=lambda p, model=None, max_tokens=400: "??",
        token="tok", log_path=tmp_path / "log.csv",
        submit_fn=lambda qid, prob, token: None, submit_numeric_fn=submit_num)
    assert out["fallbacks"] == 1
    assert len(sent["cdf"]) == 201           # a flat, honest CDF still covers


# ---- the mandatory comment ----

def test_every_submission_gets_a_private_comment(tmp_path):
    comments = []
    def comment_fn(post_id, text, token):
        comments.append((post_id, text))
        return True
    card = dict(cards(1)[0], post_id=30099)
    tournament.one_cycle(
        cards=[card], ask_fn=ask_futures, token="tok",
        log_path=tmp_path / "log.csv",
        submit_fn=lambda qid, prob, token: None, comment_fn=comment_fn)
    assert len(comments) == 1
    post_id, text = comments[0]
    assert post_id == 30099
    assert "0.8" in text or "0.7" in text     # the submitted probability appears


def test_a_failing_comment_never_blocks_the_forecast(tmp_path):
    def comment_fn(post_id, text, token):
        raise RuntimeError("comment endpoint down")
    card = dict(cards(1)[0], post_id=30099)
    out = tournament.one_cycle(
        cards=[card], ask_fn=ask_futures, token="tok",
        log_path=tmp_path / "log.csv",
        submit_fn=lambda qid, prob, token: None, comment_fn=comment_fn)
    assert out["submitted"] == 1              # forecast still went out and logged


# ---- per-question deadline ----

def test_a_hung_model_call_degrades_to_the_fallback_ladder(tmp_path, monkeypatch):
    import config, time
    monkeypatch.setattr(config, "QUESTION_DEADLINE_S", 0.2)
    def slow_ask(p, model=None, max_tokens=400):
        time.sleep(0.6)
        return FUTURES
    t0 = time.monotonic()
    out = tournament.one_cycle(
        cards=cards(1), ask_fn=slow_ask, token="tok",
        log_path=tmp_path / "log.csv",
        submit_fn=lambda qid, prob, token: None)
    elapsed = time.monotonic() - t0
    assert out["fallbacks"] == 1              # deadline hit twice -> last resort
    assert out["submitted"] == 1
    assert elapsed < 5


# ---- review findings, 2026-08-23 ----

def test_one_bad_question_never_aborts_the_rest_of_the_cycle(tmp_path):
    """Finding 1: a submit rejection on one question (an API 400) must
    not zero out every question still left in the batch."""
    def submit_mc(qid, probs, token):
        raise RuntimeError("metaculus rejected the forecast for question 600001")
    good = dict(cards(1)[0], qid=600009)
    out = tournament.one_cycle(
        cards=[dict(MC_CARD), good],
        ask_fn=lambda p, model=None, max_tokens=400: (
            MC_REPLY if "option" in p.lower() else FUTURES),
        token="tok", log_path=tmp_path / "log.csv",
        submit_fn=lambda qid, prob, token: None, submit_mc_fn=submit_mc)
    assert out["submitted"] == 1              # the good binary still went out


def test_budget_error_still_aborts_the_cycle(tmp_path):
    """The budget cap is the one error that must still stop everything."""
    def ask_fn(p, model=None, max_tokens=400):
        raise RuntimeError("engine budget cap hit ($10.00)")
    with pytest.raises(RuntimeError, match="budget cap hit"):
        tournament.one_cycle(
            cards=cards(2), ask_fn=ask_fn, token="tok",
            log_path=tmp_path / "log.csv",
            submit_fn=lambda qid, prob, token: None)


def test_a_hung_mc_call_hits_the_deadline_and_falls_back(tmp_path, monkeypatch):
    """Finding 1: the per-question deadline must cover MC and numeric
    paths, not just the binary ladder."""
    import config, time
    monkeypatch.setattr(config, "QUESTION_DEADLINE_S", 0.2)
    sent = {}
    def slow_ask(p, model=None, max_tokens=400):
        time.sleep(0.6)
        return MC_REPLY
    t0 = time.monotonic()
    out = tournament.one_cycle(
        cards=[dict(MC_CARD)], ask_fn=slow_ask, token="tok",
        log_path=tmp_path / "log.csv",
        submit_fn=lambda qid, prob, token: None,
        submit_mc_fn=lambda qid, probs, token: sent.update(probs=probs))
    assert time.monotonic() - t0 < 3
    assert out["fallbacks"] == 1              # uniform fallback still submitted
    vals = list(sent["probs"].values())
    assert max(vals) - min(vals) < 1e-6


def test_numeric_question_with_missing_bounds_is_skipped_cleanly(tmp_path):
    """Finding 2: a malformed question (no range) is skipped, never a
    crash and never a made-up range."""
    bad = dict(NUM_CARD, scaling={"range_min": None, "range_max": None,
                                  "inbound_outcome_count": 200,
                                  "continuous_range": None})
    out = tournament.one_cycle(
        cards=[bad], ask_fn=lambda p, model=None, max_tokens=400: NUM_REPLY,
        token="tok", log_path=tmp_path / "log.csv",
        submit_fn=lambda qid, prob, token: None,
        submit_numeric_fn=lambda qid, cdf, token: None)
    assert out["submitted"] == 0
    assert out["answered"] == 0


def test_log_scaled_numeric_without_its_own_grid_is_skipped(tmp_path):
    """Finding 6: a zero_point question with no continuous_range would
    get a badly mis-shaped CDF from a linear grid; skip it honestly."""
    logq = dict(NUM_CARD, scaling={"range_min": 1, "range_max": 1000,
                                   "zero_point": 0.0,
                                   "inbound_outcome_count": 200,
                                   "continuous_range": None})
    out = tournament.one_cycle(
        cards=[logq], ask_fn=lambda p, model=None, max_tokens=400: NUM_REPLY,
        token="tok", log_path=tmp_path / "log.csv",
        submit_fn=lambda qid, prob, token: None,
        submit_numeric_fn=lambda qid, cdf, token: None)
    assert out["submitted"] == 0


def test_mc_question_with_no_options_is_skipped(tmp_path):
    """Finding 9: an MC card without options can only produce an empty
    payload the API would reject; skip it instead."""
    bad = dict(MC_CARD, options=[])
    out = tournament.one_cycle(
        cards=[bad], ask_fn=lambda p, model=None, max_tokens=400: MC_REPLY,
        token="tok", log_path=tmp_path / "log.csv",
        submit_fn=lambda qid, prob, token: None,
        submit_mc_fn=lambda qid, probs, token: None)
    assert out["submitted"] == 0


def test_old_four_column_log_is_migrated_before_appending(tmp_path):
    """Finding 5: the live log still has the pre-hardening header
    (qid,question,prob,at). Appending 7-field rows under it silently
    corrupts the ledger; the header and old rows must migrate first."""
    log_path = tmp_path / "log.csv"
    log_path.write_text("qid,question,prob,at\n"
                        "111,old question,0.6,2026-07-01T00:00:00+00:00\n")
    out = tournament.one_cycle(
        cards=cards(1), ask_fn=ask_futures, token="tok", log_path=log_path,
        submit_fn=lambda qid, prob, token: None)
    assert out["submitted"] == 1
    rows = list(csv.DictReader(open(log_path, newline="")))
    assert set(rows[0].keys()) == set(tournament.LOG_COLUMNS)
    old = rows[0]
    assert old["qid"] == "111"
    assert old["prob"] == "0.6"
    assert old["at"] == "2026-07-01T00:00:00+00:00"   # migrated, not shifted
    assert rows[1]["qtype"] == "binary"


def test_dry_run_counts_mc_and_numeric_fallbacks_too(tmp_path):
    """Finding 8: dry-run and live runs must count fallbacks the same
    way on every question type."""
    out = tournament.one_cycle(
        cards=[dict(MC_CARD)], ask_fn=lambda p, model=None, max_tokens=400: "junk",
        token="tok", dry_run=True, log_path=tmp_path / "log.csv")
    assert out["fallbacks"] == 1


def test_every_cycle_writes_a_status_file_for_the_daily_brief(tmp_path, monkeypatch):
    """The cloud briefer cannot reach metaculus.com (egress proxy), so
    the bot records what it saw: open-question count, tournament slug,
    answered-all-time, and when. Written even on an empty cycle."""
    import config, json
    monkeypatch.setattr(config, "DATA", tmp_path)
    tournament.one_cycle(cards=[], ask_fn=ask_futures, token="tok",
                         log_path=tmp_path / "log.csv", now_iso=NOW_ISO,
                         submit_fn=lambda qid, prob, token: None)
    status = json.loads((tmp_path / "tournament_status.json").read_text())
    assert status["open_seen"] == 0
    assert status["at"] == NOW_ISO
    assert status["tournament"] == config.METACULUS_TOURNAMENT
    assert status["answered_all_time"] == 0
