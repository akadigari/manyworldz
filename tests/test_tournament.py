import csv
import sys
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


def ask_futures(p, model=None, max_tokens=400):
    return FUTURES


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
    assert set(rows[0].keys()) == {"qid", "question", "prob", "at"}
    assert rows[0]["qid"] == "500000"
    assert float(rows[0]["prob"]) > 0


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
