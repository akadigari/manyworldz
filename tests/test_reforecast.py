"""The per-tournament update policy (Market Pulse and anything like it).

Market Pulse scores the forecast standing at close (spot scoring), so a
forecast made days early and never touched is scored on stale
information. Tournaments listed in config.METACULUS_REFORECAST_TOURNAMENTS
therefore swap "already answered" for "answered recently": a question
is re-answered once its last forecast is REFORECAST_MIN_INTERVAL_H old,
plus once more inside the final REFORECAST_FINAL_WINDOW_H before close.
Every other tournament (FutureEval, MiniBench) keeps the strict
one-forecast-per-question rule. All fakes, no network.
"""
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import config
import tournament

NOW = datetime(2026, 9, 27, 12, 0, 0, tzinfo=timezone.utc)
NOW_ISO = NOW.isoformat()
VOTE = '{"probability": 0.8, "reason": "likely"}'
PULSE = "market-pulse-26q4"


def ask_vote(p, model=None, max_tokens=400):
    return VOTE


def card(qid, close_in_h=24 * 7, **extra):
    close = (NOW + timedelta(hours=close_in_h)).isoformat()
    return {"qid": qid, "post_id": qid, "question": f"Will thing {qid} happen?",
            "close_time": close, "url": f"https://www.metaculus.com/questions/{qid}/",
            **extra}


def seed(log_path, qid, hours_ago):
    tournament._append_log(
        {"qid": qid, "question": f"Will thing {qid} happen?", "raw_prob": 0.6,
         "prob": 0.6, "at": (NOW - timedelta(hours=hours_ago)).isoformat(),
         "source": "crowd"}, log_path)


@pytest.fixture
def pulse_enabled(monkeypatch):
    monkeypatch.setattr(config, "METACULUS_REFORECAST_TOURNAMENTS", [PULSE])
    monkeypatch.setattr(config, "REFORECAST_MIN_INTERVAL_H", 24.0)
    monkeypatch.setattr(config, "REFORECAST_FINAL_WINDOW_H", 3.0)


def run(log_path, cards, tournament_slug=PULSE):
    posted = []
    out = tournament.one_cycle(
        tournament=tournament_slug, cards=cards, ask_fn=ask_vote, token="tok",
        log_path=log_path, now_iso=NOW_ISO,
        submit_fn=lambda qid, prob, token: posted.append(qid),
        comment_fn=lambda *a: True)
    return out, posted


def test_config_defaults_keep_every_tournament_strict():
    assert config.METACULUS_REFORECAST_TOURNAMENTS == []
    assert config.REFORECAST_MIN_INTERVAL_H == 24.0
    assert config.REFORECAST_FINAL_WINDOW_H == 3.0
    assert not any("pulse" in str(s) for s in config.METACULUS_TOURNAMENTS)


def test_reforecast_env_vars_parse_like_the_tournament_list(monkeypatch):
    import importlib
    monkeypatch.setenv("METACULUS_REFORECAST_TOURNAMENTS", " 33066, market-pulse-26q4 ,")
    monkeypatch.setenv("REFORECAST_MIN_INTERVAL_HOURS", "12")
    monkeypatch.setenv("REFORECAST_FINAL_WINDOW_HOURS", "2")
    importlib.reload(config)
    try:
        assert config.METACULUS_REFORECAST_TOURNAMENTS == ["33066", "market-pulse-26q4"]
        assert config.REFORECAST_MIN_INTERVAL_H == 12.0
        assert config.REFORECAST_FINAL_WINDOW_H == 2.0
    finally:
        for name in ("METACULUS_REFORECAST_TOURNAMENTS",
                     "REFORECAST_MIN_INTERVAL_HOURS",
                     "REFORECAST_FINAL_WINDOW_HOURS"):
            monkeypatch.delenv(name)
        importlib.reload(config)


def test_stale_answer_is_refreshed_in_a_reforecast_tournament(tmp_path, pulse_enabled):
    log_path = tmp_path / "log.csv"
    seed(log_path, 1, hours_ago=25)
    out, posted = run(log_path, [card(1)])
    assert posted == [1]
    rows = list(csv.DictReader(open(log_path, newline="")))
    assert len(rows) == 2
    assert tournament._log_history(log_path)[1] == NOW_ISO   # latest row wins


def test_recent_answer_is_left_alone_in_a_reforecast_tournament(tmp_path, pulse_enabled):
    log_path = tmp_path / "log.csv"
    seed(log_path, 1, hours_ago=2)
    out, posted = run(log_path, [card(1)])
    assert posted == []
    assert out["answered"] == 0


def test_one_last_update_inside_the_final_window_before_close(tmp_path, pulse_enabled):
    log_path = tmp_path / "log.csv"
    seed(log_path, 1, hours_ago=10)                 # recent by the 24h rule...
    out, posted = run(log_path, [card(1, close_in_h=1)])
    assert posted == [1]                            # ...but close is 1h away


def test_final_window_update_happens_only_once(tmp_path, pulse_enabled):
    log_path = tmp_path / "log.csv"
    seed(log_path, 1, hours_ago=0.5)                # already inside the window
    out, posted = run(log_path, [card(1, close_in_h=1)])
    assert posted == []


def test_strict_tournament_never_refreshes_even_when_stale(tmp_path, pulse_enabled):
    log_path = tmp_path / "log.csv"
    seed(log_path, 1, hours_ago=100)
    out, posted = run(log_path, [card(1, close_in_h=1)], tournament_slug="minibench")
    assert posted == []
    assert out["answered"] == 0


def test_api_forecast_time_counts_when_the_log_has_nothing(tmp_path, pulse_enabled):
    log_path = tmp_path / "log.csv"
    recent = card(1, already_forecast=True,
                  last_forecast_at=(NOW - timedelta(hours=1)).isoformat())
    stale = card(2, already_forecast=True,
                 last_forecast_at=(NOW - timedelta(hours=30)).isoformat())
    out, posted = run(log_path, [recent, stale])
    assert posted == [2]


def test_api_forecast_time_newer_than_the_log_wins(tmp_path, pulse_enabled):
    """A forecast made by hand (or by another checkout) after the log's
    row is still a recent forecast."""
    log_path = tmp_path / "log.csv"
    seed(log_path, 1, hours_ago=40)
    c = card(1, already_forecast=True,
             last_forecast_at=(NOW - timedelta(hours=1)).isoformat())
    out, posted = run(log_path, [c])
    assert posted == []


def test_never_answered_questions_go_before_refreshes(tmp_path, pulse_enabled, monkeypatch):
    monkeypatch.setattr(config, "TOURNAMENT_QUESTIONS_PER_RUN", 2)
    log_path = tmp_path / "log.csv"
    seed(log_path, 1, hours_ago=50)
    seed(log_path, 2, hours_ago=30)
    seed(log_path, 3, hours_ago=5)
    cards = [card(1), card(2), card(3, close_in_h=2), card(4)]
    out, posted = run(log_path, cards)
    # 4 is new; 3 is in its final window; 1 and 2 are merely stale.
    assert posted == [4, 3]


def test_stalest_refresh_goes_first(tmp_path, pulse_enabled, monkeypatch):
    monkeypatch.setattr(config, "TOURNAMENT_QUESTIONS_PER_RUN", 1)
    log_path = tmp_path / "log.csv"
    seed(log_path, 1, hours_ago=30)
    seed(log_path, 2, hours_ago=50)
    out, posted = run(log_path, [card(1), card(2)])
    assert posted == [2]


def test_numeric_id_tournament_matches_the_string_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "METACULUS_REFORECAST_TOURNAMENTS", ["33066"])
    log_path = tmp_path / "log.csv"
    seed(log_path, 1, hours_ago=30)
    out, posted = run(log_path, [card(1)], tournament_slug=33066)
    assert posted == [1]
