import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masker import score_probe_answer


def test_both_right_by_nickname_counts():
    ans = '{"home": "Boston Celtics", "away": "New York Knicks"}'
    assert score_probe_answer(ans, ("BOS", "NYK")) is True


def test_one_wrong_does_not_count():
    ans = '{"home": "Boston Celtics", "away": "Brooklyn Nets"}'
    assert score_probe_answer(ans, ("BOS", "NYK")) is False


def test_garbage_answer_scores_false_not_crash():
    assert score_probe_answer("no idea, sorry", ("BOS", "NYK")) is False


def test_hornets_are_not_scored_as_nets():
    ans = '{"home": "Charlotte Hornets", "away": "Miami Heat"}'
    assert score_probe_answer(ans, ("CHA", "MIA")) is True


def test_abbreviation_and_city_answers_count():
    assert score_probe_answer('{"home": "BOS", "away": "New York Knicks"}', ("BOS", "NYK")) is True
    assert score_probe_answer('{"home": "Boston", "away": "Denver"}', ("BOS", "DEN")) is True


def test_ambiguous_city_alone_does_not_match():
    # "Los Angeles" alone could be LAL or LAC — must not count as either
    assert score_probe_answer('{"home": "Los Angeles", "away": "Utah Jazz"}', ("LAL", "UTA")) is False
