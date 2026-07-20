import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from adapters import metaculus

FIXTURE = Path(__file__).parent / "fixtures" / "metaculus_posts.json"


def payload():
    return json.loads(FIXTURE.read_text())


class FakeResp:
    """A tiny stand-in for requests.Response: just the two attributes
    submit_prediction actually looks at."""
    def __init__(self, ok, text=""):
        self.ok = ok
        self.text = text


def test_parse_questions_keeps_only_open_binary_with_required_fields():
    cards = metaculus.parse_questions(payload())
    qids = {c["qid"] for c in cards}
    # 500001: plain open binary -> kept
    assert 500001 in qids
    # 500006: open, no "type" key at all -> treated as binary, kept
    assert 500006 in qids
    card = next(c for c in cards if c["qid"] == 500001)
    assert card["question"] == "Will a new AI safety treaty be signed before 2027?"
    assert card["close_time"] == "2026-12-31T23:59:00Z"
    assert card["url"] == "https://www.metaculus.com/questions/30001/"


def test_parse_questions_skips_group_non_binary_and_closed_posts():
    cards = metaculus.parse_questions(payload())
    qids = {c["qid"] for c in cards}
    assert 500002 not in qids     # multiple_choice
    assert 500003 not in qids     # group post, no "question" key
    assert 500004 not in qids     # group post, no "question" key
    assert 500005 not in qids     # status closed
    assert len(cards) == 2


def test_parse_questions_handles_missing_results_key():
    assert metaculus.parse_questions({}) == []


def test_fetch_open_questions_paginates_until_a_short_page():
    one_result = {"id": 1, "question": {"id": 900, "type": "binary", "status": "open",
                                        "title": "Q1", "scheduled_close_time": "2026-08-01T00:00:00Z"}}
    page1 = {"results": [one_result] * metaculus.PAGE_SIZE}
    page2 = {"results": [
        {"id": 2, "question": {"id": 901, "type": "binary", "status": "open",
                               "title": "Q2", "scheduled_close_time": "2026-08-01T00:00:00Z"}}
    ]}
    calls = []

    def fake_get(tournament, token, offset):
        calls.append(offset)
        return page1 if offset == 0 else page2

    cards = metaculus.fetch_open_questions("some-slug", "tok", get_fn=fake_get)
    assert calls == [0, metaculus.PAGE_SIZE]
    assert len(cards) == metaculus.PAGE_SIZE + 1


def test_submit_prediction_payload_matches_documented_format():
    seen = {}

    def fake_post(qid, body, token):
        seen["qid"], seen["body"], seen["token"] = qid, body, token
        return FakeResp(ok=True)

    out = metaculus.submit_prediction(500001, 0.63, "tok-123", post_fn=fake_post)
    assert seen["token"] == "tok-123"
    assert seen["body"] == [{
        "question": 500001, "source": "api",
        "probability_yes": 0.63,
        "probability_yes_per_category": None,
        "continuous_cdf": None,
    }]
    assert out == {"qid": 500001, "probability": 0.63}


def test_submit_prediction_clamps_probability_into_accepted_range():
    seen = {}

    def fake_post(qid, body, token):
        seen["prob"] = body[0]["probability_yes"]
        return FakeResp(ok=True)

    metaculus.submit_prediction(1, 0.9999, "tok", post_fn=fake_post)
    assert seen["prob"] == 0.99
    metaculus.submit_prediction(1, 0.0001, "tok", post_fn=fake_post)
    assert seen["prob"] == 0.01


def test_submit_prediction_retries_once_on_network_hiccup_then_succeeds():
    attempts = {"n": 0}

    def flaky_post(qid, body, token):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("blip")
        return FakeResp(ok=True)

    out = metaculus.submit_prediction(1, 0.5, "tok", post_fn=flaky_post)
    assert attempts["n"] == 2
    assert out["qid"] == 1


def test_submit_prediction_raises_clear_error_without_retry_on_clean_rejection():
    attempts = {"n": 0}

    def rejecting_post(qid, body, token):
        attempts["n"] += 1
        return FakeResp(ok=False, text="question is closed")

    with pytest.raises(RuntimeError) as exc:
        metaculus.submit_prediction(1, 0.5, "tok", post_fn=rejecting_post)
    assert attempts["n"] == 1               # a clean "no" is never retried
    assert "question is closed" in str(exc.value)
