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


def test_parse_questions_keeps_open_questions_of_all_four_types():
    cards = metaculus.parse_questions(payload())
    by_qid = {c["qid"]: c for c in cards}
    # 500001: plain open binary -> kept
    # 500006: open, no "type" key at all -> treated as binary, kept
    # 500002: multiple choice -> kept now
    # 500007: numeric -> kept now; 500008: discrete -> kept now
    assert set(by_qid) == {500001, 500002, 500006, 500007, 500008}
    card = by_qid[500001]
    assert card["question"] == "Will a new AI safety treaty be signed before 2027?"
    assert card["close_time"] == "2026-12-31T23:59:00Z"
    assert card["url"] == "https://www.metaculus.com/questions/30001/"
    assert card["post_id"] == 30001
    assert card["qtype"] == "binary"


def test_parse_questions_skips_group_and_closed_posts():
    cards = metaculus.parse_questions(payload())
    qids = {c["qid"] for c in cards}
    assert 500003 not in qids     # group post, no "question" key
    assert 500004 not in qids     # group post, no "question" key
    assert 500005 not in qids     # status closed


def test_parse_questions_carries_resolution_criteria_and_fine_print():
    card = {c["qid"]: c for c in metaculus.parse_questions(payload())}[500001]
    assert "signed by at least 20 states" in card["criteria"]
    assert "ratification is not required" in card["criteria"]


def test_parse_questions_carries_options_for_multiple_choice():
    card = {c["qid"]: c for c in metaculus.parse_questions(payload())}[500002]
    assert card["qtype"] == "multiple_choice"
    assert card["options"] == ["France", "United Kingdom", "Singapore", "Other"]


def test_parse_questions_carries_scaling_and_bounds_for_numeric_and_discrete():
    by_qid = {c["qid"]: c for c in metaculus.parse_questions(payload())}
    num = by_qid[500007]
    assert num["qtype"] == "numeric"
    assert num["scaling"]["range_min"] == 0
    assert num["scaling"]["range_max"] == 60
    assert num["open_lower_bound"] is False
    assert num["open_upper_bound"] is True
    assert num["unit"] == "states"
    disc = by_qid[500008]
    assert disc["qtype"] == "discrete"
    assert disc["scaling"]["range_min"] == -49500


def test_parse_questions_flags_already_forecast_from_my_forecasts():
    by_qid = {c["qid"]: c for c in metaculus.parse_questions(payload())}
    assert by_qid[500007]["already_forecast"] is True
    assert by_qid[500001]["already_forecast"] is False
    assert by_qid[500006]["already_forecast"] is False   # field absent -> False


def test_submit_mc_prediction_clips_renormalizes_and_posts_per_category():
    sent = {}
    def fake_post(qid, payload, token):
        sent["payload"] = payload
        return FakeResp(True)
    out = metaculus.submit_mc_prediction(
        777, {"A": 1.0, "B": 0.0, "C": 0.0}, "tok", post_fn=fake_post)
    body = sent["payload"][0]
    probs = body["probability_yes_per_category"]
    # 0.0 floors to 0.01 before renormalizing, so nothing is a claimed 0%
    assert all(p >= 0.009 for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert body["question"] == 777
    assert body["source"] == "api"
    assert "probability_yes" not in body
    assert out["qid"] == 777


def test_submit_numeric_prediction_posts_the_cdf_and_validates_it():
    sent = {}
    def fake_post(qid, payload, token):
        sent["payload"] = payload
        return FakeResp(True)
    cdf = [i / 200 for i in range(201)]
    metaculus.submit_numeric_prediction(888, cdf, "tok", post_fn=fake_post)
    body = sent["payload"][0]
    assert body["continuous_cdf"] == cdf
    assert body["question"] == 888
    with pytest.raises(ValueError):
        metaculus.submit_numeric_prediction(888, [0.5, 0.4], "tok",
                                            post_fn=fake_post)
    with pytest.raises(ValueError):
        metaculus.submit_numeric_prediction(888, [0.0, 1.5], "tok",
                                            post_fn=fake_post)


def test_post_comment_hits_comments_create_private_and_never_raises():
    sent = {}
    def fake_post_json(url, body, token):
        sent["url"] = url; sent["body"] = body
        return FakeResp(True)
    ok = metaculus.post_comment(30001, "crowd said 0.42", "tok",
                                post_fn=fake_post_json)
    assert ok is True
    assert sent["url"].endswith("/comments/create/")
    assert sent["body"] == {"on_post": 30001, "text": "crowd said 0.42",
                            "is_private": True, "included_forecast": True}
    def broken(url, body, token):
        raise RuntimeError("network down")
    # a comment failure must never break the cycle: log-and-continue
    assert metaculus.post_comment(30001, "x", "tok", post_fn=broken) is False


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


def test_fetch_post_count_reads_the_apis_total_count():
    payload = {"count": 60, "results": [{"id": 1}]}
    assert metaculus.fetch_post_count(
        "minibench", "tok", get_fn=lambda t, tok: payload) == 60


def test_fetch_post_count_falls_back_to_len_results():
    payload = {"results": [{"id": 1}, {"id": 2}]}
    assert metaculus.fetch_post_count(
        "minibench", "tok", get_fn=lambda t, tok: payload) == 2
