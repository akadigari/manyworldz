"""Group posts ("numeric groups"), the shape Market Pulse questions come in.

The payload shape below mirrors what Metaculus's own forecasting-tools
library unpacks (MetaculusClient._unpack_group_question, read
2026-09-27): a post with no "question" key but a "group_of_questions"
dict holding the shared description / resolution_criteria / fine_print
and a "questions" list of ordinary question dicts, each carrying its own
id, "label" (the one-row name, e.g. a ticker), type, status, scaling and
scheduled_close_time. No network: every payload is built inline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters import metaculus

SCALING = {"range_min": 100, "range_max": 300, "zero_point": None,
           "inbound_outcome_count": 200, "continuous_range": None}


def sub(qid, label, status="open", qtype="numeric", **extra):
    return {"id": qid, "title": "Where will these stocks close on Sep 30?",
            "label": label, "type": qtype, "status": status,
            "scheduled_close_time": "2026-09-30T20:00:00Z",
            "unit": "USD", "open_lower_bound": True, "open_upper_bound": True,
            "scaling": dict(SCALING), **extra}


def group_post(*subs, post_id=40001):
    return {"id": post_id, "title": "Where will these stocks close on Sep 30?",
            "group_of_questions": {
                "description": "Closing prices on the last trading day.",
                "resolution_criteria": "Resolves to the official NASDAQ close.",
                "fine_print": "Splits are adjusted for.",
                "questions": list(subs)}}


def test_group_post_expands_into_one_card_per_subquestion():
    payload = {"results": [group_post(sub(700001, "AAPL"), sub(700002, "NVDA"))]}
    cards = metaculus.parse_questions(payload)
    assert [c["qid"] for c in cards] == [700001, 700002]
    first = cards[0]
    assert first["post_id"] == 40001                  # the parent post
    assert first["qtype"] == "numeric"
    assert first["question"] == "Where will these stocks close on Sep 30? (AAPL)"
    assert first["url"] == "https://www.metaculus.com/questions/40001/"
    assert first["close_time"] == "2026-09-30T20:00:00Z"
    assert "official NASDAQ close" in first["criteria"]
    assert "Splits are adjusted" in first["criteria"]
    assert first["scaling"]["range_max"] == 300
    assert first["open_lower_bound"] is True
    assert first["unit"] == "USD"
    assert first["already_forecast"] is False


def test_subquestion_own_criteria_win_over_the_groups():
    s = sub(700001, "AAPL", resolution_criteria="Use the Yahoo close.")
    card = metaculus.parse_questions({"results": [group_post(s)]})[0]
    assert "Use the Yahoo close." in card["criteria"]
    assert "NASDAQ" not in card["criteria"]
    assert "Closing prices" in card["criteria"]       # group background kept


def test_closed_unsupported_and_unlabelled_subquestions_are_skipped():
    no_label = sub(700004, "")                        # nothing tells it apart
    payload = {"results": [group_post(
        sub(700001, "AAPL"), sub(700002, "NVDA", status="closed"),
        sub(700003, "MSFT", qtype="date"), no_label)]}
    assert [c["qid"] for c in metaculus.parse_questions(payload)] == [700001]


def test_subquestion_already_forecast_and_its_time_come_from_my_forecasts():
    iso = sub(700001, "AAPL",
              my_forecasts={"latest": {"start_time": "2026-09-20T00:00:00Z"}})
    epoch = sub(700002, "NVDA",
                my_forecasts={"latest": {"start_time": 1790000000.0}})
    cards = metaculus.parse_questions({"results": [group_post(iso, epoch)]})
    assert cards[0]["already_forecast"] is True
    assert cards[0]["last_forecast_at"].startswith("2026-09-20T00:00:00")
    assert cards[1]["already_forecast"] is True
    assert cards[1]["last_forecast_at"].startswith("2026-09-21")


def test_plain_posts_carry_no_last_forecast_time_when_never_forecast():
    post = {"id": 1, "question": {"id": 9, "type": "binary", "status": "open",
                                  "title": "Q", "scheduled_close_time": ""}}
    card = metaculus.parse_questions({"results": [post]})[0]
    assert card["last_forecast_at"] is None


def test_a_malformed_group_is_skipped_not_crashed_on():
    payload = {"results": [
        {"id": 1, "title": "g", "group_of_questions": {"questions": None}},
        {"id": 2, "title": "g", "group_of_questions": {"questions": ["junk"]}},
        {"id": 3, "title": "g", "group_of_questions": "junk"},
    ]}
    assert metaculus.parse_questions(payload) == []


def test_listing_asks_the_api_for_group_posts_too(monkeypatch):
    """forecasting-tools adds "group_of_questions" to forecast_type when
    it unpacks groups; without it the API never returns them at all."""
    import requests
    seen = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": []}

    def fake_get(url, headers=None, params=None, timeout=None):
        seen.update(params)
        return Resp()

    monkeypatch.setattr(requests, "get", fake_get)
    metaculus._get_posts("some-slug", "tok", 0)
    assert "group_of_questions" in seen["forecast_type"].split(",")
    assert "numeric" in seen["forecast_type"].split(",")
