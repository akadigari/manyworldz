"""Talk to the Metaculus API: read open tournament questions, post the
crowd's probability on them.

Read-only listing plus one write endpoint (posting a forecast). No
sports filter here, no market price either: Metaculus questions come
with no "mid" at all, so the crowd is told the truth about that the
same way ask.py's typed-in questions are.

Everything below was checked against two sources on 2026-07-19:

1. The official bot template, github.com/Metaculus/metac-bot-template,
   specifically main_with_no_framework.py (the plain-requests version,
   meant for people who want to see the raw calls without the
   forecasting-tools framework in the way). It shows:
   - API_BASE_URL = "https://www.metaculus.com/api"
   - auth header: {"Authorization": f"Token {METACULUS_TOKEN}"}
   - listing open questions: GET {API_BASE_URL}/posts/ with params
     limit, offset, order_by, forecast_type, tournaments (a list, ID or
     slug both work), statuses, include_description
   - posting a forecast: POST {API_BASE_URL}/questions/forecast/ with a
     JSON list of {"question": <id>, "source": "api",
     "probability_yes": <0-1 float>, "probability_yes_per_category":
     None, "continuous_cdf": None} for a binary question
   - the template clamps its own probabilities to 1-99% before sending
2. https://www.metaculus.com/api/ ("Commercial API or Data Access"):
   confirms the token comes prefixed "Token " in the Authorization
   header, and that the whole API rejects unauthenticated requests with
   a 403 ("The API is only available to authenticated users"), which is
   why fetch_open_questions needs a token too, not just submit_prediction.

No documented numeric rate limit was found anywhere in the above (the
template's own CONCURRENT_REQUESTS_LIMIT = 5 is about its LLM calls,
not the Metaculus API). We stay well under any plausible limit anyway:
one sequential request at a time, one retry max on a network hiccup,
never a retry storm.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

API_BASE_URL = "https://www.metaculus.com/api"
QUESTIONS_URL = f"{API_BASE_URL}/questions/forecast/"
POSTS_URL = f"{API_BASE_URL}/posts/"
TIMEOUT_S = 30
PAGE_SIZE = 100     # a FutureEval season runs 300-500 questions; a few pages covers it
MAX_PAGES = 10      # hard cap so a bad "next" page can never spin forever


def _headers(token: str) -> dict:
    """The one auth header shape the whole API wants."""
    return {"Authorization": f"Token {token}"}


def parse_questions(payload: dict) -> list[dict]:
    """Turn Metaculus's raw /posts/ response into simple question cards.

    A "post" wraps one question most of the time, but it can also wrap
    a group of questions, or carry no question at all (an article, a
    discussion thread). Group posts and non-binary questions (multiple
    choice, numeric, discrete) get skipped: this project's crowd only
    knows how to give a single YES/NO probability, so only binary
    questions are usable. A post missing the fields we actually need
    (an id, a title) is skipped too, never guessed at.

    Card shape: {"qid": <question id, used to submit later>,
    "question": <title text>, "close_time": <ISO string or "">,
    "url": <the question's page on metaculus.com>}.
    """
    cards = []
    for post in payload.get("results", []) or []:
        if not isinstance(post, dict):
            continue
        question = post.get("question")
        if not isinstance(question, dict):
            continue          # a group/multi-question post, or no question attached
        # The listing call already asks for forecast_type=binary, so a
        # missing "type" key is treated as binary too; an explicit,
        # different type is trusted and skipped.
        qtype = question.get("type", "binary")
        if qtype != "binary":
            continue
        if question.get("status") not in (None, "open"):
            continue
        qid = question.get("id")
        post_id = post.get("id")
        title = question.get("title") or post.get("title") or ""
        if qid is None or post_id is None or not title:
            continue          # missing what we actually need: skip, don't guess
        cards.append({
            "qid": qid,
            "question": title,
            "close_time": question.get("scheduled_close_time", ""),
            "url": f"https://www.metaculus.com/questions/{post_id}/",
        })
    return cards


def _get_posts(tournament, token: str, offset: int) -> dict:
    """The only function that really talks to the API for reads. Split
    out so tests can replace it, same pattern as engine/llm.py's
    _call_api and adapters/kalshi_events.py's fetch_open_markets."""
    import requests
    params = {
        "limit": PAGE_SIZE, "offset": offset, "order_by": "-hotness",
        "forecast_type": "binary", "tournaments": [tournament],
        "statuses": "open", "include_description": "true",
    }
    resp = requests.get(POSTS_URL, headers=_headers(token), params=params,
                        timeout=TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def _post_forecast(qid: int, payload: list[dict], token: str):
    """The only function that really talks to the API for the write
    call. Split out so tests can replace it with a fake."""
    import requests
    return requests.post(QUESTIONS_URL, headers=_headers(token),
                         json=payload, timeout=TIMEOUT_S)


def _retry_once(call, what: str):
    """Try once, and if the network itself blew up (a connection error,
    a timeout, anything that never got back a real HTTP response), try
    exactly one more time. That is the whole retry budget: two failures
    in a row raise a clear error instead of hammering the API. A clean
    HTTP response that just says "no" (a bad token, a closed question)
    is not a network hiccup and is never retried here at all.
    """
    try:
        return call()
    except Exception:
        try:
            return call()
        except Exception as exc:
            raise RuntimeError(f"could not {what}: {exc}") from exc


def fetch_open_questions(tournament, token: str, get_fn=None) -> list[dict]:
    """Live pull of every open binary question in one tournament.

    `tournament` can be the numeric tournament ID or its slug; both
    show up in the wild (metaculus.com/tournament/<slug>/, and the
    template's own TOURNAMENT_ID constant is a plain int) and the
    "tournaments" filter on /posts/ accepts either. Paginates the same
    way adapters/kalshi_events.py paginates /events/, capped at
    MAX_PAGES so nothing can spin forever on a bad response.
    """
    get_fn = get_fn or _get_posts
    cards: list[dict] = []
    offset = 0
    for _ in range(MAX_PAGES):
        payload = _retry_once(lambda: get_fn(tournament, token, offset),
                              what="fetch open questions from Metaculus")
        if not isinstance(payload, dict):
            break
        cards.extend(parse_questions(payload))
        results = payload.get("results", []) or []
        if len(results) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return cards


def submit_prediction(qid: int, probability: float, token: str,
                      post_fn=None) -> dict:
    """Post one forecast on one binary question.

    Payload format confirmed against the official bot template (see
    the module docstring for the exact source): a JSON list holding one
    object, "source" always "api", "probability_yes_per_category" and
    "continuous_cdf" left None since this project only ever forecasts
    binary questions.

    `probability` is clamped to 0.01-0.99 first: never a claimed 0% or
    100% chance. That is the same clamp engine/swarm.py's consensus
    already applies before this function ever sees the number, and the
    same 1-99% range the official template clamps its own forecasts to;
    clamping again here is just a second, cheap safety net, not a
    change in behavior.

    Raises RuntimeError with Metaculus's own error text if the API
    cleanly rejects the forecast (bad token, closed question, and so
    on): that is never retried, since retrying the same rejected
    request would just fail the same way again. A network hiccup gets
    one retry, see _retry_once.
    """
    post_fn = post_fn or _post_forecast
    prob = min(max(float(probability), 0.01), 0.99)
    payload = [{
        "question": qid, "source": "api",
        "probability_yes": prob,
        "probability_yes_per_category": None,
        "continuous_cdf": None,
    }]
    resp = _retry_once(lambda: post_fn(qid, payload, token),
                       what="submit a forecast to Metaculus")
    if not getattr(resp, "ok", False):
        text = getattr(resp, "text", "")
        raise RuntimeError(
            f"metaculus rejected the forecast for question {qid}: {text}")
    return {"qid": qid, "probability": prob}


if __name__ == "__main__":
    # CLI smoke test (needs a real token): venv/bin/python adapters/metaculus.py TOKEN
    import os
    tok = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("METACULUS_TOKEN", "")
    if not tok:
        print("pass a token: venv/bin/python adapters/metaculus.py TOKEN")
        raise SystemExit(1)
    import config
    open_qs = fetch_open_questions(config.METACULUS_TOURNAMENT, tok)
    print(f"{len(open_qs)} open binary question(s) in "
          f"{config.METACULUS_TOURNAMENT!r}; first 5:")
    for c in open_qs[:5]:
        print(f"  {c['qid']:>8}  {c['question'][:60]}  closes {c['close_time']}")
