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
from datetime import datetime, timezone
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


SUPPORTED_TYPES = {"binary", "multiple_choice", "numeric", "discrete"}
# Not a question type but a post type: the listing's forecast_type
# filter drops group posts entirely unless this is asked for too
# (forecasting-tools appends it exactly when it unpacks groups).
GROUP_POST_TYPE = "group_of_questions"


def _criteria_text(question: dict) -> str:
    """The resolution rules a forecaster actually needs, as one block.

    Ranked bots pass description, resolution criteria, and fine print
    into the prompt; forecasting the title alone means forecasting a
    headline instead of the rules. Missing pieces are simply left out.
    """
    parts = []
    for key, label in (("description", "Background"),
                       ("resolution_criteria", "Resolution criteria"),
                       ("fine_print", "Fine print")):
        text = question.get(key)
        if text:
            parts.append(f"{label}: {text}")
    return "\n".join(parts)


def _forecast_time_iso(latest) -> str | None:
    """When this account last forecast a question, as an ISO string.

    my_forecasts.latest.start_time has been seen both as an ISO string
    (our own fixture, written from an early live payload) and as a unix
    epoch number (what the API's forecast history uses). Either is
    accepted; anything else reads as "unknown", never a guess.
    """
    if not isinstance(latest, dict):
        return None
    start = latest.get("start_time")
    if isinstance(start, bool):
        return None
    if isinstance(start, (int, float)):
        try:
            return datetime.fromtimestamp(float(start), timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(start, str) and start.strip():
        return start.strip()
    return None


def _card(post_id, question: dict, title: str, criteria: str) -> dict | None:
    """One question card, shared by plain posts and group subquestions.

    Returns None for anything this bot does not answer: an unsupported
    type, a question that is not open, or one missing its id or title.
    """
    # A missing "type" key is treated as binary (the listing filter
    # already narrows types); an unsupported type is skipped.
    qtype = question.get("type", "binary")
    if qtype not in SUPPORTED_TYPES:
        return None
    if question.get("status") not in (None, "open"):
        return None
    qid = question.get("id")
    if qid is None or post_id is None or not title:
        return None           # missing what we actually need: skip, don't guess
    my = question.get("my_forecasts") or {}
    latest = my.get("latest") if isinstance(my, dict) else None
    card = {
        "qid": qid,
        "post_id": post_id,
        "qtype": qtype,
        "question": title,
        "criteria": criteria,
        "close_time": question.get("scheduled_close_time", ""),
        "url": f"https://www.metaculus.com/questions/{post_id}/",
        "already_forecast": bool(latest),
        "last_forecast_at": _forecast_time_iso(latest),
    }
    if qtype == "multiple_choice":
        card["options"] = question.get("options") or []
    if qtype in ("numeric", "discrete"):
        card["scaling"] = question.get("scaling") or {}
        card["open_lower_bound"] = bool(question.get("open_lower_bound"))
        card["open_upper_bound"] = bool(question.get("open_upper_bound"))
        card["unit"] = question.get("unit", "")
    return card


def _group_cards(post: dict) -> list[dict]:
    """Expand a group post into one card per answerable subquestion.

    Market Pulse asks its questions as "numeric groups": one post, one
    shared set of rules, several numeric rows (one per stock, say),
    each its own question with its own id, forecast and score. The
    shape followed here is the one Metaculus's own forecasting-tools
    library unpacks (MetaculusClient._unpack_group_question, read
    2026-09-27): post["group_of_questions"] holds description,
    resolution_criteria, fine_print and a "questions" list, and each
    subquestion's "label" names its row.

    Each card keeps the PARENT post's id (comments and the page URL
    live on the post) and the SUBQUESTION's id (forecasts are posted
    per subquestion). The title is the group title with the row label
    in parentheses, because the label alone ("AAPL") is not a question.
    The group's rules apply to every row; a rule a subquestion states
    for itself wins over the group's. A subquestion with no label and
    no title of its own has nothing telling it apart from its siblings,
    so it is skipped rather than forecast blind.
    """
    group = post.get("group_of_questions")
    if not isinstance(group, dict):
        return []
    subs = group.get("questions")
    if not isinstance(subs, list):
        return []
    group_title = (post.get("title") or group.get("title") or "").strip()
    cards = []
    for sub in subs:
        if not isinstance(sub, dict):
            continue
        label = (sub.get("label") or "").strip()
        sub_title = (sub.get("title") or "").strip()
        if not label and sub_title != group_title:
            label = sub_title
        if not label or not group_title:
            continue
        merged = {key: sub.get(key) or group.get(key)
                  for key in ("description", "resolution_criteria", "fine_print")}
        card = _card(post.get("id"), sub, f"{group_title} ({label})",
                     _criteria_text(merged))
        if card is not None:
            cards.append({**card, "group_label": label})
    return cards


def parse_questions(payload: dict) -> list[dict]:
    """Turn Metaculus's raw /posts/ response into simple question cards.

    A "post" wraps one question most of the time, but it can also wrap
    a group of questions, or carry no question at all (an article, a
    discussion thread). Group posts (post["group_of_questions"]) are
    expanded into one card per subquestion, see _group_cards; that is
    how Market Pulse asks everything. A post with neither shape is
    skipped. All four tournament question types come through: binary,
    multiple choice, numeric, and discrete. A post missing the fields
    we actually need (an id, a title) is skipped too, never guessed at.

    Card shape: {"qid", "post_id", "qtype", "question" (the title),
    "criteria" (background + resolution criteria + fine print),
    "close_time", "url", "already_forecast" (from the API's own
    my_forecasts record, so a fresh checkout can't re-answer),
    "last_forecast_at" (ISO time of that record, or None), and for
    multiple choice "options", for numeric/discrete "scaling",
    "open_lower_bound", "open_upper_bound", "unit"}. Group cards add
    "group_label".
    """
    cards = []
    for post in payload.get("results", []) or []:
        if not isinstance(post, dict):
            continue
        question = post.get("question")
        if not isinstance(question, dict):
            cards.extend(_group_cards(post))   # a group post, or no question at all
            continue
        title = question.get("title") or post.get("title") or ""
        card = _card(post.get("id"), question, title, _criteria_text(question))
        if card is not None:
            cards.append(card)
    return cards


def _get_posts(tournament, token: str, offset: int) -> dict:
    """The only function that really talks to the API for reads. Split
    out so tests can replace it, same pattern as engine/llm.py's
    _call_api and adapters/kalshi_events.py's fetch_open_markets."""
    import requests
    params = {
        "limit": PAGE_SIZE, "offset": offset, "order_by": "-hotness",
        # Comma-joined, not a repeated query param: that is the exact
        # shape the official template sends (see the module docstring).
        "forecast_type": ",".join(sorted(SUPPORTED_TYPES | {GROUP_POST_TYPE})),
        "tournaments": [tournament],
        "statuses": "open", "include_description": "true",
    }
    resp = requests.get(POSTS_URL, headers=_headers(token), params=params,
                        timeout=TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def _get_posts_any_status(tournament, token: str) -> dict:
    """One listing with NO statuses filter, for fetch_post_count.

    A full page, not limit=1: the live /posts/ response carries no
    "count" field (verified against the receipt of the first live run,
    2026-09-02, which came back 1), so the count is len(results) and a
    tiny limit would cap it at the limit. One page is plenty for the
    receipt's real question, zero versus dozens; a 100+-question
    tournament just reads as 100.
    """
    import requests
    params = {"limit": PAGE_SIZE, "offset": 0, "tournaments": [tournament]}
    resp = requests.get(POSTS_URL, headers=_headers(token), params=params,
                        timeout=TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def fetch_post_count(tournament, token: str, get_fn=None) -> int:
    """How many posts this tournament has AT ALL, open or closed.

    Exists to break an ambiguity the Summer season was lost to:
    "open_seen: 0" reads the same whether the tournament simply has no
    open windows right now (MiniBench between rounds) or the slug
    matches nothing at all (a typo, or a season that has not launched).
    A real tournament shows a nonzero total within days of launching;
    a wrong slug stays at 0 forever, and the receipt makes that
    visible instead of green.
    """
    get_fn = get_fn or _get_posts_any_status
    payload = _retry_once(lambda: get_fn(tournament, token),
                          what="count a tournament's posts on Metaculus")
    if not isinstance(payload, dict):
        return 0
    count = payload.get("count")
    if isinstance(count, int):
        return count
    return len(payload.get("results", []) or [])


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


def submit_mc_prediction(qid: int, probs_by_option: dict, token: str,
                         post_fn=None) -> dict:
    """Post one multiple-choice forecast: one probability per option.

    Each option's probability is floored at 0.01 and capped at 0.99
    (never a claimed 0% or 100% on any option), then the whole set is
    renormalized to sum to 1, which is what the API requires. Payload
    shape confirmed against forecasting-tools
    post_multiple_choice_question_prediction: the same forecast list as
    a binary post, but with "probability_yes_per_category" carrying the
    option-name-to-probability dict.
    """
    post_fn = post_fn or _post_forecast
    clipped = {opt: min(max(float(p), 0.01), 0.99)
               for opt, p in probs_by_option.items()}
    total = sum(clipped.values())
    normed = {opt: p / total for opt, p in clipped.items()}
    payload = [{
        "question": qid, "source": "api",
        "probability_yes_per_category": normed,
    }]
    resp = _retry_once(lambda: post_fn(qid, payload, token),
                       what="submit a multiple-choice forecast to Metaculus")
    if not getattr(resp, "ok", False):
        raise RuntimeError(
            f"metaculus rejected the forecast for question {qid}: "
            f"{getattr(resp, 'text', '')}")
    return {"qid": qid, "probability_yes_per_category": normed}


def submit_numeric_prediction(qid: int, cdf: list, token: str,
                              post_fn=None) -> dict:
    """Post one numeric or discrete forecast as a full CDF.

    `cdf` is the 201-value list engine/cdf.py builds: P(outcome <= x)
    at each of the question's own inbound edges. Validated the same way
    forecasting-tools validates before posting: every value in [0, 1]
    and monotonically non-decreasing. A bad CDF raises ValueError here,
    before anything touches the network.
    """
    post_fn = post_fn or _post_forecast
    values = [float(v) for v in cdf]
    if not all(0.0 <= v <= 1.0 for v in values):
        raise ValueError("every CDF value must be between 0 and 1")
    if not all(a <= b for a, b in zip(values, values[1:])):
        raise ValueError("CDF values must be monotonically non-decreasing")
    payload = [{
        "question": qid, "source": "api",
        "continuous_cdf": values,
    }]
    resp = _retry_once(lambda: post_fn(qid, payload, token),
                       what="submit a numeric forecast to Metaculus")
    if not getattr(resp, "ok", False):
        raise RuntimeError(
            f"metaculus rejected the forecast for question {qid}: "
            f"{getattr(resp, 'text', '')}")
    return {"qid": qid, "cdf_len": len(values)}


COMMENTS_URL = f"{API_BASE_URL}/comments/create/"


def _post_json(url: str, body: dict, token: str):
    """Generic JSON POST, split out so tests can replace it."""
    import requests
    return requests.post(url, headers=_headers(token), json=body,
                         timeout=TIMEOUT_S)


def post_comment(post_id: int, text: str, token: str, post_fn=None) -> bool:
    """Leave the required private comment on a forecasted question.

    The tournament rules require a comment with every forecast (kept
    private; Metaculus publishes them itself at intervals). Endpoint
    and payload confirmed against forecasting-tools
    post_question_comment: POST /comments/create/ with on_post,
    text, is_private, included_forecast.

    Returns True on success, False on ANY failure. A comment that
    fails to post must never take the forecast down with it: the
    forecast is the scored thing, the comment is paperwork.
    """
    post_fn = post_fn or _post_json
    body = {"on_post": post_id, "text": text,
            "is_private": True, "included_forecast": True}
    try:
        resp = _retry_once(lambda: post_fn(COMMENTS_URL, body, token),
                           what="post a comment to Metaculus")
        ok = bool(getattr(resp, "ok", False))
    except Exception as exc:
        print(f"  comment on post {post_id} failed ({exc}); forecast stands")
        return False
    if not ok:
        print(f"  comment on post {post_id} rejected; forecast stands")
    return ok


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


# ---- resolved questions, for backtesting -------------------------------
# The tournament bot has no track record: MiniBench had nothing open
# between arming it and now, so every tuning knob (crowd size,
# deliberation, the clip) is set on judgment alone. Metaculus keeps the
# questions it has already resolved, with the outcome and the community
# forecast attached, which is the only way to get evidence without
# waiting a season for one.

def _binary_outcome(question: dict) -> float | None:
    """1.0 for yes, 0.0 for no, None for anything unscoreable.

    Annulled and ambiguous resolutions are not outcomes; scoring against
    them would be inventing a result Metaculus explicitly refused to
    declare.
    """
    if question.get("type") != "binary":
        return None
    resolution = (question.get("resolution") or "").strip().lower()
    if resolution == "yes":
        return 1.0
    if resolution == "no":
        return 0.0
    return None


def _community_forecast(question: dict) -> float | None:
    """The crowd's last standing number, or None if it is not published.

    None is recorded honestly rather than filled in: a missing community
    forecast means that one question cannot contribute to the "did we
    beat the crowd" comparison, not that the crowd said 50 percent.
    """
    aggregations = question.get("aggregations")
    if not isinstance(aggregations, dict):
        return None
    latest = (aggregations.get("recency_weighted") or {}).get("latest")
    if not isinstance(latest, dict):
        return None
    centers = latest.get("centers")
    if isinstance(centers, list) and centers:
        try:
            return float(centers[0])
        except (TypeError, ValueError):
            return None
    return None


def parse_resolved(payload: dict) -> list[dict]:
    """Resolved binary questions as scoreable cards.

    Same card shape parse_questions produces, plus "outcome" (1.0/0.0),
    "community" (float or None) and "resolved_at". Anything without a
    clean yes/no outcome is dropped, not guessed at.
    """
    base = {card["qid"]: card for card in parse_questions(payload)}
    scoreable = []
    for post in payload.get("results", []) or []:
        if not isinstance(post, dict):
            continue
        question = post.get("question")
        if not isinstance(question, dict):
            continue
        card = base.get(question.get("id"))
        if card is None:
            continue
        outcome = _binary_outcome(question)
        if outcome is None:
            continue
        scoreable.append({**card, "outcome": outcome,
                          "community": _community_forecast(question),
                          "resolved_at": question.get("actual_resolve_time")})
    return scoreable


def _get_resolved_posts(tournament, token: str, offset: int) -> dict:
    """Read side for resolved questions. Split out like _get_posts so
    tests can replace it."""
    import requests
    params = {
        "limit": PAGE_SIZE, "offset": offset, "order_by": "-resolve_time",
        "forecast_type": ["binary"],
        "tournaments": [tournament],
        "statuses": "resolved", "include_description": "true",
    }
    resp = requests.get(POSTS_URL, headers=_headers(token), params=params,
                        timeout=TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def get_posts_raw(params: dict, token: str) -> dict:
    """One /posts/ read with caller-supplied params, returned unparsed.

    Only for diagnosing which filters Metaculus actually honors; the
    normal paths go through fetch_open_questions and
    fetch_resolved_questions.
    """
    import requests
    resp = requests.get(POSTS_URL, headers=_headers(token), params=params,
                        timeout=TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def fetch_resolved_questions(tournament, token: str, get_fn=None,
                             want: int = 50) -> list[dict]:
    """Resolved binary questions from one tournament, newest first.

    Stops as soon as `want` scoreable cards are in hand: every extra
    page is a request nobody needs, and the newest resolutions are the
    ones most likely to sit after the model cutoff anyway.
    """
    get_fn = get_fn or _get_resolved_posts
    cards: list[dict] = []
    offset = 0
    for _ in range(MAX_PAGES):
        payload = _retry_once(lambda: get_fn(tournament, token, offset),
                              what="fetch resolved questions from Metaculus")
        if not isinstance(payload, dict):
            break
        cards.extend(parse_resolved(payload))
        results = payload.get("results", []) or []
        if len(cards) >= want or len(results) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return cards[:want]
