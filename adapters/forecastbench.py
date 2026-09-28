"""ForecastBench question sets in, forecast-set files out.

Format facts come from the official submission guide
(github.com/forecastingresearch/forecastbench/wiki/How-to-submit-to-ForecastBench):

- Question sets are published at 0:00 UTC on each forecast due date
  (every two weeks) in the forecastbench-datasets repo as
  `<due date>-llm.json`; `latest-llm.json` is a symlink to the newest,
  which raw.githubusercontent.com serves as the bare target filename.
- 500 questions: 250 "market" (one forecast, resolution_date null) and
  250 "dataset" (one forecast per entry in resolution_dates, usually 8).
- Pre-2025-10-26 sets also held combination questions (id is a list);
  they are no longer asked, so they are dropped here.
- The upload is `<due date>.<organization>.<N>.json`, N in 1..3, with
  exactly five top-level keys. Under 95% coverage of either question
  type keeps a model off the leaderboard; gaps are imputed 0.5.

No network in this module except through the injectable `fetch`.
"""
from __future__ import annotations

import json
import re

DATASETS_RAW = ("https://raw.githubusercontent.com/forecastingresearch/"
                "forecastbench-datasets/main/datasets/question_sets")

SOURCES = {
    "market": ("kalshi", "manifold", "metaculus", "polymarket"),
    "dataset": ("acled", "dbnomics", "fred", "wikipedia", "yfinance"),
}
MAX_SETS_PER_ROUND = 3
MIN_COVERAGE = 0.95
TOP_LEVEL_KEYS = {"organization", "model", "model_organization",
                  "question_set", "forecasts"}
FORECAST_KEYS = {"id", "source", "forecast", "resolution_date", "reasoning"}
_SET_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}-llm\.json$")
_ORG_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def _http_get(url: str) -> str:
    """The only real network call. Split out so tests never reach it."""
    import requests
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.text


def fetch_question_set(due_date: str | None = None, fetch=_http_get) -> dict:
    """Download one question set: `due_date` (YYYY-MM-DD) or the latest.

    `latest-llm.json` is a git symlink, so the raw URL returns the name
    of the file it points at rather than JSON; that name is followed once.
    Raises ValueError on anything that does not look like a question set.
    """
    name = f"{due_date}-llm.json" if due_date else "latest-llm.json"
    text = fetch(f"{DATASETS_RAW}/{name}")
    if not due_date and _SET_NAME.match(text.strip()):
        text = fetch(f"{DATASETS_RAW}/{text.strip()}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"question set {name} is not JSON: {exc}") from exc
    missing = {"forecast_due_date", "question_set", "questions"} - set(data)
    if missing or not data.get("questions"):
        raise ValueError(f"question set {name} is malformed (missing {sorted(missing)} "
                         "or no questions)")
    return data


def question_kind(question: dict) -> str | None:
    """"market", "dataset", or None for a source the guide does not list."""
    for kind, sources in SOURCES.items():
        if question.get("source") in sources:
            return kind
    return None


def standard_questions(question_set: dict) -> list[dict]:
    """Every question that needs a forecast: known source, scalar id."""
    return [q for q in question_set["questions"]
            if isinstance(q.get("id"), str) and question_kind(q)]


def resolution_dates(question: dict) -> list[str | None]:
    """The dates to forecast for: [None] for a market question."""
    if question_kind(question) == "dataset":
        dates = question.get("resolution_dates")
        return list(dates) if isinstance(dates, list) else []
    return [None]


def fill_dates(template: str, due_date: str, resolution_date: str) -> str:
    """Dataset questions are f-string templates with two date slots."""
    return (template.replace("{forecast_due_date}", due_date)
                    .replace("{resolution_date}", resolution_date))


def forecast_filename(due_date: str, organization: str, n: int) -> str:
    """`<due>.<org>.<N>.json`; the org may not contain the separator."""
    if not _ORG_NAME.match(organization or ""):
        raise ValueError(f"organization {organization!r} must be letters, digits, - or _")
    if not 1 <= n <= MAX_SETS_PER_ROUND:
        raise ValueError(f"N must be 1..{MAX_SETS_PER_ROUND}, got {n}")
    return f"{due_date}.{organization}.{n}.json"


def make_forecast(question: dict, probability: float, resolution_date: str | None,
                  reasoning: str | None) -> dict:
    return {"id": question["id"], "source": question["source"],
            "forecast": round(float(probability), 4),
            "resolution_date": resolution_date, "reasoning": reasoning}


def _check_forecast(f: dict, expected: dict) -> list[str]:
    errors = []
    if set(f) != FORECAST_KEYS:
        errors.append(f"forecast keys {sorted(f)} != {sorted(FORECAST_KEYS)}")
        return errors
    key = (f["source"], f["id"])
    p = f["forecast"]
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0:
        errors.append(f"{key}: forecast {p!r} is not a number in [0,1]")
    if f["reasoning"] is not None and not isinstance(f["reasoning"], str):
        errors.append(f"{key}: reasoning must be a string or null")
    if key not in expected:
        errors.append(f"{key}: not a question in this set")
    elif f["resolution_date"] not in expected[key]:
        errors.append(f"{key}: resolution_date {f['resolution_date']!r} not in {expected[key]}")
    return errors


def _coverage_errors(forecasts: list[dict], expected: dict, by_kind: dict) -> list[str]:
    errors = []
    given = {(f.get("source"), f.get("id"), f.get("resolution_date")) for f in forecasts}
    for kind, keys in by_kind.items():
        wanted = [(s, i, d) for (s, i) in keys for d in expected[(s, i)]]
        if not wanted:
            continue
        share = sum(w in given for w in wanted) / len(wanted)
        if share < MIN_COVERAGE:
            errors.append(f"{kind} coverage {share:.1%} is under {MIN_COVERAGE:.0%}")
    return errors


def validate_forecast_set(forecast_set: dict, question_set: dict) -> list[str]:
    """Every way `forecast_set` breaks the guide's schema. [] means valid."""
    errors = []
    if set(forecast_set) != TOP_LEVEL_KEYS:
        errors.append(f"top-level keys {sorted(forecast_set)} != {sorted(TOP_LEVEL_KEYS)}"
                      " (missing model/organization/... or extras)")
    for key in ("organization", "model", "model_organization"):
        if key in forecast_set and not (isinstance(forecast_set[key], str)
                                        and forecast_set[key].strip()):
            errors.append(f"{key} must be a non-empty string")
    if forecast_set.get("question_set") != question_set["question_set"]:
        errors.append(f"question_set {forecast_set.get('question_set')!r} != "
                      f"{question_set['question_set']!r}")
    forecasts = forecast_set.get("forecasts")
    if not isinstance(forecasts, list):
        return errors + ["forecasts must be a list"]

    questions = standard_questions(question_set)
    expected = {(q["source"], q["id"]): resolution_dates(q) for q in questions}
    by_kind = {"market": [], "dataset": []}
    for q in questions:
        by_kind[question_kind(q)].append((q["source"], q["id"]))

    seen = set()
    for f in forecasts:
        errors.extend(_check_forecast(f, expected))
        triple = (f.get("source"), f.get("id"), f.get("resolution_date"))
        if triple in seen:
            errors.append(f"duplicate forecast for {triple}")
        seen.add(triple)
    return errors + _coverage_errors(forecasts, expected, by_kind)
