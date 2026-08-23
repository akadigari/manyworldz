"""One cycle of the Metaculus FutureEval bot tournament.

FutureEval (metaculus.com/futureeval) is Metaculus's ongoing bot
tournament: AI forecasters answer real open questions, and the
tournament grades them against how the world actually turns out. This
file is the one command that lets manyworldz's own crowd take part in
it: fetch the tournament's open questions of every type it runs
(binary, multiple choice, numeric, discrete), answer each one this
account hasn't already answered, submit it with the required private
comment, and log every submission to data/tournament_log.csv.

A person only ever does two things by hand: create the bot account on
Metaculus and generate its METACULUS_TOKEN. Everything after that is
this one command:

    venv/bin/python tournament.py

Needs METACULUS_TOKEN and ANTHROPIC_API_KEY in the environment for a
real run. Missing either one is not treated as a crash: see main()
below. Pass --dry-run to see exactly what the crowd would submit
without ever calling the write endpoint.

This tournament scores by SUMMING peer scores across every question,
and prize share goes with the SQUARE of that sum. A question this bot
never answers scores a hard zero while every rival who did answer
banks theirs, so coverage matters more than cleverness. Four things
below all serve that one goal:

1. A fallback ladder (see _answer_one). The full crowd answers a
   question if at all possible. If the crowd run itself blows up, one
   single simplified run is tried next. If that also blows up, a
   documented last-resort probability is submitted instead, logged
   plainly with source="fallback", never silently. A question only
   ever gets skipped outright if the crowd ran clean, start to finish,
   and every single answer was honestly unusable: that is not a
   crash, so there is nothing to retry into being different.
2. config.TOURNAMENT_CLIP. Every submitted probability is clipped a
   bit short of 0% and 100% before it goes out, see _tournament_clip.
3. One forecast per question, never more. The tournament rules say bot
   makers should only submit one forecast per question, so an already
   answered question is left alone, whether the record of it comes
   from the local log or from the API's own my_forecasts flag.
4. A per-question deadline (config.QUESTION_DEADLINE_S). The questions
   are only open 1.5 to 3 hours; a hung call degrades to the ladder
   instead of eating the window.
5. A one-line, plain-English coverage report printed every cycle.

Question types: binary goes through the full crowd. Multiple choice
and numeric/discrete get one direct model call each (options ->
probabilities, or five percentiles -> engine/cdf.py's full CDF), with
an honest uniform fallback if the reply is unusable, because an
unanswered question scores a hard zero and a flat answer does not.
Every submission is followed by the required private comment; a failed
comment never takes the forecast down with it.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from adapters import metaculus
from engine import cdf as cdf_mod
from engine import llm, news
from engine.ensemble import build_crowd_for
from engine.swarm import run_crowd

LOG_COLUMNS = ["qid", "question", "qtype", "raw_prob", "prob", "at", "source"]
LOG_PATH = config.DATA / "tournament_log.csv"

# The honest last-resort answer when both the full crowd and a single
# simplified run fail on one question. 0.5 is not a guess dressed up as
# confidence: it is the plain truth that nothing usable came back, and
# it always gets logged with source="fallback" so it is never mistaken
# for a real crowd answer later.
LAST_RESORT_PROBABILITY = 0.5

def _clamp(probability: float) -> float:
    """Metaculus's accepted range for a binary forecast: never a claimed
    0% or 100% chance. Same clamp engine/swarm.py's consensus already
    applies and the same range the official bot template uses; this is
    just a second, cheap safety net before the number leaves this file.
    """
    return min(max(float(probability), 0.01), 0.99)


def _tournament_clip(probability: float) -> float:
    """Clip a probability into [TOURNAMENT_CLIP, 1 - TOURNAMENT_CLIP]
    right before it is submitted.

    This is stricter than _clamp's plain 1%-99% floor above. Log
    scoring punishes a confident miss brutally, and the bots that
    actually rank near the top of these tournaments clip their extremes
    for exactly that reason: giving up a little best-case score buys a
    lot less downside on the misses. Only the submitted number is
    clipped here; the crowd's own raw number is recorded untouched in
    the ledger's "raw_prob" column, see one_cycle below.
    """
    clip = config.TOURNAMENT_CLIP
    return round(min(max(float(probability), clip), 1 - clip), 4)


def _parse_iso(ts) -> datetime | None:
    """Parse an ISO timestamp defensively. None for anything that isn't
    one: a missing or corrupt timestamp should mean "we can't reason
    about this one" and get skipped, never crash the cycle."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _log_history(log_path: Path) -> dict:
    """Every question this log has ever answered, mapped to its most
    recent submission's "at" timestamp string.

    A qid can now show up on more than one row: the refresh pass (see
    _pick_refresh_targets) resubmits a stale, soon-to-close question, so
    later rows for the same qid always win, since csv rows are appended
    in time order and never rewritten.
    """
    history: dict = {}
    if not log_path.exists():
        return history
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                qid = int(row["qid"])
            except (KeyError, TypeError, ValueError):
                continue          # a corrupt row should never crash the cycle
            at = row.get("at", "")
            if at:
                history[qid] = at
    return history


def _already_answered(log_path: Path) -> set:
    """Question ids this log already has a submission for, so a rerun
    never re-answers the same question twice in a row. (Metaculus lets
    you update a forecast any time; a fresh run is just meant to visit
    each open question once, same spirit as ledger.log_pick's
    one-open-position rule. The refresh pass is the one deliberate
    exception: see _pick_refresh_targets.)
    """
    return set(_log_history(log_path).keys())


def _append_log(row: dict, log_path: Path) -> None:
    """Append one submission's row right away, not batched at the end.

    If something later in the run blows up (say, the budget cap), every
    submission that already went out to Metaculus is still safely on
    disk. Losing the record of a real submission just because a later
    question failed would be worse than a slightly messier write
    pattern.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _migrate_log(log_path)
    is_new = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _migrate_log(log_path: Path) -> None:
    """Bring an old-format log up to LOG_COLUMNS before appending.

    The pre-hardening log had four columns (qid,question,prob,at).
    Appending seven-field rows under that header would land values in
    the wrong columns and silently corrupt the ledger. Old rows keep
    their values under their own column names; the fields they never
    had are filled with honest blanks (qtype defaults to binary, the
    only type that existed back then).
    """
    if not log_path.exists():
        return
    with open(log_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if header is None or header == LOG_COLUMNS:
        return
    with open(log_path, newline="") as f:
        old_rows = list(csv.DictReader(f))
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, restval="")
        writer.writeheader()
        for old in old_rows:
            writer.writerow({
                "qid": old.get("qid", ""),
                "question": old.get("question", ""),
                "qtype": old.get("qtype", "binary"),
                "raw_prob": old.get("raw_prob", old.get("prob", "")),
                "prob": old.get("prob", ""),
                "at": old.get("at", ""),
                "source": old.get("source", ""),
            })


def _is_budget_error(exc: Exception) -> bool:
    """True for the specific RuntimeError engine/llm.py raises once
    ENGINE_BUDGET_USD is spent (see engine/llm.py's ask()).

    That error can never be treated as "this one crowd run failed, try
    the next tier down": it means the whole cycle needs to stop right
    now, cleanly, not spend more money trying a single run and then a
    fallback on every question still left in the batch.
    """
    return "budget cap hit" in str(exc)


def _question_text(card: dict) -> str:
    """The full question the model should forecast: title plus the
    resolution rules. Forecasting the title alone means forecasting a
    headline; the criteria are what actually resolves the question.
    Trimmed so one enormous description can't blow up every prompt."""
    criteria = (card.get("criteria") or "").strip()
    if not criteria:
        return card["question"]
    return f"{card['question']}\n\n{criteria[:1500]}"


def _with_deadline(fn, seconds: float):
    """Run fn() with a hard wall-clock limit.

    On timeout raises TimeoutError so the caller's ladder catches it
    like any other failure. The worker thread itself can't be killed,
    but the cycle stops waiting on it, which is what protects the
    window. Deadlines under a year run through the executor; the
    tests set tiny ones, real runs use config.QUESTION_DEADLINE_S."""
    import threading
    box: dict = {}

    def _worker():
        try:
            box["result"] = fn()
        except BaseException as exc:          # delivered to the caller below
            box["error"] = exc

    # A plain daemon thread, not a ThreadPoolExecutor: executor workers
    # are non-daemon and concurrent.futures joins them at interpreter
    # exit, so one hung call would block the process AFTER the cycle
    # printed "done". A daemon thread is simply abandoned.
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=seconds)
    if thread.is_alive():
        raise TimeoutError(f"no answer within {seconds}s")
    if "error" in box:
        raise box["error"]
    return box.get("result")


_MC_PROMPT = """You are forecasting a multiple choice question.

The question: "{question}"
{evidence}

The options, exactly as written: {options}

Start from base rates: how often does each kind of outcome actually
happen? Then adjust for the evidence. Probabilities must sum to 1.
Reply with ONLY JSON mapping every option to its probability, like
{{"Option A": 0.5, "Option B": 0.3, "Option C": 0.2}}"""


def _skip_reason(card: dict) -> str | None:
    """Why a question can't be answered honestly, or None if it can.

    Malformed or unsupported cards get skipped with a printed reason,
    never guessed at and never allowed to crash the cycle: an MC card
    with no options has nothing to submit, a numeric card without
    range bounds has no grid to build, and a log-scaled question
    (zero_point set) without its own continuous_range would get a
    badly mis-shaped CDF from a linear grid.
    """
    qtype = card.get("qtype", "binary")
    if qtype == "multiple_choice" and not card.get("options"):
        return "multiple choice with no options"
    if qtype in ("numeric", "discrete"):
        scaling = card.get("scaling") or {}
        if not scaling.get("continuous_range"):
            if scaling.get("range_min") is None or scaling.get("range_max") is None:
                return "numeric with no range bounds"
            if scaling.get("zero_point") is not None:
                return "log-scaled with no continuous_range"
    return None


def _answer_mc(card: dict, headlines: list[str], ask) -> dict:
    """One direct model call for a multiple choice question.

    The crowd machinery is binary; a single well-prompted call per MC
    question is what the official template does too. An unusable reply
    degrades to the uniform distribution, logged as a fallback: a flat
    answer scores better than the hard zero of no answer."""
    from engine.swarm import extract_json
    options = card.get("options") or []
    evidence = f'Recent headlines: {"; ".join(headlines) if headlines else "(none found)"}'
    prompt = _MC_PROMPT.format(question=_question_text(card),
                               evidence=evidence, options=options)
    probs = None
    try:
        parsed = extract_json(ask(prompt, model="sonnet"))
        if isinstance(parsed, dict):
            got = {opt: float(parsed[opt]) for opt in options if opt in parsed}
            if len(got) == len(options) and all(v >= 0 for v in got.values())                     and sum(got.values()) > 0:
                probs = got
    except Exception as exc:
        if _is_budget_error(exc):
            raise
        print(f'  MC call failed on qid {card["qid"]} ({exc})')
    if probs is None:
        probs = {opt: 1.0 / len(options) for opt in options}
        return {"probs": probs, "source": "fallback"}
    return {"probs": probs, "source": "mc"}


_NUMERIC_PROMPT = """You are forecasting a numeric question.

The question: "{question}"
{evidence}

The answer is measured in: {unit}
The plausible range runs from {lo} to {hi}.{bounds_note}

Start from base rates and any relevant historical figures, then adjust
for the evidence. Give your 5th, 25th, 50th, 75th and 95th percentile
estimates, in the question's own units, wide enough to be honest about
your uncertainty. Reply with ONLY JSON like
{{"p05": 10, "p25": 20, "p50": 30, "p75": 40, "p95": 50}}"""


def _answer_numeric(card: dict, headlines: list[str], ask) -> dict:
    """One direct model call for a numeric or discrete question: five
    percentiles in, engine/cdf.py's full CDF out. An unusable reply
    degrades to the uniform CDF over the question's own range, logged
    as a fallback, for the same coverage-first reason as _answer_mc."""
    scaling = card.get("scaling") or {}
    lo, hi = scaling.get("range_min"), scaling.get("range_max")
    notes = []
    if card.get("open_lower_bound"):
        notes.append("values below the range are possible")
    if card.get("open_upper_bound"):
        notes.append("values above the range are possible")
    bounds_note = f" ({'; '.join(notes)})" if notes else ""
    evidence = f'Recent headlines: {"; ".join(headlines) if headlines else "(none found)"}'
    prompt = _NUMERIC_PROMPT.format(question=_question_text(card),
                                    evidence=evidence,
                                    unit=card.get("unit") or "(unitless)",
                                    lo=lo, hi=hi, bounds_note=bounds_note)
    percentiles, source = None, "numeric"
    try:
        percentiles = cdf_mod.percentiles_from_json(ask(prompt, model="sonnet"))
    except Exception as exc:
        if _is_budget_error(exc):
            raise
        print(f'  numeric call failed on qid {card["qid"]} ({exc})')
    if percentiles is None:
        # the honest flat answer: percentiles evenly spread over the range
        span = float(hi) - float(lo)
        percentiles = {p: float(lo) + span * p for p in (0.05, 0.25, 0.5, 0.75, 0.95)}
        source = "fallback"
    values = cdf_mod.build_cdf(percentiles, scaling,
                               open_lower=card.get("open_lower_bound", False),
                               open_upper=card.get("open_upper_bound", False))
    return {"cdf": values, "percentiles": percentiles, "source": source}


def _run_full_crowd(market_card: dict, headlines: list[str], crowd: list[dict], ask):
    """The fallback ladder's first tier: the whole configured crowd,
    same simulate-mode run this file has always done."""
    return run_crowd(market_card, headlines, crowd, mode="simulate",
                     k=config.SIM_ROLLOUTS_K, deliberation=config.DELIBERATION,
                     ask_fn=ask)


def _run_single(market_card: dict, headlines: list[str], crowd: list[dict], ask):
    """The fallback ladder's second tier: one agent, one plain vote, no
    K-futures simulation and no deliberation round. Cheaper and simpler
    than the full crowd, so it has the best odds of surviving whatever
    just broke the full run.
    """
    seat = crowd[:1] or crowd
    return run_crowd(market_card, headlines, seat, mode="vote", k=1,
                     deliberation=False, ask_fn=ask)


def _answer_one(card: dict, headlines: list[str], crowd: list[dict], ask) -> dict:
    """Get an answer for one question, degrading instead of skipping
    whenever something actually breaks.

    Tries the full crowd first. Any failure that raises an exception
    and isn't the budget cap (a model error, a timeout, one seat dying)
    degrades to a single simplified run; if that also raises, it
    degrades again to the documented LAST_RESORT_PROBABILITY. None of
    those three outcomes ever comes back as "skipped".

    The one case that does come back with prob=None is a real
    no-quorum result: the full crowd ran clean, start to finish, but
    every single agent's answer was honestly unusable (bad JSON, no
    parseable probability). That's not a crash, just the crowd having
    nothing to say, and it's left alone exactly like it always has
    been: there's nothing broken to retry into being different, and
    fabricating a number here would be a worse lie than fabricating one
    after an actual failure.

    Returns {"prob": float | None, "source": "crowd" | "single" |
    "fallback" | None, "skipped": int}. A budget RuntimeError is never
    caught here: it always propagates straight out, and out of
    one_cycle, so the cycle stops cleanly instead of fallback-spamming
    every question still left in the batch.
    """
    market_card = {"ticker": f"META-{card['qid']}",
                   "question": _question_text(card), "mid": None}

    try:
        result = _with_deadline(
            lambda: _run_full_crowd(market_card, headlines, crowd, ask),
            config.QUESTION_DEADLINE_S)
    except Exception as exc:
        if _is_budget_error(exc):
            raise
        print(f'  crowd run failed on qid {card["qid"]} ({exc}), '
              f'retrying with a single run')
        return _answer_with_single_or_fallback(card, headlines, crowd, ask, market_card)

    if result["votes"]:
        return {"prob": result["probability"], "source": "crowd",
               "skipped": result["skipped"]}
    return {"prob": None, "source": None, "skipped": result["skipped"]}


def _answer_with_single_or_fallback(card: dict, headlines: list[str],
                                    crowd: list[dict], ask, market_card: dict) -> dict:
    """The fallback ladder's second and third tiers, called only after
    the full crowd itself has already raised a non-budget exception."""
    try:
        result = _with_deadline(
            lambda: _run_single(market_card, headlines, crowd, ask),
            config.QUESTION_DEADLINE_S)
    except Exception as exc:
        if _is_budget_error(exc):
            raise
        print(f'  single run also failed on qid {card["qid"]} ({exc})')
        result = {"votes": [], "skipped": 0}

    if result["votes"]:
        return {"prob": result["probability"], "source": "single",
               "skipped": result["skipped"]}

    print(f'  LAST RESORT: no usable answer from the crowd or a single run '
          f'on qid {card["qid"]} | {card["question"][:60]}, submitting the '
          f'documented fallback probability {LAST_RESORT_PROBABILITY} '
          f'(source=fallback, never silent)')
    return {"prob": LAST_RESORT_PROBABILITY, "source": "fallback",
           "skipped": result["skipped"]}


def one_cycle(tournament=None, cards: list[dict] | None = None, ask_fn=None,
             dry_run: bool = False, token: str | None = None,
             now_iso: str | None = None, log_path: Path | None = None,
             fetch_fn=None, submit_fn=None, submit_mc_fn=None,
             submit_numeric_fn=None, comment_fn=None) -> dict:
    """Run one full tournament cycle.

    Pass `cards` (fake question cards) and `ask_fn` (a fake crowd) to
    run the whole cycle in tests, with no network calls at all. Leave
    both blank for a real run: cards come from
    adapters/metaculus.fetch_open_questions, and unless dry_run is set,
    every answer gets posted through the matching adapters/metaculus
    submit function for its question type, followed by the required
    private comment (adapters/metaculus.post_comment; a failed comment
    never blocks anything).

    One forecast per question, per the tournament rules: a question is
    skipped when the local log has it OR the API's own my_forecasts
    flag says this account already answered it. Binary questions go
    through the fallback ladder in _answer_one under a per-question
    deadline; multiple choice and numeric/discrete get one direct call
    each with an honest uniform fallback. A budget RuntimeError from
    engine/llm.py is never caught: it propagates so the cycle stops
    cleanly.

    Returns a small summary: how many open questions were seen, how
    many were answered, how many submissions went out, and how many
    were last-resort fallbacks.
    """
    live = cards is None            # no fake cards given -> this is a real, live run
    ask = ask_fn or llm.ask
    fetch_fn = fetch_fn or metaculus.fetch_open_questions
    submit_fn = submit_fn or metaculus.submit_prediction
    submit_mc_fn = submit_mc_fn or metaculus.submit_mc_prediction
    submit_numeric_fn = submit_numeric_fn or metaculus.submit_numeric_prediction
    comment_fn = comment_fn or metaculus.post_comment
    tournament = tournament or config.METACULUS_TOURNAMENT
    log_path = log_path or LOG_PATH
    now = now_iso or datetime.now(timezone.utc).isoformat()

    if live:
        cards = fetch_fn(tournament, token)

    already = _already_answered(log_path)
    pending = [c for c in cards
               if c.get("qid") not in already
               and not c.get("already_forecast")]
    targets = pending[:config.TOURNAMENT_QUESTIONS_PER_RUN]

    crowd = build_crowd_for()
    counts = {"answered": 0, "submitted": 0, "fallbacks": 0}

    def _comment(card: dict, text: str) -> None:
        post_id = card.get("post_id")
        if post_id is None:
            return
        try:
            comment_fn(post_id, text, token)
        except Exception as exc:
            print(f'  comment on post {post_id} failed ({exc}); forecast stands')

    def _record(card: dict, raw, submitted, source: str) -> None:
        counts["answered"] += 1
        if source == "fallback":
            counts["fallbacks"] += 1
        row = {"qid": card["qid"], "question": card["question"],
              "qtype": card.get("qtype", "binary"), "raw_prob": raw,
              "prob": submitted, "at": now, "source": source}
        _append_log(row, log_path)
        counts["submitted"] += 1

    def _process(card: dict) -> None:
        qtype = card.get("qtype", "binary")
        headlines = news.research(card["question"]) if live else []

        if qtype == "multiple_choice":
            try:
                outcome = _with_deadline(
                    lambda: _answer_mc(card, headlines, ask),
                    config.QUESTION_DEADLINE_S)
            except Exception as exc:
                if _is_budget_error(exc):
                    raise
                options = card.get("options") or []
                outcome = {"probs": {o: 1.0 / len(options) for o in options},
                          "source": "fallback"}
            if dry_run:
                counts["answered"] += 1
                if outcome["source"] == "fallback":
                    counts["fallbacks"] += 1
                print(f'  DRY RUN would submit {outcome["probs"]} on qid {card["qid"]}')
                return
            submit_mc_fn(card["qid"], outcome["probs"], token)
            _record(card, json.dumps(outcome["probs"]),
                    json.dumps(outcome["probs"]), outcome["source"])
            _comment(card, f'manyworldz {outcome["source"]} forecast: '
                          f'{json.dumps(outcome["probs"])}')
            print(f'  SUBMIT MC (source={outcome["source"]}) on qid '
                  f'{card["qid"]} | {card["question"][:60]}')
            return

        if qtype in ("numeric", "discrete"):
            try:
                outcome = _with_deadline(
                    lambda: _answer_numeric(card, headlines, ask),
                    config.QUESTION_DEADLINE_S)
            except Exception as exc:
                if _is_budget_error(exc):
                    raise
                print(f'  numeric path failed on qid {card["qid"]} ({exc}), skipping')
                return
            if dry_run:
                counts["answered"] += 1
                if outcome["source"] == "fallback":
                    counts["fallbacks"] += 1
                print(f'  DRY RUN would submit a {len(outcome["cdf"])}-point CDF '
                      f'on qid {card["qid"]}')
                return
            submit_numeric_fn(card["qid"], outcome["cdf"], token)
            _record(card, json.dumps(outcome["percentiles"]),
                    f'cdf:{len(outcome["cdf"])}', outcome["source"])
            _comment(card, f'manyworldz {outcome["source"]} forecast, '
                          f'percentiles: {json.dumps(outcome["percentiles"])}')
            print(f'  SUBMIT {qtype.upper()} (source={outcome["source"]}) on qid '
                  f'{card["qid"]} | {card["question"][:60]}')
            return

        outcome = _answer_one(card, headlines, crowd, ask)
        if outcome["prob"] is None:
            print(f'  no quorum (all {outcome["skipped"]} answers unusable), '
                  f'skipping qid {card["qid"]} | {card["question"][:60]}')
            return

        raw_prob = outcome["prob"]
        submit_prob = _tournament_clip(raw_prob)
        if dry_run:
            counts["answered"] += 1
            if outcome["source"] == "fallback":
                counts["fallbacks"] += 1
            print(f'  DRY RUN would submit {submit_prob:.2f} '
                  f'(source={outcome["source"]}) on qid {card["qid"]} '
                  f'| {card["question"][:60]}')
            return

        submit_fn(card["qid"], submit_prob, token)
        _record(card, raw_prob, submit_prob, outcome["source"])
        _comment(card, f'manyworldz {outcome["source"]} forecast: '
                      f'probability {submit_prob} that this resolves YES')
        print(f'  SUBMIT {submit_prob:.2f} (source={outcome["source"]}) '
              f'on qid {card["qid"]} | {card["question"][:60]}')

    for card in targets:
        reason = _skip_reason(card)
        if reason:
            print(f'  skipping qid {card.get("qid")} ({reason})')
            continue
        try:
            _process(card)
        except Exception as exc:
            # One question's failure (a rejected submit, a malformed
            # field) must never zero out the rest of the batch. The
            # budget cap is the one exception: it means stop spending.
            if _is_budget_error(exc):
                raise
            print(f'  qid {card.get("qid")} failed this cycle ({exc}); '
                  f'moving on')

    all_time_answered = len(_log_history(log_path))
    # A small status receipt for the daily-brief routine: its cloud
    # sandbox cannot reach metaculus.com, so the bot records what it
    # saw. Committed back with the log by the workflow.
    status_path = config.DATA / "tournament_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps({
        "at": now, "tournament": tournament, "open_seen": len(cards),
        "answered_all_time": all_time_answered,
        "answered_this_cycle": counts["answered"],
        "fallbacks_this_cycle": counts["fallbacks"],
    }))
    print(f'tournament cycle done: {len(cards)} open, '
         f'{counts["answered"]} answered this cycle, '
         f'{counts["fallbacks"]} fallback(s), '
         f'{all_time_answered} answered all time, '
         f'${llm.spent_usd():.2f} spent')
    return {"considered": len(cards), "answered": counts["answered"],
           "submitted": counts["submitted"], "fallbacks": counts["fallbacks"]}


class TokenState:
    """Whether this cycle is armed, still waiting, or overdue.

    An unset token exits 0 on purpose so the schedule can stay active before
    the bot account exists. That grace has an end date: past it, a silent
    no-op is a failure, not a pending task.
    """

    def __init__(self, status: str, message: str, exit_code: int):
        self.status = status
        self.message = message
        self.exit_code = exit_code


def token_state(token, now=None, arm_by=None) -> TokenState:
    now = now or datetime.now(timezone.utc)
    arm_by = arm_by or config.TOURNAMENT_ARM_BY

    if token:
        return TokenState("armed", "METACULUS_TOKEN present; forecasting.", 0)

    if now < arm_by:
        days = (arm_by - now).days
        return TokenState(
            "waiting",
            f"METACULUS_TOKEN is not set. Nothing fetched or submitted. "
            f"{days} days left to arm this before {arm_by.date()} "
            f"(Fall FutureEval). Set it as a GitHub Actions secret.",
            0,
        )

    overdue = (now - arm_by).days
    return TokenState(
        "overdue",
        f"METACULUS_TOKEN still unset {overdue} days past the "
        f"{arm_by.date()} arm-by date. This bot has been scoring zero on "
        f"every question. Failing loudly on purpose.",
        1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one FutureEval tournament cycle.")
    parser.add_argument("--tournament", default=None,
                        help=f"tournament ID or slug (default "
                             f"{config.METACULUS_TOURNAMENT!r})")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be submitted, post nothing")
    args = parser.parse_args()

    token = os.environ.get("METACULUS_TOKEN")
    state = token_state(token)
    if state.status != "armed":
        # ::warning:: / ::error:: surface as annotations in the Actions UI,
        # so an unarmed bot is visible without opening the log.
        level = "error" if state.status == "overdue" else "warning"
        print(f"::{level}::{state.message}")
        print(state.message)
        raise SystemExit(state.exit_code)

    one_cycle(tournament=args.tournament, dry_run=args.dry_run, token=token)


if __name__ == "__main__":
    main()
