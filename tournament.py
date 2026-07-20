"""One cycle of the Metaculus FutureEval bot tournament.

FutureEval (metaculus.com/futureeval) is Metaculus's ongoing bot
tournament: AI forecasters answer real open questions, and the
tournament grades them against how the world actually turns out. This
file is the one command that lets manyworldz's own crowd take part in
it: fetch the tournament's open binary questions, run the existing
crowd on each one this run hasn't already answered, clamp the crowd's
probability into the range Metaculus accepts, submit it, and log every
submission to data/tournament_log.csv.

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
3. A refresh pass. Scoring reads the LAST forecast on file before a
   question closes, so an already-answered question whose close time
   is coming up soon gets one more fresh look, see _pick_refresh_targets.
4. A one-line, plain-English coverage report printed every cycle.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from adapters import metaculus
from engine import llm, news
from engine.ensemble import build_crowd_for
from engine.swarm import run_crowd

LOG_COLUMNS = ["qid", "question", "raw_prob", "prob", "at", "source"]
LOG_PATH = config.DATA / "tournament_log.csv"

# The honest last-resort answer when both the full crowd and a single
# simplified run fail on one question. 0.5 is not a guess dressed up as
# confidence: it is the plain truth that nothing usable came back, and
# it always gets logged with source="fallback" so it is never mistaken
# for a real crowd answer later.
LAST_RESORT_PROBABILITY = 0.5

# How old a question's last submission needs to be, in hours, before the
# refresh pass (see _pick_refresh_targets) will touch it again, on top
# of also being close to closing. Matched to
# .github/workflows/tournament.yml's own 6-hour cron cadence: a question
# answered less than one cycle ago is already fresh, no need to spend
# another crowd run on it yet.
RESUBMIT_STALE_HOURS = 6


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


def _pick_refresh_targets(cards: list[dict], history: dict, now_dt: datetime,
                          refresh_hours: float, stale_hours: float,
                          cap: int) -> list[dict]:
    """Already-answered questions worth one more fresh crowd run this
    cycle: still open, close time coming up inside `refresh_hours`, and
    the last submission on file is older than `stale_hours`.

    Scoring reads the LAST forecast before close, so a stale answer
    sitting on a soon-to-close question is worth spending another crowd
    run on. Candidates are sorted soonest-to-close first, then capped
    at `cap`: if there are more stale, urgent questions than the budget
    allows, the most urgent ones win. Anything with a missing or
    unparsable close_time or last-submission timestamp is skipped, not
    guessed at.
    """
    candidates = []
    for card in cards:
        qid = card.get("qid")
        if qid not in history:
            continue                  # never answered yet: that's targets, not refresh
        close_dt = _parse_iso(card.get("close_time", ""))
        if close_dt is None:
            continue
        hours_to_close = (close_dt - now_dt).total_seconds() / 3600.0
        if hours_to_close < 0 or hours_to_close > refresh_hours:
            continue                  # already closed, or not urgent yet
        last_dt = _parse_iso(history[qid])
        if last_dt is None:
            continue
        age_hours = (now_dt - last_dt).total_seconds() / 3600.0
        if age_hours < stale_hours:
            continue                  # already fresh enough, leave it alone
        candidates.append((hours_to_close, card))
    candidates.sort(key=lambda pair: pair[0])
    return [card for _, card in candidates[:cap]]


def _append_log(row: dict, log_path: Path) -> None:
    """Append one submission's row right away, not batched at the end.

    If something later in the run blows up (say, the budget cap), every
    submission that already went out to Metaculus is still safely on
    disk. Losing the record of a real submission just because a later
    question failed would be worse than a slightly messier write
    pattern.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _is_budget_error(exc: Exception) -> bool:
    """True for the specific RuntimeError engine/llm.py raises once
    ENGINE_BUDGET_USD is spent (see engine/llm.py's ask()).

    That error can never be treated as "this one crowd run failed, try
    the next tier down": it means the whole cycle needs to stop right
    now, cleanly, not spend more money trying a single run and then a
    fallback on every question still left in the batch.
    """
    return "budget cap hit" in str(exc)


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
                   "question": card["question"], "mid": None}

    try:
        result = _run_full_crowd(market_card, headlines, crowd, ask)
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
        result = _run_single(market_card, headlines, crowd, ask)
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
             fetch_fn=None, submit_fn=None) -> dict:
    """Run one full tournament cycle.

    Pass `cards` (fake question cards) and `ask_fn` (a fake crowd) to
    run the whole cycle in tests, with no network calls at all. Leave
    both blank for a real run: cards come from
    adapters/metaculus.fetch_open_questions, and unless dry_run is set,
    every answer gets posted through adapters/metaculus.submit_prediction.

    Every question this cycle touches, fresh or refreshed, goes through
    the fallback ladder in _answer_one, so a broken model call or a
    dead seat degrades to a worse answer instead of no answer at all.
    Every actual submission (crowd, single, or fallback) appends one
    row to the ledger with both the crowd's raw number and the clipped,
    submitted one; see LOG_COLUMNS. A budget RuntimeError from
    engine/llm.py is never caught: it propagates straight out of this
    function so the whole cycle stops cleanly, with no fallback-spam
    on whatever questions were still left.

    Returns a small summary: how many open questions were seen, how
    many got a fresh answer this cycle, how many stale near-close
    answers got refreshed, how many total submissions went out, and how
    many of those were last-resort fallbacks.
    """
    live = cards is None            # no fake cards given -> this is a real, live run
    ask = ask_fn or llm.ask
    fetch_fn = fetch_fn or metaculus.fetch_open_questions
    submit_fn = submit_fn or metaculus.submit_prediction
    tournament = tournament or config.METACULUS_TOURNAMENT
    log_path = log_path or LOG_PATH
    now = now_iso or datetime.now(timezone.utc).isoformat()
    now_dt = _parse_iso(now) or datetime.now(timezone.utc)

    if live:
        cards = fetch_fn(tournament, token)

    history = _log_history(log_path)
    already = set(history.keys())
    pending = [c for c in cards if c.get("qid") not in already]
    targets = pending[:config.TOURNAMENT_QUESTIONS_PER_RUN]
    refresh_targets = _pick_refresh_targets(
        cards, history, now_dt, config.TOURNAMENT_REFRESH_HOURS,
        RESUBMIT_STALE_HOURS, config.TOURNAMENT_REFRESH_CAP)

    crowd = build_crowd_for()
    counts = {"answered": 0, "refreshed": 0, "submitted": 0, "fallbacks": 0}

    def _process(card: dict, is_refresh: bool) -> None:
        headlines = news.research(card["question"]) if live else []
        outcome = _answer_one(card, headlines, crowd, ask)
        if outcome["prob"] is None:
            print(f'  no quorum (all {outcome["skipped"]} answers unusable), '
                  f'skipping qid {card["qid"]} | {card["question"][:60]}')
            return

        raw_prob = outcome["prob"]
        submit_prob = _tournament_clip(raw_prob)
        counts["refreshed" if is_refresh else "answered"] += 1
        if outcome["source"] == "fallback":
            counts["fallbacks"] += 1

        if dry_run:
            tag = "REFRESH (dry run)" if is_refresh else "DRY RUN"
            print(f'  {tag} would submit {submit_prob:.2f} '
                  f'(source={outcome["source"]}) on qid {card["qid"]} '
                  f'| {card["question"][:60]}')
            return

        submit_fn(card["qid"], submit_prob, token)
        row = {"qid": card["qid"], "question": card["question"],
              "raw_prob": raw_prob, "prob": submit_prob, "at": now,
              "source": outcome["source"]}
        _append_log(row, log_path)
        counts["submitted"] += 1
        label = "REFRESH" if is_refresh else "SUBMIT"
        print(f'  {label} {submit_prob:.2f} (source={outcome["source"]}) '
              f'on qid {card["qid"]} | {card["question"][:60]}')

    for card in targets:
        _process(card, is_refresh=False)
    for card in refresh_targets:
        _process(card, is_refresh=True)

    all_time_answered = len(_log_history(log_path))
    print(f'tournament cycle done: {len(cards)} open, '
         f'{counts["answered"]} answered this cycle, '
         f'{counts["refreshed"]} refreshed, {counts["fallbacks"]} fallback(s), '
         f'{all_time_answered} answered all time, '
         f'${llm.spent_usd():.2f} spent')
    return {"considered": len(cards), "answered": counts["answered"],
           "refreshed": counts["refreshed"], "submitted": counts["submitted"],
           "fallbacks": counts["fallbacks"]}


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
    if not token:
        print("METACULUS_TOKEN is not set. Nothing was fetched or "
              "submitted this cycle. Once the tournament account and "
              "its token exist, export METACULUS_TOKEN and run this "
              "again: venv/bin/python tournament.py")
        return

    one_cycle(tournament=args.tournament, dry_run=args.dry_run, token=token)


if __name__ == "__main__":
    main()
