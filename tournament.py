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
from engine.methods import build_methods
from engine.swarm import run_crowd

LOG_COLUMNS = ["qid", "question", "prob", "at"]
LOG_PATH = config.DATA / "tournament_log.csv"


def _clamp(probability: float) -> float:
    """Metaculus's accepted range for a binary forecast: never a claimed
    0% or 100% chance. Same clamp engine/swarm.py's consensus already
    applies and the same range the official bot template uses; this is
    just a second, cheap safety net before the number leaves this file.
    """
    return min(max(float(probability), 0.01), 0.99)


def _already_answered(log_path: Path) -> set:
    """Question ids this log already has a submission for, so a rerun
    never re-answers the same question twice in a row. (Metaculus lets
    you update a forecast any time; this project's cycle is just meant
    to visit each open question once per run, same spirit as
    ledger.log_pick's one-open-position rule.)
    """
    seen = set()
    if not log_path.exists():
        return seen
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                seen.add(int(row["qid"]))
            except (KeyError, TypeError, ValueError):
                continue          # a corrupt row should never crash the cycle
    return seen


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

    Returns a small summary: how many open questions were seen, how
    many got a fresh answer this cycle, and how many were actually
    submitted (0 in --dry-run).
    """
    live = cards is None            # no fake cards given -> this is a real, live run
    ask = ask_fn or llm.ask
    fetch_fn = fetch_fn or metaculus.fetch_open_questions
    submit_fn = submit_fn or metaculus.submit_prediction
    tournament = tournament or config.METACULUS_TOURNAMENT
    log_path = log_path or LOG_PATH
    now = now_iso or datetime.now(timezone.utc).isoformat()

    if live:
        cards = fetch_fn(tournament, token)

    already = _already_answered(log_path)
    pending = [c for c in cards if c.get("qid") not in already]
    targets = pending[:config.TOURNAMENT_QUESTIONS_PER_RUN]

    crowd = build_methods(config.ENGINE_N_AGENTS, config.SEED)
    answered = submitted = 0

    for card in targets:
        headlines = news.research(card["question"]) if live else []
        # No market price on a Metaculus question, same honest framing
        # ask.py uses for a typed-in question: mid=None tells the
        # crowd plainly it has nothing to anchor on.
        market_card = {"ticker": f"META-{card['qid']}",
                       "question": card["question"], "mid": None}
        result = run_crowd(market_card, headlines, crowd, mode="simulate",
                           k=config.SIM_ROLLOUTS_K,
                           deliberation=config.DELIBERATION, ask_fn=ask)
        if not result["votes"]:
            # Nobody gave a usable answer. Same rule as run.py: a fake
            # 0.5 "consensus" would fabricate an opinion out of
            # nothing, so skip the question instead of submitting one.
            print(f'  no quorum (all {result["skipped"]} answers unusable), '
                  f'skipping qid {card["qid"]} | {card["question"][:60]}')
            continue

        prob = _clamp(result["probability"])
        answered += 1
        if dry_run:
            print(f'  DRY RUN would submit {prob:.2f} on qid {card["qid"]} '
                  f'| {card["question"][:60]}')
            continue

        submit_fn(card["qid"], prob, token)
        row = {"qid": card["qid"], "question": card["question"],
              "prob": prob, "at": now}
        _append_log(row, log_path)
        submitted += 1
        print(f'  SUBMIT {prob:.2f} on qid {card["qid"]} '
              f'| {card["question"][:60]}')

    print(f"tournament cycle done: {len(cards)} open question(s), "
          f"{answered} answered, {submitted} submitted, "
          f"${llm.spent_usd():.2f} total spend")
    return {"considered": len(cards), "answered": answered, "submitted": submitted}


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
