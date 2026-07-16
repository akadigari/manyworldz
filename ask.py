"""Ask the crowd anything.

This is the front door to the whole engine. You type a yes/no question about
the future; a crowd of AI characters reads it (plus fresh headlines), each
forms its own probability — or, in simulate mode, imagines the outcome
playing out several times — and you get one honest number plus everyone's
reasoning. Add --whatif to force a fact to be true and watch the number move.

Examples:
    venv/bin/python ask.py "Will the Fed cut rates in September?"
    venv/bin/python ask.py "Will it snow in DC this December?" --simulate
    venv/bin/python ask.py "Will the album drop this month?" \\
        --whatif "the label just confirmed the release date"

Needs ANTHROPIC_API_KEY in the environment. A typical question costs about
a cent; answers are cached, so asking the same thing twice is free.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from engine import llm, news
from engine.personas import build_crowd
from engine.swarm import run_crowd
from engine.whatif import run_whatif


def ask_question(question: str, whatif: str | None = None, mode: str = "vote",
                 k: int | None = None, n_agents: int | None = None,
                 with_news: bool = True, ask_fn=None,
                 model: str | None = None) -> dict:
    """Run the crowd on any question and return the full result.

    The question becomes a "card" with no market price (mid=None), so the
    agents are told honestly that they have nothing to anchor on. Returns
    the run_crowd dict — or, when a what-if fact is given, the
    before/after/shift dict from run_whatif.
    """
    card = {"ticker": "ASK", "question": question, "mid": None}
    if n_agents is None:
        n_agents = config.ENGINE_N_AGENTS
    if k is None:
        k = config.SIM_ROLLOUTS_K
    crowd = build_crowd(n_agents, config.SEED)
    headlines = news.research(question) if with_news else []
    if ask_fn is not None:
        ask = ask_fn
    elif model is not None:
        # Carry the chosen model into every crowd call (e.g. --model fable).
        chosen = model
        def ask(prompt, model=None, max_tokens=400):
            return llm.ask(prompt, model=chosen, max_tokens=max_tokens)
    else:
        ask = llm.ask
    if whatif:
        return run_whatif(card, headlines, crowd, whatif, mode, k,
                          config.DELIBERATION, ask)
    return run_crowd(card, headlines, crowd, mode, k,
                     config.DELIBERATION, ask)


def _print_crowd(result: dict) -> None:
    """Show one crowd run in plain, readable lines."""
    print(f"\n  THE CROWD SAYS: {result['probability']:.0%} chance of YES")
    print(f"  (disagreement spread {result['spread']:.2f}, "
          f"{result['skipped']} unusable answers skipped)\n")
    for vote in result["votes"]:
        print(f"  {vote['probability']:>5.0%}  {vote['agent']:<6} "
              f"({vote['archetype']}): {vote['reason']}")
    if result["futures"]:
        print("\n  FUTURES THE CROWD SAW:")
        for future in result["futures"]:
            mark = "+" if future["resolves"] == "YES" else "-"
            print(f"   {mark} {future['story']}  ({future['agent']})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask the AI crowd for the odds of anything.")
    parser.add_argument("question", help="a yes/no question about the future")
    parser.add_argument("--whatif", metavar="FACT",
                        help="re-run the crowd with this fact forced true")
    parser.add_argument("--simulate", action="store_true",
                        help="each agent imagines K futures instead of one vote")
    parser.add_argument("--agents", type=int, default=None,
                        help=f"crowd size (default {config.ENGINE_N_AGENTS})")
    parser.add_argument("--model", default=None,
                        help="haiku, sonnet, opus, fable, or any full model ID "
                             f"(default {config.ENGINE_MODEL})")
    parser.add_argument("--no-news", action="store_true",
                        help="skip the headline lookup")
    args = parser.parse_args()

    mode = "simulate" if args.simulate else "vote"
    result = ask_question(args.question, whatif=args.whatif, mode=mode,
                          n_agents=args.agents, with_news=not args.no_news,
                          model=args.model)

    print(f'\nQ: {args.question}')
    if args.whatif:
        print(f'\n  BEFORE the what-if:')
        _print_crowd(result["before"])
        print(f'  WHAT-IF "{args.whatif}" IS TRUE:')
        _print_crowd(result["after"])
        shift = result["shift"]
        direction = "up" if shift > 0 else "down"
        print(f"  THE FACT MOVES THE ODDS {direction.upper()} "
              f"{abs(shift):.0%} ({shift:+.0%})\n")
    else:
        _print_crowd(result)
    print(f"  total engine spend so far: ${llm.spent_usd():.2f} "
          f"(cap ${config.ENGINE_BUDGET_USD:.2f})\n")


if __name__ == "__main__":
    main()
