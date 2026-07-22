"""Ask the engine anything.

This is the front door to the whole engine. You type a yes/no question about
the future; the engine splits it into several independent runs. Each run
reads the question (plus fresh headlines) and imagines the outcome playing
out several different ways. The futures fold into one honest number, and you
see every imagined world. Add --whatif to force a fact to be true and watch
the number move.

Examples:
    venv/bin/python ask.py "Will the Fed cut rates in September?"
    venv/bin/python ask.py "Will it snow in DC this December?" --vote
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
from engine.carlo import run_carlo
from engine.ensemble import build_crowd_for
from engine.explore import explore_worlds, find_paths
from engine.swarm import run_crowd
from engine.whatif import run_whatif


def _resolve_ask(ask_fn=None, model: str | None = None):
    """Figure out which ask_fn to actually call with.

    Tests pass their own ask_fn, which always wins. Otherwise, if a
    --model was picked at the command line, carry it into every crowd
    call THAT DOESN'T ALREADY HAVE ITS OWN MODEL. An ensemble seat passes
    its own model straight through (see engine/swarm.py's agent_vote and
    engine/futures.py's agent_futures), and that per-seat choice always
    wins: --model is a fallback, not a stomp. A plain methods-mode agent
    never passes a model of its own, so --model works exactly as before
    for it. With neither --model nor a per-seat model, just use the plain
    default.
    """
    if ask_fn is not None:
        return ask_fn
    if model is not None:
        chosen = model
        def ask(prompt, model=None, max_tokens=400):
            return llm.ask(prompt, model=model or chosen, max_tokens=max_tokens)
        return ask
    return llm.ask


def ask_question(question: str, whatif: str | None = None,
                 mode: str = "simulate",
                 k: int | None = None, n_agents: int | None = None,
                 with_news: bool = True, ask_fn=None,
                 model: str | None = None,
                 crowd_mode: str | None = None) -> dict:
    """Run the crowd on any question and return the full result.

    The question becomes a "card" with no market price (mid=None), so the
    agents are told honestly that they have nothing to anchor on. Returns
    the run_crowd dict, or, when a what-if fact is given, the
    before/after/shift dict from run_whatif.

    `crowd_mode` overrides config.CROWD_MODE for this one call ("methods"
    or "ensemble"); leave it None to use whatever the config/env says.
    """
    card = {"ticker": "ASK", "question": question, "mid": None}
    if k is None:
        k = config.SIM_ROLLOUTS_K
    crowd = build_crowd_for(n_agents, crowd_mode)
    headlines = news.research(question) if with_news else []
    ask = _resolve_ask(ask_fn, model)
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
        print(f"  {vote['probability']:>5.0%}  {vote['agent']:<13} "
              f"{vote['reason']}")
    if result["futures"]:
        print("\n  FUTURES THE CROWD SAW:")
        for future in result["futures"]:
            mark = "+" if future["resolves"] == "YES" else "-"
            print(f"   {mark} {future['story']}  ({future['agent']})")
    print()


def _print_worlds(result: dict) -> None:
    """Show a --deep run: every distinct world found, YES ones first,
    ranked by how often each one came up, then the plain odds."""
    print(f"\n  THE MAP OF WORLDS: {len(result['worlds'])} distinct ways seen "
          f"across {result['raw_futures']} imagined futures\n")
    yes_worlds = [w for w in result["worlds"] if w["resolves"] == "YES"]
    no_worlds = [w for w in result["worlds"] if w["resolves"] == "NO"]
    for world in yes_worlds:
        print(f"   + x{world['count']}  {world['story']}")
    for world in no_worlds:
        print(f"   - x{world['count']}  {world['story']}")
    print(f"\n  THE CROWD SAYS: {result['probability']:.0%} chance of YES")
    print(f"  ({result['rounds']} rounds, "
          f"{result['skipped']} unusable answers skipped)\n")


def _print_paths(result: dict) -> None:
    """Show a --path run: every distinct path found to the target, rated
    against base rates, then the honest neutral odds. Zero paths is
    printed plainly too: finding nothing believable is a real answer.
    """
    print(f"\n  PATHS TO {result['target']}: {len(result['paths'])} distinct ways found\n")
    if not result["paths"]:
        print("  (nothing believable found. that is a real answer, not a bug.)\n")
    for path in result["paths"]:
        print(f"   [{path['rating']}] {path['story']}  (x{path['count']})")
        print(f"      gates: {'; '.join(path['gates'])}")
    print(f"\n  THE CROWD SAYS: {result['probability']:.0%} chance of YES "
          "(from a neutral split; the path search never changes the odds)")
    print(f"  ({result['rounds']} rounds, "
          f"{result['skipped']} unusable answers skipped)\n")


def _print_carlo(result: dict) -> None:
    """Show a --carlo run: the share of a million (or however many)
    rolled futures that landed YES, the crowd's 80 percent belief band,
    and how many minds actually went into the mixture."""
    draws = result["draws"]
    # Only says "ONE MILLION" when it truly is a million, so this line
    # never claims something that isn't so if CARLO_DRAWS is ever changed.
    label = "ONE MILLION" if draws == 1_000_000 else f"{draws:,}"
    print(f"\n  {label} FUTURES ROLLED: {result['probability']:.1%} ended YES")
    print(f"  the crowd's belief band: {result['p10']:.0%} to {result['p90']:.0%} "
          "(80% of futures fell here)")
    print(f"  ({result['agents_used']} minds elicited, {result['skipped']} junk "
          f"skipped, seed {result['seed']})\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask the AI crowd for the odds of anything.")
    parser.add_argument("question", help="a yes/no question about the future")
    parser.add_argument("--whatif", metavar="FACT",
                        help="re-run the crowd with this fact forced true")
    parser.add_argument("--simulate", action="store_true",
                        help="(the default) every run imagines K futures")
    parser.add_argument("--vote", action="store_true",
                        help="cheaper mode: one probability per run, no futures")
    parser.add_argument("--deep", action="store_true",
                        help="keep splitting into futures until two rounds in a "
                             "row find nothing new, then show the map of worlds")
    parser.add_argument("--path", choices=["YES", "NO"], default=None,
                        help="hunt for distinct paths to YES or NO, each rated "
                             "against base rates; the odds still come from a "
                             "plain neutral split, never from this search")
    parser.add_argument("--agents", type=int, default=None,
                        help=f"crowd size (default {config.ENGINE_N_AGENTS}). "
                             "Large values, hundreds or thousands with "
                             "--crowd pool, cost real money per agent: "
                             "ENGINE_BUDGET_USD still hard-stops spend either way")
    parser.add_argument("--model", default=None,
                        help="haiku, sonnet, opus, fable, or any full model ID "
                             f"(default {config.ENGINE_MODEL})")
    parser.add_argument("--crowd", choices=["methods", "ensemble", "pool"], default=None,
                        help="'methods' (one model, six ways of reasoning), "
                             "'ensemble' (different models on different "
                             "evidence slices, config.ENSEMBLE_SEATS), or "
                             "'pool' (big diversity crowd, each agent blends "
                             "an independent method + emotional temperament + "
                             "lens, engine/pool.py, sized by --agents) "
                             f"(default {config.CROWD_MODE})")
    parser.add_argument("--no-news", action="store_true",
                        help="skip the headline lookup")
    parser.add_argument("--carlo", action="store_true",
                        help="roll config.CARLO_DRAWS (default 1,000,000) simulated "
                             "futures through the crowd's elicited belief bands, "
                             "and report the share that land YES plus the crowd's "
                             "80%% belief band; composes with --crowd and --agents")
    args = parser.parse_args()

    print(f'\nQ: {args.question}')

    if args.deep:
        card = {"ticker": "ASK", "question": args.question, "mid": None}
        headlines = news.research(args.question) if not args.no_news else []
        result = explore_worlds(card, headlines, _resolve_ask(model=args.model))
        _print_worlds(result)
        print(f"  total engine spend so far: ${llm.spent_usd():.2f} "
              f"(cap ${config.ENGINE_BUDGET_USD:.2f})\n")
        return

    if args.path:
        card = {"ticker": "ASK", "question": args.question, "mid": None}
        headlines = news.research(args.question) if not args.no_news else []
        result = find_paths(card, headlines, _resolve_ask(model=args.model),
                            target=args.path)
        _print_paths(result)
        print(f"  total engine spend so far: ${llm.spent_usd():.2f} "
              f"(cap ${config.ENGINE_BUDGET_USD:.2f})\n")
        return

    if args.carlo:
        card = {"ticker": "ASK", "question": args.question, "mid": None}
        headlines = news.research(args.question) if not args.no_news else []
        crowd = build_crowd_for(args.agents, args.crowd)
        result = run_carlo(card, headlines, crowd,
                           ask_fn=_resolve_ask(model=args.model))
        _print_carlo(result)
        print(f"  total engine spend so far: ${llm.spent_usd():.2f} "
              f"(cap ${config.ENGINE_BUDGET_USD:.2f})\n")
        return

    mode = "vote" if args.vote else "simulate"
    result = ask_question(args.question, whatif=args.whatif, mode=mode,
                          n_agents=args.agents, with_news=not args.no_news,
                          model=args.model, crowd_mode=args.crowd)

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
