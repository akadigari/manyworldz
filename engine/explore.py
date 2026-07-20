"""Deep split: instead of a fixed number of imagined futures, keep
exploring until no genuinely new way of it happening shows up.

Round 1 reuses the normal simulate machinery (the same crowd, the same
agent_futures calls) to get a first batch of stories. Every round after
that is one more single "imagine more" call, shown the map of distinct
worlds found so far and asked for stories that do not match anything on
it. After every round, one more call sorts that round's new stories into
the map: same underlying mechanism as something already there, or
genuinely new. We stop once a couple of rounds in a row turn up nothing
new, or once we hit a round cap, whichever comes first.

Right after round 1, one extra call plays devil's advocate: it is shown
the map and the current YES-share so far and asked to imagine ways that
exact consensus turns out wrong. This is the "saboteur": a forced
pre-mortem, so the crowd does not just keep restating its first take in
new words. Its stories go through the same classify-and-merge as
everything else, so a good pre-mortem story earns its own world instead
of getting waved through for free.

From round 2 on, every other round also forces one wildcard into the
imagine prompt: an instruction that at least one of the new futures must
involve something like a lawsuit, a leak, or a rival moving first. This
is novelty pressure: left alone, an LLM tends to reimagine the same
"soft landing" and "everything goes as planned" stories in new words.
The wildcard rotation is picked once per run from config.SEED, so it is
different from the plain list order but always the same for the same
seed.

The map (the list of distinct worlds) is a display layer. The
probability is always the plain census of every future ever seen,
duplicates included, same as engine/futures.py. Deduping never touches
the math.

Every function here takes ask_fn so tests can inject canned answers and
stay offline.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from engine.personas import build_crowd
from engine.swarm import extract_json, market_line, run_crowd

# Forced ingredients for the wildcard rounds (see explore_worlds). Each one
# is a short phrase dropped into the imagine prompt as "at least one of
# your futures must involve: <this>". The point is novelty pressure: these
# are the kinds of things a model tends to forget to imagine on its own.
WILDCARDS = [
    "the weather turns",
    "a lawsuit or ruling lands",
    "someone's health changes",
    "something leaks early",
    "a rival moves first",
    "a technical failure",
    "the money runs out",
    "the schedule slips",
    "public backlash",
    "an unexpected ally appears",
]

# Shown to one agent in rounds 2+: here is what has already been found,
# go imagine k more ways that are actually different.
_EXPLORE_PROMPT = """You are {name}, a {archetype}: {style}.

A prediction market asks: "{question}"
{market_line}
Recent headlines: {headlines}

These are the distinct ways this could play out that have ALREADY been found:
{worlds}

Imagine {k} MORE ways this could play out that are genuinely DIFFERENT from
every one listed above, not just the same idea in different words. Short,
concrete, one sentence each.{wildcard_line}
Reply with ONLY JSON like:
{{"futures": [{{"story": "one sentence", "resolves": "YES"}},
              {{"story": "another way it goes", "resolves": "NO"}}]}}
Give exactly {k} futures, each with "resolves" as "YES" or "NO"."""

# The saboteur: one pre-mortem call right after round 1. Shown the map and
# the crowd's current read, asked to imagine exactly how that consensus
# turns out wrong.
_SABOTEUR_PROMPT = """A prediction market asks: "{question}"

MAP SO FAR, the distinct ways this could play out:
{worlds}

Right now the crowd's read is about a {prob:.0%} chance of YES.

Assume that consensus turns out WRONG. Imagine {k} DIFFERENT futures
showing exactly how that happens: short, concrete, one sentence each. Do
not just restate the map above in different words; find the specific way
the crowd's current read breaks.
Reply with ONLY JSON like:
{{"futures": [{{"story": "one sentence", "resolves": "YES"}},
              {{"story": "another way it goes", "resolves": "NO"}}]}}
Give exactly {k} futures, each with "resolves" as "YES" or "NO"."""

# The dedupe pass: one call per round that sorts a batch of candidate
# stories into the growing map of distinct worlds.
_CLASSIFY_PROMPT = """A prediction market asks: "{question}"

We are sorting imagined futures into distinct WORLDS. Two stories are the
SAME world if they describe the same underlying mechanism, even if worded
differently or with different specific numbers or dates. Different
mechanisms are different worlds, even if they land on the same YES or NO.

WORLDS FOUND SO FAR (numbered):
{worlds}

NEW CANDIDATE STORIES TO SORT (numbered):
{candidates}

For each candidate, in order, answer one of three ways:
- "wN" if it is the SAME mechanism as world N above, e.g. "w2"
- "cN" if it is the SAME mechanism as an EARLIER candidate N in this list,
  e.g. "c1"
- "new" if it is a genuinely different mechanism from everything above and
  everything earlier in this list

Reply with ONLY JSON like:
{{"classifications": ["new", "w2", "c1"]}}
Give exactly {n} answers, one per candidate, in order."""


def _worlds_bullets(worlds: list[dict]) -> str:
    """The map so far, as plain bullets for a prompt."""
    if not worlds:
        return "(nothing found yet)"
    return "\n".join(f"- {w['story']}" for w in worlds)


def _wildcard_order(seed: int) -> list[str]:
    """Shuffle the wildcard list once, seeded so the same seed always
    gives the same rotation. This just decides the order wildcard rounds
    cycle through; it wraps back to the start once every wildcard has
    been used. Uses random.Random(seed), never the bare random module, so
    a run is always exactly repeatable.
    """
    rng = random.Random(seed)
    order = list(WILDCARDS)
    rng.shuffle(order)
    return order


def _parse_futures(parsed: dict | None, k: int) -> list[dict] | None:
    """Shared parsing for any "imagine k futures" style call (round-2+
    imagine, and the saboteur): pull out story + resolves, keep only
    usable ones, and refuse to guess if the model gave back less than
    half of what was asked, in a usable shape.
    """
    if not parsed:
        return None
    futures = []
    for future in parsed.get("futures", []):
        if not isinstance(future, dict):
            continue
        verdict = str(future.get("resolves", "")).strip().upper()
        story = str(future.get("story", "")).strip()
        if verdict in ("YES", "NO") and story:
            futures.append({"story": story[:200], "resolves": verdict})
    if len(futures) < max(k // 2, 1):
        return None      # the model did not really play along: skip, do not guess
    return futures


def _imagine_more(agent: dict, card: dict, headlines: list[str],
                  worlds: list[dict], k: int, ask_fn,
                  wildcard: str | None = None) -> list[dict] | None:
    """Ask one agent to imagine k more futures that do not match any
    world already on the map. If `wildcard` is set, one extra sentence
    forces at least one of the new futures to involve it.
    """
    wildcard_line = ""
    if wildcard:
        wildcard_line = f"\n\nAt least one of your futures must involve: {wildcard}."
    prompt = _EXPLORE_PROMPT.format(
        name=agent["name"], archetype=agent["archetype"], style=agent["style"],
        question=card["question"], market_line=market_line(card),
        headlines="; ".join(headlines) if headlines else "(none found)",
        worlds=_worlds_bullets(worlds), k=k, wildcard_line=wildcard_line)
    parsed = extract_json(ask_fn(prompt, max_tokens=200 + 80 * k))
    return _parse_futures(parsed, k)


def _saboteur_round(card: dict, worlds: list[dict], probability: float,
                    k: int, ask_fn) -> list[dict] | None:
    """One pre-mortem call: show the map and the current YES-share, ask
    for k futures showing exactly how that consensus turns out wrong.
    """
    prompt = _SABOTEUR_PROMPT.format(
        question=card["question"], worlds=_worlds_bullets(worlds),
        prob=probability, k=k)
    parsed = extract_json(ask_fn(prompt, max_tokens=200 + 80 * k))
    return _parse_futures(parsed, k)


def _classify_and_merge(card: dict, worlds: list[dict], candidates: list[dict],
                        ask_fn) -> tuple[int, bool]:
    """One ask_fn call: sort this round's candidate stories into the
    growing map of distinct worlds.

    Mutates `worlds` in place, appending any genuinely new ones. Returns
    (how many new worlds got added, whether the classify call itself came
    back unusable). A candidate whose classification we cannot make sense
    of is just dropped: it never creates a new world and never bumps an
    existing count, since junk answers should never inflate the map.
    """
    worlds_before = len(worlds)   # only these can be referenced as "wN"; this
                                  # round's own new worlds are not numbered yet
    worlds_block = "\n".join(
        f"{i + 1}. {w['story']}" for i, w in enumerate(worlds)) or \
        "(none yet, this is the first round)"
    cand_block = "\n".join(
        f"{i + 1}. {c['story']}" for i, c in enumerate(candidates))
    prompt = _CLASSIFY_PROMPT.format(
        question=card["question"], worlds=worlds_block, candidates=cand_block,
        n=len(candidates))
    parsed = extract_json(ask_fn(prompt, max_tokens=100 + 20 * len(candidates)))
    labels = parsed.get("classifications") if isinstance(parsed, dict) else None
    if not isinstance(labels, list):
        return 0, True    # the whole call was junk: nothing merges, map stays put

    landed_in: list[dict | None] = [None] * len(candidates)   # candidate -> its world
    added = 0
    for i, candidate in enumerate(candidates):
        label = str(labels[i]).strip().lower() if i < len(labels) else ""
        if label == "new":
            world = {"story": candidate["story"], "resolves": candidate["resolves"],
                     "count": 1}
            worlds.append(world)
            landed_in[i] = world
            added += 1
        elif len(label) > 1 and label[0] == "w" and label[1:].isdigit():
            idx = int(label[1:]) - 1
            if 0 <= idx < worlds_before:
                worlds[idx]["count"] += 1
                landed_in[i] = worlds[idx]
            # else: points at a world that does not exist. junk, drop it.
        elif len(label) > 1 and label[0] == "c" and label[1:].isdigit():
            ref = int(label[1:]) - 1
            if 0 <= ref < i and landed_in[ref] is not None:
                landed_in[ref]["count"] += 1
                landed_in[i] = landed_in[ref]
            # else: forward reference or points at a dropped candidate, junk, drop it.
        # anything else is unparseable and gets dropped the same way: never guessed at.
    return added, False


def explore_worlds(card: dict, headlines: list[str], ask_fn,
                   k_per_round: int | None = None,
                   max_rounds: int | None = None,
                   dry_rounds: int | None = None) -> dict:
    """Keep splitting into imagined futures until the map of distinct
    worlds stops growing.

    Round 1 reuses the normal simulate machinery: config.ENGINE_N_AGENTS
    agents each imagine a batch of futures, exactly like a plain
    --simulate run. Right after round 1, one extra "saboteur" call is
    shown the map and the current YES-share and asked to imagine ways
    that exact consensus turns out wrong: its stories go through the same
    classify-and-merge as everything else. Every round after that (round
    2 onward) is one more single imagine call that is shown the current
    map and asked for stories that genuinely do not match it yet; every
    OTHER one of those rounds also forces in one wildcard, rotating
    through WILDCARDS deterministically by config.SEED. After every
    round (and after the saboteur call), one more ask_fn call sorts that
    round's new stories into the map.

    We stop once `dry_rounds` rounds in a row add nothing new to the map,
    or once `max_rounds` is hit, whichever comes first. `max_rounds` and
    `dry_rounds` default to config.DEEP_MAX_ROUNDS and
    config.DEEP_DRY_ROUNDS when left as None, read fresh at call time
    (not baked in at import time) so overriding config for a test or a
    run actually takes effect.

    A budget RuntimeError from engine/llm is not caught here: it just
    bubbles up, same as everywhere else in the engine.

    Returns the distinct worlds (sorted by how often they came up), the
    plain census probability over every raw future ever seen (deduping
    never touches this number), how many rounds ran, how many raw
    futures were seen, and how many sub-calls came back unusable.
    """
    if max_rounds is None:
        max_rounds = config.DEEP_MAX_ROUNDS
    if dry_rounds is None:
        dry_rounds = config.DEEP_DRY_ROUNDS
    k = k_per_round if k_per_round is not None else config.SIM_ROLLOUTS_K
    crowd = build_crowd(config.ENGINE_N_AGENTS, config.SEED)
    wildcard_order = _wildcard_order(config.SEED)
    next_wildcard = 0

    worlds: list[dict] = []
    raw_futures: list[dict] = []
    skipped = 0
    dry_streak = 0
    rounds_run = 0

    for round_num in range(1, max_rounds + 1):
        rounds_run = round_num
        if round_num == 1:
            result = run_crowd(card, headlines, crowd, mode="simulate", k=k,
                               deliberation=False, ask_fn=ask_fn)
            candidates = result["futures"]
            skipped += result["skipped"]
        else:
            # Rotate through the crowd's personas round to round, for variety.
            agent = crowd[(round_num - 2) % len(crowd)]
            wildcard = None
            if round_num % 2 == 0:   # every OTHER round from round 2 on
                wildcard = wildcard_order[next_wildcard % len(wildcard_order)]
                next_wildcard += 1
            imagined = _imagine_more(agent, card, headlines, worlds, k, ask_fn,
                                     wildcard=wildcard)
            if imagined is None:
                candidates = []
                skipped += 1
            else:
                candidates = imagined

        raw_futures.extend(candidates)

        added = 0
        if candidates:
            added, classify_failed = _classify_and_merge(card, worlds, candidates, ask_fn)
            if classify_failed:
                skipped += 1

        if round_num == 1:
            # The saboteur: one pre-mortem call, right after round 1, and
            # only once no matter how many rounds run in total.
            yes_so_far = sum(1 for f in raw_futures if f["resolves"] == "YES")
            prob_so_far = min(max(yes_so_far / len(raw_futures), 0.01), 0.99) \
                if raw_futures else 0.5
            sab_candidates = _saboteur_round(card, worlds, prob_so_far, k, ask_fn)
            if sab_candidates is None:
                skipped += 1
            else:
                raw_futures.extend(sab_candidates)
                sab_added, sab_failed = _classify_and_merge(
                    card, worlds, sab_candidates, ask_fn)
                added += sab_added
                if sab_failed:
                    skipped += 1

        dry_streak = 0 if added else dry_streak + 1
        if dry_streak >= dry_rounds:
            break

    yes = sum(1 for f in raw_futures if f["resolves"] == "YES")
    if raw_futures:
        # Same math philosophy as engine/futures.py: the plain YES share of
        # everything imagined, clamped so nobody claims total certainty.
        probability = min(max(yes / len(raw_futures), 0.01), 0.99)
    else:
        probability = 0.5   # nothing usable ever came back: stay at total uncertainty

    worlds_sorted = sorted(worlds, key=lambda w: -w["count"])
    return {"worlds": worlds_sorted, "probability": probability,
           "rounds": rounds_run, "raw_futures": len(raw_futures),
           "skipped": skipped}
