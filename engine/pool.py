"""The diversity pool: a crowd where no two agents think the same way.

engine/methods.py gives one model six ways to reason. engine/ensemble.py
gives different models different slices of evidence. This file is a third
kind of diversity: it composes each agent from three independent pools at
once, a reasoning METHOD, an emotional/risk TEMPERAMENT, and a LENS (the
one force the agent is told to weigh heaviest). Mixing three independent
pools means a crowd of hundreds, even thousands, can all get a genuinely
different blend instead of repeating the same six voices over and over.

The temperament pool is the "emotion" dimension the owner asked for. It
never invents facts or changes what evidence an agent can see (that is
still engine/swarm.py's evidence_block); it only changes HOW an agent
weighs the evidence it already has, the same way a nervous analyst and a
bold one can look at the same numbers and land in different places.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from engine.methods import METHODS

# Ten emotional/risk stances. Each instruction is one plain line about HOW
# to weigh the evidence, never about inventing evidence that is not there.
TEMPERAMENTS: list[tuple[str, str]] = [
    ("optimist", "lean toward things working out, but say so plainly"),
    ("pessimist", "weight what could go wrong"),
    ("anxious", "overweight downside and tail risks"),
    ("bold", "give decisive numbers, avoid hedging to 50"),
    ("calm", "ignore drama, weight steady base rates"),
    ("contrarian", "distrust the obvious answer"),
    ("trusting", "take stated plans and official signals at face value"),
    ("cynical", "assume announced plans slip or fail"),
    ("hopeful", "imagine the surprising good outcome"),
    ("detached", "reason coldly from numbers only"),
]

# Eight lenses: the one force each agent is told to weigh heaviest.
LENSES: list[tuple[str, str]] = [
    ("legal/ruling", "weigh legal rulings, regulatory decisions, and official verdicts heaviest"),
    ("weather", "weigh weather and physical conditions heaviest"),
    ("money-flow", "weigh where the money is actually moving heaviest"),
    ("momentum", "weigh recent momentum and trend heaviest"),
    ("human behavior and who decides", "weigh who actually makes the call, and how people tend to act, heaviest"),
    ("timing and schedule", "weigh deadlines, calendars, and schedules heaviest"),
    ("a leak or early info", "weigh any leak or early information heaviest"),
    ("an outside shock", "weigh the chance of a sudden outside shock heaviest"),
]

# The ceiling: every distinct (method, temperament, lens) triple, before
# build_pool_crowd has to start repeating. With the pools above that is
# 6 * 10 * 8 = 480. Past this many agents, the crowd wraps around and
# starts reusing combinations from the top.
POOL_MAX_DISTINCT = len(METHODS) * len(TEMPERAMENTS) * len(LENSES)


def _compose(method: tuple[str, str], temperament: tuple[str, str],
            lens: tuple[str, str]) -> dict:
    """Build one agent dict from one method + one temperament + one lens."""
    method_label, method_instr = method
    temp_label, temp_instr = temperament
    lens_label, lens_instr = lens
    label = f"{method_label} / {temp_label} / {lens_label}"
    instruction = (
        f"Reason with the {method_label} method: {method_instr}. "
        f"Take a {temp_label} stance: {temp_instr}. "
        f"Weigh {lens_label} the most: {lens_instr}. "
        "Still start from the base rate: how often do things like this "
        "actually happen, historically. Anchor there first, then adjust "
        "for what you know. Give your own honest probability that this "
        "resolves YES."
    )
    return {"label": label, "instruction": instruction}


def build_pool_crowd(n: int, seed: int | None = None) -> list[dict]:
    """Build a crowd of n agents, each an independent blend of the three
    pools above.

    The first min(n, POOL_MAX_DISTINCT) agents are guaranteed to be all
    distinct (method, temperament, lens) combinations: no repeat until
    every combination has been used once. Past POOL_MAX_DISTINCT, the
    crowd wraps around and starts reusing combinations from the top of
    the same order, so any n, however large, always returns exactly n
    agents and never crashes.

    The walk is deterministic: build_pool_crowd(n, seed) always returns
    the exact same crowd for the same n and seed, using
    random.Random(seed), never the bare random module. `seed` defaults to
    config.SEED, the project-wide seed, so two calls with no seed given
    at all are identical too. A different seed still visits every
    combination exactly once before repeating; it just walks them in a
    different (still repeatable) order.
    """
    if n <= 0:
        return []
    rng = random.Random(seed if seed is not None else config.SEED)
    # A shuffled permutation of every flat index 0..POOL_MAX_DISTINCT-1
    # visits each (method, temperament, lens) triple exactly once before
    # any index repeats. Decoding a flat index back into three separate
    # indices is a plain mixed-radix split, one radix per pool.
    order = list(range(POOL_MAX_DISTINCT))
    rng.shuffle(order)

    n_methods = len(METHODS)
    n_temperaments = len(TEMPERAMENTS)
    agents = []
    for i in range(n):
        flat = order[i % POOL_MAX_DISTINCT]
        method_idx = flat % n_methods
        temp_idx = (flat // n_methods) % n_temperaments
        lens_idx = flat // (n_methods * n_temperaments)
        agents.append(_compose(METHODS[method_idx], TEMPERAMENTS[temp_idx],
                               LENSES[lens_idx]))
    return agents
