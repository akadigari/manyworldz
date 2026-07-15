"""The crowd roster: who is voting.

Six ways of thinking about an event, cycled across however many agents
config asks for. Names are seeded so any crowd can be rebuilt exactly.
"""
from __future__ import annotations

import random

ARCHETYPES: list[tuple[str, str]] = [
    ("stats nerd", "trusts base rates and numbers, distrusts stories"),
    ("narrative fan", "feels momentum and hype, follows the story"),
    ("sharp-money tracker", "cares only where informed money is moving"),
    ("oddsmaker", "tries to set a fair line others would bet into"),
    ("insider brain", "obsesses over who actually decides the outcome"),
    ("contrarian", "hunts for reasons the crowd is wrong"),
]

_FIRST = ["Ava", "Ben", "Cleo", "Dev", "Ember", "Finn", "Gia", "Hugo",
          "Iris", "Jax", "Kai", "Luna", "Mo", "Nia", "Oz", "Pia"]


def build_crowd(n: int, seed: int) -> list[dict]:
    """n agents cycling the six archetypes, with seeded unique names."""
    rng = random.Random(seed)
    names = rng.sample(_FIRST, k=min(n, len(_FIRST)))
    while len(names) < n:                      # crowds bigger than the name pool
        names.append(f"{rng.choice(_FIRST)}-{len(names)}")
    crowd = []
    for i in range(n):
        archetype, style = ARCHETYPES[i % len(ARCHETYPES)]
        crowd.append({"name": names[i], "archetype": archetype, "style": style})
    return crowd
