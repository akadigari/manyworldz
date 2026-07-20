"""The crowd roster: which analytical method each run uses.

Six ways of reasoning about an event, cycled across however many agents
config asks for. No fictional identity, no cast of characters: just a
method label and one line of instruction for how to use it.
"""
from __future__ import annotations

# Each entry is (short label, one-line instruction for how to reason this
# way). This is what used to be six fictional personas; the six ways of
# thinking are the same, they just aren't wearing a costume anymore.
METHODS: list[tuple[str, str]] = [
    ("base rates", "start from how often things like this actually happen, then adjust"),
    ("fresh news", "weigh the newest information hardest, but say when it is thin"),
    ("market logic", "think like someone setting a fair price others would take"),
    ("smart money", "focus on where informed people are putting their weight"),
    ("insider logic", "focus on who actually decides this outcome and what they want"),
    ("skeptic", "hunt for reasons the consensus is wrong"),
]


def build_methods(n: int, seed: int | None = None) -> list[dict]:
    """Build a list of n agents, each assigned one analytical method.

    The six methods repeat in order until there are n agents. This is
    always deterministic: no randomness and no names to pick, so the same
    n always builds the exact same list of methods. `seed` is accepted and
    ignored; it only exists so call sites written for the old
    build_crowd(n, seed) signature don't all need to change.
    """
    methods = []
    for i in range(n):
        label, instruction = METHODS[i % len(METHODS)]
        methods.append({"label": label, "instruction": instruction})
    return methods
