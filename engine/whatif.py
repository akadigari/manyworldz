"""The Eye: force a fact to be true and watch the crowd's number move.

We don't touch the agents — we edit the world they see. The injected fact
is prepended to the market question so every prompt (vote or simulate)
carries it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import llm
from engine.swarm import run_crowd


def run_whatif(card: dict, headlines: list[str], crowd: list[dict],
               inject: str, mode: str = "vote", k: int = 5,
               deliberation: bool = False, ask_fn=llm.ask) -> dict:
    """Run the crowd twice — once normally, once with a fact forced true —
    and measure how much that fact moves the crowd's number.

    `inject` is the fact to force, e.g. "the star player is out injured."
    We never change how the agents think; we only add a sentence to the
    market question itself, so every prompt sees the forced fact as given.
    Returns both runs plus the "shift": how many probability points the
    forced fact moved the consensus.
    """
    before = run_crowd(card, headlines, crowd, mode, k, deliberation, ask_fn)

    twisted = dict(card)
    twisted["question"] = (
        f"WHAT-IF (treat as definitely true: {inject}) — {card['question']}")
    after = run_crowd(twisted, headlines, crowd, mode, k, deliberation, ask_fn)

    return {"before": before, "after": after,
            "shift": round(after["probability"] - before["probability"], 4)}
