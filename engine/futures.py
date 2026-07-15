"""Simulate mode: an agent doesn't just vote — it imagines the event
playing out K times, and its probability is the share of its own futures
where the answer is YES. The stories feed the what-if view and the
dashboard's futures tree later.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import llm
from engine.swarm import extract_json

_SIM_PROMPT = """You are {name}, a {archetype} — {style}.

A prediction market asks: "{question}"
The market price right now says YES has about a {mid}% chance.
Recent headlines: {headlines}

Imagine {k} DIFFERENT ways this could actually play out — short, concrete,
one sentence each. Make them genuinely different, not five copies.
Reply with ONLY JSON like:
{{"futures": [{{"story": "one sentence", "resolves": "YES"}},
              {{"story": "another way it goes", "resolves": "NO"}}],
  "reason": "one sentence on your overall read"}}
Give exactly {k} futures, each with "resolves" as "YES" or "NO"."""


def agent_futures(agent: dict, card: dict, headlines: list[str], k: int,
                  ask_fn=llm.ask) -> dict | None:
    """Ask one agent to imagine k different ways a market could play out.

    Each imagined "future" is a short story plus a YES/NO resolution. The
    agent's probability is just the share of its own stories that ended
    YES — if it imagined 5 futures and 3 resolved YES, that's 0.6. Returns
    None if the model didn't actually give at least half of the requested
    futures in a usable shape (a sign it didn't really play along).
    """
    prompt = _SIM_PROMPT.format(
        name=agent["name"], archetype=agent["archetype"], style=agent["style"],
        question=card["question"], mid=card["mid"], k=k,
        headlines="; ".join(headlines) if headlines else "(none found)")
    parsed = extract_json(ask_fn(prompt, max_tokens=200 + 80 * k))
    if not parsed:
        return None

    futures = []
    for future in parsed.get("futures", []):
        if not isinstance(future, dict):
            continue
        verdict = str(future.get("resolves", "")).strip().upper()
        story = str(future.get("story", "")).strip()
        if verdict in ("YES", "NO") and story:
            futures.append({"story": story[:200], "resolves": verdict,
                            "agent": agent["name"]})
    if len(futures) < max(k // 2, 2):
        return None      # the model didn't really play along — skip, don't guess

    yes = sum(1 for future in futures if future["resolves"] == "YES")
    prob = min(max(yes / len(futures), 0.01), 0.99)
    return {"probability": prob,
            "reason": str(parsed.get("reason", ""))[:200],
            "futures": futures}
