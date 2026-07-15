"""The crowd itself: agents read a market, form a probability, and the
votes fold into one number plus a disagreement spread.

Every function takes ask_fn so tests can inject canned answers. Junk
answers are skipped and counted — never invented.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import llm

_VOTE_PROMPT = """You are {name}, a {archetype} — {style}.

A prediction market asks: "{question}"
The market price right now says YES has about a {mid}% chance.
Recent headlines: {headlines}

Think like your character and give YOUR OWN probability that this
resolves YES. Do not just repeat the market price.
Reply with ONLY JSON like {{"probability": 0.42, "reason": "one short sentence"}}"""

_DELIB_PROMPT = """You are {name}, a {archetype} — {style}.
Market: "{question}" (market price ~{mid}% YES). Your current view: {own}.

Other agents said:
{others}

After hearing them, give your FINAL probability. It is fine to keep your
number if they did not change your mind.
Reply with ONLY JSON like {{"probability": 0.42, "reason": "one short sentence"}}"""


def extract_json(text: str) -> dict | None:
    """Pull the first {...} out of a model answer. None if there isn't one."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return None


def _clean_prob(raw) -> float | None:
    try:
        p = float(raw)
    except (TypeError, ValueError):
        return None
    return min(max(p, 0.01), 0.99)  # never 0% or 100% — stay humble


def agent_vote(agent: dict, card: dict, headlines: list[str],
               ask_fn=llm.ask) -> dict | None:
    prompt = _VOTE_PROMPT.format(
        name=agent["name"], archetype=agent["archetype"], style=agent["style"],
        question=card["question"], mid=card["mid"],
        headlines="; ".join(headlines) if headlines else "(none found)")
    parsed = extract_json(ask_fn(prompt))
    if not parsed:
        return None
    prob = _clean_prob(parsed.get("probability"))
    if prob is None:
        return None
    return {"probability": prob, "reason": str(parsed.get("reason", ""))[:200]}


def deliberate(agent: dict, card: dict, own: dict, others: list[str],
               ask_fn=llm.ask) -> dict | None:
    prompt = _DELIB_PROMPT.format(
        name=agent["name"], archetype=agent["archetype"], style=agent["style"],
        question=card["question"], mid=card["mid"],
        own=f'{own["probability"]:.2f} ("{own["reason"]}")',
        others="\n".join(others))
    parsed = extract_json(ask_fn(prompt))
    if not parsed:
        return None
    prob = _clean_prob(parsed.get("probability"))
    if prob is None:
        return None
    return {"probability": prob, "reason": str(parsed.get("reason", ""))[:200]}


def consensus(probs: list[float]) -> tuple[float, float]:
    """Trimmed mean + spread. With 5+ votes, drop one high and one low."""
    if not probs:
        return 0.5, 0.0
    use = sorted(probs)
    if len(use) >= 5:
        use = use[1:-1]
    mean = sum(use) / len(use)
    spread = statistics.pstdev(probs) if len(probs) > 1 else 0.0
    return round(mean, 4), round(spread, 4)


def run_crowd(card: dict, headlines: list[str], crowd: list[dict],
              mode: str = "vote", k: int = 5, deliberation: bool = False,
              ask_fn=llm.ask) -> dict:
    """Run the whole crowd on one market and fold it into a consensus."""
    from engine import futures as _futures  # Task 6; local import avoids cycles

    votes, all_futures, skipped = [], [], 0
    for agent in crowd:
        if mode == "simulate":
            result = _futures.agent_futures(agent, card, headlines, k, ask_fn)
        else:
            result = agent_vote(agent, card, headlines, ask_fn)
        if result is None:
            skipped += 1
            continue
        result["agent"] = agent["name"]
        result["archetype"] = agent["archetype"]
        votes.append(result)
        all_futures.extend(result.get("futures", []))

    if deliberation and len(votes) >= 2:
        digest = [f'- {v["agent"]} ({v["archetype"]}): '
                  f'{v["probability"]:.2f} — {v["reason"]}' for v in votes]
        by_name = {a["name"]: a for a in crowd}
        revised = []
        for v in votes:
            others = [d for d in digest if not d.startswith(f'- {v["agent"]} ')]
            second = deliberate(by_name[v["agent"]], card, v, others, ask_fn)
            if second is not None:
                v = {**v, **second}
            revised.append(v)
        votes = revised

    prob, spread = consensus([v["probability"] for v in votes])
    return {"probability": prob, "spread": spread, "votes": votes,
            "futures": all_futures, "skipped": skipped}
