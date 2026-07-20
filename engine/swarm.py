"""The crowd itself: agents read a market, form a probability, and the
votes fold into one number plus a disagreement spread.

Every function takes ask_fn so tests can inject canned answers. Junk
answers are skipped and counted, never invented.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import llm

# The two prompt templates below are the actual text sent to the model.
# {curly braces} get filled in with real values before sending.
_VOTE_PROMPT = """Reason with one method only. Method: {label}. {instruction}.

The question: "{question}"
{market_line}
Recent headlines: {headlines}

Start from the base rate: how often do things like this usually happen,
historically? Anchor there first, THEN adjust for the evidence above.
Stick to your one method and give YOUR OWN probability that this
resolves YES. Do not just repeat the market price.
Reply with ONLY JSON like {{"probability": 0.42, "reason": "one short sentence"}}"""

_DELIB_PROMPT = """Reason with one method only. Method: {label}. {instruction}.
The question: "{question}". {market_line} Your current view: {own}.

Other agents said:
{others}

After hearing them, give your FINAL probability. It is fine to keep your
number if they did not change your mind.
Reply with ONLY JSON like {{"probability": 0.42, "reason": "one short sentence"}}"""


def market_line(card: dict) -> str:
    """One honest sentence about the market price, or the lack of one.

    Market cards carry a "mid" price in cents. Questions typed by a person
    (through ask.py) have no market, and lying to the crowd with a made-up
    price would bias every vote, so we tell them the truth instead.
    """
    mid = card.get("mid")
    if mid:
        return f"The market price right now says YES has about a {mid}% chance."
    return ("There is no market price for this question: you have nothing "
            "to anchor on except your own reasoning.")


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
    """Turn whatever the model sent back into a safe probability.

    Returns None if `raw` isn't a number at all. Otherwise clamps it into
    the range 0.01 to 0.99. We never let an agent claim total (100%) or
    zero (0%) certainty, since real events are basically never that sure.
    """
    try:
        p = float(raw)
    except (TypeError, ValueError):
        return None
    return min(max(p, 0.01), 0.99)  # never 0% or 100%, stay humble


def agent_vote(agent: dict, card: dict, headlines: list[str],
               ask_fn=llm.ask) -> dict | None:
    """Ask one agent for its probability on one market.

    Fills in the vote prompt with this agent's method and the market's
    question, sends it, and pulls out a probability + one-line reason.
    Returns None if the model's reply couldn't be understood (bad JSON,
    missing/invalid probability). A bad answer gets skipped, never
    guessed at.
    """
    prompt = _VOTE_PROMPT.format(
        label=agent["label"], instruction=agent["instruction"],
        question=card["question"], market_line=market_line(card),
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
    """Show one agent everyone else's votes, and ask for a final answer.

    This is the optional second round (only runs when DELIBERATION is
    on): the agent sees its own first guess plus a summary of what every
    other agent said, then gets to keep its number or change its mind.
    Returns None the same way agent_vote() does, if the reply is unusable.
    """
    prompt = _DELIB_PROMPT.format(
        label=agent["label"], instruction=agent["instruction"],
        question=card["question"], market_line=market_line(card),
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
    """Fold all the agents' probabilities into one number.

    With 5 or more votes we drop the single highest and lowest first, so
    one overexcited agent can't drag the answer around. We also return the
    "spread": how much the crowd disagreed. Small spread = confident crowd.
    """
    if not probs:
        return 0.5, 0.0
    use = sorted(probs)
    if len(use) >= 5:
        use = use[1:-1]
    mean = sum(use) / len(use)
    spread = statistics.pstdev(probs) if len(probs) > 1 else 0.0  # "spread": how far apart the votes are
    return round(mean, 4), round(spread, 4)


def run_crowd(card: dict, headlines: list[str], crowd: list[dict],
              mode: str = "vote", k: int = 5, deliberation: bool = False,
              ask_fn=llm.ask) -> dict:
    """Run every agent in the crowd on one market, then fold their answers
    into one consensus number.

    `mode` picks how each agent answers: "vote" (one probability) or
    "simulate" (K imagined futures, see engine/futures.py). If
    `deliberation` is on, agents get a second round where they see what
    everyone else said before giving a final answer. Returns the crowd's
    consensus probability and spread, every individual vote, any imagined
    futures, and how many agents gave an unusable answer.
    """
    # Imported here, not at the top of the file, so this file and
    # engine/futures.py (which imports something from this file) don't
    # end up needing each other before either one has finished loading.
    from engine import futures as _futures

    votes, all_futures, skipped = [], [], 0
    for agent in crowd:
        if mode == "simulate":
            result = _futures.agent_futures(agent, card, headlines, k, ask_fn)
        else:
            result = agent_vote(agent, card, headlines, ask_fn)
        if result is None:
            skipped += 1
            continue
        result["agent"] = agent["label"]
        votes.append(result)
        all_futures.extend(result.get("futures", []))

    if deliberation and len(votes) >= 2:
        # One line of summary text per vote, e.g. "- base rates: 0.62
        # (trusts the recent form numbers)." This is what gets shown to
        # each agent as "what everyone else said." Two agents can share
        # the same method label (a crowd bigger than six wraps around),
        # so "everyone but this one" is tracked by position, not by label
        # text, or a repeated method would wrongly hide its own sibling's
        # line too.
        digest = [f'- {v["agent"]}: {v["probability"]:.2f} ({v["reason"]})'
                  for v in votes]
        by_label = {a["label"]: a for a in crowd}
        revised = []
        for i, v in enumerate(votes):
            others = [d for j, d in enumerate(digest) if j != i]  # everyone but this one
            second = deliberate(by_label[v["agent"]], card, v, others, ask_fn)
            if second is not None:
                v = {**v, **second}
            revised.append(v)
        votes = revised

    prob, spread = consensus([v["probability"] for v in votes])
    return {"probability": prob, "spread": spread, "votes": votes,
            "futures": all_futures, "skipped": skipped}
