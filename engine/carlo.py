"""The monte carlo fusion layer: makes "a million simulated outcomes"
literally true.

engine/swarm.py's crowd already judges a question and folds every vote
into one number. This file adds a numeric layer under that judgment: the
minds do the judging, the numbers do the rolling. Three steps, run in
order by run_carlo():

1. Elicit: ask every agent for its probability AND its own honest 80
   percent band on that number (how sure it is about its own guess).
2. Roll: build a mixture of those elicited beliefs and roll CARLO_DRAWS
   simulated futures through it. Each draw picks one agent's belief,
   samples a probability from a triangular distribution shaped by that
   belief's own low/peak/high, then flips a weighted coin at that
   probability. Pure stdlib random.Random(config.SEED): fully
   deterministic, no new dependencies, and free (no API calls).
3. Report: the share of draws that landed YES, plus the 10th/50th/90th
   percentile of the sampled probabilities (the crowd's belief band).

The honest limit, stated here too since it matters: rolling a million
dice through the same beliefs cannot make the beliefs smarter. It makes
the summary of them exact and exposes the true uncertainty band. That is
the whole value of this file, no more.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from engine import llm
from engine.swarm import evidence_block, extract_json

_ELICIT_PROMPT = """Reason with one method only. Method: {label}. {instruction}.

The question: "{question}"
{evidence}

Start from the base rate: how often do things like this usually happen,
historically? Anchor there first, THEN adjust for the evidence above.
Stick to your one method and give YOUR OWN probability that this
resolves YES. Do not just repeat the market price.

Also give your own honest 80 percent band on that number: "low" and
"high" should be values you think there is an 80 percent chance your
true probability falls between. A wide band means you are unsure of your
own number; a narrow band means you are confident in it.

Reply with ONLY JSON like {{"probability": 0.42, "low": 0.30, "high": 0.55, "reason": "one short sentence"}}"""


def _clamp(raw) -> float | None:
    """Same rule as engine/swarm.py's _clean_prob: turn whatever the
    model sent back into a safe probability in [0.01, 0.99], or None if
    it isn't a number at all. Repeated here, not imported, because it is
    a one-line rule and engine/futures.py already keeps its own copy the
    same way instead of reaching into swarm.py's private helper.
    """
    try:
        p = float(raw)
    except (TypeError, ValueError):
        return None
    return min(max(p, 0.01), 0.99)


def elicit(agent: dict, card: dict, headlines: list[str],
          ask_fn=llm.ask) -> dict | None:
    """Ask one agent for its probability and its own honest uncertainty
    about that probability.

    Same shape as engine/swarm.py's agent_vote: same evidence slicing
    (a narrow-slice ensemble seat only sees its own piece of evidence),
    same per-seat model routing, same extract_json parsing. Returns None
    if the reply can't be turned into a usable belief, so a junk answer
    is skipped, never invented:

    - no JSON at all, or no usable "probability" -> skipped (None)
    - "low"/"high" missing or not numbers -> skipped (None): there is no
      band to salvage, and making one up would be inventing an opinion
      the agent never gave
    - "low" and "high" swapped, or the band doesn't actually contain the
      probability -> repaired: swap them back, and stretch the band just
      enough to hold the probability, then keep going
    """
    evidence = agent.get("evidence", "everything")
    prompt = _ELICIT_PROMPT.format(
        label=agent["label"], instruction=agent["instruction"],
        question=card["question"],
        evidence=evidence_block(card, headlines, evidence))
    parsed = extract_json(ask_fn(prompt, model=agent.get("model")))
    if not parsed:
        return None
    prob = _clamp(parsed.get("probability"))
    if prob is None:
        return None
    low = _clamp(parsed.get("low"))
    high = _clamp(parsed.get("high"))
    if low is None or high is None:
        return None
    if low > high:
        low, high = high, low          # repair: the model swapped them
    low = min(low, prob)               # repair: band must actually hold the probability
    high = max(high, prob)
    return {"agent": agent["label"], "probability": prob, "low": low,
            "high": high, "reason": str(parsed.get("reason", ""))[:200]}


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile of an already-sorted list. `q` is
    a fraction from 0 to 1 (0.10 for the 10th percentile, and so on).
    Spelled out by hand so this file needs no new dependency.
    """
    if not sorted_vals:
        return 0.5
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def roll(elicited: list[dict], draws: int, seed: int) -> dict:
    """Roll `draws` simulated futures through the crowd's elicited beliefs.

    For every draw: pick one elicited belief uniformly at random, sample
    a probability from a triangular distribution shaped by that belief's
    own (low, probability, high), clip it into [0.01, 0.99], then flip a
    coin weighted by that probability. The reported probability is just
    the share of draws that landed YES.

    Fully deterministic: the same elicited list, draws count, and seed
    always produce the exact same result, since this is pure
    random.Random(seed) stdlib math with no other source of randomness.

    An empty `elicited` list returns the same honest 0.5-with-no-band
    default the rest of the engine uses for "no usable signal": rolling
    dice through zero beliefs would be inventing an opinion no agent
    actually gave.
    """
    if not elicited:
        return {"probability": 0.5, "p10": 0.5, "p50": 0.5, "p90": 0.5, "draws": 0}
    rng = random.Random(seed)
    n = len(elicited)
    yes = 0
    sampled = []
    for _ in range(draws):
        belief = elicited[rng.randrange(n)]
        p = rng.triangular(belief["low"], belief["high"], belief["probability"])
        p = min(max(p, 0.01), 0.99)      # clip at both ends, same rule as everywhere else
        sampled.append(p)
        if rng.random() < p:
            yes += 1
    sampled.sort()
    return {"probability": round(yes / draws, 4),
            "p10": round(_percentile(sampled, 0.10), 4),
            "p50": round(_percentile(sampled, 0.50), 4),
            "p90": round(_percentile(sampled, 0.90), 4),
            "draws": draws}


def run_carlo(card: dict, headlines: list[str], crowd: list[dict],
              draws: int | None = None, ask_fn=llm.ask) -> dict:
    """The whole carlo layer in one call: elicit a belief from every agent
    in the crowd, then roll `draws` (default config.CARLO_DRAWS) simulated
    futures through the mixture of those beliefs.

    Returns the roll's probability/p10/p50/p90/draws, plus how many
    agents actually gave a usable belief, how many were skipped as junk,
    the seed used, and the individual elicited beliefs (so a caller can
    see exactly what each agent said, the same way run_crowd returns
    every vote alongside the folded consensus).
    """
    if draws is None:
        draws = config.CARLO_DRAWS
    elicited, skipped = [], 0
    for agent in crowd:
        belief = elicit(agent, card, headlines, ask_fn)
        if belief is None:
            skipped += 1
            continue
        elicited.append(belief)
    result = roll(elicited, draws, config.SEED)
    result["agents_used"] = len(elicited)
    result["skipped"] = skipped
    result["seed"] = config.SEED
    result["elicited"] = elicited
    return result
