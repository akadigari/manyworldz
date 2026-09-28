"""Answer a Prophet Arena event with the same engine the Metaculus bot uses.

Prophet Arena (prophetarena.co) is an LLM forecasting leaderboard. Its
Python SDK (`prophet-arena`, imported as `arena`) hands an agent one
`Event` per open forecast window and wants back one probability per
outcome. This file is the translation layer between that shape and the
engine's question cards; prophet_agent.py is the thin SDK entrypoint.

Verified against prophet-arena 0.5.2 (arena/types.py, 2026-09-27):

- Event fields used here: slug, title, criteria, close_at,
  outcomes[name, criteria], crowd {outcome: prob}, context
  [title, url, summary], mutually_exclusive.
- Probabilities are per-outcome MARGINALS, each in [0, 1], every
  outcome exactly once, and the platform NEVER renormalizes them.

How an event is answered, reusing tournament.py's own answer paths (no
prompts are duplicated here):

1. Two outcomes (Yes/No, or two named sides like a game): one binary
   question through tournament._answer_one (the full fallback ladder
   plus the cheap/strong escalation policy). The first outcome gets p,
   the second 1 - p.
2. Three or more outcomes flagged mutually exclusive: one direct
   multiple choice call, tournament._answer_mc, normalized.
3. Three or more outcomes NOT flagged exclusive: one binary question per
   outcome ("will this outcome resolve YES?"), capped at
   MAX_CROWD_OUTCOMES crowd runs per event to bound cost.

Every submitted number is clipped exactly like the Metaculus bot's
(config.TOURNAMENT_CLIP via tournament._tournament_clip). When the
engine has nothing usable for an outcome (ladder fallback, no quorum,
past the outcome cap, or the ENGINE_BUDGET_USD cap), the fallback is the
crowd's own price for that outcome when the event carries one, else a
uniform guess, and the rationale says so. The budget cap never raises
out of here: once hit, the remaining outcomes take the fallback without
another model call, so a spent budget still costs nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tournament
from engine.ensemble import build_crowd_for

# Crowd runs per non-exclusive multi-outcome event. Each run is a full
# crowd (config.ENGINE_N_AGENTS calls), so a 20-candidate event would
# otherwise cost 20 crowds. Outcomes past the cap take the fallback.
MAX_CROWD_OUTCOMES = 6
# Frozen snapshot sources plus live research, trimmed per line.
MAX_HEADLINES = 10
_HEADLINE_CHARS = 300
_YES_NO = ("yes", "no")


def _close_time(event) -> str | None:
    close = getattr(event, "close_at", None)
    if close is None:
        return None
    return close.isoformat() if hasattr(close, "isoformat") else str(close)


def _names(event) -> list[str]:
    return [o.name for o in event.outcomes]


def _is_yes_no(event) -> bool:
    return sorted(n.casefold() for n in _names(event)) == sorted(_YES_NO)


def binary_card(event, outcome=None) -> dict:
    """The engine's binary card for the whole event, or for one outcome.

    Per-outcome cards name the outcome in the question and prefer that
    outcome's own resolution rules over the event's shared text."""
    if outcome is None:
        return {"qid": event.slug, "question": event.title,
                "criteria": getattr(event, "criteria", None) or "",
                "close_time": _close_time(event), "qtype": "binary"}
    criteria = getattr(outcome, "criteria", None) or getattr(event, "criteria", None) or ""
    question = f"{event.title} Will the outcome '{outcome.name}' resolve YES?"
    return {"qid": f"{event.slug}:{outcome.name}", "question": question,
            "criteria": criteria, "close_time": _close_time(event),
            "qtype": "binary"}


def mc_card(event) -> dict:
    """The engine's multiple choice card: options in event order."""
    return {"qid": event.slug, "question": event.title,
            "criteria": getattr(event, "criteria", None) or "",
            "close_time": _close_time(event), "qtype": "multiple_choice",
            "options": _names(event)}


def _crowd_price(event, name: str) -> float | None:
    """The event's crowd probability for one outcome, matched the way the
    platform matches names (case-insensitively), or None."""
    crowd = getattr(event, "crowd", None) or {}
    wanted = name.casefold()
    for key, value in crowd.items():
        if str(key).casefold() == wanted:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _fallback(event, name: str) -> float:
    price = _crowd_price(event, name)
    if price is not None:
        return price
    n = len(event.outcomes)
    return 1.0 / n if getattr(event, "mutually_exclusive", False) and n else 0.5


def _headlines(event, research_fn) -> list[str]:
    """Frozen snapshot sources first, then live research. A research
    failure is never fatal: the snapshot alone is still evidence."""
    lines = []
    for source in getattr(event, "context", None) or []:
        parts = [getattr(source, "title", None), getattr(source, "summary", None)]
        text = ": ".join(p for p in parts if p)
        if text:
            lines.append(text[:_HEADLINE_CHARS])
    try:
        lines.extend(h[:_HEADLINE_CHARS] for h in research_fn(event.title) or [])
    except Exception as exc:
        print(f"  research failed on {event.slug} ({exc}); snapshot only")
    return lines[:MAX_HEADLINES]


def _binary_prob(card, headlines, crowd, ask_fn, conserving) -> tuple[float | None, str]:
    """One engine answer: (probability or None when unusable, source)."""
    result = tournament._answer_one(card, headlines, crowd, ask_fn, conserving=conserving)
    prob, source = result.get("prob"), result.get("source") or "no-quorum"
    if prob is None or source == "fallback":
        return None, source
    return float(prob), source


def _answer_pair(event, headlines, crowd, ask_fn, conserving) -> tuple[dict, list[str]]:
    first, second = event.outcomes
    card = binary_card(event) if _is_yes_no(event) else binary_card(event, first)
    yes_first = not _is_yes_no(event) or first.name.casefold() == "yes"
    prob, source = _binary_prob(card, headlines, crowd, ask_fn, conserving)
    if prob is None:
        fb = _crowd_price(event, first.name)
        if fb is None:
            other = _crowd_price(event, second.name)
            fb = 1 - other if other is not None else 0.5
        return {first.name: fb, second.name: 1 - fb}, [f"fallback ({source})"]
    p_first = prob if yes_first else 1 - prob
    return {first.name: p_first, second.name: 1 - p_first}, [source]


def _answer_mc(event, headlines, ask_fn) -> tuple[dict, list[str]]:
    card = mc_card(event)
    try:
        outcome = tournament._with_deadline(
            lambda: tournament._answer_mc(card, headlines, ask_fn),
            tournament.config.QUESTION_DEADLINE_S)
    except Exception as exc:
        if tournament._is_budget_error(exc):
            raise
        outcome = {"probs": None, "source": "fallback"}
    probs = outcome.get("probs")
    if outcome.get("source") == "fallback" or not probs:
        return ({n: _fallback(event, n) for n in _names(event)},
                ["fallback (mc unusable)"])
    total = sum(probs.values())
    return {n: probs[n] / total for n in _names(event)}, [outcome["source"]]


def _answer_each(event, headlines, crowd, ask_fn, conserving) -> tuple[dict, list[str]]:
    probs, notes, budget_hit = {}, [], False
    for i, outcome in enumerate(event.outcomes):
        if budget_hit or i >= MAX_CROWD_OUTCOMES:
            probs[outcome.name] = _fallback(event, outcome.name)
            notes.append(f"{outcome.name}: fallback ({'budget' if budget_hit else 'outcome cap'})")
            continue
        try:
            prob, source = _binary_prob(binary_card(event, outcome), headlines,
                                        crowd, ask_fn, conserving)
        except Exception as exc:
            if not tournament._is_budget_error(exc):
                raise
            print(f"  budget cap hit on {event.slug}; crowd-price fallback for the rest")
            budget_hit, prob, source = True, None, "budget"
        if prob is None:
            probs[outcome.name] = _fallback(event, outcome.name)
            notes.append(f"{outcome.name}: fallback ({source})")
        else:
            probs[outcome.name] = prob
            notes.append(f"{outcome.name}: {source}")
    return probs, notes


def forecast_event(event, ask_fn, research_fn, conserving: bool | None = None) -> dict:
    """Answer one Prophet Arena event.

    Returns {"probabilities": {outcome name: clipped marginal}, "rationale":
    short text naming which engine tier (or fallback) produced it}. Keys are
    the event's own outcome spellings, every outcome exactly once.
    """
    if conserving is None:
        conserving = tournament._conserving()
    headlines = _headlines(event, research_fn)
    crowd = build_crowd_for()
    names = _names(event)
    try:
        if len(names) == 2:
            probs, notes = _answer_pair(event, headlines, crowd, ask_fn, conserving)
        elif getattr(event, "mutually_exclusive", False):
            probs, notes = _answer_mc(event, headlines, ask_fn)
        else:
            probs, notes = _answer_each(event, headlines, crowd, ask_fn, conserving)
    except Exception as exc:
        if not tournament._is_budget_error(exc):
            raise
        print(f"  budget cap hit on {event.slug}; crowd-price fallback")
        probs = {n: _fallback(event, n) for n in names}
        notes = ["fallback (budget cap)"]
    clipped = {n: tournament._tournament_clip(probs[n]) for n in names}
    return {"probabilities": clipped, "rationale": "manyworldz: " + "; ".join(notes)}
