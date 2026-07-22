"""The ensemble crowd: the research-backed diversity upgrade.

engine/methods.py builds a crowd where one model plays six different
analytical roles. That gets you six opinions, but every one of them is
still the same model looking at the same evidence and just being told to
think about it differently. Ensemble mode is a different kind of
diversity: DIFFERENT models, each looking at a DIFFERENT slice of the
evidence. A haiku seat that only sees headlines and a sonnet seat that
only sees the market price genuinely cannot agree for the same reason;
their disagreement is real, not roleplay.

build_crowd_for() is the one door both modes share: it reads
config.CROWD_MODE (or an explicit override) and hands back a crowd list
shaped the same way either way, so run_crowd, and everything that calls
it, never has to know which mode built its crowd.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from engine.methods import build_methods
from engine.pool import build_pool_crowd

# One line of instruction per evidence slice, telling the seat plainly
# what it does and doesn't get to see. This is the ensemble's version of
# engine/methods.py's per-method instruction line: not a fictional
# identity, just an honest description of this seat's evidence discipline.
_EVIDENCE_INSTRUCTIONS = {
    "headlines": ("you only see recent headlines here, no market price: "
                  "reason from the news alone"),
    "base_rates": ("you get no headlines and no market price here: reason "
                   "from how often things like this actually happen"),
    "market": ("you only see the market price line here, no headlines: "
              "reason from what the price implies"),
    "everything": "you see the market price and the headlines both: weigh them together",
}


def build_ensemble(seats: list[dict] | None = None) -> list[dict]:
    """Build the ensemble crowd: one agent dict per seat.

    `seats` defaults to config.ENSEMBLE_SEATS, read fresh at call time (not
    baked in at import time) so overriding config for a test, or editing
    the seat list, actually takes effect. Each seat needs "model" (a
    friendly name like "haiku" or any full model ID) and "evidence" (one
    of "headlines", "base_rates", "market", "everything").

    The returned agent dicts are shaped exactly like engine/methods.py's:
    "label" and "instruction", so run_crowd, agent_vote, and agent_futures
    all treat an ensemble seat like any other agent. Two extra keys ride
    along on top: "model" (this seat's own model, threaded into every
    ask_fn call it makes) and "evidence" (this seat's evidence slice,
    threaded into the prompt). A plain methods-mode agent never carries
    either key, which is exactly why "everything" and model=None are the
    defaults everywhere else in the engine: no key means "behave like it
    always did."
    """
    if seats is None:
        seats = config.ENSEMBLE_SEATS
    agents = []
    for seat in seats:
        model = seat["model"]
        evidence = seat["evidence"]
        instruction = _EVIDENCE_INSTRUCTIONS.get(
            evidence, _EVIDENCE_INSTRUCTIONS["everything"])
        agents.append({
            "label": f"{model}+{evidence}",
            "instruction": instruction,
            "model": model,
            "evidence": evidence,
        })
    return agents


def build_crowd_for(n_agents: int | None = None,
                    crowd_mode: str | None = None) -> list[dict]:
    """Build whichever crowd this run should use.

    This is the one shared helper ask.py, run.py, and tournament.py all
    call instead of picking between build_methods and build_ensemble
    themselves. `crowd_mode` overrides config.CROWD_MODE for one call (how
    ask.py's --crowd flag works); leave it None to read the configured
    default. "methods" builds the classic n-agent crowd, sized by
    `n_agents` (falls back to config.ENGINE_N_AGENTS). "ensemble" builds
    the fixed-seat crowd from config.ENSEMBLE_SEATS: `n_agents` is ignored
    there, since ensemble seats are configured directly, not sized per
    run. "pool" builds engine/pool.py's diversity crowd, sized by
    `n_agents` the same way "methods" is: every agent blends an
    independent method, temperament, and lens, so no two agents in a
    pool crowd think the same way until the combinations run out (see
    engine.pool.POOL_MAX_DISTINCT).
    """
    mode = crowd_mode if crowd_mode is not None else config.CROWD_MODE
    if mode == "ensemble":
        return build_ensemble()
    n = config.ENGINE_N_AGENTS if n_agents is None else n_agents
    if mode == "pool":
        return build_pool_crowd(n, config.SEED)
    return build_methods(n, config.SEED)
