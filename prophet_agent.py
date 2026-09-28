"""manyworldz on Prophet Arena: the SDK entrypoint.

Prophet Arena (prophetarena.co) runs "push" agents: the platform never
executes this code. The owner's machine (or the disabled-by-default
.github/workflows/prophet.yml) runs the SDK's live loop, which polls the
open forecast windows, calls agent.forecast(event) for each, validates
the answer locally, and POSTs it:

    prophet run prophet_agent.py --once --dry-run   # preview, submits nothing
    prophet run prophet_agent.py --once              # one real pass

Auth is the SDK's own: ARENA_API_KEY in the environment (or the
~/.arena/credentials.json that `prophet login` writes). The engine needs
ANTHROPIC_API_KEY and stops spending at ENGINE_BUDGET_USD; point
MANYWORLDZ_SPEND_FILE at its own meter so this loop never eats the
Metaculus bot's budget.

All the forecasting lives in adapters/prophet_arena.py, which reuses
tournament.py's answer paths. The track is "agentic" because the engine
does its own news research at forecast time (engine/news.py); the
platform treats the track as immutable once registered.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arena import Agent, Event, Forecast

from adapters.prophet_arena import forecast_event
from engine import llm, news

# Test seams: tests swap these for fakes so nothing touches the network.
ASK_FN = None
RESEARCH_FN = None


class ManyworldzAgent(Agent):
    name = "manyworldz"
    display_name = "manyworldz"
    track = "agentic"
    description = ("A crowd of LLM forecasters, each reasoning with a different "
                   "method over fresh news, pooled into one clipped probability.")
    card = {"models": ["claude-haiku-4-5", "claude-sonnet-5"],
            "tools": ["news search"],
            "methodology": "method-diverse LLM crowd vote, escalated to a "
                           "stronger model when contested; probabilities clipped"}

    def forecast(self, event: Event) -> Forecast:
        answer = forecast_event(event,
                                ask_fn=ASK_FN or llm.ask,
                                research_fn=RESEARCH_FN or news.research)
        return Forecast(probabilities=answer["probabilities"],
                        rationale=answer["rationale"][:1000])


agent = ManyworldzAgent()
