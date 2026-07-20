"""This file is the control panel for the whole project: every adjustable
number and setting lives here instead of being buried inside the code that
does the actual work. Want to change how many "agent" votes get cast, how
much money the project is allowed to spend, or what counts as a
"post-cutoff" game? Change it here, nowhere else.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent   # the folder this file lives in
DATA = ROOT / "data"                     # where results, logs, and downloads get saved
CACHE = DATA / "cache"                   # where we keep copies of network calls, so re-runs are free

# The same random seed everywhere, so every run can be repeated exactly.
# (Yes, it's the number of futures Strange checked.)
SEED = 14000605

# The probe asks these models to un-mask games. Haiku is also the model
# whose training-data cutoff defines "post-cutoff" games below.
PROBE_MODELS = ["claude-haiku-4-5", "claude-sonnet-5"]
PROBE_N = 100            # how many masked games each model sees
PROBE_BUDGET_USD = 5.00  # hard stop for probe spend

# Checked against Anthropic's model docs on 2026-07-15: claude-haiku-4-5
# lists a "reliable knowledge cutoff" of Feb 2025 and a broader TRAINING
# DATA cutoff of Jul 2025. For leak safety we use the broader one: the
# model may have seen anything up to July 2025. With a one-month buffer,
# games on/after this date count as "post-cutoff" (results the model
# cannot have seen in training).
MODEL_CUTOFF_DATE = "2025-08-01"

# Pre-registered gate numbers. Locked in GATES.md before results exist.
REID_DEMOTION_RATE = 0.10   # both-teams-named rate that demotes the backtest
GO_MIN_GAMES = 350          # post-cutoff games with a verified close
GO_MAX_JOIN_ERROR = 0.01    # allowed error rate when a person hand-checks 50 rows

# ---- M1 engine knobs (the "go harder" dials) ----

# Which AI powers the crowd. Use a friendly name or a full model ID.
# Friendly names (cheapest to strongest):
#   haiku  -> claude-haiku-4-5   (~1c per question, the default)
#   sonnet -> claude-sonnet-5    (smarter, ~3x the cost)
#   opus   -> claude-opus-4-8    (strong reasoning, ~5x)
#   fable  -> claude-fable-5     (the frontier, ~10x, a cycle costs real cents)
# Anyone's key works: the engine reads ANTHROPIC_API_KEY from the
# environment. Set MANYWORLDZ_MODEL to override the model without
# editing this file (that's how the cloud run picks its model too).
import os as _os
MODELS = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-4-8",
    "fable": "claude-fable-5",
}
ENGINE_MODEL = _os.environ.get("MANYWORLDZ_MODEL", "haiku")
ENGINE_N_AGENTS = 8       # agents per market
SIM_ROLLOUTS_K = 5        # "futures" (imagined ways the event could play out) each agent dreams up in simulate mode
SIM_MODE = "simulate"     # "simulate" (K imagined futures per run, the default) or "vote" (one number each)
DELIBERATION = False      # one round of agents seeing each other's takes
MIN_EDGE_CENTS = 10       # crowd must differ from market by this much...
FEE_BUFFER_CENTS = 3      # ...plus this cushion for fees/spread, to log a pick
MARKETS_PER_RUN = 5       # markets the crowd votes on per cycle
ENGINE_BUDGET_USD = 10.00 # hard stop for cumulative engine spend
EXCLUDED_CATEGORIES = {"Sports"}  # skip sports markets: Maryland law only lets us simulate trades on non-sports ones

# ---- deep split (engine/explore.py: keep imagining until nothing new shows up) ----
DEEP_MAX_ROUNDS = 8   # give up after this many rounds even if still finding new worlds
DEEP_DRY_ROUNDS = 2   # stop early once this many rounds in a row add nothing new

# ---- path mode (engine/explore.py: find_paths, the "beat Thanos" search) ----
PATH_MAX_ROUNDS = 5   # target-conditioned rounds run this many times, no early stop

# ---- Metaculus FutureEval tournament (tournament.py + adapters/metaculus.py) ----
# The tournament to forecast on, by ID or slug. FutureEval runs three
# seasons a year under a new slug each time. As of 2026-07-19 the live
# one is "summer-futureeval-2026" (metaculus.com/tournament/
# summer-futureeval-2026/), which is also the official bot template's
# own default tournament for this season. Seasons rotate, so this reads
# from the environment: set METACULUS_TOURNAMENT to the current slug
# (or numeric ID) once a new season starts, no code change needed.
METACULUS_TOURNAMENT = _os.environ.get("METACULUS_TOURNAMENT", "summer-futureeval-2026")
TOURNAMENT_QUESTIONS_PER_RUN = 5   # new questions the crowd answers per cycle, same cost-control idea as MARKETS_PER_RUN
