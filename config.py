"""This file is the control panel for the whole project: every adjustable
number and setting lives here instead of being buried inside the code that
does the actual work. Want to change how many "agent" votes get cast, how
much money the project is allowed to spend, or what counts as a
"post-cutoff" game? Change it here — nowhere else.
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
# DATA cutoff of Jul 2025. For leak safety we use the broader one — the
# model may have seen anything up to July 2025. With a one-month buffer,
# games on/after this date count as "post-cutoff" (results the model
# cannot have seen in training).
MODEL_CUTOFF_DATE = "2025-08-01"

# Pre-registered gate numbers. Locked in GATES.md before results exist.
REID_DEMOTION_RATE = 0.10   # both-teams-named rate that demotes the backtest
GO_MIN_GAMES = 350          # post-cutoff games with a verified close
GO_MAX_JOIN_ERROR = 0.01    # allowed error rate when a person hand-checks 50 rows

# ---- M1 engine knobs (the "go harder" dials) ----

ENGINE_MODEL = "claude-haiku-4-5"  # cheap crowd voices; raise tier here to go harder
ENGINE_N_AGENTS = 8       # agents per market
SIM_ROLLOUTS_K = 5        # "futures" (imagined ways the event could play out) each agent dreams up in simulate mode
SIM_MODE = "vote"         # "vote" (one number per agent) or "simulate" (K futures each)
DELIBERATION = False      # one round of agents seeing each other's takes
MIN_EDGE_CENTS = 10       # crowd must differ from market by this much...
FEE_BUFFER_CENTS = 3      # ...plus this cushion for fees/spread, to log a pick
MARKETS_PER_RUN = 5       # markets the crowd votes on per cycle
ENGINE_BUDGET_USD = 10.00 # hard stop for cumulative engine spend
EXCLUDED_CATEGORIES = {"Sports"}  # skip sports markets — Maryland law only lets us paper-trade non-sports ones
