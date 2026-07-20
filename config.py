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

# Which crowd gets built. "methods" is the classic crowd: one model plays
# all six analytical methods (engine/methods.py). "ensemble" is the
# diversity upgrade: DIFFERENT models each look at a DIFFERENT slice of the
# evidence, so the crowd's disagreement comes from genuinely different
# inputs, not just different personas talking to the same facts. Set
# MANYWORLDZ_CROWD_MODE to override without editing this file. --crowd on
# ask.py overrides both, for one run only.
CROWD_MODE = _os.environ.get("MANYWORLDZ_CROWD_MODE", "methods")

# The ensemble's seats. Each seat is one model looking at one evidence
# slice: "headlines" (recent news only, no market price), "base_rates"
# (neither headlines nor market price, reason from how often things like
# this happen), "market" (just the market price line, no headlines), or
# "everything" (all of it, same view a methods-mode agent gets). Edit this
# list freely: add seats, drop seats, change who sees what. "model" takes
# a friendly name (haiku, sonnet, opus, fable) or any full model ID the
# API serves. fable and opus cost more; this default keeps the ensemble
# cheap by leaning on haiku and sonnet.
ENSEMBLE_SEATS = [
    {"model": "haiku", "evidence": "headlines"},
    {"model": "sonnet", "evidence": "base_rates"},
    {"model": "haiku", "evidence": "market"},
    {"model": "sonnet", "evidence": "everything"},
]
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
# Default target is MiniBench, not the big seasonal tournament. MiniBench
# starts fresh every two weeks with about 60 questions, so a late entrant
# carries no deficit and gets a real answer in two weeks instead of four
# months. The seasonal tournament scores by a running SUM, so joining it
# late means every question that already closed is a permanent zero.
# Point this at "summer-futureeval-2026" (or the current season slug) when
# a season starts fresh and the prize run is worth it.
METACULUS_TOURNAMENT = _os.environ.get("METACULUS_TOURNAMENT", "minibench")
# How many unanswered questions one cycle takes on. Coverage is the whole
# game here: an unanswered question scores zero, and prize share goes with
# the square of the total, so skipping questions hurts twice. The budget
# cap in ENGINE_BUDGET_USD is still the real brake on spending.
TOURNAMENT_QUESTIONS_PER_RUN = 25

# How far short of 0% and 100% every submitted binary probability gets
# clipped, e.g. 0.98 instead of 0.999. Metaculus scores forecasts with a
# log rule, and a log rule punishes a confident miss brutally: a 99%
# "yes" that resolves no scores about as badly as a forecast can. The
# bots that actually rank near the top of these tournaments clip their
# extremes for exactly this reason, trading away a sliver of best-case
# score for a lot less downside if the crowd is confidently wrong.
TOURNAMENT_CLIP = 0.02

# Scoring reads the LAST forecast on file before a question closes, not
# the first one. A question this bot answered days ago can still get a
# stale, worse number graded if nothing refreshes it. Any already
# answered question whose close time is within this many hours gets one
# more fresh crowd run before the window shuts.
TOURNAMENT_REFRESH_HOURS = 24

# How many stale, soon-to-close questions one cycle will re-run and
# resubmit. Keeps refresh spend bounded the same way
# TOURNAMENT_QUESTIONS_PER_RUN bounds spend on brand new questions.
TOURNAMENT_REFRESH_CAP = 10
