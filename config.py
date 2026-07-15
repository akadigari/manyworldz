"""One place for every knob in Agamotto M0.

Change numbers here, never inside the pipeline files.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / "cache"

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
GO_MAX_JOIN_ERROR = 0.01    # allowed error rate in the 50-row hand audit
