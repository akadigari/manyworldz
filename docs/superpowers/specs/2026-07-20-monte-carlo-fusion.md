# manyworldz: Monte Carlo fusion layer (spec)

**Date:** 2026-07-20
**Status:** approved by owner ("ok lets do it build it"); built from this spec, not from chat
**Owner:** Ayush (akadigari)

## Goal

Make "a million simulated outcomes" literally true. Add a numeric simulation
layer under the crowd: the minds do the judging, the numbers do the rolling.
One command runs the crowd, then rolls one million futures through the
crowd's beliefs and reports what share of those futures end YES.

## The honest limit (stated up front)

The million rolls measure the crowd's belief precisely. They do not create
new knowledge. Rolling a million dice through the same beliefs cannot make
the beliefs smarter; it makes the summary of them exact, exposes the true
uncertainty band, and makes the output defensible. The spec says this so
the marketing never outruns the math.

## Design

New file `engine/carlo.py`. Three steps:

1. **Elicit.** Each agent in the crowd (any crowd mode: methods, ensemble,
   pool) answers one structured prompt: `{"probability": p, "low": a,
   "high": b, "reason": "..."}` where `low`/`high` are the agent's honest
   80 percent band on its own probability (how sure it is about its own
   number). Reuse the existing vote machinery, `extract_json`, junk
   skipping, and clamps. Junk answers are skipped and counted, never
   fabricated.
2. **Roll.** Build a mixture of the elicited beliefs. For each of
   `CARLO_DRAWS` (config, default 1,000,000) draws: pick one agent
   uniformly, sample p from a triangular distribution (low, peak at
   probability, high) clipped to [0.01, 0.99], then draw the outcome as a
   Bernoulli(p) coin flip. Pure stdlib `random.Random(config.SEED)`,
   fully deterministic. No new dependencies.
3. **Report.** Probability = share of draws that ended YES. Also report
   the 10th/50th/90th percentile of the sampled p values (the crowd's
   belief band), the number of futures rolled, agents used, junk skipped.

## CLI

`ask.py "question" --carlo` runs elicit + roll and prints:

```
  ONE MILLION FUTURES ROLLED: 71.8% ended YES
  the crowd's belief band: 62% to 81% (80% of futures fell here)
  (8 minds elicited, 0 junk skipped, seed 14000605)
```

`--carlo` composes with `--crowd` and `--agents`. It does not change
`--deep`, `--path`, the default mode, run.py, or tournament.py in v1.

## Non-goals (v1)

- No market-data fitting, no time series, no continuous questions.
- No change to consensus math, budget cap, clipping, or any existing mode.
- No wiring into the tournament or the Kalshi loop until the ledger shows
  the carlo number is at least as calibrated as the plain fold.

## Config

`CARLO_DRAWS = 1_000_000` with a comment on cost: draws are free (no API
calls); only the elicitation costs money, same as one normal crowd run.

## Tests (offline, injected ask_fn)

Determinism (same seed, same result); all-agents-at-p sanity (crowd
unanimous at 0.7 with tight bands rolls to ~0.70 within 0.01); band
ordering (low <= p <= high enforced, junk bands repaired or skipped);
junk elicitation skipped and counted; percentile math; clamps at both
ends; 1M draws complete in under ~10 seconds.

## Milestones

- M1: engine/carlo.py + tests green.
- M2: ask.py --carlo + README section.
