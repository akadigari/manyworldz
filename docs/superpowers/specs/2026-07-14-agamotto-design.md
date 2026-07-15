# Agamotto — design doc

**Date:** 2026-07-14
**Status:** approved in brainstorm (this doc is the written record)
**Owner:** Ayush (akadigari)

---

## What this is

Agamotto is the "Dr Strange machine": instead of one model guessing an outcome,
it builds a **crowd of AI forecasters**, lets each one form its own probability,
and reads the prediction off the crowd. It has a **god's-eye what-if mode** —
inject a scenario ("star player is OUT tonight") and watch the whole crowd's
number shift. First target: **NBA games**, scored against real market closing
prices over 3 seasons. Later targets (WNBA, MLB, Kalshi non-sports) are plug-in
adapters, not rebuilds.

Named for the Eye of Agamotto — the relic Doctor Strange looks through to run
the 14,000,605 futures.

**PAPER-FIRST. Pre-registered gates. The verdict gets published either way.**

## Why we believe what we believe (the evidence, honestly)

A deep Scout sweep (2026-07-14, 25 agents, adversarially refuted — filed at
`wiki/resources/scout-trading-multiagent-simulation-for-prediction-and-forecasting.md`)
found:

- **Nobody in print beats market closing prices with LLM agent swarms.** The
  best published result is *parity with human crowds* (Schoenegger's 12-LLM
  ensemble matched 925 humans), and multi-agent Brier gains are measured
  against LLM baselines, never against market prices.
- **The viral "$1.49M MiroFish NBA system" story is a mashup.** The real trader
  (Polymarket's 0p0jogggg, $1.48M PnL) made their money on elections, Champions
  League, Norwegian football, Valorant, and Dota 2 — not NBA — and MiroFish's
  own README makes zero profit claims and contains no betting code.
- **Anonymized backtests leak.** Models re-identify "anonymized" games at high
  rates (expect ≥30%). So we don't *trust* the masker — we **measure the leak
  and publish the number**, and the backtest's status depends on it.
- **The one realistic money lane is elsewhere:** thin Kalshi **non-sports**
  markets, where an LLM+market-mid blend can quote — but fills are the killer
  (median 24h volume in wide-spread books is zero), so that lane gets its own
  brutal pre-registered fills test.

So Agamotto has two honest goals, in priority order:

1. **The flagship:** a leakage-audited answer to "can an AI crowd beat the
   market?" — the audit machinery (re-ID probe, post-cutoff scoring, ablations,
   pre-registered gates) is what the 68k-star LLM-forecasting ecosystem never
   ships. Valuable for the portfolio and recruiting no matter which way the
   verdict lands.
2. **The money arm:** the same engine pointed at Kalshi non-sports thin
   markets, which is legal in Maryland regardless of the 4th Circuit sports
   ruling, gated by its own kill-fast paper test.

We are explicit about expectations: the likely headline is "the crowd ties a
boring baseline and neither beats the close." If instead something passes every
gate, we found something real. Either result is worth publishing.

## Repo layout

```
agamotto/
├── README.md            # the story + the live verdict, updated by report.py
├── GATES.md             # pass/fail rules, written BEFORE any results exist
├── config.py            # every dial: mode, model(s), crowd size, deliberation
│                        #   on/off, backtest sample size, hard $ budget cap
├── engine/
│   ├── personas.py      # persona mode: archetype library (stat analyst,
│   │                    #   narrative bettor, sharp-money tracker, oddsmaker,
│   │                    #   insider, contrarian) built per game
│   ├── ensemble.py      # ensemble mode: different model families, each given
│   │                    #   a DIFFERENT slice of evidence (news / base rates /
│   │                    #   price history), trimmed-mean aggregation
│   ├── swarm.py         # run a crowd (either mode) → votes → optional single
│   │                    #   deliberation round → consensus + disagreement
│   ├── calibrate.py     # logistic layer that learns when the crowd runs
│   │                    #   hot/cold (train: earlier seasons; test: held out)
│   └── whatif.py        # god's-eye: inject a scenario, re-run, show the shift
├── adapters/
│   └── nba.py           # nba_api → per-game features (last-10 form, home/away
│                        #   splits, pace, defensive rating, rest/back-to-backs,
│                        #   lineup availability, head-to-head) → MASKED
│                        #   stat-sheets (Team A vs Team B, no names)
├── masker.py            # the anonymizer + the re-identification probe
├── markets/
│   └── closes.py        # verified closing prices: MGM Kaggle NBA dataset
│                        #   (2021-22 → 2026-02-12) + Kalshi candlesticks
│                        #   (Apr 2025 →); every join hand-auditable
├── baseline.py          # boring logistic model on the SAME features (control)
├── backtest.py          # replay games, Brier + CLV vs close, per source
├── ledger.py, report.py # paper picks, grading, live REPORT.md
└── run.py               # one live cycle: tonight's games → crowd → edge →
                         #   paper pick (October, only if gates allow)
```

Boundaries: the **engine never knows what a rebound is** (stat-sheets in,
probabilities out); the **adapter never knows what a persona is**. A new sport
or market category = one new adapter file.

## The two ways to build the crowd (both get measured)

- **Persona mode** (the Dr Strange story): one model plays N different
  characters — the stats nerd, the narrative fan, the sharp-money watcher, the
  contrarian — each votes with a reason.
- **Ensemble mode** (what the evidence says works better): N *different model
  families* (Claude, GPT, Gemini, one open model), each shown a *different
  slice* of the evidence, combined with a trimmed mean. Diversity from
  genuinely different brains and different information, not one brain in
  costumes.

The backtest runs the **ablation** that answers, head-to-head:
persona crowd vs ensemble crowd vs one strong model prompted once vs the
boring logistic baseline vs the market close. Also measured: deliberation ON
vs OFF (does letting agents see each other's arguments help or hurt?). These
comparisons are themselves publishable results.

## Milestones (kill-cheap order)

**M0 — leak + data pilot (weeks 1–2). No swarm code allowed.**
1. Build `masker.py`, run the **re-ID probe**: 100 masked 2024-25 game
   descriptions into 2–3 models — "name these teams." Log the re-ID rate.
   Whatever the number is, it gets published and it decides the backtest's
   status (see Gates).
2. Build the **post-cutoff scoring table** the whole project stands on: for a
   model with a documented training cutoff, join nba_api results to verified
   closes (MGM Kaggle through 2026-02-12; Kalshi candlesticks after) for
   games dated *after* the cutoff.
- **GO:** ≥350 post-cutoff games each carry a verified close AND a 50-game
  hand audit shows ≤1% join errors.
- **NO-GO:** re-scope the closing-price source before writing anything else.

**M1 — engine + backtest (weeks 3–5).** Swarm (both modes), baseline,
calibration (train on earlier seasons, test only on held-out/post-cutoff),
sampled backtest sized to budget (~300–500 games at v1), full ablation.

**M2 — verdict (week 6).** `backtest.py` output vs GATES.md, REPORT.md and
README updated, result published either way.

**M3 — live NBA paper arm (late October, opening night).** Only if gates
allow. Real names permitted going forward (no leak risk on future games).
Hourly GitHub Actions + Telegram, patterns lifted from kayfabe.

**M-money — Kalshi non-sports blend lane (optional, after M1).** Same engine,
non-sports adapter, LLM + market-mid blend (~0.25 model weight), paper quotes
one tick inside the book on the blend-favored side. Its own 14-day
pre-registered test: ≥150 strict trade-through fills across a stratified
~200-market sample AND fee-adjusted CLV > 0 (bootstrap p < 0.10) AND blend
Brier ≤ market-mid Brier on in-window resolutions. Any failure = kill or
re-scope. This lane is legal in MD regardless of the 4th Circuit.

## The gates (pre-registered — locked before results exist)

- **G-leak:** if the re-ID probe shows the masker leaks (rate meaningfully
  above chance), the pre-cutoff backtest is **demoted to calibration-training
  only** and every accuracy claim rests on post-cutoff games. The re-ID rate
  is published either way.
- **G0 (power):** ≥300 graded games in the held-out set before judging.
- **G1 (the Dr Strange test):** crowd's Brier beats the **market close** on
  held-out games.
- **G2 (the theater test):** crowd beats the **boring logistic baseline** on
  the same features — else the swarm is expensive theater and we say so.
- **G3 (luck test):** sign-randomization on edge picks (wallet_screener
  method) — the edge must beat shuffled labels.
- **G4 (cost test):** hypothetical edge picks stay +EV after Kalshi fees and
  spread, with a buffer.
- **G5 (worth-it test):** enough market depth that passing G1–G4 means real
  money could actually be placed.
- **Verdict:** passes all → candidate for a human to place small real bets.
  Fails any → honest "no edge," published as a portfolio result — same as
  MechLab and TrendLab.

## Cost control

- Hard budget cap in `config.py`; the engine counts tokens and **halts** at
  the cap. v1 budget: **$20–50 total.**
- Every crowd vote cached to disk keyed by (game, agent, config-hash): reruns
  are free, backtests resume, scaling up never re-pays for old votes.
- "Go harder" = config change only: crowd size, model tier, sample size,
  deliberation. Nothing structural.
- Default models: Haiku for crowd votes, stronger model only where measured to
  matter.

## Error handling

- API failure on a vote → skip, log, never fabricate. A game with missing
  votes below quorum is dropped from scoring, counted in the report.
- Data-join anomalies (missing close, duplicate game id, date mismatch) →
  quarantine file, never silently patched.
- Deterministic seeds for agent assignment; any run reproducible exactly.
- Live mode inherits kayfabe's dead-man's-switch pattern (health check +
  watchdog) when M3 arrives.

## Testing

- Unit tests: masker (no team/player/city/arena strings survive), the
  closes join (spot-checked against the 50-game hand audit), ledger math
  (CLV/Brier/Kelly), config cap enforcement.
- The re-ID probe doubles as the leak regression test — rerun on masker
  changes.
- One recorded end-to-end fixture game so the whole pipeline replays in CI
  without API keys.

## Code style

Plain, simple English in comments, docstrings, and output — high-school level,
no jargon (owner preference, applies to everything). Match the kayfabe/tipoff
house style. Commits never carry AI co-author trailers.

## What we are NOT building (on purpose)

- No full social-world simulation (fake Twitter, multi-round herd dynamics) —
  evidence says LLM crowds polarize faster than humans; a single deliberation
  round is the tested, affordable slice of that idea.
- No transformer calibrator — logistic regression is the honest tool at this
  data size.
- No real-money execution of any kind; a person places any real bet, and only
  after every gate passes.
- No Polymarket trading (geo-blocked in MD; data use only).
- No copy-trading features, ever — the MiroFish-brand copy-trade funnel is the
  scam this project's honesty is designed to be the opposite of.

## Key references

- Scout board: `wiki/resources/scout-trading-multiagent-simulation-for-prediction-and-forecasting.md`
- Schoenegger et al. — LLM ensemble ↔ human crowd parity (arXiv 2311.10054 → Sci. Adv.)
- InfoDelphi / partitioned-evidence ensembles (arXiv 2607.01661)
- AIA Forecaster — blend weights vs market prices (arXiv 2511.07678)
- MGM Kaggle NBA closing-odds dataset (2021-22 → 2026 All-Star break)
- Kalshi single-game NBA candlesticks (Apr 2025 →), via kayfabe's data layer
- MiroFish (github.com/666ghj/MiroFish) — idea lineage; no code used (AGPL)
- Our own prior art: kayfabe/sim.py (persona crowd v0), GATES.md pattern,
  wallet_screener's sign-randomization luck test
