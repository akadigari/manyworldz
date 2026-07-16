# manyworldz — design doc (formerly "Agamotto")

**RENAMED 2026-07-15 (owner call):** the project is now **manyworldz** — after
the many-worlds interpretation of quantum mechanics, where every possibility
branches into its own universe. Visual identity moves from the Marvel-relic
"Immortal Weapon" theme to a **Deep Space** theme. Name verified unclaimed in
this domain (existing GitHub "manyworldz" repos are shader art + Minecraft
plugins). knaves.ai remains the site brand.

**Date:** 2026-07-14
**Status:** approved in brainstorm (this doc is the written record)
**Owner:** Ayush (akadigari)

---

## What this is

manyworldz is the "Dr Strange machine": instead of one model guessing an outcome,
it builds a **crowd of AI forecasters**, lets each one form its own probability,
and reads the prediction off the crowd. It has a **god's-eye what-if mode** —
inject a scenario ("star player is OUT tonight") and watch the whole crowd's
number shift. First target: **NBA games**, scored against real market closing
prices over 3 seasons. Later targets (WNBA, MLB, Kalshi non-sports) are plug-in
adapters, not rebuilds.

Named for the many-worlds interpretation: every question splits reality into
branches, and the engine watches them all.

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

So manyworldz has two honest goals, in priority order:

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
│   ├── swarm.py         # run a crowd (either mode) → each agent VOTES or
│   │                    #   SIMULATES (see "seeing the futures") → optional
│   │                    #   single deliberation round → consensus + spread
│   ├── learn.py         # the learning loop: agent track records → earned
│   │                    #   voice weights, walk-forward only (see below)
│   ├── memory.py        # lessons store: what each agent got wrong and why,
│   │                    #   retrieved for similar future games
│   ├── calibrate.py     # logistic layer that learns when the crowd runs
│   │                    #   hot/cold (train: earlier seasons; test: held out)
│   └── whatif.py        # god's-eye: inject a scenario, re-run, show the shift
├── adapters/
│   └── nba.py           # nba_api → per-game features (last-10 form, home/away
│                        #   splits, pace, defensive rating, rest/back-to-backs,
│                        #   lineup availability, head-to-head) → MASKED
│                        #   stat-sheets (Team A vs Team B, no names)
├── ingest.py            # feed-it-anything layer: drop a file/URL into
│                        #   evidence/, it becomes time-stamped evidence cards
│                        #   the agents read on their next run
├── evidence/            # the drop folder (plus auto-pulled news per game)
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
  costumes. **Key fallback:** if only an Anthropic key is available, ensemble
  mode runs on different Claude tiers (Haiku / Sonnet / Opus) with partitioned
  evidence — weaker diversity, and the report says so explicitly.

## Seeing the futures (simulate mode — the Dr Strange head)

Each agent can run in two modes, set in config:

- **Vote mode:** the agent reads its evidence and gives one probability with a
  reason. Cheap, one call.
- **Simulate mode:** the agent *imagines the event playing out* K times
  (default K=5) — short scenario rollouts ("Team A's pace wears them down in
  the 4th…", "foul trouble flips it…") — and its probability is the fraction
  of its own futures where the outcome happens. The crowd collectively holds
  **N agents × K rollouts** of simulated futures, and the report can show the
  most common storylines, not just a number. Costs ~K× more per agent, so the
  cap decides how far it scales.

The what-if engine reuses this directly: inject a scenario and the rollouts
re-run with that fact forced true — that's the 14,000,605-futures move, sized
to a college budget.

## The learning loop (how it trains itself)

Honest version of "trains itself": the underlying models never change — what
learns is everything wrapped around them, and it learns from every settled
game automatically:

1. **Regrade:** when an event settles, the ledger grades every agent's call.
2. **Earned voice (learn.py):** each agent/archetype/model-family carries a
   running track record; aggregation weights shift toward agents that have
   been right (inverse-Brier with shrinkage, so a lucky streak doesn't take
   over). The contrarian that keeps nailing upsets earns a louder voice; the
   narrative fan that keeps losing gets quieter.
3. **Calibration refit:** the hot/cold correction layer refits on the growing
   ledger.
4. **Lessons memory (memory.py):** each miss is stored as a short plain-English
   lesson ("crowd overweighted home streaks in back-to-backs"); before voting
   on a similar future game, agents are shown their own relevant past
   mistakes.

5. **Punishment with teeth (the bench):** losing voice is the fine; the bench
   is the suspension. An agent whose rolling Brier stays worse than the
   crowd's average over its last 25 graded calls gets **benched** — it stops
   voting entirely. A benched agent serves probation: it keeps making silent
   shadow picks (logged, zero voice), and only re-enters the crowd after its
   shadow record beats the crowd average over the next 15 calls. Persistent
   busts get **retired** and a fresh agent variant is drafted in its place —
   the roster itself evolves under survival pressure.
6. **The scoring rule IS the pain:** agents are graded on Brier score, which
   punishes confident-and-wrong far harder than humble-and-wrong. An agent
   that screams 95% and misses eats ~9× the damage of one that said 65% and
   missed. Overconfidence — the thing LLMs are worst at — is exactly what
   hurts most.

**The no-fooling-ourselves rule:** all learned state is walk-forward only —
weights, lessons, benchings, and roster changes used on game N were learned
strictly from games that settled before N. And adaptivity itself is an
ablation arm: **adaptive weights + bench vs frozen equal weights**. If the
punishment system doesn't beat not-having-it on later games, we say so and
ship it off by default.

## Feed it anything (the ingest layer)

- **Auto news:** before each live vote, `ingest.py` pulls fresh headlines per
  game (Google News RSS pattern from kayfabe's research.py — no API key) into
  time-stamped evidence cards.
- **Drop anything:** put a file or URL into `evidence/` (scouting notes, an
  injury tweet screenshot's text, a stats CSV, an article) and it becomes
  evidence cards tagged to team/date with provenance. Agents read matching
  cards on their next run. New data source = drop it in, not a rebuild.
- **Time discipline:** every card carries a timestamp, and the backtest
  replayer only shows agents cards dated *before* the game — new data can
  never leak the future into a historical run.

## The ablation (what the backtest answers head-to-head)

Persona crowd vs ensemble crowd vs one strong model prompted once vs the
boring logistic baseline vs the market close — plus deliberation ON vs OFF,
**vote vs simulate mode**, and **adaptive vs frozen weights**. Every one of
these comparisons is a publishable result on its own.

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

**RESEQUENCED 2026-07-15 (owner call: "NBA is over — build the Dr Strange
predictor now, add the lab later").** The engine comes first, pointed at
live markets; the NBA backtest lab plugs back in whenever its data arrives
(M0's code is complete and waiting on the closes CSV + probe run).

**M1 — the engine, live (now).** The crowd core: personas + ensemble modes,
vote + simulate (K rollouts), one deliberation round, the what-if god's-eye,
paper ledger with CLV grading, and a **live Kalshi non-sports adapter** —
the crowd votes on real open markets (entertainment, econ, culture; MD-legal
year-round) and logs paper picks when the edge clears fees. No leak risk:
live events post-date every model's training.

**M2 — the learning loop (as picks settle).** Earned voice, the bench +
shadow picks + retirement, lessons memory, calibration refit — all
walk-forward on the growing live ledger, plus REPORT.md.

**M-lab — the NBA backtest lab (deferred; resumes when data lands or by
October).** M0's pilot (closes CSV join, 50-row audit, re-ID probe, GO/NO-GO)
then the sampled backtest + full ablation vs the boring baseline and the
market close. This is where the headline accuracy claims get their evidence.

**M-viz — the knaves.ai dashboard (after M1; mock-data demo anytime).** The
Multiverse Map front door + War Room and Receipts tabs, reading `report.py`'s
exported JSON. See "The face" section.

**M-money — Kalshi non-sports blend lane (optional, after M1).** Same engine,
non-sports adapter, LLM + market-mid blend (~0.25 model weight), paper quotes
one tick inside the book on the blend-favored side. Its own 14-day
pre-registered test: ≥150 strict trade-through fills across a stratified
~200-market sample AND fee-adjusted CLV > 0 (bootstrap p < 0.10) AND blend
Brier ≤ market-mid Brier on in-window resolutions. Any failure = kill or
re-scope. This lane is legal in MD regardless of the 4th Circuit.

## The face (knaves.ai — the Multiverse Map)

manyworldz gets a public dashboard at **knaves.ai** (owner's existing domain,
repointed from the old horse-betting use). Decided 2026-07-15 via visual
mockups:

- **Front door — the Multiverse Map:** tonight's events rendered as branching
  futures trees built from the crowd's simulate-mode rollouts. Branch
  thickness = share of futures that agree; the winning storyline glows. A
  **what-if chip** ("star player OUT") re-runs the swarm and animates the tree
  re-growing — the 14,000,605 experience in a browser.
- **Tab: the Sanctum War Room** — the Eye as a consensus dial vs the market
  price, the crowd as a constellation where each agent's dot grows as it earns
  voice (the learning loop, visible), the lessons feed, and **the bench**:
  suspended agents shown dimmed with their shadow record, so the punishment
  system is publicly visible too.
- **Theme — "Deep Space"** (owner pivot 2026-07-15, replacing "Immortal
  Weapon"): space-black background `#060913` with a faint starfield;
  star-white text `#e8ecf5`; **starlight gold** `#f5c542` for the wordmark,
  chosen worlds, and primary actions; **nebula crimson** `#d4435c` for energy
  branches; faint violet `#8b7ff0` for secondary worlds. The Multiverse Map
  reads as branching light-trails across space — every future is a star, the
  brightest branch is the world the crowd believes in.
- **Tab: the Receipts Board** — gate lights (pass/fail, verbatim), calibration
  curve, CLV track record, every pick with its story.

Rules that keep it honest and cheap:

- **Single source of truth:** the site is a static page reading JSON exported
  by `report.py` — the same ledger the gates read. The dashboard can never
  show a rosier number than REPORT.md.
- **$0 to run:** static hosting (GitHub Pages or Vercel), no backend, no
  accounts, no API keys in the browser. Live mode = the hourly Action rewrites
  the JSON.
- **No funnels:** gate fails display as fails; no copy-trading, no
  "join my picks" anything. It's a lab window, not a tip service.
- **Milestone M-viz** (after M1, needs real backtest output): build the three
  views on exported JSON. A mock-data version can be built anytime for demos.
- Repointing the domain's DNS is a human step (owner's registrar), not code.

## The gates (pre-registered — locked before results exist)

- **G-leak:** if the re-ID probe correctly names both teams on **≥10%** of
  masked games (chance is well under 1%), the pre-cutoff backtest is
  **demoted to calibration-training only** and every accuracy claim rests on
  post-cutoff games. The re-ID rate is published either way.
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
  deliberation, vote↔simulate mode, rollouts-per-agent K. Nothing structural.
- Simulate mode multiplies cost by ~K, so v1 runs it on a subset of the
  backtest sample; the cap, not enthusiasm, decides how far it scales.
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

## Public framing (README + knaves.ai rules)

The GitHub repo and the site tell **manyworldz's own story**, on its own terms:

- Lead with what it is: a multi-agent outcome-simulation engine — a crowd of
  AI forecasters, simulated futures, a what-if god's-eye, a learning loop, and
  an honest evaluation lab with pre-registered gates.
- **No MiroFish mentions** in the README or site — no comparisons, no debunk
  content. That research stays in the private wiki. (Nothing dishonest about
  this: zero shared code, and the underlying idea is decades-old public
  research. Related-work credits go to academic papers, which read stronger
  anyway.)
- **Don't frame it as a betting product.** Sports is "the first evaluation
  domain" (rich data, hard benchmark), not the identity. It's a paper-only
  research lab that scores itself against real-world outcomes and market
  prices — that framing is also the recruiting-friendly one.
- The internal spec (this doc) keeps the full evidence trail; public docs
  keep the receipts about *our own* results.

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
- No model fine-tuning or "retraining the AI" — at our data size that's
  overfitting sold as learning. The learning loop (weights, calibration,
  lessons memory) is the version that can actually be proven to help.
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
