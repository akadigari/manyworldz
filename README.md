# manyworldz

An LLM-driven forecasting engine that runs a crowd of AI agents on any yes/no
question about the future — each agent researches, votes (or simulates the
outcome multiple times), the votes fold into one probability, and every
prediction gets logged and graded against real prediction-market prices.

Paper only. The engine never places bets — it writes CSV rows and keeps score.

## What it does

- **Ask it anything** — give it `"Will the Fed cut rates in September?"`; six
  to eight AI personas (a stats nerd, a contrarian, an oddsmaker, a
  sharp-money tracker...) each pull headlines, anchor on a base rate, and
  return their own probability with a one-line reason
- **Simulate mode** — each agent imagines the event playing out K different
  ways; its probability is how many of its futures land YES
- **What-if** — re-runs the whole crowd with a fact forced true and shows how
  much the odds move
- **Live loop** — scans ~4,000 open Kalshi markets (non-sports), votes on the
  biggest ones, and logs a paper pick when the crowd disagrees with the
  market by more than fees could explain
- **Scorecard** — every pick graded against closing prices (CLV). Results
  written to CSV and a dashboard automatically

## How it works

1. `adapters/kalshi_events.py` pulls open markets and turns them into simple
   "cards" (question, price, volume). `engine/news.py` grabs headlines from
   two search angles — no API key needed
2. `engine/personas.py` builds the crowd; `engine/swarm.py` collects votes,
   throws out junk answers (never invents one), trims the extremes, and
   returns one probability plus a disagreement spread
3. `run.py` compares the crowd's number to the market price. Gap bigger than
   edge + fee buffer → paper pick goes in `data/ledger.csv`
4. Next cycle, `ledger.py` re-checks every open pick: did the market move
   toward us (CLV), did it settle, did we win
5. `report.py` writes `web/data.json` — the dashboard draws straight from it

## Quick Start

```bash
git clone https://github.com/akadigari/manyworldz && cd manyworldz
python3 -m venv venv && venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key   # console.anthropic.com

venv/bin/python ask.py "Will it snow in DC this December?"
venv/bin/python ask.py "Will the album drop this month?" --simulate --whatif "the label confirmed the date"
venv/bin/python run.py              # one live market cycle
```

A question costs about a cent. Answers are cached — asking twice is free.

## Models

Pick the crowd's brain with `--model` or the `MANYWORLDZ_MODEL` env var:

| Name | Model | Cost |
|---|---|---|
| `haiku` | claude-haiku-4-5 | ~1c per question (default) |
| `sonnet` | claude-sonnet-5 | ~3x |
| `opus` | claude-opus-4-8 | ~5x |
| `fable` | claude-fable-5 | ~10x |

Any full model ID also works. Hard budget cap in `config.py`
(`ENGINE_BUDGET_USD`, default $10) — the engine stops calling the API when
it's spent, no surprises.

## The rules it can't break

- Paper only — a person makes any real decision, and only if the
  pre-registered gates pass (`GATES.md`, written before any results existed:
  beat the closing line, beat a boring baseline, survive a luck test,
  survive fees)
- Junk model answers get skipped and counted, never fabricated. All-junk
  crowd → no pick
- Questions with no market price are told so — the crowd never gets a fake
  anchor
- The dashboard reads the exact same ledger the gates read

## Run it in the cloud

`.github/workflows/manyworldz.yml` runs the live loop 4x/day on GitHub
Actions and commits the scorecard back. Setup: push to GitHub, add
`ANTHROPIC_API_KEY` as a repository secret (Settings → Secrets → Actions).
Done — laptop can stay off.

## Files

```
ask.py            ask the crowd anything (CLI)
run.py            one live cycle: grade -> vote -> log picks
engine/           llm client (cached, budget-capped), personas, swarm,
                  simulate mode, what-if, news research
adapters/         kalshi market cards (+ a dormant NBA backtest lab)
ledger.py         the scorecard: picks, CLV, settlement
report.py         ledger -> web/data.json + REPORT.md
web/index.html    the dashboard (static, no server) — branching-worlds map,
                  browser ask-the-crowd on your own key, receipts
GATES.md          pre-registered pass/fail rules
docs/             architecture + a plain-English owner's tour
```

## Requirements

- Python 3.11+
- An Anthropic API key (only for asking/voting — market scanning and the
  dashboard need none)
- `pip install -r requirements.txt` (anthropic, requests, pandas, pytest)

79 tests, all offline — `venv/bin/pytest` runs without a key or network.

## License

MIT
