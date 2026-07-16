# 🌌 manyworldz

**Every question splits into many worlds. This engine watches them all — and keeps score.**

Ask it anything about the future. Instead of one AI giving you one opinion,
manyworldz builds a small crowd of AI characters — a stats nerd, a contrarian,
a hype-follower, an oddsmaker — and each one independently works out the
odds. In simulate mode, each agent *imagines the event playing out* several
times and counts how the futures land. The crowd's views fold into one
honest probability.

Then the part almost nobody ships: **it grades itself.** On real prediction
markets, every paper pick is logged and scored against what actually
happened — with pass/fail rules written down *before* any results existed.

```
$ python ask.py "Will the Fed cut rates in September?"

  THE CROWD SAYS: 62% chance of YES
  (disagreement spread 0.11, 0 unusable answers skipped)

   70%  Ava    (stats nerd): futures pricing implies a cut is likelier than not
   55%  Ben    (contrarian): everyone expects it, which is exactly when it slips
   ...

$ python ask.py "Will the album drop this month?" --whatif "the label confirmed the date"

  THE FACT MOVES THE ODDS UP 34% (+34%)
```

## Quickstart (5 minutes)

```bash
git clone <this repo> && cd manyworldz
python3 -m venv venv && venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key        # get one at console.anthropic.com
venv/bin/python ask.py "Will it snow in DC this December?" --simulate
```

**Any key, any Claude model.** The engine reads `ANTHROPIC_API_KEY` from
your environment — nothing is hardcoded. Pick the crowd's brain with
`--model` (or the `MANYWORLDZ_MODEL` env var):

| Name | Model | Vibe |
|---|---|---|
| `haiku` | claude-haiku-4-5 | the default — a question costs ~a cent |
| `sonnet` | claude-sonnet-5 | smarter voices, ~3x the cost |
| `opus` | claude-opus-4-8 | strong reasoning, ~5x |
| `fable` | claude-fable-5 | the frontier — real cents per question |

```bash
venv/bin/python ask.py "Will it snow in DC this December?" --model fable --simulate
```

Answers are cached on your machine — asking the same thing twice is free.
A hard budget cap (`ENGINE_BUDGET_USD` in `config.py`, default $10) means
no model choice can ever surprise you.

## What's in the box

| Piece | What it does |
|---|---|
| `ask.py` | Ask the crowd anything; `--whatif` re-runs with a fact forced true |
| `run.py` | The live loop: crowd votes on real open prediction markets, logs paper picks |
| `engine/` | The crowd: personas, voting, simulated futures, deliberation, what-if |
| `ledger.py` | The scorecard — every pick graded against real closing prices (CLV) |
| `GATES.md` | The pre-registered rules for what would count as real skill |
| `docs/OWNERS_TOUR.md` | The whole project explained in plain English |

## The honesty rules

1. **Paper only.** The machine writes CSV rows; it cannot spend money or
   place bets. Any real-world decision belongs to a human.
2. **Pre-registered gates.** To ever claim "edge," the crowd must beat the
   market's closing price, beat a boring statistics baseline, survive a
   luck test, survive fees, and have enough depth to matter — rules locked
   before results existed. Failures get published too.
3. **Never fabricate.** Junk model answers are skipped and counted. If the
   whole crowd fails on a question, there is no answer — not a made-up one.
4. **Honest prompts.** Questions without a market price say so — the crowd
   is never fed a fake anchor.

## Run it in the cloud (laptop off)

The included GitHub Action (`.github/workflows/manyworldz.yml`) runs the live
loop four times a day on GitHub's servers and commits the scorecard back to
the repo. Setup: push to GitHub, add `ANTHROPIC_API_KEY` as a repository
secret (Settings → Secrets and variables → Actions), done.

## Why this exists

Multi-agent "prediction" demos are everywhere; published, graded track
records are almost nowhere. manyworldz is built backwards from that gap: the
crowd is the show, but the scorecard is the product. Expectations are set
honestly — published research says AI forecasters roughly match human
crowds at best — and the gates exist to find out, not to assume.

Research-lab project. Not financial advice, not a betting product.

MIT licensed. Built by [@akadigari](https://github.com/akadigari).
