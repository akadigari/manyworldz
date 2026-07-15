# The Owner's Tour — what manyworlds actually is, in plain English

This is the document you read twice and then never need again. No jargon.
After this, you can explain your own project to anyone — including you.

## The 30-second version (say this in an interview)

> "manyworlds is a forecasting engine. Instead of asking one AI a question, it
> builds a small crowd of AI 'characters' with different personalities — a
> stats nerd, a contrarian, a hype-follower — and each one independently
> estimates the odds of a real-world event. Some of them literally imagine
> the event playing out several times, like running simulations. The crowd's
> answers get averaged into one probability. Then — the part nobody else
> does — it checks its own answers against real prediction-market prices and
> keeps a public scorecard. Most AI forecasting projects show you a demo;
> mine keeps receipts."

That's it. Everything else is detail.

## What happens when it runs (one cycle, step by step)

1. **It looks at real markets.** It pulls every open market on Kalshi (a
   regulated prediction exchange) and keeps the non-sports ones that are
   actually liquid — about 4,000 right now. Things like "Will CPI come in
   above 3%?" or "Will this album drop in July?"
2. **It picks the five biggest** and grabs a few fresh news headlines for
   each one (free, from Google News).
3. **The crowd votes.** Eight AI characters each read the question, the
   market's current price, and the headlines — and give their OWN probability
   with a one-sentence reason. Junk answers get thrown out, never invented.
   (In "simulate mode," each character instead imagines 5 different ways the
   event could play out and counts how many end in YES.)
4. **The votes become one number.** The highest and lowest votes get dropped
   (so one weird agent can't hijack it), the rest are averaged. How much the
   agents disagree becomes a confidence measure.
5. **It only "acts" when it strongly disagrees with the market.** If the
   crowd says 65% and the market says 43%, that's a 22-cent gap — bigger than
   the 13-cent hurdle (edge + fees) — so it logs a pretend bet ("paper pick")
   in a spreadsheet. No real money ever moves. Ever.
6. **It grades itself.** Every later run, it re-checks its old picks: did the
   market move toward the crowd's number (good sign — called CLV, "closing
   line value") and when the event settles, was the crowd right?

## The what-if button (the fun one)

`engine/whatif.py` lets you inject a fact — "treat as true: the star player
is out" — and re-run the crowd. You watch the probability move. That's the
"simulate the future under different conditions" feature.

## What each file does (one line each)

| File | What it is |
|---|---|
| `run.py` | The conductor: one full cycle (grade old picks → crowd votes → log new picks) |
| `engine/personas.py` | The cast: builds the 8 AI characters |
| `engine/swarm.py` | The voting room: collects opinions, trims extremes, makes the number |
| `engine/futures.py` | Simulate mode: each character imagines the event K times |
| `engine/whatif.py` | The what-if button |
| `engine/llm.py` | The only door to the AI: caches every answer, hard $10 spending cap |
| `engine/news.py` | Free headline fetcher (never crashes the run) |
| `adapters/kalshi_events.py` | Translator: turns Kalshi's market data into simple cards |
| `ledger.py` | The scorecard: every pick, graded |
| `config.py` | Every setting in one file (crowd size, budget, edge threshold) |
| `GATES.md` | The rules for calling something "an edge," written BEFORE results |
| `masker.py`, `adapters/nba.py`, `markets/`, `jointable.py`, `verdict_m0.py` | The dormant NBA test-lab (built, waiting for its data; horses may replace it) |

## The three honesty rules (why this isn't a gambling bot)

1. **Paper only.** The machine writes rows in a CSV. It cannot spend money.
2. **Pre-registered gates.** GATES.md was locked before any results existed:
   to ever count as "real edge," picks must beat the market's closing price,
   beat a boring statistics model, survive a luck test, survive fees, and
   have enough market depth to matter. Fail any → published as "no edge."
3. **Never fabricate.** If the AI gives a garbage answer, it's skipped and
   counted — the crowd can't invent an opinion from nothing.

## Numbers to have in your pocket

- 8 agents per market, 5 markets per cycle, ~40 AI calls/cycle ≈ pennies
- $10 hard budget cap, every answer cached (re-runs are free)
- 74 automated tests, all runnable offline with zero API keys
- 3,969 live tradeable markets found on the last real scan
- Edge rule: crowd must differ from market by 10¢ + 3¢ fee buffer

## What it is NOT

- Not a betting service. Not financial advice. A person makes any real
  decision, and only if the gates ever pass (so far, nothing has earned it).
- Not a clone of anything: the crowd members are independent and get judged
  individually — there's no fake social network, because research shows AI
  agents herd worse than humans. The scoreboard is the whole identity.
