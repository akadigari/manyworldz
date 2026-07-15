# manyworlds — the gates (locked before any results exist)

Same idea as kayfabe/MechLab/TrendLab: we write the pass/fail rules first,
so we can't fool ourselves later. Verdicts get published either way.

## M0 gates (this milestone)

- **GO:** the post-cutoff scoring table has >= 350 games where each game
  carries a verified market close, AND a 50-row hand audit of the join
  finds <= 1% errors.
- **NO-GO:** either fails -> fix the closing-price source before writing
  any engine code.
- **G-leak:** if any probed model correctly names BOTH teams of a masked
  game on >= 10% of probes (chance is far under 1%), the pre-cutoff
  backtest is demoted to calibration-training only. The re-ID rate is
  published either way.

## Project gates (M1+, restated from the spec so they're in one place)

- **G0 (power):** >= 300 graded games in the held-out set before judging.
- **G1 (the Dr Strange test):** crowd Brier beats the market close.
- **G2 (the theater test):** crowd beats the boring logistic baseline.
- **G3 (luck test):** sign-randomization on edge picks beats shuffled labels.
- **G4 (cost test):** edge survives Kalshi fees + spread with a buffer.
- **G5 (worth-it test):** enough depth that the edge means real money.

Passing everything = a human may place small real bets. Failing any = an
honest "no edge," published as a portfolio result.
