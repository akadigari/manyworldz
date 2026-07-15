"""This file decides whether M0 (the first phase of the project) passed or
failed, using rules that were written down in GATES.md and config.py
BEFORE any results existed. That order matters: the rules can't be
quietly loosened after the fact just because the numbers came out badly.
This file only checks the rules — it never changes them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def evaluate(n_games: int, audit_errors: int, audit_n: int,
             worst_reid_rate: float) -> dict:
    """Check the three pre-registered gates and decide GO or NO-GO.

    Gate 1: did we end up with enough games that have a verified market
    price? Gate 2: when a person hand-checked a sample, was the error
    rate low enough to trust the matching? Gate 3 (doesn't block GO by
    itself, but matters): did the re-ID probe show the mask is leaky? If
    so, the older ("pre-cutoff") games get downgraded to
    "calibration-only" instead of counting as a real backtest, since a
    model that can partly guess the teams isn't purely predicting them.

    Returns a dict with the overall go/no-go call, whether the
    pre-cutoff backtest gets demoted, and a plain-English reason for
    each gate.
    """
    reasons = []
    if n_games >= config.GO_MIN_GAMES:
        reasons.append(f"games with verified closes: {n_games} (need {config.GO_MIN_GAMES}) — pass")
        games_ok = True
    else:
        reasons.append(f"games with verified closes: {n_games} (need {config.GO_MIN_GAMES}) — FAIL")
        games_ok = False

    err_rate = audit_errors / max(audit_n, 1)
    if err_rate <= config.GO_MAX_JOIN_ERROR:
        reasons.append(f"hand-audit errors: {audit_errors}/{audit_n} ({err_rate:.1%}) — pass")
        audit_ok = True
    else:
        reasons.append(f"hand-audit errors: {audit_errors}/{audit_n} ({err_rate:.1%}) — FAIL")
        audit_ok = False

    demote = worst_reid_rate >= config.REID_DEMOTION_RATE
    reasons.append(
        f"worst re-ID rate: {worst_reid_rate:.1%} — "
        + ("pre-cutoff backtest DEMOTED to calibration-only" if demote
           else "mask holds; pre-cutoff backtest stays scoreable"))

    return {"go": games_ok and audit_ok, "demote_precutoff": demote,
            "reasons": reasons}


if __name__ == "__main__":
    import pandas as pd
    table = pd.read_csv(config.DATA / "scoring_table.csv")
    probe = json.loads((config.DATA / "probe_results.json").read_text())
    # AUDIT.md line format (human writes it): "errors: 0 of 50"
    audit_line = next(l for l in (config.ROOT / "AUDIT.md").read_text().splitlines()
                      if l.startswith("errors:"))
    parts = audit_line.replace("errors:", "").split("of")
    audit_errors, audit_n = int(parts[0]), int(parts[1])

    verdict = evaluate(len(table), audit_errors, audit_n,
                 max(probe["per_model"].values()))
    lines = ["# M0 verdict", "",
             f"**{'GO' if verdict['go'] else 'NO-GO'}** — " +
             ("engine work may start." if verdict["go"]
              else "fix the data before any engine code."), ""]
    lines += [f"- {r}" for r in verdict["reasons"]]
    (config.ROOT / "M0_VERDICT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
