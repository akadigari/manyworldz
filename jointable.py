"""Join game results to verified market closes — the table M0 stands on.

Anything odd goes to a quarantine file with a reason. We never guess and
never silently drop a post-cutoff game.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def build_table(games: list[dict], closes: pd.DataFrame,
                cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    post = [g for g in games if g["date"] >= cutoff]
    key = lambda d, h, a: f"{d}|{h}|{a}"
    by_key: dict[str, list[dict]] = {}
    for _, r in closes.iterrows():
        by_key.setdefault(key(r["date"], r["home"], r["away"]), []).append(dict(r))

    rows, bad = [], []
    for g in post:
        matches = by_key.get(key(g["date"], g["home"], g["away"]), [])
        if len(matches) == 1:
            rows.append({"date": g["date"], "home": g["home"], "away": g["away"],
                         "home_won": g["home_won"],
                         "home_close_prob": matches[0]["home_close_prob"],
                         "provenance": matches[0]["provenance"]})
        else:
            bad.append({"date": g["date"], "home": g["home"], "away": g["away"],
                        "reason": "no_close" if not matches else "duplicate_close"})
    cols = ["date", "home", "away", "home_won", "home_close_prob", "provenance"]
    return (pd.DataFrame(rows, columns=cols),
            pd.DataFrame(bad, columns=["date", "home", "away", "reason"]))


def write_outputs(table: pd.DataFrame, quarantine: pd.DataFrame) -> None:
    config.DATA.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.DATA / "scoring_table.csv", index=False)
    quarantine.to_csv(config.DATA / "quarantine.csv", index=False)
    n = min(50, len(table))
    table.sample(n, random_state=config.SEED).to_csv(
        config.DATA / "audit_sample.csv", index=False)


if __name__ == "__main__":
    # CLI: venv/bin/python jointable.py
    from adapters.nba import fetch_season_results
    from markets.closes import load_kaggle_closes
    games = []
    for season in ("2024-25", "2025-26"):
        games += fetch_season_results(season)
    closes = load_kaggle_closes(config.DATA / "nba_closes.csv")
    table, quarantine = build_table(games, closes, config.MODEL_CUTOFF_DATE)
    write_outputs(table, quarantine)
    print(f"scoring table: {len(table)} games | quarantine: {len(quarantine)}")
