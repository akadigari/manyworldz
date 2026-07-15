"""This file matches up two separate lists — real NBA game results and the
market's verified closing prices — into one clean table. That table is
the foundation the whole M0 backtest is built on.

If a game can't be matched cleanly (no verified price found, or more than
one match), it goes into a separate "quarantine" file with a reason
instead of being guessed at or silently dropped. We never fake a number
just to fill a row.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def build_table(games: list[dict], closes: pd.DataFrame,
                cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match each post-cutoff game to exactly one verified closing price.

    `games` is the full list of NBA results; `closes` is the table of
    verified market prices; `cutoff` is the date where "post-cutoff"
    starts (games on/after this date are the ones we actually score).
    Returns two tables: the clean matches ready to score, and the
    "quarantine" list of games that couldn't be matched cleanly (with a
    reason for each).
    """
    post = [g for g in games if g["date"] >= cutoff]
    # A "key" strings a game's date + home team + away team together so we
    # can look up "does this exact game exist in the closes table?" in one
    # dictionary lookup instead of scanning the whole table each time.
    key = lambda date, home, away: f"{date}|{home}|{away}"
    by_key: dict[str, list[dict]] = {}
    for _, r in closes.iterrows():
        by_key.setdefault(key(r["date"], r["home"], r["away"]), []).append(dict(r))

    rows, quarantined = [], []
    for game in post:
        matches = by_key.get(key(game["date"], game["home"], game["away"]), [])
        if len(matches) == 1:
            # Exactly one verified price for this game — a clean match.
            rows.append({"date": game["date"], "home": game["home"], "away": game["away"],
                         "home_won": game["home_won"],
                         "home_close_prob": matches[0]["home_close_prob"],
                         "provenance": matches[0]["provenance"]})
        else:
            # Zero matches (no price found) or 2+ matches (which one is
            # real?) — either way, we can't trust a single answer, so this
            # game gets set aside instead of guessed at.
            quarantined.append({"date": game["date"], "home": game["home"], "away": game["away"],
                        "reason": "no_close" if not matches else "duplicate_close"})
    cols = ["date", "home", "away", "home_won", "home_close_prob", "provenance"]
    return (pd.DataFrame(rows, columns=cols),
            pd.DataFrame(quarantined, columns=["date", "home", "away", "reason"]))


def write_outputs(table: pd.DataFrame, quarantine: pd.DataFrame) -> None:
    """Save the scoring table and quarantine list to CSV files in data/,
    plus a random 50-row sample for a person to hand-check.
    """
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
