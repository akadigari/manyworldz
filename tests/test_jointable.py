import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointable import build_table


def games():
    return [
        {"game_id": "1", "date": "2025-03-05", "home": "BOS", "away": "NYK",
         "home_pts": 120, "away_pts": 100, "home_won": True},
        {"game_id": "2", "date": "2025-03-06", "home": "LAL", "away": "DEN",
         "home_pts": 100, "away_pts": 110, "home_won": False},
        {"game_id": "3", "date": "2025-03-07", "home": "MIA", "away": "ORL",
         "home_pts": 99, "away_pts": 98, "home_won": True},   # no close -> quarantine
        {"game_id": "4", "date": "2025-01-01", "home": "BOS", "away": "MIA",
         "home_pts": 100, "away_pts": 90, "home_won": True},  # pre-cutoff -> excluded silently
    ]


def closes():
    return pd.DataFrame([
        {"date": "2025-03-05", "home": "BOS", "away": "NYK", "home_close_prob": 0.66, "provenance": "kaggle"},
        {"date": "2025-03-06", "home": "LAL", "away": "DEN", "home_close_prob": 0.45, "provenance": "kaggle"},
        {"date": "2025-03-06", "home": "LAL", "away": "DEN", "home_close_prob": 0.46, "provenance": "kaggle"},
    ])


def test_join_quarantines_instead_of_guessing():
    table, quarantine = build_table(games(), closes(), cutoff="2025-03-01")
    assert len(table) == 1                       # only the clean BOS game
    assert table.iloc[0]["home_close_prob"] == 0.66
    assert bool(table.iloc[0]["home_won"]) is True
    reasons = dict(zip(quarantine["home"], quarantine["reason"]))
    assert reasons == {"LAL": "duplicate_close", "MIA": "no_close"}


def test_pre_cutoff_games_are_excluded_not_quarantined():
    table, quarantine = build_table(games(), closes(), cutoff="2025-03-01")
    assert "2025-01-01" not in set(table["date"]) | set(quarantine["date"])
