"""Load verified market closing prices for NBA games.

The Kaggle CSV is downloaded by hand into data/ (see the plan's execution
task). We validate its columns loudly instead of assuming — if the real
file names differ, edit COLUMN_MAP and nothing else.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masker import NBA_TEAMS

# our name -> the column we expect in the raw CSV. One dict to edit.
COLUMN_MAP = {
    "date": "game_date",
    "home": "home_team",
    "away": "away_team",
    "home_ml": "home_ml_close",
    "away_ml": "away_ml_close",
}

_FULLNAME_TO_ABBREV = {f"{t[1]} {t[2]}".lower(): t[0] for t in NBA_TEAMS}


def american_to_prob(ml: int) -> float:
    """American moneyline -> raw implied probability (still has vig)."""
    ml = int(ml)
    if ml < 0:
        return -ml / (-ml + 100)
    return 100 / (ml + 100)


def devig(p_home_raw: float, p_away_raw: float) -> float:
    """Strip the bookmaker's margin: scale the pair to sum to 1."""
    return p_home_raw / (p_home_raw + p_away_raw)


def validate_schema(df: pd.DataFrame) -> None:
    missing = [v for v in COLUMN_MAP.values() if v not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing expected columns {missing}. "
            f"Found columns: {list(df.columns)}. Fix COLUMN_MAP in closes.py.")


def _to_abbrev(name: str) -> str | None:
    return _FULLNAME_TO_ABBREV.get(str(name).strip().lower())


def load_kaggle_closes(csv_path: Path) -> pd.DataFrame:
    """Return one clean row per game: date, home, away, home_close_prob."""
    raw = pd.read_csv(csv_path)
    validate_schema(raw)
    rows = []
    for _, r in raw.iterrows():
        home, away = _to_abbrev(r[COLUMN_MAP["home"]]), _to_abbrev(r[COLUMN_MAP["away"]])
        try:
            p_home = devig(american_to_prob(r[COLUMN_MAP["home_ml"]]),
                           american_to_prob(r[COLUMN_MAP["away_ml"]]))
        except (ValueError, TypeError):
            home = None  # bad odds -> drop below
        if home is None or away is None:
            continue
        rows.append({"date": str(r[COLUMN_MAP["date"]])[:10], "home": home,
                     "away": away, "home_close_prob": round(p_home, 4),
                     "provenance": "kaggle"})
    return pd.DataFrame(rows, columns=["date", "home", "away",
                                       "home_close_prob", "provenance"])
