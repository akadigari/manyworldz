"""Load verified market closing prices for NBA games.

The Kaggle CSV is downloaded by hand into data/ (see the plan's execution
task). We validate its columns loudly instead of assuming — if the real
file names differ, edit COLUMN_MAP and nothing else.
"""
from __future__ import annotations

import re
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
# Some datasets write the Clippers with the official league city ("LA"
# Clippers) even though the more common way people write it is "Los
# Angeles Clippers" — accept that spelling too.
_FULLNAME_TO_ABBREV["los angeles clippers"] = "LAC"

_DATE_FORMAT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


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

    # Catch a wrong date format early and loudly, instead of silently
    # producing garbage "date" strings later (a "03/05/2025"-style column
    # would slice down to "03/05/2" and nobody would notice).
    date_col = raw[COLUMN_MAP["date"]].dropna()
    if len(date_col):
        first_date = str(date_col.iloc[0])
        if not _DATE_FORMAT_RE.match(first_date):
            raise ValueError(
                f"column '{COLUMN_MAP['date']}' doesn't look like YYYY-MM-DD. "
                f"First value found: {first_date!r}. Dates must be in "
                f"YYYY-MM-DD format — reformat the CSV before loading.")

    rows = []
    dropped = 0
    bad_team_names: list[str] = []

    def _note_bad_name(raw_name) -> None:
        name_str = str(raw_name).strip()
        if name_str and name_str.lower() != "nan" and name_str not in bad_team_names \
                and len(bad_team_names) < 10:
            bad_team_names.append(name_str)

    for _, r in raw.iterrows():
        home_name, away_name = r[COLUMN_MAP["home"]], r[COLUMN_MAP["away"]]
        home, away = _to_abbrev(home_name), _to_abbrev(away_name)
        odds_ok = True
        try:
            p_home = devig(american_to_prob(r[COLUMN_MAP["home_ml"]]),
                           american_to_prob(r[COLUMN_MAP["away_ml"]]))
        except (ValueError, TypeError):
            odds_ok = False

        if home is None:
            _note_bad_name(home_name)
        if away is None:
            _note_bad_name(away_name)

        if home is None or away is None or not odds_ok:
            dropped += 1
            continue
        rows.append({"date": str(r[COLUMN_MAP["date"]])[:10], "home": home,
                     "away": away, "home_close_prob": round(p_home, 4),
                     "provenance": "kaggle"})

    if dropped:
        print(f"kept {len(rows)} rows, dropped {dropped} "
              f"(unmatched team names: {bad_team_names})")

    return pd.DataFrame(rows, columns=["date", "home", "away",
                                       "home_close_prob", "provenance"])
