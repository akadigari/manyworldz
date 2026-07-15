"""Pull NBA game results and turn them into simple game records.

The only network call lives in fetch_season_results(); everything else is
pure parsing so tests never need the internet.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def parse_gamefinder_rows(rows: list[dict]) -> list[dict]:
    """Combine per-team rows into one record per game.

    A game only counts when we see BOTH of its rows (home and away).
    Orphans are dropped — we never guess a missing side.
    """
    by_id: dict[str, list[dict]] = {}
    for r in rows:
        by_id.setdefault(r["GAME_ID"], []).append(r)

    games = []
    for gid, pair in by_id.items():
        if len(pair) != 2:
            continue
        home = next((r for r in pair if "vs." in r["MATCHUP"]), None)
        away = next((r for r in pair if "@" in r["MATCHUP"]), None)
        if home is None or away is None:
            continue
        games.append({
            "game_id": gid,
            "date": home["GAME_DATE"][:10],
            "home": home["TEAM_ABBREVIATION"],
            "away": away["TEAM_ABBREVIATION"],
            "home_pts": int(home["PTS"]),
            "away_pts": int(away["PTS"]),
            "home_won": int(home["PTS"]) > int(away["PTS"]),
        })
    games.sort(key=lambda g: (g["date"], g["game_id"]))
    return games


def fetch_season_results(season: str) -> list[dict]:
    """Fetch one regular season's results, with a disk cache.

    season looks like "2024-25". Cached forever — settled games don't change.
    """
    config.CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = config.CACHE / f"results_{season}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    from nba_api.stats.endpoints import leaguegamefinder
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
    )
    rows = finder.get_normalized_dict()["LeagueGameFinderResults"]
    time.sleep(1)  # be polite to the free endpoint
    games = parse_gamefinder_rows(rows)
    cache_file.write_text(json.dumps(games, indent=1))
    return games


if __name__ == "__main__":
    # CLI: venv/bin/python adapters/nba.py 2024-25
    season = sys.argv[1]
    games = fetch_season_results(season)
    print(f"{season}: {len(games)} games cached")
