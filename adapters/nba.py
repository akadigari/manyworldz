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
    Orphans are dropped: we never guess a missing side.
    """
    by_id: dict[str, list[dict]] = {}
    for r in rows:
        by_id.setdefault(r["GAME_ID"], []).append(r)

    games = []
    dropped = 0
    for game_id, pair in by_id.items():
        if len(pair) != 2:
            dropped += 1
            continue
        home = next((r for r in pair if "vs." in r["MATCHUP"]), None)
        away = next((r for r in pair if "@" in r["MATCHUP"]), None)
        if home is None or away is None:
            dropped += 1
            continue
        games.append({
            "game_id": game_id,
            "date": home["GAME_DATE"][:10],
            "home": home["TEAM_ABBREVIATION"],
            "away": away["TEAM_ABBREVIATION"],
            "home_pts": int(home["PTS"]),
            "away_pts": int(away["PTS"]),
            "home_won": int(home["PTS"]) > int(away["PTS"]),
        })
    games.sort(key=lambda g: (g["date"], g["game_id"]))
    if dropped:
        print(f"note: dropped {dropped} incomplete game(s) missing a home or away row")
    return games


def fetch_season_results(season: str) -> list[dict]:
    """Fetch one regular season's results, with a disk cache.

    season looks like "2024-25". Cached forever: settled games don't change.
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


def team_history(games: list[dict], team: str, before_date: str) -> list[dict]:
    """Every game this team played strictly before the date, oldest first."""
    return [g for g in games
            if g["date"] < before_date and team in (g["home"], g["away"])]


def _days_between(d1: str, d2: str) -> int:
    """How many days sit between two "YYYY-MM-DD" date strings (d2 - d1)."""
    from datetime import date
    a = date(*map(int, d1.split("-")))
    b = date(*map(int, d2.split("-")))
    return (b - a).days


def _form(history: list[dict], team: str) -> dict:
    """Summarize a team's last 10 games: wins, and average points scored
    and allowed, all seen from that team's own point of view (not home
    vs. away).
    """
    last10 = history[-10:]
    wins, pts_for, pts_against = 0, [], []
    for g in last10:
        we_are_home = g["home"] == team
        our_pts = g["home_pts"] if we_are_home else g["away_pts"]
        their_pts = g["away_pts"] if we_are_home else g["home_pts"]
        pts_for.append(our_pts)
        pts_against.append(their_pts)
        if (our_pts > their_pts):
            wins += 1
    n = max(len(last10), 1)
    return {"last10_wins": wins,
            "avg_pts_for": round(sum(pts_for) / n, 1) if pts_for else 0.0,
            "avg_pts_against": round(sum(pts_against) / n, 1) if pts_against else 0.0}


def build_statsheet(games: list[dict], idx: int) -> dict:
    """Everything an agent may know about a game, from BEFORE tipoff only.

    Assumes `games` is already sorted by date (parse_gamefinder_rows always
    sorts it). team_history() relies on that order, and hist[-1] below only
    grabs the most recent prior game if the list is in date order.
    """
    game = games[idx]
    sheet = {"date": game["date"], "home": game["home"], "away": game["away"]}
    for side in ("home", "away"):
        team = game[side]
        hist = team_history(games, team, game["date"])
        sheet[f"{side}_form"] = _form(hist, team)
        # Rest days capped at 9: "well rested" reads the same for a season
        # opener and a long break, so the number can't fingerprint the date.
        rest = _days_between(hist[-1]["date"], game["date"]) if hist else 9
        sheet[f"{side}_rest_days"] = min(rest, 9)
    return sheet


if __name__ == "__main__":
    # CLI: venv/bin/python adapters/nba.py 2024-25
    season = sys.argv[1]
    games = fetch_season_results(season)
    print(f"{season}: {len(games)} games cached")
