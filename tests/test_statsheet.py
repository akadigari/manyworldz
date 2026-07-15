import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.nba import build_statsheet, team_history


def little_season():
    """Three games: BOS beats NYK, then LAL beats BOS two days later."""
    return [
        {"game_id": "1", "date": "2025-01-01", "home": "BOS", "away": "NYK",
         "home_pts": 120, "away_pts": 100, "home_won": True},
        {"game_id": "2", "date": "2025-01-03", "home": "LAL", "away": "BOS",
         "home_pts": 111, "away_pts": 99, "home_won": True},
    ]


def test_history_only_looks_backward():
    games = little_season()
    h = team_history(games, "BOS", before_date="2025-01-03")
    assert len(h) == 1 and h[0]["game_id"] == "1"


def test_statsheet_form_and_rest():
    games = little_season()
    sheet = build_statsheet(games, 1)  # the Jan 3 game
    assert sheet["home"] == "LAL" and sheet["away"] == "BOS"
    assert sheet["away_form"]["last10_wins"] == 1
    assert sheet["away_form"]["avg_pts_for"] == 120.0
    assert sheet["away_rest_days"] == 2
    assert sheet["home_rest_days"] == 9  # no prior game -> capped, not a leak
