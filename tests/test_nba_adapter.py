import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.nba import parse_gamefinder_rows

FIXTURE = Path(__file__).parent / "fixtures" / "gamefinder_rows.json"


def rows():
    return json.loads(FIXTURE.read_text())


def test_pairs_two_rows_into_one_game():
    games = parse_gamefinder_rows(rows())
    complete = [g for g in games if g["game_id"] == "0022400001"]
    assert len(complete) == 1
    g = complete[0]
    assert g["home"] == "BOS" and g["away"] == "NYK"
    assert g["home_pts"] == 132 and g["away_pts"] == 109
    assert g["home_won"] is True
    assert g["date"] == "2024-10-22"


def test_orphan_rows_are_dropped_not_guessed():
    games = parse_gamefinder_rows(rows())
    assert all(g["game_id"] != "0022400999" for g in games)


def test_output_sorted_by_date():
    games = parse_gamefinder_rows(rows() + rows())  # dupes exercise stability
    dates = [g["date"] for g in games]
    assert dates == sorted(dates)
