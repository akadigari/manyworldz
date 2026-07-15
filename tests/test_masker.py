import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masker import BANNED_TOKENS, NBA_TEAMS, mask_statsheet


def sheet_for(home, away):
    return {"date": "2025-01-15", "home": home, "away": away,
            "home_form": {"last10_wins": 7, "avg_pts_for": 118.2, "avg_pts_against": 110.0},
            "away_form": {"last10_wins": 3, "avg_pts_for": 105.4, "avg_pts_against": 114.9},
            "home_rest_days": 2, "away_rest_days": 1}


def test_thirty_teams_and_no_identity_survives():
    assert len(NBA_TEAMS) == 30
    abbrevs = [t[0] for t in NBA_TEAMS]
    for i in range(0, 30, 2):  # 15 masked sheets covers all 30 teams
        text = mask_statsheet(sheet_for(abbrevs[i], abbrevs[i + 1])).lower()
        for token in BANNED_TOKENS:
            assert not re.search(rf"\b{re.escape(token.lower())}\b", text), \
                f"banned token '{token}' leaked for {abbrevs[i]} vs {abbrevs[i+1]}"


def test_exact_date_is_hidden():
    text = mask_statsheet(sheet_for("BOS", "NYK"))
    assert "2025-01-15" not in text and "January)" in text


def test_teams_become_a_and_b():
    text = mask_statsheet(sheet_for("BOS", "NYK"))
    assert "Team A" in text and "Team B" in text
