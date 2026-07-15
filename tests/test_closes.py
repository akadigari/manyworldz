import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from markets.closes import american_to_prob, devig, load_kaggle_closes

FIXTURE = Path(__file__).parent / "fixtures" / "closes_sample.csv"


def test_american_odds_to_probability():
    assert american_to_prob(-220) == pytest.approx(0.6875, abs=1e-4)
    assert american_to_prob(+180) == pytest.approx(0.3571, abs=1e-4)


def test_devig_makes_the_pair_sum_to_one():
    p = devig(american_to_prob(-220), american_to_prob(+180))
    assert 0.65 < p < 0.67  # fair home prob, vig stripped


def test_loader_normalizes_names_and_drops_bad_rows():
    df = load_kaggle_closes(FIXTURE)
    assert list(df.columns) == ["date", "home", "away", "home_close_prob", "provenance"]
    assert len(df) == 2                      # the row with a blank team is dropped
    assert set(df["home"]) == {"BOS", "LAL"}  # full names became abbrevs
    assert ((df["home_close_prob"] > 0) & (df["home_close_prob"] < 1)).all()
    assert (df["provenance"] == "kaggle").all()


def test_los_angeles_clippers_full_name_resolves_to_lac(tmp_path):
    # Datasets often write "Los Angeles Clippers" even though the league's
    # own city field for that team is just "LA" — both must resolve to LAC.
    csv_path = tmp_path / "clippers.csv"
    csv_path.write_text(
        "game_date,home_team,away_team,home_ml_close,away_ml_close\n"
        "2025-03-05,Los Angeles Clippers,Boston Celtics,-150,+130\n"
    )
    df = load_kaggle_closes(csv_path)
    assert list(df["home"]) == ["LAC"]


def test_wrong_date_format_raises_value_error(tmp_path):
    csv_path = tmp_path / "bad_dates.csv"
    csv_path.write_text(
        "game_date,home_team,away_team,home_ml_close,away_ml_close\n"
        "03/05/2025,Boston Celtics,New York Knicks,-220,+180\n"
    )
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        load_kaggle_closes(csv_path)
