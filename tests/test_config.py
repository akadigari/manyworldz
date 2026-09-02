import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def test_gate_numbers_match_the_spec():
    assert config.GO_MIN_GAMES == 350
    assert config.GO_MAX_JOIN_ERROR == 0.01
    assert config.REID_DEMOTION_RATE == 0.10
    assert config.SEED == 14000605


def test_cutoff_date_is_a_plain_string():
    assert isinstance(config.MODEL_CUTOFF_DATE, str)
    assert len(config.MODEL_CUTOFF_DATE) == 10  # YYYY-MM-DD


def test_engine_budget_is_env_overridable(monkeypatch):
    """The monthly budget must be raisable from the workflow without a
    code change: the tournament's question volume is Metaculus's call,
    not ours."""
    import importlib
    import config as config_module
    monkeypatch.setenv("ENGINE_BUDGET_USD", "75")
    importlib.reload(config_module)
    try:
        assert config_module.ENGINE_BUDGET_USD == 75.0
    finally:
        monkeypatch.delenv("ENGINE_BUDGET_USD")
        importlib.reload(config_module)


def test_tournament_slugs_cover_minibench_and_the_fall_season():
    """2026-09-02: the bot enters the Fall FutureEval season alongside
    MiniBench. MiniBench stays primary (backtest and probe tooling key
    off METACULUS_TOURNAMENT)."""
    assert config.METACULUS_TOURNAMENTS[0] == "minibench"
    assert any("fall" in slug for slug in config.METACULUS_TOURNAMENTS)
    assert config.METACULUS_TOURNAMENT == config.METACULUS_TOURNAMENTS[0]


def test_tournament_slug_list_is_env_overridable(monkeypatch):
    import importlib
    import config as config_module
    monkeypatch.setenv("METACULUS_TOURNAMENTS", " a, b ,")
    importlib.reload(config_module)
    try:
        assert config_module.METACULUS_TOURNAMENTS == ["a", "b"]
        assert config_module.METACULUS_TOURNAMENT == "a"
    finally:
        monkeypatch.delenv("METACULUS_TOURNAMENTS")
        importlib.reload(config_module)


def test_singular_tournament_env_still_works(monkeypatch):
    """The pre-2026-09 override spelling keeps working: one slug in
    METACULUS_TOURNAMENT narrows the whole run to that tournament."""
    import importlib
    import config as config_module
    monkeypatch.setenv("METACULUS_TOURNAMENT", "solo-slug")
    importlib.reload(config_module)
    try:
        assert config_module.METACULUS_TOURNAMENTS == ["solo-slug"]
        assert config_module.METACULUS_TOURNAMENT == "solo-slug"
    finally:
        monkeypatch.delenv("METACULUS_TOURNAMENT")
        importlib.reload(config_module)
