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
