import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.personas import ARCHETYPES, build_crowd


def test_six_archetypes_and_deterministic_crowd():
    assert len(ARCHETYPES) == 6
    a = build_crowd(8, seed=14000605)
    b = build_crowd(8, seed=14000605)
    assert a == b                       # same seed -> same crowd
    assert len({agent["name"] for agent in a}) == 8  # names unique


def test_crowd_cycles_all_archetypes():
    crowd = build_crowd(8, seed=1)
    used = {agent["archetype"] for agent in crowd}
    assert len(used) == 6  # with 8 agents every archetype shows up
