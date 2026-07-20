import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.methods import METHODS, build_methods


def test_six_methods_and_deterministic_cycling():
    assert len(METHODS) == 6
    a = build_methods(8)
    b = build_methods(8)
    assert a == b                       # same n -> same list, every time
    assert [m["label"] for m in a[:6]] == [label for label, _ in METHODS]


def test_seed_is_accepted_and_ignored():
    # Old call sites passed a seed; build_methods still takes one so they
    # don't all need editing, but it never changes the result.
    assert build_methods(6, seed=1) == build_methods(6, seed=99) == build_methods(6)


def test_crowd_cycles_all_six_methods():
    crowd = build_methods(8)
    used = {agent["label"] for agent in crowd}
    assert len(used) == 6  # with 8 agents every method shows up
