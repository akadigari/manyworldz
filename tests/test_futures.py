import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.futures import agent_futures
from engine.personas import build_crowd

CARD = {"ticker": "T", "question": "Will the album drop in July?", "mid": 43}
AGENT = build_crowd(1, seed=1)[0]


def ask_with(payload):
    return lambda prompt, model=None, max_tokens=400: payload


def test_probability_is_yes_fraction_of_futures():
    payload = ('{"futures": ['
               '{"story": "surprise midnight drop", "resolves": "YES"},'
               '{"story": "label delays to August", "resolves": "NO"},'
               '{"story": "single first, album July 30", "resolves": "YES"},'
               '{"story": "tour pushes it back", "resolves": "NO"},'
               '{"story": "deluxe drops July 25", "resolves": "YES"}],'
               '"reason": "momentum is real"}')
    out = agent_futures(AGENT, CARD, [], k=5, ask_fn=ask_with(payload))
    assert out["probability"] == pytest.approx(0.6)
    assert len(out["futures"]) == 5
    assert out["futures"][0]["agent"] == AGENT["name"]


def test_too_few_parsed_futures_returns_none():
    payload = '{"futures": [{"story": "only one", "resolves": "YES"}], "reason": "meh"}'
    assert agent_futures(AGENT, CARD, [], k=5, ask_fn=ask_with(payload)) is None


def test_junk_resolves_values_are_dropped():
    payload = ('{"futures": ['
               '{"story": "a", "resolves": "YES"},'
               '{"story": "b", "resolves": "maybe"},'
               '{"story": "c", "resolves": "NO"},'
               '{"story": "d", "resolves": "YES"}], "reason": "r"}')
    out = agent_futures(AGENT, CARD, [], k=4, ask_fn=ask_with(payload))
    assert len(out["futures"]) == 3     # "maybe" dropped
    assert out["probability"] == pytest.approx(2 / 3)
