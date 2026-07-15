import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.personas import build_crowd
from engine.swarm import agent_vote, consensus, extract_json, run_crowd

CARD = {"ticker": "T", "question": "Will the album drop in July?", "mid": 43,
        "close_time": "2026-07-31T23:59:00Z", "yes_bid": 40, "yes_ask": 46,
        "volume": 5200, "category": "Entertainment"}


def canned_ask(answers):
    """ask_fn that replays a list of canned model answers in order."""
    stack = list(answers)
    def _ask(prompt, model=None, max_tokens=400):
        return stack.pop(0)
    return _ask


def test_extract_json_finds_object_in_noise():
    assert extract_json('sure! {"probability": 0.7, "reason": "hype"} done')["probability"] == 0.7
    assert extract_json("no json here") is None


def test_agent_vote_parses_and_clamps():
    agent = build_crowd(1, seed=1)[0]
    ask = canned_ask(['{"probability": 1.7, "reason": "too sure"}'])
    vote = agent_vote(agent, CARD, ["headline"], ask_fn=ask)
    assert vote["probability"] == 0.99  # clamped into [0.01, 0.99]


def test_agent_vote_returns_none_on_junk():
    agent = build_crowd(1, seed=1)[0]
    assert agent_vote(agent, CARD, [], ask_fn=canned_ask(["garbage"])) is None


def test_consensus_trims_extremes_with_five_plus():
    prob, spread = consensus([0.01, 0.5, 0.5, 0.5, 0.99])
    assert prob == pytest.approx(0.5)   # extremes dropped
    assert spread > 0


def test_run_crowd_vote_mode_counts_skips():
    crowd = build_crowd(4, seed=1)
    answers = ['{"probability": 0.6, "reason": "a"}',
               '{"probability": 0.7, "reason": "b"}',
               "junk",
               '{"probability": 0.5, "reason": "c"}']
    out = run_crowd(CARD, [], crowd, mode="vote", k=0,
                    deliberation=False, ask_fn=canned_ask(answers))
    assert out["skipped"] == 1
    assert len(out["votes"]) == 3
    assert 0.5 <= out["probability"] <= 0.7


def test_deliberation_second_round_updates():
    crowd = build_crowd(2, seed=1)
    answers = ['{"probability": 0.2, "reason": "low"}',
               '{"probability": 0.8, "reason": "high"}',
               # deliberation round answers:
               '{"probability": 0.4, "reason": "moved up"}',
               '{"probability": 0.6, "reason": "moved down"}']
    out = run_crowd(CARD, [], crowd, mode="vote", k=0,
                    deliberation=True, ask_fn=canned_ask(answers))
    probs = sorted(v["probability"] for v in out["votes"])
    assert probs == [0.4, 0.6]  # final numbers are the revised ones
