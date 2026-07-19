"""Tests for ask.py (the ask-anything door) and the no-market prompt line."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ask import ask_question
from engine.personas import build_crowd
from engine.swarm import agent_vote, market_line


def test_market_line_tells_the_truth_both_ways():
    # A real market card cites its price; a typed question admits there is none.
    assert "43" in market_line({"mid": 43})
    assert "no market price" in market_line({"mid": None})
    assert "no market price" in market_line({})


def test_agent_vote_works_without_a_market_price():
    agent = build_crowd(1, seed=1)[0]
    seen = []

    def ask(prompt, model=None, max_tokens=400):
        seen.append(prompt)
        return '{"probability": 0.7, "reason": "seems likely"}'

    card = {"ticker": "ASK", "question": "Will it rain tomorrow?", "mid": None}
    vote = agent_vote(agent, card, [], ask_fn=ask)
    assert vote["probability"] == 0.7
    assert "no market price" in seen[0]      # the crowd was told the truth
    assert "% chance" not in seen[0]         # and no fake price was invented


def test_ask_question_returns_a_crowd_result_offline():
    confident = '{"probability": 0.8, "reason": "signs point to yes"}'
    result = ask_question("Will the sequel be announced this year?",
                          mode="vote", with_news=False,
                          ask_fn=lambda p, model=None, max_tokens=400: confident)
    assert 0.7 <= result["probability"] <= 0.9
    assert len(result["votes"]) >= 1
    assert result["skipped"] == 0


def test_simulate_mode_works_without_a_market_price():
    # The --simulate path must also tell the crowd the truth (no fake price).
    seen = []
    payload = ('{"futures": ['
               '{"story": "a", "resolves": "YES"},'
               '{"story": "b", "resolves": "NO"},'
               '{"story": "c", "resolves": "YES"},'
               '{"story": "d", "resolves": "YES"}], "reason": "r"}')

    def ask(prompt, model=None, max_tokens=400):
        seen.append(prompt)
        return payload

    result = ask_question("Will the show get renewed?", mode="simulate", k=4,
                          n_agents=2, with_news=False, ask_fn=ask)
    assert result["futures"]                      # rollouts came through
    assert all("no market price" in p for p in seen)
    assert all("% chance" not in p for p in seen)


def test_ask_question_whatif_reports_shift_offline():
    def ask(prompt, model=None, max_tokens=400):
        if "WHAT-IF" in prompt:
            return '{"probability": 0.9, "reason": "the fact settles it"}'
        return '{"probability": 0.4, "reason": "unclear"}'

    result = ask_question("Will the deal close?", whatif="the board approved it",
                          mode="vote", with_news=False, ask_fn=ask)
    assert result["before"]["probability"] == pytest.approx(0.4)
    assert result["after"]["probability"] == pytest.approx(0.9)
    assert result["shift"] == pytest.approx(0.5)
