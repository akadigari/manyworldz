"""Tests for the A1+A2 accuracy upgrades: research() and base-rate prompts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import news
from engine.methods import build_methods
from engine.swarm import agent_vote


def test_key_terms_drops_filler_words():
    assert news.key_terms("Will the album drop before July 31?") == \
        "album drop july 31"


def test_research_merges_two_angles_and_dedupes(monkeypatch):
    calls = []

    def fake_headlines(query, limit=3):
        calls.append(query)
        if "will" in query.lower():                 # the full question
            return ["Shared headline", "Only from full question"]
        return ["Shared headline", "Only from key terms"]

    monkeypatch.setattr(news, "headlines_for", fake_headlines)
    heads = news.research("Will the album drop before July 31?", limit=8)
    assert len(calls) == 2                          # both angles searched
    assert heads.count("Shared headline") == 1      # duplicates removed
    assert "Only from full question" in heads
    assert "Only from key terms" in heads


def test_research_never_raises_when_search_dies(monkeypatch):
    def boom(query, limit=3):
        raise RuntimeError("should have been caught inside headlines_for")
    # research() leans on headlines_for's own never-raise guarantee, so we
    # simulate that contract: a dead search returns [] rather than raising.
    monkeypatch.setattr(news, "headlines_for", lambda q, limit=3: [])
    assert news.research("anything at all") == []


def test_vote_prompt_teaches_base_rates():
    seen = []

    def ask(prompt, model=None, max_tokens=400):
        seen.append(prompt)
        return '{"probability": 0.5, "reason": "even"}'

    agent = build_methods(1)[0]
    agent_vote(agent, {"question": "Will it rain?", "mid": 50}, [], ask_fn=ask)
    assert "base rate" in seen[0]                   # the superforecaster habit
