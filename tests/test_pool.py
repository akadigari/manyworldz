"""Tests for the diversity pool: a crowd where every agent blends an
independent method, temperament, and lens, so hundreds or thousands of
agents can all think differently instead of repeating a handful of voices.

Everything here runs offline, same as the rest of the suite: no network,
no real API key, canned ask_fn stand-ins only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from engine.ensemble import build_crowd_for
from engine.methods import METHODS
from engine.pool import LENSES, POOL_MAX_DISTINCT, TEMPERAMENTS, build_pool_crowd
from engine.swarm import run_crowd


def canned_ask(payload):
    """ask_fn that always returns `payload`, no matter the prompt."""
    def _ask(prompt, model=None, max_tokens=400):
        return payload
    return _ask


# ---- the three pools themselves ----

def test_methods_pool_is_the_existing_six():
    assert len(METHODS) == 6


def test_temperament_pool_has_about_ten_labeled_pairs():
    assert len(TEMPERAMENTS) == 10
    for label, instruction in TEMPERAMENTS:
        assert isinstance(label, str) and label
        assert isinstance(instruction, str) and instruction


def test_lens_pool_has_about_eight_labeled_pairs():
    assert len(LENSES) == 8
    for label, instruction in LENSES:
        assert isinstance(label, str) and label
        assert isinstance(instruction, str) and instruction


def test_ceiling_is_the_product_of_the_three_pool_sizes():
    assert POOL_MAX_DISTINCT == len(METHODS) * len(TEMPERAMENTS) * len(LENSES)
    assert POOL_MAX_DISTINCT == 480


# ---- distinctness up to the ceiling ----

def test_first_ceiling_agents_are_all_distinct_combinations():
    crowd = build_pool_crowd(POOL_MAX_DISTINCT)
    labels = [agent["label"] for agent in crowd]
    assert len(labels) == POOL_MAX_DISTINCT
    assert len(set(labels)) == POOL_MAX_DISTINCT   # no repeat before the ceiling


def test_a_crowd_smaller_than_the_ceiling_is_still_all_distinct():
    crowd = build_pool_crowd(150)
    labels = [agent["label"] for agent in crowd]
    assert len(set(labels)) == 150


# ---- determinism ----

def test_same_n_twice_gives_identical_crowds():
    a = build_pool_crowd(50)
    b = build_pool_crowd(50)
    assert a == b


def test_default_seed_matches_explicit_config_seed():
    a = build_pool_crowd(50)
    b = build_pool_crowd(50, seed=config.SEED)
    assert a == b


def test_explicit_seed_is_deterministic_too():
    a = build_pool_crowd(80, seed=7)
    b = build_pool_crowd(80, seed=7)
    assert a == b


# ---- wrap-around past the ceiling ----

def test_wrap_around_past_the_ceiling_returns_n_agents_and_does_not_crash():
    n = POOL_MAX_DISTINCT + 37
    crowd = build_pool_crowd(n)
    assert len(crowd) == n


def test_wrap_around_repeats_the_same_order_after_the_ceiling():
    crowd = build_pool_crowd(POOL_MAX_DISTINCT + 5)
    for i in range(5):
        assert crowd[i] == crowd[POOL_MAX_DISTINCT + i]


def test_zero_agents_returns_an_empty_list():
    assert build_pool_crowd(0) == []


# ---- agent dict shape ----

def test_every_agent_has_label_and_instruction():
    crowd = build_pool_crowd(20)
    for agent in crowd:
        assert isinstance(agent["label"], str) and agent["label"]
        assert isinstance(agent["instruction"], str) and agent["instruction"]


def test_instruction_mentions_its_own_method_temperament_and_lens():
    crowd = build_pool_crowd(POOL_MAX_DISTINCT)
    for agent in crowd:
        method_label, temperament_label, lens_label = agent["label"].split(" / ")
        assert method_label in agent["instruction"]
        assert temperament_label in agent["instruction"]
        assert lens_label in agent["instruction"]


def test_base_rate_anchoring_present_in_every_composed_instruction():
    crowd = build_pool_crowd(POOL_MAX_DISTINCT)
    for agent in crowd:
        assert "base rate" in agent["instruction"].lower()


# ---- flows through run_crowd like any other crowd ----

def test_small_pool_crowd_flows_through_run_crowd_offline():
    crowd = build_pool_crowd(5)
    card = {"ticker": "ASK", "question": "Will it happen?", "mid": None}
    ask_fn = canned_ask('{"probability": 0.6, "reason": "ok"}')
    result = run_crowd(card, [], crowd, mode="vote", k=0,
                       deliberation=False, ask_fn=ask_fn)
    assert 0.0 <= result["probability"] <= 1.0
    assert "spread" in result
    assert len(result["votes"]) == 5
    assert result["skipped"] == 0


# ---- build_crowd_for dispatch and the --agents -> n mapping ----

def test_build_crowd_for_pool_branch_uses_n_agents():
    crowd = build_crowd_for(n_agents=25, crowd_mode="pool")
    assert len(crowd) == 25
    for agent in crowd:
        assert agent["label"].count(" / ") == 2


def test_build_crowd_for_pool_defaults_to_engine_n_agents_when_n_is_none():
    crowd = build_crowd_for(n_agents=None, crowd_mode="pool")
    assert len(crowd) == config.ENGINE_N_AGENTS


def test_ask_question_crowd_pool_maps_agents_flag_to_distinct_agents():
    from ask import ask_question
    ask_fn = canned_ask('{"probability": 0.5, "reason": "ok"}')
    result = ask_question("Will it happen?", mode="vote", n_agents=300,
                          with_news=False, crowd_mode="pool", ask_fn=ask_fn)
    assert len(result["votes"]) == 300


def test_ask_crowd_choice_list_includes_pool():
    import ask as ask_module
    src = Path(ask_module.__file__).read_text()
    assert '"pool"' in src
