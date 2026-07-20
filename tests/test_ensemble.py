"""Tests for ensemble mode: different models on different evidence slices.

Everything here runs offline, same as the rest of the suite: canned
ask_fn stand-ins, no network, no real API key. What we're checking is
plumbing, not model behavior: does the right seat get the right slice of
evidence, does the right seat's call go to the right model, and does
CROWD_MODE actually pick between the two crowd builders.
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ask as ask_module
import config
from engine import llm
from engine.ensemble import build_crowd_for, build_ensemble
from engine.futures import agent_futures
from engine.methods import build_methods
from engine.swarm import agent_vote, deliberate, run_crowd

CARD = {"ticker": "T", "question": "Will the album drop in July?", "mid": 55}
HEADLINES = ["Label teases a release date", "Studio session photo leaks"]
CONFIDENT = '{"probability": 0.6, "reason": "ok"}'
FUTURES_PAYLOAD = ('{"futures": ['
                   '{"story": "a surprise drop", "resolves": "YES"},'
                   '{"story": "a delay to August", "resolves": "NO"},'
                   '{"story": "single first, album follows", "resolves": "YES"},'
                   '{"story": "tour eats the date", "resolves": "NO"}], "reason": "r"}')


def capturing_ask(payload):
    """ask_fn that always returns `payload`, and records prompt/model/
    max_tokens for every call it gets."""
    calls = []
    def _ask(prompt, model=None, max_tokens=400):
        calls.append({"prompt": prompt, "model": model, "max_tokens": max_tokens})
        return payload
    return _ask, calls


def capturing_sequence(answers):
    """ask_fn that replays `answers` in order, same recording as above."""
    stack = list(answers)
    calls = []
    def _ask(prompt, model=None, max_tokens=400):
        calls.append({"prompt": prompt, "model": model, "max_tokens": max_tokens})
        return stack.pop(0)
    return _ask, calls


# ---- seat building ----

def test_build_ensemble_default_has_four_seats_matching_config():
    crowd = build_ensemble()
    assert len(crowd) == len(config.ENSEMBLE_SEATS) == 4
    for agent, seat in zip(crowd, config.ENSEMBLE_SEATS):
        assert agent["model"] == seat["model"]
        assert agent["evidence"] == seat["evidence"]


def test_seat_labels_combine_model_and_evidence():
    crowd = build_ensemble([{"model": "sonnet", "evidence": "market"}])
    assert crowd[0]["label"] == "sonnet+market"
    assert crowd[0]["instruction"]      # never blank
    assert isinstance(crowd[0]["instruction"], str)


def test_build_ensemble_reads_custom_seats_not_just_config():
    custom = [{"model": "opus", "evidence": "everything"}]
    crowd = build_ensemble(custom)
    assert len(crowd) == 1
    assert crowd[0]["label"] == "opus+everything"


def test_ensemble_seats_get_distinct_instructions_per_evidence():
    crowd = build_ensemble([
        {"model": "haiku", "evidence": "headlines"},
        {"model": "haiku", "evidence": "base_rates"},
        {"model": "haiku", "evidence": "market"},
        {"model": "haiku", "evidence": "everything"},
    ])
    instructions = {a["instruction"] for a in crowd}
    assert len(instructions) == 4   # every evidence slice reads differently


# ---- evidence slicing at prompt-build time ----

def test_headlines_seat_sees_headlines_not_market():
    seat = build_ensemble([{"model": "haiku", "evidence": "headlines"}])[0]
    ask_fn, calls = capturing_ask(CONFIDENT)
    agent_vote(seat, CARD, HEADLINES, ask_fn=ask_fn)
    prompt = calls[0]["prompt"]
    assert "Label teases a release date" in prompt
    assert "about a 55%" not in prompt


def test_base_rates_seat_sees_neither_headlines_nor_market():
    seat = build_ensemble([{"model": "sonnet", "evidence": "base_rates"}])[0]
    ask_fn, calls = capturing_ask(CONFIDENT)
    agent_vote(seat, CARD, HEADLINES, ask_fn=ask_fn)
    prompt = calls[0]["prompt"]
    assert "Label teases a release date" not in prompt
    assert "about a 55%" not in prompt
    assert "how often" in prompt.lower()   # told to lean on base rates instead


def test_market_seat_sees_the_price_line_not_headlines():
    seat = build_ensemble([{"model": "haiku", "evidence": "market"}])[0]
    ask_fn, calls = capturing_ask(CONFIDENT)
    agent_vote(seat, CARD, HEADLINES, ask_fn=ask_fn)
    prompt = calls[0]["prompt"]
    assert "about a 55%" in prompt
    assert "Label teases a release date" not in prompt


def test_everything_seat_sees_both():
    seat = build_ensemble([{"model": "sonnet", "evidence": "everything"}])[0]
    ask_fn, calls = capturing_ask(CONFIDENT)
    agent_vote(seat, CARD, HEADLINES, ask_fn=ask_fn)
    prompt = calls[0]["prompt"]
    assert "about a 55%" in prompt
    assert "Label teases a release date" in prompt


def test_methods_mode_agent_still_gets_everything_by_default():
    # A plain methods-mode agent has no "evidence" key at all: it must
    # keep seeing the full view, exactly like before ensemble existed.
    agent = build_methods(1)[0]
    ask_fn, calls = capturing_ask(CONFIDENT)
    agent_vote(agent, CARD, HEADLINES, ask_fn=ask_fn)
    prompt = calls[0]["prompt"]
    assert "about a 55%" in prompt
    assert "Label teases a release date" in prompt


def test_evidence_slicing_also_applies_in_simulate_mode():
    seat = build_ensemble([{"model": "haiku", "evidence": "market"}])[0]
    ask_fn, calls = capturing_ask(FUTURES_PAYLOAD)
    agent_futures(seat, CARD, HEADLINES, k=4, ask_fn=ask_fn)
    prompt = calls[0]["prompt"]
    assert "about a 55%" in prompt
    assert "Label teases a release date" not in prompt
    assert calls[0]["max_tokens"] == 200 + 80 * 4   # formula untouched


def test_deliberation_hides_market_from_a_headlines_seat():
    # Round two must not suddenly show a narrow seat evidence it never had
    # in round one.
    crowd = build_ensemble([
        {"model": "haiku", "evidence": "headlines"},
        {"model": "sonnet", "evidence": "everything"},
    ])
    answers = ['{"probability": 0.3, "reason": "low"}',
               '{"probability": 0.7, "reason": "high"}',
               '{"probability": 0.35, "reason": "still low"}',
               '{"probability": 0.65, "reason": "still high"}']
    ask_fn, calls = capturing_sequence(answers)
    run_crowd(CARD, HEADLINES, crowd, mode="vote", k=0, deliberation=True, ask_fn=ask_fn)
    delib_prompts = [c["prompt"] for c in calls[2:]]
    assert "about a 55%" not in delib_prompts[0]   # the headlines seat's turn
    assert "about a 55%" in delib_prompts[1]        # the everything seat's turn


# ---- model routing ----

def test_agent_vote_threads_seat_model_into_ask_fn():
    seat = build_ensemble([{"model": "sonnet", "evidence": "everything"}])[0]
    ask_fn, calls = capturing_ask(CONFIDENT)
    agent_vote(seat, CARD, HEADLINES, ask_fn=ask_fn)
    assert calls[0]["model"] == "sonnet"


def test_methods_agent_passes_model_none_through():
    agent = build_methods(1)[0]
    ask_fn, calls = capturing_ask(CONFIDENT)
    agent_vote(agent, CARD, HEADLINES, ask_fn=ask_fn)
    assert calls[0]["model"] is None


def test_agent_futures_threads_seat_model_into_ask_fn():
    seat = build_ensemble([{"model": "opus", "evidence": "everything"}])[0]
    ask_fn, calls = capturing_ask(FUTURES_PAYLOAD)
    agent_futures(seat, CARD, HEADLINES, k=4, ask_fn=ask_fn)
    assert calls[0]["model"] == "opus"


def test_deliberate_threads_seat_model_into_ask_fn():
    seat = build_ensemble([{"model": "fable", "evidence": "everything"}])[0]
    own = {"probability": 0.4, "reason": "first take"}
    ask_fn, calls = capturing_ask(CONFIDENT)
    deliberate(seat, CARD, own, ["- other: 0.5 (x)"], ask_fn=ask_fn)
    assert calls[0]["model"] == "fable"


def test_run_crowd_ensemble_threads_each_seats_own_model_in_order():
    crowd = build_ensemble()   # default four seats
    ask_fn, calls = capturing_ask(CONFIDENT)
    run_crowd(CARD, HEADLINES, crowd, mode="vote", k=0, deliberation=False, ask_fn=ask_fn)
    seen_models = [c["model"] for c in calls]
    expected_models = [seat["model"] for seat in config.ENSEMBLE_SEATS]
    assert seen_models == expected_models


def test_run_crowd_simulate_mode_threads_model_via_agent_futures():
    crowd = build_ensemble([{"model": "opus", "evidence": "market"}])
    ask_fn, calls = capturing_ask(FUTURES_PAYLOAD)
    run_crowd(CARD, HEADLINES, crowd, mode="simulate", k=4, deliberation=False, ask_fn=ask_fn)
    assert calls[0]["model"] == "opus"


# ---- CROWD_MODE switch ----

def test_build_crowd_for_methods_mode(monkeypatch):
    monkeypatch.setattr(config, "CROWD_MODE", "methods")
    crowd = build_crowd_for(6)
    assert len(crowd) == 6
    assert "model" not in crowd[0]


def test_build_crowd_for_ensemble_mode(monkeypatch):
    monkeypatch.setattr(config, "CROWD_MODE", "ensemble")
    crowd = build_crowd_for()
    assert len(crowd) == len(config.ENSEMBLE_SEATS)
    assert crowd[0]["model"] == config.ENSEMBLE_SEATS[0]["model"]


def test_build_crowd_for_explicit_override_beats_config(monkeypatch):
    monkeypatch.setattr(config, "CROWD_MODE", "methods")
    crowd = build_crowd_for(crowd_mode="ensemble")
    assert len(crowd) == len(config.ENSEMBLE_SEATS)


def test_build_crowd_for_ensemble_ignores_n_agents(monkeypatch):
    monkeypatch.setattr(config, "CROWD_MODE", "ensemble")
    crowd = build_crowd_for(n_agents=99)
    assert len(crowd) == len(config.ENSEMBLE_SEATS)   # seats, not n_agents, size it


def test_env_var_overrides_crowd_mode(monkeypatch):
    monkeypatch.setenv("MANYWORLDZ_CROWD_MODE", "ensemble")
    importlib.reload(config)
    try:
        assert config.CROWD_MODE == "ensemble"
    finally:
        monkeypatch.delenv("MANYWORLDZ_CROWD_MODE", raising=False)
        importlib.reload(config)
        assert config.CROWD_MODE == "methods"


# ---- CLI --crowd override ----

def test_cli_crowd_flag_overrides_config_for_one_run(monkeypatch):
    captured = {}
    def fake_ask_question(question, whatif=None, mode="simulate", k=None,
                          n_agents=None, with_news=True, ask_fn=None,
                          model=None, crowd_mode=None):
        captured["crowd_mode"] = crowd_mode
        return {"probability": 0.5, "spread": 0.0, "votes": [],
                "futures": [], "skipped": 0}
    monkeypatch.setattr(ask_module, "ask_question", fake_ask_question)
    monkeypatch.setattr(sys, "argv",
                        ["ask.py", "Will it happen?", "--crowd", "ensemble", "--no-news"])
    ask_module.main()
    assert captured["crowd_mode"] == "ensemble"


def test_cli_without_crowd_flag_passes_none_through(monkeypatch):
    captured = {}
    def fake_ask_question(question, whatif=None, mode="simulate", k=None,
                          n_agents=None, with_news=True, ask_fn=None,
                          model=None, crowd_mode=None):
        captured["crowd_mode"] = crowd_mode
        return {"probability": 0.5, "spread": 0.0, "votes": [],
                "futures": [], "skipped": 0}
    monkeypatch.setattr(ask_module, "ask_question", fake_ask_question)
    monkeypatch.setattr(sys, "argv", ["ask.py", "Will it happen?", "--no-news"])
    ask_module.main()
    assert captured["crowd_mode"] is None   # config default decides, not forced


def test_ask_question_end_to_end_with_ensemble_crowd_offline():
    result = ask_module.ask_question(
        "Will the sequel be announced this year?", mode="vote",
        with_news=False, crowd_mode="ensemble",
        ask_fn=lambda p, model=None, max_tokens=400: CONFIDENT)
    assert len(result["votes"]) == len(config.ENSEMBLE_SEATS)
    labels = {v["agent"] for v in result["votes"]}
    assert labels == {f'{s["model"]}+{s["evidence"]}' for s in config.ENSEMBLE_SEATS}


# ---- budget propagation ----

def test_budget_error_propagates_through_ensemble_crowd():
    crowd = build_ensemble()
    calls = {"n": 0}
    def blow_budget(prompt, model=None, max_tokens=400):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("engine budget cap hit ($10.00)")
        return CONFIDENT
    with pytest.raises(RuntimeError, match="budget cap hit"):
        run_crowd(CARD, HEADLINES, crowd, mode="vote", k=0,
                 deliberation=False, ask_fn=blow_budget)
    assert calls["n"] == 2   # the second seat's call is the one that blew up


# ---- spend safety ----

def test_default_ensemble_models_have_known_per_model_prices():
    # Mixed-model ensembles only meter spend correctly if every seat's
    # model resolves to a price _PRICES actually knows, not the frontier
    # default fallback.
    for seat in config.ENSEMBLE_SEATS:
        resolved = llm.resolve_model(seat["model"])
        assert resolved in llm._PRICES, (
            f'{seat["model"]!r} -> {resolved!r} has no known price: '
            "spend would silently be metered at the frontier default")


def test_spend_meter_charges_the_right_seat_model(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm")
    monkeypatch.setattr(llm, "SPEND_FILE", tmp_path / "spend.json")
    seen_models = []
    def fake_api(prompt, model, max_tokens):
        seen_models.append(model)
        return "x", 1_000_000, 0
    monkeypatch.setattr(llm, "_call_api", fake_api)
    llm.ask("q1", model="haiku")
    llm.ask("q2", model="sonnet")
    assert seen_models == ["claude-haiku-4-5", "claude-sonnet-5"]
    # haiku: 1M input tokens = $1.00, sonnet: 1M input tokens = $3.00
    assert llm.spent_usd() == pytest.approx(4.00, abs=0.01)
