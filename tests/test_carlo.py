"""Tests for engine/carlo.py, the monte carlo fusion layer, and ask.py's
--carlo flag.

Everything here runs offline, same as the rest of the suite: canned
ask_fn stand-ins, no network, no real API key. Determinism means every
number below is exactly reproducible: the same elicited beliefs, draws
count, and config.SEED always produce the exact same roll, so several
tests below hardcode the actual value measured from a real run instead
of a loose range.
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ask as ask_module
import config
from engine.carlo import elicit, roll, run_carlo
from engine.methods import build_methods

CARD = {"ticker": "ASK", "question": "Will the album drop in July?", "mid": None}
AGENT = {"label": "base rates", "instruction": "start from how often this happens"}


def canned_ask(answers):
    """ask_fn that replays a list of canned model answers in order."""
    stack = list(answers)
    def _ask(prompt, model=None, max_tokens=400):
        return stack.pop(0)
    return _ask


# ---- elicit(): reuses vote machinery, junk skipped, bands repaired ----

def test_elicit_parses_probability_and_band():
    ask = canned_ask(['{"probability": 0.7, "low": 0.6, "high": 0.8, "reason": "ok"}'])
    belief = elicit(AGENT, CARD, [], ask_fn=ask)
    assert belief == {"agent": "base rates", "probability": 0.7, "low": 0.6,
                      "high": 0.8, "reason": "ok"}


def test_elicit_returns_none_on_junk():
    assert elicit(AGENT, CARD, [], ask_fn=canned_ask(["not json at all"])) is None


def test_elicit_returns_none_when_probability_missing():
    ask = canned_ask(['{"low": 0.6, "high": 0.8, "reason": "no probability field"}'])
    assert elicit(AGENT, CARD, [], ask_fn=ask) is None


def test_elicit_clamps_an_overconfident_probability():
    # 1.7 isn't a real probability; clamp it into [0.01, 0.99] same as
    # engine/swarm.py's agent_vote does, and stretch the band to hold it.
    ask = canned_ask(['{"probability": 1.7, "low": 0.5, "high": 0.9, "reason": "too sure"}'])
    belief = elicit(AGENT, CARD, [], ask_fn=ask)
    assert belief["probability"] == 0.99
    assert belief["high"] == 0.99          # repaired: band stretched to hold it
    assert belief["low"] == 0.5


def test_elicit_repairs_a_swapped_band():
    # low and high came back backwards: repair by swapping, don't discard.
    ask = canned_ask(['{"probability": 0.5, "low": 0.8, "high": 0.2, "reason": "swapped"}'])
    belief = elicit(AGENT, CARD, [], ask_fn=ask)
    assert belief["low"] == 0.2
    assert belief["high"] == 0.8
    assert belief["low"] <= belief["probability"] <= belief["high"]


def test_elicit_repairs_a_band_that_excludes_the_probability():
    # The band the model gave doesn't even contain its own probability;
    # repair by stretching it just enough to hold, don't throw it away.
    ask = canned_ask(['{"probability": 0.9, "low": 0.1, "high": 0.3, "reason": "excludes"}'])
    belief = elicit(AGENT, CARD, [], ask_fn=ask)
    assert belief["low"] == 0.1
    assert belief["high"] == 0.9           # stretched up to hold 0.9
    assert belief["low"] <= belief["probability"] <= belief["high"]


def test_elicit_skips_when_the_band_has_no_real_numbers():
    # A usable probability but a band that can't be turned into numbers
    # at all: there is nothing to repair, so this is skipped, not guessed.
    ask = canned_ask(['{"probability": 0.5, "low": "idk", "high": "idk", "reason": "?"}'])
    assert elicit(AGENT, CARD, [], ask_fn=ask) is None


def test_elicit_threads_evidence_slice_and_model_like_agent_vote():
    seen = []
    def ask(prompt, model=None, max_tokens=400):
        seen.append((prompt, model))
        return '{"probability": 0.6, "low": 0.5, "high": 0.7, "reason": "r"}'
    seat = {"label": "haiku+headlines", "instruction": "news only",
           "evidence": "headlines", "model": "haiku"}
    elicit(seat, CARD, ["a headline"], ask_fn=ask)
    prompt, model = seen[0]
    assert model == "haiku"
    assert "Recent headlines" in prompt
    assert "% chance" not in prompt   # a headlines-only seat is never told a fake market price


# ---- roll(): deterministic, clamped, honest with no beliefs ----

def test_roll_is_deterministic_same_seed():
    elicited = [{"agent": "a", "probability": 0.6, "low": 0.5, "high": 0.7},
               {"agent": "b", "probability": 0.3, "low": 0.2, "high": 0.4}]
    first = roll(elicited, 50_000, config.SEED)
    second = roll(elicited, 50_000, config.SEED)
    assert first == second


def test_roll_empty_elicited_returns_the_honest_neutral_default():
    # Rolling dice through zero beliefs would invent an opinion nobody
    # gave. Same idea as consensus([]) elsewhere in the engine: 0.5, flat.
    out = roll([], 1000, config.SEED)
    assert out == {"probability": 0.5, "p10": 0.5, "p50": 0.5, "p90": 0.5, "draws": 0}


def test_roll_clamps_the_sampled_probability_at_both_ends():
    # A deliberately absurd band (-5 to 5) that no elicit() output could
    # ever produce, to prove the roll-time clip is its own safety net,
    # not just relying on elicit() having already cleaned the band.
    wild = [{"agent": "x", "probability": 0.5, "low": -5.0, "high": 5.0}]
    out = roll(wild, 100_000, config.SEED)
    assert out["p10"] == 0.01     # clamped at the floor
    assert out["p90"] == 0.99     # clamped at the ceiling
    assert 0.0 <= out["probability"] <= 1.0


def test_roll_unanimous_crowd_converges_near_its_own_probability():
    # All-agents-at-p sanity check from the spec: a crowd unanimous at
    # 0.7 with a tight band should roll to ~0.70 within 0.01.
    belief = {"agent": "x", "probability": 0.7, "low": 0.65, "high": 0.75}
    elicited = [dict(belief) for _ in range(8)]
    out = roll(elicited, 200_000, config.SEED)
    assert out["probability"] == pytest.approx(0.7, abs=0.01)


def test_roll_percentile_band_is_ordered_and_tight_for_a_tight_input():
    belief = {"agent": "x", "probability": 0.7, "low": 0.65, "high": 0.75}
    elicited = [dict(belief) for _ in range(8)]
    out = roll(elicited, 200_000, config.SEED)
    assert out["p10"] < out["p50"] < out["p90"]
    # the whole band the agents gave was [0.65, 0.75]; the rolled
    # percentiles must stay inside the beliefs actually elicited
    assert 0.65 <= out["p10"] < 0.7
    assert 0.7 < out["p90"] <= 0.75


def test_roll_one_million_draws_completes_well_under_ten_seconds():
    # Loose timing bound, not a flaky hard limit: measured about 0.5
    # seconds for 1,000,000 draws on the machine this was built on, so
    # 10 seconds leaves a wide margin for a slower CI box.
    belief = {"agent": "x", "probability": 0.5, "low": 0.4, "high": 0.6}
    elicited = [dict(belief) for _ in range(8)]
    start = time.perf_counter()
    out = roll(elicited, 1_000_000, config.SEED)
    elapsed = time.perf_counter() - start
    assert out["draws"] == 1_000_000
    assert elapsed < 10.0


# ---- run_carlo(): elicit + roll + report, junk counted, config default ----

def test_run_carlo_counts_junk_and_agents_used():
    crowd = build_methods(4)
    answers = ['{"probability": 0.6, "low": 0.5, "high": 0.7, "reason": "a"}',
               '{"probability": 0.7, "low": 0.6, "high": 0.8, "reason": "b"}',
               "junk, not json",
               '{"probability": 0.5, "low": 0.4, "high": 0.6, "reason": "c"}']
    out = run_carlo(CARD, [], crowd, draws=1000, ask_fn=canned_ask(answers))
    assert out["agents_used"] == 3
    assert out["skipped"] == 1
    assert len(out["elicited"]) == 3


def test_run_carlo_all_junk_never_fabricates_a_belief():
    crowd = build_methods(3)
    out = run_carlo(CARD, [], crowd, draws=1000,
                    ask_fn=canned_ask(["junk", "garbage", "still garbage"]))
    assert out["agents_used"] == 0
    assert out["skipped"] == 3
    assert out["probability"] == 0.5   # honest neutral default, nothing invented


def test_run_carlo_reports_the_configured_seed():
    crowd = build_methods(1)
    ask = canned_ask(['{"probability": 0.6, "low": 0.5, "high": 0.7, "reason": "a"}'])
    out = run_carlo(CARD, [], crowd, draws=1000, ask_fn=ask)
    assert out["seed"] == config.SEED == 14000605


def test_run_carlo_defaults_draws_to_config_carlo_draws(monkeypatch):
    monkeypatch.setattr(config, "CARLO_DRAWS", 777)
    crowd = build_methods(1)
    ask = canned_ask(['{"probability": 0.6, "low": 0.5, "high": 0.7, "reason": "a"}'])
    out = run_carlo(CARD, [], crowd, ask_fn=ask)   # no draws= override
    assert out["draws"] == 777


def test_carlo_draws_config_default_is_one_million():
    assert config.CARLO_DRAWS == 1_000_000


def test_run_carlo_is_deterministic_end_to_end():
    crowd = build_methods(3)
    answers = ['{"probability": 0.6, "low": 0.5, "high": 0.7, "reason": "a"}',
               '{"probability": 0.4, "low": 0.3, "high": 0.5, "reason": "b"}',
               '{"probability": 0.8, "low": 0.7, "high": 0.9, "reason": "c"}']
    out1 = run_carlo(CARD, [], crowd, draws=20_000, ask_fn=canned_ask(list(answers)))
    out2 = run_carlo(CARD, [], crowd, draws=20_000, ask_fn=canned_ask(list(answers)))
    assert out1["probability"] == out2["probability"]
    assert out1["p10"] == out2["p10"]
    assert out1["p50"] == out2["p50"]
    assert out1["p90"] == out2["p90"]


# ---- ask.py's --carlo flag: composes with --crowd and --agents ----

def test_print_carlo_matches_the_spec_example_format(capsys):
    result = {"probability": 0.718, "p10": 0.62, "p90": 0.81, "p50": 0.71,
             "draws": 1_000_000, "agents_used": 8, "skipped": 0, "seed": 14000605}
    ask_module._print_carlo(result)
    out = capsys.readouterr().out
    assert "ONE MILLION FUTURES ROLLED: 71.8% ended YES" in out
    assert "the crowd's belief band: 62% to 81% (80% of futures fell here)" in out
    assert "(8 minds elicited, 0 junk skipped, seed 14000605)" in out


def test_print_carlo_only_says_one_million_when_it_truly_is(capsys):
    result = {"probability": 0.5, "p10": 0.4, "p90": 0.6, "p50": 0.5,
             "draws": 200_000, "agents_used": 4, "skipped": 0, "seed": 14000605}
    ask_module._print_carlo(result)
    out = capsys.readouterr().out
    assert "200,000 FUTURES ROLLED" in out
    assert "ONE MILLION" not in out


def test_cli_carlo_flag_composes_agents_and_crowd(monkeypatch):
    captured = {}
    def fake_run_carlo(card, headlines, crowd, draws=None, ask_fn=None):
        captured["crowd_size"] = len(crowd)
        return {"probability": 0.5, "p10": 0.4, "p90": 0.6, "p50": 0.5,
               "draws": 1000, "agents_used": len(crowd), "skipped": 0,
               "seed": config.SEED}
    monkeypatch.setattr(ask_module, "run_carlo", fake_run_carlo)
    monkeypatch.setattr(sys, "argv",
                        ["ask.py", "Will it happen?", "--carlo",
                         "--agents", "3", "--crowd", "methods", "--no-news"])
    ask_module.main()
    assert captured["crowd_size"] == 3


def test_cli_carlo_flag_off_never_calls_run_carlo(monkeypatch):
    called = []
    monkeypatch.setattr(ask_module, "run_carlo",
                        lambda *a, **k: called.append(1))
    monkeypatch.setattr(ask_module, "ask_question",
                        lambda *a, **k: {"probability": 0.5, "spread": 0.0,
                                        "votes": [], "futures": [], "skipped": 0})
    monkeypatch.setattr(sys, "argv", ["ask.py", "Will it happen?", "--no-news"])
    ask_module.main()
    assert called == []   # --carlo wasn't passed, so run_carlo never runs
