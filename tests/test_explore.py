"""Tests for engine/explore.py (--deep: keep splitting until nothing new
shows up). Everything here runs offline: a fake ask_fn stands in for the
model, dispatching on what the prompt is asking for so the test does not
have to know exactly how many agents are in the crowd.

Round 1 reuses engine.futures.agent_futures under the hood, which only
trusts an agent's answer if at least half of the requested futures parsed
(minimum 2). So round-1 imagine payloads below always hand back the full
batch of k stories; later rounds use engine.explore's own looser rule
(at least half, minimum 1), so a single story per round is enough there.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from engine.explore import explore_worlds

CARD = {"ticker": "T", "question": "Will the album drop in July?", "mid": 43}


def make_ask(imagine_answers, classify_answers):
    """A fake ask_fn that hands out imagine-call answers in order, and
    classify-call answers in order, dispatching on which kind of prompt
    it was just given (the classify prompt always names the map).
    """
    imagine_stack = list(imagine_answers)
    classify_stack = list(classify_answers)

    def ask(prompt, model=None, max_tokens=400):
        if "WORLDS FOUND SO FAR" in prompt:
            return classify_stack.pop(0)
        return imagine_stack.pop(0)
    return ask


def futures_json(pairs):
    """Build a {"futures": [...]} payload from [(story, resolves), ...]."""
    items = ",".join(
        f'{{"story": "{story}", "resolves": "{resolves}"}}'
        for story, resolves in pairs)
    return f'{{"futures": [{items}], "reason": "r"}}'


def classify_json(labels):
    inner = ", ".join(f'"{label}"' for label in labels)
    return f'{{"classifications": [{inner}]}}'


def test_round_one_reuses_simulate_and_classifies(monkeypatch):
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)
    ask = make_ask(
        imagine_answers=[futures_json([("Fed cuts 25bps", "YES"),
                                        ("Fed holds steady", "NO")])],
        classify_answers=[classify_json(["new", "new"])])

    out = explore_worlds(CARD, [], ask, k_per_round=2, max_rounds=1)

    assert out["rounds"] == 1
    assert out["raw_futures"] == 2
    assert len(out["worlds"]) == 2
    assert {w["count"] for w in out["worlds"]} == {1}
    assert out["skipped"] == 0
    assert out["probability"] == pytest.approx(0.5)


def test_dedupe_merges_same_mechanism_and_sums_counts(monkeypatch):
    # One agent's own round-1 batch imagines the same mechanism twice.
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)
    ask = make_ask(
        imagine_answers=[futures_json([("Fed cuts 25bps", "YES"),
                                        ("Fed cuts rates a quarter point", "YES")])],
        classify_answers=[classify_json(["new", "c1"])])

    out = explore_worlds(CARD, [], ask, k_per_round=2, max_rounds=1)

    assert len(out["worlds"]) == 1
    assert out["worlds"][0]["count"] == 2
    assert out["raw_futures"] == 2


def test_new_classifications_grow_the_map_across_rounds(monkeypatch):
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)
    ask = make_ask(
        imagine_answers=[futures_json([("A", "YES"), ("B", "NO")]),
                         futures_json([("C", "YES")]),
                         futures_json([("D", "NO")])],
        classify_answers=[classify_json(["new", "new"]),
                          classify_json(["new"]),
                          classify_json(["new"])])

    out = explore_worlds(CARD, [], ask, k_per_round=2, max_rounds=3, dry_rounds=2)

    assert out["rounds"] == 3   # never dry, so it runs until the round cap
    assert len(out["worlds"]) == 4
    assert out["raw_futures"] == 4


def test_two_dry_rounds_stop_the_loop_early(monkeypatch):
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)
    ask = make_ask(
        imagine_answers=[futures_json([("Fed cuts 25bps", "YES"),
                                        ("Fed slashes rates", "YES")]),
                         futures_json([("Another cut story", "YES")]),
                         futures_json([("Yet another cut story", "YES")])],
        classify_answers=[classify_json(["new", "c1"]),  # round 1: 1 world, not dry
                          classify_json(["w1"]),          # round 2: duplicate, dry 1
                          classify_json(["w1"])])         # round 3: duplicate, dry 2, stop

    out = explore_worlds(CARD, [], ask, k_per_round=2, max_rounds=8, dry_rounds=2)

    assert out["rounds"] == 3
    assert len(out["worlds"]) == 1
    assert out["worlds"][0]["count"] == 4
    assert out["raw_futures"] == 4


def test_max_rounds_caps_it_even_when_still_finding_new_worlds(monkeypatch):
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)
    ask = make_ask(
        imagine_answers=[futures_json([("A", "YES"), ("B", "YES")]),
                         futures_json([("C", "YES")])],
        classify_answers=[classify_json(["new", "new"]),
                          classify_json(["new"])])

    out = explore_worlds(CARD, [], ask, k_per_round=2, max_rounds=2, dry_rounds=99)

    assert out["rounds"] == 2   # capped, even though round 2 still found something new
    assert len(out["worlds"]) == 3


def test_probability_is_raw_census_not_deduped_count(monkeypatch):
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)
    ask = make_ask(
        imagine_answers=[futures_json([("Fed cuts 25bps", "YES"),
                                        ("Fed holds steady", "NO")]),
                         futures_json([("Fed slashes rates again", "YES")])],
        classify_answers=[classify_json(["new", "new"]),  # world 1 (YES), world 2 (NO)
                          classify_json(["w1"])])          # world 1 (YES) again

    out = explore_worlds(CARD, [], ask, k_per_round=2, max_rounds=2, dry_rounds=99)

    assert out["raw_futures"] == 3
    assert len(out["worlds"]) == 2                 # two distinct worlds
    # 2 YES out of 3 raw futures, not 1 YES out of 2 worlds (which would be 0.5)
    assert out["probability"] == pytest.approx(2 / 3)
    assert out["worlds"][0]["count"] >= out["worlds"][1]["count"]   # sorted desc


def test_junk_classification_answers_never_add_worlds(monkeypatch):
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)
    ask = make_ask(
        imagine_answers=[futures_json([("Fed cuts 25bps", "YES"),
                                        ("Fed holds steady", "NO")])],
        classify_answers=["not valid json at all"])

    out = explore_worlds(CARD, [], ask, k_per_round=2, max_rounds=1)

    assert out["worlds"] == []               # junk never inflates the map
    assert out["raw_futures"] == 2           # the raw futures still count for the math
    assert out["skipped"] == 1               # the failed classify call is counted
    assert out["probability"] == pytest.approx(0.5)   # 1 YES of 2


def test_skipped_counts_a_failed_imagine_call_in_a_later_round(monkeypatch):
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)
    ask = make_ask(
        imagine_answers=[futures_json([("A", "YES"), ("B", "NO")]),
                         "not json either"],           # round 2's imagine call fails
        classify_answers=[classify_json(["new", "new"])])  # only round 1 gets that far

    out = explore_worlds(CARD, [], ask, k_per_round=2, max_rounds=2, dry_rounds=1)

    assert out["rounds"] == 2
    assert out["skipped"] == 1
    assert out["raw_futures"] == 2
    assert len(out["worlds"]) == 2


def test_budget_error_bubbles_up(monkeypatch):
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)

    def blows_the_budget(prompt, model=None, max_tokens=400):
        raise RuntimeError("engine budget cap hit ($10.00)")

    with pytest.raises(RuntimeError):
        explore_worlds(CARD, [], blows_the_budget, k_per_round=2, max_rounds=1)


def test_defaults_are_read_from_config_at_call_time(monkeypatch):
    # Round 1 grows the map, round 2 is a duplicate. With DEEP_DRY_ROUNDS
    # patched to 1, that single dry round should be enough to stop the
    # loop, proving explore_worlds actually reads config instead of a
    # default that got frozen in at import time.
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)
    monkeypatch.setattr(config, "DEEP_MAX_ROUNDS", 5)
    monkeypatch.setattr(config, "DEEP_DRY_ROUNDS", 1)
    ask = make_ask(
        imagine_answers=[futures_json([("A", "YES"), ("B", "YES")]),
                         futures_json([("Another A-like story", "YES")])],
        classify_answers=[classify_json(["new", "new"]), classify_json(["w1"])])

    out = explore_worlds(CARD, [], ask, k_per_round=2)

    assert out["rounds"] == 2
