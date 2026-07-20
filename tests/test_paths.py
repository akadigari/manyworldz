"""Tests for engine/explore.py's find_paths (--path: the "beat Thanos"
search). Everything here runs offline, same pattern as test_explore.py: a
fake ask_fn dispatches on what the prompt is asking for.

Every find_paths run makes three kinds of calls: the target-conditioned
imagine rounds (the ONLY calls where the model is told to imagine YES-only
or NO-only futures), the classify/dedupe call for paths, and exactly two
calls that are NOT about paths at all: a rating call at the end, and the
neutral run_crowd split whose plain probability becomes the headline
number. The dispatcher below tells them apart by unique phrases pulled
straight from the actual prompt templates.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from engine.explore import find_paths

CARD = {"ticker": "T", "question": "Will the Fed cut rates in September?", "mid": 43}


def paths_json(items):
    """Build a {"futures": [...]} payload with story+gates, from
    [(story, [gate, ...]), ...].
    """
    parts = []
    for story, gates in items:
        gates_block = ", ".join(f'"{g}"' for g in gates)
        parts.append(f'{{"story": "{story}", "gates": [{gates_block}]}}')
    return f'{{"futures": [{", ".join(parts)}]}}'


def futures_json(pairs):
    """Build a plain {"futures": [...]} payload for the neutral split (no
    gates needed there: it is a normal --simulate style call).
    """
    items = ",".join(
        f'{{"story": "{story}", "resolves": "{resolves}"}}'
        for story, resolves in pairs)
    return f'{{"futures": [{items}], "reason": "r"}}'


def classify_json(labels):
    inner = ", ".join(f'"{label}"' for label in labels)
    return f'{{"classifications": [{inner}]}}'


def ratings_json(labels):
    inner = ", ".join(f'"{label}"' for label in labels)
    return f'{{"ratings": [{inner}]}}'


# A generic, always-safe neutral answer: 4 futures, half YES, half NO.
NEUTRAL_50_50 = futures_json([("x", "YES"), ("y", "NO"), ("z", "YES"), ("w", "NO")])


def make_ask(path_imagine_answers, classify_answers, rating_answer, neutral_answer):
    """A fake ask_fn that dispatches four ways: the neutral split (its
    prompt is the plain agent_futures one, unique phrase "DIFFERENT ways
    this could actually play out"), the rating call ("Rate each path"),
    the paths classify call ("PATHS FOUND SO FAR"), and everything else
    falls to the path-imagine stack, in order.
    """
    imagine_stack = list(path_imagine_answers)
    classify_stack = list(classify_answers)

    def ask(prompt, model=None, max_tokens=400):
        if "DIFFERENT ways this could actually play out" in prompt:
            return neutral_answer
        if "Rate each path" in prompt:
            return rating_answer
        if "PATHS FOUND SO FAR" in prompt:
            return classify_stack.pop(0)
        return imagine_stack.pop(0)
    return ask


def test_gates_parse_into_each_path(monkeypatch):
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)
    ask = make_ask(
        path_imagine_answers=[paths_json([
            ("Powell signals a cut clearly",
             ["Powell speaks at Jackson Hole", "markets read it as dovish"])])],
        classify_answers=[classify_json(["new"])],
        rating_answer=ratings_json(["likely"]),
        neutral_answer=NEUTRAL_50_50)

    out = find_paths(CARD, [], ask, target="YES", k_per_round=1, max_rounds=1)

    assert out["target"] == "YES"
    assert len(out["paths"]) == 1
    path = out["paths"][0]
    assert path["story"] == "Powell signals a cut clearly"
    assert path["gates"] == ["Powell speaks at Jackson Hole", "markets read it as dovish"]
    assert path["count"] == 1
    assert path["rating"] == "likely"
    assert out["rounds"] == 1


def test_dedupe_merges_same_mechanism_across_rounds(monkeypatch):
    # A second round's path is really the same mechanism, just worded
    # differently: the classifier says "w1" and it should merge, not
    # double the map.
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)
    ask = make_ask(
        path_imagine_answers=[
            paths_json([("Fed cuts because inflation cools",
                        ["CPI comes in soft", "Powell signals comfort"])]),
            paths_json([("Fed lowers rates as prices ease",
                        ["inflation data cools further"])]),
        ],
        classify_answers=[classify_json(["new"]), classify_json(["w1"])],
        rating_answer=ratings_json(["possible"]),
        neutral_answer=NEUTRAL_50_50)

    out = find_paths(CARD, [], ask, target="YES", k_per_round=1, max_rounds=2)

    assert len(out["paths"]) == 1
    assert out["paths"][0]["count"] == 2
    assert out["rounds"] == 2


def test_ratings_sort_likely_then_possible_then_longshot(monkeypatch):
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)
    ask = make_ask(
        path_imagine_answers=[paths_json([
            ("story A", ["gate a1"]),
            ("story B", ["gate b1"]),
            ("story C", ["gate c1"]),
        ])],
        classify_answers=[classify_json(["new", "new", "new"])],
        rating_answer=ratings_json(["longshot", "likely", "possible"]),
        neutral_answer=NEUTRAL_50_50)

    out = find_paths(CARD, [], ask, target="YES", k_per_round=3, max_rounds=1)

    ratings_in_order = [p["rating"] for p in out["paths"]]
    assert ratings_in_order == ["likely", "possible", "longshot"]


def test_junk_rating_defaults_to_longshot(monkeypatch):
    # The rater is allowed to call everything longshot, and an answer we
    # cannot parse at all must default the same way: never upgrade on junk.
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)
    ask = make_ask(
        path_imagine_answers=[paths_json([("story A", ["gate a1"])])],
        classify_answers=[classify_json(["new"])],
        rating_answer="not valid json",
        neutral_answer=NEUTRAL_50_50)

    out = find_paths(CARD, [], ask, target="YES", k_per_round=1, max_rounds=1)

    assert out["paths"][0]["rating"] == "longshot"
    assert out["skipped"] >= 1   # the failed rating call is counted


def test_probability_comes_from_neutral_split_only(monkeypatch):
    # The path-conditioned rounds only ever imagine YES futures by design.
    # If that leaked into the probability it would read close to 100%.
    # The neutral split below is deliberately a 25% YES read: the
    # returned number must match that, not the conditioned rounds.
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)
    ask = make_ask(
        path_imagine_answers=[paths_json([("story A", ["gate a1"])])],
        classify_answers=[classify_json(["new"])],
        rating_answer=ratings_json(["possible"]),
        neutral_answer=futures_json([("x", "YES"), ("y", "NO"),
                                     ("z", "NO"), ("w", "NO")]))

    out = find_paths(CARD, [], ask, target="YES", k_per_round=1, max_rounds=1)

    assert out["probability"] == pytest.approx(0.25)


def test_zero_paths_case_is_a_real_answer(monkeypatch):
    # Every path-imagine call fails outright: the search should come back
    # with an empty paths list, no invented content, and it should not
    # even bother asking for ratings on nothing.
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)
    calls = {"rating": 0}

    def ask(prompt, model=None, max_tokens=400):
        if "DIFFERENT ways this could actually play out" in prompt:
            return NEUTRAL_50_50
        if "Rate each path" in prompt:
            calls["rating"] += 1
            return ratings_json([])
        return "not valid json"   # every path-imagine call is unusable

    out = find_paths(CARD, [], ask, target="YES", k_per_round=1, max_rounds=2)

    assert out["paths"] == []
    assert calls["rating"] == 0   # nothing to rate, so we never even ask
    assert out["probability"] == pytest.approx(0.5)
    assert out["skipped"] == 2    # both path rounds failed


def test_failed_imagine_round_counts_as_skipped_but_others_still_land(monkeypatch):
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)
    ask = make_ask(
        path_imagine_answers=[paths_json([("story A", ["gate a1"])]),
                              "not valid json"],
        classify_answers=[classify_json(["new"])],
        rating_answer=ratings_json(["possible"]),
        neutral_answer=NEUTRAL_50_50)

    out = find_paths(CARD, [], ask, target="YES", k_per_round=1, max_rounds=2)

    assert out["rounds"] == 2
    assert out["skipped"] == 1     # round 2's imagine call failed
    assert len(out["paths"]) == 1


def test_target_no_is_supported(monkeypatch):
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)
    ask = make_ask(
        path_imagine_answers=[paths_json([("story A", ["gate a1"])])],
        classify_answers=[classify_json(["new"])],
        rating_answer=ratings_json(["longshot"]),
        neutral_answer=NEUTRAL_50_50)

    out = find_paths(CARD, [], ask, target="NO", k_per_round=1, max_rounds=1)

    assert out["target"] == "NO"
    assert len(out["paths"]) == 1
    # the odds are always "chance of YES", even when hunting paths to NO
    assert out["probability"] == pytest.approx(0.5)


def test_defaults_are_read_from_config_at_call_time(monkeypatch):
    # Same proof as engine/explore.py's own version of this test: patching
    # config.PATH_MAX_ROUNDS after import must still be picked up, since
    # the default is read fresh at call time, not frozen in.
    monkeypatch.setattr(config, "ENGINE_N_AGENTS", 1)
    monkeypatch.setattr(config, "PATH_MAX_ROUNDS", 1)
    ask = make_ask(
        path_imagine_answers=[paths_json([("story A", ["gate a1"])])],
        classify_answers=[classify_json(["new"])],
        rating_answer=ratings_json(["possible"]),
        neutral_answer=NEUTRAL_50_50)

    out = find_paths(CARD, [], ask, target="YES")

    assert out["rounds"] == 1
