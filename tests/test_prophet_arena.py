"""Tests for the Prophet Arena adapter and entrypoint.

Same injected-fake pattern as tests/test_tournament.py: a fake `ask`
stands in for the model and a fake `research` stands in for the news
search, so nothing here ever touches the network. Events are plain
SimpleNamespace objects shaped like the SDK's arena.Event (slug, title,
criteria, close_at, outcomes[name, criteria], crowd, context,
mutually_exclusive), so the adapter is tested without the SDK installed.
The entrypoint test at the bottom runs only where the SDK is importable.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import config
from adapters import prophet_arena as pa

CLOSE = datetime(2026, 12, 31, tzinfo=timezone.utc)
CLIP = config.TOURNAMENT_CLIP


def make_event(outcomes, title="Will thing happen?", criteria="Resolves YES if thing.",
               crowd=None, context=None, mutually_exclusive=False, slug="evt-1"):
    return SimpleNamespace(
        slug=slug, title=title, criteria=criteria, close_at=CLOSE,
        outcomes=[SimpleNamespace(name=n, criteria=None) for n in outcomes],
        crowd=crowd or {}, context=context or [],
        mutually_exclusive=mutually_exclusive)


def vote(p):
    return f'{{"probability": {p}, "reason": "fake"}}'


class FakeAsk:
    """Records every prompt; answers with a fixed or prompt-keyed reply."""

    def __init__(self, reply=None, by_marker=None):
        self.reply = reply
        self.by_marker = by_marker or {}
        self.prompts = []

    def __call__(self, prompt, model=None, max_tokens=400):
        self.prompts.append(prompt)
        for marker, answer in self.by_marker.items():
            if marker in prompt:
                return answer
        return self.reply


def no_research(question):
    return []


def run(event, ask, research=no_research):
    return pa.forecast_event(event, ask_fn=ask, research_fn=research, conserving=False)


def test_yes_no_event_maps_crowd_probability_to_yes_and_complement_to_no():
    ask = FakeAsk(vote(0.8))
    out = run(make_event(["Yes", "No"]), ask)
    assert out["probabilities"] == {"Yes": 0.8, "No": 0.2}
    assert ask.prompts                                   # the crowd actually ran
    assert "Will thing happen?" in ask.prompts[0]
    assert "Resolves YES if thing." in ask.prompts[0]    # criteria reach the prompt


def test_yes_no_detection_is_case_insensitive_and_keeps_event_spelling():
    out = run(make_event(["YES", "no"]), FakeAsk(vote(0.7)))
    assert out["probabilities"] == {"YES": 0.7, "no": 0.3}


def test_extreme_answers_are_clipped_like_the_tournament():
    out = run(make_event(["Yes", "No"]), FakeAsk(vote(0.99)))
    assert out["probabilities"]["Yes"] == round(1 - CLIP, 4)
    assert out["probabilities"]["No"] == CLIP


def test_two_named_outcomes_are_one_crowd_run_on_the_first_outcome():
    ask = FakeAsk(vote(0.6))
    event = make_event(["New Orleans", "Las Vegas"], title="LV Raiders vs NO Saints")
    out = run(event, ask)
    assert out["probabilities"] == {"New Orleans": 0.6, "Las Vegas": 0.4}
    assert "New Orleans" in ask.prompts[0]
    assert len(ask.prompts) == len(pa.build_crowd_for())  # one crowd, not two


def test_mutually_exclusive_multi_outcome_is_one_direct_mc_call():
    ask = FakeAsk('{"A": 0.5, "B": 0.3, "C": 0.2}')
    out = run(make_event(["A", "B", "C"], mutually_exclusive=True), ask)
    assert len(ask.prompts) == 1
    assert out["probabilities"] == {"A": 0.5, "B": 0.3, "C": 0.2}


def test_mc_reply_is_normalized_then_clipped():
    ask = FakeAsk('{"A": 2.0, "B": 0.0, "C": 0.0}')
    out = run(make_event(["A", "B", "C"], mutually_exclusive=True), ask)
    assert out["probabilities"] == {"A": round(1 - CLIP, 4), "B": CLIP, "C": CLIP}


def test_non_exclusive_multi_outcome_runs_one_crowd_per_outcome():
    ask = FakeAsk(by_marker={"'Alpha'": vote(0.7), "'Beta'": vote(0.2),
                             "'Gamma'": vote(0.5)})
    out = run(make_event(["Alpha", "Beta", "Gamma"], title="Which will happen?"), ask)
    assert out["probabilities"] == {"Alpha": 0.7, "Beta": 0.2, "Gamma": 0.5}


def test_outcomes_past_the_crowd_cap_fall_back_to_the_crowd_price(monkeypatch):
    monkeypatch.setattr(pa, "MAX_CROWD_OUTCOMES", 2)
    ask = FakeAsk(vote(0.6))
    names = ["A", "B", "C", "D"]
    event = make_event(names, crowd={"C": 0.33, "D": 0.001})
    out = run(event, ask)
    assert out["probabilities"]["A"] == 0.6 and out["probabilities"]["B"] == 0.6
    assert out["probabilities"]["C"] == 0.33
    assert out["probabilities"]["D"] == CLIP             # crowd price is clipped too
    assert not any("'C'" in p or "'D'" in p for p in ask.prompts)
    assert "fallback" in out["rationale"]


def test_budget_cap_never_crashes_and_falls_back_to_crowd_prices():
    def broke(prompt, model=None, max_tokens=400):
        raise RuntimeError("engine budget cap hit ($10.00): raise ENGINE_BUDGET_USD")
    out = run(make_event(["Yes", "No"], crowd={"Yes": 0.64, "No": 0.36}), broke)
    assert out["probabilities"] == {"Yes": 0.64, "No": 0.36}
    assert "budget" in out["rationale"]


def test_budget_cap_with_no_crowd_falls_back_to_uniform():
    def broke(prompt, model=None, max_tokens=400):
        raise RuntimeError("engine budget cap hit ($10.00)")
    out = run(make_event(["A", "B", "C"], mutually_exclusive=True), broke)
    third = round(1 / 3, 4)
    assert out["probabilities"] == {"A": third, "B": third, "C": third}


def test_budget_cap_stops_spending_on_the_remaining_outcomes():
    calls = []

    def broke(prompt, model=None, max_tokens=400):
        calls.append(prompt)
        raise RuntimeError("engine budget cap hit ($10.00)")
    out = run(make_event(["A", "B", "C"]), broke)
    assert set(out["probabilities"]) == {"A", "B", "C"}
    # The first outcome's crowd hits the wall; later outcomes never ask.
    assert not any("'B'" in p or "'C'" in p for p in calls)


def test_unusable_engine_answer_prefers_crowd_price_over_coin_flip():
    ask = FakeAsk("no json here")
    out = run(make_event(["Yes", "No"], crowd={"Yes": 0.9, "No": 0.1}), ask)
    assert out["probabilities"] == {"Yes": 0.9, "No": 0.1}


def test_frozen_context_and_research_both_reach_the_prompt():
    seen = []

    def research(question):
        seen.append(question)
        return ["Live headline from research"]
    context = [SimpleNamespace(title="Snapshot source title", url="https://x",
                               summary="Snapshot summary text")]
    ask = FakeAsk(vote(0.5))
    run(make_event(["Yes", "No"], context=context), ask, research=research)
    assert seen and "Will thing happen?" in seen[0]
    assert "Snapshot source title" in ask.prompts[0]
    assert "Live headline from research" in ask.prompts[0]


def test_research_failure_is_not_fatal():
    def research(question):
        raise ConnectionError("offline")
    out = run(make_event(["Yes", "No"]), FakeAsk(vote(0.55)), research=research)
    assert out["probabilities"] == {"Yes": 0.55, "No": 0.45}


def test_every_outcome_gets_exactly_one_finite_probability():
    event = make_event(["A", "B", "C", "D", "E"])
    out = run(event, FakeAsk(vote(0.3)))
    assert list(out["probabilities"]) == ["A", "B", "C", "D", "E"]
    assert all(CLIP <= p <= 1 - CLIP for p in out["probabilities"].values())


def test_binary_card_matches_the_engines_card_shape():
    event = make_event(["Yes", "No"], slug="kx-1")
    card = pa.binary_card(event)
    assert card["qid"] == "kx-1"
    assert card["question"] == "Will thing happen?"
    assert card["criteria"] == "Resolves YES if thing."
    assert card["close_time"] == CLOSE.isoformat()
    assert card.get("qtype", "binary") == "binary"


def test_outcome_card_names_the_outcome_and_prefers_its_own_criteria():
    event = make_event(["Alpha", "Beta", "Gamma"])
    event.outcomes[1] = SimpleNamespace(name="Beta", criteria="Beta-specific rule.")
    card = pa.binary_card(event, event.outcomes[1])
    assert "'Beta'" in card["question"]
    assert card["criteria"] == "Beta-specific rule."
    assert card["qid"] == "evt-1:Beta"


def test_mc_card_carries_options_in_event_order():
    card = pa.mc_card(make_event(["X", "Y", "Z"], mutually_exclusive=True))
    assert card["qtype"] == "multiple_choice"
    assert card["options"] == ["X", "Y", "Z"]


# ── entrypoint (needs the prophet-arena SDK) ────────────────────────────────


def test_entrypoint_returns_a_forecast_the_platform_would_accept(monkeypatch):
    arena = pytest.importorskip("arena")
    import prophet_agent
    monkeypatch.setattr(prophet_agent, "ASK_FN", FakeAsk(vote(0.8)))
    monkeypatch.setattr(prophet_agent, "RESEARCH_FN", no_research)
    event = arena.Event(slug="kx-1", title="Will thing happen?", close_at=CLOSE,
                        outcomes=[arena.Outcome(name="Yes"), arena.Outcome(name="No")],
                        crowd={"Yes": 0.5, "No": 0.5})
    forecast = prophet_agent.agent.forecast(event)
    forecast.validate_for(event)                       # raises if the server would reject
    assert forecast.as_floats() == {"Yes": 0.8, "No": 0.2}
    assert isinstance(prophet_agent.agent, arena.Agent)
    assert prophet_agent.agent.name == "manyworldz"
    assert prophet_agent.agent.track == "agentic"      # it does its own news research
