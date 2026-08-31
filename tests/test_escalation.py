"""Two-tier compute allocation and budget pacing (tournament.py).

The policy: run the crowd on the cheap model first, and pay for the
strong model only where the cheap crowd disagreed with itself or landed
near the coin flip, because peer score is won and lost on exactly those
questions. And when the month's budget gets low, stop escalating and
answer everything cheaply, because a cheap answer scores infinitely
better than the hard zero of silence.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import tournament

BIN_CARD = {"qid": 800001, "question": "Will the thing happen?",
            "close_time": "2026-12-31T00:00:00Z",
            "url": "https://www.metaculus.com/questions/800001/"}
BUDGET_ERROR = RuntimeError(
    "engine budget cap hit ($10.00): refusing to spend more this month")


def two_tier(monkeypatch):
    """Make the tiers genuinely different, whatever the local default."""
    monkeypatch.setattr(config, "ENGINE_MODEL", "sonnet")
    monkeypatch.setattr(config, "TOURNAMENT_CHEAP_MODEL", "haiku")


# --- the cheap-ask wrapper ------------------------------------------------

def test_cheap_ask_fills_in_the_cheap_model_only_when_unpinned(monkeypatch):
    two_tier(monkeypatch)
    seen = []
    def ask(prompt, model=None, max_tokens=400):
        seen.append(model)
        return "ok"
    cheap = tournament._cheap_ask(ask)
    cheap("q")                      # unpinned -> cheap model
    cheap("q", model="opus")        # an explicitly pinned seat keeps its model
    assert seen == ["haiku", "opus"]


# --- what counts as contested ---------------------------------------------

def test_wide_crowd_disagreement_is_contested():
    assert tournament._is_contested(0.8, config.ESCALATE_SPREAD + 0.01)


def test_a_near_coin_flip_is_contested_even_when_unanimous():
    assert tournament._is_contested(0.55, 0.0)


def test_confident_and_agreed_is_not_contested():
    assert not tournament._is_contested(0.9, 0.03)


def test_no_quorum_is_contested_because_a_stronger_model_may_parse():
    assert tournament._is_contested(None, None)


def test_a_missing_spread_reads_as_agreement_not_a_crash():
    assert not tournament._is_contested(0.9, None)


# --- the escalation policy, driven through a fake ladder ------------------

def scripted_ladder(results, record):
    """A _ladder stand-in that replays `results` and records each ask."""
    queue = list(results)
    def _fake(card, headlines, crowd, ask):
        record.append(ask)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return dict(item)
    return _fake


def test_uncontested_cheap_answer_stands_and_strong_is_never_paid_for(monkeypatch):
    two_tier(monkeypatch)
    asks = []
    monkeypatch.setattr(tournament, "_ladder", scripted_ladder(
        [{"prob": 0.9, "source": "crowd", "skipped": 0, "spread": 0.02}], asks))
    seen = []
    out = tournament._answer_one(BIN_CARD, [], [], 
                                 lambda p, model=None, max_tokens=400: seen.append(model) or "x")
    assert out["prob"] == 0.9 and out["source"] == "crowd"
    assert out["escalated"] is False
    assert len(asks) == 1
    asks[0]("probe")                # the one pass ran on the cheap tier
    assert seen == ["haiku"]


def test_contested_answer_is_escalated_and_the_strong_result_adopted(monkeypatch):
    two_tier(monkeypatch)
    asks = []
    monkeypatch.setattr(tournament, "_ladder", scripted_ladder(
        [{"prob": 0.52, "source": "crowd", "skipped": 0, "spread": 0.20},
         {"prob": 0.71, "source": "crowd", "skipped": 0, "spread": 0.05}], asks))
    seen = []
    out = tournament._answer_one(BIN_CARD, [], [],
                                 lambda p, model=None, max_tokens=400: seen.append(model) or "x")
    assert out["prob"] == 0.71
    assert out["source"] == "crowd+esc"
    assert out["escalated"] is True
    assert len(asks) == 2
    asks[0]("probe"); asks[1]("probe")
    assert seen == ["haiku", None]  # tier 1 cheap, tier 2 the configured voice


def test_a_strong_pass_that_degrades_to_fallback_is_not_adopted(monkeypatch):
    two_tier(monkeypatch)
    asks = []
    monkeypatch.setattr(tournament, "_ladder", scripted_ladder(
        [{"prob": 0.52, "source": "crowd", "skipped": 0, "spread": 0.20},
         {"prob": 0.5, "source": "fallback", "skipped": 0, "spread": 0.0}], asks))
    out = tournament._answer_one(BIN_CARD, [], [],
                                 lambda p, model=None, max_tokens=400: "x")
    assert out["prob"] == 0.52 and out["source"] == "crowd"
    assert out["escalated"] is False


def test_a_budget_error_during_escalation_keeps_the_cheap_answer(monkeypatch):
    """The cheap answer is already in hand; losing it because the
    escalation attempt hit the cap would turn one wall into two."""
    two_tier(monkeypatch)
    asks = []
    monkeypatch.setattr(tournament, "_ladder", scripted_ladder(
        [{"prob": 0.52, "source": "crowd", "skipped": 0, "spread": 0.20},
         BUDGET_ERROR], asks))
    out = tournament._answer_one(BIN_CARD, [], [],
                                 lambda p, model=None, max_tokens=400: "x")
    assert out["prob"] == 0.52 and out["source"] == "crowd"


def test_a_budget_error_on_the_cheap_pass_still_stops_the_cycle(monkeypatch):
    two_tier(monkeypatch)
    monkeypatch.setattr(tournament, "_ladder", scripted_ladder([BUDGET_ERROR], []))
    try:
        tournament._answer_one(BIN_CARD, [], [],
                               lambda p, model=None, max_tokens=400: "x")
        assert False, "budget error must propagate from the first pass"
    except RuntimeError as exc:
        assert tournament._is_budget_error(exc)


def test_no_quorum_on_the_cheap_pass_escalates(monkeypatch):
    two_tier(monkeypatch)
    asks = []
    monkeypatch.setattr(tournament, "_ladder", scripted_ladder(
        [{"prob": None, "source": None, "skipped": 8, "spread": None},
         {"prob": 0.66, "source": "crowd", "skipped": 0, "spread": 0.04}], asks))
    out = tournament._answer_one(BIN_CARD, [], [],
                                 lambda p, model=None, max_tokens=400: "x")
    assert out["prob"] == 0.66 and out["source"] == "crowd+esc"


def test_conserving_mode_makes_one_cheap_pass_and_never_escalates(monkeypatch):
    two_tier(monkeypatch)
    asks = []
    monkeypatch.setattr(tournament, "_ladder", scripted_ladder(
        [{"prob": 0.5, "source": "crowd", "skipped": 0, "spread": 0.30}], asks))
    seen = []
    out = tournament._answer_one(BIN_CARD, [], [],
                                 lambda p, model=None, max_tokens=400: seen.append(model) or "x",
                                 conserving=True)
    assert out["prob"] == 0.5
    assert len(asks) == 1
    asks[0]("probe")
    assert seen == ["haiku"]


def test_matching_tiers_disable_escalation_and_run_the_plain_ask(monkeypatch):
    monkeypatch.setattr(config, "ENGINE_MODEL", "haiku")
    monkeypatch.setattr(config, "TOURNAMENT_CHEAP_MODEL", "haiku")
    asks = []
    monkeypatch.setattr(tournament, "_ladder", scripted_ladder(
        [{"prob": 0.5, "source": "crowd", "skipped": 0, "spread": 0.30}], asks))
    seen = []
    tournament._answer_one(BIN_CARD, [], [],
                           lambda p, model=None, max_tokens=400: seen.append(model) or "x")
    assert len(asks) == 1
    asks[0]("probe")
    assert seen == [None]           # no wrapper: one tier is just the voice


# --- budget pacing --------------------------------------------------------

def test_conserving_trips_at_the_reserve_line():
    budget = 15.0
    line = budget * (1 - config.BUDGET_RESERVE_FRACTION)
    assert not tournament._conserving(spent=line - 0.01, budget=budget)
    assert tournament._conserving(spent=line, budget=budget)


def test_conserving_is_off_at_a_fresh_month():
    assert not tournament._conserving(spent=0.0, budget=15.0)


def test_conserving_cycle_answers_numeric_on_the_cheap_model(monkeypatch, tmp_path):
    two_tier(monkeypatch)
    monkeypatch.setattr(tournament.llm, "spent_usd", lambda: 14.0)
    monkeypatch.setattr(config, "ENGINE_BUDGET_USD", 15.0)
    num_card = {"qid": 800002, "post_id": 31002, "qtype": "numeric",
                "question": "How many?", "criteria": "",
                "close_time": "2026-12-31T00:00:00Z",
                "url": "https://www.metaculus.com/questions/31002/",
                "already_forecast": False, "unit": "things",
                "open_lower_bound": False, "open_upper_bound": True,
                "scaling": {"range_min": 0, "range_max": 60,
                            "inbound_outcome_count": 200, "continuous_range": None}}
    seen = []
    def ask(prompt, model=None, max_tokens=400):
        seen.append(model)
        return '{"p05": 5, "p25": 15, "p50": 30, "p75": 45, "p95": 55}'
    tournament.one_cycle(cards=[num_card], ask_fn=ask, token="tok",
                         log_path=tmp_path / "log.csv",
                         submit_fn=lambda qid, prob, token: None,
                         submit_numeric_fn=lambda qid, cdf, token: None)
    assert seen and all(m == "haiku" for m in seen)


def test_receipt_records_conserving_and_escalations(monkeypatch, tmp_path):
    import json
    two_tier(monkeypatch)
    monkeypatch.setattr(tournament.llm, "spent_usd", lambda: 0.5)
    monkeypatch.setattr(config, "ENGINE_BUDGET_USD", 15.0)
    monkeypatch.setattr(tournament, "_answer_one",
                        lambda card, headlines, crowd, ask, conserving=False:
                        {"prob": 0.7, "source": "crowd+esc", "skipped": 0,
                         "escalated": True})
    tournament.one_cycle(cards=[dict(BIN_CARD)], ask_fn=lambda p, model=None, max_tokens=400: "x",
                         token="tok", log_path=tmp_path / "log.csv",
                         submit_fn=lambda qid, prob, token: None)
    status = json.loads((tmp_path / "tournament_status.json").read_text())
    assert status["conserving"] is False
    assert status["escalated_this_cycle"] == 1


def test_numeric_normally_runs_on_the_configured_voice_not_a_hardcode(monkeypatch, tmp_path):
    """model="sonnet" was hardcoded in the numeric and MC paths, which
    both ignored MANYWORLDZ_MODEL and blocked the conservation floor.
    They now pass model=None and let the engine resolve the voice."""
    two_tier(monkeypatch)
    monkeypatch.setattr(tournament.llm, "spent_usd", lambda: 0.0)
    seen = []
    def ask(prompt, model=None, max_tokens=400):
        seen.append(model)
        return '{"p05": 5, "p25": 15, "p50": 30, "p75": 45, "p95": 55}'
    card = {"qid": 800003, "post_id": 31003, "qtype": "numeric",
            "question": "How many?", "criteria": "",
            "close_time": "2026-12-31T00:00:00Z",
            "url": "https://www.metaculus.com/questions/31003/",
            "already_forecast": False, "unit": "things",
            "open_lower_bound": False, "open_upper_bound": True,
            "scaling": {"range_min": 0, "range_max": 60,
                        "inbound_outcome_count": 200, "continuous_range": None}}
    tournament.one_cycle(cards=[card], ask_fn=ask, token="tok",
                         log_path=tmp_path / "log.csv",
                         submit_fn=lambda qid, prob, token: None,
                         submit_numeric_fn=lambda qid, cdf, token: None)
    assert seen and all(m is None for m in seen)
