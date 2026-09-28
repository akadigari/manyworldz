"""ForecastBench runner: question set in, schema-valid forecast set out.

No network anywhere: the question set comes from a fixture through an
injected fetch, the model is a fake ask_fn, and headlines are a fake.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest

import forecastbench
from adapters import forecastbench as fb

FIXTURE = Path(__file__).parent / "fixtures" / "forecastbench_question_set.json"
MARKET_IDS = {"11045", "0x9f94a3f64e8db20bf9d1766643b3bf4c345e3439d282d779775e171c780e9ddf"}
VOTE = '{"probability": 0.7, "reason": "fake"}'


def question_set():
    return json.loads(FIXTURE.read_text())


def fake_ask(prompt, model=None, max_tokens=400):
    """Answers a market vote with 0.7 and a dataset prompt with a rising
    probability per horizon, keyed on whatever dates the prompt lists."""
    if "RESOLUTION DATES:" in prompt:
        dates = prompt.split("RESOLUTION DATES:")[1].split("\n")[0].split(",")
        probs = {d.strip(): 0.4 + 0.02 * i for i, d in enumerate(dates)}
        return json.dumps({"probabilities": probs, "reason": "fake"})
    return VOTE


def no_news(question, limit=8):
    return []


class Meter:
    """Stands in for the engine's spend meter: every ask costs 0.01."""
    def __init__(self):
        self.usd = 0.0

    def ask(self, prompt, model=None, max_tokens=400):
        self.usd += 0.01
        return fake_ask(prompt, model, max_tokens)

    def spent(self):
        return self.usd


def run(**kw):
    meter = kw.pop("meter", Meter())
    defaults = dict(ask_fn=meter.ask, spent_fn=meter.spent, headlines_fn=no_news,
                    budget_usd=5.0, n_agents=2, model="claude-haiku-4-5",
                    organization="manyworldz", model_name="kayfabe (test)")
    defaults.update(kw)
    return forecastbench.forecast_question_set(question_set(), **defaults)


# ---- adapter: fetching and reading the question set ----

def test_fetch_follows_latest_symlink_to_the_dated_file():
    urls = []

    def fetch(url):
        urls.append(url)
        if url.endswith("latest-llm.json"):
            return "2026-09-27-llm.json"        # raw GitHub serves the symlink text
        return FIXTURE.read_text()

    qs = fb.fetch_question_set(fetch=fetch)
    assert qs["question_set"] == "2026-09-27-llm.json"
    assert urls[-1].endswith("/2026-09-27-llm.json")


def test_fetch_by_date_goes_straight_to_that_file():
    urls = []
    qs = fb.fetch_question_set("2026-09-27",
                               fetch=lambda u: urls.append(u) or FIXTURE.read_text())
    assert len(urls) == 1 and urls[0].endswith("/2026-09-27-llm.json")
    assert qs["forecast_due_date"] == "2026-09-27"


def test_fetch_rejects_a_malformed_question_set():
    with pytest.raises(ValueError):
        fb.fetch_question_set("2026-09-27", fetch=lambda u: '{"questions": []}')


def test_question_kinds_and_combination_questions_are_dropped():
    qs = question_set()
    combo = dict(qs["questions"][0], id=["a", "b"], source="metaculus")
    qs["questions"].append(combo)
    kept = fb.standard_questions(qs)
    assert len(kept) == 4
    assert {fb.question_kind(q) for q in kept} == {"market", "dataset"}


def test_fill_dates_replaces_the_template_placeholders():
    q = next(q for q in question_set()["questions"] if q["id"] == "IHLIDXUS")
    text = fb.fill_dates(q["question"], "2026-09-27", "2027-09-27")
    assert "{" not in text and "2026-09-27" in text and "2027-09-27" in text


def test_filename_follows_the_due_date_org_n_convention():
    assert fb.forecast_filename("2026-09-27", "manyworldz", 1) == "2026-09-27.manyworldz.1.json"
    with pytest.raises(ValueError):
        fb.forecast_filename("2026-09-27", "many.worldz", 1)   # a dot breaks the name
    with pytest.raises(ValueError):
        fb.forecast_filename("2026-09-27", "manyworldz", 4)    # max 3 sets a round


# ---- runner: a full, schema-valid forecast set ----

def test_full_run_writes_a_schema_valid_forecast_set():
    out = run()
    fs = out["forecast_set"]
    assert set(fs) == {"organization", "model", "model_organization",
                       "question_set", "forecasts"}
    assert fs["question_set"] == "2026-09-27-llm.json"
    assert fb.validate_forecast_set(fs, question_set()) == []
    # 2 market forecasts + 8 + 7 dataset horizons
    assert len(fs["forecasts"]) == 2 + 8 + 7
    for f in fs["forecasts"]:
        assert set(f) == {"id", "source", "forecast", "resolution_date", "reasoning"}
        assert 0.0 <= f["forecast"] <= 1.0
        if f["id"] in MARKET_IDS:
            assert f["resolution_date"] is None
        else:
            assert isinstance(f["resolution_date"], str)


def test_market_questions_use_the_crowd_and_dataset_horizons_keep_their_order():
    fs = run()["forecast_set"]
    market = [f for f in fs["forecasts"] if f["id"] in MARKET_IDS]
    assert all(f["forecast"] == 0.7 for f in market)
    fred = [f for f in fs["forecasts"] if f["id"] == "IHLIDXUS"]
    assert [f["resolution_date"] for f in fred] == question_set()["questions"][2]["resolution_dates"]
    assert fred[0]["forecast"] < fred[-1]["forecast"]        # the fake rises with horizon


def test_limit_answers_n_questions_and_fills_the_rest_with_labelled_fallbacks():
    out = run(limit=1)
    assert out["answered"] == 1 and out["fallback"] == 3
    fs = out["forecast_set"]
    assert fb.validate_forecast_set(fs, question_set()) == []
    poly = next(f for f in fs["forecasts"] if f["source"] == "polymarket")
    assert poly["forecast"] == pytest.approx(0.073)           # its freeze market price
    assert poly["reasoning"].startswith("fallback")


def test_budget_cap_stops_asking_before_it_is_crossed():
    meter = Meter()
    out = run(meter=meter, budget_usd=0.03, n_agents=2)
    assert meter.usd <= 0.03 + 1e-9
    assert out["answered"] < 4 and out["stopped_by"] == "budget"
    assert fb.validate_forecast_set(out["forecast_set"], question_set()) == []


def test_engine_budget_error_mid_run_degrades_to_fallbacks_not_a_crash():
    def broke(prompt, model=None, max_tokens=400):
        raise RuntimeError("engine budget cap hit ($10.00): raise ENGINE_BUDGET_USD")
    out = run(ask_fn=broke)
    assert out["answered"] == 0 and out["stopped_by"] == "budget"
    assert fb.validate_forecast_set(out["forecast_set"], question_set()) == []


def test_unparseable_dataset_answer_falls_back_for_that_question_only():
    def half_broken(prompt, model=None, max_tokens=400):
        return "no json here" if "RESOLUTION DATES:" in prompt else VOTE
    out = run(ask_fn=half_broken, spent_fn=lambda: 0.0)
    fs = out["forecast_set"]
    assert fb.validate_forecast_set(fs, question_set()) == []
    fred = [f for f in fs["forecasts"] if f["source"] == "fred"]
    assert all(f["forecast"] == 0.5 and f["reasoning"].startswith("fallback") for f in fred)


def test_dataset_prompt_states_the_dates_and_freeze_value():
    prompts = []

    def spy(prompt, model=None, max_tokens=400):
        prompts.append(prompt)
        return fake_ask(prompt, model, max_tokens)
    run(ask_fn=spy, spent_fn=lambda: 0.0, n_agents=1)
    ds = [p for p in prompts if "IHLIDXUS" in p or "job postings" in p]
    assert ds and "2026-09-27" in ds[0] and "2036-09-24" in ds[0]
    assert "{resolution_date}" not in ds[0]


def test_write_puts_the_named_file_on_disk(tmp_path):
    out = run()
    path = forecastbench.write_forecast_set(out["forecast_set"], "2026-09-27",
                                            "manyworldz", 1, tmp_path)
    assert path.name == "2026-09-27.manyworldz.1.json"
    assert json.loads(path.read_text()) == out["forecast_set"]


# ---- validator catches what the guide forbids ----

@pytest.mark.parametrize("mutate, needle", [
    (lambda fs: fs.pop("model"), "model"),
    (lambda fs: fs["forecasts"][0].update(forecast=1.2), "forecast"),
    (lambda fs: fs["forecasts"][0].update(resolution_date="2026-10-04"), "resolution_date"),
    (lambda fs: fs["forecasts"][-1].update(resolution_date=None), "resolution_date"),
    (lambda fs: fs["forecasts"].append(dict(fs["forecasts"][0])), "duplicate"),
    (lambda fs: fs.update(question_set="2026-09-13-llm.json"), "question_set"),
])
def test_validator_flags_schema_breaks(mutate, needle):
    fs = run()["forecast_set"]
    mutate(fs)
    errors = fb.validate_forecast_set(fs, question_set())
    assert errors and any(needle in e for e in errors)


def test_validator_flags_coverage_under_95_percent():
    fs = run()["forecast_set"]
    fs["forecasts"] = [f for f in fs["forecasts"] if f["source"] != "fred"]
    assert any("coverage" in e for e in fb.validate_forecast_set(fs, question_set()))


def test_budget_env_default_is_small(monkeypatch):
    monkeypatch.delenv("FORECASTBENCH_BUDGET_USD", raising=False)
    assert 0 < forecastbench.budget_from_env() <= 5
    monkeypatch.setenv("FORECASTBENCH_BUDGET_USD", "2.5")
    assert forecastbench.budget_from_env() == 2.5
