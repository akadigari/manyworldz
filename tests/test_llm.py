import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import llm


def test_cache_returns_same_answer_without_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm")
    monkeypatch.setattr(llm, "SPEND_FILE", tmp_path / "spend.json")
    calls = []

    def fake_api(prompt, model, max_tokens):
        calls.append(prompt)
        return "hello", 10, 5  # text, input tokens, output tokens

    monkeypatch.setattr(llm, "_call_api", fake_api)
    a = llm.ask("What is 2+2?")
    b = llm.ask("What is 2+2?")
    assert a == b == "hello"
    assert len(calls) == 1  # second answer came from disk, not the API


def test_budget_cap_halts_before_calling(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm")
    monkeypatch.setattr(llm, "SPEND_FILE", tmp_path / "spend.json")
    monkeypatch.setattr(llm, "_call_api", lambda p, m, t: ("x", 1000, 1000))
    import config
    monkeypatch.setattr(config, "ENGINE_BUDGET_USD", 0.0)
    with pytest.raises(RuntimeError, match="budget"):
        llm.ask("anything new")


def test_spend_meter_accumulates(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm")
    monkeypatch.setattr(llm, "SPEND_FILE", tmp_path / "spend.json")
    monkeypatch.setattr(llm, "_call_api", lambda p, m, t: ("x", 1_000_000, 0))
    llm.ask("one")  # 1M input tokens on haiku = $1.00
    assert llm.spent_usd() == pytest.approx(1.00, abs=0.01)
