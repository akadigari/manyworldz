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


def test_friendly_model_names_resolve_to_full_ids():
    # "haiku"/"sonnet"/"opus"/"fable" are shortcuts; full IDs pass through.
    assert llm.resolve_model("haiku") == "claude-haiku-4-5"
    assert llm.resolve_model("FABLE") == "claude-fable-5"
    assert llm.resolve_model("claude-sonnet-5") == "claude-sonnet-5"
    assert llm.resolve_model("some-future-model") == "some-future-model"


def test_refusal_becomes_empty_answer_not_a_crash():
    # Frontier models can decline a request (stop_reason "refusal").
    # That must read as "no usable answer", never as invented text.
    from types import SimpleNamespace
    block = SimpleNamespace(type="text", text="should not appear")
    refused = SimpleNamespace(stop_reason="refusal", content=[block])
    assert llm._extract_text(refused) == ""
    normal = SimpleNamespace(stop_reason="end_turn", content=[block])
    assert llm._extract_text(normal) == "should not appear"


def test_thinking_blocks_are_skipped():
    # Reasoning models return "thinking" blocks alongside the answer text:
    # only the text should come through.
    from types import SimpleNamespace
    thinking = SimpleNamespace(type="thinking", thinking="internal notes")
    answer = SimpleNamespace(type="text", text='{"probability": 0.6}')
    msg = SimpleNamespace(stop_reason="end_turn", content=[thinking, answer])
    assert llm._extract_text(msg) == '{"probability": 0.6}'


# ---- monthly budget window ----

def test_spend_resets_when_the_month_changes(tmp_path, monkeypatch):
    """The budget is a monthly allowance, not a lifetime total: last
    month's spending must not silence the bot mid-season."""
    import json as _json
    monkeypatch.setattr(llm, "SPEND_FILE", tmp_path / "spend.json")
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    llm.SPEND_FILE.write_text(_json.dumps(
        {"calls": 999, "est_usd": 99.0, "month": "2020-01"}))
    assert llm.spent_usd() == 0.0            # old month reads as a fresh window

    monkeypatch.setattr(llm, "_call_api", lambda p, m, mt: ("hi", 100, 10))
    out = llm.ask("a fresh question after the month turned")
    assert out == "hi"
    state = _json.loads(llm.SPEND_FILE.read_text())
    assert state["calls"] == 1               # the counter restarted with the month
    assert state["month"] != "2020-01"


def test_concurrent_asks_never_lose_a_spend_record(tmp_path, monkeypatch):
    import json as _json
    import threading
    monkeypatch.setattr(llm, "SPEND_FILE", tmp_path / "spend.json")
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm, "_call_api", lambda p, m, mt: ("hi", 1000, 100))
    threads = [threading.Thread(target=llm.ask, args=(f"q{i}",))
              for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    state = _json.loads(llm.SPEND_FILE.read_text())
    assert state["calls"] == 8


# ---- openrouter routing ----

def test_ask_routes_through_openrouter_when_the_key_is_present(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "SPEND_FILE", tmp_path / "spend.json")
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    seen = {}
    def fake_or(prompt, model, max_tokens):
        seen["model"] = model
        return "routed", 10, 5
    monkeypatch.setattr(llm, "_call_openrouter", fake_or)
    monkeypatch.setattr(llm, "_call_api",
                        lambda p, m, mt: (_ for _ in ()).throw(
                            AssertionError("anthropic path must not be used")))
    assert llm.ask("route me", model="haiku") == "routed"
    assert seen["model"].startswith("anthropic/")


def test_ask_uses_the_anthropic_path_without_the_openrouter_key(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "SPEND_FILE", tmp_path / "spend.json")
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_call_api", lambda p, m, mt: ("direct", 10, 5))
    assert llm.ask("no routing here") == "direct"


def test_parallel_asks_cannot_overshoot_the_budget_by_a_full_batch(tmp_path, monkeypatch):
    """Review finding: the budget check was outside the lock, so N
    concurrent seats could all pass it and overshoot by N-1 calls.
    With reservation, at most one call can straddle the line."""
    import threading
    import config
    monkeypatch.setattr(llm, "SPEND_FILE", tmp_path / "spend.json")
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "ENGINE_BUDGET_USD", 0.010)
    calls = {"n": 0}
    lock = threading.Lock()
    def fake_call(p, m, mt):
        with lock:
            calls["n"] += 1
        return "hi", 1000, 400
    monkeypatch.setattr(llm, "_call_api", fake_call)
    barrier = threading.Barrier(8)
    def worker(i):
        barrier.wait()
        try:
            llm.ask(f"q{i}", model="sonnet")
        except RuntimeError:
            pass
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    # sonnet at 1000 in / 400 out estimates ~0.009 per call: the old
    # racy check let all 8 through; reservation allows at most 2.
    assert calls["n"] <= 2


def test_a_corrupt_cache_file_reads_as_a_miss_and_heals(tmp_path, monkeypatch):
    """Review finding: a killed process could leave truncated JSON in
    the cache, making that prompt raise forever."""
    import hashlib
    monkeypatch.setattr(llm, "SPEND_FILE", tmp_path / "spend.json")
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm, "_call_api", lambda p, m, mt: ("fresh", 10, 5))
    model = llm.resolve_model("haiku")
    key = hashlib.sha256(f"{model}\nbroken one".encode()).hexdigest()
    llm.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (llm.CACHE_DIR / f"{key}.json").write_text('{"text": "trunca')
    assert llm.ask("broken one", model="haiku") == "fresh"
    assert llm.ask("broken one", model="haiku") == "fresh"   # healed cache hit


# --- budget isolation -----------------------------------------------------
# Found 2026-08-30. The Kalshi loop (manyworldz.yml, four times a day) and
# the tournament loop (every 5 minutes) both metered into one shared
# data/spend.json, so Kalshi spending ate the tournament's headroom. Once
# the month's cap is hit, _is_budget_error stops the WHOLE tournament
# cycle, which means a hard zero on every question in an open round. The
# meter path is now overridable so each loop can carry its own.

def test_spend_file_follows_the_environment_when_set(tmp_path, monkeypatch):
    import importlib
    target = tmp_path / "spend_tournament.json"
    monkeypatch.setenv("MANYWORLDZ_SPEND_FILE", str(target))
    import engine.llm as llm_mod
    importlib.reload(llm_mod)
    assert llm_mod.SPEND_FILE == target


def test_spend_file_defaults_to_the_shared_meter(tmp_path, monkeypatch):
    import importlib
    monkeypatch.delenv("MANYWORLDZ_SPEND_FILE", raising=False)
    import engine.llm as llm_mod
    importlib.reload(llm_mod)
    assert llm_mod.SPEND_FILE.name == "spend.json"
