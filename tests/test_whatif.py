import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.methods import build_methods
from engine.whatif import run_whatif

CARD = {"ticker": "T", "question": "Will the album drop in July?", "mid": 43}


def test_whatif_reruns_and_reports_shift():
    crowd = build_methods(2)
    seen_prompts = []

    def ask(prompt, model=None, max_tokens=400):
        seen_prompts.append(prompt)
        # crowd leans low before the injection, high after it
        if "WHAT-IF" in prompt:
            return '{"probability": 0.9, "reason": "fact changes everything"}'
        return '{"probability": 0.3, "reason": "doubtful"}'

    out = run_whatif(CARD, [], crowd, inject="the label confirms the date",
                     mode="vote", k=0, deliberation=False, ask_fn=ask)
    assert out["before"]["probability"] == pytest.approx(0.3)
    assert out["after"]["probability"] == pytest.approx(0.9)
    assert out["shift"] == pytest.approx(0.6)
    assert any("label confirms the date" in p for p in seen_prompts)
