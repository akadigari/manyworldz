"""Tests for report.py — the honest bridge from ledger to website."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report import build_report


def pick(status="open", side="YES", result="", clv=5):
    return {"logged_at": "2026-07-15T12:00:00Z", "ticker": "T",
            "question": "Will it happen?", "side": side, "entry_mid": 43,
            "crowd_prob": 0.61, "edge_cents": 18, "mode": "vote",
            "status": status, "result": result, "latest_mid": 48,
            "clv_cents": clv, "settled_at": ""}


def test_stats_count_wins_and_clv_honestly():
    picks = [
        pick("settled", "YES", "yes", clv=10),   # win
        pick("settled", "YES", "no", clv=-8),    # loss
        pick("settled", "NO", "no", clv=6),      # win (NO pick, resolved no)
        pick("open", clv=4),
    ]
    r = build_report(picks, None, {"est_usd": 1.234})
    s = r["stats"]
    assert s["total_picks"] == 4 and s["open"] == 1 and s["settled"] == 3
    assert s["wins"] == 2 and s["losses"] == 1
    assert s["avg_clv_cents"] == pytest.approx((10 - 8 + 6 + 4) / 4)
    assert s["spend_usd"] == 1.23


def test_empty_ledger_reports_zeros_not_crash():
    r = build_report([], None, None)
    s = r["stats"]
    assert s["total_picks"] == 0 and s["avg_clv_cents"] == 0.0
    assert r["cycle"]["markets"] == []
