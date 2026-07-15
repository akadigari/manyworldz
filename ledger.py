"""The paper ledger: every pick the crowd makes, graded against reality.

Append-only CSV. CLV (closing line value) is the honest score: did the
market move toward our pick after we made it?
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

LEDGER_COLUMNS = ["logged_at", "ticker", "question", "side", "entry_mid",
                  "crowd_prob", "edge_cents", "mode", "status", "result",
                  "latest_mid", "clv_cents", "settled_at"]

_DEFAULT = config.DATA / "ledger.csv"


def load(path: Path | None = None) -> list[dict]:
    path = path or _DEFAULT
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_all(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def log_pick(row: dict, path: Path | None = None) -> None:
    """Append one pick. A second open pick on the same ticker+side is a
    repeat opinion, not a new position — refused."""
    path = path or _DEFAULT
    rows = load(path)
    for r in rows:
        if (r["ticker"] == row["ticker"] and r["side"] == row["side"]
                and r["status"] == "open"):
            return
    rows.append({col: row.get(col, "") for col in LEDGER_COLUMNS})
    _write_all(rows, path)


def grade(latest_by_ticker: dict[str, dict], path: Path | None = None) -> dict:
    """Refresh open picks against the latest market state."""
    path = path or _DEFAULT
    rows = load(path)
    updated = settled = 0
    for r in rows:
        if r["status"] != "open" or r["ticker"] not in latest_by_ticker:
            continue
        latest = latest_by_ticker[r["ticker"]]
        entry = int(r["entry_mid"])
        mid = int(latest.get("mid") or entry)
        r["latest_mid"] = mid
        r["clv_cents"] = (mid - entry) if r["side"] == "YES" else (entry - mid)
        updated += 1
        if latest.get("status") == "settled" and latest.get("result"):
            r["status"] = "settled"
            r["result"] = latest["result"]
            r["settled_at"] = datetime.now(timezone.utc).isoformat()
            settled += 1
    _write_all(rows, path)
    return {"updated": updated, "settled": settled}
