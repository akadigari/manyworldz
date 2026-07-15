"""The paper ledger keeps a record of every pick the crowd makes, and later
checks it against what really happened. Nothing here is real money — it's
a spreadsheet-style CSV file we can grade over time.

The honest scorecard is CLV (closing line value): after we log a pick, did
the market's price move toward our side or away from it? If the market
kept drifting toward us, that's a sign the crowd was onto something real,
not just lucky.
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

LEDGER_COLUMNS = ["logged_at", "ticker", "question", "side", "entry_mid",
                  "crowd_prob", "edge_cents", "mode", "status", "result",
                  "latest_mid", "clv_cents", "settled_at"]

_DEFAULT = config.DATA / "ledger.csv"


def load(path: Path | None = None) -> list[dict]:
    """Read the ledger CSV into a list of plain dicts, one per row.

    If the file doesn't exist yet (nothing has been logged), this returns
    an empty list instead of crashing.
    """
    path = path or _DEFAULT
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_all(rows: list[dict], path: Path) -> None:
    """Write the whole ledger out without ever leaving it half-written.

    We write to a temp file in the same folder first, then swap it into
    the real path with one atomic rename (os.replace). If the process
    dies mid-write, the old ledger file is still sitting there intact —
    a crash can never truncate it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".ledger-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


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
    """Update every open pick with fresh market prices, and close out any
    that have settled.

    Takes a dict of the latest known price/status for each ticker (from
    the Kalshi adapter). For each open pick still in that dict, this
    updates its CLV score, and if the market has actually resolved, marks
    the pick "settled" with the real result. Returns a count of how many
    rows were touched and how many got settled.
    """
    path = path or _DEFAULT
    rows = load(path)
    updated = settled = 0
    for r in rows:
        if r["status"] != "open" or r["ticker"] not in latest_by_ticker:
            continue
        latest = latest_by_ticker[r["ticker"]]
        mid = latest.get("mid")
        if mid:                    # None or 0 means "price unknown" — keep old values
            entry = int(r["entry_mid"])
            r["latest_mid"] = mid
            # CLV: cents the price moved in our favor since we logged the
            # pick. Positive = the market is drifting toward us.
            r["clv_cents"] = (mid - entry) if r["side"] == "YES" else (entry - mid)
        updated += 1
        if latest.get("status") == "settled" and latest.get("result"):
            r["status"] = "settled"
            r["result"] = latest["result"]
            r["settled_at"] = datetime.now(timezone.utc).isoformat()
            settled += 1
    _write_all(rows, path)
    return {"updated": updated, "settled": settled}
