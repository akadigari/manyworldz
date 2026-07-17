"""Turn the scorecard into something the website (and a human) can read.

Reads three local files (the pick ledger, the latest cycle snapshot, and
the spending meter) and writes two things:

  web/data.json   what the knaves.ai dashboard draws
  REPORT.md       a plain-English summary for the repo page

The dashboard can never look better than reality, because this is its only
source and it reads the exact same ledger the gates read.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import ledger


def build_report(picks: list[dict], cycle: dict | None,
                 spend: dict | None) -> dict:
    """Fold the raw files into one honest summary dictionary."""
    open_picks = [p for p in picks if p["status"] == "open"]
    settled = [p for p in picks if p["status"] == "settled"]

    # A settled pick "won" if the side we took matched how it resolved.
    wins = sum(1 for p in settled
               if (p["side"] == "YES") == (p["result"] == "yes"))

    # CLV = closing line value: did the market move toward our pick after
    # we made it? Positive average = the crowd was ahead of the market.
    clv_values = [int(p["clv_cents"]) for p in picks if p.get("clv_cents") not in ("", None)]
    avg_clv = round(sum(clv_values) / len(clv_values), 2) if clv_values else 0.0

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "total_picks": len(picks),
            "open": len(open_picks),
            "settled": len(settled),
            "wins": wins,
            "losses": len(settled) - wins,
            "avg_clv_cents": avg_clv,
            "spend_usd": round((spend or {}).get("est_usd", 0.0), 2),
        },
        "picks": picks,
        "cycle": cycle or {"at": "", "markets": []},
    }


def write_outputs(report: dict) -> None:
    """Write web/data.json (for the site) and REPORT.md (for the repo)."""
    web = config.ROOT / "web"
    web.mkdir(exist_ok=True)
    (web / "data.json").write_text(json.dumps(report, indent=1))

    s = report["stats"]
    lines = [
        "# manyworldz: the live scorecard",
        "",
        f"_Updated {report['generated_at'][:16]}Z. Paper picks only. A"
        " person makes any real decision, and only if the gates pass._",
        "",
        f"- **Picks:** {s['total_picks']} total, {s['open']} open,"
        f" {s['settled']} settled",
        f"- **Record on settled:** {s['wins']} wins, {s['losses']} losses",
        f"- **Average CLV:** {s['avg_clv_cents']:+.1f} cents"
        " (positive = the market moved toward the crowd)",
        f"- **Engine spend:** ${s['spend_usd']:.2f}"
        f" (hard cap ${config.ENGINE_BUDGET_USD:.2f})",
        "",
        "The full pick-by-pick record lives in `data/ledger.csv`; the"
        " dashboard at knaves.ai draws from the same file.",
    ]
    (config.ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    picks = ledger.load()
    cycle_file = config.DATA / "latest_cycle.json"
    cycle = json.loads(cycle_file.read_text()) if cycle_file.exists() else None
    spend_file = config.DATA / "spend.json"
    spend = json.loads(spend_file.read_text()) if spend_file.exists() else None
    report = build_report(picks, cycle, spend)
    write_outputs(report)
    s = report["stats"]
    print(f"report written: {s['total_picks']} picks, "
          f"{s['settled']} settled, avg CLV {s['avg_clv_cents']:+.1f}c")


if __name__ == "__main__":
    main()
