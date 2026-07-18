"""One live cycle of the engine.

Grade what's open, look at the biggest open non-sports markets, let the
crowd form its number, and log a tracked pick when the crowd disagrees with
the market by more than fees could explain. A person places any real bet.
This only writes CSV rows.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import ledger
from adapters import kalshi_events as kalshi
from engine import llm, news
from engine.personas import build_crowd
from engine.swarm import run_crowd


def _save_cycle_snapshot(markets: list[dict], now: str, path: Path) -> None:
    """Save everything the crowd just saw and said, for the dashboard.

    The website draws its branching-futures map from this file: every
    vote, every imagined future, every verdict from the latest cycle.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"at": now, "markets": markets}, indent=1))


def pick_side(crowd_prob: float, mid: int) -> tuple[str, int] | None:
    """Decide whether the crowd disagrees with the market by enough to bet.

    Takes the crowd's probability (0 to 1) and the market's mid price (in
    cents). Returns ("YES", edge) or ("NO", edge) if the gap is wide enough
    to clear fees, or None if the two are too close to call.
    """
    need = config.MIN_EDGE_CENTS + config.FEE_BUFFER_CENTS
    yes_edge = round(crowd_prob * 100) - mid
    if yes_edge >= need:
        return "YES", yes_edge
    if -yes_edge >= need:
        return "NO", -yes_edge
    return None


def one_cycle(cards: list[dict] | None = None, ask_fn=None,
             now_iso: str | None = None,
             snapshot_path: Path | None = None) -> dict:
    """Run one full round of the engine: grade old picks, then make new ones.

    Pass in `cards` (fake market data) and `ask_fn` (a fake crowd) to run
    the whole cycle in tests, with no network or API calls. Leave both
    blank for a real run. Returns a small summary: how many markets were
    looked at, how many new picks got logged, and how grading went.
    """
    live = cards is None            # no fake cards given -> this is a real, live run
    ask = ask_fn or llm.ask
    now = now_iso or datetime.now(timezone.utc).isoformat()

    # 1. Grade open picks (live only, needs per-ticker fetches).
    graded = {"updated": 0, "settled": 0}
    if live:
        open_tickers = {r["ticker"] for r in ledger.load()
                        if r["status"] == "open"}
        latest = {}
        fetch_failures = 0
        for ticker in open_tickers:
            try:
                latest[ticker] = kalshi.fetch_market(ticker)
            except Exception:
                fetch_failures += 1        # one bad ticker shouldn't kill the cycle
        if fetch_failures:
            print(f"note: could not refresh {fetch_failures} open pick(s)")
        graded = ledger.grade(latest)
        cards = kalshi.fetch_open_markets()

    # 2. The crowd votes on the biggest tradeable markets.
    targets = kalshi.tradeable(cards, now)[:config.MARKETS_PER_RUN]
    crowd = build_crowd(config.ENGINE_N_AGENTS, config.SEED)
    mode = config.SIM_MODE

    picks = 0
    snapshot = []          # what the crowd saw + said, market by market
    for card in targets:
        headlines = news.research(card["question"]) if live else []
        result = run_crowd(card, headlines, crowd, mode=mode,
                           k=config.SIM_ROLLOUTS_K,
                           deliberation=config.DELIBERATION, ask_fn=ask)
        snap = {"ticker": card["ticker"], "question": card["question"],
                "category": card.get("category", ""), "mid": card["mid"],
                "probability": result["probability"],
                "spread": result["spread"], "skipped": result["skipped"],
                "votes": result["votes"], "futures": result["futures"],
                "verdict": None}
        snapshot.append(snap)
        if not result["votes"]:
            # Nobody gave a usable answer. A fake 0.5 "consensus" would
            # fabricate a pick out of nothing. Skip the market instead.
            snap["verdict"] = "no_quorum"
            print(f'  no quorum (all {result["skipped"]} answers unusable) '
                  f'| {card["question"]}')
            continue
        verdict = pick_side(result["probability"], card["mid"])
        # "mid" = the mid price: halfway between what buyers will pay and
        # sellers will take, in cents. The market's best guess at "fair."
        line = (f'{card["mid"]:>3}c market | {result["probability"]:.2f} crowd '
                f'(spread {result["spread"]:.2f}, {result["skipped"]} skipped) '
                f'| {card["question"][:60]}')
        if verdict is None:
            snap["verdict"] = "pass"
            print(f"  pass  {line}")
            continue
        side, edge = verdict
        snap["verdict"] = {"side": side, "edge_cents": edge}
        ledger.log_pick({
            "logged_at": now, "ticker": card["ticker"],
            "question": card["question"], "side": side,
            "entry_mid": card["mid"], "crowd_prob": result["probability"],
            "edge_cents": edge, "mode": mode, "status": "open",
            "result": "", "latest_mid": card["mid"], "clv_cents": 0,
            "settled_at": "",
        })
        picks += 1
        print(f"  PICK  {side} +{edge}c {line}")

    # Save the snapshot: always when a path is given (tests), and on live
    # runs by default (the dashboard reads it via report.py).
    if snapshot_path is None and live:
        snapshot_path = config.DATA / "latest_cycle.json"
    if snapshot_path is not None:
        _save_cycle_snapshot(snapshot, now, snapshot_path)

    print(f"cycle done: {len(targets)} markets considered, {picks} picks, "
          f"{graded['settled']} settled, ${llm.spent_usd():.2f} total spend")
    return {"considered": len(targets), "picks": picks, "graded": graded}


if __name__ == "__main__":
    one_cycle()
