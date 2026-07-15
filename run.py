"""One live cycle of the engine.

Grade what's open, look at the biggest open non-sports markets, let the
crowd form its number, and log a paper pick when the crowd disagrees with
the market by more than fees could explain. A person places any real bet —
this only writes CSV rows.
"""
from __future__ import annotations

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


def pick_side(crowd_prob: float, mid: int) -> tuple[str, int] | None:
    """Which side (if any) does the crowd's number justify, after costs?"""
    need = config.MIN_EDGE_CENTS + config.FEE_BUFFER_CENTS
    yes_edge = round(crowd_prob * 100) - mid
    if yes_edge >= need:
        return "YES", yes_edge
    if -yes_edge >= need:
        return "NO", -yes_edge
    return None


def one_cycle(cards: list[dict] | None = None, ask_fn=None,
             now_iso: str | None = None) -> dict:
    live = cards is None
    ask = ask_fn or llm.ask
    now = now_iso or datetime.now(timezone.utc).isoformat()

    # 1. Grade open picks (live only — needs per-ticker fetches).
    graded = {"updated": 0, "settled": 0}
    if live:
        open_tickers = {r["ticker"] for r in ledger.load()
                        if r["status"] == "open"}
        latest = {}
        fetch_failures = 0
        for t in open_tickers:
            try:
                latest[t] = kalshi.fetch_market(t)
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
    for card in targets:
        heads = news.headlines_for(card["question"]) if live else []
        result = run_crowd(card, heads, crowd, mode=mode,
                           k=config.SIM_ROLLOUTS_K,
                           deliberation=config.DELIBERATION, ask_fn=ask)
        if not result["votes"]:
            # Nobody gave a usable answer — a fake 0.5 "consensus" would
            # fabricate a pick out of nothing. Skip the market instead.
            print(f'  no quorum (all {result["skipped"]} answers unusable) '
                  f'| {card["question"]}')
            continue
        verdict = pick_side(result["probability"], card["mid"])
        line = (f'{card["mid"]:>3}c market | {result["probability"]:.2f} crowd '
                f'(spread {result["spread"]:.2f}, {result["skipped"]} skipped) '
                f'| {card["question"][:60]}')
        if verdict is None:
            print(f"  pass  {line}")
            continue
        side, edge = verdict
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

    print(f"cycle done: {len(targets)} markets considered, {picks} picks, "
          f"{graded['settled']} settled, ${llm.spent_usd():.2f} total spend")
    return {"considered": len(targets), "picks": picks, "graded": graded}


if __name__ == "__main__":
    one_cycle()
