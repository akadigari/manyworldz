"""Answer a ForecastBench question set with the manyworldz engine and write
the forecast-set file locally. This script NEVER uploads anything.

    python forecastbench.py --limit 5            # dry run: 5 real answers, rest fallbacks
    python forecastbench.py                      # the whole latest round
    python forecastbench.py --date 2026-10-11 --agents 3 --model haiku

Market questions get one methods-crowd vote each (engine/swarm.py, the
same path the Metaculus bot uses) with the freeze-date market price as
the anchor. Dataset questions ask each crowd seat for every resolution
date in ONE call and fold the seats per date with swarm.consensus.

Spend: a hard per-run cap, FORECASTBENCH_BUDGET_USD (default $3). The
run stops asking before a question whose worst-case cost would cross
it; every question left unanswered gets a labelled fallback (the market
price at freeze for market questions, 0.5 for dataset questions, which
is exactly what ForecastBench imputes for a gap) so the file stays
complete and schema-valid. Spend is metered in its own file,
data/spend_forecastbench.json, so this never eats the live Metaculus
bot's budget.

Uploading (owner only, after registration; see the wiki, section 6):
    gcloud storage cp data/forecastbench/<due>.<org>.<N>.json \\
        gs://<bucket-and-folder-from-the-registration-reply>/
Due by 23:59:59 UTC on the forecast due date; max 3 sets per round.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from adapters import forecastbench as fb
from engine import llm, news
from engine.methods import build_methods
from engine.swarm import _clean_prob, consensus, extract_json, run_crowd

DEFAULT_BUDGET_USD = 3.00
DEFAULT_AGENTS = 3
OUT_DIR = config.DATA / "forecastbench"
SPEND_FILE = config.DATA / "spend_forecastbench.json"
DATASET_FALLBACK = 0.5          # what ForecastBench itself imputes for a gap
MARKET_MAX_TOKENS = 400
DATASET_MAX_TOKENS = 600
EST_PROMPT_TOKENS = 1500        # generous: prompts run ~600-1000 tokens
TEXT_TRIM = 1200                # background/criteria characters per prompt
REASON_TRIM = 300

_DATASET_PROMPT = """Reason with one method only. Method: {label}. {instruction}.

Today is {due}. You are forecasting a question generated from a public
data series ({source}). It will be asked at several resolution dates.

Question (for resolution date D): "{question}"
Resolution: {criteria}
Latest value when the question was frozen: {freeze_value} ({freeze_explanation})
Background: {background}
{headlines}

Work outside-in: the status quo, then the base rate for moves of this
kind over each horizon, then the evidence. Longer horizons usually carry
more uncertainty; do not hedge every date to 0.5 without a reason.

RESOLUTION DATES: {dates}
Reply with ONLY JSON like {{"probabilities": {{"<date>": 0.42, ...}}, "reason": "one short sentence"}}
with one probability for EVERY date above."""


class BudgetStop(Exception):
    """The per-run cap (or the engine's monthly cap) says stop asking."""


def budget_from_env() -> float:
    return float(os.environ.get("FORECASTBENCH_BUDGET_USD", DEFAULT_BUDGET_USD))


def _pinned(ask_fn, default_model: str):
    """Unpinned calls run on the chosen model (same idea as tournament._cheap_ask)."""
    def _wrapped(prompt, model=None, max_tokens=MARKET_MAX_TOKENS, **kw):
        return ask_fn(prompt, model=model or default_model, max_tokens=max_tokens, **kw)
    return _wrapped


def _worst_case_usd(model: str, n_agents: int, max_tokens: int) -> float:
    price_in, price_out = llm._PRICES.get(model, llm._DEFAULT_PRICE)
    return n_agents * (EST_PROMPT_TOKENS * price_in + max_tokens * price_out) / 1e6


def _freeze_price(q: dict) -> float | None:
    try:
        p = float(q.get("freeze_datetime_value"))
    except (TypeError, ValueError):
        return None
    return p if 0.0 <= p <= 1.0 else None


def _fallbacks(q: dict, why: str) -> list[dict]:
    if fb.question_kind(q) == "market":
        price = _freeze_price(q)
        p, basis = (price, "market price at freeze") if price is not None else (0.5, "0.5")
        return [fb.make_forecast(q, p, None, f"fallback ({why}): {basis}")]
    return [fb.make_forecast(q, DATASET_FALLBACK, d, f"fallback ({why}): 0.5")
            for d in fb.resolution_dates(q)]


def _market_card(q: dict, due: str) -> dict:
    criteria = (q.get("market_info_resolution_criteria") or "")
    if criteria in ("", "N/A"):
        criteria = q.get("background") or ""
    text = (f"{q['question']}\n\n{criteria[:TEXT_TRIM]}\n\nToday is {due}. "
            f"The market closes {q.get('market_info_close_datetime', 'N/A')}; "
            "the price below is from four days ago.")
    price = _freeze_price(q)
    return {"question": text, "mid": round(price * 100) if price is not None else None}


def _answer_market(q: dict, due: str, crowd, ask, headlines_fn, model: str) -> list[dict]:
    result = run_crowd(_market_card(q, due), headlines_fn(q["question"], limit=8),
                       crowd, mode="vote", ask_fn=ask)
    if not result["votes"]:
        return _fallbacks(q, "no usable crowd answer")
    reason = f"{len(result['votes'])}-seat crowd ({model}): {result['votes'][0]['reason']}"
    return [fb.make_forecast(q, result["probability"], None, reason[:REASON_TRIM])]


def _dataset_prompt(agent: dict, q: dict, due: str, headlines: list[str]) -> str:
    dates = fb.resolution_dates(q)
    return _DATASET_PROMPT.format(
        label=agent["label"], instruction=agent["instruction"], due=due,
        source=q["source"],
        question=fb.fill_dates(q["question"], due, "D"),
        criteria=(q.get("resolution_criteria") or "")[:TEXT_TRIM],
        freeze_value=q.get("freeze_datetime_value", "N/A"),
        freeze_explanation=q.get("freeze_datetime_value_explanation", ""),
        background=(q.get("background") or "N/A")[:TEXT_TRIM],
        headlines=f"Recent headlines: {'; '.join(headlines) if headlines else '(none found)'}",
        dates=", ".join(dates))


def _parse_dataset_reply(text: str, dates: list[str]) -> dict | None:
    """{date: prob} only if EVERY date got a usable probability."""
    parsed = extract_json(text or "")
    probs = (parsed or {}).get("probabilities")
    if not isinstance(probs, dict):
        return None
    out = {d: _clean_prob(probs.get(d)) for d in dates}
    if any(p is None for p in out.values()):
        return None
    return {"probs": out, "reason": str(parsed.get("reason", ""))[:200]}


def _answer_dataset(q: dict, due: str, crowd, ask, headlines_fn, model: str) -> list[dict]:
    dates = fb.resolution_dates(q)
    headlines = headlines_fn(fb.fill_dates(q["question"], due, dates[-1]), limit=5)

    def _seat(agent):
        reply = ask(_dataset_prompt(agent, q, due, headlines),
                    max_tokens=DATASET_MAX_TOKENS)
        return _parse_dataset_reply(reply, dates)

    with ThreadPoolExecutor(max_workers=max(len(crowd), 1)) as pool:
        seats = [s for s in pool.map(_seat, crowd) if s]
    if not seats:
        return _fallbacks(q, "no usable crowd answer")
    reason = f"{len(seats)}-seat crowd ({model}): {seats[0]['reason']}"[:REASON_TRIM]
    return [fb.make_forecast(q, consensus([s["probs"][d] for s in seats])[0], d, reason)
            for d in dates]


def _answer(q, due, crowd, ask, headlines_fn, model) -> list[dict]:
    """One question's forecasts. Budget errors become BudgetStop; any other
    failure costs only this question (labelled fallback)."""
    answer = _answer_market if fb.question_kind(q) == "market" else _answer_dataset
    try:
        return answer(q, due, crowd, ask, headlines_fn, model)
    except Exception as exc:
        if "budget cap hit" in str(exc):
            raise BudgetStop(str(exc)) from exc
        print(f"  {q['source']}:{q['id']} failed ({exc}); fallback")
        return _fallbacks(q, "error")


def forecast_question_set(question_set: dict, *, ask_fn=llm.ask, spent_fn=llm.spent_usd,
                          headlines_fn=news.research, budget_usd: float | None = None,
                          n_agents: int = DEFAULT_AGENTS, model: str | None = None,
                          limit: int | None = None, organization: str = "manyworldz",
                          model_name: str | None = None,
                          model_organization: str = "Anthropic") -> dict:
    """Answer every standard question; return the forecast set plus counts.

    Stops asking (never crashes) when `limit` questions are answered, when
    the next question's worst-case cost would cross `budget_usd`, or when
    the engine raises its own budget error. Unanswered questions get
    labelled fallbacks so the set stays complete.
    """
    budget = budget_from_env() if budget_usd is None else budget_usd
    model = llm.resolve_model(model or config.TOURNAMENT_CHEAP_MODEL)
    ask = _pinned(ask_fn, model)
    crowd = build_methods(n_agents)
    due = question_set["forecast_due_date"]
    start, worst_seen = spent_fn(), 0.0
    forecasts, answered, fallback, stopped_by = [], 0, 0, None

    for q in fb.standard_questions(question_set):
        max_tokens = MARKET_MAX_TOKENS if fb.question_kind(q) == "market" else DATASET_MAX_TOKENS
        worst = max(_worst_case_usd(model, n_agents, max_tokens), worst_seen)
        before = spent_fn()
        if stopped_by is None and limit is not None and answered >= limit:
            stopped_by = "limit"
        if stopped_by is None and before - start + worst > budget:
            stopped_by = "budget"
        if stopped_by is None:
            try:
                forecasts.extend(_answer(q, due, crowd, ask, headlines_fn, model))
                answered += 1
                worst_seen = max(worst_seen, spent_fn() - before)
                continue
            except BudgetStop as exc:
                print(f"  engine budget stop: {exc}")
                stopped_by = "budget"
        forecasts.extend(_fallbacks(q, f"not asked: {stopped_by}"))
        fallback += 1

    forecast_set = {"organization": organization,
                    "model": model_name or f"{model} (kayfabe crowd)",
                    "model_organization": model_organization,
                    "question_set": question_set["question_set"],
                    "forecasts": forecasts}
    return {"forecast_set": forecast_set, "answered": answered, "fallback": fallback,
            "stopped_by": stopped_by, "spent_usd": round(spent_fn() - start, 4)}


def write_forecast_set(forecast_set: dict, due_date: str, organization: str, n: int,
                       out_dir: Path = OUT_DIR) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fb.forecast_filename(due_date, organization, n)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(forecast_set, indent=1))
    os.replace(tmp, path)
    return path


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--date", help="forecast due date YYYY-MM-DD (default: latest set)")
    p.add_argument("--question-file", help="read a local question set instead of fetching")
    p.add_argument("--limit", type=int, help="answer at most N questions (dry runs)")
    p.add_argument("--agents", type=int, default=DEFAULT_AGENTS)
    p.add_argument("--model", help="engine model (default: MANYWORLDZ_CHEAP_MODEL / haiku)")
    p.add_argument("--org", default=os.environ.get("FORECASTBENCH_ORG", "manyworldz"))
    p.add_argument("--model-name", help="leaderboard model label; cannot change once posted")
    p.add_argument("--model-org", default="Anthropic")
    p.add_argument("--n", type=int, default=1, help="forecast set number 1..3")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if not os.environ.get("MANYWORLDZ_SPEND_FILE"):
        llm.SPEND_FILE = SPEND_FILE     # own meter: never touch the live bot's
    if args.question_file:
        qs = json.loads(Path(args.question_file).read_text())
    else:
        qs = fb.fetch_question_set(args.date)
    out = forecast_question_set(qs, n_agents=args.agents, model=args.model,
                                limit=args.limit, organization=args.org,
                                model_name=args.model_name,
                                model_organization=args.model_org)
    errors = fb.validate_forecast_set(out["forecast_set"], qs)
    path = write_forecast_set(out["forecast_set"], qs["forecast_due_date"], args.org,
                              args.n, Path(args.out_dir))
    print(f"wrote {path}: {out['answered']} answered, {out['fallback']} fallback, "
          f"stopped_by={out['stopped_by']}, spent ~${out['spent_usd']:.2f}")
    for e in errors[:20]:
        print(f"  SCHEMA: {e}")
    print("NOT uploaded. Upload is a separate owner step (see module docstring).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
