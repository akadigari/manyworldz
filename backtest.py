"""Score the bot against questions Metaculus has already resolved.

The tournament bot has never been graded. MiniBench had nothing open
between arming it and today, so every tuning knob (crowd size,
deliberation, the clip) is set on judgment alone, and judgment without
feedback is just taste. Metaculus keeps its resolved questions, with the
outcome and the community forecast attached, so the evidence exists
right now. This reads it.

    python3 backtest.py --limit 25 --dry-run   # costs nothing, plans the run
    python3 backtest.py --limit 25             # spends, writes the table
    python3 backtest.py --inspect              # one API call, dumps the shape

LEAKAGE, WHICH IS THE WHOLE GAME HERE
A backtest that can see the answer is worse than no backtest, because it
produces a number people believe. Two controls, both enforced in code and
covered by tests:

1. No research. engine/news.py searches CURRENT news; pointed at a
   question that resolved last week it hands the model the answer. The
   backtest passes empty headlines and never calls the news layer, and
   every scored row records research="disabled" so no one can misread
   the result later.
2. Nothing resolved before config.MODEL_CUTOFF_DATE. An outcome the
   model could have memorized in training is not a forecast.

Even with both, treat this as calibration evidence and not as a
tournament score: the real thing is peer-scored against other bots, and
here the community forecast is the only stand-in available for that.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import tournament
from adapters import metaculus
from engine import llm
from engine.ensemble import build_crowd_for

OUT_DIR = config.DATA
# Clip before taking a log: a bot that says 0 and is wrong deserves a bad
# score, not an infinite one that swallows every other row in the mean.
LOG_CLIP = 1e-4
CALIBRATION_BUCKETS = [(i / 10, (i + 1) / 10) for i in range(10)]


def after_cutoff(cards: list[dict]) -> list[dict]:
    """Only questions that resolved after the model's training cutoff.

    A card with no resolve time is dropped rather than assumed recent:
    the whole point is to exclude outcomes the model might already know,
    so the missing case has to fail closed.
    """
    cutoff = datetime.fromisoformat(config.MODEL_CUTOFF_DATE).replace(
        tzinfo=timezone.utc)
    kept = []
    for card in cards:
        stamp = card.get("resolved_at")
        if not stamp:
            continue
        try:
            when = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        except ValueError:
            continue
        if when > cutoff:
            kept.append(card)
    return kept


def brier(prob: float, outcome: float) -> float:
    """Squared error. Lower is better, 0.25 is a coin flip every time."""
    return (prob - outcome) ** 2


def log_score(prob: float, outcome: float) -> float:
    """Surprisal in nats. Lower is better, punishes confident and wrong."""
    p = min(max(prob, LOG_CLIP), 1 - LOG_CLIP)
    return -math.log(p if outcome >= 0.5 else 1 - p)


def score_one(card: dict, ask, crowd: list[dict] | None = None) -> dict:
    """Run the real answer path on one resolved question and score it.

    Deliberately calls tournament._answer_one, the same ladder the live
    cycle uses, so this measures the bot rather than a lookalike of it.
    Headlines are empty by construction: see the leakage note up top.
    """
    crowd = crowd if crowd is not None else build_crowd_for()
    answer = tournament._answer_one(card, [], crowd, ask)
    prob = answer.get("prob")
    row = {
        "qid": card.get("qid"),
        "question": card.get("question"),
        "qtype": card.get("qtype", "binary"),
        "prob": prob,
        "source": answer.get("source"),
        "outcome": card.get("outcome"),
        "community": card.get("community"),
        "resolved_at": card.get("resolved_at"),
        "research": "disabled",
    }
    if prob is not None:
        row["brier"] = brier(prob, card["outcome"])
        row["log_score"] = log_score(prob, card["outcome"])
    return row


def summarize(rows: list[dict]) -> dict:
    """Headline numbers: how good, and better or worse than the crowd.

    Rows the crowd never priced still count toward our own score; they
    just sit out the comparison, which is why the two means are taken
    over different sets and the community one can come back None.
    """
    scored = [r for r in rows if r.get("prob") is not None
              and r.get("outcome") is not None]
    if not scored:
        return {"n": 0, "brier": None, "log_score": None,
                "community_brier": None, "beat_community": None,
                "calibration": []}

    mean = lambda xs: sum(xs) / len(xs)
    ours = mean([brier(r["prob"], r["outcome"]) for r in scored])
    paired = [r for r in scored if r.get("community") is not None]
    theirs = (mean([brier(r["community"], r["outcome"]) for r in paired])
              if paired else None)
    ours_paired = (mean([brier(r["prob"], r["outcome"]) for r in paired])
                   if paired else None)

    calibration = []
    for lo, hi in CALIBRATION_BUCKETS:
        in_bucket = [r for r in scored if lo <= r["prob"] < hi
                     or (hi == 1.0 and r["prob"] == 1.0)]
        if in_bucket:
            calibration.append({
                "bucket": f"{lo:.1f}-{hi:.1f}", "n": len(in_bucket),
                "hit_rate": mean([r["outcome"] for r in in_bucket]),
            })

    return {
        "n": len(scored),
        "brier": ours,
        "log_score": mean([log_score(r["prob"], r["outcome"]) for r in scored]),
        "community_brier": theirs,
        "compared_on": len(paired),
        "beat_community": (None if theirs is None else ours_paired < theirs),
        "calibration": calibration,
    }


def run(cards: list[dict], ask, dry_run: bool = False) -> dict:
    """Score every card, or count them and spend nothing."""
    if dry_run:
        return {"rows": [], "summary": summarize([]), "would_score": len(cards)}
    crowd = build_crowd_for()
    rows = []
    for card in cards:
        try:
            rows.append(score_one(card, ask, crowd))
        except RuntimeError as exc:
            if tournament._is_budget_error(exc):
                print(f"budget reached after {len(rows)} questions; stopping")
                break
            raise
    return {"rows": rows, "summary": summarize(rows), "would_score": len(cards)}


def _write(result: dict, stamp: str) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"backtest_{stamp}.csv"
    json_path = OUT_DIR / f"backtest_{stamp}.json"
    columns = ["qid", "question", "qtype", "prob", "source", "outcome",
               "community", "brier", "log_score", "resolved_at", "research"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["rows"])
    json_path.write_text(json.dumps(result["summary"], indent=2))
    return csv_path, json_path


def inspect_report(payload: dict) -> str:
    """Plain description of what the API actually returned.

    Kept pure and separate from the API call so it can be tested, and
    written to a file as well as printed: workflow logs need admin
    rights to read, so a schema receipt nobody can open is no receipt.
    """
    results = payload.get("results", []) or []
    lines = [f"{len(results)} raw posts back"]
    if results:
        post = results[0]
        question = post.get("question") or {}
        lines += [
            f"post keys: {sorted(post.keys())}",
            f"question keys: {sorted(question.keys())}",
            f"resolution: {question.get('resolution')}",
            f"actual_resolve_time: {question.get('actual_resolve_time')}",
            f"aggregations present: {bool(question.get('aggregations'))}",
        ]
    lines.append(f"parsed -> {len(metaculus.parse_resolved(payload))} scoreable")
    return "\n".join(lines)


def probe_report(probes: list[dict]) -> str:
    """What each parameter shape actually returned.

    The first inspect came back with zero posts for minibench/resolved,
    which leaves the harness with no data source. Rather than guess at
    Metaculus's filter vocabulary, try the plausible shapes and write
    down what each one gave, so the answer is a receipt and not a theory.
    """
    lines = []
    for p in probes:
        line = f"{p['label']}: {p['count']} posts"
        if p.get("statuses_seen"):
            line += f"  statuses: {sorted(set(str(x) for x in p['statuses_seen']))}"
        if "scoreable" in p:
            line += f"  scoreable: {p['scoreable']}"
        if p.get("resolutions"):
            line += f"  resolutions: {p['resolutions']}"
        lines.append(line)
    if all(p["count"] == 0 for p in probes):
        lines.append("VERDICT: no resolved questions found by any filter; "
                     "the backtest has no data source yet")
    return "\n".join(lines)


def _use_own_spend_meter() -> None:
    """Point the engine's meter at the backtest's own file.

    A backtest is a burst of calls on demand. Metering it into the
    tournament's budget would let a curiosity run silence a live
    MiniBench round, which is the same failure the Kalshi loop had.
    """
    import os
    llm.SPEND_FILE = Path(os.environ.get("MANYWORLDZ_SPEND_FILE")
                          or config.DATA / "spend_backtest.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tournament", default=config.METACULUS_TOURNAMENT)
    parser.add_argument("--limit", type=int, default=25,
                        help="how many resolved questions to score")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan the run and spend nothing")
    parser.add_argument("--inspect", action="store_true",
                        help="one API call, print the raw shape, spend nothing")
    args = parser.parse_args(argv)

    import os
    _use_own_spend_meter()
    token = os.environ.get("METACULUS_TOKEN")
    if not token:
        print("METACULUS_TOKEN is not set; the API refuses anonymous reads")
        return 1

    if args.inspect:
        payload = metaculus._get_resolved_posts(args.tournament, token, 0)
        shapes = [
            ("minibench/resolved", {"tournaments": [args.tournament],
                                    "statuses": "resolved"}),
            ("minibench/closed", {"tournaments": [args.tournament],
                                  "statuses": "closed"}),
            ("minibench/any-status", {"tournaments": [args.tournament]}),
            ("any-tournament/resolved", {"statuses": "resolved"}),
        ]
        probes = []
        for label, params in shapes:
            try:
                raw = metaculus.get_posts_raw({**params, "limit": 20,
                                               "forecast_type": ["binary"],
                                               "include_description": "true"},
                                              token)
                results = raw.get("results", []) or []
                probes.append({
                    "label": label, "count": len(results),
                    "statuses_seen": [(r.get("question") or {}).get("status")
                                      or r.get("status") for r in results],
                    # the number that actually decides whether this shape
                    # is a data source: posts carrying a yes/no outcome
                    "scoreable": len(metaculus.parse_resolved(raw)),
                    "resolutions": [(r.get("question") or {}).get("resolution")
                                    for r in results][:6],
                })
            except Exception as exc:                      # noqa: BLE001
                probes.append({"label": label, "count": 0,
                               "statuses_seen": [f"ERROR {exc}"]})
        report = inspect_report(payload) + "\n\n" + probe_report(probes)

        # The resolution came back None even on posts the site itself
        # calls resolved, so the outcome is not in the list response.
        # Dump the real field names, and try the post detail endpoint,
        # which is the usual place a list view thins out.
        detail_lines = ["", "WHERE IS THE OUTCOME",
                        "id | type | status | resolved | resolution | "
                        "res_set_time | community"]
        try:
            raw = metaculus.get_posts_raw(
                {"statuses": "resolved", "forecast_type": ["binary"],
                 "limit": 10, "include_description": "true",
                 # Metaculus gates the community prediction behind an
                 # explicit flag; if resolution is gated the same way
                 # this is where it shows up.
                 "with_cp": "true"}, token)
            for post in (raw.get("results") or []):
                q = post.get("question") or {}
                agg = ((q.get("aggregations") or {}).get("recency_weighted")
                       or {}).get("latest") or {}
                centers = agg.get("centers")
                detail_lines.append(
                    f"{post.get('id')} | {q.get('type')} | {q.get('status')} | "
                    f"{post.get('resolved')!r} | {q.get('resolution')!r} | "
                    f"{q.get('resolution_set_time')!r} | "
                    f"perm={post.get('user_permission')!r} | "
                    f"{centers[0] if isinstance(centers, list) and centers else None}")
        except Exception as exc:                          # noqa: BLE001
            detail_lines.append(f"TABLE ERROR {exc}")

        # The ten above are one recent benchmark batch (ids 45xxx), which
        # may be exactly the set whose outcomes are withheld from bots.
        # Old ordinary questions are the control.
        detail_lines.append("CONTROL: old questions, oldest resolve first")
        try:
            old = metaculus.get_posts_raw(
                {"statuses": "resolved", "forecast_type": ["binary"],
                 "limit": 5, "order_by": "resolve_time",
                 "include_description": "true", "with_cp": "true"}, token)
            for post in (old.get("results") or []):
                q = post.get("question") or {}
                detail_lines.append(
                    f"{post.get('id')} | {q.get('resolution')!r} | "
                    f"{q.get('actual_resolve_time')!r} | {post.get('title','')[:40]}")
        except Exception as exc:                          # noqa: BLE001
            detail_lines.append(f"CONTROL ERROR {exc}")

        try:
            raw = metaculus.get_posts_raw(
                {"statuses": "resolved", "forecast_type": ["binary"],
                 "limit": 1, "include_description": "true"}, token)
            posts = raw.get("results", []) or []
            if posts:
                post, question = posts[0], (posts[0].get("question") or {})
                detail_lines += [
                    f"post id {post.get('id')} keys: {sorted(post.keys())}",
                    f"question keys: {sorted(question.keys())}",
                ]
                import requests
                url = f"https://www.metaculus.com/api/posts/{post.get('id')}/"
                resp = requests.get(url, headers=metaculus._headers(token),
                                    timeout=metaculus.TIMEOUT_S)
                detail_lines.append(f"detail GET {url} -> {resp.status_code}")
                if resp.ok:
                    detail = resp.json()
                    dq = detail.get("question") or {}
                    detail_lines += [
                        f"detail question keys: {sorted(dq.keys())}",
                        f"detail resolution: {dq.get('resolution')!r}",
                        f"detail actual_resolve_time: {dq.get('actual_resolve_time')!r}",
                    ]
        except Exception as exc:                          # noqa: BLE001
            detail_lines.append(f"ERROR {exc}")
        report += "\n".join(detail_lines)
        print(report)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "backtest_schema.txt").write_text(
            f"{datetime.now(timezone.utc).isoformat()}\n{report}\n")
        return 0

    fetched = metaculus.fetch_resolved_questions(args.tournament, token,
                                                 want=args.limit * 3)
    cards = after_cutoff(fetched)[:args.limit]
    print(f"{len(fetched)} resolved, {len(cards)} usable after the "
          f"{config.MODEL_CUTOFF_DATE} cutoff")
    if not cards:
        print("nothing to score")
        return 0

    result = run(cards, llm.ask, dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry run: would score {result['would_score']} questions, "
              f"no model calls made")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path, json_path = _write(result, stamp)
    s = result["summary"]
    print(f"scored {s['n']} | brier {s['brier']:.4f} | "
          f"log {s['log_score']:.4f}")
    if s["community_brier"] is not None:
        verdict = "BEAT" if s["beat_community"] else "lost to"
        print(f"{verdict} the community on {s['compared_on']} shared "
              f"questions: community brier {s['community_brier']:.4f}")
    print(f"wrote {csv_path.name} and {json_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
