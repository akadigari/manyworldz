"""Turn a handful of elicited percentiles into the full CDF Metaculus
wants for numeric and discrete questions.

Metaculus scores numeric forecasts as a probability distribution: a
list of cumulative probabilities, one per edge of the question's own
value grid (201 edges for the standard 200 bins). The model is only
asked for five percentiles; this file does the rest, the same way the
official forecasting-tools library does it:

1. Interpolate the percentiles into a raw CDF over the grid.
2. Standardize it with the affine map from forecasting-tools
   numeric_report.py (their _standardize_cdf), which guarantees the
   API's own rules: closed bounds hold all the mass, open bounds keep
   at least 0.1 percent outside, and every step is at least 0.01/200.
3. Flatten any spike whose single step would exceed the API's 0.2 cap.

No scipy, no numpy: plain monotone linear interpolation. PCHIP would
be smoother, but smoothness is not scored and one less dependency is.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The percentiles the model is asked for, and the JSON keys it answers
# with. Five points is the official template's own resolution.
PERCENTILE_KEYS = [("p05", 0.05), ("p25", 0.25), ("p50", 0.5),
                   ("p75", 0.75), ("p95", 0.95)]
MIN_STEP = 5e-05
# The API's per-step cap is 0.2 for the standard 200-bin grid and
# scales inversely with the bin count (forecasting-tools
# get_max_pmf_value): coarser discrete grids allow bigger steps.
MAX_STEP_200 = 0.2
# How much probability mass is spread flat across the grid purely to
# keep every step above MIN_STEP. The obvious value is 0.01, which puts
# a 200-bin grid at exactly 5e-05 per step, but "exactly" does not
# survive binary floating point: measured steps came out at
# 4.999999999999449e-05, and the API rejects a CDF whose step is under
# the floor by any amount at all. The extra 2 percent is the margin
# that makes the guarantee real, and it costs 0.02 percent of the mass.
FLOOR_MASS = 0.0102
# Open bounds must keep at least this much mass outside each bound.
EDGE_MASS = 0.001


def _grid(scaling: dict) -> list[float]:
    """The question's own value grid: one entry per CDF edge.

    Uses the API's `continuous_range` when the question carries it,
    otherwise an even grid from range_min to range_max with
    inbound_outcome_count bins (so count+1 edges).
    """
    given = scaling.get("continuous_range")
    if given:
        return [float(x) for x in given]
    lo = float(scaling["range_min"])
    hi = float(scaling["range_max"])
    n = int(scaling.get("inbound_outcome_count") or 200)
    zero = scaling.get("zero_point")
    if zero is None:
        return [lo + (hi - lo) * i / n for i in range(n + 1)]

    # Log-scaled question. Metaculus spaces the grid so that equal steps
    # in CDF location are equal RATIOS in value, which is why a linear
    # grid mis-shapes these badly: on a 10 to 1000 question the middle
    # edge sits at 100, not at 505. Transform copied from the official
    # template's _cdf_location_to_nominal_location.
    zero = float(zero)
    ratio = (hi - zero) / (lo - zero)
    return [lo + (hi - lo) * (ratio ** (i / n) - 1) / (ratio - 1)
            for i in range(n + 1)]


def _raw_cdf_at(x: float, points: list[tuple[float, float]]) -> float:
    """P(outcome <= x) from sorted (value, cumulative prob) points,
    linear between points, clamped to the elicited tails outside them."""
    if x <= points[0][0]:
        return points[0][1] if x == points[0][0] else 0.0
    if x >= points[-1][0]:
        return 1.0 if x > points[-1][0] else points[-1][1]
    for (v0, p0), (v1, p1) in zip(points, points[1:]):
        if v0 <= x <= v1:
            if v1 == v0:
                return p1
            return p0 + (p1 - p0) * (x - v0) / (v1 - v0)
    return 1.0


def build_cdf(percentiles: dict, scaling: dict,
              open_lower: bool, open_upper: bool) -> list[float]:
    """The 201-value (or grid-sized) CDF for one numeric question.

    `percentiles` maps cumulative probability (0.05 ... 0.95) to the
    model's predicted value at that percentile. Values are sorted
    before use so a slightly inconsistent elicitation can never
    produce a non-monotone CDF.
    """
    probs = sorted(percentiles.keys())
    values = sorted(float(percentiles[p]) for p in probs)
    points = list(zip(values, probs))
    grid = _grid(scaling)
    raw = [_raw_cdf_at(x, points) for x in grid]

    # forecasting-tools' _standardize_cdf affine, verbatim: rescale the
    # inbound mass, then map so bounds and the minimum step both hold.
    scale_lo = 0.0 if open_lower else raw[0]
    scale_hi = 1.0 if open_upper else raw[-1]
    mass = (scale_hi - scale_lo) or 1.0
    last = len(grid) - 1

    # Enough flat mass that every step clears MIN_STEP with margin, on
    # this question's own grid however many bins it has.
    floor = max(FLOOR_MASS, last * MIN_STEP * 1.02)
    edges = EDGE_MASS * (int(open_lower) + int(open_upper))

    def apply_minimum(f: float, loc: float) -> float:
        rescaled = (f - scale_lo) / mass
        body = (1.0 - floor - edges) * rescaled + floor * loc
        return body + EDGE_MASS if open_lower else body

    out = [apply_minimum(v, i / last) for i, v in enumerate(raw)]
    out = [min(max(v, 0.0), 1.0) for v in out]
    for i in range(1, len(out)):             # monotone after clamping
        out[i] = max(out[i], out[i - 1])
    if not open_lower:
        out[0] = 0.0
    if not open_upper:
        out[-1] = 1.0

    # A spike forecast can still break the API's per-step cap. Blend
    # toward the uniform CDF until every step fits; each pass keeps
    # the endpoints and monotonicity intact.
    max_step = min(MAX_STEP_200 * 200 / last, 1.0)
    uniform = [out[0] + (out[-1] - out[0]) * i / last
               for i in range(len(out))]
    for _ in range(50):
        steps = [b - a for a, b in zip(out, out[1:])]
        # the 0.999 margin keeps the final rounding from landing a step
        # exactly on the cap
        if max(steps) <= max_step * 0.999:
            break
        out = [0.8 * v + 0.2 * u for v, u in zip(out, uniform)]
    return [round(v, 7) for v in out]


def percentiles_from_json(text: str) -> dict | None:
    """Parse the model's percentile reply into {prob: value}.

    Expects JSON with keys p05, p25, p50, p75, p95 somewhere in the
    reply. Values are sorted ascending across the percentiles so the
    CDF stays monotone even if the model's numbers cross. None for
    anything unusable: a garbage reply is a skipped answer, never a
    guessed one.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    try:
        raw = [float(parsed[key]) for key, _ in PERCENTILE_KEYS]
    except (KeyError, TypeError, ValueError):
        return None
    ordered = sorted(raw)
    return {prob: value
            for (_, prob), value in zip(PERCENTILE_KEYS, ordered)}


def cdf_problems(values: list[float], open_lower: bool,
                 open_upper: bool) -> list[str]:
    """Every reason Metaculus would reject this CDF, in plain English.

    Called before a numeric submission goes out. The API validates all
    of this server side, and a rejected submission is a scored zero on
    a question the crowd already paid to think about, so it is worth
    the microsecond to check first and fall back deliberately instead.
    Constraints mirror NumericDistribution's validators in the official
    template. An empty list means it is safe to send.
    """
    problems: list[str] = []
    if len(values) < 2:
        problems.append(f"a CDF needs at least 2 values, got {len(values)}")
        return problems

    steps = [b - a for a, b in zip(values, values[1:])]
    smallest = min(steps)
    if smallest < MIN_STEP:
        problems.append(
            f"step {smallest:.3e} is under the API minimum of {MIN_STEP:.0e}; "
            "the CDF must rise at every single edge")

    cap = MAX_STEP_200 * 200 / (len(values) - 1)
    largest = max(steps)
    if largest > cap:
        problems.append(
            f"step {largest:.4f} is over the API cap of {cap:.4f}; "
            "the distribution is too tall for this grid")

    if not open_lower and abs(values[0]) > 1e-9:
        problems.append(
            f"closed lower bound must start at 0, starts at {values[0]:.6f}")
    if not open_upper and abs(values[-1] - 1.0) > 1e-9:
        problems.append(
            f"closed upper bound must end at 1, ends at {values[-1]:.6f}")
    if open_lower and values[0] < EDGE_MASS - 1e-9:
        problems.append(
            f"open lower bound needs at least {EDGE_MASS} mass below it, "
            f"has {values[0]:.6f}")
    if open_upper and values[-1] > 1.0 - EDGE_MASS + 1e-9:
        problems.append(
            f"open upper bound needs at least {EDGE_MASS} mass above it, "
            f"ends at {values[-1]:.6f}")
    return problems
