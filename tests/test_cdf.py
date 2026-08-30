"""engine/cdf.py: turn a handful of elicited percentiles into the
201-value CDF Metaculus wants, honoring the question's own bounds."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import cdf as cdf_mod

SCALING = {"range_min": 0, "range_max": 60, "inbound_outcome_count": 200,
           "continuous_range": None}
PCTS = {0.05: 5, 0.25: 15, 0.5: 30, 0.75: 45, 0.95: 55}


def test_cdf_has_201_monotone_values_in_range():
    out = cdf_mod.build_cdf(PCTS, SCALING, open_lower=False, open_upper=False)
    assert len(out) == 201
    assert all(0.0 <= v <= 1.0 for v in out)
    assert all(a <= b for a, b in zip(out, out[1:]))


def test_cdf_endpoints_respect_closed_bounds():
    out = cdf_mod.build_cdf(PCTS, SCALING, open_lower=False, open_upper=False)
    assert out[0] == 0.0
    assert out[-1] == 1.0


def test_cdf_endpoints_leave_mass_outside_open_bounds():
    out = cdf_mod.build_cdf(PCTS, SCALING, open_lower=True, open_upper=True)
    assert 0.0009 <= out[0] <= 0.05
    assert 0.95 <= out[-1] <= 0.9991


def test_cdf_median_lands_near_the_elicited_median():
    out = cdf_mod.build_cdf(PCTS, SCALING, open_lower=False, open_upper=False)
    # value 30 on a 0-60 grid of 201 edges is index 100
    assert abs(out[100] - 0.5) < 0.03


def test_cdf_never_steps_less_than_the_api_minimum():
    # a spike forecast: all percentiles piled on one value
    spike = {0.05: 30, 0.25: 30, 0.5: 30, 0.75: 30, 0.95: 30}
    out = cdf_mod.build_cdf(spike, SCALING, open_lower=True, open_upper=True)
    steps = [b - a for a, b in zip(out, out[1:])]
    assert min(steps) >= 0.01 / 200 - 1e-12


def test_cdf_never_steps_more_than_the_api_maximum():
    spike = {0.05: 30, 0.25: 30, 0.5: 30, 0.75: 30, 0.95: 30}
    out = cdf_mod.build_cdf(spike, SCALING, open_lower=True, open_upper=True)
    steps = [b - a for a, b in zip(out, out[1:])]
    assert max(steps) <= 0.2 + 1e-12


def test_cdf_uses_the_questions_own_continuous_range_when_given():
    scaling = {"range_min": 0, "range_max": 60, "inbound_outcome_count": 4,
               "continuous_range": [0, 10, 20, 40, 60]}
    out = cdf_mod.build_cdf(PCTS, scaling, open_lower=False, open_upper=False)
    assert len(out) == 5   # one value per edge in the question's own grid


def test_percentiles_from_json_parses_and_sorts():
    text = ('here you go {"p05": 5, "p25": 20, "p50": 15, "p75": 45, '
            '"p95": 55}')
    got = cdf_mod.percentiles_from_json(text)
    # p50 below p25 in the reply: values are sorted so the CDF stays monotone
    assert list(got.keys()) == [0.05, 0.25, 0.5, 0.75, 0.95]
    assert list(got.values()) == [5, 15, 20, 45, 55]


def test_percentiles_from_json_rejects_garbage():
    assert cdf_mod.percentiles_from_json("no json here") is None
    assert cdf_mod.percentiles_from_json('{"p05": "many"}') is None


def test_step_cap_scales_with_the_questions_own_bin_count():
    """Review finding: the API cap is 0.2 * 200 / bins, so a coarse
    discrete grid allows bigger steps and must not be over-flattened."""
    scaling = {"range_min": 0, "range_max": 10, "inbound_outcome_count": 5,
               "continuous_range": None}
    spike = {0.05: 4, 0.25: 4.5, 0.5: 5, 0.75: 5.5, 0.95: 6}
    out = cdf_mod.build_cdf(spike, scaling, open_lower=True, open_upper=True)
    steps = [b - a for a, b in zip(out, out[1:])]
    # 0.2 * 200 / 5 = 8: effectively uncapped, so the middle step keeps
    # nearly all the mass instead of being blended flat
    assert max(steps) > 0.5


# --- log-scaled questions -------------------------------------------------
# Metaculus puts the 201 CDF points at log-spaced values when a question
# carries a zero_point. Building them on a linear grid submits a badly
# mis-shaped distribution, so these lock the official transform in place.
# Reference: metac-bot-template main_with_no_framework.py,
# _cdf_location_to_nominal_location.

LOG_SCALING = {"range_min": 10, "range_max": 1000, "zero_point": 0,
               "inbound_outcome_count": 200, "continuous_range": None}


def _official_nominal(location, range_min, range_max, zero_point):
    deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
    return range_min + (range_max - range_min) * (
        deriv_ratio ** location - 1) / (deriv_ratio - 1)


def test_log_scaled_grid_matches_the_official_transform():
    grid = cdf_mod._grid(LOG_SCALING)
    assert len(grid) == 201
    assert grid[0] == 10 and abs(grid[-1] - 1000) < 1e-9
    for i in (1, 50, 100, 150, 199):
        expected = _official_nominal(i / 200, 10, 1000, 0)
        assert abs(grid[i] - expected) < 1e-9, f"edge {i}"


def test_log_scaled_grid_is_not_the_linear_one():
    # The whole point: on a log grid the midpoint sits near 100, not 505.
    grid = cdf_mod._grid(LOG_SCALING)
    assert abs(grid[100] - 100) < 1.0
    assert all(b > a for a, b in zip(grid, grid[1:]))


def test_log_scaled_cdf_is_submittable():
    pcts = {0.05: 20, 0.25: 60, 0.5: 120, 0.75: 300, 0.95: 800}
    out = cdf_mod.build_cdf(pcts, LOG_SCALING, open_lower=False, open_upper=False)
    assert cdf_mod.cdf_problems(out, open_lower=False, open_upper=False) == []


# --- the pre-submit guard -------------------------------------------------
# Every constraint the API enforces, checked before we spend a submission
# on a forecast it will reject. Reference: NumericDistribution's validators.

def test_cdf_problems_passes_a_good_cdf():
    out = cdf_mod.build_cdf(PCTS, SCALING, open_lower=False, open_upper=False)
    assert cdf_mod.cdf_problems(out, open_lower=False, open_upper=False) == []


def test_cdf_problems_catches_a_step_under_the_api_minimum():
    flat = [0.0] + [0.5] * 199 + [1.0]      # 199 steps of exactly zero
    problems = cdf_mod.cdf_problems(flat, open_lower=False, open_upper=False)
    assert any("5e-05" in p or "minimum" in p for p in problems)


def test_cdf_problems_catches_a_spike_over_the_api_cap():
    spike = [0.0] * 100 + [1.0] * 101
    problems = cdf_mod.cdf_problems(spike, open_lower=False, open_upper=False)
    assert any("cap" in p or "0.2" in p for p in problems)


def test_cdf_problems_catches_closed_bounds_that_do_not_hold_all_mass():
    out = cdf_mod.build_cdf(PCTS, SCALING, open_lower=True, open_upper=True)
    # a genuinely open-bounded CDF, judged against closed bounds, must complain
    problems = cdf_mod.cdf_problems(out, open_lower=False, open_upper=False)
    assert problems


def test_every_built_cdf_clears_the_minimum_step_after_rounding():
    # Rounding the final values can shave a step that sat exactly on the
    # 5e-05 floor, which the API rejects outright.
    for open_lo in (True, False):
        for open_hi in (True, False):
            for pcts in (PCTS, {0.05: 29.9, 0.25: 30.0, 0.5: 30.0,
                                0.75: 30.0, 0.95: 30.1}):
                out = cdf_mod.build_cdf(pcts, SCALING, open_lo, open_hi)
                steps = [b - a for a, b in zip(out, out[1:])]
                assert min(steps) >= 5e-05, (
                    f"open_lo={open_lo} open_hi={open_hi} min step {min(steps)}")
