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
