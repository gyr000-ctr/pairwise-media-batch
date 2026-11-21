import math

import pytest

import noise

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _n(seed=1):
    return noise.ValueNoise(seed)


def _check_range():
    return 0.0 <= _n().sample(0.5, 0.5) <= 1.0


def _check_deterministic():
    return _n(7).sample(1.2, 3.4) == _n(7).sample(1.2, 3.4)


def _check_different_seed():
    return _n(1).sample(1.2, 3.4) != _n(2).sample(1.2, 3.4)


def _check_grid_point():
    a = _n().sample(0.0, 0.0)
    b = _n().sample(0.0, 0.0)
    return a == b


def _check_continuity():
    a = _n().sample(0.5, 0.5)
    b = _n().sample(0.5001, 0.5)
    return abs(a - b) < 1e-3


def _check_bad_seed():
    return _raises(lambda: noise.ValueNoise(-1))


def _check_type():
    return isinstance(_n().sample(0.1, 0.1), float)


def _check_two_points():
    return _n().sample(0.1, 0.1) != _n().sample(5.1, 5.1)


def _check_negative_coords():
    return 0.0 <= _n().sample(-3.2, -7.8) <= 1.0


def _check_large_coords():
    return 0.0 <= _n().sample(1000.5, 2000.5) <= 1.0


def _check_int_coords():
    return 0.0 <= _n().sample(3.0, 4.0) <= 1.0


def _check_no_state():
    n = _n()
    n.sample(1.0, 1.0)
    return n.sample(2.0, 2.0) == _n().sample(2.0, 2.0)


def _check_float():
    return isinstance(_n().sample(2.5, 2.5), float)


def _check_repeat_calls():
    n = _n()
    return n.sample(0.3, 0.7) == n.sample(0.3, 0.7)


def _check_adjacent():
    return abs(_n().sample(1.0, 1.0) - _n().sample(1.0001, 1.0)) < 1e-2

CASES = [("range", "_check_range"), ("deterministic", "_check_deterministic"), ("different_seed", "_check_different_seed"), ("grid_point", "_check_grid_point"), ("continuity", "_check_continuity"), ("bad_seed", "_check_bad_seed"), ("type", "_check_type"), ("two_points", "_check_two_points"), ("negative_coords", "_check_negative_coords"), ("large_coords", "_check_large_coords"), ("int_coords", "_check_int_coords"), ("no_state", "_check_no_state"), ("float", "_check_float"), ("repeat_calls", "_check_repeat_calls"), ("adjacent", "_check_adjacent")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
