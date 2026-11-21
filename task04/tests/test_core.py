import math

import pytest

import easing

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _e():
    return easing.Easing()


def _check_linear_zero():
    return _e().apply("linear", 0.0) == 0.0


def _check_linear_one():
    return _e().apply("linear", 1.0) == 1.0


def _check_linear_mid():
    return abs(_e().apply("linear", 0.5) - 0.5) < 1e-9


def _check_ease_in_monotonic():
    a = _e().apply("ease_in", 0.2)
    b = _e().apply("ease_in", 0.6)
    return b >= a


def _check_ease_out():
    return _e().apply("ease_out", 1.0) == 1.0


def _check_bad_name():
    return _raises(lambda: _e().apply("bogus", 0.5))


def _check_bad_t():
    return _raises(lambda: _e().apply("linear", 1.5))


def _check_negative():
    return _raises(lambda: _e().apply("linear", -0.1))


def _check_det():
    return _e().apply("ease_in", 0.3) == _e().apply("ease_in", 0.3)


def _check_type():
    return isinstance(_e().apply("linear", 0.2), float)


def _check_range():
    return 0.0 <= _e().apply("ease_in_out", 0.7) <= 1.0


def _check_ease_in_slow():
    return _e().apply("ease_in", 0.25) < 0.25


def _check_ease_out_fast():
    return _e().apply("ease_out", 0.25) > 0.25


def _check_no_state():
    e = _e()
    e.apply("linear", 0.1)
    return e.apply("linear", 0.9) == 0.9


def _check_all_names():
    return all(_e().apply(n, 1.0) == 1.0 for n in ["linear", "ease_in", "ease_out", "ease_in_out"])

CASES = [("linear_zero", "_check_linear_zero"), ("linear_one", "_check_linear_one"), ("linear_mid", "_check_linear_mid"), ("ease_in_monotonic", "_check_ease_in_monotonic"), ("ease_out", "_check_ease_out"), ("bad_name", "_check_bad_name"), ("bad_t", "_check_bad_t"), ("negative", "_check_negative"), ("det", "_check_det"), ("type", "_check_type"), ("range", "_check_range"), ("ease_in_slow", "_check_ease_in_slow"), ("ease_out_fast", "_check_ease_out_fast"), ("no_state", "_check_no_state"), ("all_names", "_check_all_names")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
