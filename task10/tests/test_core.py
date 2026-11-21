import math

import pytest

import peakmeter

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _m():
    return peakmeter.PeakMeter()


def _check_initial():
    return _m().peak() == 0.0


def _check_positive():
    m = _m()
    m.push([0.5, 0.25])
    return m.peak() == 0.5


def _check_negative():
    m = _m()
    m.push([-0.75])
    return m.peak() == 0.75


def _check_max_kept():
    m = _m()
    m.push([0.9])
    m.push([0.1])
    return m.peak() == 0.9


def _check_reset():
    m = _m()
    m.push([0.9])
    m.reset()
    return m.peak() == 0.0


def _check_empty():
    return _m().peak() == 0.0


def _check_empty_push():
    m = _m()
    m.push([])
    return m.peak() == 0.0


def _check_bad_sample():
    m = _m()
    return _raises(lambda: m.push([float("nan")]))


def _check_type():
    m = _m()
    m.push([0.3])
    return isinstance(m.peak(), float)


def _check_det():
    m1 = _m()
    m1.push([0.4])
    m2 = _m()
    m2.push([0.4])
    return m1.peak() == m2.peak()


def _check_range():
    m = _m()
    m.push([1.0])
    return 0.0 <= m.peak() <= 1.0


def _check_multi():
    m = _m()
    m.push([0.1, 0.6, 0.2])
    return m.peak() == 0.6


def _check_no_state_leak():
    m = _m()
    m.push([0.5])
    m.reset()
    m.push([0.2])
    return m.peak() == 0.2


def _check_zero():
    m = _m()
    m.push([0.0])
    return m.peak() == 0.0


def _check_after_reset_type():
    m = _m()
    m.reset()
    return isinstance(m.peak(), float)

CASES = [("initial", "_check_initial"), ("positive", "_check_positive"), ("negative", "_check_negative"), ("max_kept", "_check_max_kept"), ("reset", "_check_reset"), ("empty", "_check_empty"), ("empty_push", "_check_empty_push"), ("bad_sample", "_check_bad_sample"), ("type", "_check_type"), ("det", "_check_det"), ("range", "_check_range"), ("multi", "_check_multi"), ("no_state_leak", "_check_no_state_leak"), ("zero", "_check_zero"), ("after_reset_type", "_check_after_reset_type")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
