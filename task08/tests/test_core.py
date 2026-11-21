import math

import pytest

import fixedstep

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _f():
    return fixedstep.FixedStep(0.016, 5)


def _check_none():
    return _f().advance(0.008) == 0


def _check_one():
    return _f().advance(0.016) == 1


def _check_two():
    return _f().advance(0.032) == 2


def _check_accumulate():
    f = _f()
    f.advance(0.008)
    return f.advance(0.008) == 1


def _check_max_steps():
    return _f().advance(1.0) == 5


def _check_zero():
    return _f().advance(0.0) == 0


def _check_negative():
    return _raises(lambda: _f().advance(-0.016))


def _check_bad_step():
    return _raises(lambda: fixedstep.FixedStep(0.0))


def _check_bad_max():
    return _raises(lambda: fixedstep.FixedStep(0.016, 0))


def _check_type():
    return isinstance(_f().advance(0.016), int)


def _check_det():
    return _f().advance(0.032) == _f().advance(0.032)


def _check_no_state():
    f = _f()
    f.advance(0.016)
    return f.advance(0.016) == 1


def _check_float_dt():
    return _f().advance(0.0159) == 0


def _check_large_dt():
    return _f().advance(10.0) == 5


def _check_reset_by_max():
    f = fixedstep.FixedStep(0.1, 3)
    return f.advance(1.0) == 3

CASES = [("none", "_check_none"), ("one", "_check_one"), ("two", "_check_two"), ("accumulate", "_check_accumulate"), ("max_steps", "_check_max_steps"), ("zero", "_check_zero"), ("negative", "_check_negative"), ("bad_step", "_check_bad_step"), ("bad_max", "_check_bad_max"), ("type", "_check_type"), ("det", "_check_det"), ("no_state", "_check_no_state"), ("float_dt", "_check_float_dt"), ("large_dt", "_check_large_dt"), ("reset_by_max", "_check_reset_by_max")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
