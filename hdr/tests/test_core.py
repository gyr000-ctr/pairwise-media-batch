import math

import pytest

import hdr

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _t():
    return hdr.ToneMap()


def _check_zero():
    return _t().tonemap(0.0) == 0.0


def _check_monotonic():
    return _t().tonemap(1.0) <= _t().tonemap(4.0)


def _check_bounded():
    return _t().tonemap(100.0) <= 1.0


def _check_negative():
    return _raises(lambda: _t().tonemap(-1.0))


def _check_exposure():
    a = _t().tonemap(1.0, 1.0)
    b = _t().tonemap(1.0, 2.0)
    return b >= a


def _check_bad_exposure():
    return _raises(lambda: _t().tonemap(1.0, 0.0))


def _check_apply_shape():
    out = _t().apply([[1.0, 2.0]])
    return len(out) == 1 and len(out[0]) == 2


def _check_apply_range():
    out = _t().apply([[0.5, 8.0]])
    return all(0.0 <= v <= 1.0 for v in out[0])


def _check_apply_empty():
    return _t().apply([]) == []


def _check_det():
    return _t().tonemap(3.0) == _t().tonemap(3.0)


def _check_type():
    return isinstance(_t().tonemap(1.0), float)


def _check_no_mutation():
    img = [[2.0]]
    _t().apply(img)
    return img == [[2.0]]


def _check_small():
    return _t().tonemap(0.5) > 0.0


def _check_large():
    return _t().tonemap(1000.0) <= 1.0


def _check_two_values():
    a = _t().tonemap(0.5)
    b = _t().tonemap(1.5)
    return b > a

CASES = [("zero", "_check_zero"), ("monotonic", "_check_monotonic"), ("bounded", "_check_bounded"), ("negative", "_check_negative"), ("exposure", "_check_exposure"), ("bad_exposure", "_check_bad_exposure"), ("apply_shape", "_check_apply_shape"), ("apply_range", "_check_apply_range"), ("apply_empty", "_check_apply_empty"), ("det", "_check_det"), ("type", "_check_type"), ("no_mutation", "_check_no_mutation"), ("small", "_check_small"), ("large", "_check_large"), ("two_values", "_check_two_values")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
