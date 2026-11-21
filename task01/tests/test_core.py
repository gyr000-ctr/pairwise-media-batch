import math

import pytest

import histeq

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[0.0, 0.5], [0.5, 1.0]]


def _check_range():
    out = histeq.HistEq().equalize(IMG)
    return all(0.0 <= v <= 1.0 for row in out for v in row)


def _check_shape():
    out = histeq.HistEq().equalize(IMG)
    return len(out) == 2 and len(out[0]) == 2


def _check_constant():
    return histeq.HistEq().equalize([[0.3, 0.3]]) == [[0.3, 0.3]]


def _check_monotonic():
    out = histeq.HistEq().equalize([[0.0, 0.2, 0.8, 1.0]])
    return all(out[0][i] <= out[0][i + 1] + 1e-9 for i in range(3))


def _check_full_range():
    out = histeq.HistEq().equalize([[0.0, 1.0]])
    return abs(out[0][0]) < 1e-9 and abs(out[0][1] - 1.0) < 1e-9


def _check_empty():
    return _raises(lambda: histeq.HistEq().equalize([]))


def _check_bad_value():
    return _raises(lambda: histeq.HistEq().equalize([[1.5]]))


def _check_det():
    return histeq.HistEq().equalize(IMG) == histeq.HistEq().equalize(IMG)


def _check_type():
    return isinstance(histeq.HistEq().equalize(IMG), list)


def _check_no_mutation():
    src = [row[:] for row in IMG]
    histeq.HistEq().equalize(IMG)
    return IMG == src


def _check_single():
    return histeq.HistEq().equalize([[0.5]]) == [[0.5]]


def _check_ragged():
    return _raises(lambda: histeq.HistEq().equalize([[0.1, 0.2], [0.3]]))


def _check_two_rows():
    return len(histeq.HistEq().equalize([[0.0], [1.0]])) == 2


def _check_float():
    return isinstance(histeq.HistEq().equalize(IMG)[0][0], float)


def _check_midpoint():
    out = histeq.HistEq().equalize([[0.0, 0.5, 1.0]])
    return abs(out[0][1] - 0.5) <= 0.5

CASES = [("range", "_check_range"), ("shape", "_check_shape"), ("constant", "_check_constant"), ("monotonic", "_check_monotonic"), ("full_range", "_check_full_range"), ("empty", "_check_empty"), ("bad_value", "_check_bad_value"), ("det", "_check_det"), ("type", "_check_type"), ("no_mutation", "_check_no_mutation"), ("single", "_check_single"), ("ragged", "_check_ragged"), ("two_rows", "_check_two_rows"), ("float", "_check_float"), ("midpoint", "_check_midpoint")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
