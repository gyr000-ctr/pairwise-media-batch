import math

import pytest

import gradient

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

STOPS = [(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))]


def _g():
    return gradient.Gradient(STOPS)


def _check_start():
    return _g().sample(0.0) == (0.0, 0.0, 0.0)


def _check_end():
    return _g().sample(1.0) == (1.0, 1.0, 1.0)


def _check_mid():
    return abs(_g().sample(0.5)[0] - 0.5) < 1e-9


def _check_out_of_range():
    return _raises(lambda: _g().sample(1.5))


def _check_negative():
    return _raises(lambda: _g().sample(-0.1))


def _check_unsorted():
    return _raises(lambda: gradient.Gradient([(1.0, (1, 1, 1)), (0.0, (0, 0, 0))]))


def _check_empty():
    return _raises(lambda: gradient.Gradient([]))


def _check_add_stop():
    g = _g()
    g.add_stop(0.5, (0.5, 0.5, 0.5))
    return abs(g.sample(0.5)[0] - 0.5) < 1e-9


def _check_add_out_of_order():
    return _raises(lambda: _g().add_stop(1.5, (0, 0, 0)))


def _check_det():
    return _g().sample(0.3) == _g().sample(0.3)


def _check_type():
    return isinstance(_g().sample(0.2), tuple)


def _check_range():
    return all(0.0 <= v <= 1.0 for v in _g().sample(0.75))


def _check_three_stops():
    g = gradient.Gradient([(0.0, (0, 0, 0)), (0.5, (1, 0, 0)), (1.0, (1, 1, 1))])
    return abs(g.sample(0.25)[0] - 0.5) < 1e-9


def _check_no_mutation():
    src = list(STOPS)
    _g().sample(0.5)
    return STOPS == src


def _check_single_stop():
    return gradient.Gradient([(0.0, (0.3, 0.3, 0.3))]).sample(0.0) == (0.3, 0.3, 0.3)

CASES = [("start", "_check_start"), ("end", "_check_end"), ("mid", "_check_mid"), ("out_of_range", "_check_out_of_range"), ("negative", "_check_negative"), ("unsorted", "_check_unsorted"), ("empty", "_check_empty"), ("add_stop", "_check_add_stop"), ("add_out_of_order", "_check_add_out_of_order"), ("det", "_check_det"), ("type", "_check_type"), ("range", "_check_range"), ("three_stops", "_check_three_stops"), ("no_mutation", "_check_no_mutation"), ("single_stop", "_check_single_stop")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
