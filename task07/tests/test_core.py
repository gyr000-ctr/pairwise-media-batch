import math

import pytest

import tilemap

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _t():
    return tilemap.TileMap(16.0)


def _check_origin():
    return _t().to_tile(0.0, 0.0) == (0, 0)


def _check_positive():
    return _t().to_tile(17.0, 33.0) == (1, 2)


def _check_negative():
    return _t().to_tile(-1.0, -17.0) == (-1, -2)


def _check_boundary():
    return _t().to_tile(16.0, 0.0) == (1, 0)


def _check_world():
    return _t().to_world(1, 2) == (16.0, 32.0)


def _check_roundtrip():
    tx, ty = _t().to_tile(35.5, -12.25)
    x, y = _t().to_world(tx, ty)
    return abs(x - 32.0) < 1e-9 and abs(y + 16.0) < 1e-9


def _check_bad_size():
    return _raises(lambda: tilemap.TileMap(0.0))


def _check_negative_size():
    return _raises(lambda: tilemap.TileMap(-1.0))


def _check_type():
    return isinstance(_t().to_tile(1.0, 1.0), tuple)


def _check_det():
    return _t().to_tile(5.0, 7.0) == _t().to_tile(5.0, 7.0)


def _check_float_world():
    return isinstance(_t().to_world(0, 0)[0], float)


def _check_large():
    return _t().to_tile(1e6, 1e6) == (62500, 62500)


def _check_center():
    return _t().to_tile(8.0, 8.0) == (0, 0)


def _check_two_cells():
    return _t().to_tile(31.9, 0.0) == (1, 0)


def _check_no_state():
    t = _t()
    t.to_tile(1.0, 1.0)
    return t.to_tile(2.0, 2.0) == (0, 0)

CASES = [("origin", "_check_origin"), ("positive", "_check_positive"), ("negative", "_check_negative"), ("boundary", "_check_boundary"), ("world", "_check_world"), ("roundtrip", "_check_roundtrip"), ("bad_size", "_check_bad_size"), ("negative_size", "_check_negative_size"), ("type", "_check_type"), ("det", "_check_det"), ("float_world", "_check_float_world"), ("large", "_check_large"), ("center", "_check_center"), ("two_cells", "_check_two_cells"), ("no_state", "_check_no_state")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
