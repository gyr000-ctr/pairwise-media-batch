import math

import pytest

import sorter

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _s():
    return sorter.Sorter()


A = {"id": "a", "layer": 1, "depth": 0.5}
B = {"id": "b", "layer": 0, "depth": 0.9}
C = {"id": "c", "layer": 1, "depth": 0.5}


def _check_layer_first():
    out = _s().sort([A, B])
    return [x["id"] for x in out] == ["b", "a"]


def _check_depth_second():
    d = {"id": "d", "layer": 1, "depth": 0.1}
    out = _s().sort([A, d])
    return [x["id"] for x in out] == ["d", "a"]


def _check_stable():
    out = _s().sort([A, C])
    return [x["id"] for x in out] == ["a", "c"]


def _check_empty():
    return _s().sort([]) == []


def _check_single():
    return _s().sort([A]) == [A]


def _check_bad_layer():
    return _raises(lambda: _s().sort([{"id": "x", "layer": -1, "depth": 0.0}]))


def _check_bad_item():
    return _raises(lambda: _s().sort([{"id": "x"}]))


def _check_len():
    return len(_s().sort([A, B, C])) == 3


def _check_det():
    return _s().sort([A, B, C]) == _s().sort([A, B, C])


def _check_type():
    return isinstance(_s().sort([A]), list)


def _check_no_mutation():
    items = [A, B]
    _s().sort(items)
    return [x["id"] for x in items] == ["a", "b"]


def _check_all_layers():
    out = _s().sort([A, B, C])
    return [x["layer"] for x in out] == [0, 1, 1]


def _check_two_layers():
    return [x["layer"] for x in _s().sort([A, B])] == [0, 1]


def _check_ids_preserved():
    return sorted(x["id"] for x in _s().sort([A, B, C])) == ["a", "b", "c"]


def _check_depth_order():
    x = {"id": "x", "layer": 2, "depth": 0.2}
    y = {"id": "y", "layer": 2, "depth": 0.8}
    return [i["id"] for i in _s().sort([y, x])] == ["x", "y"]

CASES = [("layer_first", "_check_layer_first"), ("depth_second", "_check_depth_second"), ("stable", "_check_stable"), ("empty", "_check_empty"), ("single", "_check_single"), ("bad_layer", "_check_bad_layer"), ("bad_item", "_check_bad_item"), ("len", "_check_len"), ("det", "_check_det"), ("type", "_check_type"), ("no_mutation", "_check_no_mutation"), ("all_layers", "_check_all_layers"), ("two_layers", "_check_two_layers"), ("ids_preserved", "_check_ids_preserved"), ("depth_order", "_check_depth_order")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
