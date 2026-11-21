import math

import pytest

import palettemap

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

PAL = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]


def _p():
    return palettemap.PaletteMap(PAL)


def _check_black():
    return _p().index_of((0.0, 0.0, 0.0)) == 0


def _check_white():
    return _p().index_of((1.0, 1.0, 1.0)) == 1


def _check_near_black():
    return _p().index_of((0.1, 0.1, 0.1)) == 0


def _check_near_white():
    return _p().index_of((0.9, 0.9, 0.9)) == 1


def _check_range():
    return 0 <= _p().index_of((0.5, 0.5, 0.5)) < 2


def _check_bad_palette():
    return _raises(lambda: palettemap.PaletteMap([]))


def _check_bad_color():
    return _raises(lambda: _p().index_of((0.1, 0.1)))


def _check_map_shape():
    out = _p().map_image([[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]])
    return out == [[0, 1]]


def _check_map_empty():
    return _p().map_image([]) == []


def _check_det():
    return _p().index_of((0.4, 0.4, 0.4)) == _p().index_of((0.4, 0.4, 0.4))


def _check_type():
    return isinstance(_p().index_of((0.2, 0.2, 0.2)), int)


def _check_no_mutation():
    img = [[(0.0, 0.0, 0.0)]]
    _p().map_image(img)
    return img == [[(0.0, 0.0, 0.0)]]


def _check_three_colors():
    p = palettemap.PaletteMap([(0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1)])
    return p.index_of((0.5, 0.5, 0.5)) == 1


def _check_map_len():
    return len(_p().map_image([[(0, 0, 0)] * 3])) == 1


def _check_float_out():
    return isinstance(_p().map_image([[(0, 0, 0)]])[0][0], int)

CASES = [("black", "_check_black"), ("white", "_check_white"), ("near_black", "_check_near_black"), ("near_white", "_check_near_white"), ("range", "_check_range"), ("bad_palette", "_check_bad_palette"), ("bad_color", "_check_bad_color"), ("map_shape", "_check_map_shape"), ("map_empty", "_check_map_empty"), ("det", "_check_det"), ("type", "_check_type"), ("no_mutation", "_check_no_mutation"), ("three_colors", "_check_three_colors"), ("map_len", "_check_map_len"), ("float_out", "_check_float_out")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
