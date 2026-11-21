import math

import pytest

import adsr

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _a():
    return adsr.Adsr(1000, 0.1, 0.1, 0.5, 0.1)


def _check_len():
    return len(_a().render(400)) == 400


def _check_attack_rise():
    out = _a().render(100)
    return out[-1] > out[0]


def _check_peak():
    return abs(max(_a().render(400)) - 1.0) < 0.05


def _check_release_zero():
    return _a().render(400)[-1] < 1e-9


def _check_sustain_level():
    return abs(_a().render(400)[150] - 0.5) < 0.05


def _check_bad_rate():
    return _raises(lambda: adsr.Adsr(0, 0.1, 0.1, 0.5, 0.1))


def _check_bad_sustain():
    return _raises(lambda: adsr.Adsr(1000, 0.1, 0.1, 1.5, 0.1))


def _check_negative_attack():
    return _raises(lambda: adsr.Adsr(1000, -0.1, 0.1, 0.5, 0.1))


def _check_monotonic_attack():
    out = _a().render(100)
    return all(out[i] <= out[i + 1] + 1e-9 for i in range(99))


def _check_monotonic_release():
    out = _a().render(400)
    return all(out[i] >= out[i + 1] - 1e-9 for i in range(300, 399))


def _check_det():
    return _a().render(200) == _a().render(200)


def _check_value_at():
    return abs(_a().value_at(0.05) - 0.5) < 0.05


def _check_range():
    return all(0.0 <= v <= 1.0 for v in _a().render(400))


def _check_zero_attack():
    return _a().render(10)[0] > 0.0


def _check_no_jump():
    out = _a().render(400)
    return max(abs(out[i + 1] - out[i]) for i in range(399)) < 0.1

CASES = [("len", "_check_len"), ("attack_rise", "_check_attack_rise"), ("peak", "_check_peak"), ("release_zero", "_check_release_zero"), ("sustain_level", "_check_sustain_level"), ("bad_rate", "_check_bad_rate"), ("bad_sustain", "_check_bad_sustain"), ("negative_attack", "_check_negative_attack"), ("monotonic_attack", "_check_monotonic_attack"), ("monotonic_release", "_check_monotonic_release"), ("det", "_check_det"), ("value_at", "_check_value_at"), ("range", "_check_range"), ("zero_attack", "_check_zero_attack"), ("no_jump", "_check_no_jump")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
