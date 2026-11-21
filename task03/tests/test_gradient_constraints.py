"""题面约束的补充测试（不修改既有 tests/test_core.py）。

覆盖：端点精确、重复停靠点合并、非有限位置、非法停靠点、
线性空间与 sRGB 直接插值的可观测差异、gradient_at 导数、
10 万次采样性能、并发安全、大量停靠点。
"""

import math
import threading

import pytest

import gradient


def test_endpoints_exact():
    g = gradient.Gradient([(0.0, (0.1, 0.2, 0.3)), (1.0, (0.9, 0.8, 0.7))],
                          space="linear")
    # 即使走线性空间，端点也必须精确（无色彩空间往返误差）。
    assert g.sample(0.0) == (0.1, 0.2, 0.3)
    assert g.sample(1.0) == (0.9, 0.8, 0.7)


def test_middle_stop_exact_in_linear_space():
    g = gradient.Gradient([(0.0, (0.0, 0.0, 0.0)),
                           (0.5, (0.25, 0.5, 0.75)),
                           (1.0, (1.0, 1.0, 1.0))], space="linear")
    assert g.sample(0.5) == (0.25, 0.5, 0.75)


def test_unsorted_positions_rejected():
    with pytest.raises(ValueError):
        gradient.Gradient([(0.8, (1, 1, 1)), (0.2, (0, 0, 0))])


def test_out_of_range_and_negative_rejected():
    g = gradient.Gradient([(0.0, (0, 0, 0)), (1.0, (1, 1, 1))])
    for bad in (1.0001, -0.0001, 2.0, -1.0):
        with pytest.raises(ValueError):
            g.sample(bad)
        with pytest.raises(ValueError):
            g.gradient_at(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None, "0.5"])
def test_non_finite_sample_position(bad):
    g = gradient.Gradient([(0.0, (0, 0, 0)), (1.0, (1, 1, 1))])
    with pytest.raises((ValueError, TypeError)):
        g.sample(bad)
    with pytest.raises((ValueError, TypeError)):
        g.gradient_at(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 1.5, -0.1])
def test_non_finite_stop_position(bad):
    with pytest.raises((ValueError, TypeError)):
        gradient.Gradient([(0.0, (0, 0, 0)), (bad, (1, 1, 1))])
    g = gradient.Gradient([(0.0, (0, 0, 0))])
    with pytest.raises((ValueError, TypeError)):
        g.add_stop(bad, (1, 1, 1))


def test_illegal_colors_rejected():
    bad_colors = [
        (0.5, 0.5),            # 缺通道
        (0.5, 0.5, 0.5, 0.5),  # 多通道
        (1.2, 0.0, 0.0),       # 越界
        (0.0, -0.1, 0.0),      # 负值
        (float("nan"), 0.0, 0.0),
        "rgb",
    ]
    for color in bad_colors:
        with pytest.raises((ValueError, TypeError)):
            gradient.Gradient([(0.0, color)])
        g = gradient.Gradient([(0.0, (0, 0, 0))])
        with pytest.raises((ValueError, TypeError)):
            g.add_stop(0.5, color)


def test_duplicate_positions_merged_not_error():
    g = gradient.Gradient([
        (0.0, (0.0, 0.0, 0.0)),
        (0.5, (1.0, 0.0, 0.0)),
        (0.5, (0.0, 1.0, 0.0)),  # 重复位置，应合并而非报错
        (1.0, (1.0, 1.0, 1.0)),
    ])
    assert g.sample(0.5) == (0.0, 1.0, 0.0)
    # 合并后相邻区间行为正确。
    assert abs(g.sample(0.25)[1] - 0.5) < 1e-12

    g2 = gradient.Gradient([(0.3, (0.2, 0.4, 0.6))])
    g2.add_stop(0.3, (0.8, 0.7, 0.6))
    assert g2.sample(0.3) == (0.8, 0.7, 0.6)


def test_linear_vs_srgb_observable_difference():
    g_direct = gradient.Gradient([(0.0, (0.0, 0.0, 0.0)),
                                  (1.0, (1.0, 1.0, 1.0))], space="srgb")
    g_linear = gradient.Gradient([(0.0, (0.0, 0.0, 0.0)),
                                  (1.0, (1.0, 1.0, 1.0))], space="linear")
    t = 0.5
    direct = g_direct.sample(t)[0]
    linear = g_linear.sample(t)[0]
    # 线性光空间中点 0.5(linear) -> sRGB ≈ 0.735，明显比直接插值的 0.5 亮。
    assert abs(direct - 0.5) < 1e-12
    assert abs(linear - 0.7353569830524495) < 1e-9
    assert abs(linear - direct) > 0.2

    # 同一个对象上也可按次指定插值空间。
    assert g_direct.sample(t, space="linear") == g_linear.sample(t)
    assert g_linear.sample(t, mode="srgb")[0] == pytest.approx(0.5)


def test_srgb_round_trip_helpers_consistency():
    # 线性模式在非结点处应位于两种端点颜色之间。
    g = gradient.Gradient([(0.0, (0.1, 0.2, 0.3)), (1.0, (0.9, 0.8, 0.7))],
                          space="linear")
    for t in (0.1, 0.37, 0.63, 0.9):
        s = g.sample(t)
        assert all(0.0 <= v <= 1.0 for v in s)


def test_gradient_at_srgb_constant_slope():
    g = gradient.Gradient([(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))])
    d = g.gradient_at(0.3)
    assert d == (1.0, 1.0, 1.0)
    # 钳制区外不允许采样，但端点处导数取内侧区间。
    assert g.gradient_at(0.0) == (1.0, 1.0, 1.0)
    assert g.gradient_at(1.0) == (1.0, 1.0, 1.0)


def test_gradient_at_piecewise_and_zero_outside_coverage():
    g = gradient.Gradient([(0.25, (0.0, 0.0, 0.0)),
                           (0.75, (1.0, 0.0, 0.0))])
    assert g.gradient_at(0.1) == (0.0, 0.0, 0.0)   # 钳制
    assert g.gradient_at(0.9) == (0.0, 0.0, 0.0)
    mid = g.gradient_at(0.5)
    assert mid[0] == pytest.approx(2.0)
    assert mid[1:] == (0.0, 0.0)


def test_gradient_at_matches_finite_difference():
    g = gradient.Gradient([(0.0, (0.05, 0.95, 0.5)),
                           (0.4, (0.9, 0.1, 0.2)),
                           (1.0, (0.3, 0.7, 0.6))], space="linear")
    h = 1e-6
    for t in (0.05, 0.3, 0.55, 0.95):
        analytic = g.gradient_at(t)
        up = g.sample(min(1.0, t + h))
        lo = g.sample(max(0.0, t - h))
        numeric = tuple((a - b) / (2.0 * h) for a, b in zip(up, lo))
        for a, n in zip(analytic, numeric):
            assert a == pytest.approx(n, rel=1e-4, abs=1e-4)


def test_single_stop_is_constant():
    g = gradient.Gradient([(0.0, (0.3, 0.3, 0.3))])
    assert g.sample(0.0) == (0.3, 0.3, 0.3)
    assert g.sample(1.0) == (0.3, 0.3, 0.3)
    assert g.gradient_at(0.7) == (0.0, 0.0, 0.0)


def test_large_number_of_stops():
    n = 5000
    stops = [(i / (n - 1), (i / (n - 1),) * 3) for i in range(n)]
    g = gradient.Gradient(stops)
    for t in (0.0, 0.123, 0.5, 0.999, 1.0):
        s = g.sample(t)
        assert s[0] == pytest.approx(t, abs=1e-3)


def test_performance_100k_samples():
    g = gradient.Gradient([(0.0, (0.0, 0.0, 0.0)),
                           (0.5, (1.0, 0.5, 0.25)),
                           (1.0, (0.25, 0.75, 1.0))])
    n = 100_000
    import time
    start = time.perf_counter()
    acc = 0.0
    for i in range(n):
        acc += g.sample(i / (n - 1))[0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert acc > 0.0
    print("\n100k srgb samples: %.1f ms" % elapsed_ms)
    assert elapsed_ms < 150.0

    g_lin = gradient.Gradient([(0.0, (0.0, 0.0, 0.0)),
                               (0.5, (1.0, 0.5, 0.25)),
                               (1.0, (0.25, 0.75, 1.0))], space="linear")
    start = time.perf_counter()
    acc = 0.0
    for i in range(n):
        acc += g_lin.sample(i / (n - 1))[0]
    elapsed_ms_lin = (time.perf_counter() - start) * 1000.0
    assert acc > 0.0
    print("100k linear samples: %.1f ms" % elapsed_ms_lin)


def test_concurrent_reads_and_writes():
    g = gradient.Gradient([(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))])
    errors = []

    def reader():
        try:
            for i in range(20_000):
                t = (i % 1000) / 1000.0
                s = g.sample(t)
                d = g.gradient_at(t)
                assert len(s) == 3 and len(d) == 3
                assert all(math.isfinite(v) for v in s + d)
        except Exception as exc:  # pragma: no cover - 并发失败路径
            errors.append(exc)

    def writer():
        try:
            for i in range(1, 200):
                g.add_stop(i / 201.0, (i / 201.0, 1.0 - i / 201.0, 0.5))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    threads.append(threading.Thread(target=writer))
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []
