"""电平表扩展验收测试（峰值 / RMS / true-peak / hold / 并发 / 性能）。

用 -k peak 可选中本文件全部用例（函数名均含 peak）。
既有 tests/test_core.py 未做任何改动。
"""

import math
import threading
import time

import pytest

import peakmeter
from peakmeter import (
    InvalidHoldError,
    NonFiniteSampleError,
    PeakMeter,
)


# ---------- 基础峰值语义（绝对值 / 空输入 / 重置 / 类型） ----------

def test_peak_absolute_value_and_empty():
    m = PeakMeter()
    assert m.peak() == 0.0
    m.push([-0.75, 0.5, -0.25])
    assert m.peak() == 0.75
    assert isinstance(m.peak(), float)


def test_peak_empty_push_is_zero():
    m = PeakMeter()
    m.push([])
    assert m.peak() == 0.0
    assert m.rms() == 0.0
    assert m.true_peak() == 0.0


def test_peak_reset_clears_all_state():
    m = PeakMeter()
    m.push([0.9, -0.8])
    m.reset()
    assert m.peak() == 0.0
    assert m.rms() == 0.0
    assert m.true_peak() == 0.0
    m.push([0.2])
    assert m.peak() == 0.2  # 无状态泄漏


def test_peak_accepts_iterables():
    m = PeakMeter()
    m.push(abs(v) for v in (-0.3, -0.6, 0.1))
    assert m.peak() == 0.6


# ---------- RMS ----------

def test_peak_rms_basic():
    m = PeakMeter()
    m.push([0.3, -0.4])
    assert m.rms() == pytest.approx(math.sqrt(0.25 / 2))  # sqrt((.09+.16)/2)
    m.push([0.0])
    assert m.rms() == pytest.approx(math.sqrt(0.25 / 3))


# ---------- true-peak：4 倍过采样近似 ----------

def test_peak_true_peak_ge_sample_peak():
    m = PeakMeter()
    samples = [math.sin(0.9 * i) * 0.8 for i in range(5000)]
    m.push(samples)
    sp = max(abs(v) for v in samples)
    assert m.true_peak() >= sp
    assert m.true_peak() >= m.peak()


def test_peak_true_peak_overshoots_on_sine():
    # Fs/4 正弦、45° 相位：采样峰值仅 sqrt(2)/2，真实峰值为 1.0。
    # 4 倍过采样应当捕捉到明显过冲。
    samples = [math.sin(math.pi * i / 2 + math.pi / 4) for i in range(256)]
    m = PeakMeter()
    m.push(samples)
    sample_peak = max(abs(v) for v in samples)
    assert sample_peak == pytest.approx(math.sqrt(2) / 2, abs=1e-9)
    assert m.true_peak() > sample_peak * 1.1


def test_peak_true_peak_within_three_db():
    # 多种信号下 true-peak 与采样峰值之差均不超过 3 dB
    signals = [
        [math.sin(0.37 * i) for i in range(3000)],
        [((-1) ** i) * (0.5 + 0.5 * math.sin(i * 0.01)) for i in range(3000)],
        [0.9 * math.sin(math.pi * i / 2 + math.pi / 4) for i in range(3000)],
        [1.0, -1.0] * 1500,
    ]
    for sig in signals:
        m = PeakMeter()
        m.push(sig)
        sp = max(abs(v) for v in sig)
        if sp == 0:
            assert m.true_peak() == 0.0
            continue
        db = 20 * math.log10(m.true_peak() / sp)
        assert -1e-9 <= db <= 3.0 + 1e-9


def test_peak_true_peak_cross_batch_continuous():
    # 拆批推送与整段推送的 true-peak 必须一致（上下文跨批衔接）
    sig = [math.sin(math.pi * i / 2 + math.pi / 4) for i in range(100)]
    a, b = PeakMeter(), PeakMeter()
    a.push(sig)
    b.push(sig[:30])
    b.push(sig[30:70])
    b.push(sig[70:])
    assert a.true_peak() == b.true_peak()


# ---------- hold 保持窗口 ----------

def test_peak_hold_window_keeps_max():
    m = PeakMeter(hold=3)
    m.push([0.2, 0.9])        # idx 0,1
    assert m.peak() == 0.9    # 窗口 [0.2,0.9]
    m.push([0.1, 0.3])        # idx 2,3 → 窗口 idx1..3
    assert m.peak() == 0.9    # 0.9 仍在窗口内
    m.push([0.2])             # idx 4 → 窗口 idx2..4
    assert m.peak() == 0.3    # 0.9（idx1）已过期
    m.push([0.05])            # idx 5 → 窗口 idx3..5
    assert m.peak() == 0.3    # 0.3（idx3）仍在
    m.push([0.01, 0.02])      # idx 6,7 → 窗口 idx5..7
    assert m.peak() == 0.05


def test_peak_hold_zero_reflects_latest_batch():
    m = PeakMeter(hold=0)
    m.push([0.9])
    assert m.peak() == 0.0
    m.push([0.4, -0.6])
    assert m.peak() == 0.0
    assert m.rms() == 0.0
    assert m.true_peak() == 0.0


def test_peak_hold_rms_matches_window():
    m = PeakMeter(hold=3)
    m.push([0.3, 0.4])                       # idx0,1
    m.push([0.0, 0.0])                       # idx2,3 → 窗口 idx1..3 = [.4,0,0]
    assert m.rms() == pytest.approx(math.sqrt(0.16 / 3))
    m.push([0.0])                            # idx4 → 窗口 idx2..4 全为 0
    assert m.rms() == 0.0                    # 0.4（idx1）已过期


def test_peak_hold_reset():
    m = PeakMeter(hold=2)
    m.push([0.8, 0.7])
    m.reset()
    assert m.peak() == 0.0
    assert m.rms() == 0.0
    m.push([0.3])
    assert m.peak() == 0.3


def test_peak_invalid_hold_is_distinct_error():
    for bad in (-1, 1.5, "3", True):
        with pytest.raises(InvalidHoldError):
            PeakMeter(hold=bad)
    # 与非有限样本错误可区分
    assert issubclass(InvalidHoldError, ValueError)
    assert issubclass(NonFiniteSampleError, ValueError)
    assert InvalidHoldError is not NonFiniteSampleError


# ---------- 非有限样本 ----------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_peak_non_finite_sample_raises_distinct_error(bad):
    m = PeakMeter()
    with pytest.raises(NonFiniteSampleError):
        m.push([0.5, bad, 0.3])
    # 非法批次原子失败，不污染既有状态
    assert m.peak() == 0.0
    m.push([0.42])
    with pytest.raises(NonFiniteSampleError):
        m.push([bad])
    assert m.peak() == 0.42


def test_peak_non_finite_in_hold_meter():
    m = PeakMeter(hold=4)
    m.push([0.7])
    with pytest.raises(NonFiniteSampleError):
        m.push([float("nan")])
    assert m.peak() == 0.7


# ---------- 并发安全 ----------

def test_peak_concurrent_push_thread_safe():
    n_threads = 8
    per_thread = 12_500
    m = PeakMeter()

    def worker(tid):
        # 每线程唯一区间，全体合起来覆盖 ±0.9992
        for i in range(per_thread):
            v = ((tid * per_thread + i) % 10000) / 10000.0 - 0.5
            m.push([v * 1.9984])

    threads = [threading.Thread(target=worker, args=(t,))
               for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = n_threads * per_thread
    # u 最大取 0.5（k=10000 时），故 v 最大为 0.5*1.9984 = 0.9992
    assert m.peak() == pytest.approx(0.9992, rel=1e-12)
    # 无样本丢失：RMS 与单线程参考一致
    ref = PeakMeter()
    for t in range(n_threads):
        for i in range(per_thread):
            v = ((t * per_thread + i) % 10000) / 10000.0 - 0.5
            ref.push([v * 1.9984])
    assert m.rms() == pytest.approx(ref.rms(), rel=1e-12)


def test_peak_concurrent_with_reset_safe():
    m = PeakMeter()
    errors = []

    def pusher():
        try:
            for i in range(2000):
                m.push([0.5 * ((-1) ** i)])
        except Exception as exc:  # pragma: no cover - 仅并发健壮性
            errors.append(exc)

    def resetter():
        try:
            for _ in range(200):
                m.reset()
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=pusher) for _ in range(4)]
    threads.append(threading.Thread(target=resetter))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert 0.0 <= m.peak() <= 1.0


# ---------- 性能：100 万样本 ≤ 300 ms ----------

def test_peak_one_million_samples_under_300ms():
    n = 1_000_000
    # 确定性 LCG，避免随机数生成时间混入测量
    state = 123456789
    samples = []
    for _ in range(n):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        samples.append(state / 0x7FFFFFFF * 2.0 - 1.0)

    m = PeakMeter()
    start = time.perf_counter()
    # 分批推送，覆盖跨批路径
    for i in range(0, n, 100_000):
        m.push(samples[i:i + 100_000])
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert m.peak() > 0.99
    assert m.rms() == pytest.approx(1.0 / math.sqrt(3), abs=0.01)
    assert m.true_peak() >= m.peak()
    assert elapsed_ms <= 300.0, f"耗时 {elapsed_ms:.1f} ms 超过 300 ms"
