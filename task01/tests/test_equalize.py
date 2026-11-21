"""histeq 扩展测试：覆盖题面的边界要求。

运行方式（与验收命令一致）::

    cd task01 && python -m pytest tests -q -k equalize
"""

import math
import random
import sys
import threading
import time
import tracemalloc

import pytest

import histeq
from histeq import (
    EmptyImageError,
    HistEq,
    InvalidBlockError,
    NonFinitePixelError,
    NonRectangularImageError,
    PixelRangeError,
)


def _noise(height, width, seed=7):
    rng = random.Random(seed)
    return [[rng.random() for _ in range(width)] for _ in range(height)]


def _gradient(height, width):
    return [
        [(x + y) / float(height + width - 2) for x in range(width)]
        for y in range(height)
    ]


def _min_step(output):
    """返回 (水平最小增量, 垂直最小增量)；负值代表映射反转。"""
    height = len(output)
    width = len(output[0])
    horizontal = min(
        (output[y][x + 1] - output[y][x] for y in range(height) for x in range(width - 1)),
        default=0.0,
    )
    vertical = min(
        (output[y + 1][x] - output[y][x] for y in range(height - 1) for x in range(width)),
        default=0.0,
    )
    return horizontal, vertical


def _max_step(output):
    height = len(output)
    width = len(output[0])
    horizontal = max(
        (abs(output[y][x + 1] - output[y][x]) for y in range(height) for x in range(width - 1)),
        default=0.0,
    )
    vertical = max(
        (abs(output[y + 1][x] - output[y][x]) for y in range(height - 1) for x in range(width)),
        default=0.0,
    )
    return horizontal, vertical


# ---------------------------------------------------------------------------
# 主功能
# ---------------------------------------------------------------------------


def test_equalize_output_contract():
    output = HistEq().equalize([[0.0, 0.5], [0.5, 1.0]])
    assert isinstance(output, list)
    assert [len(row) for row in output] == [2, 2]
    assert all(isinstance(value, float) for row in output for value in row)
    assert all(0.0 <= value <= 1.0 for row in output for value in row)


def test_equalize_constant_image_is_identity():
    image = [[0.3] * 9 for _ in range(6)]
    assert HistEq().equalize(image) == image
    sparse = [[0.0, 0.0], [0.0, 0.0]]
    assert HistEq().equalize(sparse) == sparse


def test_equalize_empty_raises():
    with pytest.raises(EmptyImageError):
        HistEq().equalize([])
    with pytest.raises(EmptyImageError):
        HistEq().equalize([[]])


def test_equalize_ragged_raises():
    with pytest.raises(NonRectangularImageError):
        HistEq().equalize([[0.1, 0.2], [0.3]])


def test_equalize_non_finite_raises():
    with pytest.raises(NonFinitePixelError):
        HistEq().equalize([[float("nan")]])
    with pytest.raises(NonFinitePixelError):
        HistEq().equalize([[0.5, float("inf")]])


def test_equalize_out_of_range_raises():
    with pytest.raises(PixelRangeError):
        HistEq().equalize([[1.5]])
    with pytest.raises(PixelRangeError):
        HistEq().equalize([[-0.1]])


def test_equalize_does_not_mutate_input():
    image = _noise(16, 16)
    snapshot = [row[:] for row in image]
    HistEq().equalize(image)
    assert image == snapshot


def test_equalize_is_deterministic():
    image = _noise(24, 18)
    first = HistEq().equalize(image)
    second = HistEq().equalize(image)
    assert first == second


# ---------------------------------------------------------------------------
# 分块与单调性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", [
    {},
    {"tiles": (4, 4)},
    {"tiles": (2, 3)},
    {"block_size": 8},
    {"block_size": (7, 5)},
    {"blocks": (6, 4)},
    {"grid": 3},
])
def test_equalize_monotone_on_gradient(config):
    for shape in ((97, 53), (128, 160), (301, 299)):
        output = HistEq(**config).equalize(_gradient(*shape))
        horizontal, vertical = _min_step(output)
        assert horizontal >= -1e-9, (shape, config, horizontal)
        assert vertical >= -1e-9, (shape, config, vertical)


@pytest.mark.parametrize("size", [16, 64, 256, 512, 1000])
def test_equalize_monotone_on_ramp(size):
    ramp = [[index / (size - 1) for index in range(size)]]
    output = HistEq().equalize(ramp)
    assert _min_step(output)[0] >= -1e-9


def test_equalize_block_size_is_configurable():
    image = _noise(64, 64, seed=3)
    fine = HistEq(block_size=8).equalize(image)
    coarse = HistEq(block_size=32).equalize(image)
    assert fine != coarse
    # 两种配置都必须保持结构性约束。
    assert all(0.0 <= value <= 1.0 for row in fine for value in row)
    assert all(0.0 <= value <= 1.0 for row in coarse for value in row)
    assert _max_step(fine)[0] <= 1.0 and _max_step(coarse)[0] <= 1.0


def test_equalize_no_seam_at_tile_boundary():
    # 块边界恰好落在 128 / 512 列；接缝会表现为该列相邻差分显著偏大。
    width, height = 512, 16
    image = [[0.1 if x < 128 else 0.9 for x in range(width)] for _ in range(height)]
    output = HistEq().equalize(image)
    horizontal = [
        abs(output[y][x + 1] - output[y][x])
        for y in range(height)
        for x in range(width - 1)
    ]
    assert max(horizontal) < 0.05
    left = [abs(output[y][128] - output[y][127]) for y in range(height)]
    interior = [abs(output[y][120] - output[y][119]) for y in range(height)]
    assert max(left) <= max(interior) * 4 + 0.01


def test_equalize_per_call_override_does_not_leak():
    image = _noise(20, 20, seed=11)
    engine = HistEq(tiles=(2, 2))
    baseline = engine.equalize(image)
    engine.equalize(image, tiles=1)
    assert engine.equalize(image) == baseline


# ---------------------------------------------------------------------------
# 非法块配置
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", [
    {"block_size": 0},
    {"block_size": -8},
    {"tiles": 0},
    {"tiles": (0, 4)},
    {"tiles": (4, 4, 4)},
    {"block_size": 1.5},
    {"block_size": True},
    {"tiles": float("nan")},
    {"tiles": (2, float("inf"))},
])
def test_equalize_invalid_block_config_is_distinguishable(config):
    with pytest.raises(InvalidBlockError):
        HistEq(**config)


def test_equalize_invalid_block_errors_are_not_confused_with_pixel_errors():
    assert not issubclass(InvalidBlockError, (NonFinitePixelError, NonRectangularImageError))
    with pytest.raises(InvalidBlockError):
        HistEq(block_size=0)
    with pytest.raises(NonFinitePixelError):
        HistEq().equalize([[float("nan")]])
    with pytest.raises(NonRectangularImageError):
        HistEq().equalize([[0.5, 0.4], [0.3]])


# ---------------------------------------------------------------------------
# 性能 / 内存 / 并发
# ---------------------------------------------------------------------------


def test_equalize_million_pixels_under_600ms():
    image = _noise(1000, 1000, seed=17)
    start = time.perf_counter()
    output = HistEq().equalize(image)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.6, "1MP took %.1f ms" % (elapsed * 1000)
    assert len(output) == 1000 and len(output[0]) == 1000


def test_equalize_memory_within_twice_input():
    image = _noise(500, 500, seed=19)
    input_bytes = (
        sys.getsizeof(image)
        + sum(sys.getsizeof(row) for row in image)
        + sum(sys.getsizeof(value) for row in image for value in row)
    )
    tracemalloc.start()
    try:
        HistEq().equalize(image)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < input_bytes * 2, "peak %.1f MB vs input %.1f MB" % (
        peak / 1e6,
        input_bytes / 1e6,
    )


def test_equalize_concurrent_images_are_safe():
    engine = HistEq()
    inputs = [_noise(64, 64, seed=index) for index in range(8)]
    expected = [engine.equalize(image) for image in inputs]
    failures = []

    def worker(index):
        try:
            for _ in range(3):
                if engine.equalize(inputs[index]) != expected[index]:
                    failures.append(index)
                    return
        except Exception as error:  # pragma: no cover - 失败时收集信息
            failures.append((index, error))

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures


def test_equalize_accepts_numpy_or_lists_and_agrees():
    numpy = pytest.importorskip("numpy")
    image = _noise(33, 47, seed=23)
    as_array = numpy.asarray(image)
    assert HistEq().equalize(image) == HistEq().equalize(as_array)


def test_equalize_constant_single_pixel():
    assert HistEq().equalize([[0.0]]) == [[0.0]]
    assert HistEq().equalize([[1.0]]) == [[1.0]]
    assert HistEq().equalize([[0.5]]) == [[0.5]]


def test_equalize_preserves_region_levels():
    """锚定全局 CDF 后，明暗区域的电平应保持分离，而不是被压平。"""
    size = 128
    rng = random.Random(31)
    image = [[0.0] * size for _ in range(size)]
    for y in range(size):
        for x in range(size):
            if y < size // 2 and x < size // 2:
                level = 0.8
            elif y >= size // 2 and x >= size // 2:
                level = 0.2
            else:
                level = 0.5
            image[y][x] = level + 0.05 * (rng.random() * 2 - 1)

    output = HistEq(tiles=(4, 4)).equalize(image)
    bright = sum(v for row in output[:64] for v in row[:64]) / 64 / 64
    dark = sum(v for row in output[64:] for v in row[64:]) / 64 / 64
    assert bright - dark > 0.5, (bright, dark)


def test_equalize_enhances_low_contrast():
    width = height = 240

    def value(x, y):
        return 0.5 + 0.02 * math.sin(x * 0.7) * math.cos(y * 0.5)

    image = [[value(x, y) for x in range(width)] for y in range(height)]
    output = HistEq().equalize(image)

    def spread(matrix):
        values = [v for row in matrix for v in row]
        mean = sum(values) / len(values)
        return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

    assert spread(output) > spread(image) * 5


def test_equalize_large_tile_count_stays_within_budget():
    image = _noise(300, 300, seed=37)
    output = HistEq(block_size=8).equalize(image)
    assert all(0.0 <= value <= 1.0 for row in output for value in row)
    assert [len(row) for row in output] == [300] * 300
