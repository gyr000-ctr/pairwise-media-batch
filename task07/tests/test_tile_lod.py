"""task07 补充测试：LOD 金字塔、可见性裁剪、无限分页、错误区分、并发与性能。

所有用例名均含 ``tile``，可被 ``pytest -k tile`` 选中；不改动既有测试。
"""

import math
import threading

import pytest

import tilemap
from tilemap import (
    InvalidPageSizeError,
    InvalidTileSizeError,
    NonFiniteCoordinateError,
    TileMap,
)


# --------------------------------------------------------------------------- #
# 基础取整与往返
# --------------------------------------------------------------------------- #
def test_tile_floor_negative_positive():
    t = TileMap(16.0)
    assert t.to_tile(0.0, 0.0) == (0, 0)
    assert t.to_tile(17.0, 33.0) == (1, 2)
    assert t.to_tile(-1.0, -17.0) == (-1, -2)
    # 负侧紧贴边界：-16 恰好属于瓦片 -1
    assert t.to_tile(-16.0, -0.0001) == (-1, -1)
    assert t.to_tile(-16.0001, -32.0) == (-2, -2)


def test_tile_boundary_ownership():
    t = TileMap(16.0)
    # 边界点归属编号更大的瓦片（半开区间 [k*s, (k+1)*s)）
    assert t.to_tile(16.0, 0.0) == (1, 0)
    assert t.to_tile(31.999999, 0.0) == (1, 0)
    assert t.to_tile(32.0, 0.0) == (2, 0)
    assert t.to_tile(-16.0, 0.0) == (-1, 0)


def test_tile_world_roundtrip_error():
    t = TileMap(16.0)
    for x, y in [(35.5, -12.25), (-1e5 + 7.5, 1e5 - 3.25),
                 (0.3, -0.7), (1234567.89, -9876543.21)]:
        tx, ty = t.to_tile(x, y)
        wx, wy = t.to_world(tx, ty)
        assert abs(wx - math.floor(x / 16.0) * 16.0) <= 1e-9
        assert abs(wy - math.floor(y / 16.0) * 16.0) <= 1e-9
        # 往返后的瓦片原点必然落在所属瓦片内
        assert t.to_tile(wx, wy) == (tx, ty)


def test_tile_to_world_returns_float_and_deterministic():
    t = TileMap(16.0)
    wx, wy = t.to_world(1, 2)
    assert (wx, wy) == (16.0, 32.0)
    assert isinstance(wx, float) and isinstance(wy, float)
    assert t.to_tile(5.0, 7.0) == t.to_tile(5.0, 7.0)
    # 无实例状态：连续查询互不影响
    t.to_tile(1.0, 1.0)
    assert t.to_tile(2.0, 2.0) == (0, 0)


def test_tile_huge_range():
    t = TileMap(4.0)
    assert t.to_tile(1e12, -1e12) == (250_000_000_000, -250_000_000_000)
    big = 1e300
    tx, ty = t.to_tile(big, -big)
    assert tx == 2.5e299 and ty == -2.5e299
    # 大坐标附近的相机矩形也能稳定枚举，且首元素确定
    tiles = t.visible_tiles(1e9 - 8.0, -1e9, 1e9 + 8.0, -1e9 + 8.0)
    assert tiles[0] == (0, 249_999_998, -250_000_000)
    assert tiles[-1] == (0, 250_000_002, -249_999_998)
    assert len(tiles) == 5 * 3


# --------------------------------------------------------------------------- #
# 四叉树 LOD
# --------------------------------------------------------------------------- #
def test_tile_lod_size_doubles_each_level():
    t = TileMap(16.0)
    assert t.tile_size(0) == 16.0
    assert t.tile_size(1) == 32.0
    assert t.tile_size(3) == 128.0
    assert t.to_tile(33.0, 0.0, level=1) == (1, 0)
    assert t.to_tile(-1.0, -33.0, level=1) == (-1, -2)
    assert t.to_world(2, -3, level=2) == (128.0, -192.0)


def test_tile_visible_matches_brute_force_and_order():
    t = TileMap(8.0)
    rects = [
        (0.0, 0.0, 16.0, 16.0),
        (-20.0, -20.0, 20.0, 20.0),
        (7.999, 7.999, 8.001, 8.001),       # 跨 level-0 瓦片缝
        (-8.0, -8.0, 0.0, 0.0),             # 负区域边界
        (3.0, 3.0, 3.0, 3.0),               # 零面积点
        (100.0, -50.0, 132.0, -18.0),
    ]
    for level in range(5):
        for rect in rects:
            fast = t.visible_tiles(rect, level=level)
            brute = t.brute_force_tiles(*rect, level=level)
            assert fast == brute
            # 顺序确定：level 固定时按 (ty, tx) 行优先升序
            assert fast == sorted(fast, key=lambda item: (item[2], item[1]))
            # 集合内无重复
            assert len(set(fast)) == len(fast)


def test_tile_visible_rect_crossing_lod_boundaries_no_gap_no_dup():
    t = TileMap(16.0)
    # 相机矩形同时跨越多个 LOD 级别的瓦片分界线
    rect = (-48.0, -48.0, 48.0, 48.0)
    seen = set()
    merged = t.visible_tiles_levels(rect, range(4))
    for item in merged:
        assert item not in seen
        seen.add(item)
    # 每个级别都应覆盖同一相机矩形（无漏瓦片：与暴力实现逐级别一致）
    for level in range(4):
        assert t.visible_tiles(rect, level=level) == \
            t.brute_force_tiles(*rect, level=level)
    # 单瓦片宽的退化矩形在任意级别恰好命中一个瓦片
    for level in range(6):
        s = 16.0 * (1 << level)
        got = t.visible_tiles((s * 3.0, -s * 2.0, s * 3.0, -s * 2.0),
                              level=level)
        assert got == [(level, 3, -2)]


def test_tile_visible_alias_and_tuple_form():
    t = TileMap(10.0)
    a = t.visible_tiles(0, 0, 25, 15, level=0)
    b = t.query_tiles((0, 0, 25, 15))
    c = t.cull((0, 0, 25, 15))
    assert a == b == c
    # 反序矩形等价于归一化后的矩形
    assert t.visible_tiles(25, 15, 0, 0) == a
    # 全零矩形（0.0 为假值，不得误判为矩形序列入参）
    assert t.visible_tiles(0.0, 0.0, 0.0, 0.0) == [(0, 0, 0)]
    assert t.visible_tiles((0.0, 0.0, 0.0, 0.0), 2) == [(2, 0, 0)]


# --------------------------------------------------------------------------- #
# 无限分页
# --------------------------------------------------------------------------- #
def test_tile_paging_modulo_negative_wraps_to_last():
    t = TileMap(16.0, page_w=4, page_h=8)
    assert t.wrap_tile(0, 0) == (0, 0)
    assert t.wrap_tile(3, 7) == (3, 7)
    assert t.wrap_tile(4, 8) == (0, 0)
    # -1 映射到最后一页/最后一个位置
    assert t.wrap_tile(-1, -1) == (3, 7)
    assert t.wrap_tile(-4, -8) == (0, 0)
    assert t.wrap_tile(-5, -9) == (3, 7)


def test_tile_paging_page_of_consistent_with_wrap():
    t = TileMap(16.0, 4, 4)
    for tx, ty in [(-7, -1), (-4, 0), (-1, 3), (0, 0), (5, -9), (40, -40)]:
        px, py = t.page_of(tx, ty)
        wx, wy = t.wrap_tile(tx, ty)
        assert tx == px * 4 + wx and ty == py * 4 + wy
        assert 0 <= wx < 4 and 0 <= wy < 4
    assert t.page_of(-1, -1) == (-1, -1)
    assert t.page_of(-4, -4) == (-1, -1)


def test_tile_paging_square_page_constructor():
    t = TileMap(8.0, 16)
    assert (t.page_width, t.page_height) == (16, 16)
    assert t.wrap_tile(-1, 16) == (15, 0)
    px, py, wx, wy = t.to_page_tile(-8.0, 8.0)
    assert (px, py, wx, wy) == (-1, 0, 15, 1)


def test_tile_paging_disabled_raises_distinct_error():
    t = TileMap(8.0)
    with pytest.raises(InvalidPageSizeError):
        t.wrap_tile(0, 0)
    with pytest.raises(InvalidPageSizeError):
        t.page_of(0, 0)


# --------------------------------------------------------------------------- #
# 可区分的错误
# --------------------------------------------------------------------------- #
def test_tile_invalid_size_error_distinguishable():
    for bad in (0.0, -1.0, -16):
        with pytest.raises(InvalidTileSizeError):
            TileMap(bad)
    # 非整数瓦片尺寸单独可识别
    with pytest.raises(InvalidTileSizeError):
        TileMap(16.5)
    # 仍是 ValueError，兼容旧用法
    with pytest.raises(ValueError):
        TileMap(0.0)


def test_tile_invalid_page_size_error_distinguishable():
    with pytest.raises(InvalidPageSizeError):
        TileMap(16.0, page_w=0, page_h=4)
    with pytest.raises(InvalidPageSizeError):
        TileMap(16.0, page_w=4, page_h=-1)
    with pytest.raises(InvalidPageSizeError):
        TileMap(16.0, page_w=2.5, page_h=4)


def test_tile_non_finite_coordinate_error_distinguishable():
    t = TileMap(16.0)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(NonFiniteCoordinateError):
            t.to_tile(bad, 0.0)
        with pytest.raises(NonFiniteCoordinateError):
            t.to_tile(0.0, bad)
        with pytest.raises(NonFiniteCoordinateError):
            t.visible_tiles(0.0, 0.0, bad, 1.0)
    # 非有限的“尺寸”归类为非法瓦片尺寸（与非有限坐标错误可区分）
    with pytest.raises(InvalidTileSizeError):
        TileMap(float("nan"))
    # 三类错误彼此可区分
    assert InvalidTileSizeError is not InvalidPageSizeError
    assert InvalidPageSizeError is not NonFiniteCoordinateError
    with pytest.raises(ValueError):
        t.to_tile(float("inf"), 0.0)


def test_tile_bad_level():
    t = TileMap(16.0)
    with pytest.raises(ValueError):
        t.visible_tiles(0, 0, 1, 1, level=-1)
    with pytest.raises(ValueError):
        t.tile_size(1.5)


# --------------------------------------------------------------------------- #
# 并发与性能
# --------------------------------------------------------------------------- #
def test_tile_concurrent_queries_are_safe():
    t = TileMap(16.0, page_w=8, page_h=8)
    errors = []
    expected = t.visible_tiles(-1000, -1000, 1000, 1000, level=2)

    def worker(seed):
        try:
            for i in range(2000):
                v = (i * 7 + seed) % 500 - 250
                assert t.to_tile(float(v), float(-v)) == \
                    (math.floor(v / 16.0), math.floor(-v / 16.0))
                assert t.visible_tiles(v, v, v + 33, v + 33, level=1) == \
                    t.brute_force_tiles(v, v, v + 33, v + 33, level=1)
                assert t.wrap_tile(v - 1000, v + 1000) == \
                    ((v - 1000) % 8, (v + 1000) % 8)
            assert t.visible_tiles(-1000, -1000, 1000, 1000, level=2) == \
                expected
        except BaseException as exc:  # noqa: BLE001 - 测试线程需要回传全部异常
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []


def test_tile_100k_conversions_under_200ms():
    import time

    t = TileMap(16.0)
    start = time.perf_counter()
    for i in range(100_000):
        t.to_tile(i * 0.5 - 25000.0, 25000.0 - i * 0.25)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 200.0, f"10 万次转换耗时 {elapsed_ms:.1f} ms"
