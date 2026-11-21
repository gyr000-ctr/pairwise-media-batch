"""histeq 核心 API。

实现分块 CLAHE（Contrast Limited Adaptive Histogram Equalization）：

* 输入为矩形二维序列，像素值必须落在 ``[0, 1]`` 且为有限实数；
* 将图像划分为 ``ny x nx`` 个块，对每块直方图做裁剪/再分配后得到映射表；
* 输出像素由相邻四个块的映射表做双线性插值，块边界没有硬接缝；
* 输出形状与输入一致、取值 ``[0, 1]``，映射单调不反转；
* 常量图原样返回（与输入逐值相等）；
* 实例无可变状态，``equalize`` 可被多个线程并发调用。

块的划分有四种配置方式（构造参数或 ``equalize`` 关键字参数均可）：

* 按像素尺寸：``block_size=16`` / ``tile_size=(16, 8)`` / ``block=32``；
* 按块数量：``tiles=8`` / ``blocks=(8, 8)`` / ``grid=4``。

默认采用 ``8 x 8`` 块。

单调性的实现方式：每个块的映射表都以图像的全局 CDF 为锚点 —— 块的局部
CDF 只用于在 ``[G(lo), G(hi)]`` 区间内重排电平（``lo`` / ``hi`` 是该块实际
出现的最小 / 最大 bin），块外的 bin 直接沿用 ``G`` 本身。之后对块的映射表
做二维前缀最大（monotone envelope）排序，使相邻块的映射在所有 bin 上都不
交叉；再配合块中心之间的双线性插值，输出在任意方向上都不会出现局部反转，
块与块之间也没有电平跳变。以全局 CDF 锚定（而非把每块单独拉满 [0, 1]）使
明暗区域的电平保持分离，不会因块间排序被压平。
"""

from __future__ import annotations

import math
import numbers

try:  # numpy 只是可选的加速路径；没有它时仍有纯 Python 实现
    import numpy as _np
except Exception:  # pragma: no cover - 仅在缺少 numpy 的环境中触发
    _np = None

__all__ = [
    "HistEq",
    "HistEqError",
    "InvalidBlockError",
    "NonFinitePixelError",
    "NonRectangularImageError",
    "EmptyImageError",
    "PixelRangeError",
    "PixelTypeError",
]


# ---------------------------------------------------------------------------
# 可区分的错误类型（全部继承 ValueError，便于既有调用方按 ValueError 捕获）
# ---------------------------------------------------------------------------


class HistEqError(ValueError):
    """histeq 所有输入错误的基类。"""


class InvalidBlockError(HistEqError):
    """块大小 / 块数量配置非法，或两种配置互相冲突。"""


class NonFinitePixelError(HistEqError):
    """图像中存在 NaN 或无穷大像素。"""


class NonRectangularImageError(HistEqError):
    """输入不是矩形二维序列（行长度不一致 / 不是二维）。"""


class EmptyImageError(HistEqError):
    """输入为空图（没有像素）。"""


class PixelRangeError(HistEqError):
    """像素值超出 [0, 1]。"""


class PixelTypeError(HistEqError):
    """像素值不是实数。"""


_DEFAULT_GRID = (8, 8)
_DEFAULT_CLIP_LIMIT = 2.0
_DEFAULT_NBINS = 256

# 块数上限：块映射表按「块行」流式生成，任一时刻只保留两行 (2 * nx * nbins)，
# 内存与总块数无关；这里限制的是每块的固定计算开销——过多的小块会让
# 100 万像素的耗时超出预算。65536 块对应 100 万像素上约 15x15 像素的块。
_MAX_TILES = 65536


def _check_tile_budget(nx, ny):
    if nx * ny > _MAX_TILES:
        raise InvalidBlockError(
            "invalid block configuration: %d x %d = %d tiles exceeds the "
            "limit of %d tiles" % (nx, ny, nx * ny, _MAX_TILES)
        )


# ---------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------


def _parse_positive_pair(value, name):
    """把 ``4`` / ``(4, 8)`` 解析为正整数二元组。"""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise InvalidBlockError(
                "invalid block configuration: %s must be an int or a pair, got %r"
                % (name, value)
            )
        first, second = value
    else:
        first = second = value

    pair = []
    for item in (first, second):
        if isinstance(item, bool) or not isinstance(item, numbers.Real):
            raise InvalidBlockError(
                "invalid block configuration: %s must be positive integers, got %r"
                % (name, value)
            )
        if isinstance(item, numbers.Integral):
            number = int(item)
        else:
            as_float = float(item)
            if not math.isfinite(as_float) or as_float != math.floor(as_float):
                raise InvalidBlockError(
                    "invalid block configuration: %s must be integers, got %r"
                    % (name, value)
                )
            number = int(as_float)
        if number <= 0:
            raise InvalidBlockError(
                "invalid block configuration: %s must be positive, got %r"
                % (name, value)
            )
        pair.append(number)
    return tuple(pair)


def _parse_clip_limit(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise HistEqError("clip_limit must be a positive finite number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise HistEqError("clip_limit must be a positive finite number")
    return value


def _parse_nbins(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise HistEqError("nbins must be an integer >= 2")
    value = int(value)
    if value < 2 or value > 65536:
        raise HistEqError("nbins must be an integer in [2, 65536]")
    return value


def _tile_counts(height, width, block_px, grid):
    """返回 (nx, ny)，并保证每块至少覆盖一个像素。"""
    if block_px is not None:
        block_w, block_h = block_px
        nx = (width + block_w - 1) // block_w
        ny = (height + block_h - 1) // block_h
    else:
        nx, ny = grid
    nx = max(1, min(nx, width))
    ny = max(1, min(ny, height))
    _check_tile_budget(nx, ny)
    return nx, ny


def _bounds(length, count):
    """把长度 ``length`` 均匀切成 ``count`` 段，返回每段的 [start, end)。"""
    return [(i * length) // count for i in range(count)], [
        ((i + 1) * length) // count for i in range(count)
    ]


def _tile_of(length, count):
    """每个坐标所属的块下标（与 ``_bounds`` 划分严格一致）。

    ``_bounds`` 用 ``i * length // count`` 切分，所以第 ``x`` 个坐标属于
    满足 ``ends[i] > x`` 的第一个块；直接按边界二分，避免两套公式在
    非整除时错位。
    """
    _, ends = _bounds(length, count)
    result = []
    tile = 0
    for x in range(length):
        while tile + 1 < count and ends[tile] <= x:
            tile += 1
        result.append(tile)
    return result


def _interp_axis(length, count, starts, ends):
    """返回每个坐标参与插值的两个块下标与低端权重。

    块中心为 ``(start + end - 1) / 2``；中心以外的像素夹到最近的中心，
    因此整幅图（包括第一块与最后一块的外侧）都被覆盖。
    """
    zeros = [0] * length
    weights = [0.0] * length
    if count <= 1:
        return zeros, zeros, weights

    centers = [(starts[i] + ends[i] - 1) / 2.0 for i in range(count)]
    low = [0] * length
    high = [0] * length
    position = 0
    for x in range(length):
        while position + 1 < count - 1 and centers[position + 1] <= x:
            position += 1
        while position > 0 and centers[position] > x:
            position -= 1
        if x <= centers[0]:
            low[x] = 0
            high[x] = 1
            weights[x] = 0.0
        elif x >= centers[count - 1]:
            low[x] = count - 2
            high[x] = count - 1
            weights[x] = 1.0
        else:
            span = centers[position + 1] - centers[position]
            weight = (x - centers[position]) / span if span > 0.0 else 0.0
            low[x] = position
            high[x] = position + 1
            weights[x] = min(1.0, max(0.0, weight))
    return low, high, weights


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def _non_finite_message(arr):
    index = _np.argwhere(~_np.isfinite(arr))[0].tolist()
    return (
        "image contains a non-finite pixel (NaN or infinity) "
        "at row %d, column %d" % (index[0], index[1])
    )


def _range_message(arr):
    index = _np.argwhere((arr < 0.0) | (arr > 1.0))[0].tolist()
    return "pixel value %r at row %d, column %d is outside [0, 1]" % (
        float(arr[index[0], index[1]]),
        index[0],
        index[1],
    )


def _validate_numpy(image):
    """把输入转换为 float64 的二维数组，并完成全部输入校验。"""
    try:
        raw = _np.asarray(image)
    except (ValueError, TypeError) as exc:
        raise NonRectangularImageError(
            "image must be a rectangular 2-D sequence"
        ) from exc

    if raw.ndim != 2:
        if raw.size == 0 and raw.ndim <= 2:
            raise EmptyImageError("image must contain at least one pixel")
        raise NonRectangularImageError(
            "image must be a rectangular 2-D sequence (got a %d-D array)" % raw.ndim
        )
    height, width = raw.shape
    if height == 0 or width == 0:
        raise EmptyImageError("image must contain at least one pixel")

    if raw.dtype.kind in "cUSO":
        if raw.dtype.kind != "O":
            raise PixelTypeError(
                "pixel values must be real numbers (got dtype %s)" % raw.dtype
            )
        try:
            arr = raw.astype(_np.float64)
        except (TypeError, ValueError) as exc:
            raise PixelTypeError("pixel values must be real numbers") from exc
        if not _np.isfinite(arr).all():
            raise NonFinitePixelError(_non_finite_message(arr))
        return arr

    if raw.dtype.kind not in "biuf":
        raise PixelTypeError(
            "pixel values must be real numbers (got dtype %s)" % raw.dtype
        )
    if raw.dtype.kind == "f" and not _np.isfinite(raw).all():
        raise NonFinitePixelError(_non_finite_message(raw))
    arr = raw.astype(_np.float64, copy=False)

    minimum = float(arr.min())
    maximum = float(arr.max())
    if minimum < 0.0 or maximum > 1.0:
        raise PixelRangeError(_range_message(arr))
    return arr


def _validate_python(image):
    """纯 Python 输入校验（numpy 不可用时的后备路径）。"""
    try:
        rows = list(image)
    except TypeError as exc:
        raise NonRectangularImageError(
            "image must be a rectangular 2-D sequence"
        ) from exc

    if not rows:
        raise EmptyImageError("image must contain at least one pixel")

    width = None
    for row in rows:
        if isinstance(row, (str, bytes)):
            raise NonRectangularImageError(
                "image must be a rectangular 2-D sequence (found a scalar row)"
            )
        try:
            row_length = len(row)
        except TypeError as exc:
            raise NonRectangularImageError(
                "image must be a rectangular 2-D sequence (found a scalar row)"
            ) from exc
        if width is None:
            width = row_length
        elif row_length != width:
            raise NonRectangularImageError(
                "image must be rectangular: row lengths differ (%d != %d)"
                % (row_length, width)
            )
    if width == 0:
        raise EmptyImageError("image must contain at least one pixel")

    result = []
    for row_index, row in enumerate(rows):
        converted = []
        for column_index, value in enumerate(row):
            try:
                pixel = float(value)
            except (TypeError, ValueError) as exc:
                raise PixelTypeError(
                    "pixel value at row %d, column %d is not a real number"
                    % (row_index, column_index)
                ) from exc
            if not math.isfinite(pixel):
                raise NonFinitePixelError(
                    "image contains a non-finite pixel (NaN or infinity) "
                    "at row %d, column %d" % (row_index, column_index)
                )
            if pixel < 0.0 or pixel > 1.0:
                raise PixelRangeError(
                    "pixel value %r at row %d, column %d is outside [0, 1]"
                    % (pixel, row_index, column_index)
                )
            converted.append(pixel)
        result.append(converted)
    return result


# ---------------------------------------------------------------------------
# 全局锚定 CDF
# ---------------------------------------------------------------------------


def _global_cdf(hist):
    """由全局直方图得到锚定映射 G（长度 nbins，取值 [0,1]）。

    若整幅图只出现一个 bin（常量图），返回 ``None``。
    """
    nbins = len(hist)
    occupied = [i for i, count in enumerate(hist) if count > 0]
    if len(occupied) <= 1:
        return None

    total = float(sum(hist))
    minimum = float(hist[occupied[0]])
    denominator = total - minimum
    if denominator <= 0.0:  # pragma: no cover - 只有单 bin 时才会发生
        return None

    cumulative = 0.0
    cdf = [0.0] * nbins
    for index in range(nbins):
        cumulative += hist[index]
        value = (cumulative - minimum) / denominator
        cdf[index] = 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)
    return cdf


# ---------------------------------------------------------------------------
# 单块映射表
# ---------------------------------------------------------------------------


def _tile_lut_python(counts, area, clip_limit, nbins, anchor):
    """单块映射表（纯 Python）。

    ``counts`` 为块内直方图，``anchor`` 为全局 CDF ``G``。块的局部 CDF 把
    ``[G(lo), G(hi)]`` 重新分配；块外（``< lo`` / ``> hi`` 的 bin）直接沿用
    ``G`` 本身。由于 ``G`` 单调不减、块内映射在端点处与 ``G`` 相接，整表
    单调不减，且不会把块外的电平钳成平台（那是结构压缩的来源）。
    块内只有一个 bin 时整块映射为常数 ``G(lo)``。
    """
    occupied = [i for i, count in enumerate(counts) if count > 0]
    if len(occupied) <= 1:
        if not occupied:
            return list(anchor)
        value = anchor[occupied[0]]
        return [value] * nbins

    lo = occupied[0]
    hi = occupied[-1]

    limit = max(1.0, math.floor(clip_limit * area / nbins))
    clipped = [min(float(count), limit) for count in counts]
    excess = float(area) - sum(clipped)
    share = excess / nbins
    clipped = [count + share for count in clipped]

    cumulative = []
    running = 0.0
    for count in clipped:
        running += count
        cumulative.append(running)
    low_cdf = cumulative[lo]
    span = cumulative[hi] - low_cdf
    bottom = anchor[lo]
    spread = anchor[hi] - bottom

    lut = anchor[:lo]
    if span <= 0.0:  # pragma: no cover - 只有退化直方图才会发生
        middle = [bottom] * (hi - lo + 1)
    else:
        middle = [
            bottom + (cumulative[i] - low_cdf) / span * spread
            for i in range(lo, hi + 1)
        ]
    lut.extend(middle)
    lut.extend(anchor[hi + 1:])
    return lut


def _tile_lut_numpy(counts, area, clip_limit, nbins, anchor):
    """单块映射表（numpy 版本，语义与 ``_tile_lut_python`` 一致）。"""
    used = counts > 0
    occupied = int(_np.count_nonzero(used))
    if occupied <= 1:
        index = int(_np.flatnonzero(used)[0]) if occupied else nbins // 2
        return _np.full(nbins, anchor[index], dtype=_np.float64)

    lo = int(_np.argmax(used))
    hi = nbins - 1 - int(_np.argmax(used[::-1]))

    limit = max(1.0, math.floor(clip_limit * area / nbins))
    clipped = _np.minimum(counts, limit).astype(_np.float64, copy=True)
    excess = float(area) - float(clipped.sum())
    clipped += excess / nbins

    cumulative = _np.cumsum(clipped)
    low_cdf = float(cumulative[lo])
    span = float(cumulative[hi]) - low_cdf
    bottom = float(anchor[lo])
    spread = float(anchor[hi]) - bottom

    lut = _np.empty(nbins, dtype=_np.float64)
    lut[:lo] = anchor[:lo]
    lut[hi + 1:] = anchor[hi + 1:]
    if span <= 0.0:  # pragma: no cover - 只有退化直方图才会发生
        lut[lo:hi + 1] = bottom
    else:
        lut[lo:hi + 1] = bottom + (cumulative[lo:hi + 1] - low_cdf) / span * spread
    return lut


def _tile_luts_numpy(counts, areas, clip_limit, nbins, anchor):
    """整块行批量构造映射表 (nx, nbins)，语义与 ``_tile_lut_numpy`` 一致。

    当所有块宽度相同时走这条向量化路径，避免逐块 Python 循环；块宽不一致
    时调用方仍使用 ``_tile_lut_numpy``。
    """
    rows = counts.shape[0]
    used = counts > 0
    occupied = used.sum(axis=1)

    limit = _np.maximum(1.0, _np.floor(clip_limit * areas / nbins))
    clipped = _np.minimum(counts, limit[:, None]).astype(_np.float64)
    excess = areas - clipped.sum(axis=1)
    clipped += (excess / nbins)[:, None]

    cumulative = _np.cumsum(clipped, axis=1)
    indices = _np.arange(rows, dtype=_np.intp)
    lo = _np.argmax(used, axis=1)
    hi = nbins - 1 - _np.argmax(used[:, ::-1], axis=1)
    low_cdf = cumulative[indices, lo]
    span = cumulative[indices, hi] - low_cdf
    bottom = anchor[lo]
    spread = anchor[hi] - bottom

    # 块外沿用全局锚定 G，块内按局部 CDF 在 [G(lo), G(hi)] 内重排。
    lut = _np.broadcast_to(anchor, (rows, nbins)).copy()
    positions = _np.arange(nbins, dtype=_np.intp)[None, :]
    inside = (positions >= lo[:, None]) & (positions <= hi[:, None])
    safe_span = _np.where(span > 0.0, span, 1.0)
    middle = bottom[:, None] + (
        (cumulative - low_cdf[:, None]) / safe_span[:, None] * spread[:, None]
    )
    lut = _np.where(inside, middle, lut)
    degenerate = span <= 0.0
    if degenerate.any():
        lut[degenerate] = bottom[degenerate, None]

    single = occupied <= 1
    if single.any():
        single_indices = lo[single]
        lut[single] = anchor[single_indices][:, None]
    empty = occupied == 0
    if empty.any():
        lut[empty] = anchor[nbins // 2]
    return lut


def _enforce_monotone(luts):
    """对块映射表做二维前缀最大，使相邻块在任意 bin 上都不交叉。"""
    ny = len(luts)
    nx = len(luts[0])
    for ty in range(ny):
        row = luts[ty]
        for tx in range(1, nx):
            current = row[tx]
            previous = row[tx - 1]
            row[tx] = [a if a > b else b for a, b in zip(current, previous)]
    for ty in range(1, ny):
        row = luts[ty]
        previous_row = luts[ty - 1]
        for tx in range(nx):
            current = row[tx]
            previous = previous_row[tx]
            row[tx] = [a if a > b else b for a, b in zip(current, previous)]


def _enforce_monotone_numpy(luts):
    _np.maximum.accumulate(luts, axis=1, out=luts)
    _np.maximum.accumulate(luts, axis=0, out=luts)


# ---------------------------------------------------------------------------
# 纯 Python 后端
# ---------------------------------------------------------------------------


def _equalize_python(rows, nx, ny, clip_limit, nbins):
    height = len(rows)
    width = len(rows[0])
    top = nbins - 1

    bins = [[int(round(value * top)) for value in row] for row in rows]
    for row in bins:
        for index, value in enumerate(row):
            if value < 0:
                row[index] = 0
            elif value > top:
                row[index] = top

    hist = [0] * nbins
    for row in bins:
        for value in row:
            hist[value] += 1
    anchor = _global_cdf(hist)
    if anchor is None:
        # 常量图：原样返回。
        return [row[:] for row in rows]

    x_starts, x_ends = _bounds(width, nx)
    y_starts, y_ends = _bounds(height, ny)
    x_tile = _tile_of(width, nx)
    y_tile = _tile_of(height, ny)

    # 与 numpy 路径相同，按块行统计直方图，只为当前块行保留 (nx * nbins)。
    luts = [[None] * nx for _ in range(ny)]
    for ty in range(ny):
        tile_row = [[0] * nbins for _ in range(nx)]
        for y in range(y_starts[ty], y_ends[ty]):
            bins_row = bins[y]
            for x in range(width):
                tile_row[x_tile[x]][bins_row[x]] += 1
        for tx in range(nx):
            luts[ty][tx] = _tile_lut_python(
                tile_row[tx],
                (y_ends[ty] - y_starts[ty]) * (x_ends[tx] - x_starts[tx]),
                clip_limit,
                nbins,
                anchor,
            )
    _enforce_monotone(luts)

    x0, x1, wx = _interp_axis(width, nx, x_starts, x_ends)
    y0, y1, wy = _interp_axis(height, ny, y_starts, y_ends)

    output = []
    for y in range(height):
        row_low = luts[y0[y]]
        if y0[y] == y1[y]:
            blended = row_low
        else:
            w_y = wy[y]
            row_high = luts[y1[y]]
            blended = [
                [(1.0 - w_y) * a + w_y * b for a, b in zip(left, right)]
                for left, right in zip(row_low, row_high)
            ]
        out_row = []
        bins_row = bins[y]
        for x in range(width):
            value = bins_row[x]
            left = blended[x0[x]][value]
            if x0[x] == x1[x]:
                out_row.append(left)
            else:
                out_row.append((1.0 - wx[x]) * left + wx[x] * blended[x1[x]][value])
        output.append(out_row)
    return output


# ---------------------------------------------------------------------------
# numpy 后端
# ---------------------------------------------------------------------------


def _equalize_numpy(arr, nx, ny, clip_limit, nbins):
    height, width = arr.shape

    # 量化到 [0, nbins-1] 的整数 bin；arr 之后不再需要，尽早释放。
    bins = _np.rint(arr * (nbins - 1)).astype(_np.int32)
    _np.clip(bins, 0, nbins - 1, out=bins)
    del arr

    hist = _np.bincount(bins.reshape(-1), minlength=nbins)[:nbins].astype(_np.float64)
    anchor_list = _global_cdf(hist.tolist())
    del hist
    if anchor_list is None:  # pragma: no cover - 常量图在调用方已提前返回
        return (bins.astype(_np.float64) / (nbins - 1)).tolist()
    anchor = _np.asarray(anchor_list, dtype=_np.float64)

    x_starts, x_ends = _bounds(width, nx)
    y_starts, y_ends = _bounds(height, ny)
    x_tile = _np.fromiter(_tile_of(width, nx), dtype=_np.intp, count=width)
    x_keys = x_tile * nbins

    x0, x1, wx = _interp_axis(width, nx, x_starts, x_ends)
    y0, y1, wy = _interp_axis(height, ny, y_starts, y_ends)
    x0 = _np.asarray(x0, dtype=_np.intp)
    x1 = _np.asarray(x1, dtype=_np.intp)
    y0 = _np.asarray(y0, dtype=_np.intp)
    y1 = _np.asarray(y1, dtype=_np.intp)
    wx = _np.asarray(wx, dtype=_np.float64)
    wy = _np.asarray(wy, dtype=_np.float64)

    tile_widths = _np.asarray(
        [x_ends[tx] - x_starts[tx] for tx in range(nx)], dtype=_np.float64
    )

    def raw_row(ty):
        """统计第 ty 个块行的直方图并生成各块映射表 (nx, nbins)。"""
        band = bins[y_starts[ty]:y_ends[ty]]
        keys = (x_keys[None, :] + band).reshape(-1)
        counts = _np.bincount(keys, minlength=nx * nbins)[: nx * nbins]
        counts = counts.reshape(nx, nbins)
        areas = (y_ends[ty] - y_starts[ty]) * tile_widths
        return _tile_luts_numpy(counts, areas, clip_limit, nbins, anchor)

    # 映射表按块行流式生成并做二维前缀最大（行内前缀最大 + 与上一行取最大），
    # 任一时刻只保留上一行和当前行的累计结果，内存与总块数无关。
    # 插值按 y 递增访问，只会请求 index-1 或 index 两个块行，故缓存两行。
    state = {"index": -1, "cumulative_previous": None, "cumulative": None}

    def ordered_row(ty):
        if ty == state["index"]:
            return state["cumulative"]
        if ty == state["index"] - 1:
            return state["cumulative_previous"]
        while state["index"] < ty:
            index = state["index"] + 1
            row = raw_row(index)
            _np.maximum.accumulate(row, axis=0, out=row)
            if state["cumulative"] is not None:
                row = _np.maximum(row, state["cumulative"])
            state["cumulative_previous"] = state["cumulative"]
            state["cumulative"] = row
            state["index"] = index
        return state["cumulative"]

    output = [None] * height
    for y in range(height):
        low_index = int(y0[y])
        high_index = int(y1[y])
        row_low = ordered_row(low_index)
        if low_index == high_index:
            blended = row_low
        else:
            weight = wy[y]
            blended = (1.0 - weight) * row_low + weight * ordered_row(high_index)
        row_bins = bins[y]
        left = blended[x0, row_bins]
        right = blended[x1, row_bins]
        output[y] = ((1.0 - wx) * left + wx * right).tolist()
    return output


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------


class HistEq:
    """分块 CLAHE 均衡器。"""

    def __init__(
        self,
        block_size=None,
        clip_limit=_DEFAULT_CLIP_LIMIT,
        *,
        tile_size=None,
        block=None,
        tiles=None,
        blocks=None,
        grid=None,
        nbins=_DEFAULT_NBINS,
    ):
        pixel_values = [v for v in (block_size, tile_size, block) if v is not None]
        grid_values = [v for v in (tiles, blocks, grid) if v is not None]
        if len(pixel_values) > 1:
            raise InvalidBlockError(
                "invalid block configuration: specify only one of "
                "block_size/tile_size/block"
            )
        if len(grid_values) > 1:
            raise InvalidBlockError(
                "invalid block configuration: specify only one of "
                "tiles/blocks/grid"
            )
        if pixel_values and grid_values:
            raise InvalidBlockError(
                "invalid block configuration: block size and tile count "
                "cannot be combined"
            )

        self._block_px = (
            _parse_positive_pair(pixel_values[0], "block_size")
            if pixel_values
            else None
        )
        self._grid = (
            _parse_positive_pair(grid_values[0], "tiles")
            if grid_values
            else (_DEFAULT_GRID if not pixel_values else None)
        )
        self._clip_limit = _parse_clip_limit(clip_limit)
        self._nbins = _parse_nbins(nbins)

    # -- 配置 ------------------------------------------------------------
    def _override_config(self, block_size, clip_limit, options):
        """把 ``equalize`` 的关键字参数合并成局部配置（不修改 self）。"""
        tile_size = options.pop("tile_size", None)
        block = options.pop("block", None)
        tiles = options.pop("tiles", None)
        blocks = options.pop("blocks", None)
        grid = options.pop("grid", None)
        if options:
            unknown = ", ".join(sorted(options))
            raise TypeError(
                "equalize() got unexpected keyword argument(s): %s" % unknown
            )

        local = HistEq.__new__(HistEq)
        local._block_px = self._block_px
        local._grid = self._grid
        local._clip_limit = (
            self._clip_limit if clip_limit is None else _parse_clip_limit(clip_limit)
        )
        local._nbins = self._nbins

        pixel_values = [v for v in (block_size, tile_size, block) if v is not None]
        grid_values = [v for v in (tiles, blocks, grid) if v is not None]
        if len(pixel_values) > 1 or len(grid_values) > 1:
            raise InvalidBlockError(
                "invalid block configuration: please supply a single block spec"
            )
        if pixel_values and grid_values:
            raise InvalidBlockError(
                "invalid block configuration: block size and tile count "
                "cannot be combined"
            )
        if pixel_values:
            local._block_px = _parse_positive_pair(pixel_values[0], "block_size")
            local._grid = None
        elif grid_values:
            local._grid = _parse_positive_pair(grid_values[0], "tiles")
            local._block_px = None
        return local

    # -- 主入口 ----------------------------------------------------------
    def equalize(self, image, block_size=None, clip_limit=None, **options):
        """对 ``image`` 做 CLAHE，返回新的二维 float 列表。"""
        config = self
        if block_size is not None or clip_limit is not None or options:
            config = self._override_config(block_size, clip_limit, options)

        if _np is not None:
            arr = _validate_numpy(image)
            if arr.min() == arr.max():
                # 常量图原样返回，避免被映射到别的常数。
                return arr.tolist()
            nx, ny = _tile_counts(
                arr.shape[0], arr.shape[1], config._block_px, config._grid
            )
            return _equalize_numpy(arr, nx, ny, config._clip_limit, config._nbins)

        rows = _validate_python(image)
        first = rows[0][0]
        if all(value == first for row in rows for value in row):
            return [row[:] for row in rows]
        nx, ny = _tile_counts(
            len(rows), len(rows[0]), config._block_px, config._grid
        )
        return _equalize_python(rows, nx, ny, config._clip_limit, config._nbins)
