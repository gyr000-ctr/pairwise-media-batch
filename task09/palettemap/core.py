"""palettemap 核心 API：RGB 最近色索引映射。

最近色查询基于 3 维 KD 树（空间加速结构），不做逐像素线性扫描：

* 有 C 编译器时使用 ``palettemap/_kd.c`` 编译出的原生内核（按源码哈希缓存到
  临时目录），100 万像素在数十毫秒内完成；该内核精确、平局取最小索引，
  与线性扫描结果一致；
* 没有编译器时退化为同样只读的 numpy 向量化 KD 树，结果一致（较慢）。

构造完成后树为只读结构，查询只写调用方传入的缓冲区，因此同一实例可在
多线程中并发查询。所有错误都是 :class:`PaletteMapError` 的子类（同时兼容
``except ValueError``），空表、维度不符、非有限颜色彼此可区分。
"""

from __future__ import annotations

import itertools as _itertools
import math

try:  # numpy 是主路径；缺失时退化为纯 Python KD 树
    import numpy as _np
except ImportError:  # pragma: no cover - 正常环境均带 numpy
    _np = None

__all__ = [
    "PaletteMap",
    "PaletteMapError",
    "EmptyPaletteError",
    "PaletteDimensionError",
    "ColorDimensionError",
    "NonFiniteColorError",
]

_LEAF_SIZE = 16


# ---------------------------------------------------------------------------
# 可区分的错误类型
# ---------------------------------------------------------------------------
class PaletteMapError(ValueError):
    """调色板映射相关错误的基类（兼容 ``except ValueError``）。"""


class EmptyPaletteError(PaletteMapError):
    """调色板为空。"""


class PaletteDimensionError(PaletteMapError):
    """调色板条目的维度不是 3。"""


class ColorDimensionError(PaletteMapError):
    """待映射颜色的维度不是 3（或图像维度不合法）。"""


class NonFiniteColorError(PaletteMapError):
    """调色板或颜色中包含 NaN / Inf。"""


# ---------------------------------------------------------------------------
# numpy 向量化 KD 树（无编译器时的回退路径）
# ---------------------------------------------------------------------------
class _NumpyKDTree:
    """只读 KD 树，查询整批点（构建后无共享可变状态）。"""

    leaf_size = _LEAF_SIZE

    def __init__(self, points):
        self.points = points
        n = points.shape[0]
        self.n = n
        self._sorted_pts = _np.empty_like(points)
        self._sorted_ids = _np.empty(n, dtype=_np.intp)
        self._left = []
        self._right = []
        self._start = []
        self._end = []
        self._bmin = []
        self._bmax = []
        self._cursor = 0
        self._root = self._build(_np.arange(n, dtype=_np.intp), points)

        self._left = _np.asarray(self._left, dtype=_np.intp)
        self._right = _np.asarray(self._right, dtype=_np.intp)
        self._start = _np.asarray(self._start, dtype=_np.intp)
        self._end = _np.asarray(self._end, dtype=_np.intp)
        self._bmin = _np.asarray(self._bmin, dtype=_np.float64)
        self._bmax = _np.asarray(self._bmax, dtype=_np.float64)

        # 构建后只读，保证并发查询安全。
        for arr in (self._sorted_pts, self._sorted_ids, self._left, self._right,
                    self._start, self._end, self._bmin, self._bmax):
            arr.flags.writeable = False

    def _build(self, ids, points):
        node = len(self._left)
        self._left.append(-1)
        self._right.append(-1)
        self._start.append(0)
        self._end.append(0)

        sub = points[ids]
        lo = sub.min(axis=0)
        hi = sub.max(axis=0)
        self._bmin.append(lo)
        self._bmax.append(hi)

        n = len(ids)
        if n <= self.leaf_size or bool((lo == hi).all()):
            ids = _np.sort(ids)  # 叶子内按索引升序，平局取最小索引
            start = self._cursor
            self._sorted_pts[start:start + n] = points[ids]
            self._sorted_ids[start:start + n] = ids
            self._cursor += n
            self._start[node] = start
            self._end[node] = start + n
            return node

        dim = int(_np.argmax(hi - lo))
        mid = n // 2
        order = _np.argpartition(sub[:, dim], mid)
        self._left[node] = self._build(ids[order[:mid]], points)
        self._right[node] = self._build(ids[order[mid:]], points)
        return node

    @staticmethod
    def _lower_bound(q, lo, hi):
        d = _np.maximum(lo - q, 0.0) + _np.maximum(q - hi, 0.0)
        return d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]

    def query(self, queries):
        """返回每个查询点的最近色索引（距离相同取最小索引）。"""
        m = queries.shape[0]
        best_d2 = _np.full(m, _np.inf, dtype=_np.float64)
        best_idx = _np.full(m, self.n, dtype=_np.intp)
        if m == 0 or self.n == 0:
            return best_idx

        left, right = self._left, self._right
        start, end = self._start, self._end
        bmin, bmax = self._bmin, self._bmax
        sorted_pts, sorted_ids = self._sorted_pts, self._sorted_ids

        stack = [(self._root, _np.arange(m, dtype=_np.intp))]
        while stack:
            node, sel = stack.pop()
            if sel.size == 0:
                continue
            # 包围盒下界剪枝；用 <= 保留等距候选以维持最小索引语义。
            lb = self._lower_bound(queries[sel], bmin[node], bmax[node])
            keep = lb <= best_d2[sel]
            if not keep.all():
                sel = sel[keep]
                if sel.size == 0:
                    continue

            lft = left[node]
            if lft < 0:  # 叶子：对叶子内少量候选点做精确比较
                s = start[node]
                e = end[node]
                q = queries[sel]
                d2 = ((q[:, None, :] - sorted_pts[s:e][None, :, :]) ** 2).sum(axis=2)
                j = _np.argmin(d2, axis=1)
                rows = _np.arange(sel.size)
                d = d2[rows, j]
                cand = sorted_ids[s:e][j]
                cur_d = best_d2[sel]
                cur_i = best_idx[sel]
                better = (d < cur_d) | ((d == cur_d) & (cand < cur_i))
                if better.any():
                    target = sel[better]
                    best_d2[target] = d[better]
                    best_idx[target] = cand[better]
            else:
                rgt = right[node]
                q = queries[sel]
                lb_l = self._lower_bound(q, bmin[lft], bmax[lft])
                lb_r = self._lower_bound(q, bmin[rgt], bmax[rgt])
                near_l = lb_l <= lb_r
                # 远侧先按当前上界剪枝；上界只会变小，剪掉是安全的。
                keep_l = near_l & (lb_r <= best_d2[sel])
                keep_r = (~near_l) & (lb_l <= best_d2[sel])
                if keep_r.any():
                    stack.append((lft, sel[keep_r]))
                if keep_l.any():
                    stack.append((rgt, sel[keep_l]))
                # 近侧全部保留，出栈时再按当时上界剪枝。
                if near_l.any():
                    stack.append((lft, sel[near_l]))
                if (~near_l).any():
                    stack.append((rgt, sel[~near_l]))
        return best_idx


# ---------------------------------------------------------------------------
# 纯 Python KD 树（无 numpy 时的回退路径）
# ---------------------------------------------------------------------------
class _PyKDTree:
    leaf_size = _LEAF_SIZE

    def __init__(self, points):
        self.points = points
        self.nodes = []
        self._root = self._build(list(range(len(points))))

    def _build(self, ids):
        slot = len(self.nodes)
        self.nodes.append(None)
        pts = self.points
        lo = [min(pts[i][d] for i in ids) for d in range(3)]
        hi = [max(pts[i][d] for i in ids) for d in range(3)]
        n = len(ids)
        if n <= self.leaf_size or lo == hi:
            self.nodes[slot] = ("leaf", sorted(ids), lo, hi)
            return slot
        dim = max(range(3), key=lambda d: hi[d] - lo[d])
        ids = sorted(ids, key=lambda i: pts[i][dim])
        mid = n // 2
        left = self._build(ids[:mid])
        right = self._build(ids[mid:])
        self.nodes[slot] = ("node", left, right, lo, hi)
        return slot

    @staticmethod
    def _box_dist(q, lo, hi):
        total = 0.0
        for d in range(3):
            if q[d] < lo[d]:
                delta = lo[d] - q[d]
                total += delta * delta
            elif q[d] > hi[d]:
                delta = q[d] - hi[d]
                total += delta * delta
        return total

    def query(self, queries):
        out = []
        nodes = self.nodes
        pts = self.points
        for q in queries:
            best_d2 = math.inf
            best_i = len(pts)
            stack = [self._root]
            while stack:
                node = nodes[stack.pop()]
                if self._box_dist(q, node[-2], node[-1]) > best_d2:
                    continue
                if node[0] == "leaf":
                    for i in node[1]:
                        p = pts[i]
                        d2 = ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
                              + (q[2] - p[2]) ** 2)
                        if d2 < best_d2 or (d2 == best_d2 and i < best_i):
                            best_d2 = d2
                            best_i = i
                else:
                    lft, rgt = node[1], node[2]
                    dl = self._box_dist(q, nodes[lft][-2], nodes[lft][-1])
                    dr = self._box_dist(q, nodes[rgt][-2], nodes[rgt][-1])
                    if dl <= dr:
                        if dr <= best_d2:
                            stack.append(rgt)
                        stack.append(lft)
                    else:
                        if dl <= best_d2:
                            stack.append(lft)
                        stack.append(rgt)
            out.append(best_i)
        return out


# ---------------------------------------------------------------------------
# 输入解析
# ---------------------------------------------------------------------------
def _manual_color(color, dim_error, what="color"):
    """把颜色解析成长度 3 的 float 元组；错误类型可区分。"""
    if isinstance(color, (str, bytes)):
        raise TypeError(f"{what} must be a sequence of 3 numbers")
    try:
        items = list(color)
    except TypeError:
        raise dim_error(f"{what} must be a sequence of 3 numbers") from None
    if len(items) != 3:
        raise dim_error(f"{what} dimension must be 3, got {len(items)}")
    vals = []
    for v in items:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            raise TypeError(f"{what} components must be numbers") from None
        if not math.isfinite(fv):
            raise NonFiniteColorError(f"{what} contains non-finite value {fv!r}")
        vals.append(fv)
    return tuple(vals)


def _parse_palette(palette):
    if palette is None:
        raise TypeError("palette must be a non-empty sequence of colors")
    entries = list(palette)
    if not entries:
        raise EmptyPaletteError("palette must not be empty")

    if _np is not None:
        try:
            arr = _np.asarray(entries, dtype=_np.float64)
        except (TypeError, ValueError):
            arr = None
        if arr is not None and arr.dtype == _np.float64:
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise PaletteDimensionError(
                    f"palette colors must have dimension 3, got shape {arr.shape}"
                )
            if not bool(_np.isfinite(arr).all()):
                raise NonFiniteColorError("palette contains non-finite values")
            return arr.reshape(arr.shape[0], 3)

    parsed = [_manual_color(c, PaletteDimensionError, "palette color") for c in entries]
    if _np is not None:
        return _np.asarray(parsed, dtype=_np.float64)
    return parsed


def _parse_index_color(color):
    if _np is not None:
        if isinstance(color, (str, bytes)):
            raise TypeError("color must be a sequence of 3 numbers")
        try:
            arr = _np.asarray(color, dtype=_np.float64)
        except (TypeError, ValueError):
            return _manual_color(color, ColorDimensionError)
        if arr.dtype != _np.float64:
            return _manual_color(color, ColorDimensionError)
        if arr.ndim != 1 or arr.shape[0] != 3:
            raise ColorDimensionError(
                f"color dimension must be 3, got shape {arr.shape}"
            )
        if not bool(_np.isfinite(arr).all()):
            raise NonFiniteColorError("color contains non-finite values")
        return arr
    return _manual_color(color, ColorDimensionError)


def _manual_image(image):
    try:
        outer = list(image)
    except TypeError:
        raise TypeError("image must be a nested sequence of RGB colors") from None
    rows = []
    for row in outer:
        if isinstance(row, (str, bytes)):
            raise ColorDimensionError("image rows must be sequences of colors")
        try:
            colors = list(row)
        except TypeError:
            raise ColorDimensionError("image rows must be sequences of colors") from None
        rows.append([_manual_color(c, ColorDimensionError, "image color") for c in colors])
    return rows


def _parse_image(image):
    if image is None:
        raise TypeError("image must be a nested sequence of RGB colors")

    if _np is not None:
        if isinstance(image, _np.ndarray):
            arr = _np.asarray(image, dtype=_np.float64)
            if arr.size == 0:
                return _np.empty((len(arr), 0, 3), dtype=_np.float64)
            if arr.ndim == 3 and arr.shape[2] == 3:
                if not bool(_np.isfinite(arr).all()):
                    raise NonFiniteColorError("image contains non-finite values")
                return arr
            raise ColorDimensionError(
                f"image must have shape (h, w, 3), got {arr.shape}"
            )

        if not isinstance(image, (list, tuple)):
            raise TypeError("image must be a nested sequence of RGB colors")
        height = len(image)
        if height == 0:
            return _np.empty((0, 0, 3), dtype=_np.float64)
        first = image[0]
        try:
            width = len(first)
        except TypeError:
            raise ColorDimensionError("image rows must be sequences of colors") from None
        if width == 0:
            for row in image:
                if not isinstance(row, (list, tuple)) or len(row):
                    raise ColorDimensionError("image rows must be rectangular")
            return _np.empty((height, 0, 3), dtype=_np.float64)

        arr = _fast_flatten(image, height, width)
        if arr is None:
            arr = _slow_image(image)
        if not bool(_np.isfinite(arr).all()):
            raise NonFiniteColorError("image contains non-finite values")
        return arr

    rows = _manual_image(image)
    return rows


def _slow_image(image):
    """逐项解析，给出维度/非有限等可区分错误。"""
    rows = _manual_image(image)
    arr = _np.asarray(rows, dtype=_np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ColorDimensionError(f"image must have shape (h, w, 3), got {arr.shape}")
    return arr


def _fast_flatten(image, height, width):
    """单趟展平规则的嵌套 list/tuple；形状不规则时返回 None。"""
    chain_from = _itertools.chain.from_iterable
    total = height * width * 3
    values = chain_from(chain_from(image))
    try:
        arr = _np.fromiter(values, dtype=_np.float64, count=total)
        extra = next(values)  # 多出元素说明行或颜色长度不齐
    except StopIteration:
        extra = None
    except (TypeError, ValueError):
        return None
    if extra is not None:
        return None
    try:
        return arr.reshape(height, width, 3)
    except ValueError:  # 行长度不齐导致元素总数对不上
        return None


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------
_native_tree = None
_native_checked = False


def _get_native():
    """惰性加载原生内核；不可用时返回 None（只探测一次）。"""
    global _native_tree, _native_checked
    if _native_checked:
        return _native_tree
    _native_checked = True
    try:
        from . import _native
    except Exception:  # pragma: no cover - 缺少 ctypes 等极端情况
        return None
    lib = _native.load()
    if lib is not None:
        _native_tree = _native.NativeTree
    return _native_tree


class PaletteMap:
    """把 RGB 颜色映射到调色板中最近颜色的索引。"""

    def __init__(self, palette):
        """按调色板构造 KD 树（只读，可并发查询）。"""
        points = _parse_palette(palette)
        if _np is None:  # pragma: no cover - 无 numpy 环境
            self._palette = [tuple(float(v) for v in c) for c in points]
            self._n = len(self._palette)
            self._tree = _PyKDTree(self._palette)
            self._arr = None
            return

        arr = _np.ascontiguousarray(points, dtype=_np.float64).reshape(-1, 3)
        self._palette = arr
        self._n = int(arr.shape[0])
        self._arr = arr

        native_cls = _get_native()
        self._tree = None
        if native_cls is not None:
            try:
                lib = _native_lib()
                if lib is not None:
                    self._tree = native_cls(lib, arr)
            except (MemoryError, OSError):
                self._tree = None
        if self._tree is None:
            self._tree = _NumpyKDTree(arr)
            self._native = False
        else:
            self._native = True

    @property
    def palette_size(self):
        """调色板颜色数量。"""
        return self._n

    def index_of(self, color):
        """返回 color 的最近调色板索引（int，落在 [0, n)）。"""
        if self._arr is None:  # pragma: no cover - 无 numpy 环境
            vals = _manual_color(color, ColorDimensionError)
            return int(self._tree.query([vals])[0])
        q = _parse_index_color(color).reshape(1, 3)
        return int(self._tree.query(q)[0])

    def map_image(self, image):
        """整图映射；返回与输入同形状的二维索引列表（不修改输入）。"""
        parsed = _parse_image(image)
        if self._arr is None:  # pragma: no cover - 无 numpy 环境
            return [[int(i) for i in self._tree.query(row)] for row in parsed]
        if parsed.size == 0 or parsed.shape[1] == 0:
            return [[] for _ in range(parsed.shape[0])]
        h, w, _ = parsed.shape
        # 索引直接算成 int32：tolist() 更快，也不必先升到 int64 再降回来。
        query = self._tree.query
        idx = query(parsed.reshape(-1, 3), _np.int32) if self._native else query(
            parsed.reshape(-1, 3)
        )
        return idx.reshape(h, w).tolist()


def _native_lib():
    from . import _native

    return _native.load()


if _np is not None:  # 首次导入就把内核编译好，避免第一次 map_image 才付编译耗时
    try:
        _get_native()
    except Exception:  # pragma: no cover - 任何异常都退化为 numpy 实现
        pass
