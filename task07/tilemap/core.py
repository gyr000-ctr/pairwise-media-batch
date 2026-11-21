"""tilemap 核心 API。

瓦片坐标 / 世界坐标互转、四叉树式 LOD 金字塔、可见性裁剪与无限分页。

设计约定
--------
* 瓦片边界归属：半开区间 ``[tx*s, (tx+1)*s)``，坐标 -> 瓦片使用向下取整
  （floor），因此负坐标取整与正坐标语义一致、边界点归属右侧瓦片。
* LOD：四叉树金字塔，``level=0`` 使用基础瓦片尺寸，每升一级瓦片在世界
  空间中的尺寸翻倍（``size * 2**level``）。
* 可见性裁剪按"相交即可见"（边界点也算相交，与 floor 语义一致），结果按
  ``(level, ty, tx)`` 升序输出，顺序确定。
* 无限分页：瓦片坐标按页尺寸取模，采用 Python 地板除取模语义，负数正确
  回绕（``-1`` 映射到该轴最后一页）。
* 实例只保存不可变配置，查询不写任何实例状态，多线程并发查询安全。

三类错误彼此可区分（且均为 ``ValueError`` 子类，兼容按 ValueError 捕获）：
``InvalidTileSizeError`` / ``InvalidPageSizeError`` / ``NonFiniteCoordinateError``。
"""

import math

__all__ = [
    "TileMap",
    "TileMapError",
    "InvalidTileSizeError",
    "InvalidPageSizeError",
    "NonFiniteCoordinateError",
]


class TileMapError(ValueError):
    """tilemap 相关错误的基类（同时是 ValueError，便于统一捕获）。"""


class InvalidTileSizeError(TileMapError):
    """非法瓦片尺寸：非有限、非正数或非整数。"""


class InvalidPageSizeError(TileMapError):
    """非法页尺寸：非有限、非正整数。"""


class NonFiniteCoordinateError(TileMapError):
    """坐标非有限（NaN / +-inf）。"""


def _is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


def _is_integral(value):
    if isinstance(value, int):
        return not isinstance(value, bool)
    if isinstance(value, float):
        return math.isfinite(value) and float(value).is_integer()
    return False


def _check_finite(value, name):
    if not _is_finite_number(value):
        raise NonFiniteCoordinateError(
            "%s 必须是有限数值，收到 %r" % (name, value))


def _check_level(level):
    if not _is_integral(level) or level < 0:
        raise ValueError("level 必须是非负整数，收到 %r" % (level,))
    return int(level)


class TileMap:
    """瓦片地图。

    Parameters
    ----------
    size:
        基础瓦片（level 0）的世界尺寸，必须是有限的正整数（允许
        ``16.0`` 这类整值浮点）。
    page_w, page_h:
        分页尺寸（以瓦片数计），均为正整数。两者都给定时启用无限分页；
        也可只给一个整数表示正方形页。
    """

    def __init__(self, size, page_w=None, page_h=None):
        if not _is_finite_number(size):
            raise InvalidTileSizeError(
                "瓦片尺寸必须是有限数值，收到 %r" % (size,))
        if size <= 0:
            raise InvalidTileSizeError(
                "瓦片尺寸必须为正数，收到 %r" % (size,))
        if not _is_integral(size):
            raise InvalidTileSizeError(
                "瓦片尺寸必须是整数，收到 %r" % (size,))

        if page_w is None and page_h is None:
            # 未启用分页
            page_w = page_h = 0
        else:
            if page_w is None or page_h is None:
                # 只给一个维度时视为正方形页
                page_w = page_h = page_w if page_h is None else page_h
            for dim, name in ((page_w, "page_w"), (page_h, "page_h")):
                if not _is_finite_number(dim):
                    raise InvalidPageSizeError(
                        "页尺寸 %s 必须是有限数值，收到 %r" % (name, dim))
                if dim <= 0 or not _is_integral(dim):
                    raise InvalidPageSizeError(
                        "页尺寸 %s 必须是正整数，收到 %r" % (name, dim))

        # 全部转成不可变的原生 int/float：实例状态此后只读，天然并发安全。
        self._size = float(size)
        self._page_w = int(page_w)
        self._page_h = int(page_h)
        self._paged = self._page_w > 0

    # ------------------------------------------------------------------ #
    # 基础属性
    # ------------------------------------------------------------------ #
    @property
    def size(self):
        """基础瓦片尺寸（浮点）。"""
        return self._size

    @property
    def page_width(self):
        """页宽（瓦片数）；未启用分页时为 0。"""
        return self._page_w

    @property
    def page_height(self):
        """页高（瓦片数）；未启用分页时为 0。"""
        return self._page_h

    def tile_size(self, level=0):
        """返回指定 LOD 级别的瓦片世界尺寸（浮点）。"""
        level = _check_level(level)
        return self._size * (1 << level)

    # ------------------------------------------------------------------ #
    # 坐标互转
    # ------------------------------------------------------------------ #
    def to_tile(self, x, y, level=0):
        """世界坐标 -> 瓦片坐标 ``(tx, ty)``。

        向下取整，负坐标取整正确；边界点归属编号更大的瓦片。
        """
        _check_finite(x, "x")
        _check_finite(y, "y")
        level = _check_level(level)
        s = self._size * (1 << level)
        return int(math.floor(x / s)), int(math.floor(y / s))

    def to_world(self, tx, ty, level=0):
        """瓦片坐标 -> 该瓦片左下角的世界坐标 ``(x, y)``（浮点）。"""
        _check_finite(tx, "tx")
        _check_finite(ty, "ty")
        level = _check_level(level)
        s = self._size * (1 << level)
        return float(tx) * s, float(ty) * s

    # ------------------------------------------------------------------ #
    # 四叉树式 LOD 可见性裁剪
    # ------------------------------------------------------------------ #
    def visible_tiles(self, x0, y0=None, x1=None, y1=None, level=0):
        """相机矩形与指定 LOD 级别相交的瓦片列表。

        支持的调用形式：

        * ``visible_tiles((x0, y0, x1, y1), level)`` —— 矩形序列 + 级别
        * ``visible_tiles(x0, y0, x1, y1, level)`` —— 四个坐标 + 级别
        * 级别也可用关键字 ``level=`` 传递。

        返回 ``[(level, tx, ty), ...]``，按 ``(level, ty, tx)`` 行优先升序；
        相交即可见（边界接触也算），与逐级暴力遍历结果集合完全一致。
        """
        # 矩形序列形式：visible_tiles(rect) 或 visible_tiles(rect, level)。
        # 注意必须判 None 而非真值，坐标 0.0 是合法入参。
        if isinstance(x0, (tuple, list)):
            if not (y0 is None or _is_integral(y0)) or \
                    not (x1 is None and y1 is None):
                raise TypeError(
                    "矩形序列形式应为 visible_tiles((x0, y0, x1, y1), level)")
            if y0 is not None:
                level = y0
            try:
                x0, y0, x1, y1 = x0
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "相机矩形需要 4 个坐标 (x0, y0, x1, y1)") from exc
        elif y0 is None or x1 is None or y1 is None:
            raise TypeError(
                "坐标形式需要 4 个参数 visible_tiles(x0, y0, x1, y1)")
        rect = (x0, y0, x1, y1)
        for i, name in enumerate(("x0", "y0", "x1", "y1")):
            _check_finite(rect[i], name)
        level = _check_level(level)

        xa, xb = rect[0], rect[2]
        ya, yb = rect[1], rect[3]
        if xa > xb:
            xa, xb = xb, xa
        if ya > yb:
            ya, yb = yb, ya

        s = self._size * (1 << level)
        # 相交即可见：左端 floor，右端 floor（边界点属于右侧瓦片）。
        tx0 = int(math.floor(xa / s))
        tx1 = int(math.floor(xb / s))
        ty0 = int(math.floor(ya / s))
        ty1 = int(math.floor(yb / s))
        return [(level, tx, ty)
                for ty in range(ty0, ty1 + 1)
                for tx in range(tx0, tx1 + 1)]

    # 语义别名：题面用语为 query_tiles / cull。
    def query_tiles(self, *args, **kwargs):
        """``visible_tiles`` 的别名。"""
        return self.visible_tiles(*args, **kwargs)

    def cull(self, *args, **kwargs):
        """``visible_tiles`` 的别名（可见性裁剪）。"""
        return self.visible_tiles(*args, **kwargs)

    def visible_tiles_levels(self, rect, levels):
        """多个 LOD 级别的可见瓦片，按级别升序拼接，不漏不重。

        ``levels`` 为可迭代的非负整数级别集合。
        """
        result = []
        for level in sorted({_check_level(lv) for lv in levels}):
            result.extend(self.visible_tiles(rect, level=level))
        return result

    def brute_force_tiles(self, x0, y0, x1, y1, level=0):
        """参考实现：逐级暴力遍历（逐瓦片做几何相交判断）。

        与 :meth:`visible_tiles` 结果集合与顺序完全一致，供交叉验证。
        """
        for value, name in ((x0, "x0"), (y0, "y0"),
                            (x1, "x1"), (y1, "y1")):
            _check_finite(value, name)
        level = _check_level(level)
        xa, xb = (x0, x1) if x0 <= x1 else (x1, x0)
        ya, yb = (y0, y1) if y0 <= y1 else (y1, y0)

        s = self._size * (1 << level)
        tx0 = int(math.floor(xa / s))
        tx1 = int(math.floor(xb / s))
        ty0 = int(math.floor(ya / s))
        ty1 = int(math.floor(yb / s))
        result = []
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                left, bottom = float(tx) * s, float(ty) * s
                # 半开瓦片 [left, left+s) × [bottom, bottom+s)：
                # 相交即可见（含边界接触）。
                if left <= xb and left + s >= xa \
                        and bottom <= yb and bottom + s >= ya:
                    result.append((level, tx, ty))
        return result

    # ------------------------------------------------------------------ #
    # 无限分页
    # ------------------------------------------------------------------ #
    def _require_paging(self):
        if not self._paged:
            raise InvalidPageSizeError(
                "未配置分页尺寸：构造时需传入 page_w/page_h")

    def wrap_tile(self, tx, ty, level=0):
        """瓦片坐标按页取模，返回页内坐标 ``(wx, wy)``。

        Python 地板除取模语义：``-1`` 回绕到该轴最后一个位置。
        """
        _check_finite(tx, "tx")
        _check_finite(ty, "ty")
        level = _check_level(level)
        self._require_paging()
        return (int(tx) % self._page_w, int(ty) % self._page_h)

    def page_of(self, tx, ty, level=0):
        """返回瓦片所在页坐标 ``(page_x, page_y)``，负坐标页号为负。"""
        _check_finite(tx, "tx")
        _check_finite(ty, "ty")
        level = _check_level(level)
        self._require_paging()
        itx, ity = int(tx), int(ty)
        # 与取模配套的地板除，保证 tx == page*size + wrap。
        return (itx // self._page_w, ity // self._page_h)

    def to_page_tile(self, x, y, level=0):
        """世界坐标 -> ``(page_x, page_y, local_tx, local_ty)``。"""
        tx, ty = self.to_tile(x, y, level)
        px, py = self.page_of(tx, ty, level)
        wx, wy = self.wrap_tile(tx, ty, level)
        return px, py, wx, wy
