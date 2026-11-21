"""gradient 核心 API。

约束：
- 端点精确：停靠点位置采样直接返回该停靠点颜色，不经插值往返。
- 停靠点位置单调不减；相等位置合并（后者覆盖前者），不报错。
- 采样位置越界 / 非有限、非法停靠点（空序列、颜色通道缺失或越界等）抛 ValueError。
- 支持两种插值色彩空间：
  * ``srgb``（默认）：在 sRGB 编码值上直接线性插值；
  * ``linear``：先 sRGB→linear 插值再 linear→sRGB，中间调与直接插值有可观测差异。
- :meth:`Gradient.gradient_at` 返回逐通道解析导数（同色彩空间约定）。
- 内部状态为不可变元组，写入时整体替换，读路径无锁，线程安全。
"""

import bisect
import math
import threading

__all__ = ["Gradient"]

_SRGB_ALPHA = 0.055
_SRGB_CUTOFF = 0.04045
_LINEAR_CUTOFF = 0.0031308
_INV_GAMMA = 1.0 / 2.4

_SPACE_SRGB = "srgb"
_SPACE_LINEAR = "linear"
_SPACE_ALIASES = {
    "srgb": _SPACE_SRGB,
    "rgb": _SPACE_SRGB,
    "direct": _SPACE_SRGB,
    "linear": _SPACE_LINEAR,
    "linear-rgb": _SPACE_LINEAR,
    "linear_rgb": _SPACE_LINEAR,
    "linearsrgb": _SPACE_LINEAR,
}


def _srgb_to_linear(c):
    if c <= _SRGB_CUTOFF:
        return c / 12.92
    return ((c + _SRGB_ALPHA) / (1.0 + _SRGB_ALPHA)) ** 2.4


def _linear_to_srgb(c):
    if c <= _LINEAR_CUTOFF:
        return c * 12.92
    return (1.0 + _SRGB_ALPHA) * (c ** _INV_GAMMA) - _SRGB_ALPHA


def _resolve_space(space):
    if not isinstance(space, str):
        raise ValueError("space 必须是字符串: %r" % (space,))
    key = space.strip().lower()
    try:
        return _SPACE_ALIASES[key]
    except KeyError:
        raise ValueError("未知插值色彩空间: %r" % (space,))


def _is_real(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _validate_position(t):
    if not _is_real(t):
        raise ValueError("位置必须是有限实数: %r" % (t,))
    if not (0.0 <= float(t) <= 1.0):
        raise ValueError("位置越界，要求 [0, 1]: %r" % (t,))
    return float(t)


def _validate_color(color):
    try:
        n = len(color)
    except TypeError:
        raise ValueError("颜色必须是 3 通道序列: %r" % (color,))
    if n != 3:
        raise ValueError("颜色必须恰好包含 3 个通道: %r" % (color,))
    out = []
    for v in color:
        if not _is_real(v):
            raise ValueError("颜色通道必须是有限实数: %r" % (color,))
        f = float(v)
        if not (0.0 <= f <= 1.0):
            raise ValueError("颜色通道越界，要求 [0, 1]: %r" % (color,))
        out.append(f)
    return tuple(out)


class Gradient:
    """分段线性颜色渐变。

    Parameters
    ----------
    stops:
        ``(位置, (r, g, b))`` 序列，位置单调不减，相等位置会被合并。
    space:
        插值色彩空间，``"srgb"``（默认，直接插值）或 ``"linear"``（线性光空间插值）。
    mode:
        ``space`` 的别名参数，同时给出时以 ``mode`` 为准。
    """

    def __init__(self, stops, space="srgb", mode=None):
        target = _resolve_space(mode if mode is not None else space)
        try:
            items = list(stops)
        except TypeError:
            raise ValueError("stops 必须是 (位置, 颜色) 序列: %r" % (stops,))
        if not items:
            raise ValueError("至少需要一个停靠点")

        positions = []
        colors = []
        prev = -math.inf
        for stop in items:
            try:
                raw_t, raw_color = stop
            except (TypeError, ValueError):
                raise ValueError("非法停靠点，应为 (位置, 颜色): %r" % (stop,))
            t = _validate_position(raw_t)
            color = _validate_color(raw_color)
            if t < prev:
                raise ValueError("停靠点位置必须单调不减: %r 出现在 %r 之后" % (t, prev))
            if positions and t == positions[-1]:
                # 重复位置：合并，后者覆盖前者，而不是报错。
                colors[-1] = color
            else:
                positions.append(t)
                colors.append(color)
            prev = t

        self._lock = threading.RLock()
        self._state = self._build_state(tuple(positions), tuple(colors), target)

    @staticmethod
    def _build_state(positions, colors, space):
        linears = tuple(
            tuple(_srgb_to_linear(c) for c in color) for color in colors
        )
        # 不可变状态元组：读路径只取一次快照，写入整体替换。
        return (positions, colors, linears, space)

    @property
    def space(self):
        return self._state[3]

    def add_stop(self, t, color):
        """追加（或合并）一个停靠点。"""
        pos = _validate_position(t)
        rgb = _validate_color(color)
        with self._lock:
            positions, colors, _, space = self._state
            idx = bisect.bisect_left(positions, pos)
            if idx < len(positions) and positions[idx] == pos:
                new_colors = colors[:idx] + (rgb,) + colors[idx + 1:]
                new_positions = positions
            else:
                new_positions = positions[:idx] + (pos,) + positions[idx:]
                new_colors = colors[:idx] + (rgb,) + colors[idx:]
            self._state = self._build_state(new_positions, new_colors, space)

    def _resolve_call_space(self, space, mode):
        if mode is not None:
            space = mode
        if space is None:
            return self._state[3]
        return _resolve_space(space)

    def sample(self, t, space=None, mode=None):
        """在位置 t 采样，返回 (r, g, b) 元组（sRGB 编码值，通道 ∈ [0, 1]）。"""
        if not _is_real(t):
            raise ValueError("采样位置必须是有限实数: %r" % (t,))
        t = float(t)
        if not (0.0 <= t <= 1.0):
            raise ValueError("采样位置越界，要求 [0, 1]: %r" % (t,))

        positions, colors, linears, default_space = self._state
        target = self._resolve_call_space(space, mode)
        n = len(positions)

        i = bisect.bisect_left(positions, t)
        if i < n and positions[i] == t:
            # 端点 / 停靠点精确：直接返回存储颜色，避免色彩空间往返误差。
            return colors[i]
        if n == 1 or i == 0 or i == n:
            # 单停靠点常量，或落在停靠点覆盖范围之外：钳制到最近端点。
            return colors[0] if i == 0 else colors[-1]

        p0, p1 = positions[i - 1], positions[i]
        u = (t - p0) / (p1 - p0)
        c0, c1 = colors[i - 1], colors[i]

        if target == _SPACE_SRGB:
            return (
                c0[0] + (c1[0] - c0[0]) * u,
                c0[1] + (c1[1] - c0[1]) * u,
                c0[2] + (c1[2] - c0[2]) * u,
            )

        l0, l1 = linears[i - 1], linears[i]
        return (
            min(1.0, max(0.0, _linear_to_srgb(l0[0] + (l1[0] - l0[0]) * u))),
            min(1.0, max(0.0, _linear_to_srgb(l0[1] + (l1[1] - l0[1]) * u))),
            min(1.0, max(0.0, _linear_to_srgb(l0[2] + (l1[2] - l0[2]) * u))),
        )

    def gradient_at(self, t, space=None, mode=None):
        """返回 t 处逐通道导数 d(r,g,b)/dt。

        钳制区段（停靠点覆盖范围之外）导数为 0；停靠点结点处取右侧区间导数，
        末端点取左侧区间导数。
        """
        if not _is_real(t):
            raise ValueError("采样位置必须是有限实数: %r" % (t,))
        t = float(t)
        if not (0.0 <= t <= 1.0):
            raise ValueError("采样位置越界，要求 [0, 1]: %r" % (t,))

        positions, colors, linears, default_space = self._state
        target = self._resolve_call_space(space, mode)
        n = len(positions)
        zero = (0.0, 0.0, 0.0)
        if n == 1:
            return zero

        i = bisect.bisect_right(positions, t)
        if i == 0:
            # t < 首个停靠点：钳制区。
            return zero
        if i >= n:
            if t > positions[-1]:
                return zero
            lo, hi = n - 2, n - 1
        else:
            lo, hi = i - 1, i

        p0, p1 = positions[lo], positions[hi]
        inv_dp = 1.0 / (p1 - p0)
        c0, c1 = colors[lo], colors[hi]

        if target == _SPACE_SRGB:
            return (
                (c1[0] - c0[0]) * inv_dp,
                (c1[1] - c0[1]) * inv_dp,
                (c1[2] - c0[2]) * inv_dp,
            )

        l0, l1 = linears[lo], linears[hi]
        result = []
        for k in range(3):
            dlin = (l1[k] - l0[k]) * inv_dp
            lin = l0[k] + (l1[k] - l0[k]) * ((t - p0) * inv_dp)
            if lin <= _LINEAR_CUTOFF:
                deriv = 12.92
            else:
                deriv = (1.0 + _SRGB_ALPHA) * _INV_GAMMA * lin ** (_INV_GAMMA - 1.0)
            result.append(deriv * dlin)
        return tuple(result)
