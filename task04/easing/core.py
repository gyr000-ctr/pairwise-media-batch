"""缓动函数库：标准缓动、自定义 cubic-bezier、弹簧缓动与反解。

设计约束：
- 所有内置缓动满足 f(0)=0、f(1)=1、单调不减（弹簧缓动是有超调的
  阶跃响应，属于文档化的例外，不出现在 names() 枚举中）。
- cubic-bezier 求值：牛顿迭代 + 二分回退，精度远优于 1e-6；
  退化控制点（p1x=p2x、全 0、全 1）不除零、不死循环。
- inverse(name, y)：反解精度远优于 1e-9；非单调目标抛出可区分的
  NonMonotonicError。
- 采样表缓存命中与逐次求值逐位一致（表由同一求值器生成，缓存的是
  同一批 float 对象）。
- 全部函数无状态、并发安全（唯一共享状态是加锁的采样表缓存）。
"""

import math
import operator
import threading

__all__ = [
    "Easing",
    "EasingError",
    "UnknownEasingError",
    "ParameterError",
    "NonMonotonicError",
    "NAMES",
]


# ---------------------------------------------------------------------------
# 可区分的错误类型（均为 ValueError 子类，兼容既有测试）
# ---------------------------------------------------------------------------

class EasingError(ValueError):
    """缓动库所有错误的基类。"""


class UnknownEasingError(EasingError):
    """未知缓动名称。"""


class ParameterError(EasingError):
    """参数越界、非有限或类型非法。"""


class NonMonotonicError(EasingError):
    """对非单调函数请求反解。"""


# ---------------------------------------------------------------------------
# 内置缓动：f(0)=0、f(1)=1（精确）、单调不减
# ---------------------------------------------------------------------------

def _linear(t):
    return t


def _ease_in(t):
    return t * t


def _ease_out(t):
    u = 1.0 - t
    return 1.0 - u * u


def _ease_in_out(t):
    if t < 0.5:
        return 2.0 * t * t
    u = -2.0 * t + 2.0
    return 1.0 - 0.5 * u * u


def _ease_in_cubic(t):
    return t * t * t


def _ease_out_cubic(t):
    u = 1.0 - t
    return 1.0 - u * u * u


def _ease_in_out_cubic(t):
    if t < 0.5:
        return 4.0 * t * t * t
    u = -2.0 * t + 2.0
    return 1.0 - 0.5 * u * u * u


_REGISTRY = {
    "linear": _linear,
    "ease_in": _ease_in,
    "ease_out": _ease_out,
    "ease_in_out": _ease_in_out,
    "ease_in_cubic": _ease_in_cubic,
    "ease_out_cubic": _ease_out_cubic,
    "ease_in_out_cubic": _ease_in_out_cubic,
}

#: 可枚举的内置缓动名称（全部为单调不减函数）。
NAMES = tuple(sorted(_REGISTRY))

_SPRING_NAME = "spring"


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------

def _finite(value, label):
    """转为 float 并要求有限；失败抛 ParameterError。"""
    if type(value) is float:
        v = value
    else:
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ParameterError("%s 必须是实数，得到 %r" % (label, value)) from None
    if not math.isfinite(v):
        raise ParameterError("%s 必须有限，得到 %r" % (label, value))
    return v


def _check_t(t):
    """t 必须在 [0, 1] 且有限（nan/inf 一律拒绝）。"""
    v = _finite(t, "t")
    if not 0.0 <= v <= 1.0:
        raise ParameterError("t 必须在 [0, 1] 内，得到 %r" % (t,))
    return v


# ---------------------------------------------------------------------------
# cubic-bezier 求值（牛顿迭代 + 二分回退）
# ---------------------------------------------------------------------------

def _solve_bezier_x(ax, bx, cx, x):
    """给定 x，求 t 使 x(t)=x。x(t)=((ax*t+bx)*t+cx)*t。

    牛顿迭代失败（导数过小或越界）时回退定次二分，绝不死循环。
    """
    t = x
    for _ in range(8):
        err = ((ax * t + bx) * t + cx) * t - x
        if -1e-12 < err < 1e-12:
            return t
        deriv = (3.0 * ax * t + 2.0 * bx) * t + cx
        if -1e-12 < deriv < 1e-12:
            break  # 退化导数（如 p1x=p2x=0 在 t=0 处），转二分
        t -= err / deriv
        if t < 0.0 or t > 1.0:
            break
    # 二分回退：固定次数，保证终止
    lo, hi = 0.0, 1.0
    t = x
    for _ in range(64):
        est = ((ax * t + bx) * t + cx) * t
        err = est - x
        if -1e-12 < err < 1e-12:
            break
        if err < 0.0:
            lo = t
        else:
            hi = t
        t = 0.5 * (lo + hi)
    return t


def _make_cubic_bezier(p1x, p1y, p2x, p2y):
    ax = 3.0 * p1x - 3.0 * p2x + 1.0
    bx = 3.0 * p2x - 6.0 * p1x
    cx = 3.0 * p1x
    ay = 3.0 * p1y - 3.0 * p2y + 1.0
    by = 3.0 * p2y - 6.0 * p1y
    cy = 3.0 * p1y

    def bezier(t):
        t = _check_t(t)
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0
        ts = _solve_bezier_x(ax, bx, cx, t)
        return ((ay * ts + by) * ts + cy) * ts

    bezier._easing_monotonic = True
    return bezier


# ---------------------------------------------------------------------------
# 弹簧缓动（二阶系统阶跃响应闭式解）
# ---------------------------------------------------------------------------

def _spring_eval(t, zeta, omega):
    """单位阶跃响应：f(0)=0，稳态 1，欠阻尼时超调 e^(-pi*zeta/sqrt(1-zeta^2))。"""
    if t <= 0.0:
        return 0.0
    if abs(zeta - 1.0) < 1e-9:
        # 临界阻尼：1 - e^(-w t) (1 + w t)
        w = omega * t
        return 1.0 - math.exp(-w) * (1.0 + w)
    if zeta < 1.0:
        # 欠阻尼
        s = math.sqrt(1.0 - zeta * zeta)
        wd = omega * s
        return 1.0 - math.exp(-zeta * omega * t) * (
            math.cos(wd * t) + (zeta / s) * math.sin(wd * t)
        )
    # 过阻尼
    r = omega * math.sqrt(zeta * zeta - 1.0)
    zw = zeta * omega
    s1 = r - zw
    s2 = -r - zw
    return 1.0 - (s2 * math.exp(s1 * t) - s1 * math.exp(s2 * t)) / (s2 - s1)


def _spring_default(t):
    return _spring_eval(t, 0.5, 5.0)


# ---------------------------------------------------------------------------
# 名称解析
# ---------------------------------------------------------------------------

def _resolve(name):
    """返回 (fn, monotonic)。未知名称抛 UnknownEasingError。"""
    if isinstance(name, str):
        fn = _REGISTRY.get(name)
        if fn is not None:
            return fn, True
        if name == _SPRING_NAME:
            return _spring_default, False
        raise UnknownEasingError("未知缓动名称: %r" % (name,))
    if callable(name):
        return name, bool(getattr(name, "_easing_monotonic", False))
    raise UnknownEasingError("未知缓动名称: %r" % (name,))


# ---------------------------------------------------------------------------
# 采样表缓存（命中与逐次求值逐位一致：缓存的就是同一求值器产出的 float）
# ---------------------------------------------------------------------------

_TABLE_LOCK = threading.Lock()
_TABLE_CACHE = {}


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

class Easing:
    """无状态缓动求值器；所有方法并发安全。"""

    def names(self):
        """返回可枚举的内置缓动名称（均单调不减，f(0)=0，f(1)=1）。"""
        return NAMES

    def apply(self, name, t):
        """应用缓动。name 为注册名、"spring" 或 cubic_bezier 返回的函数。"""
        fn, _ = _resolve(name)
        return fn(_check_t(t))

    def inverse(self, name, y):
        """反解：求 t 使 apply(name, t) == y。精度远优于 1e-9。

        非单调目标（如 "spring"）抛 NonMonotonicError；y 越界抛
        ParameterError；未知名称抛 UnknownEasingError。
        """
        fn, monotonic = _resolve(name)
        if not monotonic:
            raise NonMonotonicError("缓动 %r 非单调，无法反解" % (name,))
        y = _check_t(y)
        lo, hi = 0.0, 1.0
        for _ in range(48):  # 2^-48 ≈ 3.6e-15，远优于 1e-9
            mid = 0.5 * (lo + hi)
            if fn(mid) < y:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def cubic_bezier(p1x, p1y, p2x, p2y):
        """构造自定义 cubic-bezier 缓动函数。

        控制点必须有限且落在 [0,1]（保证单调不减）；越界抛
        ParameterError。退化控制点（p1x=p2x、全 0、全 1）安全。
        """
        p1x = _finite(p1x, "p1x")
        p1y = _finite(p1y, "p1y")
        p2x = _finite(p2x, "p2x")
        p2y = _finite(p2y, "p2y")
        for label, v in (("p1x", p1x), ("p1y", p1y), ("p2x", p2x), ("p2y", p2y)):
            if not 0.0 <= v <= 1.0:
                raise ParameterError("%s 必须在 [0, 1] 内，得到 %r" % (label, v))
        return _make_cubic_bezier(p1x, p1y, p2x, p2y)

    @staticmethod
    def spring(t, damping_ratio=0.5, frequency=5.0):
        """弹簧缓动（二阶阶跃响应闭式解）。

        damping_ratio: 阻尼比 zeta（>0）；frequency: 自然频率 omega（>0）。
        欠阻尼超调量恰为理论值 e^(-pi*zeta/sqrt(1-zeta^2))。
        """
        t = _check_t(t)
        zeta = _finite(damping_ratio, "damping_ratio")
        omega = _finite(frequency, "frequency")
        if zeta <= 0.0:
            raise ParameterError("阻尼比必须为正，得到 %r" % (damping_ratio,))
        if omega <= 0.0:
            raise ParameterError("频率必须为正，得到 %r" % (frequency,))
        return _spring_eval(t, zeta, omega)

    def sample_table(self, name, n):
        """生成 n 点采样表（含端点）。缓存命中与逐次求值逐位一致。"""
        fn, _ = _resolve(name)
        if isinstance(n, bool):
            raise ParameterError("采样数必须是 >= 2 的整数")
        try:
            n = operator.index(n)
        except TypeError:
            raise ParameterError("采样数必须是整数，得到 %r" % (n,)) from None
        if n < 2:
            raise ParameterError("采样数必须 >= 2，得到 %r" % (n,))
        key = (name, n) if isinstance(name, str) else None
        if key is not None:
            with _TABLE_LOCK:
                hit = _TABLE_CACHE.get(key)
            if hit is not None:
                return hit
        last = n - 1
        table = tuple(fn(i / last) for i in range(n))
        if key is not None:
            with _TABLE_LOCK:
                return _TABLE_CACHE.setdefault(key, table)
        return table
