"""ADSR 包络生成器。

特性：
- attack 升到峰值 1.0，decay 降到 sustain 电平并保持，release 降到 0；
- 支持线性（linear）与指数（exponential）两种曲线；
- 支持重触发：release 期间 note_on 从当前电平继续 attack，不跳变；
- 包络形状只取决于秒级时间参数，与采样率无关（改采样率总时长不变）；
- 非法参数抛出可区分的错误（AdsrParamError，含 param 属性）；
- render/note_on/note_off 加锁，并发渲染安全；value_at 为纯函数。

时间语义（rate-based）：a/d/r 均为“满量程时间”。
attack 以 1/a 的速率上升，decay 以 1/d 的速率下降，release 以 1/r 的
速率下降；因此 decay 实际持续 (1-s)*d 秒，release 实际持续 电平*r 秒。
"""

from __future__ import annotations

import math
import operator
import threading
from numbers import Real

__all__ = ["Adsr", "AdsrError", "AdsrParamError"]

_LINEAR = "linear"
_EXP = "exponential"
_CURVE_ALIASES = {
    "linear": _LINEAR,
    "lin": _LINEAR,
    "exponential": _EXP,
    "exp": _EXP,
}


class AdsrError(Exception):
    """adsr 相关错误的基类。"""


class AdsrParamError(ValueError, AdsrError):
    """非法参数错误。``param`` 属性指出出错的参数名，便于区分。"""

    def __init__(self, param, message):
        self.param = param
        super().__init__("%s: %s" % (param, message))


def _finite(param, value):
    """校验为有限实数，返回 float。类型错误与非有限值可区分。"""
    if not isinstance(value, Real):
        raise TypeError(
            "%s must be a real number, got %s" % (param, type(value).__name__)
        )
    v = float(value)
    if not math.isfinite(v):
        raise AdsrParamError(param, "must be finite, got %r" % (v,))
    return v


def _clamp01(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


class Adsr:
    """ADSR 包络。

    参数：
        rate: 采样率（>0，有限）。
        a: attack 时间（秒，>=0），从 0 升到峰值 1.0 的满量程时间。
        d: decay 时间（秒，>=0），满量程下降时间。
        s: sustain 电平（0..1）。
        r: release 时间（秒，>=0），满量程下降时间。
        curve: "linear"（默认）或 "exponential"/"exp"。
        hold: sustain 保持秒数，默认 None 表示与 r 相同。

    构造即隐式 note_on（t=0，电平 0），并在 t = a+d+hold 处自动 note_off。
    """

    def __init__(self, rate, a, d, s, r, curve="linear", hold=None):
        rate = _finite("rate", rate)
        if rate <= 0.0:
            raise AdsrParamError("rate", "must be > 0, got %r" % (rate,))
        a = _finite("attack", a)
        if a < 0.0:
            raise AdsrParamError("attack", "must be >= 0, got %r" % (a,))
        d = _finite("decay", d)
        if d < 0.0:
            raise AdsrParamError("decay", "must be >= 0, got %r" % (d,))
        s = _finite("sustain", s)
        if not 0.0 <= s <= 1.0:
            raise AdsrParamError("sustain", "must be in [0, 1], got %r" % (s,))
        r = _finite("release", r)
        if r < 0.0:
            raise AdsrParamError("release", "must be >= 0, got %r" % (r,))
        if hold is None:
            hold = r
        hold = _finite("hold", hold)
        if hold < 0.0:
            raise AdsrParamError("hold", "must be >= 0, got %r" % (hold,))
        if not isinstance(curve, str):
            raise TypeError(
                "curve must be a string, got %s" % type(curve).__name__
            )
        key = curve.strip().lower()
        if key not in _CURVE_ALIASES:
            raise AdsrParamError(
                "curve", "must be 'linear' or 'exponential', got %r" % (curve,)
            )

        self._rate = rate
        self._dt = 1.0 / rate
        self._a = a
        self._d = d
        self._s = s
        self._r = r
        self._hold = hold
        self._curve = _CURVE_ALIASES[key]
        self._gate = a + d + hold  # 自动 note_off 时刻（相对 note_on）

        self._lock = threading.Lock()
        # 播放状态（相对当前 note_on；位置用整数采样计数，保证分块渲染
        # 与一次性渲染逐位一致）
        self._pos = 0            # 已渲染的采样数
        self._note_level = 0.0   # 本次 note_on 时的电平
        self._off_time = self._gate
        self._off_level = s

    # ---- 只读属性 ----
    @property
    def rate(self):
        return self._rate

    @property
    def attack(self):
        return self._a

    @property
    def decay(self):
        return self._d

    @property
    def sustain(self):
        return self._s

    @property
    def release(self):
        return self._r

    @property
    def curve(self):
        return self._curve

    @property
    def hold(self):
        return self._hold

    @property
    def gate(self):
        """自动 note_off 时刻（秒，相对 note_on）。"""
        return self._gate

    @property
    def duration(self):
        """从 note_on 到包络降到 0 的总时长（秒），与采样率无关。"""
        return self._gate + self._s * self._r

    # ---- 触发控制 ----
    def note_on(self):
        """触发/重触发：从当前电平继续 attack，不跳变。"""
        with self._lock:
            level = self._level_at(self._pos * self._dt)
            self._pos = 0
            self._note_level = level
            self._off_time = self._gate
            self._off_level = self._s

    def note_off(self):
        """松开键：从当前电平进入 release。"""
        with self._lock:
            self._off_level = self._level_at(self._pos * self._dt)
            self._off_time = self._pos * self._dt

    # ---- 取值 ----
    def value_at(self, t):
        """名义包络（note_on 于 0、电平 0 起）在 t 秒处的值，纯函数。"""
        t = _finite("t", t)
        if t < 0.0:
            return 0.0
        return self._envelope(t, 0.0, self._gate, self._s)

    def render(self, n):
        """渲染 n 个采样并推进内部时间，返回 float 列表。线程安全。"""
        n = self._check_n(n)
        if n == 0:
            return []
        with self._lock:
            base = self._pos
            self._pos = base + n
            return self._render_span(base, n)

    # ---- 内部实现 ----
    @staticmethod
    def _check_n(n):
        try:
            n = operator.index(n)
        except TypeError:
            raise TypeError(
                "n must be an integer, got %s" % type(n).__name__
            ) from None
        if n < 0:
            raise AdsrParamError("n", "must be >= 0, got %r" % (n,))
        return n

    def _level_at(self, tau):
        return self._envelope(tau, self._note_level, self._off_time,
                              self._off_level)

    def _envelope(self, tau, nl, off, off_level):
        """tau 秒处的包络值（note_on 电平 nl，note_off 于 off、电平 off_level）。"""
        a, d, s, r = self._a, self._d, self._s, self._r
        t_peak = (1.0 - nl) * a
        if tau <= 0.0:
            return _clamp01(nl)
        if tau < t_peak:
            if self._curve == _LINEAR:
                return _clamp01(nl + tau / a)
            return _clamp01(1.0 - (1.0 - nl) * math.exp(-tau / (t_peak / 5.0)))
        t_dec = t_peak + (1.0 - s) * d
        if tau < t_dec:
            if self._curve == _LINEAR:
                return _clamp01(1.0 - (tau - t_peak) / d)
            return _clamp01(
                s + (1.0 - s) * math.exp(-(tau - t_peak) / ((t_dec - t_peak) / 5.0))
            )
        if tau <= off:
            return s
        t_zero = off + off_level * r
        if tau >= t_zero:
            return 0.0
        if self._curve == _LINEAR:
            return _clamp01(off_level - (tau - off) / r)
        return _clamp01(
            off_level * math.exp(-(tau - off) / ((off_level * r) / 5.0))
        )

    def _render_span(self, base, n):
        """生成第 base+1 .. base+n 个采样（时刻 p*dt），按段批量生成。"""
        dt = self._dt
        nl = self._note_level
        a, d, s, r = self._a, self._d, self._s, self._r
        off = self._off_time
        L0 = self._off_level
        t_peak = (1.0 - nl) * a
        t_dec = t_peak + (1.0 - s) * d
        t_zero = off + L0 * r

        # 段列表 (start, end, kind)：release 前的段在 off 处截断，
        # 以兼容用户在 attack/decay 期间手动 note_off 的情况。
        segs = []
        for st, en, kind in ((0.0, t_peak, "attack"),
                             (t_peak, t_dec, "decay"),
                             (t_dec, math.inf, "sustain")):
            en = min(en, off)
            if en > st:
                segs.append((st, en, kind))
        if t_zero > off:
            segs.append((off, t_zero, "release"))
        segs.append((max(off, t_zero), math.inf, "idle"))

        out = []
        j = 1  # 第 j 个采样对应绝对位置 base+j、时刻 (base+j)*dt
        for st, en, kind in segs:
            if j > n:
                break
            if en == math.inf:
                k = n
            else:
                # 满足 (base+j)*dt < en 的最大 j
                k = math.ceil(en / dt - base) - 1
                if k > n:
                    k = n
            cnt = k - j + 1
            if cnt <= 0:
                continue
            self._emit(kind, out, dt, base + j, cnt, nl, t_peak, t_dec, off, L0)
            j += cnt
        return out

    def _emit(self, kind, out, dt, p0, cnt, nl, t_peak, t_dec, off, L0):
        """生成位置 p0 .. p0+cnt-1（时刻 p*dt）的 cnt 个采样值。"""
        s, r = self._s, self._r
        if kind == "sustain":
            out.extend([s] * cnt)
        elif kind == "idle":
            out.extend([0.0] * cnt)
        elif self._curve == _LINEAR:
            if kind == "attack":
                self._emit_linear(out, nl, 1.0 / self._a, dt, p0, cnt)
            elif kind == "decay":
                d = self._d
                self._emit_linear(out, 1.0 + t_peak / d, -1.0 / d, dt, p0, cnt)
            else:  # release
                self._emit_linear(out, L0 + off / r, -1.0 / r, dt, p0, cnt)
        else:
            exp = math.exp
            if kind == "attack":
                tau = t_peak / 5.0
                c = 1.0 - nl
                out.extend([_clamp01(1.0 - c * exp(-((p0 + i) * dt) / tau))
                            for i in range(cnt)])
            elif kind == "decay":
                tau = (t_dec - t_peak) / 5.0
                c = 1.0 - s
                out.extend([_clamp01(s + c * exp(-((p0 + i) * dt - t_peak) / tau))
                            for i in range(cnt)])
            else:  # release
                tau = (L0 * r) / 5.0
                out.extend([_clamp01(L0 * exp(-((p0 + i) * dt - off) / tau))
                            for i in range(cnt)])

    @staticmethod
    def _emit_linear(out, alpha, beta, dt, p0, cnt):
        out.extend([_clamp01(alpha + beta * ((p0 + i) * dt))
                    for i in range(cnt)])
