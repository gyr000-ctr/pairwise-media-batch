"""fixedstep 核心 API：固定步长累加器。

特性：
- 按步长切分，不足一步返回 0，余量保留在累加器中；
- 单帧最多追赶 max_steps 步（spiral-of-death 保护），超出的时间不丢弃，
  留在累加器中后续帧继续消费；
- 支持时间缩放（set_time_scale）与暂停（pause/resume）；
- 负时间、非有限 dt、非法步长、非法缩放分别抛出可区分的错误类型
  （均为 ValueError 子类）；
- 并发推进同一实例安全（内部持锁）。
"""

import threading

__all__ = [
    "FixedStep",
    "FixedStepError",
    "NegativeTimeError",
    "NonFiniteTimeError",
    "InvalidStepError",
    "InvalidMaxStepsError",
    "InvalidScaleError",
]

_INF = float("inf")


class FixedStepError(ValueError):
    """固定步长累加器错误基类（所有可区分错误的公共父类）。"""


class NegativeTimeError(FixedStepError):
    """推进时间 dt 为负。"""


class NonFiniteTimeError(FixedStepError):
    """推进时间 dt 为 NaN 或无穷。"""


class InvalidStepError(FixedStepError):
    """步长非法（非有限或 <= 0）。"""


class InvalidMaxStepsError(FixedStepError):
    """最大步数非法（非正整数）。"""


class InvalidScaleError(FixedStepError):
    """时间缩放非法（非有限或 < 0）。"""


class FixedStep:
    __slots__ = ("_step", "_max_steps", "_acc", "_scale", "_paused", "_lock")

    def __init__(self, step, max_steps=5):
        """构造：step 为有限正数步长，max_steps 为正整数单帧步数上限。"""
        try:
            step = float(step)
        except (TypeError, ValueError):
            raise InvalidStepError("step 必须为有限正数, got %r" % (step,)) from None
        if not 0.0 < step < _INF:  # 同时排除 NaN、inf、0、负数
            raise InvalidStepError("step 必须为有限正数, got %r" % (step,))
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise InvalidMaxStepsError("max_steps 必须为正整数, got %r" % (max_steps,))
        if max_steps < 1:
            raise InvalidMaxStepsError("max_steps 必须为正整数, got %r" % (max_steps,))
        self._step = step
        self._max_steps = max_steps
        self._acc = 0.0
        self._scale = 1.0
        self._paused = False
        self._lock = threading.Lock()

    def advance(self, dt):
        """推进 dt 秒（先经时间缩放），返回本次消费的整步数（int）。

        不足一步返回 0；单帧最多返回 max_steps 步，剩余时间保留不丢弃。
        暂停时直接返回 0 且不累积时间。
        """
        if dt < 0.0:
            raise NegativeTimeError("dt 不能为负, got %r" % (dt,))
        if not dt <= _INF:  # NaN
            raise NonFiniteTimeError("dt 必须为有限值, got %r" % (dt,))
        if dt == _INF:
            raise NonFiniteTimeError("dt 必须为有限值, got %r" % (dt,))
        with self._lock:
            if self._paused:
                return 0
            step = self._step
            acc = self._acc + dt * self._scale
            n = acc / step
            if n >= self._max_steps:  # 浮点比较，避免构造巨整型
                n = self._max_steps
            else:
                n = int(n)
            # 一次性乘减而非逐步减，累积误差保持在 ~1e-16/次 量级
            self._acc = acc - n * step
            return n

    def set_time_scale(self, scale):
        """设置时间缩放系数，须为有限非负数（0 等价于时间静止）。"""
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            raise InvalidScaleError("scale 必须为有限非负数, got %r" % (scale,)) from None
        if not 0.0 <= scale < _INF:  # 同时排除 NaN、inf、负数
            raise InvalidScaleError("scale 必须为有限非负数, got %r" % (scale,))
        with self._lock:
            self._scale = scale

    def pause(self):
        """暂停：暂停期间 advance 返回 0 且不累积时间。"""
        with self._lock:
            self._paused = True

    def resume(self):
        """恢复推进。"""
        with self._lock:
            self._paused = False

    @property
    def paused(self):
        return self._paused

    @property
    def time_scale(self):
        return self._scale

    @property
    def step(self):
        return self._step

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def accumulator(self):
        """当前累加器中尚未消费的剩余时间。"""
        return self._acc
