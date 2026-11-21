"""peakmeter 核心 API。

纯标准库实现的音频电平表：

- ``peak``：绝对采样峰值（可重置，空输入为 0.0）
- ``rms``：均方根电平（空输入为 0.0）
- ``true_peak``：4 倍过采样近似峰值。采用三样本拉格朗日二次插值
  （等效 4 倍过采样的多项式核），允许过冲；始终不小于采样峰值，
  且插值核的 L1 范数为 1.25，因此与采样峰值的差不超过
  ``20*log10(1.25) ≈ 1.94 dB``（满足 3 dB 上限）。
- ``hold``：保持窗口（按样本计）。窗口内保留最大峰值；``None``
  表示永久保持（累计峰值）。
- 所有公开方法均加锁，多线程并发推入安全；单次 ``push`` 原子，
  非法输入不会污染既有状态。
"""

import math
import threading
from collections import deque

__all__ = [
    "PeakMeter",
    "PeakMeterError",
    "NonFiniteSampleError",
    "InvalidHoldError",
]

# 4 倍过采样三个相位的二次插值权重：
# y(f) = w_a*a + w_b*b + w_c*c，a/b/c 为连续三个样本，
# f 取 0.25 / 0.5 / 0.75（b 与 c 之间的过采样点）。
# 三个相位的权重 L1 范数最大为 1.25（f=0.5），
# 故过冲相对局部样本幅度的上界为 1.25 倍，即约 +1.94 dB。
_PHASES = (
    (-0.09375, 0.9375, 0.15625),   # f = 0.25
    (-0.125, 0.75, 0.375),         # f = 0.5
    (-0.09375, 0.4375, 0.65625),   # f = 0.75
)
_KERNEL_L1 = 1.25
_TP_GAIN_CAP = 1.25  # true_peak <= 1.25 * 采样峰值（约 1.94 dB）
_SMALL_HOLD = 64     # 小窗口阈值：读时取 max 的成本可接受


class PeakMeterError(ValueError):
    """电平表错误基类（均为 ValueError 的子类，兼容既有断言）。"""


class NonFiniteSampleError(PeakMeterError):
    """输入包含 NaN 或 ±Inf 等非有限样本。"""


class InvalidHoldError(PeakMeterError):
    """hold 参数非法（非 None、非非负整数，或为 bool）。"""


def _validate_hold(hold):
    if hold is None:
        return None
    # bool 是 int 的子类，但语义上不是合法的保持长度
    if isinstance(hold, bool) or not isinstance(hold, int) or hold < 0:
        raise InvalidHoldError(
            "hold 必须是 None 或非负整数，实际为 %r" % (hold,)
        )
    return hold


class PeakMeter:
    def __init__(self, hold=None):
        """构造电平表。

        ``hold`` 为保持窗口长度（样本数）：``None`` 表示永久保持；
        非负整数表示只保留最近 ``hold`` 个样本内的最大值（``0``
        表示不保持，读数在两次推入之间不保留旧样本）。
        """
        self._hold = _validate_hold(hold)
        self._lock = threading.RLock()
        self._reset_locked()

    def _reset_locked(self):
        # 累计口径（hold=None）专用
        self._peak = 0.0       # 累计绝对采样峰值
        self._tp = 0.0         # 累计 true-peak
        self._sum_sq = 0.0     # 累计平方和
        self._count = 0        # 累计样本数
        # 过采样需要跨批次的相邻样本上下文（两种口径都需要）
        self._prev2 = None
        self._prev1 = None
        # 有限保持窗口专用，两种布局：
        # 小窗口（hold <= _SMALL_HOLD）：定长裸值 deque，读时取 max，
        # 省去逐样本单调队列维护与 (值, 序号) 元组分配；
        # 大窗口：并行 list 单调队列 + 预分配平方环，避免热路径上的
        # 元组分配与 deque 两端操作开销，整体摊销 O(1)。
        small = self._small_window()
        if small:
            # hold=0 时 maxlen=0 的 deque 不接受元素，语义即空窗口
            self._win_peak = deque(maxlen=self._hold)
            self._win_tp = deque(maxlen=self._hold)
            self._sq_ring = deque(maxlen=self._hold)
        else:
            self._win_peak = None
            self._win_tp = None
            self._sq_ring = None
        # 大窗口布局字段（小窗口不使用）
        self._peak_vals = []      # 单调递减的峰值
        self._peak_idx = []       # 对应全局序号
        self._peak_head = 0
        self._tp_vals = []        # 单调递减的 true-peak
        self._tp_idx = []
        self._tp_head = 0
        if small or self._hold is None:
            self._sq_values = None
        else:
            self._sq_values = [0.0] * self._hold
        self._sq_pos = 0
        self._win_sum_sq = 0.0    # 窗口内平方和（摊销 O(1) 维护）
        self._expire_count = 0    # 距上次重算以来淘汰的样本数（大窗口）
        self._seq = 0             # 已推入样本总数（下一序号）

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def push(self, samples):
        """推入一批样本（任何可迭代对象），原子更新全部电平。"""
        with self._lock:
            # 先物化为列表：保证只遍历一次，且校验失败时不改动状态
            data = list(samples)
            for v in data:
                if not math.isfinite(v):
                    raise NonFiniteSampleError(
                        "样本必须是有限实数，遇到 %r" % (v,)
                    )
            if not data:
                return
            self._update_locked(data)

    def peak(self):
        """当前峰值（保持窗口内的绝对采样最大值），无数据时为 0.0。"""
        with self._lock:
            if self._hold is None:
                return float(self._peak)
            if self._small_window():
                return float(max(self._win_peak)) if self._win_peak else 0.0
            if self._peak_head >= len(self._peak_vals):
                return 0.0
            return float(self._peak_vals[self._peak_head])

    def true_peak(self):
        """4 倍过采样近似峰值，无数据时为 0.0。"""
        with self._lock:
            if self._hold is None:
                return float(self._tp)
            if self._small_window():
                return float(max(self._win_tp)) if self._win_tp else 0.0
            if self._tp_head >= len(self._tp_vals):
                return 0.0
            return float(self._tp_vals[self._tp_head])

    def rms(self):
        """当前均方根电平（保持窗口内），无数据时为 0.0。"""
        with self._lock:
            if self._hold is None:
                if self._count == 0:
                    return 0.0
                return math.sqrt(self._sum_sq / self._count)
            if self._small_window():
                if not self._sq_ring:
                    return 0.0
                total = self._win_sum_sq
                if total <= 1e-12:
                    total = 0.0
                return math.sqrt(total / len(self._sq_ring))
            n = min(self._seq, self._hold)
            if n == 0:
                return 0.0
            total = self._win_sum_sq
            if total <= 1e-12:
                total = 0.0
            return math.sqrt(total / n)

    def reset(self):
        """清空全部状态，重置后所有电平读数为 0.0。"""
        with self._lock:
            self._reset_locked()

    # ------------------------------------------------------------------
    # 内部实现（调用时已持锁）
    # ------------------------------------------------------------------
    def _small_window(self):
        return self._hold is not None and self._hold <= _SMALL_HOLD

    def _expire_locked(self):
        """淘汰保持窗口外的旧样本（仅有限保持路径使用）。

        小窗口下定长 deque 在入队时即完成淘汰；大窗口下推进两条
        单调队列的 head 指针，并在死前缀超过存活段时整体压缩，
        内存有界且摊销 O(1)。
        """
        if self._small_window():
            return
        cutoff = self._seq - self._hold
        pv, pi, ph = self._peak_vals, self._peak_idx, self._peak_head
        while ph < len(pi) and pi[ph] < cutoff:
            ph += 1
        self._peak_head = ph
        tv, ti, th = self._tp_vals, self._tp_idx, self._tp_head
        while th < len(ti) and ti[th] < cutoff:
            th += 1
        self._tp_head = th
        # 死前缀长于存活段时压缩，保证 list 长度不超过 2*hold
        if ph > len(pv) - ph:
            del pv[:ph]
            del pi[:ph]
            self._peak_head = 0
        if th > len(tv) - th:
            del tv[:th]
            del ti[:th]
            self._tp_head = 0
        # 摊销扣减会累积浮点漂移：每淘汰 4096 个样本就以平方环为
        # 唯一事实来源重算一次，重算成本摊销可忽略，整体仍为 O(n)。
        if self._expire_count >= 4096:
            n = min(self._seq, self._hold)
            pos = self._sq_pos
            ring = self._sq_values
            self._win_sum_sq = sum(ring) if n == self._hold else sum(ring[:n])
            self._expire_count = 0

    def _update_locked(self, data):
        """单遍融合更新峰值、RMS 平方和与 4 倍过采样 true-peak。

        true-peak 按**样本序号**归属：样本 ``c`` 与前一样本 ``b``
        之间的三个插值点归入 ``c`` 的序号，保持窗口因此可以按
        样本精确淘汰，不存在跨批次归属偏差。窗口维护就地内联在
        热循环中（避免每样本一次方法调用）。
        """
        hold = self._hold
        windowed = hold is not None
        small = self._small_window()
        prev2 = self._prev2
        prev1 = self._prev1
        idx = self._seq          # 本批首个样本占用的全局序号
        n_pushed = 0
        first_batch = prev1 is None
        # 先快照累计量，循环中只写局部变量，结束时一次性落账
        cum_peak = self._peak
        cum_tp = self._tp
        cum_sq = self._sum_sq
        batch_peak = 0.0
        batch_tp = 0.0

        # 窗口结构局部化（整批只取一次属性）
        if windowed:
            win_sum_sq = self._win_sum_sq
            if small:
                win_peak = self._win_peak
                win_tp = self._win_tp
                sq_ring = self._sq_ring
                shold = hold
            else:
                pv = self._peak_vals
                pi = self._peak_idx
                ph = self._peak_head
                tv = self._tp_vals
                ti = self._tp_idx
                th = self._tp_head
                ring = self._sq_values
                pos = self._sq_pos
                pos_limit = hold - 1
                expire_count = self._expire_count
                pv_len = len(pv)     # 队列长度本地化，省掉热路径 len()
                tv_len = len(tv)

        # 首批首样本：三元组上下文不完整，只能取采样值本身
        if first_batch:
            first = data[0]
            af = abs(first)
            sq0 = first * first
            cum_peak = af
            cum_tp = af
            cum_sq = sq0
            if windowed:
                if small:
                    if shold and len(sq_ring) == shold:
                        win_sum_sq -= sq_ring[0]
                    win_peak.append(af)
                    win_tp.append(af)
                    sq_ring.append(sq0)
                    win_sum_sq += sq0
                else:
                    if idx >= hold:
                        win_sum_sq -= ring[pos]
                        expire_count += 1
                    ring[pos] = sq0
                    pos += 1
                    if pos > pos_limit:
                        pos = 0
                    win_sum_sq += sq0
                    pv.append(af); pi.append(idx); pv_len += 1
                    tv.append(af); ti.append(idx); tv_len += 1
            prev2 = first
            prev1 = first
            idx += 1
            n_pushed += 1
            data = data[1:]

        for c in data:
            ac = abs(c)
            square = c * c
            if ac > batch_peak:
                batch_peak = ac
            cum_sq += square
            b = prev1
            a = prev2
            # 样本 c 自身 + 它与 b 之间三个插值点，统一归入序号 idx
            local_tp = ac
            # 线性界：插值点不超过 KERNEL_L1 * 相邻样本幅度最大值；
            # 上界都超不过当前批次 tp 时，跳过昂贵的插值运算。
            ab = abs(b)
            bound = ab if ab > ac else ac
            if _KERNEL_L1 * bound > batch_tp:
                for wa, wb, wc in _PHASES:
                    ay = abs(wa * a + wb * b + wc * c)
                    if ay > local_tp:
                        local_tp = ay
            # 逐样本收口：保证任意保持窗口内 tp <= 1.25 * 该样本幅度，
            # 从而窗口聚合后增益仍 <= 1.25（约 1.94 dB，< 3 dB）。
            local_cap = _TP_GAIN_CAP * ac
            if local_tp > local_cap:
                local_tp = local_cap
            if local_tp > batch_tp:
                batch_tp = local_tp
            if windowed:
                if small:
                    if shold and len(sq_ring) == shold:
                        win_sum_sq -= sq_ring[0]
                    win_peak.append(ac)
                    win_tp.append(local_tp)
                    sq_ring.append(square)
                    win_sum_sq += square
                else:
                    if idx >= hold:
                        win_sum_sq -= ring[pos]
                        expire_count += 1
                    ring[pos] = square
                    pos += 1
                    if pos > pos_limit:
                        pos = 0
                    win_sum_sq += square
                    while pv_len > ph and pv[-1] <= ac:
                        pv.pop()
                        pi.pop()
                        pv_len -= 1
                    pv.append(ac)
                    pi.append(idx)
                    pv_len += 1
                    while tv_len > th and tv[-1] <= local_tp:
                        tv.pop()
                        ti.pop()
                        tv_len -= 1
                    tv.append(local_tp)
                    ti.append(idx)
                    tv_len += 1
            prev2, prev1 = b, c
            idx += 1
            n_pushed += 1

        self._prev2 = prev2
        self._prev1 = prev1
        self._seq = idx

        if not windowed:
            if batch_peak > cum_peak:
                cum_peak = batch_peak
            if batch_tp > cum_tp:
                cum_tp = batch_tp
            # 全局数值安全上界（双保险）
            cap = _TP_GAIN_CAP * cum_peak
            if cum_tp > cap:
                cum_tp = cap
            self._peak = cum_peak
            self._tp = cum_tp
            self._sum_sq = cum_sq
            self._count += n_pushed
            return

        self._win_sum_sq = win_sum_sq
        if small:
            return  # 定长 deque 已在入队时完成淘汰
        self._sq_pos = pos
        self._expire_count = expire_count
        self._expire_locked()
