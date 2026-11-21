"""peakmeter 核心 API（初始未实现）。"""


class PeakMeter:
    def __init__(self):
        """构造"""
        raise NotImplementedError

    def push(self, samples):
        """推入样本"""
        raise NotImplementedError

    def peak(self):
        """当前峰值"""
        raise NotImplementedError

    def reset(self):
        """重置"""
        raise NotImplementedError
