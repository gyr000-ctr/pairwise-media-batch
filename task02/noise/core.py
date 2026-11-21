"""noise 核心 API（初始未实现）。"""


class ValueNoise:
    def __init__(self, seed):
        """按种子构造"""
        raise NotImplementedError

    def sample(self, x, y):
        """采样"""
        raise NotImplementedError
