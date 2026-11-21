"""gradient 核心 API（初始未实现）。"""


class Gradient:
    def __init__(self, stops):
        """按停靠点构造"""
        raise NotImplementedError

    def sample(self, t):
        """采样"""
        raise NotImplementedError

    def add_stop(self, t, color):
        """追加停靠点"""
        raise NotImplementedError
