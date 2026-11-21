"""adsr 核心 API（初始未实现）。"""


class Adsr:
    def __init__(self, rate, a, d, s, r):
        """构造"""
        raise NotImplementedError

    def render(self, n):
        """渲染"""
        raise NotImplementedError

    def value_at(self, t):
        """取值"""
        raise NotImplementedError
