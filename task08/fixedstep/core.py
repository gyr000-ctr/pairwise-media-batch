"""fixedstep 核心 API（初始未实现）。"""


class FixedStep:
    def __init__(self, step, max_steps=5):
        """构造"""
        raise NotImplementedError

    def advance(self, dt):
        """推进并返回步数"""
        raise NotImplementedError
