"""hdr 核心 API（初始未实现）。"""


class ToneMap:
    def tonemap(self, value, exposure=1.0):
        """单值映射"""
        raise NotImplementedError

    def apply(self, image):
        """整图映射"""
        raise NotImplementedError
