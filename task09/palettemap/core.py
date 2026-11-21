"""palettemap 核心 API（初始未实现）。"""


class PaletteMap:
    def __init__(self, palette):
        """按调色板构造"""
        raise NotImplementedError

    def index_of(self, color):
        """最近索引"""
        raise NotImplementedError

    def map_image(self, image):
        """整图映射"""
        raise NotImplementedError
