"""tilemap 核心 API（初始未实现）。"""


class TileMap:
    def __init__(self, size):
        """按瓦片尺寸构造"""
        raise NotImplementedError

    def to_tile(self, x, y):
        """世界转瓦片"""
        raise NotImplementedError

    def to_world(self, tx, ty):
        """瓦片转世界"""
        raise NotImplementedError
