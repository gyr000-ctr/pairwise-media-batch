"""sorter 核心 API：稳定排序精灵批次。

排序键依次为 ``layer``、``depth``，键完全相同时保持输入中的插入序。
实例不保存任何可变状态，可被多个线程同时复用。
"""

import math
from numbers import Integral, Real
from operator import itemgetter

_KEY = itemgetter(0)


class Sorter:
    """无状态、线程安全的精灵批次排序器。"""

    def sort(self, items):
        """返回按 图层 → 深度 → 插入序 稳定排序后的新列表。

        不修改调用方传入的序列。图层必须是 >= 0 的整数，深度必须是
        有限实数，否则抛出 ``ValueError``。空输入返回空列表。
        """
        batch = list(items)
        if not batch:
            return []

        pairs = []
        append = pairs.append
        for item in batch:
            try:
                layer = item["layer"]
                depth = item["depth"]
            except (KeyError, TypeError, IndexError):
                raise ValueError(
                    "每个精灵必须包含 layer 和 depth 字段"
                ) from None

            if isinstance(layer, bool) or not isinstance(layer, Integral):
                raise ValueError("非法图层: {!r}（必须是 >= 0 的整数）".format(layer))
            if layer < 0:
                raise ValueError("非法图层: {!r}（必须 >= 0）".format(layer))

            if isinstance(depth, bool) or not isinstance(depth, Real):
                raise ValueError("非法深度: {!r}（必须是有限实数）".format(depth))
            if not math.isfinite(depth):
                raise ValueError("非法深度: {!r}（必须是有限实数）".format(depth))

            append(((layer, depth), item))

        pairs.sort(key=_KEY)
        return [item for _, item in pairs]
