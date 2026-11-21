"""分形值噪声（fractal value noise / fBm）。

- 同种子可复现、不同种子不同，输出归一化到 [0, 1]
- 多倍频叠加（octaves / lacunarity / gain 可配），三次 fade 保证格点连续无断层
- 提供解析梯度 ``gradient``（与有限差分一致）
- 无状态、并发安全：构造后实例数据全部只读，采样不写任何状态
"""

import math
from random import Random

__all__ = [
    "ValueNoise",
    "NoiseError",
    "SeedError",
    "OctavesError",
    "CoordinateError",
    "MAX_OCTAVES",
]

MAX_OCTAVES = 16
_MAX_SEED = (1 << 64) - 1
_MASK = 255


class NoiseError(ValueError):
    """noise 参数错误基类（是 ValueError，便于按类别区分）。"""


class SeedError(NoiseError):
    """非法种子（非整数、负数或超过 2**64-1）。"""


class OctavesError(NoiseError):
    """octaves 越界（非整数或不在 [1, MAX_OCTAVES]）。"""


class CoordinateError(NoiseError):
    """坐标非法（非实数或非有限值）。"""


def _check_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SeedError("seed 必须为 int，got %r" % (seed,))
    if seed < 0 or seed > _MAX_SEED:
        raise SeedError("seed 超出范围 [0, 2**64 - 1]: %r" % (seed,))
    return seed


def _check_octaves(octaves):
    if isinstance(octaves, bool) or not isinstance(octaves, int):
        raise OctavesError("octaves 必须为 int，got %r" % (octaves,))
    if octaves < 1 or octaves > MAX_OCTAVES:
        raise OctavesError(
            "octaves 超出范围 [1, %d]: %r" % (MAX_OCTAVES, octaves)
        )
    return octaves


def _check_finite_param(name, value, lo, hi):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s 必须为实数，got %r" % (name, value)) from None
    if not math.isfinite(v) or v < lo or v > hi:
        raise ValueError("%s 超出范围 [%s, %s]: %r" % (name, lo, hi, value))
    return v


def _coords(x, y):
    try:
        fx = float(x)
        fy = float(y)
    except (TypeError, ValueError):
        raise CoordinateError("坐标必须为实数，got (%r, %r)" % (x, y)) from None
    if not (math.isfinite(fx) and math.isfinite(fy)):
        raise CoordinateError("坐标必须为有限值，got (%r, %r)" % (x, y))
    return fx, fy


class ValueNoise:
    """分形值噪声采样器。构造后不可变，采样纯函数式，线程安全。"""

    __slots__ = (
        "_seed",
        "_octaves",
        "_lacunarity",
        "_gain",
        "_perm",
        "_vals",
        "_layers",
        "_norm",
    )

    def __init__(self, seed, octaves=4, lacunarity=2.0, gain=0.5):
        _check_seed(seed)
        _check_octaves(octaves)
        lac = _check_finite_param("lacunarity", lacunarity, 1.0, 1024.0)
        g = _check_finite_param("gain", gain, 0.0, 1.0)

        rng = Random(seed)
        perm = list(range(256))
        rng.shuffle(perm)
        perm = perm + perm  # 512，省一次取模
        vals = [p / 255.0 for p in perm]

        layers = []
        norm = 0.0
        amp = 1.0
        freq = 1.0
        for _ in range(octaves):
            # 每个倍频一个整数格偏移，避免各层格子对齐产生相关
            layers.append((freq, amp, rng.randrange(256), rng.randrange(256)))
            norm += amp
            amp *= g
            freq *= lac

        self._seed = seed
        self._octaves = octaves
        self._lacunarity = lac
        self._gain = g
        self._perm = perm
        self._vals = vals
        self._layers = tuple(layers)
        self._norm = norm

    def sample(self, x, y):
        """采样 (x, y)，返回 [0, 1] 的 float。"""
        fx, fy = _coords(x, y)
        perm = self._perm
        vals = self._vals
        floor = math.floor
        total = 0.0
        for freq, amp, ox, oy in self._layers:
            sx = fx * freq + ox
            sy = fy * freq + oy
            xi = floor(sx)
            yi = floor(sy)
            tx = sx - xi
            ty = sy - yi
            a = perm[xi & _MASK]
            b = perm[(xi + 1) & _MASK]
            i = yi & _MASK
            j = (yi + 1) & _MASK
            v00 = vals[(a + i) & _MASK]
            v10 = vals[(b + i) & _MASK]
            v01 = vals[(a + j) & _MASK]
            v11 = vals[(b + j) & _MASK]
            u = tx * tx * (3.0 - 2.0 * tx)
            v = ty * ty * (3.0 - 2.0 * ty)
            total += amp * (
                (v00 + u * (v10 - v00)) * (1.0 - v)
                + (v01 + u * (v11 - v01)) * v
            )
        return total / self._norm

    def gradient(self, x, y):
        """解析梯度，返回 (df/dx, df/dy) 二元组。"""
        fx, fy = _coords(x, y)
        perm = self._perm
        vals = self._vals
        floor = math.floor
        gx = 0.0
        gy = 0.0
        for freq, amp, ox, oy in self._layers:
            sx = fx * freq + ox
            sy = fy * freq + oy
            xi = floor(sx)
            yi = floor(sy)
            tx = sx - xi
            ty = sy - yi
            a = perm[xi & _MASK]
            b = perm[(xi + 1) & _MASK]
            i = yi & _MASK
            j = (yi + 1) & _MASK
            v00 = vals[(a + i) & _MASK]
            v10 = vals[(b + i) & _MASK]
            v01 = vals[(a + j) & _MASK]
            v11 = vals[(b + j) & _MASK]
            u = tx * tx * (3.0 - 2.0 * tx)
            v = ty * ty * (3.0 - 2.0 * ty)
            du = 6.0 * tx * (1.0 - tx)
            dv = 6.0 * ty * (1.0 - ty)
            i1 = v00 + u * (v10 - v00)
            i2 = v01 + u * (v11 - v01)
            gx += amp * freq * du * ((v10 - v00) * (1.0 - v) + (v11 - v01) * v)
            gy += amp * freq * dv * (i2 - i1)
        norm = self._norm
        return gx / norm, gy / norm

    # 常用别名
    grad = gradient
