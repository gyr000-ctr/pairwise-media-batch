"""Native KD-tree kernel used by :mod:`palettemap` when a C compiler exists.

The source (``_kd.c``) is compiled once into a per-user cache directory and
loaded with :mod:`ctypes`.  Every tree keeps its own state, so concurrent
builds and queries are safe; the query call releases the GIL, which lets a
thread pool overlap chunks of a large image.

If anything goes wrong (no compiler, unsupported platform, build error) the
loader returns ``None`` and :mod:`palettemap` uses its numpy implementation.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
import tempfile
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(_HERE, "_kd.c")

# Pixels above this threshold are split across worker threads.
_PARALLEL_MIN_PIXELS = 65536
_MAX_WORKERS = min(32, (os.cpu_count() or 1))

_lib = None
_lock = threading.Lock()
_pool = None
_pool_lock = threading.Lock()


def _compiler():
    cc = os.environ.get("CC")
    if cc:
        return cc
    return "cc"


def _cache_path():
    with open(_SOURCE, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()[:16]
    stem = "palettemap-kd-%s-%s-%s" % (
        digest,
        sys.platform,
        ctypes.sizeof(ctypes.c_void_p) * 8,
    )
    base = os.environ.get("PALETTEMAP_CACHE_DIR") or os.path.join(
        tempfile.gettempdir(), "palettemap-cache"
    )
    return os.path.join(base, stem + ".so")


def _build(path):
    """Compile the kernel to *path* atomically; returns True on success."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".so", dir=os.path.dirname(path))
    os.close(fd)
    cmd = [
        _compiler(),
        "-O2",
        "-shared",
        "-fPIC",
        "-o",
        tmp,
        _SOURCE,
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=25,
        )
        if proc.returncode != 0:
            return False
        os.replace(tmp, path)
        return True
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _configure(lib):
    from ctypes import c_double, c_int, c_void_p, POINTER

    lib.kd_create.restype = c_void_p
    lib.kd_create.argtypes = [POINTER(c_double), c_int, c_int, c_int]
    lib.kd_query.restype = c_int
    lib.kd_query.argtypes = [c_void_p, POINTER(c_double), c_int, POINTER(c_int)]
    lib.kd_destroy.restype = None
    lib.kd_destroy.argtypes = [c_void_p]
    return lib


def load():
    """Return the loaded kernel, or ``None`` when it is unavailable."""
    global _lib
    if _lib is not None:
        return _lib if _lib is not False else None
    with _lock:
        if _lib is not None:
            return _lib if _lib is not False else None
        if sys.platform.startswith("win"):
            _lib = False
            return None
        try:
            path = _cache_path()
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                _build(path)
            if not os.path.exists(path):
                _lib = False
                return None
            _lib = _configure(ctypes.CDLL(path))
        except (OSError, AttributeError):
            _lib = False
            return None
        return _lib


def _executor():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                from concurrent.futures import ThreadPoolExecutor

                _pool = ThreadPoolExecutor(
                    max_workers=_MAX_WORKERS,
                    thread_name_prefix="palettemap",
                )
    return _pool


class NativeTree:
    """Thin RAII wrapper around one compiled KD tree."""

    __slots__ = ("_lib", "_ptr", "_n", "__weakref__")

    def __init__(self, lib, points):
        import numpy as np

        pts = np.ascontiguousarray(points, dtype=np.float64)
        self._lib = lib
        self._n = int(pts.shape[0])
        ptr = lib.kd_create(
            pts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._n,
            32,
            8,
        )
        if not ptr:
            raise MemoryError("palettemap: native KD-tree allocation failed")
        self._ptr = ptr

    def __del__(self):
        try:
            if getattr(self, "_ptr", None):
                self._lib.kd_destroy(self._ptr)
                self._ptr = None
        except Exception:  # pragma: no cover - interpreter shutdown
            pass

    def query(self, queries, dtype=None):
        """Return an integer array of nearest palette indices."""
        import numpy as np

        pts = np.ascontiguousarray(queries, dtype=np.float64).reshape(-1, 3)
        count = pts.shape[0]
        out = np.empty(count, dtype=np.int32)
        if count == 0:
            return out if dtype is None else out.astype(dtype, copy=False)
        q_ptr = pts.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        o_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        if count < _PARALLEL_MIN_PIXELS or _MAX_WORKERS <= 1:
            self._lib.kd_query(self._ptr, q_ptr, count, o_ptr)
        else:
            workers = min(_MAX_WORKERS, 4 + count // 131072)
            step = -(-count // workers)
            chunks = [(s, min(s + step, count)) for s in range(0, count, step)]
            errors = []

            def run(span):
                s, e = span
                rc = self._lib.kd_query(
                    self._ptr,
                    pts[s:e].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    e - s,
                    out[s:e].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                )
                if rc != 0:
                    errors.append(rc)

            try:
                list(_executor().map(run, chunks))
            except RuntimeError:  # 解释器关闭时线程池已销毁：串行收尾
                for span in chunks:
                    run(span)
            if errors:
                raise MemoryError("palettemap: native KD-tree query failed")
        if dtype is None or out.dtype == dtype:
            return out
        return out.astype(dtype, copy=False)
