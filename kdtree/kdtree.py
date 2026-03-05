import os
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libkdtree.so"))

_void_p = ctypes.c_void_p

_int = ctypes.c_int
_f64 = ctypes.c_double

_int_np = ndpointer(dtype=np.intc, flags="C")
_f64_np = ndpointer(dtype=np.float64, flags="C")

_lib.kdtree_init.restype = _void_p
_lib.kdtree_init.argtypes = [_f64_np, _int, _int, _int]

_lib.kdtree_deinit.restype = None
_lib.kdtree_deinit.argtypes = [_void_p]

_lib.kdtree_query.restype = _int
_lib.kdtree_query.argtypes = [_void_p, _f64_np, _int_np, _f64_np, _int]

_lib.kdtree_query_radius.restype = _int
_lib.kdtree_query_radius.argtypes = [_void_p, _f64_np, _f64, _int_np, _f64_np, _int, _int]


class KDTree:
    def __init__(self, points, leaf_size=0):
        points = np.ascontiguousarray(points, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError("points must be a 2D array of shape (num, dim)")
        self._points = points
        num, dim = points.shape
        self._ptr = _lib.kdtree_init(points, num, dim, leaf_size)

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.kdtree_deinit(self._ptr)

    def query(self, point, k=1):
        point = np.ascontiguousarray(point, dtype=np.float64)
        index = np.empty(k, dtype=np.intc)
        distance = np.empty(k, dtype=np.float64)
        _lib.kdtree_query(self._ptr, point, index, distance, k)
        return index, distance

    def query_radius(self, point, radius, cap=64, sorted=False):
        point = np.ascontiguousarray(point, dtype=np.float64)
        while True:
            index = np.empty(cap, dtype=np.intc)
            distance = np.empty(cap, dtype=np.float64)
            num = _lib.kdtree_query_radius(
                self._ptr, point, radius, index, distance, cap, int(sorted)
            )
            if num <= cap:
                return index[:num], distance[:num]
            cap = num
