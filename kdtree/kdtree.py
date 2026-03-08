import os
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libkdtree.so"))

_int = ctypes.c_int
_f64 = ctypes.c_double
_void_p = ctypes.c_void_p

_int_p = ctypes.POINTER(_int)
_int_pp = ctypes.POINTER(_int_p)

_f64_p = ctypes.POINTER(_f64)
_f64_pp = ctypes.POINTER(_f64_p)

_int2 = _int * 2
_int2_p = ctypes.POINTER(_int2)
_int2_pp = ctypes.POINTER(_int2_p)

_int_np = ndpointer(dtype=np.intc, flags="C")
_f64_np = ndpointer(dtype=np.float64, flags="C")
_i64_np = ndpointer(dtype=np.int64, flags="C")

_lib.kdtree_init.restype = _void_p
_lib.kdtree_init.argtypes = [_f64_np, _int, _int, _int]

_lib.kdtree_deinit.restype = None
_lib.kdtree_deinit.argtypes = [_void_p]

_lib.kdtree_nearest.restype = _int
_lib.kdtree_nearest.argtypes = [_void_p, _f64_np, _int_np, _f64_np, _int, _int, _int]

_lib.kdtree_radius.restype = _int
_lib.kdtree_radius.argtypes = [_void_p, _f64_np, _f64, _int_pp, _int_pp, _f64_pp, _int, _int]

_lib.kdtree_pairs.restype = _int
_lib.kdtree_pairs.argtypes = [_void_p, _void_p, _f64, _int2_pp]

_lib.kdtree_counts.restype = None
_lib.kdtree_counts.argtypes = [_void_p, _void_p, _f64_np, _i64_np, _int, _int]

_lib.kdtree_weighted.restype = None
_lib.kdtree_weighted.argtypes = [_void_p, _void_p, _f64_p, _f64_p, _f64_np, _f64_np, _int, _int]

_lib.kdtree_dump.restype = None
_lib.kdtree_dump.argtypes = [_void_p, ctypes.c_char_p]

_libc = ctypes.CDLL(None)
_libc.free.argtypes = [_void_p]
_libc.free.restype = None


class KDTree:
    def __init__(self, points, leaf_size=0):
        points = np.ascontiguousarray(points, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError("points must be a 2D array of shape (num, dim)")
        num, dim = points.shape
        self._ptr = _lib.kdtree_init(points, num, dim, leaf_size)

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.kdtree_deinit(self._ptr)

    def nearest(self, point, cap=1, sorted=True):
        point = np.ascontiguousarray(point, dtype=np.float64)
        single = point.ndim == 1
        if single:
            point = point[np.newaxis]
        num = len(point)
        index = np.empty((num, cap), dtype=np.intc)
        distance = np.empty((num, cap), dtype=np.float64)
        found = _lib.kdtree_nearest(self._ptr, point, index, distance, num, cap, int(sorted))
        index = index[:, :found]
        distance = distance[:, :found]
        if single:
            return index[0], distance[0]
        return index, distance

    def radius(self, point, radius, sorted=False):
        point = np.ascontiguousarray(point, dtype=np.float64)
        single = point.ndim == 1
        if single:
            point = point[np.newaxis]
        num = len(point)
        offset_p = _int_p()
        index_p = _int_p()
        distance_p = _f64_p()
        total = _lib.kdtree_radius(
            self._ptr,
            point,
            radius,
            ctypes.byref(offset_p),
            ctypes.byref(index_p),
            ctypes.byref(distance_p),
            num,
            int(sorted),
        )
        offset = np.empty(num + 1, dtype=np.intc)
        index = np.empty(total, dtype=np.intc)
        distance = np.empty(total, dtype=np.float64)
        ctypes.memmove(offset.ctypes.data, offset_p, (num + 1) * ctypes.sizeof(ctypes.c_int))
        ctypes.memmove(index.ctypes.data, index_p, total * ctypes.sizeof(ctypes.c_int))
        ctypes.memmove(distance.ctypes.data, distance_p, total * ctypes.sizeof(ctypes.c_double))
        _libc.free(ctypes.cast(offset_p, _void_p))
        _libc.free(ctypes.cast(index_p, _void_p))
        _libc.free(ctypes.cast(distance_p, _void_p))
        if single:
            return index, distance
        return [
            (index[offset[i] : offset[i + 1]], distance[offset[i] : offset[i + 1]])
            for i in range(num)
        ]

    def pairs(self, radius, other=None, output_type="set"):
        other_ptr = other._ptr if other is not None else None
        pairs = _int2_p()
        total = _lib.kdtree_pairs(self._ptr, other_ptr, radius, ctypes.byref(pairs))
        result = np.empty((total, 2), dtype=np.intc)
        ctypes.memmove(result.ctypes.data, pairs, total * 2 * ctypes.sizeof(ctypes.c_int))
        _libc.free(ctypes.cast(pairs, ctypes.c_void_p))
        if output_type == "ndarray":
            return result
        return set(result.view(np.dtype("i,i")).reshape(-1).tolist())

    def counts(self, radius, other=None, cumulative=True):
        other_ptr = other._ptr if other is not None else None
        radius = np.asarray(radius, dtype=np.float64)
        scalar = radius.ndim == 0
        radius = np.atleast_1d(radius)
        order = np.argsort(radius)
        sorted = np.ascontiguousarray(radius[order])
        result = np.empty(len(sorted), dtype=np.int64)
        _lib.kdtree_counts(self._ptr, other_ptr, sorted, result, len(sorted), int(cumulative))
        result = result[np.argsort(order)]
        return result[0].item() if scalar else result

    def counts_weighted(self, radius, weights, other=None, cumulative=True):
        other_ptr = other._ptr if other is not None else None
        if isinstance(weights, tuple):
            weight_self, weight_other = weights
        else:
            weight_self, weight_other = weights, None
        if weight_self is not None:
            weight_self = np.ascontiguousarray(weight_self, dtype=np.float64)
            weight_ptr = weight_self.ctypes.data_as(_f64_p)
        else:
            weight_ptr = None
        if weight_other is not None:
            weight_other = np.ascontiguousarray(weight_other, dtype=np.float64)
            weight_other_ptr = weight_other.ctypes.data_as(_f64_p)
        else:
            weight_other_ptr = None
        radius = np.asarray(radius, dtype=np.float64)
        scalar = radius.ndim == 0
        radius = np.atleast_1d(radius)
        order = np.argsort(radius)
        sorted = np.ascontiguousarray(radius[order])
        result = np.empty(len(sorted), dtype=np.float64)
        _lib.kdtree_weighted(
            self._ptr,
            other_ptr,
            weight_ptr,
            weight_other_ptr,
            sorted,
            result,
            len(sorted),
            int(cumulative),
        )
        result = result[np.argsort(order)]
        return result[0].item() if scalar else result

    def dump(self, path):
        _lib.kdtree_dump(self._ptr, path.encode())
