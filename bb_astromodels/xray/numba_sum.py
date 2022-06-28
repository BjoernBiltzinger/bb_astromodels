from functools import lru_cache

import numba
import numpy as np
from numpy.core.numeric import normalize_axis_index


@numba.njit()
def _numba_sum_into(arr, out, mask_out, mask_agg, shape_out, shape_agg):
    # get the multipliers
    # these times the position gives the index in the flattened array
    # this uses the same logic as np.ravel_multi_index
    multipliers = np.ones(arr.ndim, dtype="int32")
    for i in range(arr.ndim - 2, -1, -1):
        multipliers[i] = arr.shape[i + 1] * multipliers[i + 1]
    # multiplier components
    multipliers_agg = multipliers[mask_agg]
    multipliers_out = multipliers[mask_out]
    # flattened array
    a = arr.flatten()
    # loop over the kept values
    for pos_out in np.ndindex(shape_out):
        total = 0.0
        # this uses the same logic as np.ravel_multi_index
        i = 0
        for p, m in zip(pos_out, multipliers_out):
            i += p * m
        # loop over the aggregate values
        for pos_agg in np.ndindex(shape_agg):
            # this uses the same logic as np.ravel_multi_index
            j = 0
            for p, m in zip(pos_agg, multipliers_agg):
                j += p * m
            # update the total
            total += a[i + j]
        # save the total
        out[pos_out] = total
    # done!
    return out


@lru_cache()
def _normalize_axis(axis, ndim: int) -> tuple:
    if axis is None:
        axis = np.arange(ndim)
    axis = np.core.numeric.normalize_axis_tuple(axis, ndim, allow_duplicate=False)
    return axis


@lru_cache()
def _get_out_and_agg_parts(norm_axis, ndim: int, shape: tuple):
    axis = np.array(norm_axis)
    # get used dims
    mask_agg = np.zeros(ndim, dtype="bool")
    mask_agg[axis] = True
    mask_out = ~mask_agg
    # make immutable
    mask_agg.flags["WRITEABLE"] = False
    mask_out.flags["WRITEABLE"] = False
    # get the shape
    shape = np.array(shape)
    shape_agg = tuple(shape[mask_agg].tolist())
    shape_out = tuple(shape[mask_out].tolist())
    # done
    return mask_out, mask_agg, shape_out, shape_agg


def numba_sum(arr, axis=None):
    axis = _normalize_axis(axis, arr.ndim)
    # get the various shapes
    mask_out, mask_agg, shape_out, shape_agg = _get_out_and_agg_parts(
        axis, arr.ndim, arr.shape
    )
    # make the output array
    out = np.zeros(shape_out, dtype=arr.dtype)
    # write into the array
    _numba_sum_into(arr, out, mask_out, mask_agg, shape_out, shape_agg)
    # done!
    return out
