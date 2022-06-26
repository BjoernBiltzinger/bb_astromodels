import numba as nb
import numpy as np
from interpolation import interp


@nb.njit(fastmath=True, parallel=True, cache=True)
def _interp_loop(x, y, v):

    out = np.empty((v.shape[0], y.shape[1]))

    for i in nb.prange(y.shape[1]):

        out[:, i] = interp(x, y[:, i], v)

    return out


class UnivariateSpline(object):
    def __init__(self, xi, yi, axis=-1):

        self._x = xi
        # self._y = y

        self._y_axis = axis
        self._y_extra_shape = None

        if yi is not None:
            self._set_yi(yi, xi=xi, axis=axis)

        self._y = self._reshape_yi(yi)

    def __call__(self, v):

        x, x_shape = self._prepare_x(v)

        # out = interp(self._x, self._y, x)
        out = _interp_loop(self._x, self._y, x)

        return self._finish_y(out, x_shape)

    def _prepare_x(self, x):
        """Reshape input x array to 1-D"""

        x_shape = x.shape
        return x.ravel(), x_shape

    def _finish_y(self, y, x_shape):
        """Reshape interpolated y back to an N-D array similar to initial y"""
        y = y.reshape(x_shape + self._y_extra_shape)
        if self._y_axis != 0 and x_shape != ():
            nx = len(x_shape)
            ny = len(self._y_extra_shape)
            s = (
                list(range(nx, nx + self._y_axis))
                + list(range(nx))
                + list(range(nx + self._y_axis, nx + ny))
            )
            y = y.transpose(s)
        return y

    def _reshape_yi(self, yi):
        yi = np.moveaxis(np.asarray(yi), self._y_axis, 0)

        return yi.reshape((yi.shape[0], -1))

    def _set_yi(self, yi, xi=None, axis=None):
        if axis is None:
            axis = self._y_axis
        if axis is None:
            raise ValueError("no interpolation axis specified")

        yi = np.asarray(yi)

        shape = yi.shape
        if shape == ():
            shape = (1,)
        if xi is not None and shape[axis] != len(xi):
            raise ValueError(
                "x and y arrays must be equal in length along " "interpolation axis."
            )

        self._y_axis = axis % yi.ndim
        self._y_extra_shape = yi.shape[: self._y_axis] + yi.shape[self._y_axis + 1 :]
        self.dtype = None
