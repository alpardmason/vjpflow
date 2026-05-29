"""The numpy backend -- the reference implementation.

Every method is a thin wrapper over numpy. This backend has *full* coverage:
all primitives, all layers, and the GPT-2 capstone run on it. It doubles as the
ground truth in our parity tests (the Metal backend is checked against it).

Because numpy is so direct, this file is also the clearest place to *read* what
each primitive means operationally before tackling the Metal kernels.
"""

from __future__ import annotations

import numpy as np

from vjpflow.backends.base import Backend, Native

# The engine standardises on float32 for floating data (matching MLX/PyTorch
# defaults) and int64 for indices.
DEFAULT_FLOAT = np.float32


class NumpyBackend(Backend):
    name = "numpy"

    def from_numpy(self, array: np.ndarray) -> Native:
        # Floats are normalised to float32; integer index arrays are preserved.
        if np.issubdtype(array.dtype, np.floating):
            return np.ascontiguousarray(array, dtype=DEFAULT_FLOAT)
        return np.ascontiguousarray(array)

    def to_numpy(self, array: Native) -> np.ndarray:
        return np.asarray(array)

    # -- elementwise binary ---------------------------------------------
    def add(self, a: Native, b: Native) -> Native:
        return np.add(a, b)

    def sub(self, a: Native, b: Native) -> Native:
        return np.subtract(a, b)

    def mul(self, a: Native, b: Native) -> Native:
        return np.multiply(a, b)

    def div(self, a: Native, b: Native) -> Native:
        return np.divide(a, b)

    def pow(self, a: Native, b: Native) -> Native:
        return np.power(a, b)

    def maximum(self, a: Native, b: Native) -> Native:
        return np.maximum(a, b)

    def greater(self, a: Native, b: Native) -> Native:
        return np.greater(a, b)

    # -- elementwise unary ----------------------------------------------
    def neg(self, a: Native) -> Native:
        return np.negative(a)

    def exp(self, a: Native) -> Native:
        return np.exp(a)

    def log(self, a: Native) -> Native:
        return np.log(a)

    def sqrt(self, a: Native) -> Native:
        return np.sqrt(a)

    def tanh(self, a: Native) -> Native:
        return np.tanh(a)

    # -- reductions ------------------------------------------------------
    def sum(self, a: Native, axis: tuple[int, ...] | None, keepdims: bool) -> Native:
        return np.sum(a, axis=axis, keepdims=keepdims)

    def mean(self, a: Native, axis: tuple[int, ...] | None, keepdims: bool) -> Native:
        return np.mean(a, axis=axis, keepdims=keepdims)

    def max(self, a: Native, axis: tuple[int, ...] | None, keepdims: bool) -> Native:
        return np.max(a, axis=axis, keepdims=keepdims)

    # -- linear algebra --------------------------------------------------
    def matmul(self, a: Native, b: Native) -> Native:
        return np.matmul(a, b)

    # -- shape -----------------------------------------------------------
    def reshape(self, a: Native, shape: tuple[int, ...]) -> Native:
        return np.reshape(a, shape)

    def transpose(self, a: Native, axes: tuple[int, ...]) -> Native:
        return np.transpose(a, axes)

    def broadcast_to(self, a: Native, shape: tuple[int, ...]) -> Native:
        return np.broadcast_to(a, shape)

    # -- indexing --------------------------------------------------------
    def gather(self, table: Native, indices: Native) -> Native:
        return table[indices]

    def scatter_add(
        self, out_shape: tuple[int, ...], indices: Native, updates: Native
    ) -> Native:
        out = np.zeros(out_shape, dtype=updates.dtype)
        # np.add.at performs *unbuffered* scatter-add, so repeated indices
        # accumulate instead of overwriting -- the whole point of the op.
        np.add.at(out, indices, updates)
        return out

    # -- structural ------------------------------------------------------
    def concat(self, arrays: list[Native], axis: int) -> Native:
        return np.concatenate(arrays, axis=axis)

    def slice(self, a: Native, start: int, stop: int, axis: int) -> Native:
        index = [slice(None)] * a.ndim
        index[axis] = slice(start, stop)
        return a[tuple(index)]

    # -- selection -------------------------------------------------------
    def where(self, cond: Native, a: Native, b: Native) -> Native:
        return np.where(cond, a, b)
