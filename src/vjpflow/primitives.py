"""The primitive operations -- the ~20 ops everything else is built from.

Each primitive is a small class with two responsibilities:

- :meth:`Primitive.forward` -- *what to compute*, delegated to the backend. This
  runs during :func:`vjpflow.graph.evaluate`.
- :meth:`Primitive.vjp` -- the **vector-Jacobian product**: given the cotangent
  ``g_out = dL/d(output)``, return ``dL/d(input)`` for each input, *as new graph
  nodes*. This is the local gradient rule, and it is the only place the chain
  rule lives.

A vjp returns ``None`` for an input that is non-differentiable (e.g. integer
indices). Because vjps build ordinary ``Tensor`` nodes, the backward pass is
itself a lazy graph -- which is why higher-order gradients fall out for free.

Module-level functions (``add``, ``matmul``, ``softmax`` building blocks, ...)
are the user/author-facing way to apply a primitive: they do shape inference and
return a new lazy ``Tensor``. The arithmetic operators on ``Tensor``
(``__add__`` etc.) are defined in :mod:`vjpflow.tensor` and dispatch here.

Notation follows ``notes/``: ``G_X = dL/dX`` has the same shape as ``X``
(denominator layout).
"""

from __future__ import annotations

import builtins

import numpy as np

from vjpflow.backends import Native
from vjpflow.tensor import Tensor, as_tensor


def _scalar(value: float, ref: Tensor) -> Tensor:
    """A 0-d float32 constant on ``ref``'s backend (broadcasts against anything)."""
    return _const(np.array(value, dtype=np.float32), ref)


# ---------------------------------------------------------------------------
# Primitive base class
# ---------------------------------------------------------------------------


class Primitive:
    """Base class: a differentiable op with a forward and a vjp."""

    def forward(self, backend, *inputs: Native) -> Native:  # noqa: ANN001
        raise NotImplementedError

    def vjp(self, g: Tensor, out: Tensor, *inputs: Tensor) -> tuple[Tensor | None, ...]:
        """Return cotangents for each input given the output cotangent ``g``."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Graph-building + shape-inference helpers
# ---------------------------------------------------------------------------


def _node(prim: Primitive, inputs: tuple[Tensor, ...], shape, dtype=None) -> Tensor:
    """Wrap a primitive application in a new lazy ``Tensor`` node."""
    return Tensor(
        tuple(shape),
        op=prim,
        inputs=inputs,
        backend=inputs[0].backend,
        dtype=dtype or inputs[0].dtype,
    )


def _const(array: np.ndarray, ref: Tensor) -> Tensor:
    """A constant leaf living on the same backend as ``ref``."""
    return Tensor(
        array.shape,
        backend=ref.backend,
        dtype=array.dtype,
        data=ref.backend.from_numpy(array),
    )


def ones_like(ref: Tensor) -> Tensor:
    return _const(np.ones(ref.shape, dtype=np.float32), ref)


def zeros_like(ref: Tensor) -> Tensor:
    return _const(np.zeros(ref.shape, dtype=np.float32), ref)


def _norm_axis(axis, ndim: int) -> tuple[int, ...] | None:
    if axis is None:
        return None
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(sorted(ax % ndim for ax in axis))


def _reduced_shape(in_shape, axis, keepdims) -> tuple[int, ...]:
    if axis is None:
        return (1,) * len(in_shape) if keepdims else ()
    out = []
    for i, dim in enumerate(in_shape):
        if i in axis:
            if keepdims:
                out.append(1)
        else:
            out.append(dim)
    return tuple(out)


def sum_to_shape(g: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Reduce ``g`` back to ``shape`` by summing over broadcasted axes.

    This is *the* trick that makes broadcasting differentiable: when a forward
    op stretched a ``(d,)`` bias across ``(n, d)``, the gradient must be summed
    back over the stretched axis. PyTorch/MLX do exactly this internally.
    """
    shape = tuple(shape)
    if g.shape == shape:
        return g
    # 1. Sum away extra leading dims that broadcasting prepended.
    while len(g.shape) > len(shape):
        g = sum(g, axis=0, keepdims=False)
    # 2. Sum (keeping the axis) over dims that were size-1 in the target.
    axes = tuple(
        i for i, (gs, ss) in enumerate(zip(g.shape, shape, strict=True)) if ss == 1 and gs != 1
    )
    if axes:
        g = sum(g, axis=axes, keepdims=True)
    return g if g.shape == shape else reshape(g, shape)


def _swap_last_two(x: Tensor) -> Tensor:
    axes = list(range(x.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return transpose(x, tuple(axes))


# ---------------------------------------------------------------------------
# Elementwise binary
# ---------------------------------------------------------------------------


class Add(Primitive):
    def forward(self, backend, a, b):  # noqa: ANN001
        return backend.add(a, b)

    def vjp(self, g, out, a, b):  # noqa: ANN001
        return sum_to_shape(g, a.shape), sum_to_shape(g, b.shape)


class Sub(Primitive):
    def forward(self, backend, a, b):  # noqa: ANN001
        return backend.sub(a, b)

    def vjp(self, g, out, a, b):  # noqa: ANN001
        return sum_to_shape(g, a.shape), sum_to_shape(neg(g), b.shape)


class Mul(Primitive):
    def forward(self, backend, a, b):  # noqa: ANN001
        return backend.mul(a, b)

    def vjp(self, g, out, a, b):  # noqa: ANN001
        return sum_to_shape(mul(g, b), a.shape), sum_to_shape(mul(g, a), b.shape)


class Div(Primitive):
    def forward(self, backend, a, b):  # noqa: ANN001
        return backend.div(a, b)

    def vjp(self, g, out, a, b):  # noqa: ANN001
        ga = div(g, b)
        gb = neg(div(mul(g, a), mul(b, b)))
        return sum_to_shape(ga, a.shape), sum_to_shape(gb, b.shape)


class Pow(Primitive):
    def forward(self, backend, a, b):  # noqa: ANN001
        return backend.pow(a, b)

    def vjp(self, g, out, a, b):  # noqa: ANN001
        # d/da a**b = b * a**(b-1);  d/db a**b = (a**b) * log(a)
        ga = mul(g, mul(b, pow(a, sub(b, _scalar(1.0, b)))))
        gb = mul(g, mul(out, log(a)))
        return sum_to_shape(ga, a.shape), sum_to_shape(gb, b.shape)


class Maximum(Primitive):
    def forward(self, backend, a, b):  # noqa: ANN001
        return backend.maximum(a, b)

    def vjp(self, g, out, a, b):  # noqa: ANN001
        # Gradient flows to the larger operand; ties (a == b) go to b. This
        # gives the standard ReLU subgradient when b == 0.
        mask_a = greater(a, b)  # 1 where a > b, else 0
        ga = mul(g, mask_a)
        gb = sub(g, ga)  # g * (1 - mask_a)
        return sum_to_shape(ga, a.shape), sum_to_shape(gb, b.shape)


# ---------------------------------------------------------------------------
# Elementwise unary
# ---------------------------------------------------------------------------


class Neg(Primitive):
    def forward(self, backend, a):  # noqa: ANN001
        return backend.neg(a)

    def vjp(self, g, out, a):  # noqa: ANN001
        return (neg(g),)


class Exp(Primitive):
    def forward(self, backend, a):  # noqa: ANN001
        return backend.exp(a)

    def vjp(self, g, out, a):  # noqa: ANN001
        return (mul(g, out),)  # d/dx exp(x) = exp(x) = out


class Log(Primitive):
    def forward(self, backend, a):  # noqa: ANN001
        return backend.log(a)

    def vjp(self, g, out, a):  # noqa: ANN001
        return (div(g, a),)


class Sqrt(Primitive):
    def forward(self, backend, a):  # noqa: ANN001
        return backend.sqrt(a)

    def vjp(self, g, out, a):  # noqa: ANN001
        # d/dx sqrt(x) = 1/(2 sqrt(x)) = 1/(2 out)
        return (div(g, mul(_scalar(2.0, out), out)),)


class Tanh(Primitive):
    def forward(self, backend, a):  # noqa: ANN001
        return backend.tanh(a)

    def vjp(self, g, out, a):  # noqa: ANN001
        return (mul(g, sub(_scalar(1.0, out), mul(out, out))),)  # 1 - tanh^2


class Greater(Primitive):
    """Non-differentiable comparison producing a 0/1 mask."""

    def forward(self, backend, a, b):  # noqa: ANN001
        return backend.greater(a, b)

    def vjp(self, g, out, a, b):  # noqa: ANN001
        return None, None


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


class Sum(Primitive):
    def __init__(self, axis, keepdims: bool) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, backend, a):  # noqa: ANN001
        return backend.sum(a, self.axis, self.keepdims)

    def vjp(self, g, out, a):  # noqa: ANN001
        # Broadcast the (reduced) cotangent back across the summed axes.
        if not self.keepdims and self.axis is not None:
            g = reshape(g, _reduced_shape(a.shape, self.axis, keepdims=True))
        return (broadcast_to(g, a.shape),)


class Mean(Primitive):
    def __init__(self, axis, keepdims: bool) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, backend, a):  # noqa: ANN001
        return backend.mean(a, self.axis, self.keepdims)

    def vjp(self, g, out, a):  # noqa: ANN001
        if self.axis is None:
            count = a.size
        else:
            count = int(np.prod([a.shape[ax] for ax in self.axis]))
        if not self.keepdims and self.axis is not None:
            g = reshape(g, _reduced_shape(a.shape, self.axis, keepdims=True))
        scaled = div(g, _scalar(float(count), g))
        return (broadcast_to(scaled, a.shape),)


class Max(Primitive):
    def __init__(self, axis, keepdims: bool) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, backend, a):  # noqa: ANN001
        return backend.max(a, self.axis, self.keepdims)

    def vjp(self, g, out, a):  # noqa: ANN001
        # Route gradient to the maximal element(s). Build a keepdims view of the
        # max so it broadcasts against `a`.
        out_kd = out
        g_kd = g
        if not self.keepdims and self.axis is not None:
            kd = _reduced_shape(a.shape, self.axis, keepdims=True)
            out_kd = reshape(out, kd)
            g_kd = reshape(g, kd)
        # a <= max everywhere, so (a == max) == 1 - (max > a).
        mask = sub(ones_like(a), greater(broadcast_to(out_kd, a.shape), a))
        return (mul(broadcast_to(g_kd, a.shape), mask),)


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------


class MatMul(Primitive):
    def forward(self, backend, a, b):  # noqa: ANN001
        return backend.matmul(a, b)

    def vjp(self, g, out, a, b):  # noqa: ANN001
        # Y = A @ B  =>  G_A = G @ B^T,  G_B = A^T @ G  (transpose last two axes).
        ga = matmul(g, _swap_last_two(b))
        gb = matmul(_swap_last_two(a), g)
        # Sum away any broadcasted batch dims.
        return sum_to_shape(ga, a.shape), sum_to_shape(gb, b.shape)


# ---------------------------------------------------------------------------
# Shape ops
# ---------------------------------------------------------------------------


class Reshape(Primitive):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def forward(self, backend, a):  # noqa: ANN001
        return backend.reshape(a, self.shape)

    def vjp(self, g, out, a):  # noqa: ANN001
        return (reshape(g, a.shape),)


class Transpose(Primitive):
    def __init__(self, axes: tuple[int, ...]) -> None:
        self.axes = axes

    def forward(self, backend, a):  # noqa: ANN001
        return backend.transpose(a, self.axes)

    def vjp(self, g, out, a):  # noqa: ANN001
        inv = [0] * len(self.axes)
        for i, ax in enumerate(self.axes):
            inv[ax] = i
        return (transpose(g, tuple(inv)),)


class BroadcastTo(Primitive):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def forward(self, backend, a):  # noqa: ANN001
        return backend.broadcast_to(a, self.shape)

    def vjp(self, g, out, a):  # noqa: ANN001
        return (sum_to_shape(g, a.shape),)


class Slice(Primitive):
    def __init__(self, start: int, stop: int, axis: int) -> None:
        self.start = start
        self.stop = stop
        self.axis = axis

    def forward(self, backend, a):  # noqa: ANN001
        return backend.slice(a, self.start, self.stop, self.axis)

    def vjp(self, g, out, a):  # noqa: ANN001
        # Pad the gradient back to the input shape with zeros on either side of
        # the slice (expressed as a concat -- no new backend op needed).
        ax = self.axis
        pieces: list[Tensor] = []
        if self.start > 0:
            before = list(a.shape)
            before[ax] = self.start
            pieces.append(_const(np.zeros(before, np.float32), g))
        pieces.append(g)
        after_len = a.shape[ax] - self.stop
        if after_len > 0:
            after = list(a.shape)
            after[ax] = after_len
            pieces.append(_const(np.zeros(after, np.float32), g))
        return (concat(pieces, ax) if len(pieces) > 1 else g,)


class Concat(Primitive):
    def __init__(self, axis: int, sizes: tuple[int, ...]) -> None:
        self.axis = axis
        self.sizes = sizes  # length of each input along `axis`, for the backward split

    def forward(self, backend, *arrays):  # noqa: ANN001
        return backend.concat(list(arrays), self.axis)

    def vjp(self, g, out, *inputs):  # noqa: ANN001
        grads: list[Tensor] = []
        offset = 0
        for size in self.sizes:
            grads.append(slice_(g, offset, offset + size, self.axis))
            offset += size
        return tuple(grads)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


class Gather(Primitive):
    """Row-gather ``table[indices]`` (the embedding lookup)."""

    def __init__(self, indices: np.ndarray, table_shape: tuple[int, ...]) -> None:
        self.indices = indices
        self.table_shape = table_shape

    def forward(self, backend, table):  # noqa: ANN001
        return backend.gather(table, self.indices)

    def vjp(self, g, out, table):  # noqa: ANN001
        return (scatter_add(g, self.indices, self.table_shape),)


class ScatterAdd(Primitive):
    """Accumulate ``updates`` into a zero array at ``indices`` (gather's inverse)."""

    def __init__(self, indices: np.ndarray, out_shape: tuple[int, ...]) -> None:
        self.indices = indices
        self.out_shape = out_shape

    def forward(self, backend, updates):  # noqa: ANN001
        return backend.scatter_add(self.out_shape, self.indices, updates)

    def vjp(self, g, out, updates):  # noqa: ANN001
        # d(out)/d(updates) gathers the cotangent at the same indices. The tail
        # is the table's non-batch dims (``out_shape[1:]``), NOT
        # ``updates.shape[1:]`` -- those differ once ``indices`` is
        # multidimensional (``updates`` then has ``indices.ndim`` leading axes).
        return (gather(g, self.indices, self.out_shape[1:]),)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


class Where(Primitive):
    """``cond ? a : b`` with a static boolean ``cond`` (e.g. dropout mask)."""

    def __init__(self, cond: np.ndarray) -> None:
        self.cond = cond

    def forward(self, backend, a, b):  # noqa: ANN001
        return backend.where(self.cond, a, b)

    def vjp(self, g, out, a, b):  # noqa: ANN001
        mask_a = _const(self.cond.astype(np.float32), g)
        mask_b = _const((~self.cond).astype(np.float32), g)
        return sum_to_shape(mul(g, mask_a), a.shape), sum_to_shape(mul(g, mask_b), b.shape)


# ---------------------------------------------------------------------------
# Functional API (these build lazy nodes; vjps and layers call them)
# ---------------------------------------------------------------------------


def add(a: Tensor, b) -> Tensor:  # noqa: ANN001
    a, b = as_tensor(a), as_tensor(b)
    return _node(Add(), (a, b), np.broadcast_shapes(a.shape, b.shape))


def sub(a: Tensor, b) -> Tensor:  # noqa: ANN001
    a, b = as_tensor(a), as_tensor(b)
    return _node(Sub(), (a, b), np.broadcast_shapes(a.shape, b.shape))


def mul(a: Tensor, b) -> Tensor:  # noqa: ANN001
    a, b = as_tensor(a), as_tensor(b)
    return _node(Mul(), (a, b), np.broadcast_shapes(a.shape, b.shape))


def div(a: Tensor, b) -> Tensor:  # noqa: ANN001
    a, b = as_tensor(a), as_tensor(b)
    return _node(Div(), (a, b), np.broadcast_shapes(a.shape, b.shape))


def pow(a: Tensor, b) -> Tensor:  # noqa: ANN001, A001
    a, b = as_tensor(a), as_tensor(b)
    return _node(Pow(), (a, b), np.broadcast_shapes(a.shape, b.shape))


def maximum(a: Tensor, b) -> Tensor:  # noqa: ANN001
    a, b = as_tensor(a), as_tensor(b)
    return _node(Maximum(), (a, b), np.broadcast_shapes(a.shape, b.shape))


def greater(a: Tensor, b) -> Tensor:  # noqa: ANN001
    a, b = as_tensor(a), as_tensor(b)
    return _node(Greater(), (a, b), np.broadcast_shapes(a.shape, b.shape))


def neg(a: Tensor) -> Tensor:
    return _node(Neg(), (a,), a.shape)


def exp(a: Tensor) -> Tensor:
    return _node(Exp(), (a,), a.shape)


def log(a: Tensor) -> Tensor:
    return _node(Log(), (a,), a.shape)


def sqrt(a: Tensor) -> Tensor:
    return _node(Sqrt(), (a,), a.shape)


def tanh(a: Tensor) -> Tensor:
    return _node(Tanh(), (a,), a.shape)


def sum(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:  # noqa: A001
    axis = _norm_axis(axis, a.ndim)
    return _node(Sum(axis, keepdims), (a,), _reduced_shape(a.shape, axis, keepdims))


def mean(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    axis = _norm_axis(axis, a.ndim)
    return _node(Mean(axis, keepdims), (a,), _reduced_shape(a.shape, axis, keepdims))


def max(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:  # noqa: A001
    axis = _norm_axis(axis, a.ndim)
    return _node(Max(axis, keepdims), (a,), _reduced_shape(a.shape, axis, keepdims))


def matmul(a: Tensor, b: Tensor) -> Tensor:
    a, b = as_tensor(a), as_tensor(b)
    batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    shape = (*batch, a.shape[-2], b.shape[-1])
    return _node(MatMul(), (a, b), shape)


def reshape(a: Tensor, shape: tuple[int, ...]) -> Tensor:
    shape = tuple(shape)
    return _node(Reshape(shape), (a,), shape)


def transpose(a: Tensor, axes: tuple[int, ...]) -> Tensor:
    out_shape = tuple(a.shape[ax] for ax in axes)
    return _node(Transpose(tuple(axes)), (a,), out_shape)


def broadcast_to(a: Tensor, shape: tuple[int, ...]) -> Tensor:
    return _node(BroadcastTo(tuple(shape)), (a,), tuple(shape))


def slice_(a: Tensor, start: int, stop: int, axis: int = -1) -> Tensor:
    axis = axis % a.ndim
    out_shape = list(a.shape)
    out_shape[axis] = stop - start
    return _node(Slice(start, stop, axis), (a,), tuple(out_shape))


def concat(tensors: list[Tensor], axis: int = -1) -> Tensor:
    tensors = [as_tensor(t) for t in tensors]
    axis = axis % tensors[0].ndim
    sizes = tuple(t.shape[axis] for t in tensors)
    out_shape = list(tensors[0].shape)
    out_shape[axis] = builtins.sum(sizes)
    return _node(Concat(axis, sizes), tuple(tensors), tuple(out_shape))


def gather(table: Tensor, indices: np.ndarray, tail_shape: tuple[int, ...] | None = None) -> Tensor:
    indices = np.asarray(indices)
    tail = tail_shape if tail_shape is not None else table.shape[1:]
    out_shape = (*indices.shape, *tail)
    return _node(Gather(indices, table.shape), (table,), out_shape)


def scatter_add(updates: Tensor, indices: np.ndarray, out_shape: tuple[int, ...]) -> Tensor:
    indices = np.asarray(indices)
    return _node(ScatterAdd(indices, tuple(out_shape)), (updates,), tuple(out_shape))


def where(cond: np.ndarray, a: Tensor, b) -> Tensor:  # noqa: ANN001
    a, b = as_tensor(a), as_tensor(b)
    cond = np.asarray(cond, dtype=bool)
    return _node(Where(cond), (a, b), np.broadcast_shapes(a.shape, b.shape))


# ---------------------------------------------------------------------------
# Attach operators to Tensor
# ---------------------------------------------------------------------------
# The arithmetic operators / convenience methods are defined directly on
# ``Tensor`` (in tensor.py) and dispatch into the functions above. Importing
# this module is still required at startup so those methods resolve -- which is
# why ``vjpflow/__init__.py`` imports ``primitives`` eagerly.
