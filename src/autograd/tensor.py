"""The :class:`Tensor` -- a node in the lazy computation graph.

A ``Tensor`` does *not* hold a computed value by default. It holds a recipe:
"I am the output of primitive ``op`` applied to inputs ``parents``." Calling an
operation (``a + b``, ``a @ b``, ``softmax(x)``) returns a *new* ``Tensor`` that
records the op and its parents but computes nothing. The numbers are produced
only when you ask for them -- :meth:`Tensor.eval`, :meth:`Tensor.numpy`, or
:meth:`Tensor.item` -- which triggers the evaluation engine in
:mod:`autograd.graph`.

This is the MLX / JAX execution model. Why bother?

- **Whole-graph view before execution.** Because the graph exists as data
  before anything runs, you can rewrite it -- fuse elementwise chains, pick
  better kernels, allocate buffers once. That is what :mod:`autograd.jit` does.
- **One autodiff mechanism.** ``grad`` builds a *backward graph* out of the same
  ``Tensor`` nodes; differentiation is just another graph-to-graph
  transformation (see :mod:`autograd.autodiff`).

The operator overloads (``__add__`` and friends) are attached at import time by
:mod:`autograd.primitives`, to avoid a circular import (primitives need
``Tensor``; ``Tensor`` would otherwise need primitives).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from autograd.backends import Backend, Native, default_backend

if TYPE_CHECKING:
    from autograd.primitives import Primitive


class Tensor:
    """A lazily-evaluated node in the computation graph.

    Attributes
    ----------
    shape:
        Static output shape, known at graph-build time (we never need the data
        to know the shape -- crucial for building the backward graph lazily).
    op:
        The :class:`~autograd.primitives.Primitive` that produces this node, or
        ``None`` for a *leaf* (data the user supplied).
    inputs:
        The parent tensors this node consumes. Empty for a leaf.
    backend:
        Where this node evaluates. Inherited from inputs / the default.
    """

    __slots__ = ("shape", "dtype", "op", "inputs", "backend", "_data")

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        op: Primitive | None = None,
        inputs: tuple[Tensor, ...] = (),
        backend: Backend | None = None,
        dtype: np.dtype | None = None,
        data: Native | None = None,
    ) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype if dtype is not None else np.dtype(np.float32)
        self.op = op
        self.inputs = inputs
        self.backend = backend or default_backend()
        # ``_data`` is the cached, materialised native array. ``None`` means
        # "not evaluated yet". The evaluation engine fills it in.
        self._data: Native | None = data

    # -- construction ----------------------------------------------------
    @property
    def is_leaf(self) -> bool:
        """A leaf has no producing op; its data was given, not computed."""
        return self.op is None

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    # -- evaluation ------------------------------------------------------
    def eval(self) -> Tensor:
        """Force materialisation of this node (and its dependencies)."""
        from autograd.graph import evaluate

        evaluate(self)
        return self

    def numpy(self) -> np.ndarray:
        """Evaluate and copy the result to a host numpy array."""
        self.eval()
        return self.backend.to_numpy(self._data)

    def item(self) -> float:
        """Evaluate a scalar tensor and return a Python float."""
        return float(self.numpy().reshape(-1)[0])

    def detach(self) -> Tensor:
        """A new *leaf* with this tensor's evaluated value (cuts the graph).

        Used by optimisers: after a gradient step you want fresh parameter
        leaves, not a graph that reaches back through every past update.
        """
        return Tensor(
            self.shape,
            backend=self.backend,
            dtype=self.dtype,
            data=self.eval()._data,
        )

    # -- operators -------------------------------------------------------
    # These dispatch into the functional ops in ``autograd.primitives``. We
    # import lazily inside each method: ``primitives`` imports ``Tensor`` at
    # module load, so a top-level import here would be circular. Python caches
    # the module, so the per-call import is just a dict lookup.
    def __add__(self, other: Any) -> Tensor:
        from autograd import primitives as P

        return P.add(self, other)

    def __radd__(self, other: Any) -> Tensor:
        from autograd import primitives as P

        return P.add(other, self)

    def __sub__(self, other: Any) -> Tensor:
        from autograd import primitives as P

        return P.sub(self, other)

    def __rsub__(self, other: Any) -> Tensor:
        from autograd import primitives as P

        return P.sub(other, self)

    def __mul__(self, other: Any) -> Tensor:
        from autograd import primitives as P

        return P.mul(self, other)

    def __rmul__(self, other: Any) -> Tensor:
        from autograd import primitives as P

        return P.mul(other, self)

    def __truediv__(self, other: Any) -> Tensor:
        from autograd import primitives as P

        return P.div(self, other)

    def __rtruediv__(self, other: Any) -> Tensor:
        from autograd import primitives as P

        return P.div(other, self)

    def __neg__(self) -> Tensor:
        from autograd import primitives as P

        return P.neg(self)

    def __pow__(self, other: Any) -> Tensor:
        from autograd import primitives as P

        return P.pow(self, other)

    def __matmul__(self, other: Any) -> Tensor:
        from autograd import primitives as P

        return P.matmul(self, other)

    # -- convenience methods (mirror numpy / torch ergonomics) -----------
    def sum(self, axis: Any = None, keepdims: bool = False) -> Tensor:
        from autograd import primitives as P

        return P.sum(self, axis, keepdims)

    def mean(self, axis: Any = None, keepdims: bool = False) -> Tensor:
        from autograd import primitives as P

        return P.mean(self, axis, keepdims)

    def max(self, axis: Any = None, keepdims: bool = False) -> Tensor:
        from autograd import primitives as P

        return P.max(self, axis, keepdims)

    def reshape(self, *shape: Any) -> Tensor:
        from autograd import primitives as P

        resolved = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        return P.reshape(self, resolved)

    def transpose(self, *axes: int) -> Tensor:
        from autograd import primitives as P

        return P.transpose(self, axes)

    def exp(self) -> Tensor:
        from autograd import primitives as P

        return P.exp(self)

    def log(self) -> Tensor:
        from autograd import primitives as P

        return P.log(self)

    @property
    def T(self) -> Tensor:  # noqa: N802  (conventional transpose attribute, as in numpy/torch)
        """Swap the last two axes (matrix transpose for 2-D, batched otherwise)."""
        from autograd.primitives import _swap_last_two

        return _swap_last_two(self)

    # -- niceties --------------------------------------------------------
    def __repr__(self) -> str:
        state = "evaluated" if self._data is not None else "lazy"
        kind = "leaf" if self.is_leaf else type(self.op).__name__
        return f"Tensor(shape={self.shape}, {kind}, {state}, backend={self.backend.name})"

    def __len__(self) -> int:
        return self.shape[0]


def tensor(data: Any, *, backend: Backend | None = None) -> Tensor:
    """Create a *leaf* tensor from array-like ``data``.

    Floats are normalised to float32; integer arrays (token ids, indices) are
    preserved so they can drive ``gather`` without being treated as
    differentiable.
    """
    backend = backend or default_backend()
    array = np.asarray(data)
    native = backend.from_numpy(array)
    # Read the (possibly dtype-normalised) shape/dtype back from the native.
    host = backend.to_numpy(native)
    return Tensor(host.shape, backend=backend, dtype=host.dtype, data=native)


def as_tensor(x: Any, *, backend: Backend | None = None) -> Tensor:
    """Coerce ``x`` to a ``Tensor`` (pass-through if it already is one)."""
    if isinstance(x, Tensor):
        return x
    return tensor(x, backend=backend)
