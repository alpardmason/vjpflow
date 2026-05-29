"""The backend abstraction: where numbers actually get crunched.

A :class:`Backend` knows how to execute the engine's *primitive* operations on
some concrete array representation (``Native``). The lazy graph in
:mod:`autograd.tensor` is backend-agnostic; only :meth:`Primitive.forward
<autograd.primitives.Primitive.forward>` ever touches a backend.

Design pattern -- *Strategy*: the graph holds a reference to a backend object
and delegates the "how do I add two arrays" decision to it. Swapping numpy for
Metal is a one-line change (:func:`autograd.backends.set_default_backend`) with
no edits to the graph, the primitives' gradient rules, or the layers.

``Native`` is intentionally ``Any``. For the numpy backend it is
``numpy.ndarray``; for the Metal backend it is a small buffer wrapper. The graph
never inspects it -- it only passes natives back into backend methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

# A backend-specific array. The engine treats it as an opaque handle.
Native = Any


class Backend(ABC):
    """Executes primitive ops on a concrete array type.

    Subclasses implement every method below. The numpy backend is the
    reference implementation (full coverage); other backends may delegate
    unsupported ops back to numpy -- see :class:`MetalBackend`.
    """

    name: str

    # -- interop ---------------------------------------------------------
    @abstractmethod
    def from_numpy(self, array: np.ndarray) -> Native:
        """Move a numpy array into this backend's representation."""

    @abstractmethod
    def to_numpy(self, array: Native) -> np.ndarray:
        """Copy a native array back to host (numpy) memory."""

    # -- elementwise binary ---------------------------------------------
    @abstractmethod
    def add(self, a: Native, b: Native) -> Native: ...
    @abstractmethod
    def sub(self, a: Native, b: Native) -> Native: ...
    @abstractmethod
    def mul(self, a: Native, b: Native) -> Native: ...
    @abstractmethod
    def div(self, a: Native, b: Native) -> Native: ...
    @abstractmethod
    def pow(self, a: Native, b: Native) -> Native: ...
    @abstractmethod
    def maximum(self, a: Native, b: Native) -> Native: ...
    @abstractmethod
    def greater(self, a: Native, b: Native) -> Native:
        """Elementwise ``a > b`` returning a boolean-ish native array."""

    # -- elementwise unary ----------------------------------------------
    @abstractmethod
    def neg(self, a: Native) -> Native: ...
    @abstractmethod
    def exp(self, a: Native) -> Native: ...
    @abstractmethod
    def log(self, a: Native) -> Native: ...
    @abstractmethod
    def sqrt(self, a: Native) -> Native: ...
    @abstractmethod
    def tanh(self, a: Native) -> Native: ...

    # -- reductions ------------------------------------------------------
    @abstractmethod
    def sum(self, a: Native, axis: tuple[int, ...] | None, keepdims: bool) -> Native: ...
    @abstractmethod
    def mean(self, a: Native, axis: tuple[int, ...] | None, keepdims: bool) -> Native: ...
    @abstractmethod
    def max(self, a: Native, axis: tuple[int, ...] | None, keepdims: bool) -> Native: ...

    # -- linear algebra --------------------------------------------------
    @abstractmethod
    def matmul(self, a: Native, b: Native) -> Native: ...

    # -- shape -----------------------------------------------------------
    @abstractmethod
    def reshape(self, a: Native, shape: tuple[int, ...]) -> Native: ...
    @abstractmethod
    def transpose(self, a: Native, axes: tuple[int, ...]) -> Native: ...
    @abstractmethod
    def broadcast_to(self, a: Native, shape: tuple[int, ...]) -> Native: ...

    # -- indexing --------------------------------------------------------
    @abstractmethod
    def gather(self, table: Native, indices: Native) -> Native:
        """Row-gather: ``table[indices]`` along axis 0.

        ``indices`` is an integer array of arbitrary shape ``I``; the result
        has shape ``I + table.shape[1:]``. This is the embedding lookup.
        """

    @abstractmethod
    def scatter_add(
        self, out_shape: tuple[int, ...], indices: Native, updates: Native
    ) -> Native:
        """Inverse of :meth:`gather`: accumulate ``updates`` into a zero array.

        Repeated indices accumulate additively -- exactly the embedding
        backward pass (a token appearing twice gets two gradient contributions).
        """

    # -- structural ------------------------------------------------------
    @abstractmethod
    def concat(self, arrays: list[Native], axis: int) -> Native: ...
    @abstractmethod
    def slice(self, a: Native, start: int, stop: int, axis: int) -> Native:
        """Narrow ``a`` to ``[start:stop]`` along ``axis`` (a contiguous view).

        We use ``slice`` rather than a multi-output ``split`` so every graph
        node has exactly one output -- which keeps the lazy graph and the
        backward pass simple. A QKV split becomes three ``slice`` nodes; the
        backward of ``concat`` is itself a set of ``slice`` ops.
        """

    # -- selection -------------------------------------------------------
    @abstractmethod
    def where(self, cond: Native, a: Native, b: Native) -> Native: ...
