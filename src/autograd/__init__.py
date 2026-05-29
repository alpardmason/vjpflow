"""A tiny, educational autograd engine.

This package implements a *lazy* (define-by-run-but-deferred) computation graph
together with *functional* reverse-mode automatic differentiation -- the same
mental model used by MLX and JAX. It is deliberately small: around twenty
primitive operations are enough to express every layer in a GPT-2 model.

Read the ``guides/`` folder alongside the source; each guide maps onto one
module here.

Public API
----------
- :class:`~autograd.tensor.Tensor` -- a node in the lazy graph.
- :func:`~autograd.tensor.tensor` -- build a leaf tensor from data.
- :func:`~autograd.autodiff.grad` / :func:`~autograd.autodiff.value_and_grad`
  -- functional differentiation transforms.
- The functional ops (``add``, ``matmul``, ``softmax`` ...) live in
  :mod:`autograd.primitives` and :mod:`autograd.nn.functional`.
"""

from __future__ import annotations

from autograd import primitives as primitives  # noqa: F401  (registers Tensor operators)
from autograd.autodiff import grad, value_and_grad
from autograd.backends import (
    Backend,
    default_backend,
    get_backend,
    set_default_backend,
)
from autograd.tensor import Tensor, tensor

__all__ = [
    "Tensor",
    "tensor",
    "grad",
    "value_and_grad",
    "Backend",
    "default_backend",
    "get_backend",
    "set_default_backend",
]
