"""Functional reverse-mode autodiff: ``grad`` and ``value_and_grad``.

This is the JAX/MLX style. Instead of ``loss.backward()`` mutating ``.grad``
fields, we have a *transform*: ``grad(f)`` returns a new function that computes
gradients of ``f``. No global state, no in-place mutation -- differentiation is
a pure function of the graph.

How it works (the whole algorithm in four steps):

1. Run ``f`` on the inputs. Because ops are lazy, this *builds the forward
   graph* without computing anything.
2. Seed the output cotangent: ``g_out = dL/dL = 1`` (we use ``ones_like`` so a
   non-scalar output is implicitly summed -- but you should return a scalar).
3. Walk the graph in **reverse topological order**. For each node, call its
   primitive's ``vjp`` to turn the node's cotangent into cotangents for its
   inputs, and *accumulate* (add) them. Accumulation matters: a tensor used in
   two places receives a gradient contribution from each -- the multivariate
   chain rule.
4. The accumulated cotangents for the requested arguments are the gradients.
   They are themselves lazy graphs; we ``eval`` them before returning.

Everything ``vjp`` builds is an ordinary ``Tensor``, so the backward pass is
just more graph -- which is why ``grad(grad(f))`` works without extra machinery.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import overload

from autograd.primitives import add, ones_like, zeros_like
from autograd.tensor import Tensor


def _reverse_topo(root: Tensor) -> list[Tensor]:
    """All tensors reachable from ``root``, inputs-before-outputs.

    Iterative post-order DFS (no recursion -- transformer graphs are deep).
    Reversing this list gives outputs-before-inputs, the order backprop needs.
    """
    order: list[Tensor] = []
    visited: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if id(node) in visited:
            continue
        if expanded:
            visited.add(id(node))
            order.append(node)
            continue
        stack.append((node, True))
        for inp in node.inputs:
            if id(inp) not in visited:
                stack.append((inp, False))
    return order


def _compute_cotangents(output: Tensor, wanted: Sequence[Tensor]) -> list[Tensor]:
    """Reverse-mode sweep; returns the cotangent for each tensor in ``wanted``."""
    cotangents: dict[int, Tensor] = {id(output): ones_like(output)}

    # reversed(post-order) == reverse-topological == process consumers first.
    for node in reversed(_reverse_topo(output)):
        g = cotangents.get(id(node))
        if g is None or node.op is None:
            # No gradient reached this node, or it is a leaf (nothing to push).
            continue
        input_grads = node.op.vjp(g, node, *node.inputs)
        for inp, ig in zip(node.inputs, input_grads, strict=True):
            if ig is None:
                continue  # non-differentiable input (e.g. integer indices)
            existing = cotangents.get(id(inp))
            cotangents[id(inp)] = ig if existing is None else add(existing, ig)

    return [cotangents.get(id(t)) or zeros_like(t) for t in wanted]


def _normalise_argnums(argnums: int | Sequence[int]) -> tuple[tuple[int, ...], bool]:
    """Return (tuple of arg indices, is_single)."""
    if isinstance(argnums, int):
        return (argnums,), True
    return tuple(argnums), False


@overload
def value_and_grad(
    fn: Callable[..., Tensor], argnums: int = 0
) -> Callable[..., tuple[Tensor, Tensor]]: ...
@overload
def value_and_grad(
    fn: Callable[..., Tensor], argnums: Sequence[int]
) -> Callable[..., tuple[Tensor, tuple[Tensor, ...]]]: ...
def value_and_grad(
    fn: Callable[..., Tensor], argnums: int | Sequence[int] = 0
) -> Callable[..., tuple[Tensor, Tensor | tuple[Tensor, ...]]]:
    """Transform ``fn`` into one returning ``(value, grad(s))``.

    ``argnums`` selects which positional arguments to differentiate w.r.t.
    (an int for one argument, a tuple for several). ``fn`` should return a
    scalar ``Tensor``.
    """
    indices, single = _normalise_argnums(argnums)

    def wrapped(*args: Tensor, **kwargs):  # noqa: ANN002, ANN003
        value = fn(*args, **kwargs)
        wanted = [args[i] for i in indices]
        grads = _compute_cotangents(value, wanted)
        # Force evaluation so callers get concrete results, not pending graphs.
        value.eval()
        for g in grads:
            g.eval()
        return value, (grads[0] if single else tuple(grads))

    return wrapped


@overload
def grad(fn: Callable[..., Tensor], argnums: int = 0) -> Callable[..., Tensor]: ...
@overload
def grad(
    fn: Callable[..., Tensor], argnums: Sequence[int]
) -> Callable[..., tuple[Tensor, ...]]: ...
def grad(
    fn: Callable[..., Tensor], argnums: int | Sequence[int] = 0
) -> Callable[..., Tensor | tuple[Tensor, ...]]:
    """Transform ``fn`` into one returning just its gradient(s).

    Thin wrapper over :func:`value_and_grad` -- the two share all the work; this
    one simply discards the value.
    """
    vg = value_and_grad(fn, argnums)

    def wrapped(*args: Tensor, **kwargs):  # noqa: ANN002, ANN003
        _, grads = vg(*args, **kwargs)
        return grads

    return wrapped
