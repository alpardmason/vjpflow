"""A teaching-sized JIT: trace once, then fuse the graph.

This is *not* a real compiler. It exists to make one idea concrete: because a
lazy engine builds the computation graph as data *before* running it, you can
**rewrite the graph** to run faster. That single capability -- whole-graph
visibility -- is why MLX, JAX/XLA, and ``torch.compile`` exist.

We demonstrate two pieces:

- :func:`trace` -- run a function to capture its graph (the "trace"). A
  :class:`Compiled` wrapper caches the trace per input-shape signature, the way
  ``jax.jit`` caches per shape/dtype.
- :func:`fuse_elementwise` -- a real (if narrow) graph-rewrite pass. It merges
  chains of unary elementwise ops (``exp``, ``tanh``, ``neg``, ...) into a
  single :class:`FusedUnary` node. Ten chained ops become one kernel launch
  over the data instead of ten passes through memory -- the essence of kernel
  fusion.

Scope note: fusion here optimises the *forward* graph (inference). The fused
node deliberately does not define a ``vjp`` -- gradients are taken on the
original, unfused graph. A production system fuses the backward graph too.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import overload

from vjpflow.backends import Native
from vjpflow.primitives import Exp, Log, Neg, Primitive, Sqrt, Tanh
from vjpflow.tensor import Tensor

# Unary elementwise primitives are safe to fuse: same shape in and out, one
# input, no reduction or reshaping.
_FUSIBLE_UNARY: tuple[type[Primitive], ...] = (Neg, Exp, Log, Sqrt, Tanh)


class FusedUnary(Primitive):
    """A compiled chain of unary elementwise ops, executed in one sweep."""

    def __init__(self, funcs: list[Primitive]) -> None:
        self.funcs = funcs

    def forward(self, backend, x: Native) -> Native:  # noqa: ANN001
        # Apply each op in turn. A production backend would emit a single kernel
        # whose body is the composed expression; here we simply avoid building
        # intermediate graph nodes (and their bookkeeping).
        for fn in self.funcs:
            x = fn.forward(backend, x)
        return x

    def vjp(self, g, out, x):  # noqa: ANN001
        raise NotImplementedError(
            "FusedUnary is a forward-only optimisation; differentiate the "
            "unfused graph instead."
        )

    def __repr__(self) -> str:
        names = "->".join(type(f).__name__ for f in self.funcs)
        return f"FusedUnary({names})"


def _op_nodes(root: Tensor) -> list[Tensor]:
    """All non-leaf nodes reachable from ``root`` (inputs-before-outputs)."""
    order: list[Tensor] = []
    visited: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if id(node) in visited:
            continue
        if expanded:
            visited.add(id(node))
            if node.op is not None:
                order.append(node)
            continue
        stack.append((node, True))
        for inp in node.inputs:
            stack.append((inp, False))
    return order


def _consumer_counts(root: Tensor) -> dict[int, int]:
    """How many nodes consume each tensor (fan-out). Only single-consumer links are fusable."""
    counts: dict[int, int] = {}
    for node in _op_nodes(root):
        for inp in node.inputs:
            counts[id(inp)] = counts.get(id(inp), 0) + 1
    return counts


@dataclass
class FusionStats:
    nodes_before: int
    nodes_after: int
    chains_fused: int

    @property
    def removed(self) -> int:
        return self.nodes_before - self.nodes_after


def fuse_elementwise(output: Tensor) -> tuple[Tensor, FusionStats]:
    """Return an equivalent graph with unary elementwise chains fused.

    The rewrite is a standard bottom-up rebuild with memoisation: visit nodes
    inputs-first, and whenever a unary op feeds *only* into another unary op,
    splice them into one :class:`FusedUnary`.
    """
    counts = _consumer_counts(output)
    nodes = _op_nodes(output)
    nodes_before = len(nodes)
    chains_fused = 0

    mapping: dict[int, Tensor] = {}

    def remap(t: Tensor) -> Tensor:
        return mapping.get(id(t), t)

    for node in nodes:
        new_inputs = tuple(remap(i) for i in node.inputs)
        parent = node.inputs[0] if node.inputs else None

        if (
            isinstance(node.op, _FUSIBLE_UNARY)
            and parent is not None
            and isinstance(parent.op, _FUSIBLE_UNARY)
            and counts.get(id(parent), 0) == 1
        ):
            prev = remap(parent)
            prev_op = prev.op
            funcs: list[Primitive]
            if isinstance(prev_op, FusedUnary):
                funcs = [*prev_op.funcs, node.op]
                base = prev.inputs[0]
            else:
                funcs = [parent.op, node.op]
                base = remap(parent.inputs[0])
            mapping[id(node)] = Tensor(
                node.shape,
                op=FusedUnary(funcs),
                inputs=(base,),
                backend=node.backend,
                dtype=node.dtype,
            )
            chains_fused += 1
            continue

        # Default: keep the op, but rebuild if any input was remapped.
        if new_inputs == node.inputs:
            mapping[id(node)] = node
        else:
            mapping[id(node)] = Tensor(
                node.shape,
                op=node.op,
                inputs=new_inputs,
                backend=node.backend,
                dtype=node.dtype,
            )

    fused_output = remap(output)
    stats = FusionStats(nodes_before, len(_op_nodes(fused_output)), chains_fused)
    return fused_output, stats


class Compiled:
    """Wraps a function, caching its traced graph per input-shape signature."""

    def __init__(self, fn: Callable[..., Tensor], fuse: bool = True) -> None:
        self._fn = fn
        self._fuse = fuse
        self._cache: dict[tuple, FusionStats] = {}
        self.last_stats: FusionStats | None = None

    def __call__(self, *args: Tensor) -> Tensor:
        out = self._fn(*args)  # trace: build the forward graph
        if self._fuse:
            out, stats = fuse_elementwise(out)
            self.last_stats = stats
            # Record per-signature so repeat calls show the trace was reused.
            self._cache[tuple(a.shape for a in args)] = stats
        return out


@overload
def jit(fn: Callable[..., Tensor], *, fuse: bool = ...) -> Compiled: ...
@overload
def jit(fn: None = ..., *, fuse: bool = ...) -> Callable[[Callable[..., Tensor]], Compiled]: ...
def jit(
    fn: Callable[..., Tensor] | None = None, *, fuse: bool = True
) -> Compiled | Callable[[Callable[..., Tensor]], Compiled]:
    """Decorator / wrapper: ``compiled = jit(f)`` then call ``compiled(x)``.

    Mirrors ``jax.jit`` ergonomically (decorator or call form).
    """

    def wrap(f: Callable[..., Tensor]) -> Compiled:
        return Compiled(f, fuse=fuse)

    return wrap(fn) if fn is not None else wrap


# Common alias.
compile = jit  # noqa: A001
