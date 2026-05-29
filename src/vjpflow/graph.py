"""The evaluation engine: turning a lazy graph into numbers.

A :class:`~vjpflow.tensor.Tensor` is a recipe. :func:`evaluate` cooks it:

1. **Topological sort** of all not-yet-evaluated ancestors, so every node is
   visited only after its inputs.
2. **Execute** each node's primitive ``forward`` on its backend, caching the
   result on the node (``_data``).

Caching gives us *memoisation*: a shared sub-expression (e.g. an activation fed
into two branches) is computed once. Re-evaluating a tensor whose ``_data`` is
already set is free.

The sort is iterative (an explicit stack) rather than recursive so that deep
graphs -- a 12-layer transformer is hundreds of nodes deep -- never blow the
Python recursion limit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vjpflow.tensor import Tensor


def topological_sort(root: Tensor) -> list[Tensor]:
    """Return ancestors of ``root`` in dependency order (inputs before outputs).

    Only nodes that still need evaluating are included; anything already
    materialised (a leaf, or a cached result) is treated as a boundary and not
    revisited.
    """
    order: list[Tensor] = []
    visited: set[int] = set()

    # Iterative post-order DFS. Each stack frame is (node, children_pushed?).
    stack: list[tuple[Tensor, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if node._data is not None or id(node) in visited:
            # Already computed (boundary) or already scheduled.
            continue
        if expanded:
            # Second visit: all inputs are now on `order` before us.
            visited.add(id(node))
            order.append(node)
            continue
        # First visit: re-push ourselves to run *after* our inputs, then push
        # inputs. Inputs whose data exists are skipped at the top of the loop.
        stack.append((node, True))
        for inp in node.inputs:
            if inp._data is None and id(inp) not in visited:
                stack.append((inp, False))
    return order


def evaluate(root: Tensor) -> None:
    """Materialise ``root`` in place, computing any missing ancestors."""
    for node in topological_sort(root):
        if node._data is not None:  # may have been filled by a shared path
            continue
        assert node.op is not None, "non-leaf node without an op"
        input_data = [inp._data for inp in node.inputs]
        node._data = node.op.forward(node.backend, *input_data)
