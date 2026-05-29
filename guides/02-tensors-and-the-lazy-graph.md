# 02 - Tensors and the Lazy Computation Graph

**Code:** `src/autograd/tensor.py`, `src/autograd/graph.py`

This is the foundation. Everything else -- primitives, autodiff, backends, the
GPT-2 model -- is built on the idea in this chapter: a `Tensor` is not a number,
it is a *node in a graph that knows how to become a number*.

## Theory: deferred evaluation

A normal value is eager: `3 + 4` is `7` right now. A *thunk* is a deferred
computation: "add 3 and 4, but not yet." A lazy tensor is a thunk that also
remembers its shape and how it was produced.

When you write `c = a + b`, our engine does **not** add anything. It creates a
new `Tensor` whose recipe is "I am `Add` applied to `a` and `b`," with a cached
value of `None`. The numbers appear only when you call `eval()`, `numpy()`, or
`item()`.

Why do this? Three reasons, in increasing order of importance:

1. **Memoization** of shared sub-expressions (compute once, use many times).
2. **Whole-graph rewriting** before execution -- the basis of JIT/fusion ([10](10-jit-and-fusion.md)).
3. **Differentiation as a graph transform** -- the backward pass is just more
   nodes ([04](04-functional-autodiff.md)).

## The `Tensor` node

A node carries exactly what is needed to (a) know its shape without computing,
and (b) compute itself when asked:

```python
class Tensor:
    __slots__ = ("shape", "dtype", "op", "inputs", "backend", "_data")
    # op:     the Primitive that produces this node (None for a leaf)
    # inputs: parent Tensors this node consumes
    # _data:  cached native array; None means "not evaluated yet"
```

Two kinds of node:

- **Leaf** (`op is None`): data the user supplied via `tensor(...)`. Its `_data`
  is set at construction.
- **Derived** (`op` set): the output of a primitive. Its `_data` is `None` until
  evaluation fills it in.

> [!important] Shape is static, value is lazy
> Notice the shape is known at construction time, *before* any data exists. This
> is essential: to build the **backward** graph we need shapes (e.g. to make a
> "ones" seed), and we must do that without forcing a forward computation.

### Operators build nodes

The arithmetic operators are defined on `Tensor` and dispatch into the
functional ops in `primitives.py`:

```python
def __add__(self, other):
    from autograd import primitives as P   # lazy import avoids a cycle
    return P.add(self, other)
```

The lazy import is a deliberate pattern: `primitives` imports `Tensor`, so a
top-level import here would be circular. Python caches modules, so the per-call
import is just a dictionary lookup. (PyTorch attaches its tensor methods through
a similar bootstrap.)

## The evaluation engine

`graph.py` turns a recipe into numbers in two steps.

### Step 1: topological sort

We collect every not-yet-evaluated ancestor of the target, ordered so that
inputs come before the nodes that consume them:

```python
def topological_sort(root):
    # iterative post-order DFS (no recursion -- transformer graphs are deep)
    ...
```

Two design choices worth noting:

- **Iterative, not recursive.** A 12-layer transformer graph is hundreds of
  nodes deep; Python's recursion limit (~1000) would be at risk. The explicit
  stack sidesteps it.
- **Already-evaluated nodes are boundaries.** If a node's `_data` is set (a leaf,
  or a previously computed result), we stop there. This gives us free
  memoization: re-evaluating a tensor is a no-op.

### Step 2: execute

```python
def evaluate(root):
    for node in topological_sort(root):
        if node._data is not None:
            continue
        input_data = [inp._data for inp in node.inputs]
        node._data = node.op.forward(node.backend, *input_data)
```

Each node asks its primitive to run `forward` on the chosen backend, passing the
already-materialized parent arrays. After this loop the root holds its value.

## Try it

```python
from autograd import tensor
import autograd.primitives as P

a = tensor([[1., 2.], [3., 4.]])
c = P.exp(a * a)          # builds a graph; nothing computed
print(c._data)            # None
print(c.numpy())          # forces eval -> the array
print(c._data is not None) # True (now cached)
```

## Design patterns in this chapter

- **Thunk / lazy evaluation**: a value that defers its computation.
- **Interpreter over an IR**: the graph is a tiny intermediate representation;
  `evaluate` is its interpreter.
- **Memoization**: cache results on the node; shared sub-graphs compute once.
- **Bootstrap to break a cycle**: operators defined via lazy import.

## What's next

We have nodes and an interpreter, but the nodes are empty shells until each
`op` knows how to compute and differentiate itself. That is
[03 - Primitives and the VJP](03-primitives-and-vjp.md).
