# 05 - The Backend Abstraction and the numpy Reference

**Code:** `src/vjpflow/backends/base.py`, `numpy_backend.py`, `__init__.py`

The graph and autodiff never touch raw arrays directly -- they go through a
**backend**. This chapter is short because numpy makes it short, but the
abstraction it introduces is what lets the *same* GPT-2 run on a CPU or an Apple
GPU with a one-line switch.

## Theory: separate "what" from "where"

A primitive's `forward` says *what* to compute ("multiply these two arrays"); a
backend says *where* and *how* ("numpy's elementwise multiply" vs "a Metal
kernel"). This is the **Strategy pattern**: the graph holds a backend object and
delegates the how-to.

```python
class Add(Primitive):
    def forward(self, backend, a, b):
        return backend.mul(a, b)   # the graph does not know or care which backend
```

The backend operates on an opaque `Native` type:
- numpy backend: `Native = numpy.ndarray`
- Metal backend: `Native = MetalArray` (a GPU buffer + shape)

The graph never inspects a `Native`; it only passes them back into backend
methods. That opacity is what makes backends swappable.

## The protocol

`base.py` is an abstract base class listing every operation a backend must
provide: interop (`from_numpy`/`to_numpy`), elementwise binary/unary, reductions,
`matmul`, shape ops (`reshape`/`transpose`/`broadcast_to`/`slice`), indexing
(`gather`/`scatter_add`), `concat`, and `where`. Roughly twenty methods -- one per
primitive.

> [!note] Why `slice` and not `split`?
> A `split` would produce several outputs from one node, complicating the graph
> and the backward pass. We use single-output `slice` instead: a QKV split is
> three `slice`s, and `concat`'s backward is a set of `slice`s. Every graph node
> has exactly one output.

## The numpy backend

Each method is a one-liner over numpy. This file is the **reference
implementation** -- full coverage (all primitives, all layers, GPT-2) and the
ground truth the Metal backend is tested against. Two details are worth a look.

### Float32 normalization

```python
def from_numpy(self, array):
    if np.issubdtype(array.dtype, np.floating):
        return np.ascontiguousarray(array, dtype=np.float32)
    return np.ascontiguousarray(array)   # keep int index arrays as-is
```

We standardize floating data to float32 (matching PyTorch/MLX defaults) but
preserve integer arrays so token ids and indices keep their type for `gather`.

### Scatter-add must accumulate

```python
def scatter_add(self, out_shape, indices, updates):
    out = np.zeros(out_shape, dtype=updates.dtype)
    np.add.at(out, indices, updates)   # UNBUFFERED: repeats accumulate
    return out
```

`np.add.at` is subtle but essential: plain `out[indices] += updates` would
*overwrite* on repeated indices, not add. The embedding backward depends on
accumulation (a token appearing twice gets two gradient contributions) -- see
`notes/Embedding Back Propagation.md`.

## The registry: switching backends

`backends/__init__.py` keeps one cached instance per backend and a process-wide
default:

```python
from vjpflow import set_default_backend
set_default_backend("metal")   # every new leaf tensor now lives on the GPU
```

Backends are cached because they are stateful (the Metal one holds a GPU device
and a compiled-kernel cache); rebuilding per call would be wasteful. The Metal
backend is imported lazily inside `get_backend`, so simply importing the package
never requires PyObjC or a GPU.

## Design patterns in this chapter

- **Strategy**: pluggable execution backends behind one interface.
- **Opaque handle**: the graph treats `Native` as a black box.
- **Reference implementation as test oracle**: numpy defines "correct."
- **Lazy singleton registry**: construct heavy backends once, on demand.

## What's next

Time to write a real GPU backend. First the concepts, since Metal may be new:
[06 - Introduction to Metal](06-intro-to-metal.md).
