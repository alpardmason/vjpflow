# 03 - Primitives and the Vector-Jacobian Product

**Code:** `src/autograd/primitives.py` -- **Math:** `notes/Back Propagation.md`

A primitive is the atom of the engine. Each one knows two things: how to compute
itself (`forward`) and how to push a gradient back through itself (`vjp`).
Everything else -- LayerNorm, attention, GPT-2 -- is a *composition* of these
atoms, and gets its backward pass for free.

## Theory: why VJP, not "the Jacobian"

Consider an op `y = f(x)` where `x in R^n`, `y in R^m`. Its derivative is the
Jacobian `J in R^{m x n}`. For a scalar loss `L`, the chain rule says

```
G_x = J^T @ G_y          (denominator layout: G_x has the same shape as x)
```

where `G_y = dL/dy`. The crucial observation: **we never need `J` itself, only
the product `J^T @ G_y`.** That product is the *vector-Jacobian product* (VJP).

This matters enormously. For softmax over a length-`d` vector, `J` is `d x d`
(dense). Materializing it for `d = 50257` (GPT-2's vocab) is absurd. But the VJP
`J^T g` has a closed form -- `a * (g - sum(a*g))` -- that costs `O(d)`. Every
`vjp` in this file is such a closed form. This is the single most important
efficiency idea in autodiff.

> The matrix-form derivations in `notes/` are exactly these VJPs, written out by
> hand. Reading a note and its corresponding `vjp` side by side is the best way
> to internalize the chapter.

## Anatomy of a primitive

```python
class Mul(Primitive):
    def forward(self, backend, a, b):
        return backend.mul(a, b)               # what to compute

    def vjp(self, g, out, a, b):               # dL/da, dL/db given g = dL/d(out)
        return sum_to_shape(mul(g, b), a.shape), sum_to_shape(mul(g, a), b.shape)
```

Read `vjp` carefully. For `out = a * b`:
- `dL/da = g * b` (product rule), `dL/db = g * a`.
- It returns these as **graph nodes** (it calls `mul`, which builds new
  `Tensor`s), not as evaluated arrays. This is what makes the backward pass lazy
  and differentiable again (higher-order grad).
- `sum_to_shape` repairs broadcasting -- see below.

A `vjp` returns `None` for any input that is not differentiable (e.g. integer
indices in `Gather`). The autodiff loop skips those.

## The broadcasting trick: `sum_to_shape`

Broadcasting is where most hand-rolled autograd engines have bugs. If
`a` is `(n, d)` and `b` is `(d,)`, then `a + b` stretches `b` across `n` rows.
The forward is easy; the backward must **sum the gradient back over the
stretched axis**, because each element of `b` influenced `n` outputs.

```python
def sum_to_shape(g, shape):
    # 1. sum away extra leading dims broadcasting prepended
    while len(g.shape) > len(shape):
        g = sum(g, axis=0)
    # 2. sum (keepdims) over dims that were size-1 in the target
    axes = tuple(i for i,(gs,ss) in enumerate(zip(g.shape, shape)) if ss==1 and gs!=1)
    if axes:
        g = sum(g, axis=axes, keepdims=True)
    return g if g.shape == shape else reshape(g, shape)
```

Every binary op's `vjp` funnels its result through `sum_to_shape`. PyTorch and
MLX do precisely this internally; here it is one readable function.

## The primitive set (the "MVP")

About twenty primitives express every layer in `notes/`:

| Group | Ops | Used by |
|---|---|---|
| Elementwise binary | `add sub mul div pow maximum` | everything; ReLU = `maximum(x,0)` |
| Elementwise unary | `neg exp log sqrt tanh` | GELU, softmax, norms |
| Reductions | `sum mean max` | softmax, LayerNorm, loss |
| Linear algebra | `matmul` | Linear, attention |
| Shape | `reshape transpose broadcast_to slice concat` | head split/merge, QKV |
| Indexing | `gather scatter_add` | embedding, MoE |
| Selection | `where`, `greater` | masks, dropout |

A few highlights to read:

- **`MatMul.vjp`** implements the fundamental identity from the notes:
  `G_A = G @ B^T`, `G_B = A^T @ G`. With batched inputs it transposes only the
  last two axes and uses `sum_to_shape` to collapse broadcast batch dims.
- **`Gather` / `ScatterAdd`** are inverses, and each is the *other's* vjp. That
  symmetry is why a repeated token id correctly accumulates gradients (the
  embedding backward in `notes/Embedding Back Propagation.md`).
- **`Slice.vjp`** pads the gradient back to the input shape using `concat` with
  zero blocks -- no new backend op needed. `Concat.vjp` does the reverse,
  slicing the gradient into per-input pieces. Single-output nodes throughout
  keeps the graph simple.
- **`Max.vjp`** routes the gradient to the maximal element(s) via a mask; ties
  give the standard ReLU subgradient when paired with `maximum(x, 0)`.

## Numerical stability lives here too

The unary `vjp`s reuse the forward *output* where it avoids recomputation and
improves stability: `Exp.vjp` returns `g * out` (since `d/dx e^x = e^x = out`);
`Tanh.vjp` returns `g * (1 - out^2)`. The log-sum-exp trick that keeps softmax
finite is built one layer up, in `nn/functional.py` ([08](08-building-layers.md)).

## Verify it yourself

The contract for any `vjp` is "match a finite-difference estimate." That is
exactly `tests/util.check_grad`, used throughout `tests/test_primitives.py`:

```python
check_grad(lambda a, b: (a / b).sum(), [a, b], argnum=1)  # passes => vjp correct
```

## Design patterns in this chapter

- **Strategy via a base class**: every op subclasses `Primitive` and supplies
  `forward` + `vjp`.
- **VJP / reverse-mode AD**: never form the Jacobian; compute `J^T g` directly.
- **Closed-form local gradients** for stability and speed.
- **Inverse-pair ops** (`gather`/`scatter_add`) that are each other's adjoint.

## What's next

We have ops that can differentiate themselves *locally*. Stitching those local
rules into a full backward pass over an arbitrary graph is the job of
[04 - Functional Autodiff](04-functional-autodiff.md).
