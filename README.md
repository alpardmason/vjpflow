# autograd — a tiny lazy autograd engine

An educational reverse-mode automatic-differentiation engine built the way MLX
and JAX build theirs: a **lazy computation graph** plus a **functional `grad`
transform**. It is small enough to read in an afternoon (~20 primitive ops) yet
complete enough to run a GPT-2 forward and backward pass.

It is meant to be read *alongside* the `guides/` folder (a numbered reading
sequence) and the matrix-calculus derivations in `notes/`.

## Why this exists

If you already use PyTorch or MLX and want to know what actually happens when
you call `.backward()` or `mx.grad(f)`, this repo takes the lid off:

- how an operation becomes a **node in a graph** instead of running immediately;
- how `grad(f)` turns a forward graph into a **backward graph** (the vector–Jacobian
  product, op by op);
- how the same graph runs on **numpy (CPU)** or **hand-written Metal kernels (Apple GPU)**;
- what **JIT / kernel fusion** buys you, demonstrated on a real (if tiny) graph.

## Install

Uses [`uv`](https://docs.astral.sh/uv/). Python 3.12.

```bash
uv venv --python 3.12
uv sync                 # core engine (numpy backend)
uv sync --extra metal   # add the Apple-Silicon Metal backend (macOS only)
```

## Quick taste

```python
import numpy as np
from autograd import tensor, value_and_grad

def loss(W, x, y):
    pred = x @ W            # builds a graph node; nothing runs yet
    err = pred - y
    return (err * err).mean()

W = tensor(np.random.randn(3, 1))
x = tensor(np.random.randn(8, 3))
y = tensor(np.random.randn(8, 1))

L, (gW,) = value_and_grad(loss, argnums=(0,))(W, x, y)
print(L.item(), gW.numpy().shape)   # eval() is forced here
```

## Layout

- `src/autograd/tensor.py` — the lazy `Tensor` node
- `src/autograd/graph.py` — topological sort + evaluation engine
- `src/autograd/primitives.py` — the primitive ops (`forward` + `vjp`)
- `src/autograd/autodiff.py` — `grad` / `value_and_grad`
- `src/autograd/backends/` — numpy reference + raw Metal (PyObjC) backends
- `src/autograd/nn/functional.py` — layers composed from primitives
- `src/autograd/models/gpt2.py` — the end-to-end capstone
- `src/autograd/jit.py` — trace/cache + a fusion pass demo
- `guides/` — the reading companion (start at `00`)

## Tests

```bash
uv run pytest          # gradient checks, backend parity, layer + GPT-2 smoke
uv run ruff check .
uv run pyright
```
