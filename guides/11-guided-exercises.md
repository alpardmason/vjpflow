# 11 - Guided Exercises

The engine's ~20 primitives can express every layer in `notes/`, not just the
GPT-2 path. These exercises ask you to implement the rest. For each, the
*derivation already exists* in the corresponding note -- your job is the forward
pass; the engine produces the backward, and you verify with
`tests/util.check_grad`.

Two layers are already implemented as worked examples to copy the style from:
`rmsnorm` and `silu`/`sigmoid` in `src/vjpflow/nn/functional.py`.

> Workflow for every exercise:
> 1. Read the note for the math and shapes.
> 2. Write a forward function in `nn/functional.py` using only `vjpflow.primitives`.
> 3. Add a `check_grad` test (finite-difference) -- if it passes, your layer is
>    almost certainly correct, backward included.

## Tier 1 -- elementwise & norms (warm-up)

### RMSNorm (done -- study it)
`notes/RMS Normalization Back Propagation.md`. Already in the codebase. Compare
its 2-line forward to the note's backward; note there is no mean-centering term.

### BatchNorm (inference + training)
`notes/Batch Normalization Back Propagation.md`. Normalize over the *batch* axis
instead of the feature axis. Primitives: `mean`/`sub`/`mul`/`sqrt` over axis 0.
Subtlety: training uses batch stats (differentiable); inference uses running
stats (constants). Implement the training path and check gradients.

### GLU variants: SwiGLU / GeGLU / ReGLU
`notes/GLU Variants Back Propagation.md`. `GLU(x) = (xW + b) * act(xV + c)`.
Primitives: two `linear`s and an elementwise `mul` with `silu`/`gelu`/`relu`.
Hint: the gate is a `slice` of a single fused projection, like QKV.

## Tier 2 -- attention variants

### Multi-Query Attention (MQA)
`notes/Multi-Query Attention Back Propagation.md`. All query heads share *one* K
and one V. Start from `attention` in `functional.py`; project Q to `H` heads but
K, V to a single head, and broadcast K/V across heads in the batched matmul.

### Grouped-Query Attention (GQA)
`notes/Grouped-Query Attention Back Propagation.md`. The general case: `G` KV
groups, each shared by `H/G` query heads. MHA (`G=H`) and MQA (`G=1`) are the
endpoints. Hint: `reshape` Q to `(G, H/G, S, d_head)` and broadcast KV per group.

### Cross-Attention
`notes/Cross-Attention Back Propagation.md`. Q comes from the decoder, K/V from
the encoder output. Same machinery as self-attention but two inputs; no causal
mask. Gradients flow to *both* the decoder and encoder activations.

## Tier 3 -- positional & routing

### Rotary Position Embedding (RoPE)
`notes/Rotary Position Embedding Back Propagation.md`. Rotate Q and K by
position-dependent angles before the score. Precompute `cos`/`sin` tables as
constants; apply with `slice` + `concat` + elementwise `mul`/`add`. Since the
rotation is orthogonal, gradients flow cleanly -- a nice property to verify.

### Absolute (sinusoidal & learned) positional encoding
`notes/Absolute Positional Encoding Back Propagation.md`. Learned is already in
GPT-2 (`wpe`). Add the fixed sinusoidal variant as a constant tensor (no
gradient) and confirm `check_grad` reports zero gradient to it.

### Mixture of Experts (MoE)
`notes/Mixture of Experts Back Propagation.md`. A router scores experts; top-k
are selected; outputs are combined by gate weights. Primitives: `matmul` (router
+ experts), `softmax` (gates), `gather`/`scatter_add` (dispatch/combine).
Start dense (all experts, soft weights) to keep it differentiable, then discuss
why hard top-k routing needs a straight-through estimator.

## Tier 4 -- systems

### Dropout
`notes/Dropout Back Propagation.md`. Use `where` with a precomputed Bernoulli
mask and inverted scaling (`1/(1-p)`). The mask is a constant per forward; check
that the gradient is the mask times the upstream gradient.

### KV cache (inference)
`notes/KV Cache Back Propagation.md`. An inference optimization: cache past K/V
and only compute the new token's attention. Mostly a forward-path concern; the
note contrasts training vs inference gradient flow. Implement a single-step
decode and discuss why the cache carries no gradient.

### Backward fusion (extends [10](10-jit-and-fusion.md))
`jit.FusedUnary` deliberately has no `vjp`. Give it one: store the chain and, in
`vjp`, rebuild the unfused sub-graph to backprop through it. Then fusion would
help training, not just inference.

### A real Metal kernel for a reduction
`metal_backend.py` falls back to numpy for non-last-axis reductions. Write a
Metal kernel that reduces an arbitrary axis (hint: reshape to `(outer, axis,
inner)` and one thread per `(outer, inner)`), and extend the parity tests.

## How to check your work

```python
from tests.util import check_grad
check_grad(lambda x, w: your_layer(x, w).sum(), [x_np, w_np], argnum=1)
```

If the analytic gradient matches the numerical one, you have correctly
implemented a layer *and* its entire backward pass -- by writing only the forward.
That is the whole promise of an autograd engine, and now you have built the
machine that delivers it.

## Where to go beyond this repo

- **Forward-mode AD (JVP)** and how it pairs with reverse-mode for Hessians.
- **`vmap`**: automatic batching as another graph transform.
- **Real fusion/scheduling**: read MLX's lazy eval and JAX's XLA lowering.
- **Mixed precision & dtype promotion**, which we omitted for clarity.
