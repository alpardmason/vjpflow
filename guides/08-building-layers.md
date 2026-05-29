# 08 - Building Layers from Primitives

**Code:** `src/vjpflow/nn/functional.py` -- **Math:** the layer notes in `notes/`

Here is the payoff of an autograd engine: we write each layer as a **forward
pass only**, composed from the primitives of [03](03-primitives-and-vjp.md), and
the engine differentiates it for us. Not a single line of backward code appears
in this file. Read each layer beside its `notes/` derivation and confirm the
math the engine *re-derives automatically* matches the hand derivation.

The functions are **functional**: parameters are passed in explicitly, no hidden
state. Shapes follow the `notes/` row-vector convention: a sequence is `X` with
shape `(S, d)`, and weights act on the right (`Y = X W`).

## Linear -- `notes/Linear Back Propagation.md`

```python
def linear(x, weight, bias=None):
    y = P.matmul(x, weight)
    return P.add(y, bias) if bias is not None else y
```

The note derives `G_X = G_Y W^T`, `G_W = X^T G_Y`, `G_b = 1^T G_Y`. Those come
out of `MatMul.vjp` (the first two) and the broadcast-aware `Add.vjp` plus
`sum_to_shape` (the bias sum over rows). You wrote `x @ W + b`; you got three
gradients.

## Embedding -- `notes/Embedding Back Propagation.md`

```python
def embedding(table, indices):
    return P.gather(table, np.asarray(indices))
```

Forward is a row gather; backward is the scatter-add built into `Gather.vjp`. A
token id appearing twice in the sequence accumulates two gradient rows -- exactly
the note's "scatter-add" backward.

## Activations -- ReLU / GELU / SiLU / sigmoid

```python
def relu(x):  return P.maximum(x, zeros)
def gelu(x):  # tanh-approx, GPT-2 variant
    inner = sqrt(2/pi) * (x + 0.044715 * x*x*x)
    return 0.5 * x * (1.0 + P.tanh(inner))
```

GELU's derivative (`notes/GELU Back Propagation.md`) is a mouthful by hand. Here
it is just the chain rule through `mul`, `add`, `tanh`, `pow` -- the engine
assembles it. `test_gelu_grad` confirms it against finite differences.

## Softmax & the log-sum-exp trick -- `notes/Softmax Back Propagation.md`

```python
def softmax(x, axis=-1):
    m = P.max(x, axis=axis, keepdims=True)
    e = P.exp(x - m)
    return e / P.sum(e, axis=axis, keepdims=True)
```

Subtracting `max(x)` prevents `exp` overflow. Softmax is *shift-invariant*, so
the value is unchanged -- and because the shift cancels mathematically,
differentiating straight through `max` yields the exact softmax Jacobian-vector
product `a * (g - sum(a*g))` from the note. No need to stop the gradient.

## Cross-entropy -- `notes/Cross-Entropy Loss Back Propagation.md`

```python
def cross_entropy(logits, targets):
    logp = log_softmax(logits, axis=-1)        # stable log-sum-exp
    onehot = one_hot(targets)                  # constant
    nll = -(logp * onehot).sum(axis=-1)
    return nll.mean()
```

The famous gradient `G_logits = (P - Y_onehot)/N` is *not* coded. It emerges from
differentiating `log_softmax` (which contains the softmax) times the one-hot
mask. `test_cross_entropy_grad_and_value` checks both the gradient and that a
uniform input gives loss `log(V)`.

## LayerNorm -- `notes/Layer Normalization Back Propagation.md`

```python
def layernorm(x, gamma, beta, eps=1e-5):
    mu = P.mean(x, -1, keepdims=True)
    centered = x - mu
    var = P.mean(centered*centered, -1, keepdims=True)
    x_hat = centered * (1.0 / P.sqrt(var + eps))
    return x_hat * gamma + beta
```

This is the showcase. The note's boxed `G_X` -- the `O(d)` mean/variance Jacobian
with its three terms `(G_xhat - mean(G_xhat) - x_hat * mean(G_xhat * x_hat))/sigma`
-- is genuinely tricky to derive and implement by hand. We never do. We write the
five-line forward and the engine produces an algebraically equivalent backward.
`test_layernorm_grad_all_params` checks `x`, `gamma`, and `beta`. Pause on this:
it is the strongest argument for building an autograd engine at all.

## Attention -- `notes/Multi-Head Attention Back Propagation.md`

```python
def attention(x, w_qkv, b_qkv, w_out, b_out, n_head, causal=True):
    qkv = linear(x, w_qkv, b_qkv)                  # fused QKV (GPT-2 c_attn)
    q, k, v = slice into three                     # three Slice nodes
    qh, kh, vh = reshape+transpose to (H, S, d_head)
    scores = (qh @ kh.T) * scale                   # batched matmul
    if causal: scores = scores + causal_mask       # additive 0/-1e9 mask
    ctx = softmax(scores) @ vh
    return linear(merge_heads(ctx), w_out, b_out)
```

Every step is a primitive: the QK^T/softmax/V chain, the head split via
`reshape`+`transpose`, the fused-QKV split via `slice`, the causal mask as an
additive constant. The whole multi-page backward derivation in the note -- score
backward, softmax backward, per-head reassembly into `G_QKV` -- is produced by
composing those primitives' vjps. `test_attention_grad` checks gradients to `x`,
`W_qkv`, and `W_out`; `test_causal_mask_blocks_future` confirms position 0 cannot
see later tokens.

## Residual -- `notes/Residual Connection Back Propagation.md`

```python
def residual(x, sublayer_out):
    return x + sublayer_out
```

Trivial in code, profound in effect. `Add.vjp` sends the upstream gradient down
*both* branches unchanged -- the gradient "highway" `G_X = G_Y + G_F` that lets
very deep networks train.

## Design patterns in this chapter

- **Forward-only authoring**: define computation; differentiation is automatic.
- **Functional layers**: explicit params, no hidden state -> trivially
  differentiable with `grad`.
- **Numerical stability as composition**: log-sum-exp built from primitives.
- **Code-to-math traceability**: each function names its `notes/` derivation.

## What's next

We assemble these layers into the full model in
[09 - Capstone: GPT-2](09-capstone-gpt2.md).
