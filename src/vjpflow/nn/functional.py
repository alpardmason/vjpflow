"""Functional layers for the GPT-2 path (and a few activations for exercises).

Every function takes explicit parameter tensors (weights, biases, gains) -- the
*functional* style. There is no hidden state, which keeps differentiation a pure
transform: ``grad(lambda W: loss(linear(x, W)))`` just works.

Shapes follow the ``notes/`` row-vector convention: a sequence is ``X`` of shape
``(S, d)``; weights act on the right (``Y = X W``).
"""

from __future__ import annotations

import math

import numpy as np

from vjpflow import primitives as P
from vjpflow.tensor import Tensor, tensor


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Affine map ``Y = X W (+ b)``  -- ``notes/Linear Back Propagation.md``.

    The backward (``G_X = G_Y W^T``, ``G_W = X^T G_Y``, ``G_b = 1^T G_Y``) is
    produced automatically by :class:`~vjpflow.primitives.MatMul` and the
    broadcast-aware :class:`~vjpflow.primitives.Add`.
    """
    y = P.matmul(x, weight)
    return P.add(y, bias) if bias is not None else y


def embedding(table: Tensor, indices: np.ndarray) -> Tensor:
    """Row lookup ``table[indices]`` -- ``notes/Embedding Back Propagation.md``.

    Backward is a scatter-add (a repeated token id accumulates gradients), which
    is exactly :class:`~vjpflow.primitives.Gather`'s vjp.
    """
    return P.gather(table, np.asarray(indices))


def relu(x: Tensor) -> Tensor:
    """``max(x, 0)`` -- ``notes/ReLU Back Propagation.md``."""
    return P.maximum(x, tensor(np.zeros((), np.float32), backend=x.backend))


def gelu(x: Tensor) -> Tensor:
    """Tanh-approximated GELU (GPT-2 variant) -- ``notes/GELU Back Propagation.md``.

    ``0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))``. Written purely with
    primitives, so its (somewhat involved) derivative is assembled by the engine.
    """
    kappa = math.sqrt(2.0 / math.pi)
    c = 0.044715
    x3 = x * x * x
    inner = kappa * (x + c * x3)
    return 0.5 * x * (1.0 + P.tanh(inner))


def silu(x: Tensor) -> Tensor:
    """SiLU / Swish ``x * sigmoid(x)`` -- ``notes/SiLU Back Propagation.md``."""
    return x * sigmoid(x)


def sigmoid(x: Tensor) -> Tensor:
    one = tensor(np.ones((), np.float32), backend=x.backend)
    return one / (one + P.exp(-x))


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable row-softmax -- ``notes/Softmax Back Propagation.md``.

    We subtract ``max(x)`` before ``exp``. Softmax is shift-invariant, so the
    *value* is unchanged; the subtraction only prevents ``exp`` from overflowing.
    Because the shift cancels mathematically, differentiating through it yields
    the exact softmax Jacobian -- no need to stop the gradient.
    """
    m = P.max(x, axis=axis, keepdims=True)
    e = P.exp(x - m)
    return e / P.sum(e, axis=axis, keepdims=True)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """``log(softmax(x))`` via the log-sum-exp trick (avoids ``log`` of a tiny number)."""
    m = P.max(x, axis=axis, keepdims=True)
    shifted = x - m
    lse = P.log(P.sum(P.exp(shifted), axis=axis, keepdims=True))
    return shifted - lse


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean cross-entropy -- ``notes/Cross-Entropy Loss Back Propagation.md``.

    ``logits`` is ``(N, V)``; ``targets`` is an integer array ``(N,)``. We build
    a one-hot matrix as a constant and reduce, so the famous ``(P - Y)/N``
    gradient emerges automatically from log-softmax + the one-hot mask.
    """
    targets = np.asarray(targets)
    n, vocab = logits.shape
    logp = log_softmax(logits, axis=-1)
    onehot = np.zeros((n, vocab), dtype=np.float32)
    onehot[np.arange(n), targets] = 1.0
    onehot_t = tensor(onehot, backend=logits.backend)
    nll = P.neg(P.sum(logp * onehot_t, axis=-1))  # (N,)
    return P.mean(nll)


def layernorm(
    x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5
) -> Tensor:
    """LayerNorm over the last axis -- ``notes/Layer Normalization Back Propagation.md``.

    The full ``O(d)`` LayerNorm backward (mean/variance Jacobian) is *not* hand
    coded -- it falls out of differentiating mean/sub/mul/sqrt below. Compare the
    note's boxed ``G_X`` formula with what the engine assembles; they match.
    """
    mu = P.mean(x, axis=-1, keepdims=True)
    centered = x - mu
    var = P.mean(centered * centered, axis=-1, keepdims=True)
    inv_std = 1.0 / P.sqrt(var + eps)
    x_hat = centered * inv_std
    return x_hat * gamma + beta


def rmsnorm(x: Tensor, gamma: Tensor, eps: float = 1e-5) -> Tensor:
    """RMSNorm -- ``notes/RMS Normalization Back Propagation.md`` (no mean centering)."""
    ms = P.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / P.sqrt(ms + eps)) * gamma


def causal_mask(seq_len: int, backend) -> Tensor:  # noqa: ANN001
    """Additive mask: 0 on/below the diagonal, large-negative above.

    Added to attention scores so softmax assigns ~0 weight to future positions.
    We use ``-1e9`` rather than ``-inf`` to keep ``exp`` finite (it underflows to
    0 cleanly).
    """
    mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
    return tensor(mask, backend=backend)


def attention(
    x: Tensor,
    w_qkv: Tensor,
    b_qkv: Tensor,
    w_out: Tensor,
    b_out: Tensor,
    n_head: int,
    causal: bool = True,
) -> Tensor:
    """Causal multi-head self-attention -- ``notes/Multi-Head Attention Back Propagation.md``.

    ``x`` is ``(S, d)``. Uses GPT-2's fused QKV projection, splits into heads,
    runs scaled dot-product attention per head (batched matmul), concatenates,
    and projects out. All standard ops -> automatic backward.
    """
    seq_len, d_model = x.shape
    d_head = d_model // n_head
    scale = 1.0 / math.sqrt(d_head)

    qkv = linear(x, w_qkv, b_qkv)  # (S, 3d)
    q = P.slice_(qkv, 0, d_model, axis=-1)
    k = P.slice_(qkv, d_model, 2 * d_model, axis=-1)
    v = P.slice_(qkv, 2 * d_model, 3 * d_model, axis=-1)

    def to_heads(t: Tensor) -> Tensor:
        # (S, d) -> (S, H, d_head) -> (H, S, d_head)
        return P.transpose(P.reshape(t, (seq_len, n_head, d_head)), (1, 0, 2))

    qh, kh, vh = to_heads(q), to_heads(k), to_heads(v)

    scores = P.matmul(qh, kh.T) * scale  # (H, S, S); .T swaps last two axes
    if causal:
        scores = scores + causal_mask(seq_len, x.backend)
    weights = softmax(scores, axis=-1)
    context = P.matmul(weights, vh)  # (H, S, d_head)

    # (H, S, d_head) -> (S, H, d_head) -> (S, d)
    merged = P.reshape(P.transpose(context, (1, 0, 2)), (seq_len, d_model))
    return linear(merged, w_out, b_out)


def residual(x: Tensor, sublayer_out: Tensor) -> Tensor:
    """``x + F(x)`` -- ``notes/Residual Connection Back Propagation.md``.

    Trivial in code, profound in effect: ``Add``'s vjp sends the upstream
    gradient down *both* branches unchanged, giving the gradient "highway" that
    lets very deep networks train.
    """
    return x + sublayer_out
