"""GPT-2 capstone: the whole architecture, forward and backward.

This ties together every layer in :mod:`autograd.nn.functional` and mirrors
``notes/GPT-2 Back Propagation.md`` end to end. The remarkable part is what is
*absent*: there is no backward code here at all. We write the forward pass; the
engine differentiates it.

Design choices (kept MVP, for clarity):

- **Functional, not object-oriented.** Parameters live in a plain ``dict`` of
  leaf tensors. The model is a pure function ``params -> logits``. This makes the
  ``grad`` transform trivial to apply and mirrors how JAX/Flax separate params
  from computation.
- **Pre-LayerNorm blocks** (the GPT-2 layout): LN is applied *inside* each
  residual branch.
- **Weight tying**: the LM head reuses the token-embedding matrix
  (``logits = X W_E^T``). Its gradient correctly accumulates contributions from
  both the head and the embedding lookup -- automatically, because both uses
  point at the same leaf tensor and cotangents add up.
- Batch size is 1 (a single sequence). Batching is an exercise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autograd import primitives as P
from autograd.autodiff import value_and_grad
from autograd.nn import functional as F
from autograd.tensor import Tensor, tensor


@dataclass(frozen=True)
class GPT2Config:
    """Hyperparameters. Defaults are a tiny model good for tests/teaching."""

    vocab_size: int = 64
    block_size: int = 32  # max sequence length
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 32

    @property
    def d_ff(self) -> int:
        return 4 * self.n_embd


Params = dict[str, Tensor]


def init_params(config: GPT2Config, seed: int = 0) -> Params:
    """Initialise all parameters as leaf tensors (GPT-2-style small inits)."""
    rng = np.random.default_rng(seed)
    d = config.n_embd

    def normal(*shape: int, std: float = 0.02) -> Tensor:
        return tensor(rng.normal(0.0, std, size=shape).astype(np.float32))

    def ones(*shape: int) -> Tensor:
        return tensor(np.ones(shape, dtype=np.float32))

    def zeros(*shape: int) -> Tensor:
        return tensor(np.zeros(shape, dtype=np.float32))

    params: Params = {
        "wte": normal(config.vocab_size, d),  # token embedding (tied to LM head)
        "wpe": normal(config.block_size, d),  # positional embedding
        "lnf_g": ones(d),
        "lnf_b": zeros(d),
    }
    for i in range(config.n_layer):
        params |= {
            f"h{i}.ln1_g": ones(d),
            f"h{i}.ln1_b": zeros(d),
            f"h{i}.attn_qkv_w": normal(d, 3 * d),
            f"h{i}.attn_qkv_b": zeros(3 * d),
            f"h{i}.attn_out_w": normal(d, d),
            f"h{i}.attn_out_b": zeros(d),
            f"h{i}.ln2_g": ones(d),
            f"h{i}.ln2_b": zeros(d),
            f"h{i}.mlp_fc_w": normal(d, config.d_ff),
            f"h{i}.mlp_fc_b": zeros(config.d_ff),
            f"h{i}.mlp_proj_w": normal(config.d_ff, d),
            f"h{i}.mlp_proj_b": zeros(d),
        }
    return params


def _mlp(x: Tensor, p: Params, i: int) -> Tensor:
    h = F.gelu(F.linear(x, p[f"h{i}.mlp_fc_w"], p[f"h{i}.mlp_fc_b"]))
    return F.linear(h, p[f"h{i}.mlp_proj_w"], p[f"h{i}.mlp_proj_b"])


def forward(params: Params, idx: np.ndarray, config: GPT2Config) -> Tensor:
    """Map token ids ``idx`` (shape ``(S,)``) to logits ``(S, vocab)``."""
    idx = np.asarray(idx)
    seq_len = idx.shape[0]

    # Embedding: token lookup + first S learned positions (notes 3.1).
    tok = F.embedding(params["wte"], idx)  # (S, d)
    pos = P.slice_(params["wpe"], 0, seq_len, axis=0)  # (S, d)
    x = tok + pos

    for i in range(config.n_layer):
        # Residual branch 1: pre-LN -> attention.
        x_norm = F.layernorm(x, params[f"h{i}.ln1_g"], params[f"h{i}.ln1_b"])
        attn = F.attention(
            x_norm,
            params[f"h{i}.attn_qkv_w"],
            params[f"h{i}.attn_qkv_b"],
            params[f"h{i}.attn_out_w"],
            params[f"h{i}.attn_out_b"],
            n_head=config.n_head,
            causal=True,
        )
        x = F.residual(x, attn)
        # Residual branch 2: pre-LN -> MLP.
        x_norm = F.layernorm(x, params[f"h{i}.ln2_g"], params[f"h{i}.ln2_b"])
        x = F.residual(x, _mlp(x_norm, params, i))

    x = F.layernorm(x, params["lnf_g"], params["lnf_b"])
    # Weight-tied LM head: logits = X W_E^T  (notes 3.5 / 5.2).
    return P.matmul(x, params["wte"].T)


def loss(params: Params, idx: np.ndarray, targets: np.ndarray, config: GPT2Config) -> Tensor:
    """Mean next-token cross-entropy over the sequence."""
    logits = forward(params, idx, config)
    return F.cross_entropy(logits, np.asarray(targets))


def value_and_grad_params(
    params: Params, idx: np.ndarray, targets: np.ndarray, config: GPT2Config
) -> tuple[Tensor, Params]:
    """Loss and a dict of per-parameter gradients.

    ``value_and_grad`` differentiates *positional* arguments, so we flatten the
    params dict to a list, differentiate w.r.t. all of them, and re-key the
    results. (JAX's ``grad`` does this for you via pytrees; here we keep it
    explicit so the mechanism is visible.)
    """
    names = list(params)

    def flat_loss(*values: Tensor) -> Tensor:
        return loss(dict(zip(names, values, strict=True)), idx, targets, config)

    vg = value_and_grad(flat_loss, argnums=tuple(range(len(names))))
    value, grads = vg(*[params[n] for n in names])
    assert isinstance(grads, tuple)  # tuple argnums -> tuple of grads
    return value, dict(zip(names, grads, strict=True))


def sgd_step(params: Params, grads: Params, lr: float) -> Params:
    """One vanilla SGD update, returning fresh parameter leaves.

    We ``detach`` after the update so the next step starts from clean leaves
    rather than a graph that reaches back through the whole optimisation
    history (which would leak memory and recompute the past).
    """
    updated: Params = {}
    for name, weight in params.items():
        new_weight = weight - lr * grads[name]
        updated[name] = new_weight.detach()
    return updated
