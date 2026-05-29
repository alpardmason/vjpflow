"""End-to-end smoke + correctness for the GPT-2 capstone."""

from __future__ import annotations

import numpy as np

from autograd.models import gpt2
from tests.util import check_grad


def _tiny_config() -> gpt2.GPT2Config:
    return gpt2.GPT2Config(vocab_size=16, block_size=8, n_layer=2, n_head=2, n_embd=16)


def test_forward_shape_and_initial_loss() -> None:
    cfg = _tiny_config()
    params = gpt2.init_params(cfg, seed=0)
    idx = np.arange(6)
    logits = gpt2.forward(params, idx, cfg)
    assert logits.shape == (6, cfg.vocab_size)
    # A freshly initialised model is roughly uniform -> loss ~ ln(vocab).
    loss = gpt2.loss(params, idx, np.arange(6), cfg).item()
    assert abs(loss - np.log(cfg.vocab_size)) < 0.3


def test_overfits_single_sequence() -> None:
    cfg = _tiny_config()
    params = gpt2.init_params(cfg, seed=1)
    idx = np.array([1, 2, 3, 4, 5, 6, 7, 0])
    tgt = np.array([2, 3, 4, 5, 6, 7, 0, 1])

    first, _ = gpt2.value_and_grad_params(params, idx, tgt, cfg)
    for _ in range(60):
        _, grads = gpt2.value_and_grad_params(params, idx, tgt, cfg)
        params = gpt2.sgd_step(params, grads, lr=0.5)
    last = gpt2.loss(params, idx, tgt, cfg).item()
    assert last < first.item() * 0.2  # loss dropped substantially


def test_gpt2_parameter_gradients_match_numeric() -> None:
    # Spot-check a couple of parameters against finite differences using a
    # fixed sequence. We test the LM-head/embedding (weight-tied) and a block
    # layernorm gain.
    cfg = gpt2.GPT2Config(vocab_size=8, block_size=4, n_layer=1, n_head=2, n_embd=8)
    base = gpt2.init_params(cfg, seed=2)
    idx = np.array([1, 2, 3, 0])
    tgt = np.array([2, 3, 0, 1])
    names = list(base)

    for target_name in ("wte", "h0.ln1_g"):
        arrays = [base[n].numpy() for n in names]
        ti = names.index(target_name)

        def fn(*vals, _names=names):  # noqa: ANN002
            params = dict(zip(_names, vals, strict=True))
            return gpt2.loss(params, idx, tgt, cfg)

        check_grad(fn, arrays, argnum=ti, eps=1e-3, atol=5e-2)
