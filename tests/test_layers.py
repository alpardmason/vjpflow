"""Gradient checks and forward sanity for the nn layers."""

from __future__ import annotations

import numpy as np
import pytest

import autograd.nn.functional as F
from autograd import tensor
from tests.util import check_grad


@pytest.fixture(autouse=True)
def _seed() -> None:
    np.random.seed(1)


def test_linear_forward_and_grad() -> None:
    x = np.random.randn(4, 3).astype(np.float32)
    w = np.random.randn(3, 2).astype(np.float32)
    b = np.random.randn(2).astype(np.float32)
    out = F.linear(tensor(x), tensor(w), tensor(b)).numpy()
    assert np.allclose(out, x @ w + b, atol=1e-4)
    check_grad(lambda x, w, b: F.linear(x, w, b).sum(), [x, w, b], argnum=1)
    check_grad(lambda x, w, b: F.linear(x, w, b).sum(), [x, w, b], argnum=2)


def test_softmax_forward_and_grad() -> None:
    x = np.random.randn(3, 5).astype(np.float32)
    out = F.softmax(tensor(x)).numpy()
    assert np.allclose(out.sum(axis=-1), 1.0, atol=1e-5)  # rows sum to 1
    target = np.arange(5).astype(np.float32)
    check_grad(lambda x: (F.softmax(x) * tensor(target)).sum(), [x])


def test_layernorm_grad_all_params() -> None:
    x = np.random.randn(4, 8).astype(np.float32)
    g = np.random.randn(8).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    fn = lambda x, g, b: F.layernorm(x, g, b).sum()  # noqa: E731
    for argnum in (0, 1, 2):
        check_grad(fn, [x, g, b], argnum=argnum)


def test_layernorm_normalises() -> None:
    x = np.random.randn(4, 16).astype(np.float32)
    g = np.ones(16, np.float32)
    b = np.zeros(16, np.float32)
    out = F.layernorm(tensor(x), tensor(g), tensor(b)).numpy()
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-4)
    assert np.allclose(out.std(axis=-1), 1.0, atol=1e-2)


def test_gelu_grad() -> None:
    x = np.random.randn(4, 4).astype(np.float32)
    check_grad(lambda x: F.gelu(x).sum(), [x])


def test_cross_entropy_grad_and_value() -> None:
    logits = np.random.randn(5, 7).astype(np.float32)
    targets = np.array([0, 1, 2, 3, 4])
    check_grad(lambda lg: F.cross_entropy(lg, targets), [logits])
    # Uniform logits -> loss = log(V).
    uniform = np.zeros((3, 7), np.float32)
    loss = F.cross_entropy(tensor(uniform), np.array([0, 1, 2])).item()
    assert np.allclose(loss, np.log(7), atol=1e-4)


def test_attention_grad() -> None:
    s, d, h = 3, 4, 2
    x = np.random.randn(s, d).astype(np.float32)
    wqkv = np.random.randn(d, 3 * d).astype(np.float32)
    bqkv = np.zeros(3 * d, np.float32)
    wo = np.random.randn(d, d).astype(np.float32)
    bo = np.zeros(d, np.float32)

    # attention signature is (x, w_qkv, b_qkv, w_out, b_out, ...); wrap to pick
    # the differentiable matrices.
    def loss(x, wqkv, wo):
        return F.attention(x, wqkv, tensor(bqkv), wo, tensor(bo), n_head=h).sum()

    for argnum in (0, 1, 2):
        check_grad(loss, [x, wqkv, wo], argnum=argnum)


def test_causal_mask_blocks_future() -> None:
    # With a causal mask, position 0's output must not depend on later tokens.
    s, d, h = 4, 4, 1
    rng = np.random.default_rng(0)
    x = rng.normal(size=(s, d)).astype(np.float32)
    wqkv = rng.normal(size=(d, 3 * d)).astype(np.float32)
    wo = np.eye(d, dtype=np.float32)
    z = np.zeros
    args = (tensor(wqkv), tensor(z(3 * d, np.float32)), tensor(wo), tensor(z(d, np.float32)))
    out_a = F.attention(tensor(x), *args, n_head=h).numpy()
    x2 = x.copy()
    x2[3] += 10.0  # perturb the last token
    out_b = F.attention(tensor(x2), *args, n_head=h).numpy()
    assert np.allclose(out_a[0], out_b[0], atol=1e-4)  # row 0 unaffected
