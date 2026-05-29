"""Forward correctness and vjp gradient checks for the primitive ops."""

from __future__ import annotations

import numpy as np
import pytest

from tests.util import check_grad
from vjpflow import primitives as P
from vjpflow import tensor


@pytest.fixture(autouse=True)
def _seed() -> None:
    np.random.seed(0)


# --- forward correctness vs numpy ------------------------------------------


def test_elementwise_forward() -> None:
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    ta, tb = tensor(a), tensor(b)
    assert np.allclose((ta + tb).numpy(), a + b)
    assert np.allclose((ta * tb).numpy(), a * b)
    assert np.allclose((ta - tb).numpy(), a - b)
    assert np.allclose((ta / tb).numpy(), a / b, atol=1e-5)
    assert np.allclose(P.exp(ta).numpy(), np.exp(a), atol=1e-5)
    assert np.allclose(P.maximum(ta, tb).numpy(), np.maximum(a, b))


def test_matmul_forward_batched() -> None:
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(2, 4, 5).astype(np.float32)
    assert np.allclose(P.matmul(tensor(a), tensor(b)).numpy(), a @ b, atol=1e-4)


def test_reduction_forward() -> None:
    a = np.random.randn(3, 4).astype(np.float32)
    ta = tensor(a)
    assert np.allclose(P.sum(ta, axis=1).numpy(), a.sum(axis=1), atol=1e-4)
    assert np.allclose(P.mean(ta, axis=0).numpy(), a.mean(axis=0), atol=1e-5)
    assert np.allclose(P.max(ta, axis=1).numpy(), a.max(axis=1))


def test_shape_ops_forward() -> None:
    a = np.random.randn(2, 6).astype(np.float32)
    ta = tensor(a)
    assert np.allclose(P.reshape(ta, (3, 4)).numpy(), a.reshape(3, 4))
    assert np.allclose(P.transpose(ta, (1, 0)).numpy(), a.T)
    assert np.allclose(P.slice_(ta, 1, 4, axis=1).numpy(), a[:, 1:4])


def test_gather_scatter_roundtrip() -> None:
    table = np.random.randn(5, 3).astype(np.float32)
    idx = np.array([0, 2, 2, 4])
    gathered = P.gather(tensor(table), idx)
    assert np.allclose(gathered.numpy(), table[idx])
    # scatter_add of ones should count occurrences per row.
    upd = np.ones((4, 3), np.float32)
    scattered = P.scatter_add(tensor(upd), idx, (5, 3)).numpy()
    assert np.allclose(scattered[2], 2.0)  # index 2 appears twice
    assert np.allclose(scattered[1], 0.0)  # index 1 never appears


def test_scatter_add_grad_multidimensional_indices() -> None:
    # Regression: with 2-D indices, ``updates`` has shape indices.shape +
    # out_shape[1:], so the backward gather tail must be out_shape[1:], NOT
    # updates.shape[1:] (which would over-keep an axis and mis-shape the grad).
    from vjpflow import value_and_grad

    idx = np.array([[0, 2, 4], [1, 1, 3]])  # shape (2, 3)
    out_shape = (5, 4)
    upd = np.random.randn(2, 3, 4).astype(np.float32)  # indices.shape + (4,)

    _, (g,) = value_and_grad(
        lambda u: P.scatter_add(u, idx, out_shape).sum(), argnums=(0,)
    )(tensor(upd))
    assert g.numpy().shape == upd.shape
    assert np.allclose(g.numpy(), 1.0)  # each update contributes once


# --- gradient checks -------------------------------------------------------


def test_grad_arithmetic() -> None:
    a = np.random.randn(4).astype(np.float32)
    b = np.abs(np.random.randn(4)).astype(np.float32) + 0.5  # keep div/log safe
    check_grad(lambda a, b: (a * b).sum(), [a, b], argnum=0)
    check_grad(lambda a, b: (a / b).sum(), [a, b], argnum=0)
    check_grad(lambda a, b: (a / b).sum(), [a, b], argnum=1)


def test_grad_broadcasting() -> None:
    # (3,4) * (4,) exercises the sum-to-shape reduction in the backward pass.
    x = np.random.randn(3, 4).astype(np.float32)
    w = np.random.randn(4).astype(np.float32)
    check_grad(lambda x, w: (x * w).sum(), [x, w], argnum=1)


def test_grad_unary() -> None:
    x = np.abs(np.random.randn(5)).astype(np.float32) + 0.5
    check_grad(lambda x: P.exp(x).sum(), [x])
    check_grad(lambda x: P.log(x).sum(), [x])
    check_grad(lambda x: P.sqrt(x).sum(), [x])
    check_grad(lambda x: P.tanh(x).sum(), [x])


def test_grad_matmul() -> None:
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(4, 2).astype(np.float32)
    check_grad(lambda a, b: (a @ b).sum(), [a, b], argnum=0)
    check_grad(lambda a, b: (a @ b).sum(), [a, b], argnum=1)


def test_grad_reductions() -> None:
    x = np.random.randn(3, 4).astype(np.float32)
    check_grad(lambda x: P.mean(x, axis=1).sum(), [x])
    # max routes gradient to the argmax element.
    check_grad(lambda x: P.max(x, axis=1).sum(), [x])


def test_grad_through_slice_and_concat() -> None:
    x = np.random.randn(2, 6).astype(np.float32)
    check_grad(lambda x: P.slice_(x, 0, 3, axis=1).sum(), [x])
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(2, 3).astype(np.float32)
    check_grad(lambda a, b: P.concat([a, b * 2.0], axis=1).sum(), [a, b], argnum=1)
