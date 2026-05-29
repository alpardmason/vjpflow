"""Tests for the functional autodiff transforms themselves."""

from __future__ import annotations

import numpy as np

from autograd import grad, tensor, value_and_grad
from autograd import primitives as P


def test_value_and_grad_returns_value_and_grad() -> None:
    x = tensor(np.array([3.0], np.float32))
    value, g = value_and_grad(lambda x: (x * x).sum())(x)
    assert np.allclose(value.item(), 9.0)
    assert np.allclose(g.numpy(), 6.0)  # d/dx x^2 = 2x


def test_grad_only_returns_grad() -> None:
    x = tensor(np.array([2.0], np.float32))
    g = grad(lambda x: (x * x * x).sum())(x)
    assert np.allclose(g.numpy(), 12.0)  # 3x^2 at x=2


def test_multiple_argnums() -> None:
    a = tensor(np.array([2.0], np.float32))
    b = tensor(np.array([5.0], np.float32))
    _, (ga, gb) = value_and_grad(lambda a, b: (a * b).sum(), argnums=(0, 1))(a, b)
    assert np.allclose(ga.numpy(), 5.0)  # d/da ab = b
    assert np.allclose(gb.numpy(), 2.0)  # d/db ab = a


def test_gradient_accumulation_on_shared_node() -> None:
    # x is used twice: f = x*x + x. df/dx = 2x + 1. The engine must *add* the
    # two gradient contributions, not overwrite.
    x = tensor(np.array([4.0], np.float32))
    g = grad(lambda x: (x * x + x).sum())(x)
    assert np.allclose(g.numpy(), 2 * 4.0 + 1.0)


def test_unused_argument_gets_zero_grad() -> None:
    a = tensor(np.array([1.0, 2.0], np.float32))
    b = tensor(np.array([3.0, 4.0], np.float32))
    _, (ga, gb) = value_and_grad(lambda a, b: a.sum(), argnums=(0, 1))(a, b)
    assert np.allclose(ga.numpy(), 1.0)
    assert np.allclose(gb.numpy(), 0.0)  # b never touched


def test_second_order_gradient() -> None:
    # grad(grad(x^3)) = 6x. Works because the backward pass is itself a graph.
    def d1(x):
        return grad(lambda x: (x * x * x).sum())(x).sum()

    x = tensor(np.array([2.0], np.float32))
    g2 = grad(d1)(x)
    assert np.allclose(g2.numpy(), 6 * 2.0, atol=1e-4)


def test_laziness_no_compute_until_eval() -> None:
    x = tensor(np.array([1.0, 2.0], np.float32))
    y = P.exp(x * x)  # builds a graph
    assert y._data is None  # nothing computed yet
    y.eval()
    assert y._data is not None
