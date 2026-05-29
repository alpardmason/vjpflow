"""Backend parity: the Metal GPU backend must agree with numpy.

These tests are skipped automatically when Metal is unavailable (non-Apple
hardware, or ``pyobjc-framework-Metal`` not installed), so the suite still
passes everywhere on the numpy backend.
"""

from __future__ import annotations

import numpy as np
import pytest

from autograd import primitives as P
from autograd import tensor
from autograd.backends import get_backend
from autograd.backends.metal_backend import MetalBackend

metal_available = MetalBackend.is_available()
pytestmark = pytest.mark.skipif(not metal_available, reason="Metal GPU unavailable")


def _on(backend_name: str, array: np.ndarray):  # noqa: ANN202
    return tensor(array, backend=get_backend(backend_name))


def test_elementwise_parity() -> None:
    a = np.random.randn(4, 8).astype(np.float32)
    b = np.random.randn(4, 8).astype(np.float32)
    cpu = (_on("numpy", a) * _on("numpy", b) + _on("numpy", a)).numpy()
    gpu = (_on("metal", a) * _on("metal", b) + _on("metal", a)).numpy()
    assert np.allclose(cpu, gpu, atol=1e-4)


def test_matmul_parity() -> None:
    a = np.random.randn(16, 24).astype(np.float32)
    b = np.random.randn(24, 12).astype(np.float32)
    cpu = P.matmul(_on("numpy", a), _on("numpy", b)).numpy()
    gpu = P.matmul(_on("metal", a), _on("metal", b)).numpy()
    assert np.allclose(cpu, gpu, atol=1e-3)


def test_reduction_parity() -> None:
    a = np.random.randn(6, 10).astype(np.float32)
    cpu = P.sum(_on("numpy", a), axis=-1).numpy()
    gpu = P.sum(_on("metal", a), axis=-1).numpy()
    assert np.allclose(cpu, gpu, atol=1e-3)


def test_unary_parity() -> None:
    a = np.abs(np.random.randn(32).astype(np.float32)) + 0.1
    cpu = P.log(P.exp(_on("numpy", a))).numpy()
    gpu = P.log(P.exp(_on("metal", a))).numpy()
    assert np.allclose(cpu, gpu, atol=1e-3)


def test_small_mlp_end_to_end_parity() -> None:
    # A tiny MLP + softmax forward, the headline "runs on the GPU" example.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 16)).astype(np.float32)
    w = rng.normal(size=(16, 4)).astype(np.float32)

    def forward(xt, wt):
        from autograd.nn import functional as Fn

        return Fn.softmax(Fn.relu(xt @ wt))

    cpu = forward(_on("numpy", x), _on("numpy", w)).numpy()
    gpu = forward(_on("metal", x), _on("metal", w)).numpy()
    assert np.allclose(cpu, gpu, atol=1e-3)
