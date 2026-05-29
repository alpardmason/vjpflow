"""Shared test utilities: finite-difference gradient checking.

The single most valuable test for an autograd engine is: *does the analytic
gradient match a numerical one?* We perturb each input element by +-eps, measure
the change in the (scalar) output, and compare the central-difference estimate
to what ``value_and_grad`` produced.

Because the engine runs in float32, we keep ``eps`` moderate (1e-3) and use a
loose tolerance -- tight enough to catch sign/shape/factor bugs, loose enough to
tolerate float32 rounding.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from vjpflow import Tensor, tensor, value_and_grad


def numeric_grad(
    fn: Callable[..., Tensor], arrays: Sequence[np.ndarray], argnum: int, eps: float = 1e-3
) -> np.ndarray:
    """Central-difference gradient of ``fn`` w.r.t. argument ``argnum``."""
    base = [np.asarray(a, dtype=np.float32) for a in arrays]
    x0 = base[argnum]
    grad = np.zeros_like(x0)
    it = np.nditer(x0, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        plus = [tensor(b) for b in base]
        minus = [tensor(b) for b in base]
        xp = x0.copy()
        xp[idx] += eps
        xm = x0.copy()
        xm[idx] -= eps
        plus[argnum] = tensor(xp)
        minus[argnum] = tensor(xm)
        grad[idx] = (fn(*plus).item() - fn(*minus).item()) / (2 * eps)
    return grad


def check_grad(
    fn: Callable[..., Tensor],
    arrays: Sequence[np.ndarray],
    argnum: int = 0,
    eps: float = 1e-3,
    atol: float = 2e-2,
) -> None:
    """Assert the analytic gradient matches the numerical one for one argument."""
    args = [tensor(np.asarray(a, dtype=np.float32)) for a in arrays]
    _, grad = value_and_grad(fn, argnums=(argnum,))(*args)
    assert isinstance(grad, tuple)  # tuple argnums -> tuple of grads
    analytic = grad[0].numpy()
    numerical = numeric_grad(fn, arrays, argnum, eps)
    max_err = float(np.abs(analytic - numerical).max())
    assert max_err < atol, f"grad mismatch (arg {argnum}): max_err={max_err:.4g}"
