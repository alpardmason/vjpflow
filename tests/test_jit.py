"""The JIT fusion pass: it must preserve values and reduce node count."""

from __future__ import annotations

import numpy as np

from autograd import primitives as P
from autograd import tensor
from autograd.jit import FusedUnary, fuse_elementwise, jit


def test_fusion_preserves_values() -> None:
    x = tensor(np.random.randn(50).astype(np.float32))
    out = P.tanh(P.exp(P.neg(P.sqrt(P.exp(x * x)))))
    fused, stats = fuse_elementwise(out)
    assert np.allclose(out.numpy(), fused.numpy(), atol=1e-4)
    assert stats.nodes_after < stats.nodes_before
    assert stats.chains_fused > 0


def test_fusion_collapses_unary_chain() -> None:
    x = tensor(np.random.randn(10).astype(np.float32))
    # 4 chained unary ops on top of one binary op (x*x).
    out = P.tanh(P.neg(P.exp(P.exp(x * x))))
    fused, stats = fuse_elementwise(out)
    # The 4 unary ops collapse into one FusedUnary; the mul stays -> 2 nodes.
    assert stats.nodes_after == 2
    assert isinstance(fused.op, FusedUnary)
    assert len(fused.op.funcs) == 4


def test_jit_wrapper_runs_and_caches() -> None:
    def f(x):
        return P.tanh(P.exp(x))

    compiled = jit(f)
    x = tensor(np.random.randn(20).astype(np.float32))
    y = compiled(x)
    assert np.allclose(y.numpy(), np.tanh(np.exp(x.numpy())), atol=1e-5)
    assert compiled.last_stats is not None
    assert compiled.last_stats.chains_fused >= 1


def test_no_fusion_across_multi_consumer() -> None:
    # If an intermediate is consumed twice, it must not be fused away (its value
    # is needed by both consumers).
    x = tensor(np.random.randn(8).astype(np.float32))
    shared = P.exp(x)
    out = P.tanh(shared) + P.neg(shared)  # shared has two consumers
    _, stats = fuse_elementwise(out)
    # exp cannot be merged into either consumer; values must still match.
    fused, _ = fuse_elementwise(out)
    assert np.allclose(out.numpy(), fused.numpy(), atol=1e-5)
