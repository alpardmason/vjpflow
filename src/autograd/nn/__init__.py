"""Neural-network layers, composed entirely from primitives.

Nothing here defines a backward pass. Each layer is a *pure composition* of the
ops in :mod:`autograd.primitives`, so :func:`autograd.grad` differentiates it
automatically. That is the central payoff of an autograd engine: you write the
forward pass once and get the backward pass for free.

Each function corresponds to a derivation in ``notes/`` (e.g. ``layernorm`` ->
``notes/Layer Normalization Back Propagation.md``); read them side by side.
"""

from autograd.nn import functional as functional

__all__ = ["functional"]
