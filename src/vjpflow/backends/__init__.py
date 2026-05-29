"""Backend registry and selection.

The engine keeps a single *default backend* (a module-level singleton). Leaf
tensors created without an explicit backend inherit it. Switch backends with
:func:`set_default_backend` -- e.g. ``set_default_backend("metal")`` -- and
every subsequent graph runs on the GPU, with no other code changes.
"""

from __future__ import annotations

from vjpflow.backends.base import Backend, Native
from vjpflow.backends.numpy_backend import NumpyBackend

__all__ = [
    "Backend",
    "Native",
    "NumpyBackend",
    "get_backend",
    "default_backend",
    "set_default_backend",
]

# Cache one instance per backend name; backends are stateful (they hold a Metal
# device / pipeline cache) so we must not rebuild them on every call.
_INSTANCES: dict[str, Backend] = {}
_DEFAULT_NAME = "numpy"


def get_backend(name: str) -> Backend:
    """Return the (cached) backend instance for ``name`` (``"numpy"``/``"metal"``)."""
    if name not in _INSTANCES:
        if name == "numpy":
            _INSTANCES[name] = NumpyBackend()
        elif name == "metal":
            # Imported lazily so that importing the package never requires
            # PyObjC / a Metal device.
            from vjpflow.backends.metal_backend import MetalBackend

            _INSTANCES[name] = MetalBackend()
        else:
            raise ValueError(f"Unknown backend: {name!r} (expected 'numpy' or 'metal')")
    return _INSTANCES[name]


def default_backend() -> Backend:
    """The backend new leaf tensors use when none is specified."""
    return get_backend(_DEFAULT_NAME)


def set_default_backend(name: str) -> Backend:
    """Set and return the process-wide default backend."""
    global _DEFAULT_NAME
    backend = get_backend(name)  # validates / constructs eagerly
    _DEFAULT_NAME = name
    return backend
