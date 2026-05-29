"""A raw Metal GPU backend, driven from Python via PyObjC.

This backend executes the *compute-heavy* primitives -- matmul, elementwise,
and last-axis reductions -- as hand-written Metal shaders (see
``metal/kernels.metal``). Everything else (indexing, structural, odd-axis
reductions) falls back to numpy. That is enough to run a small MLP with a
softmax cross-entropy loss end-to-end on the GPU, which is the teaching goal.

It is *not* a full second backend: GPT-2 still runs on numpy. See guide 06
(Metal concepts) and guide 07 (this file, line by line).

Availability
------------
Metal only exists on macOS with Apple Silicon (or AMD GPUs) *and* the
``pyobjc-framework-Metal`` package installed (``uv sync --extra metal``).
:meth:`MetalBackend.is_available` reports this; tests skip GPU parity when it is
False so the engine still works everywhere on numpy.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from vjpflow.backends.base import Backend, Native
from vjpflow.backends.numpy_backend import NumpyBackend

try:  # PyObjC Metal bindings -- optional, macOS only.
    import Metal as _Metal  # type: ignore[import-not-found]

    _METAL_IMPORT_OK = True
except Exception:  # pragma: no cover - depends on platform
    _Metal = None
    _METAL_IMPORT_OK = False

# Typed as Any so the static checker does not flag attribute access on the
# optional module; availability is guarded at runtime by ``is_available``.
Metal: Any = _Metal

_KERNEL_SOURCE = Path(__file__).parent / "metal" / "kernels.metal"


class MetalArray:
    """A device buffer plus the metadata numpy would normally carry.

    The engine treats this as an opaque ``Native`` handle; only
    :class:`MetalBackend` ever reaches inside it.
    """

    __slots__ = ("buffer", "shape", "size")

    def __init__(self, buffer: Any, shape: tuple[int, ...]) -> None:
        self.buffer = buffer
        self.shape = shape
        self.size = int(np.prod(shape)) if shape else 1


class MetalBackend(Backend):
    """Executes core ops on the Apple GPU; delegates the rest to numpy."""

    name = "metal"

    def __init__(self) -> None:
        if not _METAL_IMPORT_OK:
            raise RuntimeError(
                "Metal bindings unavailable. Install with `uv sync --extra metal` "
                "on an Apple Silicon Mac."
            )
        self._device = Metal.MTLCreateSystemDefaultDevice()
        if self._device is None:  # pragma: no cover - hardware dependent
            raise RuntimeError("No Metal device found (not an Apple GPU machine?).")

        # Compile the shader library once at startup. Errors here are almost
        # always typos in the .metal source -- surface them loudly.
        source = _KERNEL_SOURCE.read_text()
        library, error = self._device.newLibraryWithSource_options_error_(source, None, None)
        if library is None:  # pragma: no cover - compile failure
            raise RuntimeError(f"Metal kernel compilation failed: {error}")
        self._library = library
        self._queue = self._device.newCommandQueue()

        # Build (and cache) a compute pipeline state per kernel function. The
        # pipeline bakes the compiled function into something dispatchable.
        self._pipelines: dict[str, object] = {}
        names = [
            "ew_add", "ew_sub", "ew_mul", "ew_div", "ew_max", "ew_pow", "ew_gt",
            "un_neg", "un_exp", "un_log", "un_sqrt", "un_tanh",
            "matmul", "reduce_sum_lastdim", "reduce_max_lastdim",
        ]
        for name in names:
            fn = self._library.newFunctionWithName_(name)
            pipeline, err = self._device.newComputePipelineStateWithFunction_error_(fn, None)
            if pipeline is None:  # pragma: no cover
                raise RuntimeError(f"Failed to build pipeline for {name}: {err}")
            self._pipelines[name] = pipeline

        # Used for the numpy fallback paths.
        self._cpu = NumpyBackend()

    @staticmethod
    def is_available() -> bool:
        if not _METAL_IMPORT_OK:
            return False
        try:  # pragma: no cover - hardware dependent
            return Metal.MTLCreateSystemDefaultDevice() is not None
        except Exception:  # pragma: no cover
            return False

    # -- buffer helpers --------------------------------------------------
    def _new_buffer(self, nbytes: int) -> object:
        # MTLResourceStorageModeShared: one allocation visible to CPU and GPU.
        # On Apple Silicon CPU and GPU share physical memory, so "upload" and
        # "download" are just pointer reads -- no PCIe copy. That unified-memory
        # model is the single biggest reason MLX is fast on a Mac.
        return self._device.newBufferWithLength_options_(
            max(nbytes, 4), Metal.MTLResourceStorageModeShared
        )

    def _upload(self, array: np.ndarray) -> MetalArray:
        data = np.ascontiguousarray(array, dtype=np.float32)
        buf = self._device.newBufferWithBytes_length_options_(
            data.tobytes(), max(data.nbytes, 4), Metal.MTLResourceStorageModeShared
        )
        return MetalArray(buf, data.shape)

    def _download(self, arr: MetalArray) -> np.ndarray:
        nbytes = arr.size * 4
        mv = arr.buffer.contents().as_buffer(nbytes)
        return np.frombuffer(mv, dtype=np.float32, count=arr.size).reshape(arr.shape).copy()

    def _dispatch_1d(self, pipeline_name: str, buffers: list[object], count: int) -> None:
        cmd = self._queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self._pipelines[pipeline_name])
        for i, buf in enumerate(buffers):
            enc.setBuffer_offset_atIndex_(buf, 0, i)
        enc.setBytes_length_atIndex_(struct.pack("I", count), 4, len(buffers))
        # dispatchThreads handles non-multiple sizes (Apple GPU feature), so we
        # ask for exactly `count` threads and let the kernel's bounds check
        # cover the rounding done by the threadgroup width.
        tg = min(count, 256)
        enc.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(count, 1, 1), Metal.MTLSizeMake(tg, 1, 1)
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

    # -- interop ---------------------------------------------------------
    def from_numpy(self, array: np.ndarray) -> Native:
        return self._upload(array)

    def to_numpy(self, array: Native) -> np.ndarray:
        return self._download(array)

    # -- elementwise binary (broadcast on host, compute on GPU) ----------
    def _binary(self, name: str, a: MetalArray, b: MetalArray) -> MetalArray:
        out_shape = np.broadcast_shapes(a.shape, b.shape)
        # Broadcasting on the GPU needs strides; for clarity we expand both
        # operands to the output shape on the host, then run an equal-shape
        # kernel. A production engine would pass strides into the shader.
        if a.shape != out_shape:
            a = self._upload(np.broadcast_to(self._download(a), out_shape))
        if b.shape != out_shape:
            b = self._upload(np.broadcast_to(self._download(b), out_shape))
        count = int(np.prod(out_shape)) if out_shape else 1
        out = MetalArray(self._new_buffer(count * 4), tuple(out_shape))
        self._dispatch_1d(name, [a.buffer, b.buffer, out.buffer], count)
        return out

    def add(self, a, b):  # noqa: ANN001
        return self._binary("ew_add", a, b)

    def sub(self, a, b):  # noqa: ANN001
        return self._binary("ew_sub", a, b)

    def mul(self, a, b):  # noqa: ANN001
        return self._binary("ew_mul", a, b)

    def div(self, a, b):  # noqa: ANN001
        return self._binary("ew_div", a, b)

    def pow(self, a, b):  # noqa: ANN001
        return self._binary("ew_pow", a, b)

    def maximum(self, a, b):  # noqa: ANN001
        return self._binary("ew_max", a, b)

    def greater(self, a, b):  # noqa: ANN001
        return self._binary("ew_gt", a, b)

    # -- elementwise unary -----------------------------------------------
    def _unary(self, name: str, a: MetalArray) -> MetalArray:
        out = MetalArray(self._new_buffer(a.size * 4), a.shape)
        self._dispatch_1d(name, [a.buffer, out.buffer], a.size)
        return out

    def neg(self, a):  # noqa: ANN001
        return self._unary("un_neg", a)

    def exp(self, a):  # noqa: ANN001
        return self._unary("un_exp", a)

    def log(self, a):  # noqa: ANN001
        return self._unary("un_log", a)

    def sqrt(self, a):  # noqa: ANN001
        return self._unary("un_sqrt", a)

    def tanh(self, a):  # noqa: ANN001
        return self._unary("un_tanh", a)

    # -- reductions ------------------------------------------------------
    def _reduce_lastdim(self, name: str, a: MetalArray) -> MetalArray:
        rows = int(np.prod(a.shape[:-1])) if len(a.shape) > 1 else 1
        cols = a.shape[-1] if a.shape else 1
        out = MetalArray(self._new_buffer(rows * 4), a.shape[:-1])
        cmd = self._queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self._pipelines[name])
        enc.setBuffer_offset_atIndex_(a.buffer, 0, 0)
        enc.setBuffer_offset_atIndex_(out.buffer, 0, 1)
        enc.setBytes_length_atIndex_(struct.pack("II", rows, cols), 8, 2)
        tg = min(rows, 256)
        enc.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(rows, 1, 1), Metal.MTLSizeMake(tg, 1, 1)
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        return out

    def sum(self, a, axis, keepdims):  # noqa: ANN001
        return self._reduce("reduce_sum_lastdim", "sum", a, axis, keepdims)

    def max(self, a, axis, keepdims):  # noqa: ANN001
        return self._reduce("reduce_max_lastdim", "max", a, axis, keepdims)

    def mean(self, a, axis, keepdims):  # noqa: ANN001
        # Reduce on the GPU, then scale by 1/count. The divide is on a tiny
        # already-reduced array, so it does not undermine the GPU win.
        summed = self.sum(a, axis, keepdims)
        if axis is None:
            count = a.size
        else:
            count = int(np.prod([a.shape[ax] for ax in axis]))
        host = self._download(summed) / count
        return self._upload(host)

    def _reduce(self, kernel: str, np_name: str, a, axis, keepdims):  # noqa: ANN001
        if axis is not None and tuple(axis) == (len(a.shape) - 1,):
            out = self._reduce_lastdim(kernel, a)
            if keepdims:
                out = MetalArray(out.buffer, (*a.shape[:-1], 1))
            return out
        # Fallback for axis=None / interior axes.
        host = getattr(self._cpu, np_name)(self._download(a), axis, keepdims)
        return self._upload(host)

    # -- linear algebra --------------------------------------------------
    def matmul(self, a, b):  # noqa: ANN001
        # Only the 2D case runs on the GPU kernel; batched matmuls fall back.
        if len(a.shape) != 2 or len(b.shape) != 2:
            return self._upload(np.matmul(self._download(a), self._download(b)))
        m, k = a.shape
        k2, n = b.shape
        assert k == k2, f"matmul shape mismatch: {a.shape} @ {b.shape}"
        out = MetalArray(self._new_buffer(m * n * 4), (m, n))
        cmd = self._queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self._pipelines["matmul"])
        enc.setBuffer_offset_atIndex_(a.buffer, 0, 0)
        enc.setBuffer_offset_atIndex_(b.buffer, 0, 1)
        enc.setBuffer_offset_atIndex_(out.buffer, 0, 2)
        # uint3 is 16-byte aligned in Metal, so pad to four uints.
        enc.setBytes_length_atIndex_(struct.pack("IIII", m, k, n, 0), 16, 3)
        enc.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(n, m, 1), Metal.MTLSizeMake(min(n, 16), min(m, 16), 1)
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        return out

    # -- shape / indexing / structural: numpy fallbacks ------------------
    # These are memory-movement ops with little arithmetic; doing them on the
    # host keeps the GPU code focused on the parts that actually benefit.
    def _via_cpu(self, fn_name: str, *natives, **kwargs):  # noqa: ANN001, ANN002
        host_args = [self._download(x) if isinstance(x, MetalArray) else x for x in natives]
        result = getattr(self._cpu, fn_name)(*host_args, **kwargs)
        if isinstance(result, list):
            return [self._upload(r) for r in result]
        return self._upload(result)

    def reshape(self, a, shape):  # noqa: ANN001
        return MetalArray(a.buffer, tuple(shape))  # contiguous: a pure view

    def transpose(self, a, axes):  # noqa: ANN001
        return self._via_cpu("transpose", a, axes=axes)

    def broadcast_to(self, a, shape):  # noqa: ANN001
        return self._via_cpu("broadcast_to", a, shape=shape)

    def gather(self, table, indices):  # noqa: ANN001
        host = self._cpu.gather(self._download(table), indices)
        return self._upload(host)

    def scatter_add(self, out_shape, indices, updates):  # noqa: ANN001
        host = self._cpu.scatter_add(out_shape, indices, self._download(updates))
        return self._upload(host)

    def concat(self, arrays, axis):  # noqa: ANN001
        host = self._cpu.concat([self._download(x) for x in arrays], axis)
        return self._upload(host)

    def slice(self, a, start, stop, axis):  # noqa: ANN001
        host = self._cpu.slice(self._download(a), start, stop, axis)
        return self._upload(host)

    def where(self, cond, a, b):  # noqa: ANN001
        c = self._download(cond) if isinstance(cond, MetalArray) else cond
        host = self._cpu.where(c, self._download(a), self._download(b))
        return self._upload(host)
