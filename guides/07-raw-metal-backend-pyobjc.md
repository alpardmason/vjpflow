# 07 - Raw Metal Backend with PyObjC

**Code:** `src/autograd/backends/metal_backend.py`, `backends/metal/kernels.metal`

Now we turn the concepts from [06](06-intro-to-metal.md) into a working GPU
backend. We drive Apple's Metal C API directly from Python via **PyObjC**, which
exposes Objective-C frameworks as Python modules. Method names follow a
mechanical translation: Objective-C `newLibraryWithSource:options:error:`
becomes Python `newLibraryWithSource_options_error_(...)`.

> Requires an Apple Silicon Mac with `uv sync --extra metal`. On other machines
> this backend is unavailable and the parity tests skip.

## The shaders: `kernels.metal`

A handful of kernels written in MSL. Elementwise ops are generated with a macro
so each is a one-line expression:

```metal
#define EW_BINARY(NAME, EXPR)                                            \
kernel void NAME(device const float* a [[buffer(0)]],                    \
                 device const float* b [[buffer(1)]],                    \
                 device float* out      [[buffer(2)]],                   \
                 constant uint& count   [[buffer(3)]],                   \
                 uint gid [[thread_position_in_grid]]) {                 \
    if (gid >= count) return;                                           \
    out[gid] = (EXPR);                                                  \
}
EW_BINARY(ew_add, a[gid] + b[gid])   // (simplified)
```

Plus a naive `matmul` (one thread computes one output element via a `K`-loop) and
two last-axis reductions (`reduce_sum_lastdim`, `reduce_max_lastdim`: one thread
reduces one row). Read the file -- it is short and every line is commented.

## Startup: device, library, pipelines (once)

`MetalBackend.__init__` builds the expensive, reusable objects exactly once:

```python
self._device = Metal.MTLCreateSystemDefaultDevice()
source = _KERNEL_SOURCE.read_text()
library, error = self._device.newLibraryWithSource_options_error_(source, None, None)
self._queue = self._device.newCommandQueue()
for name in [...kernel names...]:
    fn = self._library.newFunctionWithName_(name)
    pipeline, err = self._device.newComputePipelineStateWithFunction_error_(fn, None)
    self._pipelines[name] = pipeline
```

We compile the `.metal` source at runtime (no separate build step) and cache one
pipeline per kernel. A compile error here is almost always a typo in the shader,
so we surface it loudly.

> [!note] PyObjC error-handling convention
> Objective-C methods with an `NSError**` out-parameter return a `(result,
> error)` tuple in PyObjC. That is why we write
> `library, error = ...newLibraryWithSource_options_error_(...)`.

## Buffers: the unified-memory payoff

```python
def _upload(self, array):
    data = np.ascontiguousarray(array, dtype=np.float32)
    buf = self._device.newBufferWithBytes_length_options_(
        data.tobytes(), data.nbytes, Metal.MTLResourceStorageModeShared)
    return MetalArray(buf, data.shape)

def _download(self, arr):
    mv = arr.buffer.contents().as_buffer(arr.size * 4)
    return np.frombuffer(mv, dtype=np.float32, count=arr.size).reshape(arr.shape).copy()
```

`MTLResourceStorageModeShared` puts the buffer in memory both CPU and GPU see
(unified memory, [06](06-intro-to-metal.md)). `contents()` returns a pointer into
that shared region; `as_buffer` wraps it so numpy can read it with zero copy
(we `.copy()` only to detach from the GPU-owned memory).

## Dispatch: encode, commit, wait

The heart of every op is the same five-step encode:

```python
def _dispatch_1d(self, pipeline_name, buffers, count):
    cmd = self._queue.commandBuffer()
    enc = cmd.computeCommandEncoder()
    enc.setComputePipelineState_(self._pipelines[pipeline_name])
    for i, buf in enumerate(buffers):
        enc.setBuffer_offset_atIndex_(buf, 0, i)            # bind [[buffer(i)]]
    enc.setBytes_length_atIndex_(struct.pack("I", count), 4, len(buffers))
    enc.dispatchThreads_threadsPerThreadgroup_(             # launch `count` threads
        Metal.MTLSizeMake(count, 1, 1), Metal.MTLSizeMake(min(count, 256), 1, 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
```

Notes:
- `setBuffer_offset_atIndex_` binds each array to the `[[buffer(n)]]` slot the
  kernel expects. Small scalars (the element `count`, matmul dims) go through
  `setBytes_length_atIndex_` -- no need for a buffer.
- `dispatchThreads:threadsPerThreadgroup:` uses non-uniform threadgroups (an
  Apple GPU feature) so we can ask for exactly `count` threads; the kernel's
  bounds check covers the rounded-up tail.
- `commit()` submits asynchronously; `waitUntilCompleted()` blocks until the
  result is ready. (A real engine would pipeline many command buffers; we keep
  it synchronous for clarity.)

`matmul` and the reductions follow the same shape, but with a 2-D grid
(`MTLSizeMake(N, M, 1)`) and packed dimension structs. One alignment gotcha worth
knowing: Metal's `uint3` is 16-byte aligned, so we pad the matmul dims to four
uints (`struct.pack("IIII", M, K, N, 0)`).

## Honest simplifications

Read these as "exercises for the reader," all flagged in code comments:

- **Broadcasting** for elementwise ops is done by expanding operands to a common
  shape on the host before the kernel runs. A production engine passes strides
  into the shader so the GPU broadcasts directly.
- **mean** runs the GPU sum kernel, then divides the tiny reduced result on the
  host.
- **Index/shape ops** (`gather`, `scatter_add`, `concat`, `transpose`,
  `broadcast_to`, non-last-axis reductions, batched matmul) take a **numpy
  fallback**: download, compute on CPU, re-upload. They are memory movement with
  little arithmetic, so the GPU would not help much, and it keeps the GPU code
  focused.

## Proving it works: parity tests

`tests/test_backends.py` runs the same graph on both backends and asserts they
agree (skipped automatically if no GPU):

```python
cpu = (_on("numpy", a) * _on("numpy", b) + _on("numpy", a)).numpy()
gpu = (_on("metal", a) * _on("metal", b) + _on("metal", a)).numpy()
assert np.allclose(cpu, gpu, atol=1e-4)
```

The headline test, `test_small_mlp_end_to_end_parity`, runs a full
`softmax(relu(x @ w))` -- matmul, elementwise, and a reduction -- entirely on the
GPU and matches numpy. That is the "runs on the GPU" milestone.

## Design patterns in this chapter

- **Resource pooling**: build device/library/pipelines/queue once, reuse forever.
- **Adapter**: PyObjC adapts an Objective-C API to Python; `MetalArray` adapts a
  raw buffer to the engine's `Native` contract.
- **Graceful degradation**: numpy fallback for unimplemented ops; feature
  detection (`is_available`) so the package works without a GPU.

## What's next

With both backends in place, we climb back up the stack to build real layers:
[08 - Building Layers from Primitives](08-building-layers.md).
