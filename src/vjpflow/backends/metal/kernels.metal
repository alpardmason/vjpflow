// Metal compute shaders for the GPU backend.
//
// These kernels are deliberately *naive* -- one thread per output element, no
// tiling, no shared-memory blocking. The goal is to show the full Metal
// pipeline (device -> library -> pipeline state -> buffers -> threadgroups),
// not to compete with MPS/Accelerate. Guide 07 walks through every line.
//
// Conventions:
//   * All data is float (float32). Indices are computed from the global thread id.
//   * `gid` is the flattened global thread position. We bounds-check against
//     `count` because the host rounds the grid up to a multiple of the
//     threadgroup size.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Elementwise binary ops (operands already broadcast to equal shape by host).
// ---------------------------------------------------------------------------
#define EW_BINARY(NAME, EXPR)                                                  \
kernel void NAME(device const float* a   [[buffer(0)]],                        \
                 device const float* b   [[buffer(1)]],                        \
                 device float*       out [[buffer(2)]],                        \
                 constant uint&      count [[buffer(3)]],                      \
                 uint gid [[thread_position_in_grid]]) {                       \
    if (gid >= count) return;                                                  \
    float x = a[gid];                                                          \
    float y = b[gid];                                                          \
    out[gid] = (EXPR);                                                         \
}

EW_BINARY(ew_add, x + y)
EW_BINARY(ew_sub, x - y)
EW_BINARY(ew_mul, x * y)
EW_BINARY(ew_div, x / y)
EW_BINARY(ew_max, fmax(x, y))
EW_BINARY(ew_pow, pow(x, y))
EW_BINARY(ew_gt,  x > y ? 1.0f : 0.0f)

// ---------------------------------------------------------------------------
// Elementwise unary ops.
// ---------------------------------------------------------------------------
#define EW_UNARY(NAME, EXPR)                                                   \
kernel void NAME(device const float* a   [[buffer(0)]],                        \
                 device float*       out [[buffer(1)]],                        \
                 constant uint&      count [[buffer(2)]],                      \
                 uint gid [[thread_position_in_grid]]) {                       \
    if (gid >= count) return;                                                  \
    float x = a[gid];                                                          \
    out[gid] = (EXPR);                                                         \
}

EW_UNARY(un_neg,  -x)
EW_UNARY(un_exp,  exp(x))
EW_UNARY(un_log,  log(x))
EW_UNARY(un_sqrt, sqrt(x))
EW_UNARY(un_tanh, tanh(x))

// ---------------------------------------------------------------------------
// Naive matmul: out[M,N] = a[M,K] @ b[K,N]. One thread per output element.
// ---------------------------------------------------------------------------
kernel void matmul(device const float* a   [[buffer(0)]],
                   device const float* b   [[buffer(1)]],
                   device float*       out [[buffer(2)]],
                   constant uint3&     dims [[buffer(3)]],  // (M, K, N)
                   uint2 gid [[thread_position_in_grid]]) {
    uint M = dims.x, K = dims.y, N = dims.z;
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += a[row * K + k] * b[k * N + col];
    }
    out[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Reductions over the last axis: input [rows, cols] -> output [rows].
// One thread per row; the loop reduces that row's `cols` elements.
// ---------------------------------------------------------------------------
kernel void reduce_sum_lastdim(device const float* a   [[buffer(0)]],
                               device float*       out [[buffer(1)]],
                               constant uint2&     dims [[buffer(2)]],  // (rows, cols)
                               uint gid [[thread_position_in_grid]]) {
    uint rows = dims.x, cols = dims.y;
    if (gid >= rows) return;
    float acc = 0.0f;
    for (uint c = 0; c < cols; ++c) acc += a[gid * cols + c];
    out[gid] = acc;
}

kernel void reduce_max_lastdim(device const float* a   [[buffer(0)]],
                               device float*       out [[buffer(1)]],
                               constant uint2&     dims [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    uint rows = dims.x, cols = dims.y;
    if (gid >= rows) return;
    float acc = -INFINITY;
    for (uint c = 0; c < cols; ++c) acc = fmax(acc, a[gid * cols + c]);
    out[gid] = acc;
}
