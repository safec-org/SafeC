// SafeC Standard Library — Metal compute kernels for std::ml's MPS backend
// (see gpu_mps.sc). Canonical source for every kernel gpu_mps.sc dispatches.
//
// This file is compiled OFFLINE, not at process runtime: gpu_mps.sc used to
// hand each kernel's source to newLibraryWithSource:options:error: from a
// process-startup C string and let Metal's own MSL compiler build it on the
// first call (cached after that, but still a real front-end compile the
// first time every single op ran). Compiling this file ahead of time with
// Apple's `metal`/`metallib` tools produces a .metallib (compiled AIR
// bytecode) that gpu_mps.sc loads directly via newLibraryWithData:error: —
// the MSL-source-to-AIR step happens once, here, at build time, not once per
// process for every op. Regenerate the embedded copy (gpu_mps_metallib.h)
// with gen_mps_metallib.sh after editing this file.
#include <metal_stdlib>
using namespace metal;

kernel void add_kernel(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = a[id] + b[id];
}

kernel void sub_kernel(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = a[id] - b[id];
}

kernel void mul_kernel(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * b[id];
}

kernel void div_kernel(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = a[id] / b[id];
}

kernel void pow_kernel(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = pow(a[id], b[id]);
}

kernel void log_kernel(device const float* a [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = log(a[id]);
}

kernel void exp_kernel(device const float* a [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = exp(a[id]);
}

kernel void sqrt_kernel(device const float* a [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = sqrt(a[id]);
}

kernel void scale_kernel(device const float* a [[buffer(0)]],
                          constant float& k [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * k;
}

// Serial single-thread reduction — see gpu_mps.sc's mps_sum_f32 comment for
// why this isn't a real parallel reduction.
kernel void sum_kernel(device const float* a [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& n [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    if (id != 0) return;
    float acc = 0.0;
    for (uint i = 0; i < n; i++) { acc += a[i]; }
    out[0] = acc;
}

kernel void relu_kernel(device const float* a [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = max(a[id], 0.0f);
}

// out[M,N] = a[M,K] . b[K,N] — see gpu_mps.sc's mps_matmul_f32 comment for
// why this is a naive one-thread-per-output-element kernel, not a tuned GEMM.
kernel void matmul_kernel(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           constant uint& M [[buffer(3)]],
                           constant uint& K [[buffer(4)]],
                           constant uint& N [[buffer(5)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= N || gid.y >= M) return;
    float acc = 0.0;
    for (uint p = 0; p < K; p++) {
        acc += a[gid.y * K + p] * b[p * N + gid.x];
    }
    out[gid.y * N + gid.x] = acc;
}
