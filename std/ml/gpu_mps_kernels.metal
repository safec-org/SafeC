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

// ── Tiled GEMM kernels ────────────────────────────────────────────────────────
// out[M,N] = a[M,K] . b[K,N], and the two matmul-backward variants below,
// used to each be one-thread-per-output-element with no data reuse: every
// output element re-read its whole row of A and column of B from device
// memory independently, so each element of A/B was fetched from device
// memory N (or M) times over. Standard threadgroup-memory tiling instead:
// each threadgroup cooperatively loads one TILE_SIZE x TILE_SIZE tile of A
// and B into fast on-chip threadgroup memory, then every thread in that
// threadgroup reuses those same cached values for its own accumulation —
// each element of A/B is now fetched from device memory once per tile
// instead of once per output element that uses it. This is the standard
// GEMM optimization PyTorch's MPSMatrixMultiplication and MLX's own tiled
// kernels both rely on (as does every other real GEMM implementation);
// still short of Apple Silicon's simdgroup_matrix hardware matrix-multiply
// units, which is the next tier up and not attempted here.
//
// Dispatch MUST use exactly TILE_SIZE x TILE_SIZE threads per threadgroup
// (see gpu_mps.sc — the old dispatch shrank the threadgroup for M/N/K under
// 16, which would leave part of the shared tile uninitialized here); the
// boundary checks below (row/col/tile-index against M/N/K) are what make
// that fixed-size dispatch safe for matrices smaller than one tile, not
// just ones that divide evenly by it.
#define TILE_SIZE 16

// out[M,N] = a[M,K] . b[K,N]
kernel void matmul_kernel(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           constant uint& M [[buffer(3)]],
                           constant uint& K [[buffer(4)]],
                           constant uint& N [[buffer(5)]],
                           uint2 tid [[thread_position_in_threadgroup]],
                           uint2 tgid [[threadgroup_position_in_grid]]) {
    threadgroup float Atile[TILE_SIZE][TILE_SIZE];
    threadgroup float Btile[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    float acc = 0.0;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + tid.x;
        uint bRow = t * TILE_SIZE + tid.y;
        Atile[tid.y][tid.x] = (row < M && aCol < K) ? a[row * K + aCol] : 0.0;
        Btile[tid.y][tid.x] = (bRow < K && col < N) ? b[bRow * N + col] : 0.0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE_SIZE; k++) {
            acc += Atile[tid.y][k] * Btile[k][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) {
        out[row * N + col] = acc;
    }
}

// ── matmul backward (see gpu_mps.sc's __matmul_backward_gpu comment) ─────────
// dA[M,K] = dC[M,N] . B^T[N,K], B stored [K,N] -- same math tensor_blas.sc's
// __matmul_backward_blas passes to cblas_dgemm with a transpose flag; here
// it's a kernel that reads b transposed (each Btile load pulls from a
// column of B, not a row) instead of a BLAS flag.
kernel void matmul_abt_kernel(device const float* a [[buffer(0)]],
                               device const float* b [[buffer(1)]],
                               device float* out [[buffer(2)]],
                               constant uint& M [[buffer(3)]],
                               constant uint& N [[buffer(4)]],
                               constant uint& K [[buffer(5)]],
                               uint2 tid [[thread_position_in_threadgroup]],
                               uint2 tgid [[threadgroup_position_in_grid]]) {
    threadgroup float Atile[TILE_SIZE][TILE_SIZE];
    threadgroup float Btile[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + tid.y; // M dim
    uint col = tgid.x * TILE_SIZE + tid.x; // K dim (output cols)

    float acc = 0.0;
    uint numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + tid.x; // N-dim index for A's row-major load
        uint bCol = t * TILE_SIZE + tid.y; // N-dim index for B's transposed load
        Atile[tid.y][tid.x] = (row < M && aCol < N) ? a[row * N + aCol] : 0.0;
        Btile[tid.y][tid.x] = (col < K && bCol < N) ? b[col * N + bCol] : 0.0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE_SIZE; k++) {
            acc += Atile[tid.y][k] * Btile[k][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < K) {
        out[row * K + col] = acc;
    }
}

// dB[K,N] = A^T[K,M] . dC[M,N], A stored [M,K].
kernel void matmul_atb_kernel(device const float* a [[buffer(0)]],
                               device const float* b [[buffer(1)]],
                               device float* out [[buffer(2)]],
                               constant uint& M [[buffer(3)]],
                               constant uint& K [[buffer(4)]],
                               constant uint& N [[buffer(5)]],
                               uint2 tid [[thread_position_in_threadgroup]],
                               uint2 tgid [[threadgroup_position_in_grid]]) {
    threadgroup float Atile[TILE_SIZE][TILE_SIZE];
    threadgroup float Btile[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + tid.y; // K dim (output rows)
    uint col = tgid.x * TILE_SIZE + tid.x; // N dim (output cols)

    float acc = 0.0;
    uint numTiles = (M + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < numTiles; t++) {
        uint aP = t * TILE_SIZE + tid.x; // M-dim index for A's transposed load
        uint bP = t * TILE_SIZE + tid.y; // M-dim index for B's row-major load
        Atile[tid.x][tid.y] = (aP < M && row < K) ? a[aP * K + row] : 0.0;
        Btile[tid.y][tid.x] = (bP < M && col < N) ? b[bP * N + col] : 0.0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE_SIZE; k++) {
            acc += Atile[k][tid.y] * Btile[k][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < K && col < N) {
        out[row * N + col] = acc;
    }
}

// relu backward: out[i] = a[i] > 0 ? selfGrad[i] : 0 -- fed to
// __mps_run_binary_kernel like add/sub/mul, just a different one-line body.
kernel void relu_backward_kernel(device const float* a [[buffer(0)]],
                                  device const float* selfGrad [[buffer(1)]],
                                  device float* out [[buffer(2)]],
                                  uint id [[thread_position_in_grid]]) {
    out[id] = (a[id] > 0.0f) ? selfGrad[id] : 0.0f;
}
