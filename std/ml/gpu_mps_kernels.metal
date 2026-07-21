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
#include <metal_simdgroup_matrix>
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

// Two-stage parallel reduction. This used to be a single GPU thread
// running a serial O(n) loop while every other thread returned
// immediately — correct, but using none of the GPU's actual parallelism:
// O(n) *span* (critical-path length), not just O(n) total work, on
// hardware built to do thousands of things at once. Stage 1 (this
// kernel): grid-stride accumulate (each thread sums every
// (SUM_TG_SIZE * threadgroup-count)-th element, so a BOUNDED number of
// threadgroups covers arbitrarily large n) followed by a binary-tree
// reduction in threadgroup memory — O(log SUM_TG_SIZE) span per
// threadgroup instead of O(n). Each threadgroup emits one partial sum.
// Stage 2 (mps_sum_f32 in gpu_mps.sc): the partial-sums array is capped
// at SUM_MAX_GROUPS entries regardless of n, small enough to finish on
// the CPU after one readback rather than needing a second GPU dispatch.
#define SUM_TG_SIZE 256

kernel void sum_kernel(device const float* a [[buffer(0)]],
                        device float* partialOut [[buffer(1)]],
                        constant uint& n [[buffer(2)]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint tgCount [[threadgroups_per_grid]]) {
    threadgroup float shared[SUM_TG_SIZE];

    uint globalStride = tgCount * SUM_TG_SIZE;
    uint idx = tgid * SUM_TG_SIZE + tid;
    float acc = 0.0;
    while (idx < n) {
        acc += a[idx];
        idx += globalStride;
    }
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = SUM_TG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) { shared[tid] += shared[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) { partialOut[tgid] = shared[0]; }
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

// In-place SGD step: w[id] -= lr * grad[id]. Reads and writes the SAME
// buffer -- safe because each thread only ever touches its own index, no
// thread reads another's element, so there's no ordering hazard within
// the dispatch. Lets a weight tensor's device buffer stay the single
// source of truth across an entire training run: nothing ever re-uploads
// it from a CPU array (there's no CPU array in the loop at all, once
// this runs) and nothing ever reads it back except the caller's own
// choice to do so when actually done training.
kernel void sgd_update_kernel(device float* w [[buffer(0)]],
                               device const float* grad [[buffer(1)]],
                               constant float& lr [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    w[id] = w[id] - lr * grad[id];
}

// ── simdgroup_matrix (hardware matrix-unit) GEMM kernels ────────────────────
// Apple Silicon GPUs (A14/M1 and later) have dedicated matrix-multiply
// hardware, exposed in MSL as simdgroup_matrix -- one simdgroup (32
// threads) cooperatively holds an 8x8 tile and simdgroup_multiply_
// accumulate does a full 8x8x8 multiply-add in hardware, instead of the
// tiled kernels above doing it as 512 individual scalar multiply-adds
// across 256 threads. This is the same class of hardware
// MPSMatrixMultiplication (and so PyTorch-MPS/MLX's fastest paths) uses.
//
// Each threadgroup here is exactly one simdgroup (32 threads) computing
// exactly one 8x8 output tile, accumulating over the reduction dimension
// in 8-wide steps -- the simplest correct simdgroup_matrix layout. One
// further tier below adds threadgroup-memory tiling on top (matmul_
// kernel_smma_multi, 4 simdgroups / 32x32 tile).
//
// simdgroup_load/store past a matrix's real bounds is undefined behavior
// (can read/write out of bounds device memory) -- unlike the tiled
// kernels above, which have explicit boundary checks for shapes smaller
// than one tile, these kernels have NO boundary handling at all. Callers
// (gpu_mps.sc) MUST only dispatch these when every relevant dimension is
// an exact multiple of 8, and fall back to the tiled kernels otherwise.

// out[M,N] = a[M,K] . b[K,N]
kernel void matmul_kernel_smma(device const float* a [[buffer(0)]],
                                device const float* b [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                constant uint& M [[buffer(3)]],
                                constant uint& K [[buffer(4)]],
                                constant uint& N [[buffer(5)]],
                                uint2 tgid [[threadgroup_position_in_grid]]) {
    uint row = tgid.y * 8;
    uint col = tgid.x * 8;

    simdgroup_float8x8 acc = simdgroup_float8x8(0.0f);
    uint numTiles = K / 8;
    for (uint t = 0; t < numTiles; t++) {
        uint p = t * 8;
        simdgroup_float8x8 tileA;
        simdgroup_float8x8 tileB;
        simdgroup_load(tileA, a + row * K + p, K);
        simdgroup_load(tileB, b + p * N + col, N);
        simdgroup_multiply_accumulate(acc, tileA, tileB, acc);
    }
    simdgroup_store(acc, out + row * N + col, N);
}

// dA[M,K] = a[M,N] . b^T, b stored [K,N] -- same math as matmul_abt_kernel
// (see its comment), 'b' read transposed via simdgroup_load's built-in
// transpose_matrix flag instead of a manual transposed-index load.
kernel void matmul_abt_kernel_smma(device const float* a [[buffer(0)]],
                                    device const float* b [[buffer(1)]],
                                    device float* out [[buffer(2)]],
                                    constant uint& M [[buffer(3)]],
                                    constant uint& N [[buffer(4)]],
                                    constant uint& K [[buffer(5)]],
                                    uint2 tgid [[threadgroup_position_in_grid]]) {
    uint row = tgid.y * 8; // M dim
    uint col = tgid.x * 8; // K dim (output cols)

    simdgroup_float8x8 acc = simdgroup_float8x8(0.0f);
    uint numTiles = N / 8;
    for (uint t = 0; t < numTiles; t++) {
        uint p = t * 8;
        simdgroup_float8x8 tileA;
        simdgroup_float8x8 tileBT;
        simdgroup_load(tileA, a + row * N + p, N);
        simdgroup_load(tileBT, b + col * N + p, N, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc, tileA, tileBT, acc);
    }
    simdgroup_store(acc, out + row * K + col, K);
}

// dB[K,N] = a^T . b, a stored [M,K] -- same math as matmul_atb_kernel, 'a'
// read transposed via simdgroup_load's transpose_matrix flag.
kernel void matmul_atb_kernel_smma(device const float* a [[buffer(0)]],
                                    device const float* b [[buffer(1)]],
                                    device float* out [[buffer(2)]],
                                    constant uint& M [[buffer(3)]],
                                    constant uint& K [[buffer(4)]],
                                    constant uint& N [[buffer(5)]],
                                    uint2 tgid [[threadgroup_position_in_grid]]) {
    uint row = tgid.y * 8; // K dim (output rows)
    uint col = tgid.x * 8; // N dim (output cols)

    simdgroup_float8x8 acc = simdgroup_float8x8(0.0f);
    uint numTiles = M / 8;
    for (uint t = 0; t < numTiles; t++) {
        uint p = t * 8;
        simdgroup_float8x8 tileAT;
        simdgroup_float8x8 tileB;
        simdgroup_load(tileAT, a + p * K + row, K, ulong2(0, 0), true);
        simdgroup_load(tileB, b + p * N + col, N);
        simdgroup_multiply_accumulate(acc, tileAT, tileB, acc);
    }
    simdgroup_store(acc, out + row * N + col, N);
}

// ── Multi-simdgroup tiled simdgroup_matrix kernels ──────────────────────────
// The single-simdgroup kernels above launch one simdgroup (32 threads) per
// 8x8 output tile and re-read every operand tile straight from device
// memory on every K-step -- correct, and already using the hardware matrix
// units, but with no data reuse across a threadgroup's neighbors and no
// reuse across a K-step beyond what one 8x8 load already gets. These
// kernels combine BOTH optimizations PyTorch's MPSMatrixMultiplication and
// MLX's fastest GEMM paths use together: standard threadgroup-memory tiling
// (like matmul_kernel above) to cut device-memory traffic, PLUS
// simdgroup_matrix hardware multiply-accumulate (like the kernels above)
// for the compute itself -- instead of either alone.
//
// Each threadgroup computes one 32x32 output tile using 4 simdgroups (128
// threads) arranged 2x2; each simdgroup owns a 16x16 sub-region (four 8x8
// accumulator registers) within it. Every reduction-dimension step
// cooperatively loads one 32x32 tile of each operand into threadgroup
// memory ONCE per threadgroup (not once per simdgroup), and all 4
// simdgroups then read their 8x8 sub-tiles for simdgroup_multiply_
// accumulate out of that shared on-chip copy instead of device memory.
//
// Same undefined-behavior-on-misaligned-bounds caveat as the kernels above,
// one tier stricter: dispatch MUST have M, K, and N (all three) exact
// multiples of 32 (not just 8) -- gpu_mps.sc only routes here when that
// holds, falling back to the single-simdgroup _smma kernels (8-aligned) or
// the plain tiled kernels (any shape) otherwise. Dispatch must also use
// exactly 128 threads (4 simdgroups) per threadgroup.
#define SMMA_MULTI_TILE 32

// out[M,N] = a[M,K] . b[K,N]
kernel void matmul_kernel_smma_multi(device const float* a [[buffer(0)]],
                                      device const float* b [[buffer(1)]],
                                      device float* out [[buffer(2)]],
                                      constant uint& M [[buffer(3)]],
                                      constant uint& K [[buffer(4)]],
                                      constant uint& N [[buffer(5)]],
                                      uint2 tgid [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float Atile[SMMA_MULTI_TILE][SMMA_MULTI_TILE];
    threadgroup float Btile[SMMA_MULTI_TILE][SMMA_MULTI_TILE];

    uint tgRow = tgid.y * SMMA_MULTI_TILE;
    uint tgCol = tgid.x * SMMA_MULTI_TILE;
    uint simdRow = sgid / 2;
    uint simdCol = sgid % 2;
    uint outRow0 = tgRow + simdRow * 16;
    uint outCol0 = tgCol + simdCol * 16;

    simdgroup_float8x8 acc00 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc01 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc10 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc11 = simdgroup_float8x8(0.0f);

    uint numKTiles = K / SMMA_MULTI_TILE;
    for (uint kt = 0; kt < numKTiles; kt++) {
        uint kBase = kt * SMMA_MULTI_TILE;
        for (uint e = 0; e < 8; e++) {
            uint idx = tid + e * 128;
            uint r = idx / SMMA_MULTI_TILE;
            uint c = idx % SMMA_MULTI_TILE;
            Atile[r][c] = a[(tgRow + r) * K + (kBase + c)];
            Btile[r][c] = b[(kBase + r) * N + (tgCol + c)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < SMMA_MULTI_TILE; kk += 8) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, &Atile[simdRow * 16][kk], SMMA_MULTI_TILE);
            simdgroup_load(a1, &Atile[simdRow * 16 + 8][kk], SMMA_MULTI_TILE);
            simdgroup_load(b0, &Btile[kk][simdCol * 16], SMMA_MULTI_TILE);
            simdgroup_load(b1, &Btile[kk][simdCol * 16 + 8], SMMA_MULTI_TILE);
            simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc00, out + outRow0 * N + outCol0, N);
    simdgroup_store(acc01, out + outRow0 * N + outCol0 + 8, N);
    simdgroup_store(acc10, out + (outRow0 + 8) * N + outCol0, N);
    simdgroup_store(acc11, out + (outRow0 + 8) * N + outCol0 + 8, N);
}

// dA[M,K] = a[M,N] . b^T, b stored [K,N] -- multi-simdgroup version of
// matmul_abt_kernel_smma above (see its comment for the math); the
// transposed read of 'b' is done by the cooperative-load's index formula
// (BTtile[p][k] = b[k,p]) instead of simdgroup_load's transpose flag, since
// the transpose now has to happen once per threadgroup-tile into shared
// memory, not once per simdgroup_load.
kernel void matmul_abt_kernel_smma_multi(device const float* a [[buffer(0)]],
                                          device const float* b [[buffer(1)]],
                                          device float* out [[buffer(2)]],
                                          constant uint& M [[buffer(3)]],
                                          constant uint& N [[buffer(4)]],
                                          constant uint& K [[buffer(5)]],
                                          uint2 tgid [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float Atile[SMMA_MULTI_TILE][SMMA_MULTI_TILE];  // [M-local][N-local]
    threadgroup float BTtile[SMMA_MULTI_TILE][SMMA_MULTI_TILE]; // [N-local][K-local], b read transposed

    uint tgRow = tgid.y * SMMA_MULTI_TILE; // M
    uint tgCol = tgid.x * SMMA_MULTI_TILE; // K (output cols)
    uint simdRow = sgid / 2;
    uint simdCol = sgid % 2;
    uint outRow0 = tgRow + simdRow * 16;
    uint outCol0 = tgCol + simdCol * 16;

    simdgroup_float8x8 acc00 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc01 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc10 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc11 = simdgroup_float8x8(0.0f);

    uint numTiles = N / SMMA_MULTI_TILE;
    for (uint t = 0; t < numTiles; t++) {
        uint nBase = t * SMMA_MULTI_TILE;
        for (uint e = 0; e < 8; e++) {
            uint idx = tid + e * 128;
            uint r = idx / SMMA_MULTI_TILE;
            uint c = idx % SMMA_MULTI_TILE;
            Atile[r][c] = a[(tgRow + r) * N + (nBase + c)];
            BTtile[r][c] = b[(tgCol + c) * N + (nBase + r)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < SMMA_MULTI_TILE; kk += 8) {
            simdgroup_float8x8 a0, a1, bt0, bt1;
            simdgroup_load(a0, &Atile[simdRow * 16][kk], SMMA_MULTI_TILE);
            simdgroup_load(a1, &Atile[simdRow * 16 + 8][kk], SMMA_MULTI_TILE);
            simdgroup_load(bt0, &BTtile[kk][simdCol * 16], SMMA_MULTI_TILE);
            simdgroup_load(bt1, &BTtile[kk][simdCol * 16 + 8], SMMA_MULTI_TILE);
            simdgroup_multiply_accumulate(acc00, a0, bt0, acc00);
            simdgroup_multiply_accumulate(acc01, a0, bt1, acc01);
            simdgroup_multiply_accumulate(acc10, a1, bt0, acc10);
            simdgroup_multiply_accumulate(acc11, a1, bt1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc00, out + outRow0 * K + outCol0, K);
    simdgroup_store(acc01, out + outRow0 * K + outCol0 + 8, K);
    simdgroup_store(acc10, out + (outRow0 + 8) * K + outCol0, K);
    simdgroup_store(acc11, out + (outRow0 + 8) * K + outCol0 + 8, K);
}

// dB[K,N] = a^T . b, a stored [M,K] -- multi-simdgroup version of
// matmul_atb_kernel_smma above; the transposed read of 'a' is done by the
// cooperative-load's index formula (ATtile[k][p] = a[p,k]), same reasoning
// as matmul_abt_kernel_smma_multi's BTtile above.
kernel void matmul_atb_kernel_smma_multi(device const float* a [[buffer(0)]],
                                          device const float* b [[buffer(1)]],
                                          device float* out [[buffer(2)]],
                                          constant uint& M [[buffer(3)]],
                                          constant uint& K [[buffer(4)]],
                                          constant uint& N [[buffer(5)]],
                                          uint2 tgid [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float ATtile[SMMA_MULTI_TILE][SMMA_MULTI_TILE]; // [K-local][M-local], a read transposed
    threadgroup float Btile[SMMA_MULTI_TILE][SMMA_MULTI_TILE];  // [M-local][N-local]

    uint tgRow = tgid.y * SMMA_MULTI_TILE; // K (output rows)
    uint tgCol = tgid.x * SMMA_MULTI_TILE; // N (output cols)
    uint simdRow = sgid / 2;
    uint simdCol = sgid % 2;
    uint outRow0 = tgRow + simdRow * 16;
    uint outCol0 = tgCol + simdCol * 16;

    simdgroup_float8x8 acc00 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc01 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc10 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc11 = simdgroup_float8x8(0.0f);

    uint numTiles = M / SMMA_MULTI_TILE;
    for (uint t = 0; t < numTiles; t++) {
        uint mBase = t * SMMA_MULTI_TILE;
        for (uint e = 0; e < 8; e++) {
            uint idx = tid + e * 128;
            uint r = idx / SMMA_MULTI_TILE;
            uint c = idx % SMMA_MULTI_TILE;
            ATtile[r][c] = a[(mBase + c) * K + (tgRow + r)];
            Btile[r][c] = b[(mBase + r) * N + (tgCol + c)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < SMMA_MULTI_TILE; kk += 8) {
            simdgroup_float8x8 at0, at1, b0, b1;
            simdgroup_load(at0, &ATtile[simdRow * 16][kk], SMMA_MULTI_TILE);
            simdgroup_load(at1, &ATtile[simdRow * 16 + 8][kk], SMMA_MULTI_TILE);
            simdgroup_load(b0, &Btile[kk][simdCol * 16], SMMA_MULTI_TILE);
            simdgroup_load(b1, &Btile[kk][simdCol * 16 + 8], SMMA_MULTI_TILE);
            simdgroup_multiply_accumulate(acc00, at0, b0, acc00);
            simdgroup_multiply_accumulate(acc01, at0, b1, acc01);
            simdgroup_multiply_accumulate(acc10, at1, b0, acc10);
            simdgroup_multiply_accumulate(acc11, at1, b1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc00, out + outRow0 * N + outCol0, N);
    simdgroup_store(acc01, out + outRow0 * N + outCol0 + 8, N);
    simdgroup_store(acc10, out + (outRow0 + 8) * N + outCol0, N);
    simdgroup_store(acc11, out + (outRow0 + 8) * N + outCol0 + 8, N);
}

// ── fp16 / bf16 matmul (see gpu_mps.h's mps_matmul_f16/mps_matmul_bf16) ─────
// Same tiled-threadgroup-memory shape as matmul_kernel above (the naive
// tier, not the smma/multi-simdgroup tiers -- those use
// simdgroup_float8x8, a float-only type in this Metal SDK, and porting
// them to half/bfloat accumulation is future work, not attempted here).
// Storage and multiply are in the reduced-precision type (real half-rate-
// or-better hardware throughput on Apple GPUs, and half the threadgroup
// memory traffic per tile vs matmul_kernel's float tiles); accumulation
// is in float, the standard mixed-precision-GEMM practice, so 16
// TILE_SIZE-deep partial-sum steps don't compound rounding error the way
// accumulating in half/bfloat directly would.
kernel void matmul_kernel_f16(device const half* a [[buffer(0)]],
                               device const half* b [[buffer(1)]],
                               device half* out [[buffer(2)]],
                               constant uint& M [[buffer(3)]],
                               constant uint& K [[buffer(4)]],
                               constant uint& N [[buffer(5)]],
                               uint2 tid [[thread_position_in_threadgroup]],
                               uint2 tgid [[threadgroup_position_in_grid]]) {
    threadgroup half Atile[TILE_SIZE][TILE_SIZE];
    threadgroup half Btile[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    float acc = 0.0;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + tid.x;
        uint bRow = t * TILE_SIZE + tid.y;
        Atile[tid.y][tid.x] = (row < M && aCol < K) ? a[row * K + aCol] : half(0.0);
        Btile[tid.y][tid.x] = (bRow < K && col < N) ? b[bRow * N + col] : half(0.0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE_SIZE; k++) {
            acc += float(Atile[tid.y][k]) * float(Btile[k][tid.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) {
        out[row * N + col] = half(acc);
    }
}

kernel void matmul_kernel_bf16(device const bfloat* a [[buffer(0)]],
                                device const bfloat* b [[buffer(1)]],
                                device bfloat* out [[buffer(2)]],
                                constant uint& M [[buffer(3)]],
                                constant uint& K [[buffer(4)]],
                                constant uint& N [[buffer(5)]],
                                uint2 tid [[thread_position_in_threadgroup]],
                                uint2 tgid [[threadgroup_position_in_grid]]) {
    threadgroup bfloat Atile[TILE_SIZE][TILE_SIZE];
    threadgroup bfloat Btile[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    float acc = 0.0;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + tid.x;
        uint bRow = t * TILE_SIZE + tid.y;
        Atile[tid.y][tid.x] = (row < M && aCol < K) ? a[row * K + aCol] : bfloat(0.0);
        Btile[tid.y][tid.x] = (bRow < K && col < N) ? b[bRow * N + col] : bfloat(0.0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE_SIZE; k++) {
            acc += float(Atile[tid.y][k]) * float(Btile[k][tid.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) {
        out[row * N + col] = bfloat(acc);
    }
}

// Elementwise relu in half/bfloat -- used by the fp16/bf16 tensor path's
// forward pass; storage/compute in the reduced-precision type throughout
// (no accumulation to worry about for an elementwise op).
kernel void relu_kernel_f16(device const half* a [[buffer(0)]],
                             device half* out [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= n) return;
    out[id] = max(a[id], half(0.0));
}

kernel void relu_kernel_bf16(device const bfloat* a [[buffer(0)]],
                              device bfloat* out [[buffer(1)]],
                              constant uint& n [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= n) return;
    out[id] = bfloat(max(float(a[id]), 0.0f));
}

// ── fp16 / bf16 backward-pass kernels (see gpu_mps.h's mps_*_f16/mps_*_bf16
// chained/persistent family) ────────────────────────────────────────────────
// Completes the set matmul_kernel_f16/bf16 and relu_kernel_f16/bf16 above
// started: enough ops (sub, scale, matmul^T two ways, relu-backward,
// in-place SGD) to run a full GPU-resident forward+backward+SGD training
// step in fp16/bf16, mirroring the float32 GPU-resident training loop's
// op set exactly (see gpu_mps.sc's mps_sub_f32_chained/mps_scale_f32_
// chained/mps_matmul_atb_f32_chained_ab/mps_matmul_abt_f32_persistent/
// mps_relu_backward_f32_chained_ab/mps_sgd_update_f32_chained). Elementwise
// ops (sub/scale/relu_backward/sgd_update) compute directly in the
// reduced-precision type -- no iterative accumulation to protect against
// rounding drift the way the matmul kernels' partial sums need. scale_
// kernel_f16/bf16 and sgd_update_kernel_f16/bf16 take their scalar (k / lr)
// as a plain float (promoted from half/bfloat operands at the point of
// multiply, not stored as half/bfloat) since a learning rate this small
// relative to typical gradient magnitudes is exactly the kind of value
// fp16/bf16's coarse relative precision would otherwise distort.

kernel void sub_kernel_f16(device const half* a [[buffer(0)]],
                            device const half* b [[buffer(1)]],
                            device half* out [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    out[id] = a[id] - b[id];
}

kernel void sub_kernel_bf16(device const bfloat* a [[buffer(0)]],
                             device const bfloat* b [[buffer(1)]],
                             device bfloat* out [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    out[id] = a[id] - b[id];
}

kernel void scale_kernel_f16(device const half* a [[buffer(0)]],
                              constant float& k [[buffer(1)]],
                              device half* out [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    out[id] = half(float(a[id]) * k);
}

kernel void scale_kernel_bf16(device const bfloat* a [[buffer(0)]],
                               constant float& k [[buffer(1)]],
                               device bfloat* out [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    out[id] = bfloat(float(a[id]) * k);
}

// dA[M,K] = dC[M,N] . B^T[N,K], B stored [K,N] -- see matmul_abt_kernel's
// comment for the shape/transpose reasoning; same tiling here, half/
// bfloat storage, float accumulation.
kernel void matmul_abt_kernel_f16(device const half* a [[buffer(0)]],
                                   device const half* b [[buffer(1)]],
                                   device half* out [[buffer(2)]],
                                   constant uint& M [[buffer(3)]],
                                   constant uint& N [[buffer(4)]],
                                   constant uint& K [[buffer(5)]],
                                   uint2 tid [[thread_position_in_threadgroup]],
                                   uint2 tgid [[threadgroup_position_in_grid]]) {
    threadgroup half Atile[TILE_SIZE][TILE_SIZE];
    threadgroup half Btile[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    float acc = 0.0;
    uint numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + tid.x;
        uint bCol = t * TILE_SIZE + tid.y;
        Atile[tid.y][tid.x] = (row < M && aCol < N) ? a[row * N + aCol] : half(0.0);
        Btile[tid.y][tid.x] = (col < K && bCol < N) ? b[col * N + bCol] : half(0.0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE_SIZE; k++) {
            acc += float(Atile[tid.y][k]) * float(Btile[k][tid.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < K) {
        out[row * K + col] = half(acc);
    }
}

kernel void matmul_abt_kernel_bf16(device const bfloat* a [[buffer(0)]],
                                    device const bfloat* b [[buffer(1)]],
                                    device bfloat* out [[buffer(2)]],
                                    constant uint& M [[buffer(3)]],
                                    constant uint& N [[buffer(4)]],
                                    constant uint& K [[buffer(5)]],
                                    uint2 tid [[thread_position_in_threadgroup]],
                                    uint2 tgid [[threadgroup_position_in_grid]]) {
    threadgroup bfloat Atile[TILE_SIZE][TILE_SIZE];
    threadgroup bfloat Btile[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    float acc = 0.0;
    uint numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + tid.x;
        uint bCol = t * TILE_SIZE + tid.y;
        Atile[tid.y][tid.x] = (row < M && aCol < N) ? a[row * N + aCol] : bfloat(0.0);
        Btile[tid.y][tid.x] = (col < K && bCol < N) ? b[col * N + bCol] : bfloat(0.0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE_SIZE; k++) {
            acc += float(Atile[tid.y][k]) * float(Btile[k][tid.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < K) {
        out[row * K + col] = bfloat(acc);
    }
}

// dB[K,N] = A^T[K,M] . dC[M,N], A stored [M,K] -- see matmul_atb_kernel's
// comment; same tiling here, half/bfloat storage, float accumulation.
kernel void matmul_atb_kernel_f16(device const half* a [[buffer(0)]],
                                   device const half* b [[buffer(1)]],
                                   device half* out [[buffer(2)]],
                                   constant uint& M [[buffer(3)]],
                                   constant uint& K [[buffer(4)]],
                                   constant uint& N [[buffer(5)]],
                                   uint2 tid [[thread_position_in_threadgroup]],
                                   uint2 tgid [[threadgroup_position_in_grid]]) {
    threadgroup half Atile[TILE_SIZE][TILE_SIZE];
    threadgroup half Btile[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    float acc = 0.0;
    uint numTiles = (M + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < numTiles; t++) {
        uint aP = t * TILE_SIZE + tid.x;
        uint bP = t * TILE_SIZE + tid.y;
        Atile[tid.x][tid.y] = (aP < M && row < K) ? a[aP * K + row] : half(0.0);
        Btile[tid.y][tid.x] = (bP < M && col < N) ? b[bP * N + col] : half(0.0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE_SIZE; k++) {
            acc += float(Atile[k][tid.y]) * float(Btile[k][tid.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < K && col < N) {
        out[row * N + col] = half(acc);
    }
}

kernel void matmul_atb_kernel_bf16(device const bfloat* a [[buffer(0)]],
                                    device const bfloat* b [[buffer(1)]],
                                    device bfloat* out [[buffer(2)]],
                                    constant uint& M [[buffer(3)]],
                                    constant uint& K [[buffer(4)]],
                                    constant uint& N [[buffer(5)]],
                                    uint2 tid [[thread_position_in_threadgroup]],
                                    uint2 tgid [[threadgroup_position_in_grid]]) {
    threadgroup bfloat Atile[TILE_SIZE][TILE_SIZE];
    threadgroup bfloat Btile[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    float acc = 0.0;
    uint numTiles = (M + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < numTiles; t++) {
        uint aP = t * TILE_SIZE + tid.x;
        uint bP = t * TILE_SIZE + tid.y;
        Atile[tid.x][tid.y] = (aP < M && row < K) ? a[aP * K + row] : bfloat(0.0);
        Btile[tid.y][tid.x] = (bP < M && col < N) ? b[bP * N + col] : bfloat(0.0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE_SIZE; k++) {
            acc += float(Atile[k][tid.y]) * float(Btile[k][tid.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < K && col < N) {
        out[row * N + col] = bfloat(acc);
    }
}

kernel void relu_backward_kernel_f16(device const half* a [[buffer(0)]],
                                      device const half* selfGrad [[buffer(1)]],
                                      device half* out [[buffer(2)]],
                                      uint id [[thread_position_in_grid]]) {
    out[id] = (float(a[id]) > 0.0f) ? selfGrad[id] : half(0.0);
}

kernel void relu_backward_kernel_bf16(device const bfloat* a [[buffer(0)]],
                                       device const bfloat* selfGrad [[buffer(1)]],
                                       device bfloat* out [[buffer(2)]],
                                       uint id [[thread_position_in_grid]]) {
    out[id] = (float(a[id]) > 0.0f) ? selfGrad[id] : bfloat(0.0);
}

// In-place SGD step in fp16/bf16 -- w[id] -= lr*grad[id], read+write same
// buffer, safe for the same reason sgd_update_kernel's comment gives
// (each thread only ever touches its own index). lr stays a plain float
// parameter (see this section's header comment) even though w/grad are
// half/bfloat.
kernel void sgd_update_kernel_f16(device half* w [[buffer(0)]],
                                   device const half* grad [[buffer(1)]],
                                   constant float& lr [[buffer(2)]],
                                   uint id [[thread_position_in_grid]]) {
    w[id] = half(float(w[id]) - lr * float(grad[id]));
}

kernel void sgd_update_kernel_bf16(device bfloat* w [[buffer(0)]],
                                    device const bfloat* grad [[buffer(1)]],
                                    constant float& lr [[buffer(2)]],
                                    uint id [[thread_position_in_grid]]) {
    w[id] = bfloat(float(w[id]) - lr * float(grad[id]));
}
