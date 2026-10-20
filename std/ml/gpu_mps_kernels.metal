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

// Fused gated-activation kernel (GeGLU/tanh-approx-GELU shape): 'raw' is
// one row-major [rows, 2*intermediateSize] buffer holding a gate half and
// an up half side by side per row (the natural output of a single matmul
// against a row-concatenated [gateW; upW] weight matrix, e.g. this
// library's own gemma4 benchmark's gate/up projection fusion) --
// out[row,col] = gelu(raw[row,col]) * raw[row, intermediateSize+col] for
// col in [0, intermediateSize). One dispatch computes the entire gated
// activation (matmul -> gelu -> multiply's middle two steps) without ever
// writing the gate/up split or the post-gelu intermediate back to host
// memory. GELU here is the same tanh approximation as
// std::tensor_gelu_fwd (activations.sc) -- bit-identical formula, GPU
// float32 math instead of CPU, verified against it directly.
static inline float gelu_tanh(float x) {
    float c = 0.7978845608028654f; // sqrt(2/pi)
    float inner = c * (x + 0.044715f * x * x * x);
    // Clamp before tanh: this GPU's tanh() overflows to NaN (instead of
    // saturating to +-1, the way CPU libm's tanh does) for |inner| in the
    // few-hundred range, reachable from a perfectly ordinary, finite x
    // around 25-30 once cubed and scaled -- observed directly against the
    // real gemma4-E2B checkpoint (x=27.39 -> inner=755 -> tanh()=NaN on
    // this hardware/Metal build). tanh(20) already rounds to exactly
    // 1.0f in float32, so clamping to +-20 changes nothing for any input
    // this would otherwise get right, and fixes every input it wouldn't.
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

kernel void gated_gelu_kernel(device const float* raw [[buffer(0)]],
                               constant uint& intermediateSize [[buffer(1)]],
                               device float* out [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    uint col = id % intermediateSize;
    uint row = id / intermediateSize;
    uint base = row * (2u * intermediateSize);
    out[id] = gelu_tanh(raw[base + col]) * raw[base + intermediateSize + col];
}

// bf16 storage variant of gated_gelu_kernel above -- same [gate | up] row
// layout, GELU math widened to float32.
kernel void gated_gelu_kernel_bf16(device const bfloat* raw [[buffer(0)]],
                                     constant uint& intermediateSize [[buffer(1)]],
                                     device bfloat* out [[buffer(2)]],
                                     uint id [[thread_position_in_grid]]) {
    uint col = id % intermediateSize;
    uint row = id / intermediateSize;
    uint base = row * (2u * intermediateSize);
    float x = float(raw[base + col]);
    float up = float(raw[base + intermediateSize + col]);
    out[id] = bfloat(gelu_tanh(x) * up);
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

// out[i] = a[i] + b[i] with BOTH operands device buffers (sub_kernel_bf16's
// host wrapper uploads its second operand from a CPU array instead) -- for
// residual connections whose operands are both still-pending chained ops.
kernel void add_kernel_bf16(device const bfloat* a [[buffer(0)]],
                              device const bfloat* b [[buffer(1)]],
                              device bfloat* out [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    out[id] = a[id] + b[id];
}

// out[i] = gelu(a[i]) * b[i] -- gated_gelu_kernel_bf16 with the gate and
// up operands in two separate buffers rather than one [gate | up] row.
kernel void gelu_mul_kernel_bf16(device const bfloat* a [[buffer(0)]],
                                   device const bfloat* b [[buffer(1)]],
                                   device bfloat* out [[buffer(2)]],
                                   uint id [[thread_position_in_grid]]) {
    out[id] = bfloat(gelu_tanh(float(a[id])) * float(b[id]));
}

// Row-wise RMSNorm: out[row,c] = x[row,c] * weight[c] / sqrt(mean(x[row,:]^2)
// + eps). One RMSNORM_TG-thread threadgroup per row: every thread strides
// over the row accumulating a partial sum of squares, simd_sum folds each
// simdgroup's partials, and a second simd_sum over the per-simdgroup
// totals (staged through threadgroup memory) gives the row total -- same
// two-level shape as sum_kernel above, without its second dispatch since
// a row always fits one threadgroup. An earlier one-simdgroup-per-row
// version left most of the GPU idle and ran ~3-5x slower at hidden sizes
// 1536-2560. Dispatch MUST use RMSNORM_TG threads per threadgroup, 'rows'
// threadgroups -- see gpu_mps.sc's mps_rmsnorm_bf16_chained.
#define RMSNORM_TG 256
kernel void rmsnorm_kernel_bf16(device const bfloat* x [[buffer(0)]],
                                  device const bfloat* weight [[buffer(1)]],
                                  device bfloat* out [[buffer(2)]],
                                  constant uint& cols [[buffer(3)]],
                                  constant float& eps [[buffer(4)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint tid [[thread_position_in_threadgroup]],
                                  uint sgid [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sgSums[RMSNORM_TG / 32];
    device const bfloat* xRow = x + (ulong)row * (ulong)cols;
    device bfloat* outRow = out + (ulong)row * (ulong)cols;
    float partial = 0.0f;
    for (uint c = tid; c < cols; c += RMSNORM_TG) {
        float v = float(xRow[c]);
        partial += v * v;
    }
    partial = simd_sum(partial);
    if (lane == 0) { sgSums[sgid] = partial; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sumSq = simd_sum(lane < RMSNORM_TG / 32 ? sgSums[lane] : 0.0f);
    float invRms = rsqrt(sumSq / float(cols) + eps);
    for (uint c = tid; c < cols; c += RMSNORM_TG) {
        outRow[c] = bfloat(float(xRow[c]) * invRms * float(weight[c]));
    }
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

// A single-simdgroup simdgroup_matrix bf16 variant of matmul_abt_kernel_bf16
// was tried and reverted: at decode-step shapes (M tiny next to N/K) these
// matmuls are bandwidth-bound, and without threadgroup-memory tiling it
// measured ~13% slower than the tiled kernel above. Porting the f32
// "_multi" tier's tiling+simdgroup_matrix combination is the real next
// step if revisited.

// dA[1,K] = a[1,N] . B^T, B stored [K,N] -- the M==1 (matrix-vector) case
// of matmul_abt_kernel_bf16, where the 16x16 tiled kernel computes and
// discards 15 of every 16 rows. One simdgroup per output element: lanes
// stride over N reading 4 bf16 at a time (one 8-byte load per lane, 256
// contiguous bytes per simdgroup step) and simd_sum folds the partials; no
// threadgroup memory or barriers, since an M==1 tile has nothing to share.
// The op is bound by reading the K*N weights once -- the wide loads
// measured ~8-20% faster than one bf16 per lane on an M1 Pro, while
// several rows per simdgroup measured no better than one. The bfloat4
// path needs N % 4 == 0 (8-byte-aligned rows); other N use scalar loads.
#define GEMV_SIMDGROUPS 4
kernel void matmul_abt_kernel_bf16_gemv(device const bfloat* a [[buffer(0)]],
                                          device const bfloat* b [[buffer(1)]],
                                          device bfloat* out [[buffer(2)]],
                                          constant uint& N [[buffer(3)]],
                                          constant uint& K [[buffer(4)]],
                                          uint tgid [[threadgroup_position_in_grid]],
                                          uint sgid [[simdgroup_index_in_threadgroup]],
                                          uint lane [[thread_index_in_simdgroup]]) {
    uint k = tgid * GEMV_SIMDGROUPS + sgid;
    if (k >= K) { return; }
    device const bfloat* brow = b + (ulong)k * (ulong)N;
    float partial = 0.0f;
    if ((N & 3u) == 0u) {
        device const bfloat4* a4 = (device const bfloat4*)a;
        device const bfloat4* b4 = (device const bfloat4*)brow;
        for (uint i = lane; i < (N >> 2); i += 32) {
            partial += dot(float4(a4[i]), float4(b4[i]));
        }
    } else {
        for (uint n = lane; n < N; n += 32) {
            partial += float(a[n]) * float(brow[n]);
        }
    }
    float total = simd_sum(partial);
    if (lane == 0) {
        out[k] = bfloat(total);
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

// ── Batched multi-head attention (bf16), decode-step shape ─────────────────
// One new query token per sequence against that sequence's K/V history:
// out[b,h,:] = softmax(q[b,h,:] . K[b,0:len,kvHead,:]^T) . V[b,0:len,kvHead,:]
// (no 1/sqrt(headDim) factor -- callers fold any scaling into q), with
// grouped-query heads (kvHead = h / (numHeads/numKvHeads)) and an optional
// sliding window. 'cacheLens[b]' is the post-append KV length, so the query
// sits at position len-1 and causal masking never excludes anything; only
// the window can, and its masked prefix is skipped rather than visited.
// 'kAll'/'vAll' are contiguous [batchSize, maxKvLen, numKvHeads*headDim].
//
// One threadgroup of ATTN_SIMDGROUPS(D) simdgroups per (batch, head). Each
// simdgroup walks every ATTN_SIMDGROUPS(D)-th key with an online softmax
// (running max/sum, accumulator rescaled as the max grows), so there is no
// per-key scores array and no bound on the KV length. Lane i owns head-dim
// elements [i*D/32, (i+1)*D/32) of q and of the accumulator in registers --
// contiguous, so each K/V row read is one wide load per lane -- and the
// simdgroups' partial (max, sum, acc) results are merged through
// threadgroup memory at the end. D is a template parameter so those
// per-lane arrays have a compile-time size and stay in registers (a
// runtime-headDim version with predicated loops ran 3-5x slower), and the
// simdgroup count scales inversely with D so the merge buffer is always
// 16KB: more simdgroups -- more keys in flight -- for narrower heads.
// Instantiated for headDim 64/128/256/512; gpu_mps.sc refuses others.
#define ATTN_SIMDGROUPS(D) (4096u / (D) < 32u ? 4096u / (D) : 32u)

template <uint D>
kernel void attn_batch_bf16_kernel(device const bfloat* q          [[buffer(0)]],
                                    device const bfloat* kAll       [[buffer(1)]],
                                    device const bfloat* vAll       [[buffer(2)]],
                                    device const uint* cacheLens    [[buffer(3)]],
                                    device bfloat* out              [[buffer(4)]],
                                    constant uint& numHeads         [[buffer(5)]],
                                    constant uint& numKvHeads       [[buffer(6)]],
                                    constant uint& maxKvLen         [[buffer(7)]],
                                    constant uint& isSliding        [[buffer(8)]],
                                    constant uint& slidingWindow    [[buffer(9)]],
                                    uint2 tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint sgid [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint SG = ATTN_SIMDGROUPS(D);
    constexpr uint DPL = D / 32;
    threadgroup float sgMax[SG];
    threadgroup float sgSum[SG];
    threadgroup float sgAcc[SG][D];

    // The grid is exactly batchSize x numHeads threadgroups, so (b, h) is
    // always in range -- no early return ahead of the barrier below.
    uint b = tg.x;
    uint h = tg.y;
    uint kvHead = h / (numHeads / numKvHeads);
    uint kvLen = min(cacheLens[b], maxKvLen);
    uint jStart = (isSliding != 0u && kvLen > slidingWindow) ? kvLen - slidingWindow : 0u;

    ulong kvRowStride = (ulong)numKvHeads * D;
    ulong kvOffset = (ulong)b * maxKvLen * kvRowStride + (ulong)kvHead * D + lane * DPL;
    device const bfloat* kBase = kAll + kvOffset;
    device const bfloat* vBase = vAll + kvOffset;
    device const bfloat* qRow = q + ((ulong)b * numHeads + h) * D + lane * DPL;

    float qr[DPL];
    float acc[DPL];
    for (uint i = 0; i < DPL; i++) { qr[i] = float(qRow[i]); acc[i] = 0.0f; }

    float m = -INFINITY;
    float l = 0.0f;
    for (uint j = jStart + sgid; j < kvLen; j += SG) {
        device const bfloat* kRow = kBase + (ulong)j * kvRowStride;
        float s = 0.0f;
        for (uint i = 0; i < DPL; i++) { s += qr[i] * float(kRow[i]); }
        s = simd_sum(s);
        float mNew = max(m, s);
        float corr = exp(m - mNew);
        float p = exp(s - mNew);
        l = l * corr + p;
        m = mNew;
        device const bfloat* vRow = vBase + (ulong)j * kvRowStride;
        for (uint i = 0; i < DPL; i++) { acc[i] = acc[i] * corr + p * float(vRow[i]); }
    }

    if (lane == 0) { sgMax[sgid] = m; sgSum[sgid] = l; }
    for (uint i = 0; i < DPL; i++) { sgAcc[sgid][lane * DPL + i] = acc[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // A simdgroup that saw no keys has max -inf, sum 0, acc 0 and drops out
    // of the merge; if none saw any (empty or fully-masked window), the
    // output is zeros.
    float mAll = -INFINITY;
    for (uint s = 0; s < SG; s++) { mAll = max(mAll, sgMax[s]); }
    float w[SG];
    float lAll = 0.0f;
    for (uint s = 0; s < SG; s++) {
        w[s] = sgMax[s] == -INFINITY ? 0.0f : exp(sgMax[s] - mAll);
        lAll += w[s] * sgSum[s];
    }
    float invL = lAll > 0.0f ? 1.0f / lAll : 0.0f;
    device bfloat* outRow = out + ((ulong)b * numHeads + h) * D;
    for (uint d = tid; d < D; d += SG * 32u) {
        float o = 0.0f;
        for (uint s = 0; s < SG; s++) { o += w[s] * sgAcc[s][d]; }
        outRow[d] = bfloat(o * invL);
    }
}

#define ATTN_INSTANTIATE(D)                                                                    \
    template [[host_name("attn_batch_bf16_kernel_d" #D)]] kernel void attn_batch_bf16_kernel<D>( \
        device const bfloat*, device const bfloat*, device const bfloat*, device const uint*,  \
        device bfloat*, constant uint&, constant uint&, constant uint&, constant uint&,        \
        constant uint&, uint2, uint, uint, uint);
ATTN_INSTANTIATE(64)
ATTN_INSTANTIATE(128)
ATTN_INSTANTIATE(256)
ATTN_INSTANTIATE(512)

// ── Flash and linear attention: shared parameters ──────────────────────────
// Every flash/linear attention kernel below reads one AttnParams (matching
// gpu_mps.sc's struct MpsAttnParams field for field). Layouts are given by
// element strides, so the same kernels serve gpu_mps.h's head-major API
// ([batch][heads][seq][dim]: row stride D, head stride seq*D) and its
// token-major bf16 API ([seq][heads*dim]: row stride heads*D, head stride D).
// 'out' has q's layout. Query i of a sequence sits at position
// pos = i + (kvLen - seqLen) -- the last seqLen keys are the queries' own
// tokens, any before them a KV cache. 'causal' hides keys j > pos; a sliding
// window w > 0 (flash only) also hides keys j <= pos - w. Query head h reads
// KV head h / (heads / kvHeads).
struct AttnParams {
    uint seqLen, kvLen, heads, kvHeads;
    uint causal, window, hasState, seg;           // seg: chunked linear attention's rows per state
    float scale;                                   // flash only; <= 0 means 1/sqrt(D)
    uint qRow, qHead, qBatch;                      // q/out strides (elements)
    uint kvRow, kvHead, kvBatch;                   // k/v strides
};

// ── Flash attention, simdgroup-matrix kernel (head dim <= 256) ─────────────
// FlashAttention-2 on Apple's 8x8 simdgroup matrices. One threadgroup of
// SG simdgroups per (block of SG*8 query rows, head, batch); each
// simdgroup owns 8 query rows. The Q block is staged once and kept as
// fragments; K/V are walked in tiles of BK keys staged in threadgroup memory
// (K transposed, rows padded by 4 elements against bank conflicts) and
// shared by every simdgroup. Per tile: S = Q K^T on the matrix units, an
// online softmax in registers (lane L holds row (L/4 & 4) + (L/2 % 4),
// columns (L/4 & 2)*2 + (L%2)*2 and +1 of each 8x8 fragment, so a row's
// max/sum is two simd_shuffle_xor steps), then O += P V on the matrix
// units. Fragment data lives in plain float2 registers between multiplies
// -- writing through thread_elements() references measured several times
// slower. TS is the staging/multiply type: half for fp16, float for fp32
// and bf16 (the M1 family has no bf16 matrix units; converting bf16 tiles
// to float on the way in measured 25-40% faster than bfloat fragments).
// Tiles fully below the causal diagonal skip the mask. Measured on an M1
// Pro at 2.6-3.6 TFLOPS (see the benchmarks page). SG = 8 (64 query
// rows) at head dims 96/128 and for fp16 at 64: every K/V tile staged
// serves twice the rows of SG = 4, measured 16-24% faster at head dim 128
// (bf16, fp32) and 4-14% for fp16 at 64 -- the re-staging, not the matrix
// work, was the limit there (a kernel that stopped re-staging ran ~20%
// faster, while register prefetch of the next tile and cheaper
// threadgroup stores gained nothing). bf16/fp32 at head dims 32/64 measured
// the same or slightly slower at 8 and keep SG = 4, as does head dim 256,
// whose Q block would not fit in threadgroup memory at 8. 16-row simdgroups, 32-key tiles at head dim
// 128, fp32 fragments for fp16, and MLX's arrangement (Q re-read from
// threadgroup memory per tile, K and V sharing one buffer) all measured
// slower.
#define FLASH_PAD 4

template <typename T, typename TS, uint D, uint BK, uint SG>
kernel void flash_attn_mma_kernel(device const T* q           [[buffer(0)]],
                                   device const T* k           [[buffer(1)]],
                                   device const T* v           [[buffer(2)]],
                                   device T* out               [[buffer(3)]],
                                   constant AttnParams& p      [[buffer(4)]],
                                   uint3 tg [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint BQ = SG * 8, DB = D / 8, KB = BK / 8;
    constexpr uint DP = D + FLASH_PAD;
    constexpr uint BKP = BK + FLASH_PAD;
    constexpr uint TILE = D * BKP + BK * DP;
    // The Q block is staged unpadded when padding would overflow 32KB (D 256, float).
    constexpr uint QP = BQ * DP * sizeof(TS) > 32768 ? D : DP;
    threadgroup TS smem[BQ * QP > TILE ? BQ * QP : TILE];

    uint q0 = tg.x * BQ, h = tg.y, b = tg.z;
    uint kvh = h / (p.heads / p.kvHeads);
    const device T* qh = q + (ulong)b * p.qBatch + (ulong)h * p.qHead;
    const device T* kh = k + (ulong)b * p.kvBatch + (ulong)kvh * p.kvHead;
    const device T* vh = v + (ulong)b * p.kvBatch + (ulong)kvh * p.kvHead;
    device T* oh = out + (ulong)b * p.qBatch + (ulong)h * p.qHead;

    for (uint e = tid; e < BQ * D / 4; e += SG * 32) {
        uint r = e / (D / 4), c = (e % (D / 4)) * 4;
        vec<TS, 4> x = q0 + r < p.seqLen ? vec<TS, 4>(*(const device vec<T, 4>*)(qh + (ulong)(q0 + r) * p.qRow + c))
                                         : vec<TS, 4>(0);
        *(threadgroup vec<TS, 4>*)(smem + r * QP + c) = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<TS, 8, 8> qf[DB];
    for (uint dd = 0; dd < DB; dd++) simdgroup_load(qf[dd], smem + sg * 8 * QP + dd * 8, QP);

    const uint qid = lane / 4;
    const uint fm = (qid & 4) + ((lane / 2) % 4);
    const uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    const float sc = (p.scale > 0.0f ? p.scale : rsqrt(float(D))) * 1.4426950408889634f;   // exp2 domain
    uint posOff = p.kvLen - p.seqLen;
    uint sgRow0 = q0 + sg * 8;
    int myPos = int(sgRow0 + fm + posOff);

    float2 o[DB];
    for (uint dd = 0; dd < DB; dd++) o[dd] = float2(0.0f);
    float m = -INFINITY, l = 0.0f;

    // Key range any query of this block can see.
    uint lastPos = min(q0 + BQ, p.seqLen) - 1u + posOff;
    uint kvEnd = p.causal != 0u ? min(p.kvLen, lastPos + 1u) : p.kvLen;
    uint kvBegin = (p.window != 0u && q0 + posOff + 1u > p.window) ? q0 + posOff + 1u - p.window : 0u;
    kvBegin = (kvBegin / BK) * BK;
    threadgroup TS* ks = smem;
    threadgroup TS* vs = smem + D * BKP;
    for (uint t = kvBegin; t < kvEnd; t += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < BK * D / 4; e += SG * 32) {
            uint r = e / (D / 4), c = (e % (D / 4)) * 4;
            uint j = t + r;
            vec<TS, 4> kx = vec<TS, 4>(0), vx = vec<TS, 4>(0);
            if (j < kvEnd) {
                kx = vec<TS, 4>(*(const device vec<T, 4>*)(kh + (ulong)j * p.kvRow + c));
                vx = vec<TS, 4>(*(const device vec<T, 4>*)(vh + (ulong)j * p.kvRow + c));
            }
            ks[(c + 0) * BKP + r] = kx.x; ks[(c + 1) * BKP + r] = kx.y;
            ks[(c + 2) * BKP + r] = kx.z; ks[(c + 3) * BKP + r] = kx.w;
            *(threadgroup vec<TS, 4>*)(vs + r * DP + c) = vx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float2 s[KB];
        for (uint kb = 0; kb < KB; kb++) {
            simdgroup_float8x8 acc = simdgroup_float8x8(0);
            for (uint dd = 0; dd < DB; dd++) {
                simdgroup_matrix<TS, 8, 8> kt;
                simdgroup_load(kt, ks + dd * 8 * BKP + kb * 8, BKP);
                simdgroup_multiply_accumulate(acc, qf[dd], kt, acc);
            }
            s[kb] = float2(acc.thread_elements()[0], acc.thread_elements()[1]) * sc;
        }
        bool needMask = t + BK > kvEnd || (p.causal != 0u && t + BK - 1u > sgRow0 + posOff) ||
                        (p.window != 0u && t + p.window <= sgRow0 + 7u + posOff);
        if (needMask) {
            for (uint kb = 0; kb < KB; kb++) {
                for (uint i = 0; i < 2; i++) {
                    int j = int(t + kb * 8 + fn + i);
                    int dist = myPos - j;
                    bool hidden = j >= int(kvEnd) || (p.causal != 0u && dist < 0) ||
                                  (p.window != 0u && dist >= int(p.window));
                    if (hidden) { if (i == 0) s[kb].x = -INFINITY; else s[kb].y = -INFINITY; }
                }
            }
        }
        float rowMax = -INFINITY;
        for (uint kb = 0; kb < KB; kb++) rowMax = max(rowMax, max(s[kb].x, s[kb].y));
        rowMax = max(rowMax, simd_shuffle_xor(rowMax, 1));
        rowMax = max(rowMax, simd_shuffle_xor(rowMax, 8));
        float mNew = max(m, rowMax);
        float mUse = mNew == -INFINITY ? 0.0f : mNew;   // a fully hidden row so far
        float corr = exp2(m - mUse);
        float rowSum = 0.0f;
        for (uint kb = 0; kb < KB; kb++) { s[kb] = exp2(s[kb] - mUse); rowSum += s[kb].x + s[kb].y; }
        rowSum += simd_shuffle_xor(rowSum, 1);
        rowSum += simd_shuffle_xor(rowSum, 8);
        l = l * corr + rowSum;
        m = mNew;
        for (uint dd = 0; dd < DB; dd++) {
            simdgroup_float8x8 acc;
            acc.thread_elements()[0] = o[dd].x * corr;
            acc.thread_elements()[1] = o[dd].y * corr;
            for (uint kb = 0; kb < KB; kb++) {
                simdgroup_matrix<TS, 8, 8> pm, vf;
                pm.thread_elements()[0] = TS(s[kb].x);
                pm.thread_elements()[1] = TS(s[kb].y);
                simdgroup_load(vf, vs + kb * 8 * DP + dd * 8, DP);
                simdgroup_multiply_accumulate(acc, pm, vf, acc);
            }
            o[dd] = float2(acc.thread_elements()[0], acc.thread_elements()[1]);
        }
    }

    uint row = sgRow0 + fm;
    if (row < p.seqLen) {
        float inv = l > 0.0f ? 1.0f / l : 0.0f;   // no visible key: zeros
        device T* orow = oh + (ulong)row * p.qRow;
        for (uint dd = 0; dd < DB; dd++) {
            *(device vec<T, 2>*)(orow + dd * 8 + fn) = vec<T, 2>(T(o[dd].x * inv), T(o[dd].y * inv));
        }
    }
}

// SG must match gpu_mps.sc's __mps_flash_simdgroups; BK is the kernel's
// own choice (32 for fp16 at 64).
#define FLASH_MMA(T, TS, NAME, D, BK, SG)                                                          \
    template [[host_name("flash_attn_mma_" #NAME "_d" #D)]] kernel void flash_attn_mma_kernel<T, TS, D, BK, SG>( \
        device const T*, device const T*, device const T*, device T*, constant AttnParams&,        \
        uint3, uint, uint, uint);
#define FLASH_MMA_DTYPES(D, BK, SG)                \
    FLASH_MMA(float, float, f32, D, BK, SG)        \
    FLASH_MMA(half, half, f16, D, BK, SG)          \
    FLASH_MMA(bfloat, float, bf16, D, BK, SG)
FLASH_MMA_DTYPES(32, 16, 4)
FLASH_MMA(float, float, f32, 64, 16, 4)
FLASH_MMA(half, half, f16, 64, 32, 8)
FLASH_MMA(bfloat, float, bf16, 64, 16, 4)
FLASH_MMA_DTYPES(96, 16, 8)
FLASH_MMA_DTYPES(128, 16, 8)
FLASH_MMA_DTYPES(256, 8, 4)

// ── Flash attention, simdgroup-reduction kernel (head dim 512) ──────────────
// For head dims too wide for the matrix kernel's registers. One FLASH_BQ-
// simdgroup threadgroup per (block of FLASH_BQ query rows, head, batch), one
// query row per simdgroup with lane i owning head-dim elements
// [i*D/32, (i+1)*D/32). K/V tiles of FLASH_BK(D) rows are staged in
// threadgroup memory for all FLASH_BQ queries; each simdgroup scores the
// tile (lane jj keeps key jj's score) and folds it into a running
// max/sum/accumulator with one rescale per tile.
#define FLASH_BQ 16
#define FLASH_BK(D) (4096u / (D) < 32u ? 4096u / (D) : 32u)

template <typename T, uint D>
kernel void flash_attn_simd_kernel(device const T* q          [[buffer(0)]],
                                    device const T* k          [[buffer(1)]],
                                    device const T* v          [[buffer(2)]],
                                    device T* out              [[buffer(3)]],
                                    constant AttnParams& p     [[buffer(4)]],
                                    uint3 tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint sgid [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint DPL = D / 32;
    constexpr uint BK = FLASH_BK(D);
    threadgroup T kTile[BK * D];
    threadgroup T vTile[BK * D];

    uint h = tg.y, b = tg.z;
    uint kvHead = h / (p.heads / p.kvHeads);
    uint q0 = tg.x * FLASH_BQ;
    uint qi = q0 + sgid;
    bool active = qi < p.seqLen;
    uint posOff = p.kvLen - p.seqLen;
    int qPos = int(qi + posOff);

    const device T* qRow = q + (ulong)b * p.qBatch + (ulong)h * p.qHead + (ulong)qi * p.qRow + lane * DPL;
    const device T* kh = k + (ulong)b * p.kvBatch + (ulong)kvHead * p.kvHead;
    const device T* vh = v + (ulong)b * p.kvBatch + (ulong)kvHead * p.kvHead;
    float qr[DPL];
    float acc[DPL];
    for (uint i = 0; i < DPL; i++) {
        qr[i] = active ? float(qRow[i]) * (p.scale > 0.0f ? p.scale : rsqrt(float(D))) : 0.0f;
        acc[i] = 0.0f;
    }

    uint lastQ = min(q0 + FLASH_BQ, p.seqLen) - 1u;
    uint kvEnd = p.causal != 0u ? min(p.kvLen, lastQ + posOff + 1u) : p.kvLen;
    uint kvBegin = (p.window != 0u && q0 + posOff + 1u > p.window) ? q0 + posOff + 1u - p.window : 0u;

    float m = -INFINITY;
    float l = 0.0f;
    for (uint t = kvBegin; t < kvEnd; t += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < BK * D; e += FLASH_BQ * 32u) {
            uint j = t + e / D;
            ulong src = (ulong)j * p.kvRow + e % D;
            kTile[e] = j < kvEnd ? kh[src] : T(0.0f);
            vTile[e] = j < kvEnd ? vh[src] : T(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!active) { continue; }

        float mine = -INFINITY;
        for (uint jj = 0; jj < BK; jj++) {
            float s = 0.0f;
            for (uint i = 0; i < DPL; i++) { s += qr[i] * float(kTile[jj * D + lane * DPL + i]); }
            s = simd_sum(s);
            int dist = qPos - int(t + jj);
            bool visible = t + jj < kvEnd && !(p.causal != 0u && dist < 0) &&
                           !(p.window != 0u && dist >= int(p.window));
            if (lane == jj) { mine = visible ? s : -INFINITY; }
        }
        float tileMax = simd_max(mine);
        if (tileMax == -INFINITY) { continue; }
        float mNew = max(m, tileMax);
        float corr = exp(m - mNew);
        float pr = exp(mine - mNew);
        l = l * corr + simd_sum(pr);
        m = mNew;
        for (uint i = 0; i < DPL; i++) { acc[i] *= corr; }
        for (uint jj = 0; jj < BK; jj++) {
            float pj = simd_shuffle(pr, (ushort)jj);
            for (uint i = 0; i < DPL; i++) { acc[i] += pj * float(vTile[jj * D + lane * DPL + i]); }
        }
    }

    if (active) {
        float invL = l > 0.0f ? 1.0f / l : 0.0f;
        device T* outRow = out + (ulong)b * p.qBatch + (ulong)h * p.qHead + (ulong)qi * p.qRow + lane * DPL;
        for (uint i = 0; i < DPL; i++) { outRow[i] = T(acc[i] * invL); }
    }
}

#define FLASH_SIMD(T, NAME, D)                                                                       \
    template [[host_name("flash_attn_simd_" #NAME "_d" #D)]] kernel void flash_attn_simd_kernel<T, D>( \
        device const T*, device const T*, device const T*, device T*, constant AttnParams&,          \
        uint3, uint, uint, uint);
FLASH_SIMD(float, f32, 512)
FLASH_SIMD(half, f16, 512)
FLASH_SIMD(bfloat, bf16, 512)

// ── Linear attention ────────────────────────────────────────────────────────
// GPU counterpart of std::linear_attention_forward (attention.sc) with
// feature map phi(x) = elu(x)+1:
//   out_i = phi(q_i) . S / phi(q_i) . z,  S = sum_j phi(k_j) (x) v_j,
//                                         z = sum_j phi(k_j)
// summed over every key (non-causal, as on the CPU) or, with 'causal', over
// keys j <= pos -- the recurrent form, where S/z are a running state a
// decode step extends by one token.
//
// With 'hasState', S/z start from 'state' and the final values are written
// back, so a sequence can be fed in pieces (prefill, then one token per
// decode step) with the state staying on the GPU. Every simdgroup of a
// head reads z but none owns it, so this kernel writes back only its own S
// columns and linear_attn_z_update below, dispatched after it, advances z.
// Layout per query head h (batch folded in): S^T as [D][D] floats (row d2
// = column d2 of S) then z as [D] floats -- see gpu_mps.h's
// mps_linear_attn_state_bytes. State is per query head, not per KV head,
// so GQA heads sharing K/V never write the same state.
//
// One simdgroup per (batch, head, block of LIN_COLS(D) output columns):
// lane i keeps rows [i*D/32, (i+1)*D/32) of those columns of S, plus that
// slice of z, in registers, and the numerators/denominator fold with
// simd_sum. Tokens run sequentially inside a simdgroup; the parallelism is
// across batch*heads*D/LIN_COLS(D) simdgroups. LIN_COLS(D), the largest
// power of two up to min(16, 512/D), divides every supported D and bounds
// the per-lane state at 16 floats.
#define LIN_SG 4
#define LIN_COLS(D) ((D) <= 32u ? 16u : (D) <= 64u ? 8u : (D) <= 128u ? 4u : (D) <= 256u ? 2u : 1u)

static inline float phi_elu_p1(float x) { return x > 0.0f ? x + 1.0f : exp(x); }

template <typename T, uint D>
kernel void linear_attn_kernel(device const T* q             [[buffer(0)]],
                                device const T* k             [[buffer(1)]],
                                device const T* v             [[buffer(2)]],
                                device T* out                 [[buffer(3)]],
                                device float* state           [[buffer(4)]],
                                constant AttnParams& p        [[buffer(5)]],
                                uint3 tg [[threadgroup_position_in_grid]],
                                uint sgid [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint DPL = D / 32;
    constexpr uint COLS = LIN_COLS(D);
    constexpr uint GROUPS_PER_HEAD = D / COLS;
    uint g = tg.x * LIN_SG + sgid;
    if (g >= GROUPS_PER_HEAD) { return; }
    uint h = tg.y, b = tg.z;
    uint c0 = g * COLS;
    uint kvHead = h / (p.heads / p.kvHeads);
    const device T* qh = q + (ulong)b * p.qBatch + (ulong)h * p.qHead;
    const device T* kh = k + (ulong)b * p.kvBatch + (ulong)kvHead * p.kvHead;
    const device T* vh = v + (ulong)b * p.kvBatch + (ulong)kvHead * p.kvHead;
    device T* oh = out + (ulong)b * p.qBatch + (ulong)h * p.qHead;
    uint d1 = lane * DPL;

    device float* st = state + (ulong)(b * p.heads + h) * (D * D + D);
    float S[COLS][DPL];
    float z[DPL];
    for (uint i = 0; i < DPL; i++) {
        z[i] = p.hasState != 0u ? st[D * D + d1 + i] : 0.0f;
        for (uint c = 0; c < COLS; c++) { S[c][i] = p.hasState != 0u ? st[(c0 + c) * D + d1 + i] : 0.0f; }
    }

    uint posOff = p.kvLen - p.seqLen;
    uint nAccum = p.kvLen;
    uint absorbed = 0;
    // Causal: absorb every key at or before a query's position, then emit it.
    // Non-causal: absorb all keys, then emit every query.
    for (uint i = 0; i < p.seqLen; i++) {
        uint upTo = p.causal != 0u ? i + posOff + 1u : nAccum;
        for (; absorbed < upTo; absorbed++) {
            const device T* kRow = kh + (ulong)absorbed * p.kvRow + d1;
            const device T* vRow = vh + (ulong)absorbed * p.kvRow + c0;
            float pk[DPL];
            for (uint t = 0; t < DPL; t++) { pk[t] = phi_elu_p1(float(kRow[t])); z[t] += pk[t]; }
            for (uint c = 0; c < COLS; c++) {
                float vc = float(vRow[c]);
                for (uint t = 0; t < DPL; t++) { S[c][t] += pk[t] * vc; }
            }
        }
        const device T* qRow = qh + (ulong)i * p.qRow + d1;
        float pq[DPL];
        float den = 0.0f;
        for (uint t = 0; t < DPL; t++) { pq[t] = phi_elu_p1(float(qRow[t])); den += pq[t] * z[t]; }
        den = simd_sum(den);
        float invDen = den > 0.0f ? 1.0f / den : 0.0f;
        for (uint c = 0; c < COLS; c++) {
            float num = 0.0f;
            for (uint t = 0; t < DPL; t++) { num += pq[t] * S[c][t]; }
            num = simd_sum(num);
            if (lane == c) { oh[(ulong)i * p.qRow + c0 + c] = T(num * invDen); }
        }
    }

    if (p.hasState != 0u) {
        for (uint i = 0; i < DPL; i++) {
            for (uint c = 0; c < COLS; c++) { st[(c0 + c) * D + d1 + i] = S[c][i]; }
        }
    }
}

// z[b,h,d] += sum_j phi(k[b,j,kvHead,d]) over kvLen keys -- the z half of
// linear_attn_kernel's state update (see its comment). One thread per
// (head, d); grid.z is the batch.
template <typename T>
kernel void linear_attn_z_update(device const T* k           [[buffer(0)]],
                                  device float* state         [[buffer(1)]],
                                  constant AttnParams& p      [[buffer(2)]],
                                  constant uint& headDim      [[buffer(3)]],
                                  uint3 gid [[thread_position_in_grid]]) {
    uint id = gid.x, b = gid.z;
    if (id >= p.heads * headDim) { return; }
    uint h = id / headDim;
    uint d = id % headDim;
    uint kvHead = h / (p.heads / p.kvHeads);
    const device T* kh = k + (ulong)b * p.kvBatch + (ulong)kvHead * p.kvHead + d;
    float acc = 0.0f;
    for (uint j = 0; j < p.kvLen; j++) { acc += phi_elu_p1(float(kh[(ulong)j * p.kvRow])); }
    state[(ulong)(b * p.heads + h) * (headDim * headDim + headDim) + headDim * headDim + d] += acc;
}

// ── Linear attention, chunk-parallel kernels (head dim 64 / 128) ───────────
// The sequential kernel above walks every token in order. For a whole
// sequence (no carried state, and for causal kvLen == seqLen) these three
// kernels do the same math as matrix products on the simdgroup matrix
// units, in fp32:
//  1. linear_chunk_state: per segment of p.seg keys, S_seg = phi(K)^T V
//     ([D][D]) and z_seg = column sums of phi(K), into 'states'
//     ([batch*heads][segments][D*D + D] floats). One threadgroup per
//     (segment, head * 64-column slab, batch) of D/16 simdgroups, each a
//     32 x 32 block; phi(K) is staged transposed with an odd row stride so
//     the scattered writes land in distinct banks.
//  2. linear_chunk_prefix: per state element, an exclusive prefix over
//     segments (causal: the state before each 64-row chunk) or the total
//     (non-causal, into segment 0).
//  3. linear_chunk_out: per 64-row chunk and 64-column slab, num =
//     phi(Q) S + (causal) (phi(Q) phi(K)^T masked to j <= i) V and den =
//     phi(Q) z + (causal) the masked row sums; out = num / den. 8
//     simdgroups of 8 rows; Q/K are staged 32 columns at a time and the
//     masked scores stay in registers as fragments.
// Fragment accumulators live in float2 registers with a simdgroup_matrix
// built around each multiply, and operand fragments are reloaded rather
// than kept in arrays -- arrays of simdgroup_matrix measured 2-3x slower.
// Measured 2-4x faster than the sequential kernel and 2-8x faster than
// PyTorch's vectorized linear attention on an M1 Pro (benchmarks page).
#define LINC_ROWS 64   // linear_chunk_out rows per threadgroup (= causal segment)

template <typename T, uint D>
kernel void linear_chunk_state(device const T* k            [[buffer(0)]],
                               device const T* v            [[buffer(1)]],
                               device float* states         [[buffer(2)]],
                               constant AttnParams& p       [[buffer(3)]],
                               uint3 tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint SW = 64, NS = D / SW, NT = D * 2, RT = 32, RTP = RT + 1, SWP = SW + 4, SGR = D / 32;
    threadgroup float kt[D * RTP];      // phi(K) transposed: [d][row]
    threadgroup float vt[RT * SWP];
    uint seg = tg.x, h = tg.y / NS, slab = tg.y % NS, b = tg.z;
    uint kvHead = h / (p.heads / p.kvHeads);
    uint rb = (sg % SGR) * 32, cb = (sg / SGR) * 32;
    const device T* kh = k + (ulong)b * p.kvBatch + (ulong)kvHead * p.kvHead;
    const device T* vh = v + (ulong)b * p.kvBatch + (ulong)kvHead * p.kvHead;
    uint rBeg = seg * p.seg, rEnd = min(rBeg + p.seg, p.kvLen);
    float2 acc[4][4];
    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) acc[i][j] = float2(0.0f);
    float zs = 0.0f;
    for (uint r0 = rBeg; r0 < rEnd; r0 += RT) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < RT * D; e += NT) {
            uint r = e / D, d = e % D;
            float x = r0 + r < rEnd ? phi_elu_p1(float(kh[(ulong)(r0 + r) * p.kvRow + d])) : 0.0f;
            kt[d * RTP + r] = x;
        }
        for (uint e = tid; e < RT * SW / 4; e += NT) {
            uint r = e / (SW / 4), c = (e % (SW / 4)) * 4;
            float4 x = r0 + r < rEnd ? float4(*(const device vec<T, 4>*)(vh + (ulong)(r0 + r) * p.kvRow + slab * SW + c))
                                     : float4(0);
            *(threadgroup float4*)(vt + r * SWP + c) = x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (slab == 0 && tid < D) { for (uint r = 0; r < RT; r++) zs += kt[tid * RTP + r]; }
        for (uint kk = 0; kk < RT / 8; kk++) {
            for (uint i = 0; i < 4; i++) {
                simdgroup_float8x8 a;
                simdgroup_load(a, kt + (rb + i * 8) * RTP + kk * 8, RTP);
                for (uint j = 0; j < 4; j++) {
                    simdgroup_float8x8 bm, m;
                    simdgroup_load(bm, vt + kk * 8 * SWP + cb + j * 8, SWP);
                    m.thread_elements()[0] = acc[i][j].x; m.thread_elements()[1] = acc[i][j].y;
                    simdgroup_multiply_accumulate(m, a, bm, m);
                    acc[i][j] = float2(m.thread_elements()[0], m.thread_elements()[1]);
                }
            }
        }
    }
    uint nSeg = (p.kvLen + p.seg - 1) / p.seg;
    device float* st = states + ((ulong)(b * p.heads + h) * nSeg + seg) * (D * D + D);
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            simdgroup_float8x8 m;
            m.thread_elements()[0] = acc[i][j].x; m.thread_elements()[1] = acc[i][j].y;
            simdgroup_store(m, st + (ulong)(rb + i * 8) * D + slab * SW + cb + j * 8, D);
        }
    }
    if (slab == 0 && tid < D) st[D * D + tid] = zs;
}

template <uint D>
kernel void linear_chunk_prefix(device float* states        [[buffer(0)]],
                                constant AttnParams& p      [[buffer(1)]],
                                uint2 gid [[thread_position_in_grid]]) {
    constexpr uint E = D * D + D;
    uint e = gid.x, bh = gid.y;
    if (e >= E) return;
    uint nSeg = (p.kvLen + p.seg - 1) / p.seg;
    device float* st = states + (ulong)bh * nSeg * E + e;
    float run = 0.0f;
    for (uint c = 0; c < nSeg; c++) {
        float x = st[(ulong)c * E];
        if (p.causal != 0u) st[(ulong)c * E] = run;
        run += x;
    }
    if (p.causal == 0u) st[0] = run;
}

template <typename T, uint D, bool CAUSAL>
kernel void linear_chunk_out(device const T* q               [[buffer(0)]],
                             device const T* k               [[buffer(1)]],
                             device const T* v               [[buffer(2)]],
                             device T* out                   [[buffer(3)]],
                             device const float* states      [[buffer(4)]],
                             constant AttnParams& p          [[buffer(5)]],
                             uint3 tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint C = LINC_ROWS, NS = D / 64, DCH = 32, QP = DCH + 4, CP = C + 1, SP = 68, NT = 256;
    threadgroup float buf[C * QP + DCH * CP > C * SP ? C * QP + DCH * CP : C * SP];
    threadgroup float den[C];
    threadgroup float* qt = buf;               // phi(Q) chunk [row][DCH]
    threadgroup float* kt = buf + C * QP;      // phi(K) chunk transposed [DCH][row]
    uint c = tg.x, h = tg.y / NS, slab = tg.y % NS, b = tg.z;
    uint kvHead = h / (p.heads / p.kvHeads);
    constexpr bool causal = CAUSAL;
    const device T* qh = q + (ulong)b * p.qBatch + (ulong)h * p.qHead;
    const device T* kh = k + (ulong)b * p.kvBatch + (ulong)kvHead * p.kvHead;
    const device T* vh = v + (ulong)b * p.kvBatch + (ulong)kvHead * p.kvHead;
    device T* oh = out + (ulong)b * p.qBatch + (ulong)h * p.qHead;
    uint nSeg = (p.kvLen + p.seg - 1) / p.seg;
    const device float* st = states + ((ulong)(b * p.heads + h) * nSeg + (causal ? c : 0u)) * (D * D + D);
    const uint qid = lane / 4, fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
    uint r0 = c * C, R0 = sg * 8;
    float2 num[8], am[8];
    for (uint j = 0; j < 8; j++) { num[j] = float2(0.0f); am[j] = float2(0.0f); }
    float dsum = 0.0f;
    uint dr = tid / 4, dq = tid % 4;
    for (uint dc = 0; dc < D / DCH; dc++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < C * DCH / 4; e += NT) {
            uint r = e / (DCH / 4), cc = (e % (DCH / 4)) * 4;
            float4 x = float4(0);
            if (r0 + r < p.seqLen) {
                x = float4(*(const device vec<T, 4>*)(qh + (ulong)(r0 + r) * p.qRow + dc * DCH + cc));
                x = select(exp(x), x + 1.0f, x > 0.0f);
            }
            *(threadgroup float4*)(qt + r * QP + cc) = x;
        }
        if (causal) {
            for (uint e = tid; e < C * DCH; e += NT) {
                uint r = e / DCH, d = e % DCH;
                kt[d * CP + r] = r0 + r < p.kvLen ? phi_elu_p1(float(kh[(ulong)(r0 + r) * p.kvRow + dc * DCH + d])) : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0; t < DCH / 4; t++) {
            dsum += qt[dr * QP + dq * (DCH / 4) + t] * st[D * D + dc * DCH + dq * (DCH / 4) + t];
        }
        for (uint dd = 0; dd < DCH / 8; dd++) {
            simdgroup_float8x8 qf;
            simdgroup_load(qf, qt + R0 * QP + dd * 8, QP);
            for (uint j = 0; j < 8; j++) {
                simdgroup_float8x8 sf, m;
                simdgroup_load(sf, st + (ulong)(dc * DCH + dd * 8) * D + slab * 64 + j * 8, D);
                m.thread_elements()[0] = num[j].x; m.thread_elements()[1] = num[j].y;
                simdgroup_multiply_accumulate(m, qf, sf, m);
                num[j] = float2(m.thread_elements()[0], m.thread_elements()[1]);
                if (causal) {
                    simdgroup_float8x8 kf;
                    simdgroup_load(kf, kt + dd * 8 * CP + j * 8, CP);
                    m.thread_elements()[0] = am[j].x; m.thread_elements()[1] = am[j].y;
                    simdgroup_multiply_accumulate(m, qf, kf, m);
                    am[j] = float2(m.thread_elements()[0], m.thread_elements()[1]);
                }
            }
        }
    }
    dsum += simd_shuffle_xor(dsum, 1);
    dsum += simd_shuffle_xor(dsum, 2);
    if (dq == 0) den[dr] = dsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row = R0 + fm;
    if (causal) {
        // Mask the chunk's scores to keys j <= i, add their row sums to den,
        // stage V's slab, then num += A V.
        float rs = 0.0f;
        for (uint j = 0; j < 8; j++) {
            uint col = j * 8 + fn;
            if (col > row) am[j].x = 0.0f;
            if (col + 1 > row) am[j].y = 0.0f;
            rs += am[j].x + am[j].y;
        }
        rs += simd_shuffle_xor(rs, 1);
        rs += simd_shuffle_xor(rs, 8);
        if ((lane & 9u) == 0u) den[row] += rs;
        for (uint e = tid; e < C * 16; e += NT) {
            uint r = e / 16, cc = (e % 16) * 4;
            float4 x = r0 + r < p.kvLen ? float4(*(const device vec<T, 4>*)(vh + (ulong)(r0 + r) * p.kvRow + slab * 64 + cc))
                                        : float4(0);
            *(threadgroup float4*)(buf + r * SP + cc) = x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < C / 8; kk++) {
            simdgroup_float8x8 pm;
            pm.thread_elements()[0] = am[kk].x;
            pm.thread_elements()[1] = am[kk].y;
            for (uint j = 0; j < 8; j++) {
                simdgroup_float8x8 vf, m;
                simdgroup_load(vf, buf + kk * 8 * SP + j * 8, SP);
                m.thread_elements()[0] = num[j].x; m.thread_elements()[1] = num[j].y;
                simdgroup_multiply_accumulate(m, pm, vf, m);
                num[j] = float2(m.thread_elements()[0], m.thread_elements()[1]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r0 + row < p.seqLen) {
        float dv = den[row];
        float inv = dv > 0.0f ? 1.0f / dv : 0.0f;
        device T* orow = oh + (ulong)(r0 + row) * p.qRow + slab * 64;
        for (uint j = 0; j < 8; j++) {
            float2 x = num[j] * inv;
            *(device vec<T, 2>*)(orow + j * 8 + fn) = vec<T, 2>(T(x.x), T(x.y));
        }
    }
}

#define LINEAR_CHUNK(T, NAME, D)                                                                         \
    template [[host_name("linear_chunk_state_" #NAME "_d" #D)]] kernel void linear_chunk_state<T, D>(    \
        device const T*, device const T*, device float*, constant AttnParams&, uint3, uint, uint);       \
    template [[host_name("linear_chunk_out_" #NAME "_d" #D)]] kernel void linear_chunk_out<T, D, false>( \
        device const T*, device const T*, device const T*, device T*, device const float*,               \
        constant AttnParams&, uint3, uint, uint, uint);                                                  \
    template [[host_name("linear_chunk_out_causal_" #NAME "_d" #D)]] kernel void linear_chunk_out<T, D, true>( \
        device const T*, device const T*, device const T*, device T*, device const float*,               \
        constant AttnParams&, uint3, uint, uint, uint);
LINEAR_CHUNK(float, f32, 64) LINEAR_CHUNK(half, f16, 64) LINEAR_CHUNK(bfloat, bf16, 64)
LINEAR_CHUNK(float, f32, 128) LINEAR_CHUNK(half, f16, 128) LINEAR_CHUNK(bfloat, bf16, 128)
template [[host_name("linear_chunk_prefix_d64")]] kernel void linear_chunk_prefix<64>(device float*, constant AttnParams&, uint2);
template [[host_name("linear_chunk_prefix_d128")]] kernel void linear_chunk_prefix<128>(device float*, constant AttnParams&, uint2);

#define LINEAR(T, NAME, D)                                                                       \
    template [[host_name("linear_attn_" #NAME "_d" #D)]] kernel void linear_attn_kernel<T, D>(   \
        device const T*, device const T*, device const T*, device T*, device float*,             \
        constant AttnParams&, uint3, uint, uint);
#define LINEAR_DTYPES(D) LINEAR(float, f32, D) LINEAR(half, f16, D) LINEAR(bfloat, bf16, D)
LINEAR_DTYPES(32)
LINEAR_DTYPES(64)
LINEAR_DTYPES(96)
LINEAR_DTYPES(128)
LINEAR_DTYPES(256)
LINEAR_DTYPES(512)
#define LINEAR_Z(T, NAME)                                                                            \
    template [[host_name("linear_attn_z_update_" #NAME)]] kernel void linear_attn_z_update<T>(        \
        device const T*, device float*, constant AttnParams&, constant uint&, uint3);
LINEAR_Z(float, f32)
LINEAR_Z(half, f16)
LINEAR_Z(bfloat, bf16)
