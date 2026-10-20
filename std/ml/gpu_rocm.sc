// SafeC Standard Library — AMD GPU backend (ROCm / HIP). See gpu_rocm.h.
//
// Kernels are HIP C++ source embedded below and compiled on first use with
// hiprtc for the selected GPU's own architecture (gfx908, gfx90a, gfx942,
// gfx1100, ...), then loaded with hipModuleLoadData — so one build of this
// library runs on any ROCm GPU. Compilation takes ~0.2 s once per process.
#pragma once
#include <std/ml/gpu_rocm.h>
#include <std/math.h>
#include <std/mem.h>
#include <std/time.h>

namespace std {

// ── HIP runtime / hiprtc / rocBLAS C ABI (hip_runtime_api.h, hiprtc.h, rocblas.h) ──
extern int hipInit(unsigned int flags);
extern int hipGetDeviceCount(int* count);
extern int hipDeviceGetAttribute(int* value, int attr, int device);
extern int hipDeviceGetName(char* name, int len, int device);
extern int hipSetDevice(int device);
extern int hipStreamCreate(void** stream);
extern int hipStreamSynchronize(void* stream);
extern int hipMalloc(void** ptr, unsigned long size);
extern int hipFree(void* ptr);
extern int hipMemcpyAsync(void* dst, const void* src, unsigned long size, int kind, void* stream);
extern int hipMemsetAsync(void* dst, int value, unsigned long size, void* stream);
extern int hipModuleLoadData(void** module, const void* image);
extern int hipModuleGetFunction(void** func, void* module, const char* name);
extern int hipModuleLaunchKernel(void* func, unsigned int gx, unsigned int gy, unsigned int gz,
                                 unsigned int bx, unsigned int by, unsigned int bz,
                                 unsigned int shared, void* stream, void** params, void** extra);
extern int hipStreamBeginCapture(void* stream, int mode);
extern int hipStreamEndCapture(void* stream, void** graph);
extern int hipGraphInstantiate(void** exec, void* graph, void* errNode, char* log, unsigned long logSize);
extern int hipGraphLaunch(void* exec, void* stream);
extern int hipGraphDestroy(void* graph);
extern int hipGraphExecDestroy(void* exec);

extern int hiprtcCreateProgram(void** prog, const char* src, const char* name, int numHeaders,
                               const char** headers, const char** includeNames);
extern int hiprtcCompileProgram(void* prog, int numOptions, const char** options);
extern int hiprtcGetCodeSize(void* prog, unsigned long* size);
extern int hiprtcGetCode(void* prog, char* code);
extern int hiprtcDestroyProgram(void** prog);

extern int rocblas_create_handle(void** handle);
extern int rocblas_set_stream(void* handle, void* stream);
extern int rocblas_gemm_ex(void* handle, int transA, int transB, int m, int n, int k,
                           const void* alpha, const void* A, int aType, int lda,
                           const void* B, int bType, int ldb, const void* beta,
                           const void* C, int cType, int ldc, void* D, int dType, int ldd,
                           int computeType, int algo, int solutionIndex, unsigned int flags);
extern int rocblas_gemm_ex_get_solutions(void* handle, int transA, int transB, int m, int n, int k,
                                         const void* alpha, const void* A, int aType, int lda,
                                         const void* B, int bType, int ldb, const void* beta,
                                         const void* C, int cType, int ldc, void* D, int dType, int ldd,
                                         int computeType, int algo, unsigned int flags,
                                         int* list, int* size);
extern int rocblas_sgemm(void* handle, int transA, int transB, int m, int n, int k,
                         const float* alpha, const float* A, int lda,
                         const float* B, int ldb,
                         const float* beta, float* C, int ldc);

#define ROCM_HIP_OK              0
#define ROCM_ATTR_CU_COUNT       63   // hipDeviceAttributeMultiprocessorCount
#define ROCM_ATTR_INTEGRATED     16   // hipDeviceAttributeIntegrated
#define ROCM_H2D                 1    // hipMemcpyHostToDevice
#define ROCM_D2H                 2    // hipMemcpyDeviceToHost
#define ROCM_D2D                 3    // hipMemcpyDeviceToDevice
#define ROCM_CAPTURE_GLOBAL      0    // hipStreamCaptureModeGlobal
#define ROCM_BLAS_N              111  // rocblas_operation_none
#define ROCM_BLAS_T              112  // rocblas_operation_transpose
#define ROCM_BLAS_F16            150  // rocblas_datatype_f16_r
#define ROCM_BLAS_F32            151  // rocblas_datatype_f32_r
#define ROCM_BLAS_BF16           168  // rocblas_datatype_bf16_r
#define ROCM_BLAS_ALGO_STANDARD  0    // rocblas_gemm_algo_standard
#define ROCM_BLAS_ALGO_INDEX     1    // rocblas_gemm_algo_solution_index
#define ROCM_BLOCK               256U
#define ROCM_MAX_BLOCKS          4096UL

// ── Kernel source ────────────────────────────────────────────────────────────
// Elementwise kernels use grid-stride loops; reduce_sum and mse_grad reduce
// per wavefront with shuffles, then per block through LDS, then one atomic
// per block. sgemm_nn is a 64x64-tile LDS kernel with 4x4 register tiles.
static const char* rocm_kernel_source_() {
    return
    "\n"
    "#define EW_UNARY(name, expr)                                                   \\\n"
    "extern \"C\" __global__ void name(const float* __restrict__ a,                   \\\n"
    "                                float* __restrict__ o, unsigned n, float k) {  \\\n"
    "    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;                        \\\n"
    "    unsigned stride = gridDim.x * blockDim.x;                                  \\\n"
    "    for (; i < n; i += stride) { float x = a[i]; o[i] = (expr); }              \\\n"
    "}\n"
    "#define EW_BINARY(name, expr)                                                  \\\n"
    "extern \"C\" __global__ void name(const float* __restrict__ a,                   \\\n"
    "                                const float* __restrict__ b,                   \\\n"
    "                                float* __restrict__ o, unsigned n) {           \\\n"
    "    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;                        \\\n"
    "    unsigned stride = gridDim.x * blockDim.x;                                  \\\n"
    "    for (; i < n; i += stride) { float x = a[i], y = b[i]; o[i] = (expr); }    \\\n"
    "}\n"
    "EW_BINARY(ew_add, x + y)\n"
    "EW_BINARY(ew_sub, x - y)\n"
    "EW_BINARY(ew_mul, x * y)\n"
    "EW_BINARY(ew_div, x / y)\n"
    "EW_BINARY(ew_pow, __powf(x, y))\n"
    "EW_UNARY(ew_relu, x > 0.0f ? x : 0.0f)\n"
    "EW_UNARY(ew_log, __logf(x))\n"
    "EW_UNARY(ew_exp, __expf(x))\n"
    "EW_UNARY(ew_sqrt, __fsqrt_rn(x))\n"
    "EW_UNARY(ew_scale, x * k)\n"
    "\n"
    "// relu backward: g_in = g_out where x > 0\n"
    "extern \"C\" __global__ void ew_relu_grad(const float* __restrict__ x, const float* __restrict__ g,\n"
    "                                        float* __restrict__ o, unsigned n) {\n"
    "    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    unsigned stride = gridDim.x * blockDim.x;\n"
    "    for (; i < n; i += stride) o[i] = x[i] > 0.0f ? g[i] : 0.0f;\n"
    "}\n"
    "\n"
    "// w -= lr * g, in place\n"
    "extern \"C\" __global__ void sgd_update(float* __restrict__ w, const float* __restrict__ g,\n"
    "                                      float lr, unsigned n) {\n"
    "    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    unsigned stride = gridDim.x * blockDim.x;\n"
    "    for (; i < n; i += stride) w[i] -= lr * g[i];\n"
    "}\n"
    "\n"
    "// Sum: each block reduces a grid-stride slice with wave shuffles, then one\n"
    "// atomicAdd per block. *out must be zeroed first.\n"
    "extern \"C\" __global__ void reduce_sum(const float* __restrict__ a, float* __restrict__ out, unsigned n) {\n"
    "    float s = 0.0f;\n"
    "    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    unsigned stride = gridDim.x * blockDim.x;\n"
    "    for (; i < n; i += stride) s += a[i];\n"
    "    for (int off = warpSize / 2; off > 0; off >>= 1) s += __shfl_down(s, off);\n"
    "    __shared__ float part[1024 / 32];\n"
    "    int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;\n"
    "    if (lane == 0) part[wid] = s;\n"
    "    __syncthreads();\n"
    "    if (wid == 0) {\n"
    "        int nw = (blockDim.x + warpSize - 1) / warpSize;\n"
    "        s = lane < nw ? part[lane] : 0.0f;\n"
    "        for (int off = warpSize / 2; off > 0; off >>= 1) s += __shfl_down(s, off);\n"
    "        if (lane == 0) atomicAdd(out, s);\n"
    "    }\n"
    "}\n"
    "\n"
    "// ── SGEMM: C[M,N] = op(A) * op(B) (+ optional ReLU epilogue), row-major ─────\n"
    "// 64x64 output tile per 256-thread block, BK=16, each thread a 4x4 micro-tile.\n"
    "// TA/TB: operand stored transposed (A is KxM / B is NxK).\n"
    "template <int TA, int TB, int RELU>\n"
    "__device__ void sgemm_tile(const float* __restrict__ A, const float* __restrict__ B,\n"
    "                           float* __restrict__ C, int M, int N, int K) {\n"
    "    constexpr int BM = 64, BN = 64, BK = 16;\n"
    "    __shared__ float As[BK][BM + 4];\n"
    "    __shared__ float Bs[BK][BN + 4];\n"
    "    int tid = threadIdx.x;\n"
    "    int tx = tid % 16, ty = tid / 16;          // 16x16 threads, 4x4 each\n"
    "    int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;\n"
    "    float acc[4][4] = {};\n"
    "    for (int k0 = 0; k0 < K; k0 += BK) {\n"
    "        // load A tile (BM x BK) -> As[k][m]\n"
    "        for (int l = tid; l < BM * BK; l += 256) {\n"
    "            int m = l / BK, k = l % BK;\n"
    "            int gm = row0 + m, gk = k0 + k;\n"
    "            float v = 0.0f;\n"
    "            if (gm < M && gk < K) v = TA ? A[(size_t)gk * M + gm] : A[(size_t)gm * K + gk];\n"
    "            As[k][m] = v;\n"
    "        }\n"
    "        // load B tile (BK x BN) -> Bs[k][n]\n"
    "        for (int l = tid; l < BK * BN; l += 256) {\n"
    "            int k = l / BN, n = l % BN;\n"
    "            int gk = k0 + k, gn = col0 + n;\n"
    "            float v = 0.0f;\n"
    "            if (gk < K && gn < N) v = TB ? B[(size_t)gn * K + gk] : B[(size_t)gk * N + gn];\n"
    "            Bs[k][n] = v;\n"
    "        }\n"
    "        __syncthreads();\n"
    "#pragma unroll\n"
    "        for (int k = 0; k < BK; ++k) {\n"
    "            float a[4], b[4];\n"
    "#pragma unroll\n"
    "            for (int i = 0; i < 4; ++i) a[i] = As[k][ty * 4 + i];\n"
    "#pragma unroll\n"
    "            for (int j = 0; j < 4; ++j) b[j] = Bs[k][tx * 4 + j];\n"
    "#pragma unroll\n"
    "            for (int i = 0; i < 4; ++i)\n"
    "#pragma unroll\n"
    "                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];\n"
    "        }\n"
    "        __syncthreads();\n"
    "    }\n"
    "#pragma unroll\n"
    "    for (int i = 0; i < 4; ++i) {\n"
    "        int gm = row0 + ty * 4 + i;\n"
    "        if (gm >= M) continue;\n"
    "#pragma unroll\n"
    "        for (int j = 0; j < 4; ++j) {\n"
    "            int gn = col0 + tx * 4 + j;\n"
    "            if (gn < N) { float v = acc[i][j]; C[(size_t)gm * N + gn] = RELU ? (v > 0.0f ? v : 0.0f) : v; }\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "extern \"C\" __global__ __launch_bounds__(256) void sgemm_nn(const float* A, const float* B, float* C, int M, int N, int K) { sgemm_tile<0,0,0>(A,B,C,M,N,K); }\n"
    "\n"
    "// dY = 2 (y - t), loss += sum (y - t)^2   (fused: one pass instead of sub/scale/square/sum)\n"
    "extern \"C\" __global__ void mse_grad(const float* y, const float* t, float* dy, float* loss, unsigned n) {\n"
    "    float s = 0.0f;\n"
    "    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    for (; i < n; i += gridDim.x * blockDim.x) { float d = y[i] - t[i]; dy[i] = 2.0f * d; s += d * d; }\n"
    "    for (int off = warpSize / 2; off > 0; off >>= 1) s += __shfl_down(s, off);\n"
    "    __shared__ float part[16];\n"
    "    int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;\n"
    "    if (lane == 0) part[wid] = s;\n"
    "    __syncthreads();\n"
    "    if (wid == 0) {\n"
    "        s = lane < (int)((blockDim.x + warpSize - 1) / warpSize) ? part[lane] : 0.0f;\n"
    "        for (int off = warpSize / 2; off > 0; off >>= 1) s += __shfl_down(s, off);\n"
    "        if (lane == 0) atomicAdd(loss, s);\n"
    "    }\n"
    "}\n"
    // planes x h x w -> planes x 2h x 2w, out[p][y][x] = in[p][y/2][x/2]
    "extern \"C\" __global__ void upsample2x(const float* __restrict__ x, float* __restrict__ y,\n"
    "                                          unsigned planes, unsigned h, unsigned w) {\n"
    "    unsigned w2 = 2u * w, h2 = 2u * h, total = planes * h2 * w2;\n"
    "    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {\n"
    "        unsigned ox = i % w2, t = i / w2, oy = t % h2, p = t / h2;\n"
    "        y[i] = x[((size_t)p * h + oy / 2u) * w + ox / 2u];\n"
    "    }\n"
    "}\n"
    // NCHW y[b][c][:] += bias[c], then relu if asked: the conv epilogue.
    "extern \"C\" __global__ void bias_act(float* __restrict__ y, const float* __restrict__ b,\n"
    "                                        unsigned total, unsigned plane, unsigned channels, int relu) {\n"
    "    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {\n"
    "        float v = y[i] + b[(i / plane) % channels];\n"
    "        y[i] = (relu && v < 0.0f) ? 0.0f : v;\n"
    "    }\n"
    "}\n"
    // out[b] = (a[b], c[b]) for 'batch' rows of na and nc elements: channel
    // concat of NCHW tensors in one launch.
    "extern \"C\" __global__ void concat_rows(const float* __restrict__ a, const float* __restrict__ c,\n"
    "                                           float* __restrict__ o, unsigned batch, unsigned na, unsigned nc) {\n"
    "    unsigned row = na + nc, total = batch * row;\n"
    "    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {\n"
    "        unsigned b = i / row, r = i % row;\n"
    "        o[i] = r < na ? a[(size_t)b * na + r] : c[(size_t)b * nc + (r - na)];\n"
    "    }\n"
    "}\n"
    // out[b] = (upsample2x(x[b]), skip[b]) along channels: the U-Net decoder's
    // upsample and skip concat in one pass, without the upsampled tensor.
    // x: [batch, cx, h, w]; skip: [batch, cs, 2h, 2w]; out: [batch, cx+cs, 2h, 2w].
    "extern \"C\" __global__ void upcat(const float* __restrict__ x, const float* __restrict__ s,\n"
    "                                     float* __restrict__ o, unsigned batch, unsigned cx, unsigned cs,\n"
    "                                     unsigned h, unsigned w) {\n"
    "    unsigned w2 = 2u * w, h2 = 2u * h, plane = h2 * w2, row = (cx + cs) * plane, total = batch * row;\n"
    "    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {\n"
    "        unsigned b = i / row, r = i % row, c = r / plane, p = r % plane;\n"
    "        if (c < cx) {\n"
    "            unsigned oy = p / w2, ox = p % w2;\n"
    "            o[i] = x[(((size_t)b * cx + c) * h + oy / 2u) * w + ox / 2u];\n"
    "        } else {\n"
    "            o[i] = s[((size_t)b * cs + (c - cx)) * plane + p];\n"
    "        }\n"
    "    }\n"
    "}\n"
    // fp16 / bf16 storage. Both conversions round to nearest even and give
    // the same bits as float16.h: _Float16 for fp16, the integer
    // round-half-even trick for bf16 (NaN keeps its top bits plus 0x40).
    "__device__ unsigned short f2h(float x, int bf) {\n"
    "    if (!bf) return __builtin_bit_cast(unsigned short, (_Float16)x);\n"
    "    unsigned u = __builtin_bit_cast(unsigned, x);\n"
    "    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x40u);\n"
    "    return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);\n"
    "}\n"
    "__device__ float h2f(unsigned short h, int bf) {\n"
    "    return bf ? __builtin_bit_cast(float, (unsigned)h << 16) : (float)__builtin_bit_cast(_Float16, h);\n"
    "}\n"
    "extern \"C\" __global__ void to_half(const float* __restrict__ a, unsigned short* __restrict__ o,\n"
    "                                       unsigned n, int bf) {\n"
    "    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)\n"
    "        o[i] = f2h(a[i], bf);\n"
    "}\n"
    "extern \"C\" __global__ void from_half(const unsigned short* __restrict__ a, float* __restrict__ o,\n"
    "                                         unsigned n, int bf) {\n"
    "    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)\n"
    "        o[i] = h2f(a[i], bf);\n"
    "}\n"
    // w = round(w - lr * g), computed in fp32. w == g is allowed.
    "extern \"C\" __global__ void half_sgd(unsigned short* w, const unsigned short* g, float lr,\n"
    "                                        unsigned n, int bf) {\n"
    // no FMA contraction: same rounding as the CPU's separate * and -
    "#pragma clang fp contract(off)\n"
    "    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)\n"
    "        w[i] = f2h(h2f(w[i], bf) - lr * h2f(g[i], bf), bf);\n"
    "}\n"
    // Training ops on 16-bit buffers: fp32 arithmetic, one rounding per
    // stored value (relu and relu_grad only copy bits).
    "extern \"C\" __global__ void half_relu(const unsigned short* a, unsigned short* o, unsigned n, int bf) {\n"
    "    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)\n"
    "        o[i] = h2f(a[i], bf) > 0.0f ? a[i] : (unsigned short)0;\n"
    "}\n"
    "extern \"C\" __global__ void half_relu_grad(const unsigned short* z, const unsigned short* g,\n"
    "                                              unsigned short* o, unsigned n, int bf) {\n"
    "    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)\n"
    "        o[i] = h2f(z[i], bf) > 0.0f ? g[i] : (unsigned short)0;\n"
    "}\n"
    // dy = round(2 (y - t)), loss += sum (y - t)^2 in fp32
    "extern \"C\" __global__ void half_mse(const unsigned short* y, const unsigned short* t, unsigned short* dy,\n"
    "                                        float* loss, unsigned n, int bf) {\n"
    "    float s = 0.0f;\n"
    "    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {\n"
    "        float d = h2f(y[i], bf) - h2f(t[i], bf); dy[i] = f2h(2.0f * d, bf); s += d * d;\n"
    "    }\n"
    "    for (int off = warpSize / 2; off > 0; off >>= 1) s += __shfl_down(s, off);\n"
    "    __shared__ float part[16];\n"
    "    int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;\n"
    "    if (lane == 0) part[wid] = s;\n"
    "    __syncthreads();\n"
    "    if (wid == 0) {\n"
    "        s = lane < (int)((blockDim.x + warpSize - 1) / warpSize) ? part[lane] : 0.0f;\n"
    "        for (int off = warpSize / 2; off > 0; off >>= 1) s += __shfl_down(s, off);\n"
    "        if (lane == 0) atomicAdd(loss, s);\n"
    "    }\n"
    "}\n"
    // Two SGD updates in one launch: indices [0, n1) update w1, the rest w2.
    "extern \"C\" __global__ void half_sgd2(unsigned short* w1, const unsigned short* g1, unsigned n1,\n"
    "                                         unsigned short* w2, const unsigned short* g2, unsigned n2,\n"
    "                                         float lr, int bf) {\n"
    "#pragma clang fp contract(off)\n"
    "    unsigned n = n1 + n2;\n"
    "    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {\n"
    "        if (i < n1) w1[i] = f2h(h2f(w1[i], bf) - lr * h2f(g1[i], bf), bf);\n"
    "        else w2[i - n1] = f2h(h2f(w2[i - n1], bf) - lr * h2f(g2[i - n1], bf), bf);\n"
    "    }\n"
    "}\n"
    "// Flash attention (forward): one block = 64 query rows of one head, 4 lanes\n"
    "// per row, each lane holding a quarter of the row's q and output (dims\n"
    "// part, part + 4, ...). K/V tiles of 32 rows are staged in LDS; per tile the\n"
    "// lanes compute the scores, then rescale the running max / sum / output\n"
    "// once (Dao et al.'s online softmax). The [seq, kv] score matrix is never\n"
    "// formed. Q/K/V/O are [heads][rows][d] row-major; q is pre-scaled.\n"
    "#define FA_BR 64\n"
    "#define FA_BC 32\n"
    "#define FA_TPR 4\n"
    "#define FA_DMAX 128\n"
    "template <int DM>\n"
    "__device__ __forceinline__ void flash_impl(const float* __restrict__ Q, const float* __restrict__ K,\n"
    "                                           const float* __restrict__ V, float* __restrict__ O,\n"
    "                                           unsigned seq, unsigned kv, unsigned d, float scale, int causal) {\n"
    "    constexpr int NT = DM / FA_TPR;                // dims per lane\n"
    "    __shared__ float Ks[FA_BC][DM + 1];\n"
    "    __shared__ float Vs[FA_BC][DM + 1];\n"
    "    const unsigned h = blockIdx.y;\n"
    "    const unsigned row = blockIdx.x * FA_BR + threadIdx.x / FA_TPR;\n"
    "    const unsigned part = threadIdx.x % FA_TPR;\n"
    "    const bool active = row < seq;\n"
    "    const float* Kh = K + (unsigned long long)h * kv * d;\n"
    "    const float* Vh = V + (unsigned long long)h * kv * d;\n"
    "    float qr[NT], acc[NT];\n"
    "#pragma unroll\n"
    "    for (int t = 0; t < NT; ++t) {\n"
    "        unsigned c = part + FA_TPR * t;\n"
    "        qr[t] = (active && c < d) ? Q[((unsigned long long)h * seq + row) * d + c] * scale : 0.0f;\n"
    "        acc[t] = 0.0f;\n"
    "    }\n"
    "    float m = -__builtin_inff(), l = 0.0f;\n"
    "    const int off = (int)kv - (int)seq;            // causal: key j visible to row i if j <= i + off\n"
    "    unsigned kvEnd = kv;\n"
    "    if (causal) {\n"
    "        int lim = (int)(blockIdx.x * FA_BR + FA_BR - 1) + off + 1;\n"
    "        kvEnd = lim <= 0 ? 0u : (lim < (int)kv ? (unsigned)lim : kv);\n"
    "    }\n"
    "    for (unsigned j0 = 0; j0 < kvEnd; j0 += FA_BC) {\n"
    "        for (unsigned e = threadIdx.x; e < FA_BC * d; e += blockDim.x) {\n"
    "            unsigned r = e / d, c = e % d, j = j0 + r;\n"
    "            Ks[r][c] = j < kv ? Kh[(unsigned long long)j * d + c] : 0.0f;\n"
    "            Vs[r][c] = j < kv ? Vh[(unsigned long long)j * d + c] : 0.0f;\n"
    "        }\n"
    "        __syncthreads();\n"
    "        const unsigned jn = kv - j0 < FA_BC ? kv - j0 : FA_BC;\n"
    "        float s[FA_BC];\n"
    "        float mt = -__builtin_inff();\n"
    "#pragma unroll\n"
    "        for (int jj = 0; jj < FA_BC; ++jj) {\n"
    "            float x = 0.0f;\n"
    "#pragma unroll\n"
    "            for (int t = 0; t < NT; ++t) x = fmaf(qr[t], Ks[jj][part + FA_TPR * t], x);\n"
    "            x += __shfl_xor(x, 1);\n"
    "            x += __shfl_xor(x, 2);\n"
    "            bool vis = (unsigned)jj < jn && (!causal || (int)(j0 + jj) <= (int)row + off);\n"
    "            s[jj] = vis ? x : -__builtin_inff();\n"
    "            mt = fmaxf(mt, s[jj]);\n"
    "        }\n"
    "        if (mt > -__builtin_inff()) {\n"
    "            float mn = fmaxf(m, mt);\n"
    "            float corr = __expf(m - mn);\n"
    "            l *= corr;\n"
    "#pragma unroll\n"
    "            for (int t = 0; t < NT; ++t) acc[t] *= corr;\n"
    "#pragma unroll\n"
    "            for (int jj = 0; jj < FA_BC; ++jj) {\n"
    "                float p = __expf(s[jj] - mn);\n"
    "                l += p;\n"
    "#pragma unroll\n"
    "                for (int t = 0; t < NT; ++t) acc[t] = fmaf(p, Vs[jj][part + FA_TPR * t], acc[t]);\n"
    "            }\n"
    "            m = mn;\n"
    "        }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    if (active) {\n"
    "        float inv = l > 0.0f ? 1.0f / l : 0.0f;\n"
    "#pragma unroll\n"
    "        for (int t = 0; t < NT; ++t) {\n"
    "            unsigned c = part + FA_TPR * t;\n"
    "            if (c < d) O[((unsigned long long)h * seq + row) * d + c] = acc[t] * inv;\n"
    "        }\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ __launch_bounds__(256) void flash_attn_d64(const float* Q, const float* K, const float* V, float* O,\n"
    "        unsigned seq, unsigned kv, unsigned d, float scale, int causal) { flash_impl<64>(Q, K, V, O, seq, kv, d, scale, causal); }\n"
    "extern \"C\" __global__ __launch_bounds__(256) void flash_attn_d128(const float* Q, const float* K, const float* V, float* O,\n"
    "        unsigned seq, unsigned kv, unsigned d, float scale, int causal) { flash_impl<128>(Q, K, V, O, seq, kv, d, scale, causal); }\n"
    "\n"
    "// Linear attention, chunked (32 rows): phi(x) = elu(x) + 1.\n"
    "//   state: S_c = sum_{j in chunk c} phi(k_j)^T [v_j | 1]      ([d][d+1] per chunk)\n"
    "//   prefix: causal -> exclusive prefix over chunks; else the total, in chunk 0\n"
    "//   out: row i of chunk c: (phi(q_i) P + sum_{j<=i in c} (phi(q_i).phi(k_j)) [v_j|1]) split as\n"
    "//        numerator columns 0..d-1 over the denominator column d\n"
    "#define LA_C 32\n"
    "__device__ __forceinline__ float la_phi(float x) { return x > 0.0f ? x + 1.0f : __expf(x); }\n"
    "extern \"C\" __global__ __launch_bounds__(256) void lin_attn_state(const float* __restrict__ K, const float* __restrict__ V,\n"
    "        float* __restrict__ S, unsigned kv, unsigned d, unsigned nchunks) {\n"
    "    __shared__ float pk[LA_C][FA_DMAX];\n"
    "    __shared__ float vv[LA_C][FA_DMAX + 1];\n"
    "    const unsigned c = blockIdx.x, h = blockIdx.y, w = d + 1;\n"
    "    const unsigned j0 = c * LA_C;\n"
    "    for (unsigned e = threadIdx.x; e < LA_C * w; e += blockDim.x) {\n"
    "        unsigned r = e / w, col = e % w, j = j0 + r;\n"
    "        bool in = j < kv;\n"
    "        if (col < d) {\n"
    "            pk[r][col] = in ? la_phi(K[((unsigned long long)h * kv + j) * d + col]) : 0.0f;\n"
    "            vv[r][col] = in ? V[((unsigned long long)h * kv + j) * d + col] : 0.0f;\n"
    "        } else vv[r][d] = in ? 1.0f : 0.0f;\n"
    "    }\n"
    "    __syncthreads();\n"
    "    float* Sc = S + ((unsigned long long)h * nchunks + c) * d * w;\n"
    "    for (unsigned e = threadIdx.x; e < d * w; e += blockDim.x) {\n"
    "        unsigned r = e / w, col = e % w;\n"
    "        float a = 0.0f;\n"
    "#pragma unroll 8\n"
    "        for (int jj = 0; jj < LA_C; ++jj) a = fmaf(pk[jj][r], vv[jj][col], a);\n"
    "        Sc[e] = a;\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ __launch_bounds__(256) void lin_attn_prefix(float* __restrict__ S, unsigned d, unsigned nchunks, int causal) {\n"
    "    const unsigned w = d + 1, n = d * w, h = blockIdx.y;\n"
    "    const unsigned e = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (e >= n) return;\n"
    "    float* base = S + (unsigned long long)h * nchunks * n + e;\n"
    "    float run = 0.0f;\n"
    "    for (unsigned c = 0; c < nchunks; ++c) {\n"
    "        float t = base[(unsigned long long)c * n];\n"
    "        if (causal) base[(unsigned long long)c * n] = run;\n"
    "        run += t;\n"
    "    }\n"
    "    if (!causal) base[0] = run;\n"
    "}\n"
    "extern \"C\" __global__ __launch_bounds__(256) void lin_attn_out(const float* __restrict__ Q, const float* __restrict__ K,\n"
    "        const float* __restrict__ V, const float* __restrict__ S, float* __restrict__ O,\n"
    "        unsigned seq, unsigned kv, unsigned d, unsigned nchunks, int causal) {\n"
    "    __shared__ float pq[LA_C][FA_DMAX];\n"
    "    __shared__ float pk[LA_C][FA_DMAX + 1];        // later reused for the [LA_C][d+1] results\n"
    "    __shared__ float vv[LA_C][FA_DMAX + 1];\n"
    "    __shared__ float A[LA_C][LA_C + 1];\n"
    "    const unsigned c = blockIdx.x, h = blockIdx.y, w = d + 1;\n"
    "    const unsigned i0 = c * LA_C;\n"
    "    const float* P = S + ((unsigned long long)h * nchunks + (causal ? c : 0u)) * d * w;\n"
    "    for (unsigned e = threadIdx.x; e < LA_C * w; e += blockDim.x) {\n"
    "        unsigned r = e / w, col = e % w, i = i0 + r;\n"
    "        if (col < d) {\n"
    "            pq[r][col] = i < seq ? la_phi(Q[((unsigned long long)h * seq + i) * d + col]) : 0.0f;\n"
    "            if (causal) {\n"
    "                pk[r][col] = i < kv ? la_phi(K[((unsigned long long)h * kv + i) * d + col]) : 0.0f;\n"
    "                vv[r][col] = i < kv ? V[((unsigned long long)h * kv + i) * d + col] : 0.0f;\n"
    "            }\n"
    "        } else if (causal) vv[r][d] = i < kv ? 1.0f : 0.0f;\n"
    "    }\n"
    "    __syncthreads();\n"
    "    if (causal) {\n"
    "        for (unsigned e = threadIdx.x; e < LA_C * LA_C; e += blockDim.x) {\n"
    "            unsigned i = e / LA_C, j = e % LA_C;\n"
    "            float a = 0.0f;\n"
    "            if (j <= i) for (unsigned r = 0; r < d; ++r) a = fmaf(pq[i][r], pk[j][r], a);\n"
    "            A[i][j] = a;\n"
    "        }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    float res[17];\n"
    "    int nres = 0;\n"
    "    for (unsigned e = threadIdx.x; e < LA_C * w; e += blockDim.x) {\n"
    "        unsigned i = e / w, col = e % w;\n"
    "        float a = 0.0f;\n"
    "        for (unsigned r = 0; r < d; ++r) a = fmaf(pq[i][r], P[(unsigned long long)r * w + col], a);\n"
    "        if (causal)\n"
    "            for (unsigned j = 0; j <= i; ++j) a = fmaf(A[i][j], vv[j][col], a);\n"
    "        res[nres++] = a;\n"
    "    }\n"
    "    __syncthreads();                                // pk no longer needed: results go there\n"
    "    nres = 0;\n"
    "    for (unsigned e = threadIdx.x; e < LA_C * w; e += blockDim.x) pk[e / w][e % w] = res[nres++];\n"
    "    __syncthreads();\n"
    "    for (unsigned e = threadIdx.x; e < LA_C * d; e += blockDim.x) {\n"
    "        unsigned i = e / d, col = e % d;\n"
    "        if (i0 + i < seq) {\n"
    "            float den = pk[i][d];\n"
    "            O[((unsigned long long)h * seq + i0 + i) * d + col] = den != 0.0f ? pk[i][col] / den : 0.0f;\n"
    "        }\n"
    "    }\n"
    "}\n"
    "// ── Attention on matrix cores (CDNA: gfx908 / gfx90a / gfx94x / gfx950) ──────\n"
    "// fp32 MFMA versions of the flash and linear attention kernels above, used\n"
    "// when the GPU has matrix cores (mfma_probe). On other GPUs the kernels\n"
    "// compile to empty bodies and the CUDA-core kernels above run instead.\n"
    "#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)\n"
    "#define SAFEC_MFMA 1\n"
    "#else\n"
    "#define SAFEC_MFMA 0\n"
    "#endif\n"
    "extern \"C\" __global__ void mfma_probe(float* o) { if (threadIdx.x == 0 && blockIdx.x == 0) o[0] = (float)SAFEC_MFMA; }\n"
    "#if SAFEC_MFMA\n"
    "typedef float fm_v16 __attribute__((ext_vector_type(16)));\n"
    "typedef float fm_v4 __attribute__((ext_vector_type(4)));\n"
    "// Flash attention. Grid (heads, query blocks); query blocks run last-first so\n"
    "// the longest causal blocks start first. Scores are computed transposed\n"
    "// (S^T = K Q^T), so each lane owns one query row: its q (pre-scaled by\n"
    "// scale * log2 e) stays in registers as the MFMA B operand, the row max and\n"
    "// sum need one cross-half shuffle, and P feeds the P V product straight from\n"
    "// registers. The output is rescaled only when some row's max grew by\n"
    "// more than 2^8.\n"
    "//  - head dim <= 64: 32x32x2 MFMA, 4 waves x 32 rows, K/V tiles of 64 keys\n"
    "//  - head dim <= 128: 16x16x4 MFMA, 4 waves x 16 rows (half the registers)\n"
    "#define FM_BC 32\n"
    "#define FM_T 64\n"
    "#define FM_BR 128\n"
    "#define FM16_BR 64\n"
    "template <int DM>\n"
    "__device__ __forceinline__ void flash_m32_impl(const float* __restrict__ Q, const float* __restrict__ K,\n"
    "                                                const float* __restrict__ V, float* __restrict__ O,\n"
    "                                                unsigned seq, unsigned kv, unsigned d, float scale, int causal) {\n"
    "    constexpr int HD = DM / 2;\n"
    "    constexpr int NB = DM / 32;\n"
    "    constexpr int D4 = DM / 4;\n"
    "    __shared__ float4 Ks[FM_T * D4];\n"
    "    __shared__ float Vs[FM_T * DM];\n"
    "    const unsigned h = blockIdx.x;\n"
    "    const unsigned qb = gridDim.y - 1u - blockIdx.y;\n"
    "    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;\n"
    "    const unsigned col = lane & 31u, half = lane >> 5;\n"
    "    const unsigned q0 = qb * FM_BR + wave * 32u;\n"
    "    const unsigned qrow = q0 + col;\n"
    "    const float* Kh = K + (unsigned long long)h * kv * d;\n"
    "    const float* Vh = V + (unsigned long long)h * kv * d;\n"
    "    const float qs = scale * 1.4426950408889634f;\n"
    "    float qr[HD];\n"
    "#pragma unroll\n"
    "    for (int s = 0; s < HD; ++s) {\n"
    "        unsigned c = half * HD + s;\n"
    "        qr[s] = (qrow < seq && c < d) ? Q[((unsigned long long)h * seq + qrow) * d + c] * qs : 0.0f;\n"
    "    }\n"
    "    fm_v16 acc[NB];\n"
    "#pragma unroll\n"
    "    for (int b = 0; b < NB; ++b)\n"
    "#pragma unroll\n"
    "        for (int r = 0; r < 16; ++r) acc[b][r] = 0.0f;\n"
    "    float m = -__builtin_inff(), l = 0.0f;\n"
    "    const int off = (int)kv - (int)seq;\n"
    "    unsigned kvEnd = kv, wEnd = kv;\n"
    "    if (causal) {\n"
    "        int lim = (int)(qb * FM_BR + FM_BR - 1) + off + 1;\n"
    "        kvEnd = lim <= 0 ? 0u : (lim < (int)kv ? (unsigned)lim : kv);\n"
    "        int wl = (int)(q0 + 31u) + off + 1;\n"
    "        wEnd = wl <= 0 ? 0u : (wl < (int)kv ? (unsigned)wl : kv);\n"
    "    }\n"
    "    const bool vec = (d & 3u) == 0u;\n"
    "    for (unsigned jt = 0; jt < kvEnd; jt += FM_T) {\n"
    "        __syncthreads();\n"
    "        for (unsigned e = threadIdx.x; e < FM_T * D4; e += blockDim.x) {\n"
    "            unsigned r = e / D4, c4 = e % D4, j = jt + r, c = c4 * 4u;\n"
    "            float4 kq = make_float4(0.f, 0.f, 0.f, 0.f), vq = kq;\n"
    "            if (j < kv && c < d) {\n"
    "                const float* kp = Kh + (unsigned long long)j * d + c;\n"
    "                const float* vp = Vh + (unsigned long long)j * d + c;\n"
    "                if (vec) { kq = *(const float4*)kp; vq = *(const float4*)vp; }\n"
    "                else {\n"
    "                    kq.x = kp[0]; vq.x = vp[0];\n"
    "                    if (c + 1 < d) { kq.y = kp[1]; vq.y = vp[1]; }\n"
    "                    if (c + 2 < d) { kq.z = kp[2]; vq.z = vp[2]; }\n"
    "                    if (c + 3 < d) { kq.w = kp[3]; vq.w = vp[3]; }\n"
    "                }\n"
    "            }\n"
    "            Ks[r * D4 + (c4 ^ (r & 7u))] = kq;\n"
    "            *(float4*)&Vs[r * DM + c] = vq;\n"
    "        }\n"
    "        __syncthreads();\n"
    "        for (unsigned sub = 0; sub < FM_T / FM_BC; ++sub) {\n"
    "        const unsigned j0 = jt + sub * FM_BC;\n"
    "        if (j0 >= wEnd || j0 >= kvEnd) break;\n"
    "        fm_v16 sc;\n"
    "#pragma unroll\n"
    "        for (int r = 0; r < 16; ++r) sc[r] = 0.0f;\n"
    "        const float4* kr = Ks + (sub * FM_BC + col) * D4;\n"
    "#pragma unroll\n"
    "        for (int s4 = 0; s4 < HD / 4; ++s4) {\n"
    "            float4 kk = kr[(half * (HD / 4) + s4) ^ (col & 7u)];\n"
    "            sc = __builtin_amdgcn_mfma_f32_32x32x2f32(kk.x, qr[4 * s4 + 0], sc, 0, 0, 0);\n"
    "            sc = __builtin_amdgcn_mfma_f32_32x32x2f32(kk.y, qr[4 * s4 + 1], sc, 0, 0, 0);\n"
    "            sc = __builtin_amdgcn_mfma_f32_32x32x2f32(kk.z, qr[4 * s4 + 2], sc, 0, 0, 0);\n"
    "            sc = __builtin_amdgcn_mfma_f32_32x32x2f32(kk.w, qr[4 * s4 + 3], sc, 0, 0, 0);\n"
    "        }\n"
    "        float p[16];\n"
    "        float mt = -__builtin_inff();\n"
    "        const bool edge = j0 + FM_BC > kv || (causal && (int)(j0 + FM_BC - 1) > (int)q0 + off);\n"
    "#pragma unroll\n"
    "        for (int r = 0; r < 16; ++r) {\n"
    "            float x = sc[r];\n"
    "            if (edge) {\n"
    "                unsigned kj = j0 + 8u * (r / 4) + 4u * half + (r % 4);\n"
    "                bool vis = kj < kv && (!causal || (int)kj <= (int)qrow + off);\n"
    "                x = vis ? x : -__builtin_inff();\n"
    "            }\n"
    "            p[r] = x;\n"
    "            mt = fmaxf(mt, x);\n"
    "        }\n"
    "        mt = fmaxf(mt, __shfl_xor(mt, 32));\n"
    "        // keep the old max until a row's max grows by more than 2^8 (as\n"
    "        // FlashAttention-4): p stays <= 256, and most tiles skip the rescale\n"
    "        const bool grow = mt > m + 8.0f;\n"
    "        float mn = grow ? mt : m;\n"
    "        float ms = mn == -__builtin_inff() ? 0.0f : mn;\n"
    "        float ls = 0.0f;\n"
    "#pragma unroll\n"
    "        for (int r = 0; r < 16; ++r) { p[r] = exp2f(p[r] - ms); ls += p[r]; }\n"
    "        if (__any(grow)) {\n"
    "            float corr = exp2f(m - ms);\n"
    "            l *= corr;\n"
    "#pragma unroll\n"
    "            for (int b = 0; b < NB; ++b)\n"
    "#pragma unroll\n"
    "                for (int r = 0; r < 16; ++r) acc[b][r] *= corr;\n"
    "        }\n"
    "        l += ls;\n"
    "        m = mn;\n"
    "#pragma unroll\n"
    "        for (int r = 0; r < 16; ++r) {\n"
    "            const float* vr = Vs + sub * FM_BC * DM + (8u * (r / 4) + 4u * half + (r % 4)) * DM + col;\n"
    "#pragma unroll\n"
    "            for (int b = 0; b < NB; ++b)\n"
    "                acc[b] = __builtin_amdgcn_mfma_f32_32x32x2f32(vr[32 * b], p[r], acc[b], 0, 0, 0);\n"
    "        }\n"
    "        }\n"
    "    }\n"
    "    l += __shfl_xor(l, 32);\n"
    "    if (qrow < seq) {\n"
    "        float inv = l > 0.0f ? 1.0f / l : 0.0f;\n"
    "        float* orow = O + ((unsigned long long)h * seq + qrow) * d;\n"
    "#pragma unroll\n"
    "        for (int b = 0; b < NB; ++b)\n"
    "#pragma unroll\n"
    "            for (int r = 0; r < 16; ++r) {\n"
    "                unsigned c = 32u * b + 8u * (r / 4) + 4u * half + (r % 4);\n"
    "                if (c < d) orow[c] = acc[b][r] * inv;\n"
    "            }\n"
    "    }\n"
    "}\n"
    "template <int DM>\n"
    "__device__ __forceinline__ void flash_m16_impl(const float* __restrict__ Q, const float* __restrict__ K,\n"
    "                                                const float* __restrict__ V, float* __restrict__ O,\n"
    "                                                unsigned seq, unsigned kv, unsigned d, float scale, int causal) {\n"
    "    constexpr int QD = DM / 4;          // dims per lane group in QK^T\n"
    "    constexpr int NB = DM / 16;         // 16-dim output blocks\n"
    "    constexpr int D4 = DM / 4;\n"
    "    __shared__ float4 Ks[FM_BC * D4];\n"
    "    __shared__ float4 Vs[FM_BC * D4];\n"
    "    const unsigned h = blockIdx.x;\n"
    "    const unsigned qb = gridDim.y - 1u - blockIdx.y;\n"
    "    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;\n"
    "    const unsigned col = lane & 15u, grp = lane >> 4;\n"
    "    const unsigned q0 = qb * FM16_BR + wave * 16u;\n"
    "    const unsigned qrow = q0 + col;\n"
    "    const float* Kh = K + (unsigned long long)h * kv * d;\n"
    "    const float* Vh = V + (unsigned long long)h * kv * d;\n"
    "    const float qs = scale * 1.4426950408889634f;\n"
    "    float qr[QD];\n"
    "#pragma unroll\n"
    "    for (int s = 0; s < QD; ++s) {\n"
    "        unsigned c = grp * QD + s;\n"
    "        qr[s] = (qrow < seq && c < d) ? Q[((unsigned long long)h * seq + qrow) * d + c] * qs : 0.0f;\n"
    "    }\n"
    "    fm_v4 acc[NB];\n"
    "#pragma unroll\n"
    "    for (int b = 0; b < NB; ++b) acc[b] = (fm_v4){0.f, 0.f, 0.f, 0.f};\n"
    "    float m = -__builtin_inff(), l = 0.0f;\n"
    "    const int off = (int)kv - (int)seq;\n"
    "    unsigned kvEnd = kv, wEnd = kv;\n"
    "    if (causal) {\n"
    "        int lim = (int)(qb * FM16_BR + FM16_BR - 1) + off + 1;\n"
    "        kvEnd = lim <= 0 ? 0u : (lim < (int)kv ? (unsigned)lim : kv);\n"
    "        int wl = (int)(q0 + 15u) + off + 1;\n"
    "        wEnd = wl <= 0 ? 0u : (wl < (int)kv ? (unsigned)wl : kv);\n"
    "    }\n"
    "    const bool vec = (d & 3u) == 0u;\n"
    "    for (unsigned j0 = 0; j0 < kvEnd; j0 += FM_BC) {\n"
    "        __syncthreads();\n"
    "        for (unsigned e = threadIdx.x; e < FM_BC * D4; e += blockDim.x) {\n"
    "            unsigned r = e / D4, c4 = e % D4, j = j0 + r, c = c4 * 4u;\n"
    "            float4 kq = make_float4(0.f, 0.f, 0.f, 0.f), vq = kq;\n"
    "            if (j < kv && c < d) {\n"
    "                const float* kp = Kh + (unsigned long long)j * d + c;\n"
    "                const float* vp = Vh + (unsigned long long)j * d + c;\n"
    "                if (vec) { kq = *(const float4*)kp; vq = *(const float4*)vp; }\n"
    "                else {\n"
    "                    kq.x = kp[0]; vq.x = vp[0];\n"
    "                    if (c + 1 < d) { kq.y = kp[1]; vq.y = vp[1]; }\n"
    "                    if (c + 2 < d) { kq.z = kp[2]; vq.z = vp[2]; }\n"
    "                    if (c + 3 < d) { kq.w = kp[3]; vq.w = vp[3]; }\n"
    "                }\n"
    "            }\n"
    "            Ks[r * D4 + (c4 ^ (r & 7u))] = kq;\n"
    "            Vs[r * D4 + (c4 ^ (((r >> 2) & 1u) << 2))] = vq;\n"
    "        }\n"
    "        __syncthreads();\n"
    "        if (j0 >= wEnd) continue;\n"
    "        fm_v4 sc[2];\n"
    "#pragma unroll\n"
    "        for (int t = 0; t < 2; ++t) {\n"
    "            sc[t] = (fm_v4){0.f, 0.f, 0.f, 0.f};\n"
    "            const unsigned kr = 16u * t + col;\n"
    "            const float4* krow = Ks + kr * D4;\n"
    "#pragma unroll\n"
    "            for (int s4 = 0; s4 < QD / 4; ++s4) {\n"
    "                float4 kk = krow[(grp * (QD / 4) + s4) ^ (kr & 7u)];\n"
    "                sc[t] = __builtin_amdgcn_mfma_f32_16x16x4f32(kk.x, qr[4 * s4 + 0], sc[t], 0, 0, 0);\n"
    "                sc[t] = __builtin_amdgcn_mfma_f32_16x16x4f32(kk.y, qr[4 * s4 + 1], sc[t], 0, 0, 0);\n"
    "                sc[t] = __builtin_amdgcn_mfma_f32_16x16x4f32(kk.z, qr[4 * s4 + 2], sc[t], 0, 0, 0);\n"
    "                sc[t] = __builtin_amdgcn_mfma_f32_16x16x4f32(kk.w, qr[4 * s4 + 3], sc[t], 0, 0, 0);\n"
    "            }\n"
    "        }\n"
    "        float p[8];\n"
    "        float mt = -__builtin_inff();\n"
    "        const bool edge = j0 + FM_BC > kv || (causal && (int)(j0 + FM_BC - 1) > (int)q0 + off);\n"
    "#pragma unroll\n"
    "        for (int t = 0; t < 2; ++t)\n"
    "#pragma unroll\n"
    "            for (int r = 0; r < 4; ++r) {\n"
    "                float x = sc[t][r];\n"
    "                if (edge) {\n"
    "                    unsigned kj = j0 + 16u * t + 4u * grp + r;\n"
    "                    bool vis = kj < kv && (!causal || (int)kj <= (int)qrow + off);\n"
    "                    x = vis ? x : -__builtin_inff();\n"
    "                }\n"
    "                p[4 * t + r] = x;\n"
    "                mt = fmaxf(mt, x);\n"
    "            }\n"
    "        mt = fmaxf(mt, __shfl_xor(mt, 16));\n"
    "        mt = fmaxf(mt, __shfl_xor(mt, 32));\n"
    "        // keep the old max until a row's max grows by more than 2^8 (as\n"
    "        // FlashAttention-4): p stays <= 256, and most tiles skip the rescale\n"
    "        const bool grow = mt > m + 8.0f;\n"
    "        float mn = grow ? mt : m;\n"
    "        float ms = mn == -__builtin_inff() ? 0.0f : mn;\n"
    "        float ls = 0.0f;\n"
    "#pragma unroll\n"
    "        for (int r = 0; r < 8; ++r) { p[r] = exp2f(p[r] - ms); ls += p[r]; }\n"
    "        if (__any(grow)) {\n"
    "            float corr = exp2f(m - ms);\n"
    "            l *= corr;\n"
    "#pragma unroll\n"
    "            for (int b = 0; b < NB; ++b) acc[b] *= corr;\n"
    "        }\n"
    "        l += ls;\n"
    "        m = mn;\n"
    "        const float* vf = (const float*)Vs;\n"
    "#pragma unroll\n"
    "        for (int t = 0; t < 2; ++t)\n"
    "#pragma unroll\n"
    "            for (int r = 0; r < 4; ++r) {\n"
    "                const unsigned key = 16u * t + 4u * grp + r;\n"
    "                const float* vr = vf + key * DM;\n"
    "                const unsigned sw = ((key >> 2) & 1u) << 4;\n"
    "#pragma unroll\n"
    "                for (int b = 0; b < NB; ++b)\n"
    "                    acc[b] = __builtin_amdgcn_mfma_f32_16x16x4f32(vr[(16u * b + col) ^ sw], p[4 * t + r], acc[b], 0, 0, 0);\n"
    "            }\n"
    "    }\n"
    "    l += __shfl_xor(l, 16);\n"
    "    l += __shfl_xor(l, 32);\n"
    "    if (qrow < seq) {\n"
    "        float inv = l > 0.0f ? 1.0f / l : 0.0f;\n"
    "        float* orow = O + ((unsigned long long)h * seq + qrow) * d;\n"
    "#pragma unroll\n"
    "        for (int b = 0; b < NB; ++b)\n"
    "#pragma unroll\n"
    "            for (int r = 0; r < 4; ++r) {\n"
    "                unsigned c = 16u * b + 4u * grp + r;\n"
    "                if (c < d) orow[c] = acc[b][r] * inv;\n"
    "            }\n"
    "    }\n"
    "}\n"
    "// Linear attention, chunk-parallel, three passes (rows in chunks of LA_R):\n"
    "//   la_state: W[c] = phi(K_c)^T V_c and Z[c] = sum phi(K_c) per chunk (MFMA)\n"
    "//   la_scan:  causal: exclusive prefix over chunks; else the total, in chunk 0\n"
    "//   la_out:   out = (phi(Q_c) W + A V_c) / (phi(Q_c) . Z + rowsum A), A the\n"
    "//             causal intra-chunk scores phi(Q_c) phi(K_c)^T (j <= i); phi(Q)\n"
    "//             rows sit in registers as the MFMA operand, like flash's q.\n"
    "#define LA_T 32\n"
    "#define LA_R 128\n"
    "// K/V tile staging, split so the next tile's global loads can be issued\n"
    "// before the current tile is used: la_fetch reads rows [j0, j0 + LA_T) of\n"
    "// phi(K) (all DM dims) and of V (cols c0 .. c0 + VC) into registers, la_put\n"
    "// writes them to Ks (swizzled float4, as the flash kernel) and Vs ([LA_T][VC]).\n"
    "template <int DM, int VC> struct LaTile {\n"
    "    static constexpr int KE = LA_T * DM / 4 / 256, VE = LA_T * VC / 4 / 256;\n"
    "    float4 k[KE], v[VE];\n"
    "};\n"
    "__device__ __forceinline__ float4 la_ld4(const float* p, unsigned c, unsigned d, bool vec, bool phi) {\n"
    "    float4 x = make_float4(0.f, 0.f, 0.f, 0.f);\n"
    "    if (vec) x = *(const float4*)p;\n"
    "    else {\n"
    "        x.x = p[0];\n"
    "        if (c + 1 < d) x.y = p[1];\n"
    "        if (c + 2 < d) x.z = p[2];\n"
    "        if (c + 3 < d) x.w = p[3];\n"
    "    }\n"
    "    if (phi) {\n"
    "        x.x = la_phi(x.x); x.y = c + 1 < d ? la_phi(x.y) : 0.f;\n"
    "        x.z = c + 2 < d ? la_phi(x.z) : 0.f; x.w = c + 3 < d ? la_phi(x.w) : 0.f;\n"
    "    }\n"
    "    return x;\n"
    "}\n"
    "template <int DM, int VC>\n"
    "__device__ __forceinline__ void la_fetch(LaTile<DM, VC>& t, const float* __restrict__ Kh, const float* __restrict__ Vh,\n"
    "                                         unsigned j0, unsigned jend, unsigned d, unsigned c0) {\n"
    "    constexpr int D4 = DM / 4, V4 = VC / 4;\n"
    "    const bool vec = (d & 3u) == 0u;\n"
    "#pragma unroll\n"
    "    for (int i = 0; i < t.KE; ++i) {\n"
    "        unsigned e = threadIdx.x + 256u * i, r = e / D4, c = (e % D4) * 4u, j = j0 + r;\n"
    "        t.k[i] = (j < jend && c < d) ? la_ld4(Kh + (unsigned long long)j * d + c, c, d, vec, true) : make_float4(0.f, 0.f, 0.f, 0.f);\n"
    "    }\n"
    "#pragma unroll\n"
    "    for (int i = 0; i < t.VE; ++i) {\n"
    "        unsigned e = threadIdx.x + 256u * i, r = e / V4, c = c0 + (e % V4) * 4u, j = j0 + r;\n"
    "        t.v[i] = (j < jend && c < d) ? la_ld4(Vh + (unsigned long long)j * d + c, c, d, vec, false) : make_float4(0.f, 0.f, 0.f, 0.f);\n"
    "    }\n"
    "}\n"
    "template <int DM, int VC>\n"
    "__device__ __forceinline__ void la_put(const LaTile<DM, VC>& t, float4* Ks, float* Vs) {\n"
    "    constexpr int D4 = DM / 4, V4 = VC / 4;\n"
    "#pragma unroll\n"
    "    for (int i = 0; i < t.KE; ++i) {\n"
    "        unsigned e = threadIdx.x + 256u * i, r = e / D4, c4 = e % D4;\n"
    "        Ks[r * D4 + (c4 ^ (r & 7u))] = t.k[i];\n"
    "    }\n"
    "#pragma unroll\n"
    "    for (int i = 0; i < t.VE; ++i) {\n"
    "        unsigned e = threadIdx.x + 256u * i, r = e / V4;\n"
    "        *(float4*)&Vs[r * VC + (e % V4) * 4u] = t.v[i];\n"
    "    }\n"
    "}\n"
    "\n"
    "// Pass 1: W[h][c] = phi(K_c)^T V_c ([DM][DM]) and Z[h][c] = sum phi(K_c) ([DM])\n"
    "// over rows [c*rows, (c+1)*rows). Wave w owns the 32x32 blocks w, w+4, ...\n"
    "template <int DM>\n"
    "__device__ __forceinline__ void la_state_impl(const float* __restrict__ K, const float* __restrict__ V,\n"
    "                                              float* __restrict__ W, float* __restrict__ Z,\n"
    "                                              unsigned kv, unsigned d, unsigned rows, unsigned nc) {\n"
    "    constexpr int D4 = DM / 4, NBL = (DM / 32) * (DM / 32), PW = (NBL + 3) / 4;\n"
    "    __shared__ float4 Ks[LA_T * D4];\n"
    "    __shared__ float Vs[LA_T * DM];\n"
    "    const unsigned c = blockIdx.x, h = blockIdx.y;\n"
    "    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6, col = lane & 31u, half = lane >> 5;\n"
    "    const float* Kh = K + (unsigned long long)h * kv * d;\n"
    "    const float* Vh = V + (unsigned long long)h * kv * d;\n"
    "    const unsigned r0 = c * rows, r1 = min(kv, r0 + rows);\n"
    "    fm_v16 acc[PW];\n"
    "#pragma unroll\n"
    "    for (int b = 0; b < PW; ++b)\n"
    "#pragma unroll\n"
    "        for (int r = 0; r < 16; ++r) acc[b][r] = 0.0f;\n"
    "    float z = 0.0f;\n"
    "    const float* kf = (const float*)Ks;\n"
    "    LaTile<DM, DM> tile;\n"
    "    if (r0 < r1) la_fetch<DM, DM>(tile, Kh, Vh, r0, r1, d, 0u);\n"
    "    for (unsigned j0 = r0; j0 < r1; j0 += LA_T) {\n"
    "        __syncthreads();\n"
    "        la_put<DM, DM>(tile, Ks, Vs);\n"
    "        __syncthreads();\n"
    "        if (j0 + LA_T < r1) la_fetch<DM, DM>(tile, Kh, Vh, j0 + LA_T, r1, d, 0u);\n"
    "#pragma unroll\n"
    "        for (int b = 0; b < PW; ++b) {\n"
    "            const int blk = wave + 4 * b;\n"
    "            if (blk < NBL) {\n"
    "                const unsigned bi = blk / (DM / 32), bj = blk % (DM / 32);\n"
    "                const unsigned dk = 32u * bi + col;\n"
    "#pragma unroll\n"
    "                for (int s = 0; s < LA_T / 2; ++s) {\n"
    "                    const unsigned key = 2u * s + half;\n"
    "                    float a = kf[key * DM + ((((dk >> 2) ^ (key & 7u)) << 2) | (dk & 3u))];\n"
    "                    float bb = Vs[key * DM + 32u * bj + col];\n"
    "                    acc[b] = __builtin_amdgcn_mfma_f32_32x32x2f32(a, bb, acc[b], 0, 0, 0);\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "        if (threadIdx.x < DM) {\n"
    "            const unsigned dk = threadIdx.x;\n"
    "#pragma unroll 8\n"
    "            for (int key = 0; key < LA_T; ++key) z += kf[key * DM + ((((dk >> 2) ^ (key & 7u)) << 2) | (dk & 3u))];\n"
    "        }\n"
    "    }\n"
    "    float* w = W + ((unsigned long long)h * nc + c) * DM * DM;\n"
    "#pragma unroll\n"
    "    for (int b = 0; b < PW; ++b) {\n"
    "        const int blk = wave + 4 * b;\n"
    "        if (blk < NBL) {\n"
    "            const unsigned bi = blk / (DM / 32), bj = blk % (DM / 32);\n"
    "#pragma unroll\n"
    "            for (int r = 0; r < 16; ++r)\n"
    "                w[(32u * bi + 8u * (r / 4) + 4u * half + (r % 4)) * DM + 32u * bj + col] = acc[b][r];\n"
    "        }\n"
    "    }\n"
    "    if (threadIdx.x < DM) Z[((unsigned long long)h * nc + c) * DM + threadIdx.x] = z;\n"
    "}\n"
    "\n"
    "// Pass 2: causal -> exclusive prefix over chunks, in place; else the total, in chunk 0.\n"
    "template <int DM>\n"
    "__device__ __forceinline__ void la_scan_impl(float* __restrict__ W, float* __restrict__ Z, unsigned nc, int causal) {\n"
    "    const unsigned h = blockIdx.y, e = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (e >= DM * DM + DM) return;\n"
    "    float* p;\n"
    "    unsigned long long stride;\n"
    "    if (e < DM * DM) { p = W + (unsigned long long)h * nc * DM * DM + e; stride = DM * DM; }\n"
    "    else { p = Z + (unsigned long long)h * nc * DM + (e - DM * DM); stride = DM; }\n"
    "    float run = 0.0f;\n"
    "    for (unsigned c0 = 0; c0 < nc; c0 += 8) {\n"
    "        float v[8];\n"
    "#pragma unroll\n"
    "        for (int u = 0; u < 8; ++u) v[u] = c0 + u < nc ? p[(c0 + u) * stride] : 0.0f;\n"
    "        if (causal) {\n"
    "#pragma unroll\n"
    "            for (int u = 0; u < 8; ++u)\n"
    "                if (c0 + u < nc) { p[(c0 + u) * stride] = run; run += v[u]; }\n"
    "        } else {\n"
    "#pragma unroll\n"
    "            for (int u = 0; u < 8; ++u) run += v[u];\n"
    "        }\n"
    "    }\n"
    "    if (!causal) p[0] = run;\n"
    "}\n"
    "\n"
    "// Pass 3: rows [blk*LA_R, +LA_R) of the output, columns c0 .. c0 + DO\n"
    "// (c0 = blockIdx.z * DO). Wave w: rows +32w. Lane: row col, phi(q) dims\n"
    "// half*DM/2 .. in registers.\n"
    "template <int DM, int DO>\n"
    "__device__ __forceinline__ void la_out_impl(const float* __restrict__ Q, const float* __restrict__ K,\n"
    "                                            const float* __restrict__ V, const float* __restrict__ W,\n"
    "                                            const float* __restrict__ Z, float* __restrict__ O,\n"
    "                                            unsigned seq, unsigned kv, unsigned d, unsigned nc, int causal) {\n"
    "    constexpr int HD = DM / 2, NB = DO / 32, D4 = DM / 4;\n"
    "    constexpr int WR = DM < 64 ? DM : 64;            // state rows per pass (both half-waves)\n"
    "    constexpr int NP = DM / WR, SP = HD / NP;          // passes, dims per half-wave per pass\n"
    "    constexpr int P1 = LA_T * DM + LA_T * DO, P2 = WR * DO, P3 = 4 * 32 * DO;\n"
    "    constexpr int POOL = P1 > P2 ? (P1 > P3 ? P1 : P3) : (P2 > P3 ? P2 : P3);   // floats, phases reuse it\n"
    "    __shared__ float4 pool[POOL / 4];\n"
    "    float4* Ks = pool;\n"
    "    float* Vs = (float*)(pool + LA_T * D4);\n"
    "    float* Wl = (float*)pool;\n"
    "    const unsigned blk = blockIdx.x, h = blockIdx.y, c0 = blockIdx.z * DO;\n"
    "    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6, col = lane & 31u, half = lane >> 5;\n"
    "    const unsigned row = blk * LA_R + wave * 32u + col;\n"
    "    float qr[HD];\n"
    "    {\n"
    "        // phi(Q) for the block's LA_R rows, loaded coalesced through LDS in\n"
    "        // pieces of 32 x DM per wave-round (rows of all waves together)\n"
    "        float4* qs = pool;                       // [4 waves][32 rows][D4], col4 ^ (row & 7)\n"
    "        constexpr int QCH = POOL / (4 * 32 * DM) >= 1 ? 32 : 32 * POOL / (4 * 32 * DM);   // rows per wave per round\n"
    "        static_assert(4 * QCH * DM <= POOL, \"pool\");\n"
    "        for (unsigned r0 = 0; r0 < 32u; r0 += QCH) {\n"
    "            __syncthreads();\n"
    "            for (unsigned e = threadIdx.x; e < 4u * QCH * D4; e += blockDim.x) {\n"
    "                unsigned wr = e / (QCH * D4), rr = (e / D4) % QCH, c4 = e % D4;\n"
    "                unsigned grow = blk * LA_R + wr * 32u + r0 + rr, c = 4u * c4;\n"
    "                float4 x = make_float4(0.f, 0.f, 0.f, 0.f);\n"
    "                if (grow < seq && c < d)\n"
    "                    x = la_ld4(Q + ((unsigned long long)h * seq + grow) * d + c, c, d, (d & 3u) == 0u, true);\n"
    "                qs[(wr * QCH + rr) * D4 + (c4 ^ (rr & 7u))] = x;\n"
    "            }\n"
    "            __syncthreads();\n"
    "            if (col >= r0 && col < r0 + QCH) {\n"
    "                const unsigned rr = col - r0;\n"
    "#pragma unroll\n"
    "                for (int s4 = 0; s4 < HD / 4; ++s4) {\n"
    "                    float4 x = qs[(wave * QCH + rr) * D4 + ((half * (HD / 4) + s4) ^ (rr & 7u))];\n"
    "                    qr[4 * s4] = x.x; qr[4 * s4 + 1] = x.y; qr[4 * s4 + 2] = x.z; qr[4 * s4 + 3] = x.w;\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    const float* Ws = W + ((unsigned long long)h * nc + (causal ? blk : 0u)) * DM * DM;\n"
    "    const float* Zs = Z + ((unsigned long long)h * nc + (causal ? blk : 0u)) * DM;\n"
    "    fm_v16 acc[NB];\n"
    "    float den = 0.0f;\n"
    "#pragma unroll\n"
    "    for (int b = 0; b < NB; ++b)\n"
    "#pragma unroll\n"
    "        for (int r = 0; r < 16; ++r) acc[b][r] = 0.0f;\n"
    "    if (causal) {\n"
    "        const float* Kh = K + (unsigned long long)h * kv * d;\n"
    "        const float* Vh = V + (unsigned long long)h * kv * d;\n"
    "        const unsigned cs = blk * LA_R;\n"
    "        LaTile<DM, DO> tile;\n"
    "        if (cs < kv) la_fetch<DM, DO>(tile, Kh, Vh, cs, kv, d, c0);\n"
    "        for (unsigned t = 0; t < LA_R / LA_T; ++t) {\n"
    "            const unsigned j0 = cs + t * LA_T;\n"
    "            if (j0 >= kv) break;\n"
    "            __syncthreads();\n"
    "            la_put<DM, DO>(tile, Ks, Vs);\n"
    "            __syncthreads();\n"
    "            if (t + 1 < LA_R / LA_T && j0 + LA_T < kv) la_fetch<DM, DO>(tile, Kh, Vh, j0 + LA_T, kv, d, c0);\n"
    "            if (t > wave) continue;\n"
    "            fm_v16 sc;\n"
    "#pragma unroll\n"
    "            for (int r = 0; r < 16; ++r) sc[r] = 0.0f;\n"
    "            const float4* kr = Ks + col * D4;\n"
    "#pragma unroll\n"
    "            for (int s4 = 0; s4 < HD / 4; ++s4) {\n"
    "                float4 kk = kr[(half * (HD / 4) + s4) ^ (col & 7u)];\n"
    "                sc = __builtin_amdgcn_mfma_f32_32x32x2f32(kk.x, qr[4 * s4 + 0], sc, 0, 0, 0);\n"
    "                sc = __builtin_amdgcn_mfma_f32_32x32x2f32(kk.y, qr[4 * s4 + 1], sc, 0, 0, 0);\n"
    "                sc = __builtin_amdgcn_mfma_f32_32x32x2f32(kk.z, qr[4 * s4 + 2], sc, 0, 0, 0);\n"
    "                sc = __builtin_amdgcn_mfma_f32_32x32x2f32(kk.w, qr[4 * s4 + 3], sc, 0, 0, 0);\n"
    "            }\n"
    "            float a[16];\n"
    "#pragma unroll\n"
    "            for (int r = 0; r < 16; ++r) {\n"
    "                unsigned kl = 8u * (r / 4) + 4u * half + (r % 4);\n"
    "                a[r] = (t < wave || kl <= col) ? sc[r] : 0.0f;\n"
    "                den += a[r];\n"
    "            }\n"
    "#pragma unroll\n"
    "            for (int r = 0; r < 16; ++r) {\n"
    "                const float* vr = Vs + (8u * (r / 4) + 4u * half + (r % 4)) * DO + col;\n"
    "#pragma unroll\n"
    "                for (int b = 0; b < NB; ++b)\n"
    "                    acc[b] = __builtin_amdgcn_mfma_f32_32x32x2f32(vr[32 * b], a[r], acc[b], 0, 0, 0);\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    // inter-chunk: acc^T += S^T phi(Q)^T, den += phi(q) . z\n"
    "#pragma unroll\n"
    "    for (int s = 0; s < HD; ++s) den = fmaf(qr[s], Zs[half * HD + s], den);\n"
    "#pragma unroll\n"
    "    for (int p = 0; p < NP; ++p) {\n"
    "        __syncthreads();\n"
    "        for (unsigned e = threadIdx.x; e < WR * DO / 4; e += blockDim.x) {\n"
    "            unsigned rr = e / (DO / 4), c4 = e % (DO / 4);\n"
    "            unsigned grow = (rr / SP) * HD + p * SP + (rr % SP);\n"
    "            *(float4*)&Wl[rr * DO + 4u * c4] = *(const float4*)(Ws + (unsigned long long)grow * DM + c0 + 4u * c4);\n"
    "        }\n"
    "        __syncthreads();\n"
    "#pragma unroll\n"
    "        for (int s = 0; s < SP; ++s) {\n"
    "            const float* wr = Wl + (half * SP + s) * DO + col;\n"
    "#pragma unroll\n"
    "            for (int b = 0; b < NB; ++b)\n"
    "                acc[b] = __builtin_amdgcn_mfma_f32_32x32x2f32(wr[32 * b], qr[p * SP + s], acc[b], 0, 0, 0);\n"
    "        }\n"
    "    }\n"
    "    den += __shfl_xor(den, 32);\n"
    "    const float inv = den != 0.0f ? 1.0f / den : 0.0f;\n"
    "    // stage the wave's 32 x DO tile in LDS (all of Ks/Vs/Wl is free now), then\n"
    "    // write whole row segments\n"
    "    __syncthreads();\n"
    "    float4* st = pool + wave * 32u * (DO / 4);   // [32][DO/4] float4, column index ^ (row & 15)\n"
    "#pragma unroll\n"
    "    for (int b = 0; b < NB; ++b)\n"
    "#pragma unroll\n"
    "        for (int r4 = 0; r4 < 4; ++r4) {\n"
    "            unsigned c4 = 8u * b + 2u * r4 + half;\n"
    "            st[col * (DO / 4) + (c4 ^ (col & 15u))] = make_float4(acc[b][4 * r4] * inv, acc[b][4 * r4 + 1] * inv,\n"
    "                                                           acc[b][4 * r4 + 2] * inv, acc[b][4 * r4 + 3] * inv);\n"
    "        }\n"
    "    __syncthreads();\n"
    "    const unsigned rbase = blk * LA_R + wave * 32u;\n"
    "    if ((d & 3u) == 0u) {\n"
    "        for (unsigned e = lane; e < 32u * (DO / 4); e += 64u) {\n"
    "            unsigned rr = e / (DO / 4), c = 4u * (e % (DO / 4));\n"
    "            if (rbase + rr < seq && c0 + c < d)\n"
    "                *(float4*)(O + ((unsigned long long)h * seq + rbase + rr) * d + c0 + c) = st[rr * (DO / 4) + ((c / 4u) ^ (rr & 15u))];\n"
    "        }\n"
    "    } else {\n"
    "        for (unsigned e = lane; e < 32u * DO; e += 64u) {\n"
    "            unsigned rr = e / DO, c = e % DO;\n"
    "            if (rbase + rr < seq && c0 + c < d)\n"
    "                O[((unsigned long long)h * seq + rbase + rr) * d + c0 + c] = ((const float*)&st[rr * (DO / 4) + ((c / 4u) ^ (rr & 15u))])[c & 3u];\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "// Causal linear attention in one pass. Block = (chunk c of LA_R rows, head h,\n"
    "// column slice z), taken in chunk order from an atomic ticket. It loads the\n"
    "// chunk's K/V tiles once and uses them twice: for the masked intra-chunk term\n"
    "// and for the chunk's own state S_c = phi(K_c)^T V_c (slice z). The prefix of\n"
    "// earlier chunks comes from a decoupled look-back (Merrill & Garland): each\n"
    "// block publishes S_c (flag 1) right away, then sums predecessors' states\n"
    "// back to the first one that has published an inclusive prefix (flag 2), and\n"
    "// publishes its own inclusive prefix. Blocks only ever wait on lower tickets,\n"
    "// which are already running, so the waits always end.\n"
    "__device__ __forceinline__ float la_ldg(const float* p) {\n"
    "    return __hip_atomic_load((float*)p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);\n"
    "}\n"
    "template <int DM, int DO>\n"
    "__device__ __forceinline__ void la_causal_impl(const float* __restrict__ Q, const float* __restrict__ K,\n"
    "                                               const float* __restrict__ V, float* __restrict__ R,\n"
    "                                               int* __restrict__ F, float* __restrict__ O,\n"
    "                                               unsigned seq, unsigned heads, unsigned d, unsigned nc) {\n"
    "    constexpr int HD = DM / 2, NB = DO / 32, D4 = DM / 4, NZ = DM / DO;\n"
    "    constexpr int RS = DM * DO + DM;                       // one record: state slice, then z\n"
    "    constexpr int BJ = DO / 32, NBL = (DM / 32) * BJ, PW = (NBL + 3) / 4;\n"
    "    constexpr int P1 = LA_T * DM + LA_T * DO, P2 = DM * DO, P3 = 4 * 32 * DO;\n"
    "    constexpr int PM = P1 > P2 ? (P1 > P3 ? P1 : P3) : (P2 > P3 ? P2 : P3);\n"
    "    constexpr int POOL = PM >= P2 + DM + 4 ? PM : P2 + DM + 4;   // Pl, then Zp and two flags after it\n"
    "    __shared__ float4 pool[POOL / 4];\n"
    "    float* Zp = (float*)pool + P2;\n"
    "    unsigned& s_tk = *(unsigned*)((float*)pool + P2 + DM);\n"
    "    int& s_flag = *(int*)((float*)pool + P2 + DM + 1);\n"
    "    float4* Ks = pool;\n"
    "    float* Vs = (float*)(pool + LA_T * D4);\n"
    "    float* Pl = (float*)pool;\n"
    "    const float* kf = (const float*)Ks;\n"
    "    if (threadIdx.x == 0) s_tk = atomicAdd((unsigned*)(F + heads * NZ * nc), 1u);\n"
    "    __syncthreads();\n"
    "    const unsigned tk = s_tk % (nc * heads * NZ);   // wraps if the kernel is replayed without the reset\n"
    "    __syncthreads();                         // the pool is reused below\n"
    "    const unsigned c = tk / (heads * NZ), hz = tk % (heads * NZ), h = hz / NZ, z = hz % NZ, c0 = z * DO;\n"
    "    const unsigned kv = seq;\n"
    "    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6, col = lane & 31u, half = lane >> 5;\n"
    "    const unsigned row = c * LA_R + wave * 32u + col;\n"
    "    float qr[HD];\n"
    "    {\n"
    "        float4* qs = pool;\n"
    "        constexpr int QCH = POOL / (4 * 32 * DM) >= 1 ? 32 : 32 * POOL / (4 * 32 * DM);\n"
    "        for (unsigned r0 = 0; r0 < 32u; r0 += QCH) {\n"
    "            __syncthreads();\n"
    "            for (unsigned e = threadIdx.x; e < 4u * QCH * D4; e += blockDim.x) {\n"
    "                unsigned wr = e / (QCH * D4), rr = (e / D4) % QCH, c4 = e % D4;\n"
    "                unsigned grow = c * LA_R + wr * 32u + r0 + rr, cc = 4u * c4;\n"
    "                float4 x = make_float4(0.f, 0.f, 0.f, 0.f);\n"
    "                if (grow < seq && cc < d)\n"
    "                    x = la_ld4(Q + ((unsigned long long)h * seq + grow) * d + cc, cc, d, (d & 3u) == 0u, true);\n"
    "                qs[(wr * QCH + rr) * D4 + (c4 ^ (rr & 7u))] = x;\n"
    "            }\n"
    "            __syncthreads();\n"
    "            if (col >= r0 && col < r0 + QCH) {\n"
    "                const unsigned rr = col - r0;\n"
    "#pragma unroll\n"
    "                for (int s4 = 0; s4 < HD / 4; ++s4) {\n"
    "                    float4 x = qs[(wave * QCH + rr) * D4 + ((half * (HD / 4) + s4) ^ (rr & 7u))];\n"
    "                    qr[4 * s4] = x.x; qr[4 * s4 + 1] = x.y; qr[4 * s4 + 2] = x.z; qr[4 * s4 + 3] = x.w;\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    fm_v16 acc[NB], st[PW];\n"
    "    float den = 0.0f, zl = 0.0f;\n"
    "#pragma unroll\n"
    "    for (int b = 0; b < NB; ++b)\n"
    "#pragma unroll\n"
    "        for (int r = 0; r < 16; ++r) acc[b][r] = 0.0f;\n"
    "#pragma unroll\n"
    "    for (int b = 0; b < PW; ++b)\n"
    "#pragma unroll\n"
    "        for (int r = 0; r < 16; ++r) st[b][r] = 0.0f;\n"
    "    {\n"
    "        const float* Kh = K + (unsigned long long)h * kv * d;\n"
    "        const float* Vh = V + (unsigned long long)h * kv * d;\n"
    "        const unsigned cs = c * LA_R;\n"
    "        LaTile<DM, DO> tile;\n"
    "        la_fetch<DM, DO>(tile, Kh, Vh, cs, kv, d, c0);\n"
    "        for (unsigned t = 0; t < LA_R / LA_T; ++t) {\n"
    "            const unsigned j0 = cs + t * LA_T;\n"
    "            if (j0 >= kv) break;\n"
    "            __syncthreads();\n"
    "            la_put<DM, DO>(tile, Ks, Vs);\n"
    "            __syncthreads();\n"
    "            if (t + 1 < LA_R / LA_T && j0 + LA_T < kv) la_fetch<DM, DO>(tile, Kh, Vh, j0 + LA_T, kv, d, c0);\n"
    "            // the chunk's state slice: S[dk][dv] += sum_key phi(k)[dk] v[dv]\n"
    "#pragma unroll\n"
    "            for (int b = 0; b < PW; ++b) {\n"
    "                const int bl = wave + 4 * b;\n"
    "                if (bl < NBL) {\n"
    "                    const unsigned dk = 32u * (bl / BJ) + col, dv = 32u * (bl % BJ) + col;\n"
    "#pragma unroll\n"
    "                    for (int s = 0; s < LA_T / 2; ++s) {\n"
    "                        const unsigned key = 2u * s + half;\n"
    "                        float a = kf[key * DM + ((((dk >> 2) ^ (key & 7u)) << 2) | (dk & 3u))];\n"
    "                        st[b] = __builtin_amdgcn_mfma_f32_32x32x2f32(a, Vs[key * DO + dv], st[b], 0, 0, 0);\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "            if (threadIdx.x < DM) {\n"
    "                const unsigned dk = threadIdx.x;\n"
    "#pragma unroll 8\n"
    "                for (int key = 0; key < LA_T; ++key) zl += kf[key * DM + ((((dk >> 2) ^ (key & 7u)) << 2) | (dk & 3u))];\n"
    "            }\n"
    "            if (t <= wave) {\n"
    "                fm_v16 sc;\n"
    "#pragma unroll\n"
    "                for (int r = 0; r < 16; ++r) sc[r] = 0.0f;\n"
    "                const float4* kr = Ks + col * D4;\n"
    "#pragma unroll\n"
    "                for (int s4 = 0; s4 < HD / 4; ++s4) {\n"
    "                    float4 kk = kr[(half * (HD / 4) + s4) ^ (col & 7u)];\n"
    "                    sc = __builtin_amdgcn_mfma_f32_32x32x2f32(kk.x, qr[4 * s4 + 0], sc, 0, 0, 0);\n"
    "                    sc = __builtin_amdgcn_mfma_f32_32x32x2f32(kk.y, qr[4 * s4 + 1], sc, 0, 0, 0);\n"
    "                    sc = __builtin_amdgcn_mfma_f32_32x32x2f32(kk.z, qr[4 * s4 + 2], sc, 0, 0, 0);\n"
    "                    sc = __builtin_amdgcn_mfma_f32_32x32x2f32(kk.w, qr[4 * s4 + 3], sc, 0, 0, 0);\n"
    "                }\n"
    "                float a[16];\n"
    "#pragma unroll\n"
    "                for (int r = 0; r < 16; ++r) {\n"
    "                    unsigned kl = 8u * (r / 4) + 4u * half + (r % 4);\n"
    "                    a[r] = (t < wave || kl <= col) ? sc[r] : 0.0f;\n"
    "                    den += a[r];\n"
    "                }\n"
    "#pragma unroll\n"
    "                for (int r = 0; r < 16; ++r) {\n"
    "                    const float* vr = Vs + (8u * (r / 4) + 4u * half + (r % 4)) * DO + col;\n"
    "#pragma unroll\n"
    "                    for (int b = 0; b < NB; ++b)\n"
    "                        acc[b] = __builtin_amdgcn_mfma_f32_32x32x2f32(vr[32 * b], a[r], acc[b], 0, 0, 0);\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    // publish S_c: inclusive already for chunk 0\n"
    "    float* Rh = R + (unsigned long long)(h * NZ + z) * nc * 2u * RS;\n"
    "    int* Fh = F + (h * NZ + z) * nc;\n"
    "    {\n"
    "        float* rp = Rh + ((unsigned long long)c * 2u + (c == 0 ? 1u : 0u)) * RS;\n"
    "#pragma unroll\n"
    "        for (int b = 0; b < PW; ++b) {\n"
    "            const int bl = wave + 4 * b;\n"
    "            if (bl < NBL)\n"
    "#pragma unroll\n"
    "                for (int r = 0; r < 16; ++r)\n"
    "                    rp[(32u * (bl / BJ) + 8u * (r / 4) + 4u * half + (r % 4)) * DO + 32u * (bl % BJ) + col] = st[b][r];\n"
    "        }\n"
    "        if (threadIdx.x < DM) rp[DM * DO + threadIdx.x] = zl;\n"
    "        __threadfence();\n"
    "        __syncthreads();\n"
    "        if (threadIdx.x == 0) __hip_atomic_store(&Fh[c], c == 0 ? 2 : 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);\n"
    "    }\n"
    "    // look back: Pl, Zp = sums over chunks < c\n"
    "    for (unsigned e = threadIdx.x; e < DM * DO; e += blockDim.x) Pl[e] = 0.0f;\n"
    "    if (threadIdx.x < DM) Zp[threadIdx.x] = 0.0f;\n"
    "    __syncthreads();\n"
    "    for (int u = (int)c - 1; u >= 0; --u) {\n"
    "        if (threadIdx.x == 0) {\n"
    "            int f;\n"
    "            while ((f = __hip_atomic_load(&Fh[u], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)) == 0) __builtin_amdgcn_s_sleep(1);\n"
    "            s_flag = f;\n"
    "        }\n"
    "        __syncthreads();\n"
    "        const int f = s_flag;\n"
    "        const float* rp = Rh + ((unsigned long long)u * 2u + (f == 2 ? 1u : 0u)) * RS;\n"
    "        for (unsigned e = threadIdx.x; e < DM * DO; e += blockDim.x) Pl[e] += la_ldg(rp + e);\n"
    "        if (threadIdx.x < DM) Zp[threadIdx.x] += la_ldg(rp + DM * DO + threadIdx.x);\n"
    "        __syncthreads();\n"
    "        if (f == 2) break;\n"
    "    }\n"
    "    if (c > 0) {\n"
    "        float* rp = Rh + ((unsigned long long)c * 2u + 1u) * RS;\n"
    "#pragma unroll\n"
    "        for (int b = 0; b < PW; ++b) {\n"
    "            const int bl = wave + 4 * b;\n"
    "            if (bl < NBL)\n"
    "#pragma unroll\n"
    "                for (int r = 0; r < 16; ++r) {\n"
    "                    const unsigned i = (32u * (bl / BJ) + 8u * (r / 4) + 4u * half + (r % 4)) * DO + 32u * (bl % BJ) + col;\n"
    "                    rp[i] = Pl[i] + st[b][r];\n"
    "                }\n"
    "        }\n"
    "        if (threadIdx.x < DM) rp[DM * DO + threadIdx.x] = Zp[threadIdx.x] + zl;\n"
    "        __threadfence();\n"
    "        __syncthreads();\n"
    "        if (threadIdx.x == 0) __hip_atomic_store(&Fh[c], 2, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);\n"
    "    }\n"
    "    // inter-chunk: acc^T += P^T phi(Q)^T, den += phi(q) . z\n"
    "#pragma unroll\n"
    "    for (int s = 0; s < HD; ++s) {\n"
    "        den = fmaf(qr[s], Zp[half * HD + s], den);\n"
    "        const float* wr = Pl + (half * HD + s) * DO + col;\n"
    "#pragma unroll\n"
    "        for (int b = 0; b < NB; ++b)\n"
    "            acc[b] = __builtin_amdgcn_mfma_f32_32x32x2f32(wr[32 * b], qr[s], acc[b], 0, 0, 0);\n"
    "    }\n"
    "    den += __shfl_xor(den, 32);\n"
    "    const float inv = den != 0.0f ? 1.0f / den : 0.0f;\n"
    "    __syncthreads();\n"
    "    constexpr unsigned SWM = (DO / 4 < 16 ? DO / 4 : 16) - 1;   // swizzle within a staged row\n"
    "    float4* sto = pool + wave * 32u * (DO / 4);\n"
    "#pragma unroll\n"
    "    for (int b = 0; b < NB; ++b)\n"
    "#pragma unroll\n"
    "        for (int r4 = 0; r4 < 4; ++r4) {\n"
    "            unsigned c4 = 8u * b + 2u * r4 + half;\n"
    "            sto[col * (DO / 4) + (c4 ^ (col & SWM))] = make_float4(acc[b][4 * r4] * inv, acc[b][4 * r4 + 1] * inv,\n"
    "                                                                  acc[b][4 * r4 + 2] * inv, acc[b][4 * r4 + 3] * inv);\n"
    "        }\n"
    "    __syncthreads();\n"
    "    const unsigned rbase = c * LA_R + wave * 32u;\n"
    "    if ((d & 3u) == 0u) {\n"
    "        for (unsigned e = lane; e < 32u * (DO / 4); e += 64u) {\n"
    "            unsigned rr = e / (DO / 4), cc = 4u * (e % (DO / 4));\n"
    "            if (rbase + rr < seq && c0 + cc < d)\n"
    "                *(float4*)(O + ((unsigned long long)h * seq + rbase + rr) * d + c0 + cc) = sto[rr * (DO / 4) + ((cc / 4u) ^ (rr & SWM))];\n"
    "        }\n"
    "    } else {\n"
    "        for (unsigned e = lane; e < 32u * DO; e += 64u) {\n"
    "            unsigned rr = e / DO, cc = e % DO;\n"
    "            if (rbase + rr < seq && c0 + cc < d)\n"
    "                O[((unsigned long long)h * seq + rbase + rr) * d + c0 + cc] = ((const float*)&sto[rr * (DO / 4) + ((cc / 4u) ^ (rr & SWM))])[cc & 3u];\n"
    "        }\n"
    "    }\n"
    "}\n"
    "#endif\n"
    "#if SAFEC_MFMA\n"
    "#define FM_ATTR(minb) __launch_bounds__(256, minb)\n"
    "#else\n"
    "#define FM_ATTR(minb) __launch_bounds__(256)\n"
    "#endif\n"
    "extern \"C\" __global__ FM_ATTR(2) void flash_mfma_d64(const float* Q, const float* K, const float* V, float* O,\n"
    "        unsigned seq, unsigned kv, unsigned d, float scale, int causal) {\n"
    "#if SAFEC_MFMA\n"
    "    flash_m32_impl<64>(Q, K, V, O, seq, kv, d, scale, causal);\n"
    "#endif\n"
    "}\n"
    "extern \"C\" __global__ FM_ATTR(2) void flash_mfma_d128(const float* Q, const float* K, const float* V, float* O,\n"
    "        unsigned seq, unsigned kv, unsigned d, float scale, int causal) {\n"
    "#if SAFEC_MFMA\n"
    "    flash_m16_impl<128>(Q, K, V, O, seq, kv, d, scale, causal);\n"
    "#endif\n"
    "}\n"
    "extern \"C\" __global__ FM_ATTR(3) void la_state_d64(const float* K, const float* V, float* W, float* Z,\n"
    "        unsigned kv, unsigned d, unsigned rows, unsigned nc) {\n"
    "#if SAFEC_MFMA\n"
    "    la_state_impl<64>(K, V, W, Z, kv, d, rows, nc);\n"
    "#endif\n"
    "}\n"
    "extern \"C\" __global__ FM_ATTR(2) void la_state_d128(const float* K, const float* V, float* W, float* Z,\n"
    "        unsigned kv, unsigned d, unsigned rows, unsigned nc) {\n"
    "#if SAFEC_MFMA\n"
    "    la_state_impl<128>(K, V, W, Z, kv, d, rows, nc);\n"
    "#endif\n"
    "}\n"
    "extern \"C\" __global__ __launch_bounds__(256) void la_scan_d64(float* W, float* Z, unsigned nc, int causal) {\n"
    "#if SAFEC_MFMA\n"
    "    la_scan_impl<64>(W, Z, nc, causal);\n"
    "#endif\n"
    "}\n"
    "extern \"C\" __global__ __launch_bounds__(256) void la_scan_d128(float* W, float* Z, unsigned nc, int causal) {\n"
    "#if SAFEC_MFMA\n"
    "    la_scan_impl<128>(W, Z, nc, causal);\n"
    "#endif\n"
    "}\n"
    "extern \"C\" __global__ FM_ATTR(2) void la_out_d64(const float* Q, const float* K, const float* V, const float* W,\n"
    "        const float* Z, float* O, unsigned seq, unsigned kv, unsigned d, unsigned nc, int causal) {\n"
    "#if SAFEC_MFMA\n"
    "    la_out_impl<64, 64>(Q, K, V, W, Z, O, seq, kv, d, nc, causal);\n"
    "#endif\n"
    "}\n"
    "extern \"C\" __global__ FM_ATTR(2) void la_out_d128(const float* Q, const float* K, const float* V, const float* W,\n"
    "        const float* Z, float* O, unsigned seq, unsigned kv, unsigned d, unsigned nc, int causal) {\n"
    "#if SAFEC_MFMA\n"
    "    la_out_impl<128, 64>(Q, K, V, W, Z, O, seq, kv, d, nc, causal);\n"
    "#endif\n"
    "}\n"
    "extern \"C\" __global__ FM_ATTR(2) void la_causal_d64(const float* Q, const float* K, const float* V, float* R,\n"
    "        int* F, float* O, unsigned seq, unsigned heads, unsigned d, unsigned nc) {\n"
    "#if SAFEC_MFMA\n"
    "    la_causal_impl<64, 64>(Q, K, V, R, F, O, seq, heads, d, nc);\n"
    "#endif\n"
    "}\n";
}

// Kernel table, in this order.
#define K_ADD 0
#define K_SUB 1
#define K_MUL 2
#define K_DIV 3
#define K_POW 4
#define K_RELU 5
#define K_LOG 6
#define K_EXP 7
#define K_SQRT 8
#define K_SCALE 9
#define K_RELU_GRAD 10
#define K_SGD 11
#define K_SUM 12
#define K_SGEMM 13
#define K_MSE 14
#define K_UPSAMPLE2X 15
#define K_BIAS_ACT 16
#define K_TO_HALF 17
#define K_FROM_HALF 18
#define K_HALF_SGD 19
#define K_CONCAT 20
#define K_UPCAT 21
#define K_HALF_RELU 22
#define K_HALF_RELUG 23
#define K_HALF_MSE 24
#define K_HALF_SGD2 25
#define K_FLASH64 26
#define K_FLASH128 27
#define K_LIN_STATE 28
#define K_LIN_PREFIX 29
#define K_LIN_OUT 30
#define K_MFMA_PROBE 31
#define K_FLASH_M64 32
#define K_FLASH_M128 33
#define K_LA_STATE64 34
#define K_LA_STATE128 35
#define K_LA_SCAN64 36
#define K_LA_SCAN128 37
#define K_LA_OUT64 38
#define K_LA_OUT128 39
#define K_LA_CAUSAL64 40
#define K_COUNT 41

static const char* rocm_kernel_name_(int k) {
    if (k == K_ADD) return "ew_add";
    if (k == K_SUB) return "ew_sub";
    if (k == K_MUL) return "ew_mul";
    if (k == K_DIV) return "ew_div";
    if (k == K_POW) return "ew_pow";
    if (k == K_RELU) return "ew_relu";
    if (k == K_LOG) return "ew_log";
    if (k == K_EXP) return "ew_exp";
    if (k == K_SQRT) return "ew_sqrt";
    if (k == K_SCALE) return "ew_scale";
    if (k == K_RELU_GRAD) return "ew_relu_grad";
    if (k == K_SGD) return "sgd_update";
    if (k == K_SUM) return "reduce_sum";
    if (k == K_SGEMM) return "sgemm_nn";
    if (k == K_UPSAMPLE2X) return "upsample2x";
    if (k == K_BIAS_ACT) return "bias_act";
    if (k == K_TO_HALF) return "to_half";
    if (k == K_FROM_HALF) return "from_half";
    if (k == K_HALF_SGD) return "half_sgd";
    if (k == K_CONCAT) return "concat_rows";
    if (k == K_UPCAT) return "upcat";
    if (k == K_HALF_RELU) return "half_relu";
    if (k == K_HALF_RELUG) return "half_relu_grad";
    if (k == K_HALF_MSE) return "half_mse";
    if (k == K_HALF_SGD2) return "half_sgd2";
    if (k == K_FLASH64) return "flash_attn_d64";
    if (k == K_FLASH128) return "flash_attn_d128";
    if (k == K_LIN_STATE) return "lin_attn_state";
    if (k == K_LIN_PREFIX) return "lin_attn_prefix";
    if (k == K_LIN_OUT) return "lin_attn_out";
    if (k == K_MFMA_PROBE) return "mfma_probe";
    if (k == K_FLASH_M64) return "flash_mfma_d64";
    if (k == K_FLASH_M128) return "flash_mfma_d128";
    if (k == K_LA_STATE64) return "la_state_d64";
    if (k == K_LA_STATE128) return "la_state_d128";
    if (k == K_LA_SCAN64) return "la_scan_d64";
    if (k == K_LA_SCAN128) return "la_scan_d128";
    if (k == K_LA_OUT64) return "la_out_d64";
    if (k == K_LA_OUT128) return "la_out_d128";
    if (k == K_LA_CAUSAL64) return "la_causal_d64";
    return "mse_grad";
}

// ── Backend state (one per process) ──────────────────────────────────────────
static int   g_rocm_state = 0;        // 0 = not tried, 1 = ready, -1 = unavailable
static void* g_rocm_stream = (void*)0;
static void* g_rocm_blas = (void*)0;
static void* g_rocm_fn[K_COUNT];
static char  g_rocm_name[256];
static int   g_rocm_mfma = 0;         // 1 = the MFMA attention kernels run (matrix cores, not switched off)
static int   g_rocm_mfma_has = 0;     // 1 = the GPU has matrix cores (CDNA)

// Initializes everything once: picks the GPU with the most compute units
// (discrete before integrated), creates the stream and rocBLAS handle, and
// compiles + loads the kernels with hiprtc.
static int rocm_ensure_() {
    if (g_rocm_state != 0) return g_rocm_state == 1;
    g_rocm_state = -1;
    unsafe {
        g_rocm_name[0] = (char)0;
        if (hipInit(0U) != ROCM_HIP_OK) return 0;
        int count = 0;
        if (hipGetDeviceCount(&count) != ROCM_HIP_OK || count <= 0) return 0;
        int best = -1;
        long long bestScore = -1LL;
        int d = 0;
        while (d < count) {
            int cu = 0;
            int integrated = 0;
            hipDeviceGetAttribute(&cu, ROCM_ATTR_CU_COUNT, d);
            hipDeviceGetAttribute(&integrated, ROCM_ATTR_INTEGRATED, d);
            long long score = (long long)cu + (integrated != 0 ? 0LL : 1000000LL);
            if (score > bestScore) { bestScore = score; best = d; }
            d = d + 1;
        }
        if (best < 0 || hipSetDevice(best) != ROCM_HIP_OK) return 0;
        hipDeviceGetName(&g_rocm_name[0], 256, best);
        if (hipStreamCreate(&g_rocm_stream) != ROCM_HIP_OK) return 0;
        if (rocblas_create_handle(&g_rocm_blas) != 0) return 0;
        rocblas_set_stream(g_rocm_blas, g_rocm_stream);

        void* prog = (void*)0;
        if (hiprtcCreateProgram(&prog, rocm_kernel_source_(), "safec_rocm.hip", 0,
                                (const char**)0, (const char**)0) != 0) return 0;
        const char* opts[1];
        opts[0] = "-O3";
        if (hiprtcCompileProgram(prog, 1, &opts[0]) != 0) { hiprtcDestroyProgram(&prog); return 0; }
        unsigned long codeSize = 0UL;
        hiprtcGetCodeSize(prog, &codeSize);
        char* code = (char*)alloc_raw_(codeSize);
        if (code == (char*)0) { hiprtcDestroyProgram(&prog); return 0; }
        hiprtcGetCode(prog, code);
        hiprtcDestroyProgram(&prog);
        void* module = (void*)0;
        int loaded = hipModuleLoadData(&module, (const void*)code);
        dealloc_raw_((void*)code);
        if (loaded != ROCM_HIP_OK) return 0;
        int k = 0;
        while (k < K_COUNT) {
            if (hipModuleGetFunction(&g_rocm_fn[k], module, rocm_kernel_name_(k)) != ROCM_HIP_OK) return 0;
            k = k + 1;
        }
        // Ask the compiled code whether this GPU has matrix cores (the kernel
        // source knows the target; the host API doesn't say).
        void* flag = (void*)0;
        if (hipMalloc(&flag, 4UL) == ROCM_HIP_OK) {
            void* pp[1];
            pp[0] = (void*)&flag;
            float has = 0.0f;
            if (hipModuleLaunchKernel(g_rocm_fn[K_MFMA_PROBE], 1U, 1U, 1U, 64U, 1U, 1U, 0U, g_rocm_stream,
                                      &pp[0], (void**)0) == ROCM_HIP_OK &&
                hipMemcpyAsync((void*)&has, (const void*)flag, 4UL, ROCM_D2H, g_rocm_stream) == ROCM_HIP_OK &&
                hipStreamSynchronize(g_rocm_stream) == ROCM_HIP_OK)
                g_rocm_mfma_has = has == 1.0f ? 1 : 0;
            g_rocm_mfma = g_rocm_mfma_has;
            hipFree(flag);
        }
    }
    g_rocm_state = 1;
    return 1;
}

int rocm_available() { return rocm_ensure_(); }

int rocm_attention_use_mfma(int on) {
    if (!rocm_ensure_()) return 0;
    g_rocm_mfma = on && g_rocm_mfma_has ? 1 : 0;
    return g_rocm_mfma;
}

const &char rocm_device_name() {
    rocm_ensure_();
    unsafe { return (const &char)&g_rocm_name[0]; }
}

// ── Buffers ──────────────────────────────────────────────────────────────────

struct RocmBuf rocm_alloc(unsigned long n) {
    struct RocmBuf b;
    b.dev = 0UL;
    b.len = 0UL;
    if (n == 0UL || !rocm_ensure_()) return b;
    unsafe {
        void* p = (void*)0;
        if (hipMalloc(&p, checked_mul_size(n, sizeof(float))) != ROCM_HIP_OK) return b;
        b.dev = (unsigned long)p;
    }
    b.len = n;
    return b;
}

int rocm_write(struct RocmBuf dst, const &float host, unsigned long n) {
    if (dst.dev == 0UL || n > dst.len || !rocm_ensure_()) return 0;
    unsafe {
        return hipMemcpyAsync((void*)dst.dev, (const void*)host, n * sizeof(float),
                              ROCM_H2D, g_rocm_stream) == ROCM_HIP_OK;
    }
}

struct RocmBuf rocm_upload(const &float host, unsigned long n) {
    struct RocmBuf b = rocm_alloc(n);
    if (b.dev != 0UL && !rocm_write(b, host, n)) rocm_free(&b);
    return b;
}

int rocm_read(struct RocmBuf src, &float host, unsigned long n) {
    if (src.dev == 0UL || n > src.len || !rocm_ensure_()) return 0;
    unsafe {
        if (hipMemcpyAsync((void*)host, (const void*)src.dev, n * sizeof(float),
                           ROCM_D2H, g_rocm_stream) != ROCM_HIP_OK) return 0;
        return hipStreamSynchronize(g_rocm_stream) == ROCM_HIP_OK;
    }
}

void rocm_free(&RocmBuf b) {
    if (b.dev != 0UL) { unsafe { hipFree((void*)b.dev); } }
    b.dev = 0UL;
    b.len = 0UL;
}

int rocm_sync() {
    if (!rocm_ensure_()) return 0;
    unsafe { return hipStreamSynchronize(g_rocm_stream) == ROCM_HIP_OK; }
}

?&void rocm_stream_handle() {
    if (!rocm_ensure_()) return null;
    unsafe { return (?&void)g_rocm_stream; }
}

int rocm_copy(struct RocmBuf src, unsigned long srcOff, struct RocmBuf dst, unsigned long dstOff, unsigned long n) {
    if (!rocm_ensure_() || src.dev == 0UL || dst.dev == 0UL || n == 0UL ||
        srcOff + n > src.len || dstOff + n > dst.len) return 0;
    unsafe {
        return hipMemcpyAsync((void*)(dst.dev + dstOff * sizeof(float)),
                              (const void*)(src.dev + srcOff * sizeof(float)),
                              n * sizeof(float), ROCM_D2D, g_rocm_stream) == ROCM_HIP_OK;
    }
}

// ── Launch helpers ───────────────────────────────────────────────────────────

static int rocm_launch_(int kernel, unsigned long n, &void params) {
    unsigned long blocks = (n + (unsigned long)ROCM_BLOCK - 1UL) / (unsigned long)ROCM_BLOCK;
    if (blocks > ROCM_MAX_BLOCKS) blocks = ROCM_MAX_BLOCKS;
    if (blocks == 0UL) blocks = 1UL;
    unsafe {
        return hipModuleLaunchKernel(g_rocm_fn[kernel], (unsigned int)blocks, 1U, 1U,
                                     ROCM_BLOCK, 1U, 1U, 0U, g_rocm_stream,
                                     (void**)params, (void**)0) == ROCM_HIP_OK;
    }
}

static int rocm_fits_(struct RocmBuf b, unsigned long n) { return b.dev != 0UL && b.len >= n; }

static int rocm_binary_(int kernel, struct RocmBuf a, struct RocmBuf b, struct RocmBuf out) {
    unsigned long n = out.len;
    if (!rocm_ensure_() || !rocm_fits_(a, n) || !rocm_fits_(b, n) || !rocm_fits_(out, n) ||
        n > 4294967295UL) return 0;
    unsafe {
        void* pa = (void*)a.dev;
        void* pb = (void*)b.dev;
        void* po = (void*)out.dev;
        unsigned int nn = (unsigned int)n;
        void* params[4];
        params[0] = (void*)&pa; params[1] = (void*)&pb; params[2] = (void*)&po; params[3] = (void*)&nn;
        return rocm_launch_(kernel, n, (&void)&params[0]);
    }
}

static int rocm_unary_(int kernel, struct RocmBuf a, float k, struct RocmBuf out) {
    unsigned long n = out.len;
    if (!rocm_ensure_() || !rocm_fits_(a, n) || !rocm_fits_(out, n) || n > 4294967295UL) return 0;
    unsafe {
        void* pa = (void*)a.dev;
        void* po = (void*)out.dev;
        unsigned int nn = (unsigned int)n;
        float kk = k;
        void* params[4];
        params[0] = (void*)&pa; params[1] = (void*)&po; params[2] = (void*)&nn; params[3] = (void*)&kk;
        return rocm_launch_(kernel, n, (&void)&params[0]);
    }
}

// ── Queued ops ───────────────────────────────────────────────────────────────

// ── gemm_ex solution choice ──────────────────────────────────────────────────
// rocBLAS's default solution for a gemm_ex shape can be several times slower
// than its best on some GPUs (on an MI100: 3.5x for a 128x256x1024 fp16
// product, 2-3x for small bf16 ones). So the first call for each (types,
// transposes, size) outside graph capture times every solution rocBLAS
// offers on that call's own buffers, like MIOpen's Find, and keeps the
// fastest (0 = the default, if nothing beats it). During capture an untuned
// shape uses the default. The cache holds ROCM_GEMM_PLANS shapes.
#define ROCM_GEMM_PLANS     64
#define ROCM_GEMM_MAX_SOLS  512
#define ROCM_GEMM_REPS      8

static int g_rocm_capturing = 0;
static int g_gplan_used[ROCM_GEMM_PLANS];
static unsigned long g_gplan_key[ROCM_GEMM_PLANS];   // packed types, transposes
static int g_gplan_m[ROCM_GEMM_PLANS];
static int g_gplan_n[ROCM_GEMM_PLANS];
static int g_gplan_k[ROCM_GEMM_PLANS];
static int g_gplan_sol[ROCM_GEMM_PLANS];
static int g_gplan_next = 0;
static int g_gplan_list[ROCM_GEMM_MAX_SOLS];

// One column-major rocBLAS gemm_ex call, D = C = op(A) op(B); 'sol' 0 uses
// the default solution.
static int rocm_gemm_ex_call_(int opA, int opB, int m, int n, int k,
                              unsigned long a, int aType, int lda, unsigned long b, int ldb,
                              unsigned long c, int cType, int ldc, int sol) {
    float one = 1.0f;
    float zero = 0.0f;
    unsafe {
        return rocblas_gemm_ex(g_rocm_blas, opA, opB, m, n, k, (const void*)&one,
                               (const void*)a, aType, lda, (const void*)b, aType, ldb,
                               (const void*)&zero, (const void*)c, cType, ldc, (void*)c, cType, ldc,
                               ROCM_BLAS_F32, sol == 0 ? ROCM_BLAS_ALGO_STANDARD : ROCM_BLAS_ALGO_INDEX,
                               sol, 0U) == 0;
    }
}

static double rocm_gemm_time_(int opA, int opB, int m, int n, int k,
                              unsigned long a, int aType, int lda, unsigned long b, int ldb,
                              unsigned long c, int cType, int ldc, int sol) {
    if (!rocm_gemm_ex_call_(opA, opB, m, n, k, a, aType, lda, b, ldb, c, cType, ldc, sol)) return -1.0;
    unsafe { hipStreamSynchronize(g_rocm_stream); }
    long long t0 = time_mono_ns();
    int r = 0;
    while (r < ROCM_GEMM_REPS) {
        rocm_gemm_ex_call_(opA, opB, m, n, k, a, aType, lda, b, ldb, c, cType, ldc, sol);
        r = r + 1;
    }
    unsafe { hipStreamSynchronize(g_rocm_stream); }
    return (double)(time_mono_ns() - t0);
}

static int rocm_gemm_ex_(int opA, int opB, int m, int n, int k,
                         unsigned long a, int aType, int lda, unsigned long b, int ldb,
                         unsigned long c, int cType, int ldc) {
    unsigned long key = (unsigned long)aType | ((unsigned long)cType << 16) |
                        ((unsigned long)opA << 32) | ((unsigned long)opB << 48);
    int i = 0;
    while (i < ROCM_GEMM_PLANS) {
        if (g_gplan_used[i] && g_gplan_key[i] == key && g_gplan_m[i] == m && g_gplan_n[i] == n &&
            g_gplan_k[i] == k)
            return rocm_gemm_ex_call_(opA, opB, m, n, k, a, aType, lda, b, ldb, c, cType, ldc, g_gplan_sol[i]);
        i = i + 1;
    }
    if (g_rocm_capturing) return rocm_gemm_ex_call_(opA, opB, m, n, k, a, aType, lda, b, ldb, c, cType, ldc, 0);

    int best = 0;
    double bestT = rocm_gemm_time_(opA, opB, m, n, k, a, aType, lda, b, ldb, c, cType, ldc, 0);
    if (bestT < 0.0) return 0;
    int count = ROCM_GEMM_MAX_SOLS;
    float one = 1.0f;
    float zero = 0.0f;
    unsafe {
        if (rocblas_gemm_ex_get_solutions(g_rocm_blas, opA, opB, m, n, k, (const void*)&one,
                                          (const void*)a, aType, lda, (const void*)b, aType, ldb,
                                          (const void*)&zero, (const void*)c, cType, ldc, (void*)c, cType, ldc,
                                          ROCM_BLAS_F32, ROCM_BLAS_ALGO_INDEX, 0U,
                                          &g_gplan_list[0], &count) != 0) count = 0;
    }
    if (count > ROCM_GEMM_MAX_SOLS) count = ROCM_GEMM_MAX_SOLS;
    int s = 0;
    while (s < count) {
        int sol = g_gplan_list[s];
        if (sol != 0) {
            double t = rocm_gemm_time_(opA, opB, m, n, k, a, aType, lda, b, ldb, c, cType, ldc, sol);
            // Prefer the default unless a solution is clearly faster.
            if (t >= 0.0 && t < bestT * 0.95) { bestT = t; best = sol; }
        }
        s = s + 1;
    }
    int slot = -1;
    i = 0;
    while (i < ROCM_GEMM_PLANS && slot < 0) { if (!g_gplan_used[i]) slot = i; i = i + 1; }
    if (slot < 0) { slot = g_gplan_next; g_gplan_next = (g_gplan_next + 1) % ROCM_GEMM_PLANS; }
    g_gplan_used[slot] = 1;
    g_gplan_key[slot] = key;
    g_gplan_m[slot] = m;
    g_gplan_n[slot] = n;
    g_gplan_k[slot] = k;
    g_gplan_sol[slot] = best;
    // The timing runs already computed C; run the chosen solution once more
    // so the result comes from the solution later calls will use.
    return rocm_gemm_ex_call_(opA, opB, m, n, k, a, aType, lda, b, ldb, c, cType, ldc, best);
}

int rocm_gemm(struct RocmBuf a, int ta, struct RocmBuf b, int tb, struct RocmBuf c,
              unsigned long m, unsigned long n, unsigned long k) {
    if (!rocm_ensure_() || !rocm_fits_(a, m * k) || !rocm_fits_(b, k * n) || !rocm_fits_(c, m * n))
        return 0;
    // Row-major C = op(A) op(B) is column-major C^T = op(B)^T op(A)^T, so
    // hand rocBLAS the operands swapped; each operand's leading dimension
    // is its stored row length.
    return rocm_gemm_ex_(tb ? ROCM_BLAS_T : ROCM_BLAS_N, ta ? ROCM_BLAS_T : ROCM_BLAS_N,
                         (int)n, (int)m, (int)k,
                         b.dev, ROCM_BLAS_F32, tb ? (int)k : (int)n,
                         a.dev, ta ? (int)m : (int)k,
                         c.dev, ROCM_BLAS_F32, (int)n);
}

int rocm_add(struct RocmBuf a, struct RocmBuf b, struct RocmBuf out) { return rocm_binary_(K_ADD, a, b, out); }
int rocm_sub(struct RocmBuf a, struct RocmBuf b, struct RocmBuf out) { return rocm_binary_(K_SUB, a, b, out); }
int rocm_mul(struct RocmBuf a, struct RocmBuf b, struct RocmBuf out) { return rocm_binary_(K_MUL, a, b, out); }
int rocm_div(struct RocmBuf a, struct RocmBuf b, struct RocmBuf out) { return rocm_binary_(K_DIV, a, b, out); }
int rocm_pow(struct RocmBuf a, struct RocmBuf b, struct RocmBuf out) { return rocm_binary_(K_POW, a, b, out); }
int rocm_relu_grad(struct RocmBuf z, struct RocmBuf g, struct RocmBuf out) { return rocm_binary_(K_RELU_GRAD, z, g, out); }
int rocm_scale(struct RocmBuf a, float k, struct RocmBuf out) { return rocm_unary_(K_SCALE, a, k, out); }
int rocm_relu(struct RocmBuf a, struct RocmBuf out) { return rocm_unary_(K_RELU, a, 0.0f, out); }
int rocm_exp(struct RocmBuf a, struct RocmBuf out) { return rocm_unary_(K_EXP, a, 0.0f, out); }
int rocm_log(struct RocmBuf a, struct RocmBuf out) { return rocm_unary_(K_LOG, a, 0.0f, out); }
int rocm_sqrt(struct RocmBuf a, struct RocmBuf out) { return rocm_unary_(K_SQRT, a, 0.0f, out); }

// ── fp16 / bf16 ─────────────────────────────────────────────────────────────

static int rocm_half_dtype_ok_(int dtype) { return dtype == ROCM_FP16 || dtype == ROCM_BF16; }
static int rocm_half_fits_(struct RocmHalfBuf b, unsigned long n) {
    return b.dev != 0UL && b.len >= n && rocm_half_dtype_ok_(b.dtype);
}

struct RocmHalfBuf rocm_half_alloc(unsigned long n, int dtype) {
    struct RocmHalfBuf b;
    b.dev = 0UL;
    b.len = 0UL;
    b.dtype = dtype;
    if (n == 0UL || !rocm_half_dtype_ok_(dtype) || !rocm_ensure_()) return b;
    unsafe {
        void* p = (void*)0;
        if (hipMalloc(&p, checked_mul_size(n, 2UL)) != ROCM_HIP_OK) return b;
        b.dev = (unsigned long)p;
    }
    b.len = n;
    return b;
}

void rocm_half_free(&RocmHalfBuf b) {
    if (b.dev != 0UL) { unsafe { hipFree((void*)b.dev); } }
    b.dev = 0UL;
    b.len = 0UL;
}

int rocm_half_write(struct RocmHalfBuf dst, const &unsigned short bits, unsigned long n) {
    if (dst.dev == 0UL || n > dst.len || !rocm_ensure_()) return 0;
    unsafe {
        return hipMemcpyAsync((void*)dst.dev, (const void*)bits, n * 2UL, ROCM_H2D, g_rocm_stream) == ROCM_HIP_OK;
    }
}

int rocm_half_read(struct RocmHalfBuf src, &unsigned short bits, unsigned long n) {
    if (src.dev == 0UL || n > src.len || !rocm_ensure_()) return 0;
    unsafe {
        if (hipMemcpyAsync((void*)bits, (const void*)src.dev, n * 2UL, ROCM_D2H, g_rocm_stream) != ROCM_HIP_OK)
            return 0;
        return hipStreamSynchronize(g_rocm_stream) == ROCM_HIP_OK;
    }
}

static int rocm_convert_(int kernel, unsigned long a, unsigned long o, unsigned long n, int bf) {
    if (n == 0UL || n > 4294967295UL) return 0;
    unsafe {
        void* pa = (void*)a;
        void* po = (void*)o;
        unsigned int nn = (unsigned int)n;
        int b = bf;
        void* params[4];
        params[0] = (void*)&pa; params[1] = (void*)&po; params[2] = (void*)&nn; params[3] = (void*)&b;
        return rocm_launch_(kernel, n, (&void)&params[0]);
    }
}

int rocm_to_half(struct RocmBuf src, struct RocmHalfBuf dst) {
    unsigned long n = dst.len;
    if (!rocm_ensure_() || !rocm_fits_(src, n) || !rocm_half_fits_(dst, n)) return 0;
    return rocm_convert_(K_TO_HALF, src.dev, dst.dev, n, dst.dtype == ROCM_BF16);
}

int rocm_from_half(struct RocmHalfBuf src, struct RocmBuf dst) {
    unsigned long n = dst.len;
    if (!rocm_ensure_() || !rocm_half_fits_(src, n) || !rocm_fits_(dst, n)) return 0;
    return rocm_convert_(K_FROM_HALF, src.dev, dst.dev, n, src.dtype == ROCM_BF16);
}

struct RocmHalfBuf rocm_half_upload(const &float host, unsigned long n, int dtype) {
    struct RocmHalfBuf h = rocm_half_alloc(n, dtype);
    if (h.dev == 0UL) return h;
    struct RocmBuf tmp = rocm_upload(host, n);
    int ok = tmp.dev != 0UL && rocm_to_half(tmp, h) && rocm_sync();
    rocm_free(&tmp);
    if (!ok) rocm_half_free(&h);
    return h;
}

// rocBLAS gemm_ex: 16-bit A and B, fp32 accumulation; C/D are 16-bit
// (outType = the inputs' type) or fp32. Row-major via the same operand swap
// as rocm_gemm.
static int rocm_half_gemm_(struct RocmHalfBuf a, int ta, struct RocmHalfBuf b, int tb,
                           unsigned long cdev, int outType, unsigned long m, unsigned long n, unsigned long k) {
    int t = a.dtype == ROCM_BF16 ? ROCM_BLAS_BF16 : ROCM_BLAS_F16;
    return rocm_gemm_ex_(tb ? ROCM_BLAS_T : ROCM_BLAS_N, ta ? ROCM_BLAS_T : ROCM_BLAS_N,
                         (int)n, (int)m, (int)k,
                         b.dev, t, tb ? (int)k : (int)n,
                         a.dev, ta ? (int)m : (int)k,
                         cdev, outType == 0 ? t : outType, (int)n);
}

int rocm_half_gemm(struct RocmHalfBuf a, int ta, struct RocmHalfBuf b, int tb, struct RocmHalfBuf c,
                   unsigned long m, unsigned long n, unsigned long k) {
    if (!rocm_ensure_() || !rocm_half_fits_(a, m * k) || !rocm_half_fits_(b, k * n) ||
        !rocm_half_fits_(c, m * n) || a.dtype != b.dtype || a.dtype != c.dtype) return 0;
    return rocm_half_gemm_(a, ta, b, tb, c.dev, 0, m, n, k);
}

int rocm_half_gemm_f32(struct RocmHalfBuf a, int ta, struct RocmHalfBuf b, int tb, struct RocmBuf c,
                       unsigned long m, unsigned long n, unsigned long k) {
    if (!rocm_ensure_() || !rocm_half_fits_(a, m * k) || !rocm_half_fits_(b, k * n) ||
        !rocm_fits_(c, m * n) || a.dtype != b.dtype) return 0;
    return rocm_half_gemm_(a, ta, b, tb, c.dev, ROCM_BLAS_F32, m, n, k);
}

int rocm_half_sgd(struct RocmHalfBuf w, struct RocmHalfBuf g, float lr) {
    unsigned long n = w.len;
    if (!rocm_ensure_() || !rocm_half_fits_(w, n) || !rocm_half_fits_(g, n) || w.dtype != g.dtype ||
        n > 4294967295UL) return 0;
    unsafe {
        void* pw = (void*)w.dev;
        void* pg = (void*)g.dev;
        float l = lr;
        unsigned int nn = (unsigned int)n;
        int bf = w.dtype == ROCM_BF16 ? 1 : 0;
        void* params[5];
        params[0] = (void*)&pw; params[1] = (void*)&pg; params[2] = (void*)&l; params[3] = (void*)&nn;
        params[4] = (void*)&bf;
        return rocm_launch_(K_HALF_SGD, n, (&void)&params[0]);
    }
}

int rocm_half_relu(struct RocmHalfBuf a, struct RocmHalfBuf out) {
    unsigned long n = a.len;
    if (!rocm_ensure_() || !rocm_half_fits_(a, n) || !rocm_half_fits_(out, n) || a.dtype != out.dtype ||
        n > 4294967295UL) return 0;
    unsafe {
        void* pa = (void*)a.dev;
        void* po = (void*)out.dev;
        unsigned int nn = (unsigned int)n;
        int bf = a.dtype == ROCM_BF16 ? 1 : 0;
        void* params[4];
        params[0] = (void*)&pa; params[1] = (void*)&po; params[2] = (void*)&nn; params[3] = (void*)&bf;
        return rocm_launch_(K_HALF_RELU, n, (&void)&params[0]);
    }
}

// rocBLAS has no ReLU epilogue: the GEMM, then ReLU in place.
int rocm_half_gemm_relu(struct RocmHalfBuf a, int ta, struct RocmHalfBuf b, int tb, struct RocmHalfBuf c,
                        unsigned long m, unsigned long n, unsigned long k) {
    if (!rocm_half_gemm(a, ta, b, tb, c, m, n, k)) return 0;
    struct RocmHalfBuf view = c;
    view.len = m * n;
    return rocm_half_relu(view, view);
}

int rocm_half_relu_grad(struct RocmHalfBuf z, struct RocmHalfBuf g, struct RocmHalfBuf out) {
    unsigned long n = z.len;
    if (!rocm_ensure_() || !rocm_half_fits_(z, n) || !rocm_half_fits_(g, n) || !rocm_half_fits_(out, n) ||
        z.dtype != g.dtype || z.dtype != out.dtype || n > 4294967295UL) return 0;
    unsafe {
        void* pz = (void*)z.dev;
        void* pg = (void*)g.dev;
        void* po = (void*)out.dev;
        unsigned int nn = (unsigned int)n;
        int bf = z.dtype == ROCM_BF16 ? 1 : 0;
        void* params[5];
        params[0] = (void*)&pz; params[1] = (void*)&pg; params[2] = (void*)&po; params[3] = (void*)&nn;
        params[4] = (void*)&bf;
        return rocm_launch_(K_HALF_RELUG, n, (&void)&params[0]);
    }
}

int rocm_half_mse_grad(struct RocmHalfBuf y, struct RocmHalfBuf t, struct RocmHalfBuf dy, struct RocmBuf loss1) {
    unsigned long n = y.len;
    if (!rocm_ensure_() || !rocm_half_fits_(y, n) || !rocm_half_fits_(t, n) || !rocm_half_fits_(dy, n) ||
        y.dtype != t.dtype || y.dtype != dy.dtype || !rocm_fits_(loss1, 1UL) || n > 4294967295UL) return 0;
    unsafe {
        if (hipMemsetAsync((void*)loss1.dev, 0, sizeof(float), g_rocm_stream) != ROCM_HIP_OK) return 0;
        void* py = (void*)y.dev;
        void* pt = (void*)t.dev;
        void* pd = (void*)dy.dev;
        void* pl = (void*)loss1.dev;
        unsigned int nn = (unsigned int)n;
        int bf = y.dtype == ROCM_BF16 ? 1 : 0;
        void* params[6];
        params[0] = (void*)&py; params[1] = (void*)&pt; params[2] = (void*)&pd;
        params[3] = (void*)&pl; params[4] = (void*)&nn; params[5] = (void*)&bf;
        return rocm_launch_(K_HALF_MSE, n, (&void)&params[0]);
    }
}

int rocm_half_sgd2(struct RocmHalfBuf w1, struct RocmHalfBuf g1, struct RocmHalfBuf w2, struct RocmHalfBuf g2,
                   float lr) {
    unsigned long n1 = w1.len;
    unsigned long n2 = w2.len;
    if (!rocm_ensure_() || !rocm_half_fits_(w1, n1) || !rocm_half_fits_(g1, n1) || !rocm_half_fits_(w2, n2) ||
        !rocm_half_fits_(g2, n2) || w1.dtype != g1.dtype || w1.dtype != w2.dtype || w1.dtype != g2.dtype ||
        n1 + n2 > 4294967295UL) return 0;
    unsafe {
        void* pw1 = (void*)w1.dev;
        void* pg1 = (void*)g1.dev;
        void* pw2 = (void*)w2.dev;
        void* pg2 = (void*)g2.dev;
        unsigned int nn1 = (unsigned int)n1;
        unsigned int nn2 = (unsigned int)n2;
        float rate = lr;
        int bf = w1.dtype == ROCM_BF16 ? 1 : 0;
        void* params[8];
        params[0] = (void*)&pw1; params[1] = (void*)&pg1; params[2] = (void*)&nn1;
        params[3] = (void*)&pw2; params[4] = (void*)&pg2; params[5] = (void*)&nn2;
        params[6] = (void*)&rate; params[7] = (void*)&bf;
        return rocm_launch_(K_HALF_SGD2, n1 + n2, (&void)&params[0]);
    }
}

int rocm_concat(struct RocmBuf a, unsigned long na, struct RocmBuf c, unsigned long nc,
                struct RocmBuf out, unsigned long batch) {
    unsigned long total = batch * (na + nc);
    if (!rocm_ensure_() || batch == 0UL || na == 0UL || nc == 0UL || total > 4294967295UL ||
        !rocm_fits_(a, batch * na) || !rocm_fits_(c, batch * nc) || !rocm_fits_(out, total)) return 0;
    unsafe {
        void* pa = (void*)a.dev;
        void* pc = (void*)c.dev;
        void* po = (void*)out.dev;
        unsigned int nb = (unsigned int)batch;
        unsigned int n1 = (unsigned int)na;
        unsigned int n2 = (unsigned int)nc;
        void* params[6];
        params[0] = (void*)&pa; params[1] = (void*)&pc; params[2] = (void*)&po;
        params[3] = (void*)&nb; params[4] = (void*)&n1; params[5] = (void*)&n2;
        return rocm_launch_(K_CONCAT, total, (&void)&params[0]);
    }
}

int rocm_upsample2x_concat(struct RocmBuf x, unsigned long xChannels, struct RocmBuf skip,
                           unsigned long skipChannels, struct RocmBuf out,
                           unsigned long batch, unsigned long h, unsigned long w) {
    unsigned long plane = 4UL * h * w;
    unsigned long total = batch * (xChannels + skipChannels) * plane;
    if (!rocm_ensure_() || total == 0UL || total > 4294967295UL ||
        !rocm_fits_(x, batch * xChannels * h * w) || !rocm_fits_(skip, batch * skipChannels * plane) ||
        !rocm_fits_(out, total)) return 0;
    unsafe {
        void* px = (void*)x.dev;
        void* ps = (void*)skip.dev;
        void* po = (void*)out.dev;
        unsigned int nb = (unsigned int)batch;
        unsigned int ncx = (unsigned int)xChannels;
        unsigned int ncs = (unsigned int)skipChannels;
        unsigned int nh = (unsigned int)h;
        unsigned int nw = (unsigned int)w;
        void* params[8];
        params[0] = (void*)&px; params[1] = (void*)&ps; params[2] = (void*)&po; params[3] = (void*)&nb;
        params[4] = (void*)&ncx; params[5] = (void*)&ncs; params[6] = (void*)&nh; params[7] = (void*)&nw;
        return rocm_launch_(K_UPCAT, total, (&void)&params[0]);
    }
}

int rocm_upsample2x_nearest(struct RocmBuf in, struct RocmBuf out,
                            unsigned long planes, unsigned long h, unsigned long w) {
    unsigned long total = planes * 4UL * h * w;
    if (!rocm_ensure_() || total == 0UL || total > 4294967295UL ||
        !rocm_fits_(in, planes * h * w) || !rocm_fits_(out, total)) return 0;
    unsafe {
        void* px = (void*)in.dev;
        void* py = (void*)out.dev;
        unsigned int np = (unsigned int)planes;
        unsigned int nh = (unsigned int)h;
        unsigned int nw = (unsigned int)w;
        void* params[5];
        params[0] = (void*)&px; params[1] = (void*)&py; params[2] = (void*)&np;
        params[3] = (void*)&nh; params[4] = (void*)&nw;
        return rocm_launch_(K_UPSAMPLE2X, total, (&void)&params[0]);
    }
}

int rocm_bias_channels(struct RocmBuf y, struct RocmBuf bias, unsigned long batch, unsigned long channels,
                       unsigned long plane, int relu) {
    unsigned long total = batch * channels * plane;
    if (!rocm_ensure_() || total == 0UL || total > 4294967295UL ||
        !rocm_fits_(y, total) || !rocm_fits_(bias, channels)) return 0;
    unsafe {
        void* py = (void*)y.dev;
        void* pb = (void*)bias.dev;
        unsigned int nt = (unsigned int)total;
        unsigned int npl = (unsigned int)plane;
        unsigned int nc = (unsigned int)channels;
        int r = relu ? 1 : 0;
        void* params[6];
        params[0] = (void*)&py; params[1] = (void*)&pb; params[2] = (void*)&nt;
        params[3] = (void*)&npl; params[4] = (void*)&nc; params[5] = (void*)&r;
        return rocm_launch_(K_BIAS_ACT, total, (&void)&params[0]);
    }
}

int rocm_sgd(struct RocmBuf w, struct RocmBuf g, float lr) {
    unsigned long n = w.len;
    if (!rocm_ensure_() || !rocm_fits_(w, n) || !rocm_fits_(g, n) || n > 4294967295UL) return 0;
    unsafe {
        void* pw = (void*)w.dev;
        void* pg = (void*)g.dev;
        float rate = lr;
        unsigned int nn = (unsigned int)n;
        void* params[4];
        params[0] = (void*)&pw; params[1] = (void*)&pg; params[2] = (void*)&rate; params[3] = (void*)&nn;
        return rocm_launch_(K_SGD, n, (&void)&params[0]);
    }
}

int rocm_sum(struct RocmBuf a, struct RocmBuf out1) {
    unsigned long n = a.len;
    if (!rocm_ensure_() || !rocm_fits_(a, n) || !rocm_fits_(out1, 1UL) || n > 4294967295UL) return 0;
    unsafe {
        if (hipMemsetAsync((void*)out1.dev, 0, sizeof(float), g_rocm_stream) != ROCM_HIP_OK) return 0;
        void* pa = (void*)a.dev;
        void* po = (void*)out1.dev;
        unsigned int nn = (unsigned int)n;
        void* params[3];
        params[0] = (void*)&pa; params[1] = (void*)&po; params[2] = (void*)&nn;
        return rocm_launch_(K_SUM, n, (&void)&params[0]);
    }
}

int rocm_mse_grad(struct RocmBuf y, struct RocmBuf t, struct RocmBuf dy, struct RocmBuf loss1) {
    unsigned long n = y.len;
    if (!rocm_ensure_() || !rocm_fits_(t, n) || !rocm_fits_(dy, n) || !rocm_fits_(loss1, 1UL) ||
        n > 4294967295UL) return 0;
    unsafe {
        if (hipMemsetAsync((void*)loss1.dev, 0, sizeof(float), g_rocm_stream) != ROCM_HIP_OK) return 0;
        void* py = (void*)y.dev;
        void* pt = (void*)t.dev;
        void* pd = (void*)dy.dev;
        void* pl = (void*)loss1.dev;
        unsigned int nn = (unsigned int)n;
        void* params[5];
        params[0] = (void*)&py; params[1] = (void*)&pt; params[2] = (void*)&pd;
        params[3] = (void*)&pl; params[4] = (void*)&nn;
        return rocm_launch_(K_MSE, n, (&void)&params[0]);
    }
}

// ── Graphs ───────────────────────────────────────────────────────────────────

int rocm_capture_begin() {
    if (!rocm_ensure_()) return 0;
    unsafe {
        if (hipStreamBeginCapture(g_rocm_stream, ROCM_CAPTURE_GLOBAL) != ROCM_HIP_OK) return 0;
    }
    g_rocm_capturing = 1;
    return 1;
}

struct RocmGraph rocm_capture_end() {
    struct RocmGraph g;
    g.exec = 0UL;
    if (!rocm_ensure_()) return g;
    unsafe {
        void* graph = (void*)0;
        g_rocm_capturing = 0;
        if (hipStreamEndCapture(g_rocm_stream, &graph) != ROCM_HIP_OK) return g;
        void* exec = (void*)0;
        int ok = hipGraphInstantiate(&exec, graph, (void*)0, (char*)0, 0UL) == ROCM_HIP_OK;
        hipGraphDestroy(graph);
        if (ok) g.exec = (unsigned long)exec;
    }
    return g;
}

int rocm_graph_launch(struct RocmGraph g) {
    if (g.exec == 0UL || !rocm_ensure_()) return 0;
    unsafe { return hipGraphLaunch((void*)g.exec, g_rocm_stream) == ROCM_HIP_OK; }
}

void rocm_graph_free(&RocmGraph g) {
    if (g.exec != 0UL) { unsafe { hipGraphExecDestroy((void*)g.exec); } }
    g.exec = 0UL;
}

// ── Host-array convenience ops ───────────────────────────────────────────────

static int rocm_host_binary_(int kernel, const &float a, const &float b, &float out, unsigned long n) {
    struct RocmBuf da = rocm_upload(a, n);
    struct RocmBuf db = rocm_upload(b, n);
    struct RocmBuf dout = rocm_alloc(n);
    int ok = da.dev != 0UL && db.dev != 0UL && dout.dev != 0UL &&
             rocm_binary_(kernel, da, db, dout) && rocm_read(dout, out, n);
    rocm_free(&da);
    rocm_free(&db);
    rocm_free(&dout);
    return ok;
}

static int rocm_host_unary_(int kernel, const &float a, float k, &float out, unsigned long n) {
    struct RocmBuf da = rocm_upload(a, n);
    struct RocmBuf dout = rocm_alloc(n);
    int ok = da.dev != 0UL && dout.dev != 0UL &&
             rocm_unary_(kernel, da, k, dout) && rocm_read(dout, out, n);
    rocm_free(&da);
    rocm_free(&dout);
    return ok;
}

int rocm_add_f32(const &float a, const &float b, &float out, unsigned long n) { return rocm_host_binary_(K_ADD, a, b, out, n); }
int rocm_sub_f32(const &float a, const &float b, &float out, unsigned long n) { return rocm_host_binary_(K_SUB, a, b, out, n); }
int rocm_mul_f32(const &float a, const &float b, &float out, unsigned long n) { return rocm_host_binary_(K_MUL, a, b, out, n); }
int rocm_div_f32(const &float a, const &float b, &float out, unsigned long n) { return rocm_host_binary_(K_DIV, a, b, out, n); }
int rocm_pow_f32(const &float a, const &float b, &float out, unsigned long n) { return rocm_host_binary_(K_POW, a, b, out, n); }
int rocm_relu_f32(const &float a, &float out, unsigned long n) { return rocm_host_unary_(K_RELU, a, 0.0f, out, n); }
int rocm_log_f32(const &float a, &float out, unsigned long n) { return rocm_host_unary_(K_LOG, a, 0.0f, out, n); }
int rocm_exp_f32(const &float a, &float out, unsigned long n) { return rocm_host_unary_(K_EXP, a, 0.0f, out, n); }
int rocm_sqrt_f32(const &float a, &float out, unsigned long n) { return rocm_host_unary_(K_SQRT, a, 0.0f, out, n); }
int rocm_scale_f32(const &float a, float k, &float out, unsigned long n) { return rocm_host_unary_(K_SCALE, a, k, out, n); }

int rocm_sum_f32(const &float a, &float out1, unsigned long n) {
    struct RocmBuf da = rocm_upload(a, n);
    struct RocmBuf dout = rocm_alloc(1UL);
    int ok = da.dev != 0UL && dout.dev != 0UL && rocm_sum(da, dout) && rocm_read(dout, out1, 1UL);
    rocm_free(&da);
    rocm_free(&dout);
    return ok;
}

// ── Attention ────────────────────────────────────────────────────────────────

static int rocm_launch_grid3_(int kernel, unsigned long gx, unsigned long gy, unsigned long gz, &void params) {
    if (gx == 0UL || gy == 0UL || gz == 0UL || gx > 2147483647UL || gy > 65535UL || gz > 65535UL) return 0;
    unsafe {
        return hipModuleLaunchKernel(g_rocm_fn[kernel], (unsigned int)gx, (unsigned int)gy, (unsigned int)gz,
                                     ROCM_BLOCK, 1U, 1U, 0U, g_rocm_stream,
                                     (void**)params, (void**)0) == ROCM_HIP_OK;
    }
}

static int rocm_launch_grid_(int kernel, unsigned long gx, unsigned long gy, &void params) {
    return rocm_launch_grid3_(kernel, gx, gy, 1UL, params);
}

static int rocm_attn_shape_ok_(struct RocmBuf q, struct RocmBuf k, struct RocmBuf v, struct RocmBuf out,
                               unsigned long heads, unsigned long seqLen, unsigned long kvLen, unsigned long headDim) {
    if (heads == 0UL || seqLen == 0UL || kvLen == 0UL || headDim == 0UL || headDim > 128UL) return 0;
    if (heads > 65535UL || seqLen > 2147483647UL || kvLen > 2147483647UL) return 0;
    return rocm_fits_(q, heads * seqLen * headDim) && rocm_fits_(k, heads * kvLen * headDim) &&
           rocm_fits_(v, heads * kvLen * headDim) && rocm_fits_(out, heads * seqLen * headDim);
}

int rocm_flash_attention(struct RocmBuf q, struct RocmBuf k, struct RocmBuf v, struct RocmBuf out,
                         unsigned long heads, unsigned long seqLen, unsigned long kvLen, unsigned long headDim,
                         float scale, int causal) {
    if (!rocm_ensure_() || !rocm_attn_shape_ok_(q, k, v, out, heads, seqLen, kvLen, headDim)) return 0;
    if (causal && kvLen < seqLen) return 0;
    unsafe {
        void* pq = (void*)q.dev;
        void* pk = (void*)k.dev;
        void* pv = (void*)v.dev;
        void* po = (void*)out.dev;
        unsigned int s = (unsigned int)seqLen;
        unsigned int kn = (unsigned int)kvLen;
        unsigned int d = (unsigned int)headDim;
        float sc = scale > 0.0f ? scale : (float)(1.0 / sqrt_d((double)headDim));
        int ca = causal ? 1 : 0;
        void* params[9];
        params[0] = (void*)&pq; params[1] = (void*)&pk; params[2] = (void*)&pv; params[3] = (void*)&po;
        params[4] = (void*)&s; params[5] = (void*)&kn; params[6] = (void*)&d; params[7] = (void*)&sc;
        params[8] = (void*)&ca;
        if (g_rocm_mfma) {
            // matrix cores: grid (heads, query blocks of 128 / 64 rows)
            if (headDim <= 64UL) return rocm_launch_grid_(K_FLASH_M64, heads, (seqLen + 127UL) / 128UL, (&void)&params[0]);
            return rocm_launch_grid_(K_FLASH_M128, heads, (seqLen + 63UL) / 64UL, (&void)&params[0]);
        }
        return rocm_launch_grid_(headDim <= 64UL ? K_FLASH64 : K_FLASH128, (seqLen + 63UL) / 64UL, heads,
                                 (&void)&params[0]);
    }
}

// Per-chunk states for rocm_linear_attention, kept and grown as needed
// (not during graph capture: growing would allocate).
static struct RocmBuf g_rocm_lin_scratch;
static int g_rocm_lin_init = 0;

// Grows the linear-attention scratch to 'need' floats (not during capture).
static int rocm_lin_scratch_(unsigned long need) {
    if (!g_rocm_lin_init) { g_rocm_lin_scratch.dev = 0UL; g_rocm_lin_scratch.len = 0UL; g_rocm_lin_init = 1; }
    if (g_rocm_lin_scratch.len >= need) return 1;
    if (g_rocm_capturing) return 0;
    rocm_free(&g_rocm_lin_scratch);
    g_rocm_lin_scratch = rocm_alloc(need);
    if (g_rocm_lin_scratch.dev == 0UL) { g_rocm_lin_scratch.len = 0UL; return 0; }
    return 1;
}

// Matrix-core linear attention. Causal with head dim <= 64: one fused pass
// (la_causal_d64, decoupled look-back over 128-row chunks). Otherwise three
// passes: per-chunk states (128-row chunks; non-causal uses longer row ranges
// once there would be more than 64 of them), a scan over chunks, then the
// output pass. Head dims are padded to 64 or 128.
static int rocm_linear_attention_mfma_(struct RocmBuf q, struct RocmBuf k, struct RocmBuf v, struct RocmBuf out,
                                       unsigned long heads, unsigned long seqLen, unsigned long kvLen,
                                       unsigned long headDim, int causal) {
    unsigned long dm = headDim <= 64UL ? 64UL : 128UL;
    if (causal && dm == 64UL) {
        // one pass: per-chunk records (state, then inclusive prefix) and a
        // flag per chunk plus a ticket counter after them, zeroed per call
        unsigned long ncc = (seqLen + 127UL) / 128UL;
        unsigned long rs = 64UL * 64UL + 64UL;
        unsigned long recs = heads * ncc * 2UL * rs;
        if (!rocm_lin_scratch_(recs + heads * ncc + 1UL)) return 0;
        unsafe {
            void* pq = (void*)q.dev;
            void* pk = (void*)k.dev;
            void* pv = (void*)v.dev;
            void* po = (void*)out.dev;
            void* pr = (void*)g_rocm_lin_scratch.dev;
            void* pf = (void*)(g_rocm_lin_scratch.dev + recs * 4UL);
            if (hipMemsetAsync(pf, 0, (heads * ncc + 1UL) * 4UL, g_rocm_stream) != ROCM_HIP_OK) return 0;
            unsigned int s = (unsigned int)seqLen;
            unsigned int hn = (unsigned int)heads;
            unsigned int d = (unsigned int)headDim;
            unsigned int nc = (unsigned int)ncc;
            void* pc[10];
            pc[0] = (void*)&pq; pc[1] = (void*)&pk; pc[2] = (void*)&pv; pc[3] = (void*)&pr; pc[4] = (void*)&pf;
            pc[5] = (void*)&po; pc[6] = (void*)&s; pc[7] = (void*)&hn; pc[8] = (void*)&d; pc[9] = (void*)&nc;
            return rocm_launch_grid_(K_LA_CAUSAL64, ncc * heads, 1UL, (&void)&pc[0]);
        }
    }
    unsigned long rows = 128UL;
    if (!causal) rows = 128UL * (((kvLen + 127UL) / 128UL + 63UL) / 64UL);
    unsigned long nchunks = (kvLen + rows - 1UL) / rows;
    if (!rocm_lin_scratch_(heads * nchunks * dm * (dm + 1UL))) return 0;
    int big = dm == 128UL;
    unsafe {
        void* pq = (void*)q.dev;
        void* pk = (void*)k.dev;
        void* pv = (void*)v.dev;
        void* po = (void*)out.dev;
        void* pw = (void*)g_rocm_lin_scratch.dev;
        void* pz = (void*)(g_rocm_lin_scratch.dev + heads * nchunks * dm * dm * 4UL);
        unsigned int s = (unsigned int)seqLen;
        unsigned int kn = (unsigned int)kvLen;
        unsigned int d = (unsigned int)headDim;
        unsigned int rw = (unsigned int)rows;
        unsigned int nc = (unsigned int)nchunks;
        int ca = causal ? 1 : 0;
        void* p1[8];
        p1[0] = (void*)&pk; p1[1] = (void*)&pv; p1[2] = (void*)&pw; p1[3] = (void*)&pz; p1[4] = (void*)&kn;
        p1[5] = (void*)&d; p1[6] = (void*)&rw; p1[7] = (void*)&nc;
        if (!rocm_launch_grid_(big ? K_LA_STATE128 : K_LA_STATE64, nchunks, heads, (&void)&p1[0])) return 0;
        if (causal || nchunks > 1UL) {
            void* p2[4];
            p2[0] = (void*)&pw; p2[1] = (void*)&pz; p2[2] = (void*)&nc; p2[3] = (void*)&ca;
            if (!rocm_launch_grid_(big ? K_LA_SCAN128 : K_LA_SCAN64, (dm * dm + dm + 255UL) / 256UL, heads,
                                   (&void)&p2[0])) return 0;
        }
        void* p3[11];
        p3[0] = (void*)&pq; p3[1] = (void*)&pk; p3[2] = (void*)&pv; p3[3] = (void*)&pw; p3[4] = (void*)&pz;
        p3[5] = (void*)&po; p3[6] = (void*)&s; p3[7] = (void*)&kn; p3[8] = (void*)&d; p3[9] = (void*)&nc;
        p3[10] = (void*)&ca;
        return rocm_launch_grid3_(big ? K_LA_OUT128 : K_LA_OUT64, (seqLen + 127UL) / 128UL, heads, dm / 64UL,
                                  (&void)&p3[0]);
    }
}

int rocm_linear_attention(struct RocmBuf q, struct RocmBuf k, struct RocmBuf v, struct RocmBuf out,
                          unsigned long heads, unsigned long seqLen, unsigned long kvLen, unsigned long headDim,
                          int causal) {
    if (!rocm_ensure_() || !rocm_attn_shape_ok_(q, k, v, out, heads, seqLen, kvLen, headDim)) return 0;
    if (causal && kvLen != seqLen) return 0;
    if (g_rocm_mfma) return rocm_linear_attention_mfma_(q, k, v, out, heads, seqLen, kvLen, headDim, causal);
    unsigned long nchunks = (kvLen + 31UL) / 32UL;
    if (!rocm_lin_scratch_(heads * nchunks * headDim * (headDim + 1UL))) return 0;
    unsafe {
        void* pq = (void*)q.dev;
        void* pk = (void*)k.dev;
        void* pv = (void*)v.dev;
        void* po = (void*)out.dev;
        void* ps = (void*)g_rocm_lin_scratch.dev;
        unsigned int s = (unsigned int)seqLen;
        unsigned int kn = (unsigned int)kvLen;
        unsigned int d = (unsigned int)headDim;
        unsigned int nc = (unsigned int)nchunks;
        int ca = causal ? 1 : 0;
        void* p1[6];
        p1[0] = (void*)&pk; p1[1] = (void*)&pv; p1[2] = (void*)&ps; p1[3] = (void*)&kn; p1[4] = (void*)&d;
        p1[5] = (void*)&nc;
        if (!rocm_launch_grid_(K_LIN_STATE, nchunks, heads, (&void)&p1[0])) return 0;
        void* p2[4];
        p2[0] = (void*)&ps; p2[1] = (void*)&d; p2[2] = (void*)&nc; p2[3] = (void*)&ca;
        unsigned long elems = headDim * (headDim + 1UL);
        if (!rocm_launch_grid_(K_LIN_PREFIX, (elems + 255UL) / 256UL, heads, (&void)&p2[0])) return 0;
        void* p3[10];
        p3[0] = (void*)&pq; p3[1] = (void*)&pk; p3[2] = (void*)&pv; p3[3] = (void*)&ps; p3[4] = (void*)&po;
        p3[5] = (void*)&s; p3[6] = (void*)&kn; p3[7] = (void*)&d; p3[8] = (void*)&nc; p3[9] = (void*)&ca;
        return rocm_launch_grid_(K_LIN_OUT, (seqLen + 31UL) / 32UL, heads, (&void)&p3[0]);
    }
}

static int rocm_attention_f32_(const &float q, const &float k, const &float v, &float out,
                               unsigned long heads, unsigned long seqLen, unsigned long kvLen,
                               unsigned long headDim, float scale, int causal, int linear) {
    unsigned long nq = heads * seqLen * headDim;
    unsigned long nk = heads * kvLen * headDim;
    struct RocmBuf dq = rocm_upload(q, nq);
    struct RocmBuf dk = rocm_upload(k, nk);
    struct RocmBuf dv = rocm_upload(v, nk);
    struct RocmBuf dout = rocm_alloc(nq);
    int ok = dq.dev != 0UL && dk.dev != 0UL && dv.dev != 0UL && dout.dev != 0UL;
    if (ok) {
        ok = linear ? rocm_linear_attention(dq, dk, dv, dout, heads, seqLen, kvLen, headDim, causal)
                    : rocm_flash_attention(dq, dk, dv, dout, heads, seqLen, kvLen, headDim, scale, causal);
        ok = ok && rocm_read(dout, out, nq);
    }
    rocm_free(&dq);
    rocm_free(&dk);
    rocm_free(&dv);
    rocm_free(&dout);
    return ok;
}

int rocm_flash_attention_f32(const &float q, const &float k, const &float v, &float out,
                             unsigned long heads, unsigned long seqLen, unsigned long kvLen,
                             unsigned long headDim, float scale, int causal) {
    return rocm_attention_f32_(q, k, v, out, heads, seqLen, kvLen, headDim, scale, causal, 0);
}

int rocm_linear_attention_f32(const &float q, const &float k, const &float v, &float out,
                              unsigned long heads, unsigned long seqLen, unsigned long kvLen,
                              unsigned long headDim, int causal) {
    return rocm_attention_f32_(q, k, v, out, heads, seqLen, kvLen, headDim, 0.0f, causal, 1);
}

int rocm_matmul_f32(const &float a, const &float b, &float out,
                    unsigned long M, unsigned long K, unsigned long N) {
    if (M > 2147483647UL || N > 2147483647UL || K > 2147483647UL) return 0;
    struct RocmBuf da = rocm_upload(a, M * K);
    struct RocmBuf db = rocm_upload(b, K * N);
    struct RocmBuf dc = rocm_alloc(M * N);
    int ok = da.dev != 0UL && db.dev != 0UL && dc.dev != 0UL;
    if (ok) {
        unsafe {
            void* pa = (void*)da.dev;
            void* pb = (void*)db.dev;
            void* pc = (void*)dc.dev;
            int m = (int)M;
            int n = (int)N;
            int k = (int)K;
            void* params[6];
            params[0] = (void*)&pa; params[1] = (void*)&pb; params[2] = (void*)&pc;
            params[3] = (void*)&m; params[4] = (void*)&n; params[5] = (void*)&k;
            ok = hipModuleLaunchKernel(g_rocm_fn[K_SGEMM],
                                       (unsigned int)((N + 63UL) / 64UL), (unsigned int)((M + 63UL) / 64UL), 1U,
                                       ROCM_BLOCK, 1U, 1U, 0U, g_rocm_stream,
                                       &params[0], (void**)0) == ROCM_HIP_OK;
        }
        ok = ok && rocm_read(dc, out, M * N);
    }
    rocm_free(&da);
    rocm_free(&db);
    rocm_free(&dc);
    return ok;
}

int rocm_matmul_f32_blas(const &float a, const &float b, &float out,
                         unsigned long M, unsigned long K, unsigned long N) {
    struct RocmBuf da = rocm_upload(a, M * K);
    struct RocmBuf db = rocm_upload(b, K * N);
    struct RocmBuf dc = rocm_alloc(M * N);
    int ok = da.dev != 0UL && db.dev != 0UL && dc.dev != 0UL &&
             rocm_gemm(da, 0, db, 0, dc, M, N, K) && rocm_read(dc, out, M * N);
    rocm_free(&da);
    rocm_free(&db);
    rocm_free(&dc);
    return ok;
}

} // namespace std
