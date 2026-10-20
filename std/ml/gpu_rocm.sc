// SafeC Standard Library — AMD GPU backend (ROCm / HIP). See gpu_rocm.h.
//
// Kernels are HIP C++ source embedded below and compiled on first use with
// hiprtc for the selected GPU's own architecture (gfx908, gfx90a, gfx942,
// gfx1100, ...), then loaded with hipModuleLoadData — so one build of this
// library runs on any ROCm GPU. Compilation takes ~0.2 s once per process.
#pragma once
#include <std/ml/gpu_rocm.h>
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
#define K_COUNT 26

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
    return "mse_grad";
}

// ── Backend state (one per process) ──────────────────────────────────────────
static int   g_rocm_state = 0;        // 0 = not tried, 1 = ready, -1 = unavailable
static void* g_rocm_stream = (void*)0;
static void* g_rocm_blas = (void*)0;
static void* g_rocm_fn[K_COUNT];
static char  g_rocm_name[256];

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
    }
    g_rocm_state = 1;
    return 1;
}

int rocm_available() { return rocm_ensure_(); }

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
