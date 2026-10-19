// SafeC Standard Library — AMD GPU backend (ROCm / HIP). See gpu_rocm.h.
//
// Kernels are HIP C++ source embedded below and compiled on first use with
// hiprtc for the selected GPU's own architecture (gfx908, gfx90a, gfx942,
// gfx1100, ...), then loaded with hipModuleLoadData — so one build of this
// library runs on any ROCm GPU. Compilation takes ~0.2 s once per process.
#pragma once
#include <std/ml/gpu_rocm.h>
#include <std/mem.h>

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
extern int rocblas_sgemm(void* handle, int transA, int transB, int m, int n, int k,
                         const float* alpha, const float* A, int lda,
                         const float* B, int ldb,
                         const float* beta, float* C, int ldc);

#define ROCM_HIP_OK              0
#define ROCM_ATTR_CU_COUNT       63   // hipDeviceAttributeMultiprocessorCount
#define ROCM_ATTR_INTEGRATED     16   // hipDeviceAttributeIntegrated
#define ROCM_H2D                 1    // hipMemcpyHostToDevice
#define ROCM_D2H                 2    // hipMemcpyDeviceToHost
#define ROCM_CAPTURE_GLOBAL      0    // hipStreamCaptureModeGlobal
#define ROCM_BLAS_N              111  // rocblas_operation_none
#define ROCM_BLAS_T              112  // rocblas_operation_transpose
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
#define K_COUNT 15

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
        char* code = (char*)alloc(codeSize);
        if (code == (char*)0) { hiprtcDestroyProgram(&prog); return 0; }
        hiprtcGetCode(prog, code);
        hiprtcDestroyProgram(&prog);
        void* module = (void*)0;
        int loaded = hipModuleLoadData(&module, (const void*)code);
        dealloc((void*)code);
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

const char* rocm_device_name() {
    rocm_ensure_();
    unsafe { return (const char*)&g_rocm_name[0]; }
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

int rocm_gemm(struct RocmBuf a, int ta, struct RocmBuf b, int tb, struct RocmBuf c,
              unsigned long m, unsigned long n, unsigned long k) {
    if (!rocm_ensure_() || !rocm_fits_(a, m * k) || !rocm_fits_(b, k * n) || !rocm_fits_(c, m * n))
        return 0;
    float one = 1.0f;
    float zero = 0.0f;
    // Row-major C = op(A) op(B) is column-major C^T = op(B)^T op(A)^T, so
    // hand rocBLAS the operands swapped; each operand's leading dimension
    // is its stored row length.
    unsafe {
        return rocblas_sgemm(g_rocm_blas,
                             tb ? ROCM_BLAS_T : ROCM_BLAS_N, ta ? ROCM_BLAS_T : ROCM_BLAS_N,
                             (int)n, (int)m, (int)k, &one,
                             (const float*)b.dev, tb ? (int)k : (int)n,
                             (const float*)a.dev, ta ? (int)m : (int)k,
                             &zero, (float*)c.dev, (int)n) == 0;
    }
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
    unsafe { return hipStreamBeginCapture(g_rocm_stream, ROCM_CAPTURE_GLOBAL) == ROCM_HIP_OK; }
}

struct RocmGraph rocm_capture_end() {
    struct RocmGraph g;
    g.exec = 0UL;
    if (!rocm_ensure_()) return g;
    unsafe {
        void* graph = (void*)0;
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
