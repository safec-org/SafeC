// SafeC Standard Library — NVIDIA GPU backend (CUDA). See gpu_cuda.h.
//
// Kernels are CUDA C++ source embedded below and compiled on first use with
// NVRTC to a cubin for the selected GPU's own architecture (sm_80, sm_90,
// ...), then loaded with cuModuleLoadData — so one build of this library
// runs on any CUDA GPU the installed NVRTC knows. Compilation takes
// ~0.3 s once per process. Matmuls go to cuBLAS (and cuBLASLt for the
// fused GEMM + ReLU), on the same stream as the kernels.
#pragma once
#include <std/stddef.h>
#include <std/ml/gpu_cuda.h>
#include <std/ml/float16.h>
#include <std/mem.h>
#include <std/thread.h>

namespace std {

// ── Driver API / NVRTC / cuBLAS / cuBLASLt C ABI (cuda.h, nvrtc.h, cublas_v2.h, cublasLt.h) ──
extern int cuInit(unsigned int flags);
extern int cuDeviceGetCount(int* count);
extern int cuDeviceGet(int* device, int ordinal);
extern int cuDeviceGetAttribute(int* value, int attr, int device);
extern int cuDeviceGetName(char* name, int len, int device);
extern int cuDevicePrimaryCtxRetain(void** pctx, int device);
extern int cuCtxSetCurrent(void* ctx);
extern int cuCtxGetCurrent(void** pctx);
extern int cuStreamCreate(void** stream, unsigned int flags);
extern int cuStreamSynchronize(void* stream);
extern int cuMemAlloc_v2(unsigned long long* dptr, size_t bytes);
extern int cuMemFree_v2(unsigned long long dptr);
extern int cuMemAllocHost_v2(void** pp, size_t bytes);
extern int cuMemcpyHtoDAsync_v2(unsigned long long dst, const void* src, size_t bytes, void* stream);
extern int cuMemcpyDtoHAsync_v2(void* dst, unsigned long long src, size_t bytes, void* stream);
extern int cuMemcpyDtoDAsync_v2(unsigned long long dst, unsigned long long src, size_t bytes, void* stream);
extern int cuEventCreate(void** event, unsigned int flags);
extern int cuEventRecord(void* event, void* stream);
extern int cuEventSynchronize(void* event);
extern int cuStreamWaitEvent(void* stream, void* event, unsigned int flags);
extern int cuMemsetD32Async(unsigned long long dst, unsigned int value, size_t n, void* stream);
extern int cuModuleLoadData(void** module, const void* image);
extern int cuModuleGetFunction(void** func, void* module, const char* name);
extern int cuLaunchKernel(void* f, unsigned int gx, unsigned int gy, unsigned int gz,
                          unsigned int bx, unsigned int by, unsigned int bz,
                          unsigned int shared, void* stream, void** params, void** extra);
extern int cuStreamBeginCapture_v2(void* stream, int mode);
extern int cuStreamEndCapture(void* stream, void** graph);
extern int cuGraphInstantiateWithFlags(void** exec, void* graph, unsigned long long flags);
extern int cuGraphLaunch(void* exec, void* stream);
extern int cuGraphDestroy(void* graph);
extern int cuGraphExecDestroy(void* exec);

extern int nvrtcCreateProgram(void** prog, const char* src, const char* name, int numHeaders,
                              const char** headers, const char** includeNames);
extern int nvrtcCompileProgram(void* prog, int numOptions, const char** options);
extern int nvrtcGetCUBINSize(void* prog, size_t* size);
extern int nvrtcGetCUBIN(void* prog, char* cubin);
extern int nvrtcGetProgramLogSize(void* prog, size_t* size);
extern int nvrtcGetProgramLog(void* prog, char* log);
extern int nvrtcDestroyProgram(void** prog);

extern int cublasCreate_v2(void** handle);
extern int cublasSetStream_v2(void* handle, void* stream);
extern int cublasSetWorkspace_v2(void* handle, void* workspace, size_t bytes);
extern int cublasSetMathMode(void* handle, int mode);
extern int cublasSgemm_v2(void* handle, int transa, int transb, int m, int n, int k,
                          const float* alpha, const float* A, int lda,
                          const float* B, int ldb,
                          const float* beta, float* C, int ldc);

extern int cublasGemmEx(void* handle, int transa, int transb, int m, int n, int k, const void* alpha,
                        const void* a, int aType, int lda, const void* b, int bType, int ldb,
                        const void* beta, void* c, int cType, int ldc, int computeType, int algo);
extern int cublasLtCreate(void** handle);
extern int cublasLtMatmulDescCreate(void** desc, int computeType, int scaleType);
extern int cublasLtMatmulDescSetAttribute(void* desc, int attr, const void* buf, size_t size);
extern int cublasLtMatrixLayoutCreate(void** layout, int type, unsigned long long rows,
                                      unsigned long long cols, long long ld);
extern int cublasLtMatmulPreferenceCreate(void** pref);
extern int cublasLtMatmulPreferenceSetAttribute(void* pref, int attr, const void* buf, size_t size);
extern int cublasLtMatmulPreferenceDestroy(void* pref);
extern int cublasLtMatmulAlgoGetHeuristic(void* handle, void* desc, void* a, void* b, void* c, void* d,
                                          void* pref, int requested, void* results, int* returned);
extern int cublasLtMatmul(void* handle, void* desc, const void* alpha, const void* A, void* Adesc,
                          const void* B, void* Bdesc, const void* beta, const void* C, void* Cdesc,
                          void* D, void* Ddesc, const void* algo, void* workspace, size_t wsBytes,
                          void* stream);

extern void* memcpy(void* dst, const void* src, size_t n);
extern int printf(const char* fmt, ...);

#define CUDA_OK                  0
#define CUDA_ATTR_SM_COUNT       16   // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
#define CUDA_ATTR_CC_MAJOR       75   // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
#define CUDA_ATTR_CC_MINOR       76   // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
#define CUDA_STREAM_NON_BLOCKING 1U   // CU_STREAM_NON_BLOCKING
#define CUDA_CAPTURE_GLOBAL      0    // CU_STREAM_CAPTURE_MODE_GLOBAL
#define CUDA_BLAS_N              0    // CUBLAS_OP_N
#define CUDA_BLAS_T              1    // CUBLAS_OP_T
#define CUDA_R_32F_              0    // CUDA_R_32F
#define CUDA_R_16F_              2    // CUDA_R_16F
#define CUDA_R_16BF_             14   // CUDA_R_16BF
#define CUDA_GEMM_DEFAULT_       -1   // CUBLAS_GEMM_DEFAULT
#define CUDA_COMPUTE_32F         68   // CUBLAS_COMPUTE_32F
#define CUDA_COMPUTE_32F_TF32    77   // CUBLAS_COMPUTE_32F_FAST_TF32
#define CUDA_MATH_DEFAULT        0    // CUBLAS_DEFAULT_MATH
#define CUDA_MATH_TF32           3    // CUBLAS_TF32_TENSOR_OP_MATH
#define LT_DESC_TRANSA           3    // CUBLASLT_MATMUL_DESC_TRANSA
#define LT_DESC_TRANSB           4    // CUBLASLT_MATMUL_DESC_TRANSB
#define LT_DESC_EPILOGUE         7    // CUBLASLT_MATMUL_DESC_EPILOGUE
#define LT_EPILOGUE_RELU         2    // CUBLASLT_EPILOGUE_RELU
#define LT_PREF_MAX_WORKSPACE    1    // CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
#define LT_RESULT_WORDS          12   // sizeof(cublasLtMatmulHeuristicResult_t) / 8 (algo is the first 64 bytes)
#define LT_CACHE                 32
#define CUDA_BLOCK               256U
#define CUDA_MAX_BLOCKS          4096UL
#define CUDA_WORKSPACE_BYTES     33554432UL   // 32 MiB: cuBLAS's recommendation for graph capture on Hopper
#define CUDA_STAGING_BYTES       65536UL      // pinned bounce buffer for small reads
#define CUDA_PIPE_MIN            1048576UL    // transfers at least this big go through the lanes
#define CUDA_PIPE_LANES          4            // parallel copy lanes (more only contend for bandwidth)
#define CUDA_PIPE_CHUNK          2097152UL    // bytes per pinned buffer (two per lane)
#define CUDA_EVENT_NO_TIMING     2U           // CU_EVENT_DISABLE_TIMING

// ── Kernel source ────────────────────────────────────────────────────────────
// Elementwise kernels use grid-stride loops; reduce_sum and mse_grad reduce
// per warp with shuffles, then per block through shared memory, then one
// atomic per block (or a plain store when the grid is one block, which
// also makes the zeroing memset unnecessary). sgemm_nn is a 64x64-tile
// shared-memory kernel with 4x4 register tiles. The fp16/bf16 kernels
// keep 2-byte storage and do the arithmetic in fp32.
static const char* cuda_kernel_source_() {
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
    "EW_BINARY(ew_relu_grad, x > 0.0f ? y : 0.0f)\n"
    "EW_UNARY(ew_relu, x > 0.0f ? x : 0.0f)\n"
    "EW_UNARY(ew_log, __logf(x))\n"
    "EW_UNARY(ew_exp, __expf(x))\n"
    "EW_UNARY(ew_sqrt, __fsqrt_rn(x))\n"
    "EW_UNARY(ew_scale, x * k)\n"
    "\n"
    "// w -= lr * g, in place\n"
    "extern \"C\" __global__ void sgd_update(float* __restrict__ w, const float* __restrict__ g,\n"
    "                                      float lr, unsigned n) {\n"
    "    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    unsigned stride = gridDim.x * blockDim.x;\n"
    "    for (; i < n; i += stride) w[i] -= lr * g[i];\n"
    "}\n"
    "\n"
    "// two SGD updates in one launch: index space [0, n1) is w1, [n1, n1+n2) is w2\n"
    "extern \"C\" __global__ void sgd_update2(float* __restrict__ w1, const float* __restrict__ g1, unsigned n1,\n"
    "                                       float* __restrict__ w2, const float* __restrict__ g2, unsigned n2,\n"
    "                                       float lr) {\n"
    "    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    unsigned stride = gridDim.x * blockDim.x;\n"
    "    for (; i < n1 + n2; i += stride) {\n"
    "        if (i < n1) w1[i] -= lr * g1[i]; else w2[i - n1] -= lr * g2[i - n1];\n"
    "    }\n"
    "}\n"
    "\n"
    "// Block-wide sum of 's'; thread 0 gets the result.\n"
    "__device__ float block_sum(float s) {\n"
    "    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);\n"
    "    __shared__ float part[32];\n"
    "    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;\n"
    "    if (lane == 0) part[wid] = s;\n"
    "    __syncthreads();\n"
    "    s = 0.0f;\n"
    "    if (wid == 0) {\n"
    "        s = lane < (int)((blockDim.x + 31) >> 5) ? part[lane] : 0.0f;\n"
    "        for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);\n"
    "    }\n"
    "    return s;\n"
    "}\n"
    "\n"
    "// Sum: one partial per block, then one atomicAdd per block (*out zeroed\n"
    "// first) or, for a one-block grid, a plain store.\n"
    "extern \"C\" __global__ void reduce_sum(const float* __restrict__ a, float* __restrict__ out, unsigned n) {\n"
    "    float s = 0.0f;\n"
    "    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    unsigned stride = gridDim.x * blockDim.x;\n"
    "    for (; i < n; i += stride) s += a[i];\n"
    "    s = block_sum(s);\n"
    "    if (threadIdx.x == 0) { if (gridDim.x == 1) *out = s; else atomicAdd(out, s); }\n"
    "}\n"
    "\n"
    "// dY = 2 (y - t), loss = sum (y - t)^2   (fused: one pass instead of sub/scale/square/sum)\n"
    "extern \"C\" __global__ void mse_grad(const float* __restrict__ y, const float* __restrict__ t,\n"
    "                                    float* __restrict__ dy, float* __restrict__ loss, unsigned n) {\n"
    "    float s = 0.0f;\n"
    "    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    unsigned stride = gridDim.x * blockDim.x;\n"
    "    for (; i < n; i += stride) { float d = y[i] - t[i]; dy[i] = 2.0f * d; s += d * d; }\n"
    "    s = block_sum(s);\n"
    "    if (threadIdx.x == 0) { if (gridDim.x == 1) *loss = s; else atomicAdd(loss, s); }\n"
    "}\n"
    "\n"
    "// ── SGEMM: C[M,N] = A[M,K] * B[K,N], row-major ─────────────────────────────\n"
    "// 64x64 output tile per 256-thread block, BK=16, each thread a 4x4 micro-tile.\n"
    "extern \"C\" __global__ __launch_bounds__(256) void sgemm_nn(const float* __restrict__ A,\n"
    "        const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {\n"
    "    const int BM = 64, BN = 64, BK = 16;\n"
    "    __shared__ float As[BK][BM + 4];\n"
    "    __shared__ float Bs[BK][BN + 4];\n"
    "    int tid = threadIdx.x;\n"
    "    int tx = tid % 16, ty = tid / 16;\n"
    "    int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;\n"
    "    float acc[4][4] = {};\n"
    "    for (int k0 = 0; k0 < K; k0 += BK) {\n"
    "        for (int l = tid; l < BM * BK; l += 256) {\n"
    "            int m = l / BK, k = l % BK;\n"
    "            int gm = row0 + m, gk = k0 + k;\n"
    "            As[k][m] = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;\n"
    "        }\n"
    "        for (int l = tid; l < BK * BN; l += 256) {\n"
    "            int k = l / BN, n = l % BN;\n"
    "            int gk = k0 + k, gn = col0 + n;\n"
    "            Bs[k][n] = (gk < K && gn < N) ? B[(size_t)gk * N + gn] : 0.0f;\n"
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
    "            if (gn < N) C[(size_t)gm * N + gn] = acc[i][j];\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "// Nearest-neighbour 2x upsample of 'planes' h x w planes: out[p][y][x] = in[p][y/2][x/2].\n"
    "// One thread per input value, writing its 2x2 block as two float2 stores.\n"
    "extern \"C\" __global__ void upsample2x(const float* __restrict__ in, float* __restrict__ out,\n"
    "                                      unsigned planes, unsigned h, unsigned w) {\n"
    "    unsigned hw = h * w, n = planes * hw;\n"
    "    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    unsigned stride = gridDim.x * blockDim.x;\n"
    "    for (; i < n; i += stride) {\n"
    "        unsigned p = i / hw, r = i - p * hw, y = r / w, x = r - y * w;\n"
    "        float v = in[i];\n"
    "        float2* o = (float2*)(out + (size_t)p * 4u * hw + (size_t)(2u * y) * (2u * w) + 2u * x);\n"
    "        o[0] = make_float2(v, v);\n"
    "        o[w] = make_float2(v, v);   // next output row: 2w floats = w float2s on\n"
    "    }\n"
    "}\n"
    "\n"
    "// dst[dOff + r*dStride + c] = src[sOff + r*sStride + c] for r < rows, c < len; T is\n"
    "// float or float4 (all offsets, strides and len then counted in T).\n"
    "template <typename T> __device__ void copy_rows_t(const T* __restrict__ src, T* __restrict__ dst,\n"
    "        unsigned long long sOff, unsigned long long sStride, unsigned long long dOff,\n"
    "        unsigned long long dStride, unsigned len, unsigned rows) {\n"
    "    unsigned long long n = (unsigned long long)len * rows;\n"
    "    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;\n"
    "         i += (unsigned long long)gridDim.x * blockDim.x) {\n"
    "        unsigned r = (unsigned)(i / len), c = (unsigned)(i - (unsigned long long)r * len);\n"
    "        dst[dOff + r * dStride + c] = src[sOff + r * sStride + c];\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void copy_rows(const float* src, float* dst, unsigned long long sOff,\n"
    "        unsigned long long sStride, unsigned long long dOff, unsigned long long dStride,\n"
    "        unsigned len, unsigned rows) {\n"
    "    copy_rows_t<float>(src, dst, sOff, sStride, dOff, dStride, len, rows);\n"
    "}\n"
    "extern \"C\" __global__ void copy_rows4(const float4* src, float4* dst, unsigned long long sOff,\n"
    "        unsigned long long sStride, unsigned long long dOff, unsigned long long dStride,\n"
    "        unsigned len, unsigned rows) {\n"
    "    copy_rows_t<float4>(src, dst, sOff, sStride, dOff, dStride, len, rows);\n"
    "}\n"
    "\n"
    "// ── fp16 / bf16 storage, fp32 arithmetic ───────────────────────────────────\n"
    "__device__ float h2f(unsigned short h) { float f; asm(\"cvt.f32.f16 %0, %1;\" : \"=f\"(f) : \"h\"(h)); return f; }\n"
    "__device__ unsigned short f2h(float f) { unsigned short h; asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(h) : \"f\"(f)); return h; }\n"
    "__device__ float b2f(unsigned short b) { return __uint_as_float(((unsigned)b) << 16); }\n"
    "__device__ unsigned short f2b(float f) {   // round to nearest even; NaN stays NaN\n"
    "    unsigned u = __float_as_uint(f);\n"
    "    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x40u);\n"
    "    u += 0x7fffu + ((u >> 16) & 1u);\n"
    "    return (unsigned short)(u >> 16);\n"
    "}\n"
    "template <int BF> __device__ float ld16(unsigned short v) { return BF ? b2f(v) : h2f(v); }\n"
    "template <int BF> __device__ unsigned short st16(float v) { return BF ? f2b(v) : f2h(v); }\n"
    "template <int BF> __device__ void matmul16(const unsigned short* A, const unsigned short* B,\n"
    "                                         unsigned short* C, int M, int N, int K) {\n"
    "    int col = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int row = blockIdx.y * blockDim.y + threadIdx.y;\n"
    "    if (row >= M || col >= N) return;\n"
    "    float s = 0.0f;\n"
    "    for (int k = 0; k < K; ++k) s += ld16<BF>(A[(size_t)row * K + k]) * ld16<BF>(B[(size_t)k * N + col]);\n"
    "    C[(size_t)row * N + col] = st16<BF>(s);\n"
    "}\n"
    "template <int BF> __device__ void sgd16(unsigned short* w, const unsigned short* g, float lr, unsigned n) {\n"
    "    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    unsigned stride = gridDim.x * blockDim.x;\n"
    "    for (; i < n; i += stride) w[i] = st16<BF>(ld16<BF>(w[i]) - lr * ld16<BF>(g[i]));\n"
    "}\n"
    "extern \"C\" __global__ void matmul_f16(const unsigned short* A, const unsigned short* B, unsigned short* C, int M, int N, int K) { matmul16<0>(A, B, C, M, N, K); }\n"
    "extern \"C\" __global__ void matmul_bf16(const unsigned short* A, const unsigned short* B, unsigned short* C, int M, int N, int K) { matmul16<1>(A, B, C, M, N, K); }\n"
    "extern \"C\" __global__ void sgd_f16(unsigned short* w, const unsigned short* g, float lr, unsigned n) { sgd16<0>(w, g, lr, n); }\n"
    "extern \"C\" __global__ void sgd_bf16(unsigned short* w, const unsigned short* g, float lr, unsigned n) { sgd16<1>(w, g, lr, n); }\n"
    "// CudaHalfBuf ops (grid-stride, fp32 arithmetic, one rounding per stored value)\n"
    "#define GS for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)\n"
    "template <int BF> __device__ void to16(const float* a, unsigned short* o, unsigned n) { GS o[i] = st16<BF>(a[i]); }\n"
    "template <int BF> __device__ void from16(const unsigned short* a, float* o, unsigned n) { GS o[i] = ld16<BF>(a[i]); }\n"
    "template <int BF> __device__ void relu16(const unsigned short* a, unsigned short* o, unsigned n) {\n"
    "    GS { float v = ld16<BF>(a[i]); o[i] = v > 0.0f ? a[i] : (unsigned short)0; }\n"
    "}\n"
    "template <int BF> __device__ void relug16(const unsigned short* z, const unsigned short* g, unsigned short* o, unsigned n) {\n"
    "    GS o[i] = ld16<BF>(z[i]) > 0.0f ? g[i] : (unsigned short)0;\n"
    "}\n"
    "template <int BF> __device__ void mse16(const unsigned short* y, const unsigned short* t, unsigned short* dy,\n"
    "                                      float* loss, unsigned n) {\n"
    "    float s = 0.0f;\n"
    "    GS { float d = ld16<BF>(y[i]) - ld16<BF>(t[i]); dy[i] = st16<BF>(2.0f * d); s += d * d; }\n"
    "    s = block_sum(s);\n"
    "    if (threadIdx.x == 0) { if (gridDim.x == 1) *loss = s; else atomicAdd(loss, s); }\n"
    "}\n"
    "template <int BF> __device__ void sgd2_16(unsigned short* w1, const unsigned short* g1, unsigned n1,\n"
    "        unsigned short* w2, const unsigned short* g2, unsigned n2, float lr) {\n"
    "    unsigned n = n1 + n2;\n"
    "    GS {\n"
    "        if (i < n1) w1[i] = st16<BF>(ld16<BF>(w1[i]) - lr * ld16<BF>(g1[i]));\n"
    "        else w2[i - n1] = st16<BF>(ld16<BF>(w2[i - n1]) - lr * ld16<BF>(g2[i - n1]));\n"
    "    }\n"
    "}\n"
    "extern \"C\" __global__ void to_f16(const float* a, unsigned short* o, unsigned n) { to16<0>(a, o, n); }\n"
    "extern \"C\" __global__ void to_bf16(const float* a, unsigned short* o, unsigned n) { to16<1>(a, o, n); }\n"
    "extern \"C\" __global__ void from_f16(const unsigned short* a, float* o, unsigned n) { from16<0>(a, o, n); }\n"
    "extern \"C\" __global__ void from_bf16(const unsigned short* a, float* o, unsigned n) { from16<1>(a, o, n); }\n"
    "extern \"C\" __global__ void relu_f16(const unsigned short* a, unsigned short* o, unsigned n) { relu16<0>(a, o, n); }\n"
    "extern \"C\" __global__ void relu_bf16(const unsigned short* a, unsigned short* o, unsigned n) { relu16<1>(a, o, n); }\n"
    "extern \"C\" __global__ void relug_f16(const unsigned short* z, const unsigned short* g, unsigned short* o, unsigned n) { relug16<0>(z, g, o, n); }\n"
    "extern \"C\" __global__ void relug_bf16(const unsigned short* z, const unsigned short* g, unsigned short* o, unsigned n) { relug16<1>(z, g, o, n); }\n"
    "extern \"C\" __global__ void mse_f16(const unsigned short* y, const unsigned short* t, unsigned short* dy, float* l, unsigned n) { mse16<0>(y, t, dy, l, n); }\n"
    "extern \"C\" __global__ void mse_bf16(const unsigned short* y, const unsigned short* t, unsigned short* dy, float* l, unsigned n) { mse16<1>(y, t, dy, l, n); }\n"
    "extern \"C\" __global__ void sgd2_f16(unsigned short* w1, const unsigned short* g1, unsigned n1, unsigned short* w2, const unsigned short* g2, unsigned n2, float lr) { sgd2_16<0>(w1, g1, n1, w2, g2, n2, lr); }\n"
    "extern \"C\" __global__ void sgd2_bf16(unsigned short* w1, const unsigned short* g1, unsigned n1, unsigned short* w2, const unsigned short* g2, unsigned n2, float lr) { sgd2_16<1>(w1, g1, n1, w2, g2, n2, lr); }\n";
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
#define K_SGD2 15
#define K_MATMUL_F16 16
#define K_MATMUL_BF16 17
#define K_SGD_F16 18
#define K_SGD_BF16 19
#define K_UPSAMPLE2X 20
#define K_COPY_ROWS 21
#define K_COPY_ROWS4 22
// CudaHalfBuf kernels: each fp16 index is followed by its bf16 twin
#define K_TO_F16 23
#define K_FROM_F16 25
#define K_RELU_F16 27
#define K_RELUG_F16 29
#define K_MSE_F16 31
#define K_SGD2_F16 33
#define K_COUNT 35

static const char* cuda_kernel_name_(int k) {
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
    if (k == K_MSE) return "mse_grad";
    if (k == K_SGD2) return "sgd_update2";
    if (k == K_MATMUL_F16) return "matmul_f16";
    if (k == K_MATMUL_BF16) return "matmul_bf16";
    if (k == K_SGD_F16) return "sgd_f16";
    if (k == K_SGD_BF16) return "sgd_bf16";
    if (k == K_COPY_ROWS) return "copy_rows";
    if (k == K_COPY_ROWS4) return "copy_rows4";
    if (k == K_TO_F16) return "to_f16";
    if (k == K_TO_F16 + 1) return "to_bf16";
    if (k == K_FROM_F16) return "from_f16";
    if (k == K_FROM_F16 + 1) return "from_bf16";
    if (k == K_RELU_F16) return "relu_f16";
    if (k == K_RELU_F16 + 1) return "relu_bf16";
    if (k == K_RELUG_F16) return "relug_f16";
    if (k == K_RELUG_F16 + 1) return "relug_bf16";
    if (k == K_MSE_F16) return "mse_f16";
    if (k == K_MSE_F16 + 1) return "mse_bf16";
    if (k == K_SGD2_F16) return "sgd2_f16";
    if (k == K_SGD2_F16 + 1) return "sgd2_bf16";
    return "upsample2x";
}

// ── Backend state (one per process) ──────────────────────────────────────────
static int   g_cuda_state = 0;        // 0 = not tried, 1 = ready, -1 = unavailable
static void* g_cuda_ctx = (void*)0;
static void* g_cuda_stream = (void*)0;
static void* g_cuda_blas = (void*)0;
static void* g_cuda_lt = (void*)0;
static unsigned long long g_cuda_ws = 0ULL;
static void* g_cuda_staging = (void*)0;
// Pinned transfer lanes (allocated on the first large transfer). Lane k
// has its own stream, two pinned buffers g_cuda_pipe[2k + b], each with an
// event recorded after the last copy that used it (g_cuda_pipe_busy = 1
// while that copy may still be in flight), and an end event.
static int   g_cuda_pipe_state = 0;   // 0 = not tried, 1 = ready, -1 = unavailable
static void* g_cuda_pipe[CUDA_PIPE_LANES * 2];
static void* g_cuda_pipe_ev[CUDA_PIPE_LANES * 2];
static int   g_cuda_pipe_busy[CUDA_PIPE_LANES * 2];
static void* g_cuda_lane_stream[CUDA_PIPE_LANES];
static void* g_cuda_lane_end[CUDA_PIPE_LANES];
static int   g_cuda_lane_ok[CUDA_PIPE_LANES];
static void* g_cuda_pipe_start = (void*)0;
// Lanes 1.. run on their own threads (lane 0 on the caller), parked on
// g_cuda_lane_wake between copies: waking three threads costs much less
// than a whole-machine pool would, and they never contend with it.
static unsigned long long g_cuda_lane_mu = 0ULL;
static unsigned long long g_cuda_lane_wake = 0ULL;
static unsigned long long g_cuda_lane_done = 0ULL;
static unsigned long g_cuda_lane_gen = 0UL;
static unsigned long g_cuda_lane_pending = 0UL;
static unsigned long g_cuda_lane_count = 0UL;    // lanes in the current copy
static unsigned long g_cuda_lane_threads = 0UL;  // helper threads started
static unsigned long g_cuda_lane_ids[CUDA_PIPE_LANES];
static unsigned long g_cuda_lane_born[CUDA_PIPE_LANES];   // g_cuda_lane_gen when each was started
static void* g_cuda_lane_job = (void*)0;
static void* g_cuda_fn[K_COUNT];
static char  g_cuda_name[256];
static int   g_cuda_blocks_cap = 0;   // 0 = use CUDA_MAX_BLOCKS
static int   g_cuda_tf32 = 0;

// cuBLASLt plans, one per (ta, tb, m, n, k, epilogue): descriptor, three
// layouts and the heuristic's algo. -1 in g_lt_key0 = free slot.
static long  g_lt_key0[LT_CACHE];     // ta | tb << 1 | tf32 << 2 | epilogue << 3
static long  g_lt_m[LT_CACHE];
static long  g_lt_n[LT_CACHE];
static long  g_lt_k[LT_CACHE];
static void* g_lt_desc[LT_CACHE];
static void* g_lt_la[LT_CACHE];
static void* g_lt_lb[LT_CACHE];
static void* g_lt_lc[LT_CACHE];
static unsigned long g_lt_algo[LT_CACHE * LT_RESULT_WORDS];

// Compiles the kernel source for sm_<major><minor> and loads every kernel.
static int cuda_build_kernels_(int major, int minor) {
    unsafe {
        void* prog = (void*)0;
        if (nvrtcCreateProgram(&prog, cuda_kernel_source_(), "safec_cuda.cu", 0,
                               (const char**)0, (const char**)0) != 0) return 0;
        char arch[32];
        int v = major * 10 + minor;
        // "-arch=sm_" + two or three digits
        const char* prefix = "-arch=sm_";
        int p = 0;
        while (prefix[p] != (char)0) { arch[p] = prefix[p]; p = p + 1; }
        if (v >= 100) { arch[p] = (char)(48 + v / 100); p = p + 1; }
        arch[p] = (char)(48 + (v / 10) % 10); p = p + 1;
        arch[p] = (char)(48 + v % 10); p = p + 1;
        arch[p] = (char)0;
        const char* opts[3];
        opts[0] = (const char*)&arch[0];
        opts[1] = "-std=c++17";
        opts[2] = "-default-device";
        if (nvrtcCompileProgram(prog, 3, &opts[0]) != 0) {
            size_t logSize = 0UL;
            nvrtcGetProgramLogSize(prog, &logSize);
            char* log = (char*)alloc_raw_((unsigned long)logSize + 1UL);
            if (log != (char*)0) {
                nvrtcGetProgramLog(prog, log);
                printf("std::cuda: kernel compile failed (%s):\n%s\n", (const char*)&arch[0], (const char*)log);
                dealloc_raw_((void*)log);
            }
            nvrtcDestroyProgram(&prog);
            return 0;
        }
        size_t size = 0UL;
        nvrtcGetCUBINSize(prog, &size);
        char* cubin = (char*)alloc_raw_(size);
        if (cubin == (char*)0) { nvrtcDestroyProgram(&prog); return 0; }
        nvrtcGetCUBIN(prog, cubin);
        nvrtcDestroyProgram(&prog);
        void* module = (void*)0;
        int loaded = cuModuleLoadData(&module, (const void*)cubin);
        dealloc_raw_((void*)cubin);
        if (loaded != CUDA_OK) return 0;
        int k = 0;
        while (k < K_COUNT) {
            if (cuModuleGetFunction(&g_cuda_fn[k], module, cuda_kernel_name_(k)) != CUDA_OK) return 0;
            k = k + 1;
        }
    }
    return 1;
}

// Initializes everything once: picks the GPU with the most SMs, makes its
// primary context current, creates the stream, the cuBLAS / cuBLASLt
// handles and their shared workspace, the pinned read buffer, and compiles
// + loads the kernels. Later calls only make sure the context is current
// on the calling thread.
static int cuda_ensure_() {
    if (g_cuda_state == 1) {
        unsafe {
            void* cur = (void*)0;
            cuCtxGetCurrent(&cur);
            if (cur != g_cuda_ctx) cuCtxSetCurrent(g_cuda_ctx);
        }
        return 1;
    }
    if (g_cuda_state != 0) return 0;
    g_cuda_state = -1;
    unsafe {
        g_cuda_name[0] = (char)0;
        if (cuInit(0U) != CUDA_OK) return 0;
        int count = 0;
        if (cuDeviceGetCount(&count) != CUDA_OK || count <= 0) return 0;
        int best = -1;
        int bestSms = -1;
        int i = 0;
        while (i < count) {
            int dev = 0;
            int sms = 0;
            if (cuDeviceGet(&dev, i) == CUDA_OK &&
                cuDeviceGetAttribute(&sms, CUDA_ATTR_SM_COUNT, dev) == CUDA_OK && sms > bestSms) {
                bestSms = sms;
                best = dev;
            }
            i = i + 1;
        }
        if (best < 0) return 0;
        cuDeviceGetName(&g_cuda_name[0], 256, best);
        int major = 0;
        int minor = 0;
        cuDeviceGetAttribute(&major, CUDA_ATTR_CC_MAJOR, best);
        cuDeviceGetAttribute(&minor, CUDA_ATTR_CC_MINOR, best);
        g_cuda_blocks_cap = bestSms * 8;
        if (cuDevicePrimaryCtxRetain(&g_cuda_ctx, best) != CUDA_OK) return 0;
        if (cuCtxSetCurrent(g_cuda_ctx) != CUDA_OK) return 0;
        if (cuStreamCreate(&g_cuda_stream, CUDA_STREAM_NON_BLOCKING) != CUDA_OK) return 0;
        if (cuMemAlloc_v2(&g_cuda_ws, CUDA_WORKSPACE_BYTES) != CUDA_OK) return 0;
        if (cuMemAllocHost_v2(&g_cuda_staging, CUDA_STAGING_BYTES) != CUDA_OK) return 0;
        if (cublasCreate_v2(&g_cuda_blas) != 0) return 0;
        cublasSetStream_v2(g_cuda_blas, g_cuda_stream);
        cublasSetWorkspace_v2(g_cuda_blas, (void*)(unsigned long)g_cuda_ws, CUDA_WORKSPACE_BYTES);
        if (cublasLtCreate(&g_cuda_lt) != 0) g_cuda_lt = (void*)0;   // optional: gemm_relu falls back
        int s = 0;
        while (s < LT_CACHE) { g_lt_key0[s] = -1L; s = s + 1; }
        if (!cuda_build_kernels_(major, minor)) return 0;
    }
    g_cuda_state = 1;
    return 1;
}

int cuda_available() { return cuda_ensure_(); }

const &char cuda_device_name() {
    cuda_ensure_();
    unsafe { return (const char*)&g_cuda_name[0]; }
}

// ── Buffers ──────────────────────────────────────────────────────────────────

static unsigned long long cuda_alloc_bytes_(unsigned long bytes) {
    if (bytes == 0UL || !cuda_ensure_()) return 0ULL;
    unsafe {
        unsigned long long p = 0ULL;
        if (cuMemAlloc_v2(&p, bytes) != CUDA_OK) return 0ULL;
        return p;
    }
}

// ── Host <-> device copies ───────────────────────────────────────────────────
// A copy from pageable memory makes the driver bounce it through its own
// pinned buffers on one thread, which bounds it at one core's memcpy rate
// (~16 GB/s here, a third of what PCIe 5 moves). Large copies instead split
// into CUDA_PIPE_LANES slices that the thread pool runs side by side: each
// lane copies its slice chunk by chunk between host memory and its two
// pinned buffers in turn, on its own stream, so the DMA of one chunk
// overlaps the copy of the next. Events order the lanes after the work
// already queued on the main stream, and (for writes) the main stream
// after the lanes.

static int cuda_pipe_ensure_() {
    if (g_cuda_pipe_state != 0) return g_cuda_pipe_state == 1;
    g_cuda_pipe_state = -1;
    unsafe {
        if (cuEventCreate(&g_cuda_pipe_start, CUDA_EVENT_NO_TIMING) != CUDA_OK) return 0;
        int i = 0;
        while (i < CUDA_PIPE_LANES * 2) {
            g_cuda_pipe_busy[i] = 0;
            if (cuMemAllocHost_v2(&g_cuda_pipe[i], CUDA_PIPE_CHUNK) != CUDA_OK ||
                cuEventCreate(&g_cuda_pipe_ev[i], CUDA_EVENT_NO_TIMING) != CUDA_OK) return 0;
            i = i + 1;
        }
        i = 0;
        while (i < CUDA_PIPE_LANES) {
            if (cuStreamCreate(&g_cuda_lane_stream[i], CUDA_STREAM_NON_BLOCKING) != CUDA_OK ||
                cuEventCreate(&g_cuda_lane_end[i], CUDA_EVENT_NO_TIMING) != CUDA_OK) return 0;
            i = i + 1;
        }
    }
    g_cuda_pipe_state = 1;
    return 1;
}

// Waits until no queued copy uses pinned buffer i.
static int cuda_pipe_wait_(int i) {
    if (!g_cuda_pipe_busy[i]) return 1;
    g_cuda_pipe_busy[i] = 0;
    unsafe { return cuEventSynchronize(g_cuda_pipe_ev[i]) == CUDA_OK; }
}

struct CudaLaneJob {
    char*              host;
    unsigned long long dev;
    unsigned long      bytes;
    unsigned long      part;   // bytes per lane
    int                toDevice;
};

// Lane k's slice: host -> device (queued; returns once the host bytes are
// staged) or device -> host (returns once they've landed).
static int cuda_lane_run_(&CudaLaneJob j, int k) {
    unsigned long base = (unsigned long)k * j.part;
    if (base >= j.bytes) return 1;
    unsigned long len = j.bytes - base < j.part ? j.bytes - base : j.part;
    unsafe {
        void* cur = (void*)0;
        cuCtxGetCurrent(&cur);
        if (cur != g_cuda_ctx && cuCtxSetCurrent(g_cuda_ctx) != CUDA_OK) return 0;   // pool threads
        void* st = g_cuda_lane_stream[k];
        if (cuStreamWaitEvent(st, g_cuda_pipe_start, 0U) != CUDA_OK) return 0;
        char* host = j.host + base;
        unsigned long long dev = j.dev + (unsigned long long)base;
        unsigned long off = 0UL;
        unsigned long prevOff = 0UL;
        unsigned long prevLen = 0UL;
        int b = 0;
        while (off < len) {
            unsigned long c = len - off < CUDA_PIPE_CHUNK ? len - off : CUDA_PIPE_CHUNK;
            int i = 2 * k + b;
            if (!cuda_pipe_wait_(i)) return 0;
            if (j.toDevice) {
                memcpy(g_cuda_pipe[i], (const void*)(host + off), c);
                if (cuMemcpyHtoDAsync_v2(dev + (unsigned long long)off, (const void*)g_cuda_pipe[i], c, st) != CUDA_OK)
                    return 0;
            } else {
                if (cuMemcpyDtoHAsync_v2(g_cuda_pipe[i], dev + (unsigned long long)off, c, st) != CUDA_OK) return 0;
            }
            if (cuEventRecord(g_cuda_pipe_ev[i], st) != CUDA_OK) return 0;
            g_cuda_pipe_busy[i] = 1;
            if (!j.toDevice && prevLen > 0UL) {
                int p = 2 * k + (1 - b);
                if (!cuda_pipe_wait_(p)) return 0;
                memcpy((void*)(host + prevOff), (const void*)g_cuda_pipe[p], prevLen);
            }
            prevOff = off;
            prevLen = c;
            off = off + c;
            b = 1 - b;
        }
        if (!j.toDevice) {
            int p = 2 * k + (1 - b);
            if (!cuda_pipe_wait_(p)) return 0;
            memcpy((void*)(host + prevOff), (const void*)g_cuda_pipe[p], prevLen);
        }
        return cuEventRecord(g_cuda_lane_end[k], st) == CUDA_OK;
    }
}

static ?&unsigned long cuda_lane_worker_(?&unsigned long arg) {
    unsigned long k = 0UL;
    match (arg) { case some(p): { k = *p; } case null: { } }
    mutex_lock(&g_cuda_lane_mu);
    unsigned long seen = g_cuda_lane_born[k];   // not g_cuda_lane_gen: a copy may already be posted
    while (1) {
        while (g_cuda_lane_gen == seen) cond_wait(&g_cuda_lane_wake, &g_cuda_lane_mu);
        seen = g_cuda_lane_gen;
        int take = k < g_cuda_lane_count;
        void* job = g_cuda_lane_job;
        mutex_unlock(&g_cuda_lane_mu);
        if (take) { unsafe { g_cuda_lane_ok[k] = cuda_lane_run_((&CudaLaneJob)job, (int)k); } }
        mutex_lock(&g_cuda_lane_mu);
        if (take) {
            g_cuda_lane_pending = g_cuda_lane_pending - 1UL;
            if (g_cuda_lane_pending == 0UL) cond_signal(&g_cuda_lane_done);
        }
    }
    return null;
}

// Starts the helper threads on first use; returns how many lanes can run
// (1 = only the caller's, if no thread could be started).
static int g_cuda_lane_inited = 0;

static unsigned long cuda_lanes_start_() {
    if (!g_cuda_lane_inited) {
        g_cuda_lane_inited = 1;
        if (mutex_init(&g_cuda_lane_mu) != 0 || cond_init(&g_cuda_lane_wake) != 0 ||
            cond_init(&g_cuda_lane_done) != 0) return 1UL;
        unsigned long k = 1UL;
        while (k < (unsigned long)CUDA_PIPE_LANES) {
            g_cuda_lane_ids[k] = k;
            g_cuda_lane_born[k] = g_cuda_lane_gen;
            unsigned long long tid = 0ULL;
            if (thread_create(&tid, cuda_lane_worker_, (?&unsigned long)&g_cuda_lane_ids[k]) != 0) break;
            thread_detach(tid);
            g_cuda_lane_threads = k;
            k = k + 1UL;
        }
    }
    return g_cuda_lane_threads + 1UL;
}

// Runs a large copy over the lanes; 1 on success.
static int cuda_lanes_copy_(void* host, unsigned long long dev, unsigned long bytes, int toDevice) {
    struct CudaLaneJob j;
    unsafe { j.host = (char*)host; }
    j.dev = dev;
    j.bytes = bytes;
    j.toDevice = toDevice;
    unsigned long lanes = (bytes + CUDA_PIPE_MIN - 1UL) / CUDA_PIPE_MIN;   // >= 1 MiB per lane
    unsigned long avail = cuda_lanes_start_();
    if (lanes > avail) lanes = avail;
    j.part = (bytes / lanes + 4095UL) & ~4095UL;
    unsafe {
        if (cuEventRecord(g_cuda_pipe_start, g_cuda_stream) != CUDA_OK) return 0;
        if (lanes > 1UL) {
            mutex_lock(&g_cuda_lane_mu);
            g_cuda_lane_job = (void*)&j;
            g_cuda_lane_count = lanes;
            g_cuda_lane_pending = lanes - 1UL;
            g_cuda_lane_gen = g_cuda_lane_gen + 1UL;
            cond_broadcast(&g_cuda_lane_wake);
            mutex_unlock(&g_cuda_lane_mu);
        }
        g_cuda_lane_ok[0] = cuda_lane_run_(&j, 0);
        if (lanes > 1UL) {
            mutex_lock(&g_cuda_lane_mu);
            while (g_cuda_lane_pending != 0UL) cond_wait(&g_cuda_lane_done, &g_cuda_lane_mu);
            mutex_unlock(&g_cuda_lane_mu);
        }
    }
    int ok = 1;
    unsigned long k = 0UL;
    while (k < lanes) {
        if (!g_cuda_lane_ok[k]) ok = 0;
        // later work on the main stream waits for this lane's copies
        unsafe { if (ok && toDevice && cuStreamWaitEvent(g_cuda_stream, g_cuda_lane_end[k], 0U) != CUDA_OK) ok = 0; }
        k = k + 1UL;
    }
    return ok;
}

// Queues host -> device; returns once 'host' may be reused (the last chunks
// may still be on their way from pinned memory).
static int cuda_write_bytes_(unsigned long long dst, const void* host, unsigned long bytes) {
    unsafe {
        if (bytes < CUDA_PIPE_MIN || !cuda_pipe_ensure_())
            return cuMemcpyHtoDAsync_v2(dst, host, bytes, g_cuda_stream) == CUDA_OK;
        return cuda_lanes_copy_((void*)host, dst, bytes, 1);
    }
}

// Waits for queued work, then copies 'bytes' to 'host'. Small reads (a
// loss, one output value) bounce through the pinned staging buffer: a
// pinned D2H copy is one DMA, where a pageable one adds a driver-side copy.
static int cuda_read_bytes_(void* host, unsigned long long src, unsigned long bytes) {
    unsafe {
        if (bytes <= CUDA_STAGING_BYTES) {
            if (cuMemcpyDtoHAsync_v2(g_cuda_staging, src, bytes, g_cuda_stream) != CUDA_OK) return 0;
            if (cuStreamSynchronize(g_cuda_stream) != CUDA_OK) return 0;
            memcpy(host, (const void*)g_cuda_staging, bytes);
            return 1;
        }
        if (bytes < CUDA_PIPE_MIN || !cuda_pipe_ensure_()) {
            if (cuMemcpyDtoHAsync_v2(host, src, bytes, g_cuda_stream) != CUDA_OK) return 0;
            return cuStreamSynchronize(g_cuda_stream) == CUDA_OK;
        }
        return cuda_lanes_copy_(host, src, bytes, 0);
    }
}

int cuda_copy_to_device(unsigned long dev, const &void host, unsigned long bytes) {
    if (dev == 0UL || bytes == 0UL || !cuda_ensure_()) return 0;
    unsafe { return cuda_write_bytes_((unsigned long long)dev, (const void*)host, bytes); }
}

int cuda_copy_to_host(&void host, unsigned long dev, unsigned long bytes) {
    if (dev == 0UL || bytes == 0UL || !cuda_ensure_()) return 0;
    unsafe { return cuda_read_bytes_((void*)host, (unsigned long long)dev, bytes); }
}

struct CudaBuf cuda_alloc(unsigned long n) {
    struct CudaBuf b;
    b.dev = 0UL;
    b.len = 0UL;
    if (n == 0UL) return b;
    unsigned long long p = cuda_alloc_bytes_(checked_mul_size(n, sizeof(float)));
    if (p == 0ULL) return b;
    b.dev = (unsigned long)p;
    b.len = n;
    return b;
}

int cuda_write(struct CudaBuf dst, const &float host, unsigned long n) {
    if (dst.dev == 0UL || n > dst.len || !cuda_ensure_()) return 0;
    unsafe { return cuda_write_bytes_((unsigned long long)dst.dev, (const void*)host, n * sizeof(float)); }
}

struct CudaBuf cuda_upload(const &float host, unsigned long n) {
    struct CudaBuf b = cuda_alloc(n);
    if (b.dev != 0UL && !cuda_write(b, host, n)) cuda_free(&b);
    return b;
}

int cuda_read(struct CudaBuf src, &float host, unsigned long n) {
    if (src.dev == 0UL || n > src.len || !cuda_ensure_()) return 0;
    unsafe { return cuda_read_bytes_((void*)host, (unsigned long long)src.dev, n * sizeof(float)); }
}

void cuda_free(&CudaBuf b) {
    if (b.dev != 0UL) { unsafe { cuMemFree_v2((unsigned long long)b.dev); } }
    b.dev = 0UL;
    b.len = 0UL;
}

int cuda_sync() {
    if (!cuda_ensure_()) return 0;
    unsafe { return cuStreamSynchronize(g_cuda_stream) == CUDA_OK; }
}

int cuda_set_tf32(int on) {
    if (!cuda_ensure_()) return 0;
    unsafe {
        if (cublasSetMathMode(g_cuda_blas, on ? CUDA_MATH_TF32 : CUDA_MATH_DEFAULT) != 0) return 0;
    }
    g_cuda_tf32 = on ? 1 : 0;   // cuBLASLt plans are keyed on this
    return 1;
}

// ── Launch helpers ───────────────────────────────────────────────────────────

static unsigned int cuda_blocks_(unsigned long n) {
    unsigned long blocks = (n + (unsigned long)CUDA_BLOCK - 1UL) / (unsigned long)CUDA_BLOCK;
    unsigned long cap = g_cuda_blocks_cap > 0 ? (unsigned long)g_cuda_blocks_cap : CUDA_MAX_BLOCKS;
    if (blocks > cap) blocks = cap;
    if (blocks == 0UL) blocks = 1UL;
    return (unsigned int)blocks;
}

static int cuda_launch_(int kernel, unsigned long n, &void params) {
    unsafe {
        return cuLaunchKernel(g_cuda_fn[kernel], cuda_blocks_(n), 1U, 1U,
                              CUDA_BLOCK, 1U, 1U, 0U, g_cuda_stream,
                              (void**)params, (void**)0) == CUDA_OK;
    }
}

static int cuda_fits_(struct CudaBuf b, unsigned long n) { return b.dev != 0UL && b.len >= n; }

static int cuda_binary_raw_(int kernel, unsigned long a, unsigned long b, unsigned long out, unsigned long n) {
    unsafe {
        void* pa = (void*)a;
        void* pb = (void*)b;
        void* po = (void*)out;
        unsigned int nn = (unsigned int)n;
        void* params[4];
        params[0] = (void*)&pa; params[1] = (void*)&pb; params[2] = (void*)&po; params[3] = (void*)&nn;
        return cuda_launch_(kernel, n, (&void)&params[0]);
    }
}

static int cuda_unary_raw_(int kernel, unsigned long a, float k, unsigned long out, unsigned long n) {
    unsafe {
        void* pa = (void*)a;
        void* po = (void*)out;
        unsigned int nn = (unsigned int)n;
        float kk = k;
        void* params[4];
        params[0] = (void*)&pa; params[1] = (void*)&po; params[2] = (void*)&nn; params[3] = (void*)&kk;
        return cuda_launch_(kernel, n, (&void)&params[0]);
    }
}

static int cuda_binary_(int kernel, struct CudaBuf a, struct CudaBuf b, struct CudaBuf out) {
    unsigned long n = out.len;
    if (!cuda_ensure_() || !cuda_fits_(a, n) || !cuda_fits_(b, n) || !cuda_fits_(out, n) ||
        n > 4294967295UL) return 0;
    return cuda_binary_raw_(kernel, a.dev, b.dev, out.dev, n);
}

static int cuda_unary_(int kernel, struct CudaBuf a, float k, struct CudaBuf out) {
    unsigned long n = out.len;
    if (!cuda_ensure_() || !cuda_fits_(a, n) || !cuda_fits_(out, n) || n > 4294967295UL) return 0;
    return cuda_unary_raw_(kernel, a.dev, k, out.dev, n);
}

// out1[0] = reduction over n elements; zeroes out1 first unless the grid is one block.
static int cuda_reduce_launch_(int kernel, unsigned long n, unsigned long out1, &void params) {
    unsafe {
        if (cuda_blocks_(n) > 1U &&
            cuMemsetD32Async((unsigned long long)out1, 0U, 1UL, g_cuda_stream) != CUDA_OK) return 0;
    }
    return cuda_launch_(kernel, n, params);
}

// Row-major sgemm on raw device addresses (sizes already checked).
static int cuda_gemm_raw_(unsigned long a, int ta, unsigned long b, int tb, unsigned long c,
                          unsigned long m, unsigned long n, unsigned long k) {
    float one = 1.0f;
    float zero = 0.0f;
    // Row-major C = op(A) op(B) is column-major C^T = op(B)^T op(A)^T, so
    // hand cuBLAS the operands swapped; each operand's leading dimension
    // is its stored row length.
    unsafe {
        return cublasSgemm_v2(g_cuda_blas,
                              tb ? CUDA_BLAS_T : CUDA_BLAS_N, ta ? CUDA_BLAS_T : CUDA_BLAS_N,
                              (int)n, (int)m, (int)k, &one,
                              (const float*)b, tb ? (int)k : (int)n,
                              (const float*)a, ta ? (int)m : (int)k,
                              &zero, (float*)c, (int)n) == 0;
    }
}

static int cuda_gemm_dims_ok_(unsigned long m, unsigned long n, unsigned long k) {
    return m > 0UL && n > 0UL && k > 0UL && m <= 2147483647UL && n <= 2147483647UL && k <= 2147483647UL;
}

// ── Queued ops ───────────────────────────────────────────────────────────────

int cuda_gemm(struct CudaBuf a, int ta, struct CudaBuf b, int tb, struct CudaBuf c,
              unsigned long m, unsigned long n, unsigned long k) {
    if (!cuda_ensure_() || !cuda_gemm_dims_ok_(m, n, k) ||
        !cuda_fits_(a, m * k) || !cuda_fits_(b, k * n) || !cuda_fits_(c, m * n)) return 0;
    return cuda_gemm_raw_(a.dev, ta, b.dev, tb, c.dev, m, n, k);
}

// Finds or builds the cuBLASLt plan for this shape + epilogue; -1 if none.
// 'dtype' is the element type of a, b and c (CUDA_R_32F_/16F_/16BF_); 16-bit
// types accumulate in fp32 (TF32 doesn't apply).
static int cuda_lt_plan_(int ta, int tb, unsigned long m, unsigned long n, unsigned long k, int epilogue, int dtype) {
    long key0 = (long)(ta | (tb << 1) | (g_cuda_tf32 << 2) | (epilogue << 3) | (dtype << 8));
    int slot = -1;
    int s = 0;
    while (s < LT_CACHE) {
        if (g_lt_key0[s] == key0 && g_lt_m[s] == (long)m && g_lt_n[s] == (long)n && g_lt_k[s] == (long)k)
            return s;
        if (slot < 0 && g_lt_key0[s] == -1L) slot = s;
        s = s + 1;
    }
    if (slot < 0 || g_cuda_lt == (void*)0) return -1;
    unsafe {
        // Column-major view (see cuda_gemm_raw_): D[n,m] = op(B') op(A'),
        // with B' = row-major b (stored n x k if tb, else k x n, read
        // column-major) and A' = row-major a.
        int opA = tb ? CUDA_BLAS_T : CUDA_BLAS_N;
        int opB = ta ? CUDA_BLAS_T : CUDA_BLAS_N;
        void* desc = (void*)0;
        int compute = g_cuda_tf32 && dtype == CUDA_R_32F_ ? CUDA_COMPUTE_32F_TF32 : CUDA_COMPUTE_32F;
        if (cublasLtMatmulDescCreate(&desc, compute, CUDA_R_32F_) != 0) return -1;
        cublasLtMatmulDescSetAttribute(desc, LT_DESC_TRANSA, (const void*)&opA, sizeof(int));
        cublasLtMatmulDescSetAttribute(desc, LT_DESC_TRANSB, (const void*)&opB, sizeof(int));
        cublasLtMatmulDescSetAttribute(desc, LT_DESC_EPILOGUE, (const void*)&epilogue, sizeof(int));
        void* la = (void*)0;
        void* lb = (void*)0;
        void* lc = (void*)0;
        unsigned long aRows = tb ? k : n;
        unsigned long aCols = tb ? n : k;
        unsigned long bRows = ta ? m : k;
        unsigned long bCols = ta ? k : m;
        if (cublasLtMatrixLayoutCreate(&la, dtype, (unsigned long long)aRows, (unsigned long long)aCols, (long long)aRows) != 0 ||
            cublasLtMatrixLayoutCreate(&lb, dtype, (unsigned long long)bRows, (unsigned long long)bCols, (long long)bRows) != 0 ||
            cublasLtMatrixLayoutCreate(&lc, dtype, (unsigned long long)n, (unsigned long long)m, (long long)n) != 0)
            return -1;
        void* pref = (void*)0;
        if (cublasLtMatmulPreferenceCreate(&pref) != 0) return -1;
        unsigned long long wsBytes = (unsigned long long)CUDA_WORKSPACE_BYTES;
        cublasLtMatmulPreferenceSetAttribute(pref, LT_PREF_MAX_WORKSPACE, (const void*)&wsBytes, sizeof(unsigned long long));
        int returned = 0;
        int hs = cublasLtMatmulAlgoGetHeuristic(g_cuda_lt, desc, la, lb, lc, lc, pref, 1,
                                                (void*)&g_lt_algo[slot * LT_RESULT_WORDS], &returned);
        cublasLtMatmulPreferenceDestroy(pref);
        if (hs != 0 || returned < 1) return -1;
        g_lt_desc[slot] = desc;
        g_lt_la[slot] = la;
        g_lt_lb[slot] = lb;
        g_lt_lc[slot] = lc;
        g_lt_m[slot] = (long)m;
        g_lt_n[slot] = (long)n;
        g_lt_k[slot] = (long)k;
        g_lt_key0[slot] = key0;
    }
    return slot;
}

int cuda_gemm_relu(struct CudaBuf a, int ta, struct CudaBuf b, int tb, struct CudaBuf c,
                   unsigned long m, unsigned long n, unsigned long k) {
    if (!cuda_ensure_() || !cuda_gemm_dims_ok_(m, n, k) ||
        !cuda_fits_(a, m * k) || !cuda_fits_(b, k * n) || !cuda_fits_(c, m * n)) return 0;
    int plan = cuda_lt_plan_(ta, tb, m, n, k, LT_EPILOGUE_RELU, CUDA_R_32F_);
    if (plan >= 0) {
        float one = 1.0f;
        float zero = 0.0f;
        unsafe {
            if (cublasLtMatmul(g_cuda_lt, g_lt_desc[plan], (const void*)&one,
                               (const void*)b.dev, g_lt_la[plan], (const void*)a.dev, g_lt_lb[plan],
                               (const void*)&zero, (const void*)c.dev, g_lt_lc[plan],
                               (void*)c.dev, g_lt_lc[plan],
                               (const void*)&g_lt_algo[plan * LT_RESULT_WORDS],
                               (void*)(unsigned long)g_cuda_ws, CUDA_WORKSPACE_BYTES, g_cuda_stream) == 0)
                return 1;
        }
    }
    // No cuBLASLt (or no algo for this shape): same result in two launches.
    return cuda_gemm_raw_(a.dev, ta, b.dev, tb, c.dev, m, n, k) &&
           cuda_unary_raw_(K_RELU, c.dev, 0.0f, c.dev, m * n);
}

int cuda_add(struct CudaBuf a, struct CudaBuf b, struct CudaBuf out) { return cuda_binary_(K_ADD, a, b, out); }
int cuda_sub(struct CudaBuf a, struct CudaBuf b, struct CudaBuf out) { return cuda_binary_(K_SUB, a, b, out); }
int cuda_mul(struct CudaBuf a, struct CudaBuf b, struct CudaBuf out) { return cuda_binary_(K_MUL, a, b, out); }
int cuda_div(struct CudaBuf a, struct CudaBuf b, struct CudaBuf out) { return cuda_binary_(K_DIV, a, b, out); }
int cuda_pow(struct CudaBuf a, struct CudaBuf b, struct CudaBuf out) { return cuda_binary_(K_POW, a, b, out); }
int cuda_relu_grad(struct CudaBuf z, struct CudaBuf g, struct CudaBuf out) { return cuda_binary_(K_RELU_GRAD, z, g, out); }
int cuda_scale(struct CudaBuf a, float k, struct CudaBuf out) { return cuda_unary_(K_SCALE, a, k, out); }
int cuda_relu(struct CudaBuf a, struct CudaBuf out) { return cuda_unary_(K_RELU, a, 0.0f, out); }
int cuda_exp(struct CudaBuf a, struct CudaBuf out) { return cuda_unary_(K_EXP, a, 0.0f, out); }
int cuda_log(struct CudaBuf a, struct CudaBuf out) { return cuda_unary_(K_LOG, a, 0.0f, out); }
int cuda_sqrt(struct CudaBuf a, struct CudaBuf out) { return cuda_unary_(K_SQRT, a, 0.0f, out); }

static int cuda_sgd_raw_(int kernel, unsigned long w, unsigned long g, float lr, unsigned long n) {
    unsafe {
        void* pw = (void*)w;
        void* pg = (void*)g;
        float rate = lr;
        unsigned int nn = (unsigned int)n;
        void* params[4];
        params[0] = (void*)&pw; params[1] = (void*)&pg; params[2] = (void*)&rate; params[3] = (void*)&nn;
        return cuda_launch_(kernel, n, (&void)&params[0]);
    }
}

int cuda_sgd(struct CudaBuf w, struct CudaBuf g, float lr) {
    unsigned long n = w.len;
    if (!cuda_ensure_() || !cuda_fits_(w, n) || !cuda_fits_(g, n) || n > 4294967295UL) return 0;
    return cuda_sgd_raw_(K_SGD, w.dev, g.dev, lr, n);
}

int cuda_sgd2(struct CudaBuf w1, struct CudaBuf g1, struct CudaBuf w2, struct CudaBuf g2, float lr) {
    unsigned long n1 = w1.len;
    unsigned long n2 = w2.len;
    if (!cuda_ensure_() || !cuda_fits_(w1, n1) || !cuda_fits_(g1, n1) ||
        !cuda_fits_(w2, n2) || !cuda_fits_(g2, n2) || n1 + n2 > 4294967295UL) return 0;
    unsafe {
        void* pw1 = (void*)w1.dev;
        void* pg1 = (void*)g1.dev;
        void* pw2 = (void*)w2.dev;
        void* pg2 = (void*)g2.dev;
        unsigned int nn1 = (unsigned int)n1;
        unsigned int nn2 = (unsigned int)n2;
        float rate = lr;
        void* params[7];
        params[0] = (void*)&pw1; params[1] = (void*)&pg1; params[2] = (void*)&nn1;
        params[3] = (void*)&pw2; params[4] = (void*)&pg2; params[5] = (void*)&nn2;
        params[6] = (void*)&rate;
        return cuda_launch_(K_SGD2, n1 + n2, (&void)&params[0]);
    }
}

int cuda_sum(struct CudaBuf a, struct CudaBuf out1) {
    unsigned long n = a.len;
    if (!cuda_ensure_() || !cuda_fits_(a, n) || !cuda_fits_(out1, 1UL) || n > 4294967295UL) return 0;
    unsafe {
        void* pa = (void*)a.dev;
        void* po = (void*)out1.dev;
        unsigned int nn = (unsigned int)n;
        void* params[3];
        params[0] = (void*)&pa; params[1] = (void*)&po; params[2] = (void*)&nn;
        return cuda_reduce_launch_(K_SUM, n, out1.dev, (&void)&params[0]);
    }
}

int cuda_mse_grad(struct CudaBuf y, struct CudaBuf t, struct CudaBuf dy, struct CudaBuf loss1) {
    unsigned long n = y.len;
    if (!cuda_ensure_() || !cuda_fits_(y, n) || !cuda_fits_(t, n) || !cuda_fits_(dy, n) ||
        !cuda_fits_(loss1, 1UL) || n > 4294967295UL) return 0;
    unsafe {
        void* py = (void*)y.dev;
        void* pt = (void*)t.dev;
        void* pd = (void*)dy.dev;
        void* pl = (void*)loss1.dev;
        unsigned int nn = (unsigned int)n;
        void* params[5];
        params[0] = (void*)&py; params[1] = (void*)&pt; params[2] = (void*)&pd;
        params[3] = (void*)&pl; params[4] = (void*)&nn;
        return cuda_reduce_launch_(K_MSE, n, loss1.dev, (&void)&params[0]);
    }
}

int cuda_copy(struct CudaBuf src, unsigned long srcOff, struct CudaBuf dst, unsigned long dstOff, unsigned long n) {
    if (!cuda_ensure_() || n == 0UL || !cuda_fits_(src, srcOff + n) || !cuda_fits_(dst, dstOff + n)) return 0;
    unsafe {
        return cuMemcpyDtoDAsync_v2((unsigned long long)(dst.dev + dstOff * sizeof(float)),
                                    (unsigned long long)(src.dev + srcOff * sizeof(float)),
                                    n * sizeof(float), g_cuda_stream) == CUDA_OK;
    }
}

int cuda_upsample2x_nearest(struct CudaBuf in, struct CudaBuf out,
                            unsigned long planes, unsigned long h, unsigned long w) {
    unsigned long n = planes * h * w * 4UL;
    if (!cuda_ensure_() || n == 0UL || n > 4294967295UL ||
        !cuda_fits_(in, planes * h * w) || !cuda_fits_(out, n)) return 0;
    unsafe {
        void* pi = (void*)in.dev;
        void* po = (void*)out.dev;
        unsigned int np = (unsigned int)planes;
        unsigned int nh = (unsigned int)h;
        unsigned int nw = (unsigned int)w;
        void* params[5];
        params[0] = (void*)&pi; params[1] = (void*)&po; params[2] = (void*)&np;
        params[3] = (void*)&nh; params[4] = (void*)&nw;
        return cuda_launch_(K_UPSAMPLE2X, planes * h * w, (&void)&params[0]);   // a thread per input
    }
}

int cuda_copy_rows(struct CudaBuf src, unsigned long srcOff, unsigned long srcStride,
                   struct CudaBuf dst, unsigned long dstOff, unsigned long dstStride,
                   unsigned long rowLen, unsigned long rows) {
    if (!cuda_ensure_() || rowLen == 0UL || rows == 0UL || rowLen > 4294967295UL || rows > 4294967295UL ||
        (rows > 1UL && (rowLen > srcStride || rowLen > dstStride)) ||
        !cuda_fits_(src, srcOff + (rows - 1UL) * srcStride + rowLen) ||
        !cuda_fits_(dst, dstOff + (rows - 1UL) * dstStride + rowLen)) return 0;
    // float4 when every offset, stride and the row length are multiples of 4
    // (CudaBuf bases are 256-byte aligned)
    int v4 = ((srcOff | srcStride | dstOff | dstStride | rowLen) & 3UL) == 0UL;
    unsigned long q = v4 ? 4UL : 1UL;
    unsafe {
        void* ps = (void*)src.dev;
        void* pd = (void*)dst.dev;
        unsigned long long so = (unsigned long long)(srcOff / q);
        unsigned long long ss = (unsigned long long)(srcStride / q);
        unsigned long long dof = (unsigned long long)(dstOff / q);
        unsigned long long ds = (unsigned long long)(dstStride / q);
        unsigned int len = (unsigned int)(rowLen / q);
        unsigned int nr = (unsigned int)rows;
        void* params[8];
        params[0] = (void*)&ps; params[1] = (void*)&pd; params[2] = (void*)&so; params[3] = (void*)&ss;
        params[4] = (void*)&dof; params[5] = (void*)&ds; params[6] = (void*)&len; params[7] = (void*)&nr;
        return cuda_launch_(v4 ? K_COPY_ROWS4 : K_COPY_ROWS, (rowLen / q) * rows, (&void)&params[0]);
    }
}

// ── Interop ──────────────────────────────────────────────────────────────────

?&void cuda_stream_handle() {
    if (!cuda_ensure_()) return (void*)0;
    return g_cuda_stream;
}

int cuda_tf32_enabled() { return g_cuda_tf32; }

// ── Graphs ───────────────────────────────────────────────────────────────────

int cuda_capture_begin() {
    if (!cuda_ensure_()) return 0;
    unsafe { return cuStreamBeginCapture_v2(g_cuda_stream, CUDA_CAPTURE_GLOBAL) == CUDA_OK; }
}

struct CudaGraph cuda_capture_end() {
    struct CudaGraph g;
    g.exec = 0UL;
    if (!cuda_ensure_()) return g;
    unsafe {
        void* graph = (void*)0;
        if (cuStreamEndCapture(g_cuda_stream, &graph) != CUDA_OK) return g;
        void* exec = (void*)0;
        int ok = cuGraphInstantiateWithFlags(&exec, graph, 0ULL) == CUDA_OK;
        cuGraphDestroy(graph);
        if (ok) g.exec = (unsigned long)exec;
    }
    return g;
}

int cuda_graph_launch(struct CudaGraph g) {
    if (g.exec == 0UL || !cuda_ensure_()) return 0;
    unsafe { return cuGraphLaunch((void*)g.exec, g_cuda_stream) == CUDA_OK; }
}

void cuda_graph_free(&CudaGraph g) {
    if (g.exec != 0UL) { unsafe { cuGraphExecDestroy((void*)g.exec); } }
    g.exec = 0UL;
}

// ── Host-array convenience ops ───────────────────────────────────────────────

static int cuda_host_binary_(int kernel, const float* a, const float* b, float* out, unsigned long n) {
    if (n == 0UL || n > 4294967295UL || !cuda_ensure_()) return 0;
    unsigned long bytes = checked_mul_size(n, sizeof(float));
    unsigned long long da = cuda_alloc_bytes_(bytes);
    unsigned long long db = cuda_alloc_bytes_(bytes);
    unsigned long long dout = cuda_alloc_bytes_(bytes);
    int ok = 0;
    unsafe {
        ok = da != 0ULL && db != 0ULL && dout != 0ULL &&
             cuda_write_bytes_(da, (const void*)a, bytes) && cuda_write_bytes_(db, (const void*)b, bytes) &&
             cuda_binary_raw_(kernel, (unsigned long)da, (unsigned long)db, (unsigned long)dout, n) &&
             cuda_read_bytes_((void*)out, dout, bytes);
        if (da != 0ULL) cuMemFree_v2(da);
        if (db != 0ULL) cuMemFree_v2(db);
        if (dout != 0ULL) cuMemFree_v2(dout);
    }
    return ok;
}

static int cuda_host_unary_(int kernel, const float* a, float k, float* out, unsigned long n) {
    if (n == 0UL || n > 4294967295UL || !cuda_ensure_()) return 0;
    unsigned long bytes = checked_mul_size(n, sizeof(float));
    unsigned long long da = cuda_alloc_bytes_(bytes);
    unsigned long long dout = cuda_alloc_bytes_(bytes);
    int ok = 0;
    unsafe {
        ok = da != 0ULL && dout != 0ULL && cuda_write_bytes_(da, (const void*)a, bytes) &&
             cuda_unary_raw_(kernel, (unsigned long)da, k, (unsigned long)dout, n) &&
             cuda_read_bytes_((void*)out, dout, bytes);
        if (da != 0ULL) cuMemFree_v2(da);
        if (dout != 0ULL) cuMemFree_v2(dout);
    }
    return ok;
}

int cuda_add_f32(const &float a, const &float b, &float out, unsigned long n) { return cuda_host_binary_(K_ADD, a, b, out, n); }
int cuda_sub_f32(const &float a, const &float b, &float out, unsigned long n) { return cuda_host_binary_(K_SUB, a, b, out, n); }
int cuda_mul_f32(const &float a, const &float b, &float out, unsigned long n) { return cuda_host_binary_(K_MUL, a, b, out, n); }
int cuda_div_f32(const &float a, const &float b, &float out, unsigned long n) { return cuda_host_binary_(K_DIV, a, b, out, n); }
int cuda_pow_f32(const &float a, const &float b, &float out, unsigned long n) { return cuda_host_binary_(K_POW, a, b, out, n); }
int cuda_relu_f32(const &float a, &float out, unsigned long n) { return cuda_host_unary_(K_RELU, a, 0.0f, out, n); }
int cuda_log_f32(const &float a, &float out, unsigned long n) { return cuda_host_unary_(K_LOG, a, 0.0f, out, n); }
int cuda_exp_f32(const &float a, &float out, unsigned long n) { return cuda_host_unary_(K_EXP, a, 0.0f, out, n); }
int cuda_sqrt_f32(const &float a, &float out, unsigned long n) { return cuda_host_unary_(K_SQRT, a, 0.0f, out, n); }
int cuda_scale_f32(const &float a, float k, &float out, unsigned long n) { return cuda_host_unary_(K_SCALE, a, k, out, n); }

int cuda_sum_f32(const &float a, &float out, unsigned long n) {
    if (n == 0UL || n > 4294967295UL || !cuda_ensure_()) return 0;
    unsigned long bytes = checked_mul_size(n, sizeof(float));
    unsigned long long da = cuda_alloc_bytes_(bytes);
    unsigned long long dout = cuda_alloc_bytes_(sizeof(float));
    int ok = 0;
    unsafe {
        if (da != 0ULL && dout != 0ULL && cuda_write_bytes_(da, (const void*)a, bytes)) {
            void* pa = (void*)(unsigned long)da;
            void* po = (void*)(unsigned long)dout;
            unsigned int nn = (unsigned int)n;
            void* params[3];
            params[0] = (void*)&pa; params[1] = (void*)&po; params[2] = (void*)&nn;
            ok = cuda_reduce_launch_(K_SUM, n, (unsigned long)dout, (&void)&params[0]) &&
                 cuda_read_bytes_((void*)out, dout, sizeof(float));
        }
        if (da != 0ULL) cuMemFree_v2(da);
        if (dout != 0ULL) cuMemFree_v2(dout);
    }
    return ok;
}

// Launches sgemm_nn on raw device addresses.
static int cuda_sgemm_kernel_(unsigned long a, unsigned long b, unsigned long c,
                              unsigned long M, unsigned long K, unsigned long N) {
    unsafe {
        void* pa = (void*)a;
        void* pb = (void*)b;
        void* pc = (void*)c;
        int m = (int)M;
        int n = (int)N;
        int k = (int)K;
        void* params[6];
        params[0] = (void*)&pa; params[1] = (void*)&pb; params[2] = (void*)&pc;
        params[3] = (void*)&m; params[4] = (void*)&n; params[5] = (void*)&k;
        return cuLaunchKernel(g_cuda_fn[K_SGEMM],
                              (unsigned int)((N + 63UL) / 64UL), (unsigned int)((M + 63UL) / 64UL), 1U,
                              CUDA_BLOCK, 1U, 1U, 0U, g_cuda_stream, &params[0], (void**)0) == CUDA_OK;
    }
}

// Uploads a and b, runs the custom kernel (useBlas = 0) or cuBLAS, reads out.
static int cuda_host_matmul_(const float* a, const float* b, float* out,
                             unsigned long M, unsigned long K, unsigned long N, int useBlas) {
    if (!cuda_gemm_dims_ok_(M, N, K) || !cuda_ensure_()) return 0;
    unsigned long bytesA = checked_mul_size(checked_mul_size(M, K), sizeof(float));
    unsigned long bytesB = checked_mul_size(checked_mul_size(K, N), sizeof(float));
    unsigned long bytesC = checked_mul_size(checked_mul_size(M, N), sizeof(float));
    unsigned long long da = cuda_alloc_bytes_(bytesA);
    unsigned long long db = cuda_alloc_bytes_(bytesB);
    unsigned long long dc = cuda_alloc_bytes_(bytesC);
    int ok = 0;
    unsafe {
        if (da != 0ULL && db != 0ULL && dc != 0ULL &&
            cuda_write_bytes_(da, (const void*)a, bytesA) && cuda_write_bytes_(db, (const void*)b, bytesB)) {
            if (useBlas) ok = cuda_gemm_raw_((unsigned long)da, 0, (unsigned long)db, 0, (unsigned long)dc, M, N, K);
            else ok = cuda_sgemm_kernel_((unsigned long)da, (unsigned long)db, (unsigned long)dc, M, K, N);
            ok = ok && cuda_read_bytes_((void*)out, dc, bytesC);
        }
        if (da != 0ULL) cuMemFree_v2(da);
        if (db != 0ULL) cuMemFree_v2(db);
        if (dc != 0ULL) cuMemFree_v2(dc);
    }
    return ok;
}

int cuda_matmul_f32(const &float a, const &float b, &float out,
                    unsigned long M, unsigned long K, unsigned long N) {
    return cuda_host_matmul_(a, b, out, M, K, N, 0);
}

int cuda_matmul_f32_blas(const &float a, const &float b, &float out,
                         unsigned long M, unsigned long K, unsigned long N) {
    return cuda_host_matmul_(a, b, out, M, K, N, 1);
}

// ── Raw device pointers (older API) ──────────────────────────────────────────

int cuda_persistent_available() { return cuda_ensure_(); }

?&void cuda_upload_persistent(const &float data, unsigned long bytes) {
    unsigned long long p = cuda_alloc_bytes_(bytes);
    if (p == 0ULL) return (void*)0;
    unsafe {
        if (!cuda_write_bytes_(p, (const void*)data, bytes)) { cuMemFree_v2(p); return (void*)0; }
        return (void*)(unsigned long)p;
    }
}

void cuda_release_persistent(&void devPtr) {
    if (devPtr == (void*)0 || !cuda_ensure_()) return;
    unsafe { cuMemFree_v2((unsigned long long)(unsigned long)devPtr); }
}

int cuda_matmul_f32_persistent(&void devA, &void devB, &float out,
                                unsigned long M, unsigned long K, unsigned long N) {
    if (devA == (void*)0 || devB == (void*)0 || !cuda_gemm_dims_ok_(M, N, K) || !cuda_ensure_()) return 0;
    unsigned long bytesC = checked_mul_size(checked_mul_size(M, N), sizeof(float));
    unsigned long long dc = cuda_alloc_bytes_(bytesC);
    if (dc == 0ULL) return 0;
    int ok = 0;
    unsafe {
        ok = cuda_gemm_raw_((unsigned long)devA, 0, (unsigned long)devB, 0, (unsigned long)dc, M, N, K) &&
             cuda_read_bytes_((void*)out, dc, bytesC);
        cuMemFree_v2(dc);
    }
    return ok;
}

int cuda_sgd_update_f32(&void devW, &void devGrad, float lr, unsigned long n) {
    if (devW == (void*)0 || devGrad == (void*)0 || n == 0UL || n > 4294967295UL || !cuda_ensure_()) return 0;
    unsafe { return cuda_sgd_raw_(K_SGD, (unsigned long)devW, (unsigned long)devGrad, lr, n); }
}

// ── fp16 / bf16 ──────────────────────────────────────────────────────────────

extern void* malloc(size_t size);
extern void  free(void* ptr);

static void* cuda_upload_16_(const float* data, unsigned long n, int bf) {
    if (n == 0UL) return (void*)0;
    unsafe {
        unsigned short* tmp = (unsigned short*)malloc(checked_mul_size(n, 2UL));
        if (tmp == (unsigned short*)0) return (void*)0;
        if (bf) f32_to_bf16_bulk(data, tmp, n);
        else f32_to_fp16_bulk(data, tmp, n);
        void* buf = cuda_upload_persistent((const float*)tmp, n * 2UL);
        free((void*)tmp);
        return buf;
    }
}

?&void cuda_upload_f16_persistent(const &float data, unsigned long n) { return cuda_upload_16_(data, n, 0); }
?&void cuda_upload_bf16_persistent(const &float data, unsigned long n) { return cuda_upload_16_(data, n, 1); }

static int cuda_matmul_16_(int kernel, void* devA, void* devB, unsigned short* out,
                           unsigned long M, unsigned long K, unsigned long N) {
    if (devA == (void*)0 || devB == (void*)0 || !cuda_gemm_dims_ok_(M, N, K) || !cuda_ensure_()) return 0;
    unsigned long bytesC = checked_mul_size(checked_mul_size(M, N), 2UL);
    unsigned long long dc = cuda_alloc_bytes_(bytesC);
    if (dc == 0ULL) return 0;
    int ok = 0;
    unsafe {
        void* pa = devA;
        void* pb = devB;
        void* pc = (void*)(unsigned long)dc;
        int m = (int)M;
        int n = (int)N;
        int k = (int)K;
        void* params[6];
        params[0] = (void*)&pa; params[1] = (void*)&pb; params[2] = (void*)&pc;
        params[3] = (void*)&m; params[4] = (void*)&n; params[5] = (void*)&k;
        ok = cuLaunchKernel(g_cuda_fn[kernel],
                            (unsigned int)((N + 15UL) / 16UL), (unsigned int)((M + 15UL) / 16UL), 1U,
                            16U, 16U, 1U, 0U, g_cuda_stream, &params[0], (void**)0) == CUDA_OK &&
             cuda_read_bytes_((void*)out, dc, bytesC);
        cuMemFree_v2(dc);
    }
    return ok;
}

int cuda_matmul_f16_persistent(&void devA, &void devB, &unsigned short out,
                                unsigned long M, unsigned long K, unsigned long N) {
    return cuda_matmul_16_(K_MATMUL_F16, devA, devB, out, M, K, N);
}

int cuda_matmul_bf16_persistent(&void devA, &void devB, &unsigned short out,
                                 unsigned long M, unsigned long K, unsigned long N) {
    return cuda_matmul_16_(K_MATMUL_BF16, devA, devB, out, M, K, N);
}

int cuda_sgd_update_f16(&void devW, &void devGrad, float lr, unsigned long n) {
    if (devW == (void*)0 || devGrad == (void*)0 || n == 0UL || n > 4294967295UL || !cuda_ensure_()) return 0;
    unsafe { return cuda_sgd_raw_(K_SGD_F16, (unsigned long)devW, (unsigned long)devGrad, lr, n); }
}

int cuda_sgd_update_bf16(&void devW, &void devGrad, float lr, unsigned long n) {
    if (devW == (void*)0 || devGrad == (void*)0 || n == 0UL || n > 4294967295UL || !cuda_ensure_()) return 0;
    unsafe { return cuda_sgd_raw_(K_SGD_BF16, (unsigned long)devW, (unsigned long)devGrad, lr, n); }
}

// ── fp16 / bf16 buffers (CudaHalfBuf) ────────────────────────────────────────

static int cuda_half_ok_(struct CudaHalfBuf b, unsigned long n) {
    return b.dev != 0UL && b.len >= n && (b.dtype == CUDA_FP16 || b.dtype == CUDA_BF16);
}

// Kernel index for a CudaHalfBuf op: the fp16 kernel or its bf16 twin.
static int cuda_k16_(int fp16Kernel, int dtype) { return dtype == CUDA_BF16 ? fp16Kernel + 1 : fp16Kernel; }

static int cuda_dtype_type_(int dtype) { return dtype == CUDA_BF16 ? CUDA_R_16BF_ : CUDA_R_16F_; }

struct CudaHalfBuf cuda_half_alloc(unsigned long n, int dtype) {
    struct CudaHalfBuf b;
    b.dev = 0UL;
    b.len = 0UL;
    b.dtype = dtype;
    if (n == 0UL || (dtype != CUDA_FP16 && dtype != CUDA_BF16)) return b;
    unsigned long long p = cuda_alloc_bytes_(checked_mul_size(n, 2UL));
    if (p == 0ULL) return b;
    b.dev = (unsigned long)p;
    b.len = n;
    return b;
}

struct CudaHalfBuf cuda_half_upload(const &float host, unsigned long n, int dtype) {
    struct CudaHalfBuf b = cuda_half_alloc(n, dtype);
    if (b.dev == 0UL) return b;
    struct CudaBuf tmp = cuda_upload(host, n);
    int ok = tmp.dev != 0UL && cuda_to_half(tmp, b);
    if (ok) ok = cuda_sync();   // tmp is freed next
    cuda_free(&tmp);
    if (!ok) cuda_half_free(&b);
    return b;
}

int cuda_half_write(struct CudaHalfBuf dst, const &unsigned short bits, unsigned long n) {
    if (!cuda_half_ok_(dst, n) || n == 0UL || !cuda_ensure_()) return 0;
    unsafe { return cuda_write_bytes_((unsigned long long)dst.dev, (const void*)bits, n * 2UL); }
}

int cuda_half_read(struct CudaHalfBuf src, &unsigned short bits, unsigned long n) {
    if (!cuda_half_ok_(src, n) || n == 0UL || !cuda_ensure_()) return 0;
    unsafe { return cuda_read_bytes_((void*)bits, (unsigned long long)src.dev, n * 2UL); }
}

void cuda_half_free(&CudaHalfBuf b) {
    if (b.dev != 0UL) { unsafe { cuMemFree_v2((unsigned long long)b.dev); } }
    b.dev = 0UL;
    b.len = 0UL;
}

// Launches a two-pointer CudaHalfBuf kernel over n elements: (a, out, n).
static int cuda_launch2_(int kernel, unsigned long a, unsigned long out, unsigned long n) {
    unsafe {
        void* pa = (void*)a;
        void* po = (void*)out;
        unsigned int nn = (unsigned int)n;
        void* params[3];
        params[0] = (void*)&pa; params[1] = (void*)&po; params[2] = (void*)&nn;
        return cuda_launch_(kernel, n, (&void)&params[0]);
    }
}

int cuda_to_half(struct CudaBuf src, struct CudaHalfBuf dst) {
    unsigned long n = dst.len;
    if (!cuda_ensure_() || !cuda_half_ok_(dst, n) || !cuda_fits_(src, n) || n > 4294967295UL) return 0;
    return cuda_launch2_(cuda_k16_(K_TO_F16, dst.dtype), src.dev, dst.dev, n);
}

int cuda_from_half(struct CudaHalfBuf src, struct CudaBuf dst) {
    unsigned long n = src.len;
    if (!cuda_ensure_() || !cuda_half_ok_(src, n) || !cuda_fits_(dst, n) || n > 4294967295UL) return 0;
    return cuda_launch2_(cuda_k16_(K_FROM_F16, src.dtype), src.dev, dst.dev, n);
}

// c = op(a) op(b) through cublasGemmEx: a, b of 'dtype', c of cType, fp32
// accumulation (row-major via the column-major swap of cuda_gemm_raw_).
static int cuda_gemm_ex_(unsigned long a, int ta, unsigned long b, int tb, unsigned long c, int abType, int cType,
                         unsigned long m, unsigned long n, unsigned long k) {
    float one = 1.0f;
    float zero = 0.0f;
    unsafe {
        return cublasGemmEx(g_cuda_blas, tb ? CUDA_BLAS_T : CUDA_BLAS_N, ta ? CUDA_BLAS_T : CUDA_BLAS_N,
                            (int)n, (int)m, (int)k, (const void*)&one,
                            (const void*)b, abType, tb ? (int)k : (int)n,
                            (const void*)a, abType, ta ? (int)m : (int)k,
                            (const void*)&zero, (void*)c, cType, (int)n,
                            CUDA_COMPUTE_32F, CUDA_GEMM_DEFAULT_) == 0;
    }
}

static int cuda_half_gemm_ok_(struct CudaHalfBuf a, struct CudaHalfBuf b, unsigned long m, unsigned long n,
                              unsigned long k) {
    return cuda_ensure_() && cuda_gemm_dims_ok_(m, n, k) && a.dtype == b.dtype &&
           cuda_half_ok_(a, m * k) && cuda_half_ok_(b, k * n);
}

int cuda_half_gemm(struct CudaHalfBuf a, int ta, struct CudaHalfBuf b, int tb, struct CudaHalfBuf c,
                   unsigned long m, unsigned long n, unsigned long k) {
    if (!cuda_half_gemm_ok_(a, b, m, n, k) || c.dtype != a.dtype || !cuda_half_ok_(c, m * n)) return 0;
    int t = cuda_dtype_type_(a.dtype);
    return cuda_gemm_ex_(a.dev, ta, b.dev, tb, c.dev, t, t, m, n, k);
}

int cuda_half_gemm_f32(struct CudaHalfBuf a, int ta, struct CudaHalfBuf b, int tb, struct CudaBuf c,
                       unsigned long m, unsigned long n, unsigned long k) {
    if (!cuda_half_gemm_ok_(a, b, m, n, k) || !cuda_fits_(c, m * n)) return 0;
    return cuda_gemm_ex_(a.dev, ta, b.dev, tb, c.dev, cuda_dtype_type_(a.dtype), CUDA_R_32F_, m, n, k);
}

int cuda_half_gemm_relu(struct CudaHalfBuf a, int ta, struct CudaHalfBuf b, int tb, struct CudaHalfBuf c,
                        unsigned long m, unsigned long n, unsigned long k) {
    if (!cuda_half_gemm_ok_(a, b, m, n, k) || c.dtype != a.dtype || !cuda_half_ok_(c, m * n)) return 0;
    int t = cuda_dtype_type_(a.dtype);
    int plan = cuda_lt_plan_(ta, tb, m, n, k, LT_EPILOGUE_RELU, t);
    if (plan >= 0) {
        float one = 1.0f;
        float zero = 0.0f;
        unsafe {
            if (cublasLtMatmul(g_cuda_lt, g_lt_desc[plan], (const void*)&one,
                               (const void*)b.dev, g_lt_la[plan], (const void*)a.dev, g_lt_lb[plan],
                               (const void*)&zero, (const void*)c.dev, g_lt_lc[plan],
                               (void*)c.dev, g_lt_lc[plan],
                               (const void*)&g_lt_algo[plan * LT_RESULT_WORDS],
                               (void*)(unsigned long)g_cuda_ws, CUDA_WORKSPACE_BYTES, g_cuda_stream) == 0)
                return 1;
        }
    }
    return cuda_gemm_ex_(a.dev, ta, b.dev, tb, c.dev, t, t, m, n, k) &&
           cuda_launch2_(cuda_k16_(K_RELU_F16, a.dtype), c.dev, c.dev, m * n);
}

int cuda_half_relu(struct CudaHalfBuf a, struct CudaHalfBuf out) {
    unsigned long n = a.len;
    if (!cuda_ensure_() || !cuda_half_ok_(a, n) || !cuda_half_ok_(out, n) || a.dtype != out.dtype ||
        n > 4294967295UL) return 0;
    return cuda_launch2_(cuda_k16_(K_RELU_F16, a.dtype), a.dev, out.dev, n);
}

int cuda_half_relu_grad(struct CudaHalfBuf z, struct CudaHalfBuf g, struct CudaHalfBuf out) {
    unsigned long n = z.len;
    if (!cuda_ensure_() || !cuda_half_ok_(z, n) || !cuda_half_ok_(g, n) || !cuda_half_ok_(out, n) ||
        z.dtype != g.dtype || z.dtype != out.dtype || n > 4294967295UL) return 0;
    unsafe {
        void* pz = (void*)z.dev;
        void* pg = (void*)g.dev;
        void* po = (void*)out.dev;
        unsigned int nn = (unsigned int)n;
        void* params[4];
        params[0] = (void*)&pz; params[1] = (void*)&pg; params[2] = (void*)&po; params[3] = (void*)&nn;
        return cuda_launch_(cuda_k16_(K_RELUG_F16, z.dtype), n, (&void)&params[0]);
    }
}

int cuda_half_mse_grad(struct CudaHalfBuf y, struct CudaHalfBuf t, struct CudaHalfBuf dy, struct CudaBuf loss1) {
    unsigned long n = y.len;
    if (!cuda_ensure_() || !cuda_half_ok_(y, n) || !cuda_half_ok_(t, n) || !cuda_half_ok_(dy, n) ||
        y.dtype != t.dtype || y.dtype != dy.dtype || !cuda_fits_(loss1, 1UL) || n > 4294967295UL) return 0;
    unsafe {
        void* py = (void*)y.dev;
        void* pt = (void*)t.dev;
        void* pd = (void*)dy.dev;
        void* pl = (void*)loss1.dev;
        unsigned int nn = (unsigned int)n;
        void* params[5];
        params[0] = (void*)&py; params[1] = (void*)&pt; params[2] = (void*)&pd;
        params[3] = (void*)&pl; params[4] = (void*)&nn;
        return cuda_reduce_launch_(cuda_k16_(K_MSE_F16, y.dtype), n, loss1.dev, (&void)&params[0]);
    }
}

int cuda_half_sgd(struct CudaHalfBuf w, struct CudaHalfBuf g, float lr) {
    unsigned long n = w.len;
    if (!cuda_ensure_() || !cuda_half_ok_(w, n) || !cuda_half_ok_(g, n) || w.dtype != g.dtype ||
        n > 4294967295UL) return 0;
    return cuda_sgd_raw_(w.dtype == CUDA_BF16 ? K_SGD_BF16 : K_SGD_F16, w.dev, g.dev, lr, n);
}

int cuda_half_sgd2(struct CudaHalfBuf w1, struct CudaHalfBuf g1, struct CudaHalfBuf w2, struct CudaHalfBuf g2,
                   float lr) {
    unsigned long n1 = w1.len;
    unsigned long n2 = w2.len;
    if (!cuda_ensure_() || !cuda_half_ok_(w1, n1) || !cuda_half_ok_(g1, n1) || !cuda_half_ok_(w2, n2) ||
        !cuda_half_ok_(g2, n2) || w1.dtype != g1.dtype || w1.dtype != w2.dtype || w1.dtype != g2.dtype ||
        n1 + n2 > 4294967295UL) return 0;
    unsafe {
        void* pw1 = (void*)w1.dev;
        void* pg1 = (void*)g1.dev;
        void* pw2 = (void*)w2.dev;
        void* pg2 = (void*)g2.dev;
        unsigned int nn1 = (unsigned int)n1;
        unsigned int nn2 = (unsigned int)n2;
        float rate = lr;
        void* params[7];
        params[0] = (void*)&pw1; params[1] = (void*)&pg1; params[2] = (void*)&nn1;
        params[3] = (void*)&pw2; params[4] = (void*)&pg2; params[5] = (void*)&nn2;
        params[6] = (void*)&rate;
        return cuda_launch_(cuda_k16_(K_SGD2_F16, w1.dtype), n1 + n2, (&void)&params[0]);
    }
}

} // namespace std
