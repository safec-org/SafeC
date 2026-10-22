// SafeC Standard Library — the CUDA device backend (see device.h).
// CudaBackend: device memory from cuMemAlloc, copies and kernels queued on
// gpu_cuda.h's stream (ordered with every other gpu_cuda op, cuDNN and
// cuFFT), GEMMs through cuBLAS, and device_kernels.cu -- the same source as
// the ROCm backend's -- compiled once by NVRTC for the GPU's compute
// capability. A program enables it with its cuda feature; link -lcuda
// -lcublas -lnvrtc (and -lcudnn for cuDNN's convolutions).
#pragma once
#include <std/stddef.h>
#include <std/ml/device.h>
#include <std/ml/device_kernels.h>
#include <std/ml/gpu_cuda.h>

namespace std {

extern int cuCtxGetDevice(int* device);
extern int cuDeviceGetAttribute(int* value, int attr, int device);
extern int cuStreamSynchronize(void* stream);
extern int cuMemAlloc_v2(unsigned long long* dptr, size_t bytes);
extern int cuMemFree_v2(unsigned long long dptr);
extern int cuMemcpyHtoDAsync_v2(unsigned long long dst, const void* src, size_t bytes, void* stream);
extern int cuMemcpyDtoHAsync_v2(void* dst, unsigned long long src, size_t bytes, void* stream);
extern int cuMemcpyDtoDAsync_v2(unsigned long long dst, unsigned long long src, size_t bytes, void* stream);
extern int cuModuleLoadData(void** module, const void* image);
extern int cuModuleGetFunction(void** func, void* module, const char* name);
extern int cuLaunchKernel(void* f, unsigned int gx, unsigned int gy, unsigned int gz,
                          unsigned int bx, unsigned int by, unsigned int bz,
                          unsigned int shared, void* stream, void** params, void** extra);
extern int nvrtcCreateProgram(void** prog, const char* src, const char* name, int numHeaders,
                              const char** headers, const char** includeNames);
extern int nvrtcCompileProgram(void* prog, int numOptions, const char** options);
extern int nvrtcGetCUBINSize(void* prog, size_t* size);
extern int nvrtcGetCUBIN(void* prog, char* cubin);
extern int nvrtcDestroyProgram(void** prog);
extern int cublasCreate_v2(void** handle);
extern int cublasSetStream_v2(void* handle, void* stream);
extern int cublasSgemm_v2(void* handle, int transa, int transb, int m, int n, int k,
                          const float* alpha, const float* a, int lda, const float* b, int ldb,
                          const float* beta, float* c, int ldc);
extern int cublasSgemmStridedBatched(void* handle, int transa, int transb, int m, int n, int k,
                                     const float* alpha, const float* a, int lda, long long sa,
                                     const float* b, int ldb, long long sb, const float* beta,
                                     float* c, int ldc, long long sc, int batch);
extern int cublasGemmStridedBatchedEx(void* handle, int transa, int transb, int m, int n, int k, const void* alpha,
                                      const void* a, int aType, int lda, long long sa,
                                      const void* b, int bType, int ldb, long long sb, const void* beta,
                                      void* c, int cType, int ldc, long long sc, int batch, int computeType, int algo);
#define DCUDA_R_32F  0    // CUDA_R_32F
#define DCUDA_R_16F  2    // CUDA_R_16F
#define DCUDA_R_16BF 14   // CUDA_R_16BF
#define DCUDA_COMPUTE_32F 68   // CUBLAS_COMPUTE_32F
#define DCUDA_GEMM_DEFAULT -1
extern void* malloc(size_t size);
extern void  free(void* ptr);

#define DCUDA_ATTR_CC_MAJOR 75   // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
#define DCUDA_ATTR_CC_MINOR 76
#define DCUDA_N 0                // CUBLAS_OP_N
#define DCUDA_T 1                // CUBLAS_OP_T
#define DCUDA_BLOCK 256UL
#define DCUDA_MAX_BLOCKS 4096UL

static int   g_dcuda_state = 0;
static void* g_dcuda_stream = (void*)0;
static void* g_dcuda_blas = (void*)0;
static void* g_dcuda_fn[DK_COUNT];
static void* g_dcuda_fn16[DK_COUNT];   // the DK_TAGGED build (16-bit arguments), compiled on first use
static int   g_dcuda_state16 = 0;
static char  g_dcuda_arch[32];        // -arch=sm_XY

// Compiles device_kernels.cu (with DK_TAGGED when tagged) into fns.
static int dcuda_module_(int tagged, &void fns) {
    unsafe {
        void* prog = (void*)0;
        if (nvrtcCreateProgram(&prog, device_kernel_source_(), "safec_device.cu", 0,
                               (const char**)0, (const char**)0) != 0) return 0;
        const char* opts[3];
        opts[0] = (const char*)&g_dcuda_arch[0];
        opts[1] = "-std=c++17";
        opts[2] = "-DDK_TAGGED";
        if (nvrtcCompileProgram(prog, tagged ? 3 : 2, &opts[0]) != 0) { nvrtcDestroyProgram(&prog); return 0; }
        size_t size = 0UL;
        nvrtcGetCUBINSize(prog, &size);
        char* cubin = (char*)malloc((size_t)((unsigned long)size + 1UL));
        if (cubin == (char*)0) { nvrtcDestroyProgram(&prog); return 0; }
        nvrtcGetCUBIN(prog, cubin);
        nvrtcDestroyProgram(&prog);
        void* module = (void*)0;
        int loaded = cuModuleLoadData(&module, (const void*)cubin);
        free((void*)cubin);
        if (loaded != 0) return 0;
        void** f = (void**)fns;
        int k = 0;
        while (k < DK_COUNT) {
            if (cuModuleGetFunction(&f[k], module, device_kernel_name_(k)) != 0) return 0;
            k = k + 1;
        }
    }
    return 1;
}

static int dcuda_ensure_() {
    if (g_dcuda_state != 0) {
        if (g_dcuda_state == 1) cuda_available();   // the context current on this thread
        return g_dcuda_state == 1;
    }
    g_dcuda_state = -1;
    if (!cuda_available()) return 0;
    unsafe {
        match (cuda_stream_handle()) {
            case some(s): { g_dcuda_stream = (void*)s; }
            case null: { return 0; }
        }
        if (cublasCreate_v2(&g_dcuda_blas) != 0) return 0;
        cublasSetStream_v2(g_dcuda_blas, g_dcuda_stream);
        int dev = 0;
        int major = 0;
        int minor = 0;
        if (cuCtxGetDevice(&dev) != 0) return 0;
        cuDeviceGetAttribute(&major, DCUDA_ATTR_CC_MAJOR, dev);
        cuDeviceGetAttribute(&minor, DCUDA_ATTR_CC_MINOR, dev);
        int cc = major * 10 + minor;
        char* arch = &g_dcuda_arch[0];
        const char* prefix = "-arch=sm_";
        int p = 0;
        while (prefix[p] != (char)0) { arch[p] = prefix[p]; p = p + 1; }
        if (cc >= 100) { arch[p] = (char)(48 + cc / 100); p = p + 1; }
        arch[p] = (char)(48 + (cc / 10) % 10); p = p + 1;
        arch[p] = (char)(48 + cc % 10); p = p + 1;
        arch[p] = (char)0;
        if (!dcuda_module_(0, (&void)(void*)&g_dcuda_fn[0])) return 0;
    }
    g_dcuda_state = 1;
    return 1;
}

// ── Convolutions through cuDNN (optional) ────────────────────────────────────
// Weak references: a program linked with -lcudnn gets cuDNN's convolutions;
// without it CudaBackend::conv declines (im2col + GEMM). cuDNN has no
// transposed mode, so a transposed convolution runs as the convolution it
// is the adjoint of: its forward is that convolution's backward-data, and so
// on. Each shape gets a plan; each of its three directions the algorithm
// cudnnFind timed fastest, the first time that direction ran.
extern int cudnnCreate(void** handle) __attribute__((weak));
extern int cudnnSetStream(void* handle, void* stream) __attribute__((weak));
extern int cudnnCreateTensorDescriptor(void** desc) __attribute__((weak));
extern int cudnnSetTensor4dDescriptor(void* desc, int format, int dataType, int n, int c, int h, int w) __attribute__((weak));
extern int cudnnCreateFilterDescriptor(void** desc) __attribute__((weak));
extern int cudnnSetFilter4dDescriptor(void* desc, int dataType, int format, int k, int c, int h, int w) __attribute__((weak));
extern int cudnnCreateConvolutionDescriptor(void** desc) __attribute__((weak));
extern int cudnnSetConvolution2dDescriptor(void* desc, int padH, int padW, int u, int v, int dilationH, int dilationW,
                                           int mode, int computeType) __attribute__((weak));
extern int cudnnSetConvolutionMathType(void* desc, int mathType) __attribute__((weak));
extern int cudnnFindConvolutionForwardAlgorithm(void* handle, void* xDesc, void* wDesc, void* convDesc, void* yDesc,
                                                int requested, int* returned, void* perf) __attribute__((weak));
extern int cudnnFindConvolutionBackwardDataAlgorithm(void* handle, void* wDesc, void* dyDesc, void* convDesc,
                                                     void* dxDesc, int requested, int* returned, void* perf)
    __attribute__((weak));
extern int cudnnFindConvolutionBackwardFilterAlgorithm(void* handle, void* xDesc, void* dyDesc, void* convDesc,
                                                       void* dwDesc, int requested, int* returned, void* perf)
    __attribute__((weak));
extern int cudnnConvolutionForward(void* handle, const void* alpha, void* xDesc, const void* x, void* wDesc,
                                   const void* w, void* convDesc, int algo, void* ws, size_t wsBytes, const void* beta,
                                   void* yDesc, void* y) __attribute__((weak));
extern int cudnnConvolutionBackwardData(void* handle, const void* alpha, void* wDesc, const void* w, void* dyDesc,
                                        const void* dy, void* convDesc, int algo, void* ws, size_t wsBytes,
                                        const void* beta, void* dxDesc, void* dx) __attribute__((weak));
extern int cudnnConvolutionBackwardFilter(void* handle, const void* alpha, void* xDesc, const void* x, void* dyDesc,
                                          const void* dy, void* convDesc, int algo, void* ws, size_t wsBytes,
                                          const void* beta, void* dwDesc, void* dw) __attribute__((weak));

#define DCDNN_NCHW      0      // CUDNN_TENSOR_NCHW
#define DCDNN_FLOAT     0      // CUDNN_DATA_FLOAT
#define DCDNN_XCORR     1      // CUDNN_CROSS_CORRELATION
#define DCDNN_DEFAULT_MATH 0   // CUDNN_DEFAULT_MATH (fp32 may use TF32 tensor cores)
#define DCDNN_FMA_MATH  3      // CUDNN_FMA_MATH
#define DCDNN_PERF_INTS 12     // sizeof(cudnnConvolution*AlgoPerf_t) / 4
#define DCDNN_PERF_COUNT 8
#define DCDNN_MAX_WS    268435456UL
#define DCDNN_PLANS     256
#define DCDNN_KEY       17
// the convolution-side directions
#define DCDNN_F  0
#define DCDNN_BD 1
#define DCDNN_BF 2

static int   g_dcdnn_state = 0;
static void* g_dcdnn = (void*)0;
static unsigned long long g_dcdnn_ws = 0ULL;
static unsigned long g_dcdnn_wsBytes = 0UL;
static unsigned long g_dcdnn_key[DCDNN_PLANS * DCDNN_KEY];
static void* g_dcdnn_desc[DCDNN_PLANS * 4];   // x, y, w, conv (the convolution's sides)
static int   g_dcdnn_algo[DCDNN_PLANS * 3];   // per direction; -1 not found yet, -2 can't
static int   g_dcdnn_math[DCDNN_PLANS * 3];
static int   g_dcdnn_good[DCDNN_PLANS];
static int   g_dcdnn_count = 0;

static int dcdnn_on_() {
    if (g_dcdnn_state == 0) {
        g_dcdnn_state = -1;
        unsafe {
            if (dcuda_ensure_() && (void*)cudnnCreate != (void*)0 && cudnnCreate(&g_dcdnn) == 0 &&
                cudnnSetStream(g_dcdnn, g_dcuda_stream) == 0) g_dcdnn_state = 1;
        }
    }
    return g_dcdnn_state == 1;
}

static int dcdnn_ws_grow_(unsigned long bytes) {
    if (bytes <= g_dcdnn_wsBytes) return 1;
    unsafe {
        if (g_dcdnn_ws != 0ULL) { cuStreamSynchronize(g_dcuda_stream); cuMemFree_v2(g_dcdnn_ws); }
        g_dcdnn_ws = 0ULL;
        g_dcdnn_wsBytes = 0UL;
        if (cuMemAlloc_v2(&g_dcdnn_ws, bytes) != 0) { g_dcdnn_ws = 0ULL; return 0; }
        g_dcdnn_wsBytes = bytes;
    }
    return 1;
}

static int dcdnn_plan_(const &DevConv c) {
    unsigned long key[DCDNN_KEY];
    unsafe {
        key[0] = (unsigned long)c.transpose; key[1] = c.batch; key[2] = c.inC; key[3] = c.inH; key[4] = c.inW;
        key[5] = c.outC; key[6] = c.outH; key[7] = c.outW; key[8] = c.kH; key[9] = c.kW; key[10] = c.sH;
        key[11] = c.sW; key[12] = c.pH; key[13] = c.pW; key[14] = c.dH; key[15] = c.dW; key[16] = (unsigned long)c.dtype;
    }
    int i = 0;
    while (i < g_dcdnn_count) {
        int same = 1;
        int j = 0;
        while (same && j < DCDNN_KEY) { if (g_dcdnn_key[i * DCDNN_KEY + j] != key[j]) same = 0; j = j + 1; }
        if (same) return g_dcdnn_good[i] ? i : -1;
        i = i + 1;
    }
    if (g_dcdnn_count >= DCDNN_PLANS) return -1;
    i = g_dcdnn_count;
    g_dcdnn_count = g_dcdnn_count + 1;
    int j = 0;
    while (j < DCDNN_KEY) { g_dcdnn_key[i * DCDNN_KEY + j] = key[j]; j = j + 1; }
    g_dcdnn_algo[i * 3] = -1; g_dcdnn_algo[i * 3 + 1] = -1; g_dcdnn_algo[i * 3 + 2] = -1;
    // the convolution's sides: a transposed convolution's output is its input
    unsigned long xC = c.transpose ? c.outC : c.inC;
    unsigned long xH = c.transpose ? c.outH : c.inH;
    unsigned long xW = c.transpose ? c.outW : c.inW;
    unsigned long yC = c.transpose ? c.inC : c.outC;
    unsigned long yH = c.transpose ? c.inH : c.outH;
    unsigned long yW = c.transpose ? c.inW : c.outW;
    int ok = 1;
    unsafe {
        void* xd = (void*)0; void* yd = (void*)0; void* wd = (void*)0; void* cd = (void*)0;
        ok = cudnnCreateTensorDescriptor(&xd) == 0 && cudnnCreateTensorDescriptor(&yd) == 0 &&
             cudnnCreateFilterDescriptor(&wd) == 0 && cudnnCreateConvolutionDescriptor(&cd) == 0;
        int dt = c.dtype == DEVICE_FP16 ? 2 : (c.dtype == DEVICE_BF16 ? 9 : DCDNN_FLOAT);   // CUDNN_DATA_HALF / BFLOAT16
        ok = ok && cudnnSetTensor4dDescriptor(xd, DCDNN_NCHW, dt, (int)c.batch, (int)xC, (int)xH, (int)xW) == 0 &&
             cudnnSetTensor4dDescriptor(yd, DCDNN_NCHW, dt, (int)c.batch, (int)yC, (int)yH, (int)yW) == 0 &&
             cudnnSetFilter4dDescriptor(wd, dt, DCDNN_NCHW, (int)yC, (int)xC, (int)c.kH, (int)c.kW) == 0 &&
             cudnnSetConvolution2dDescriptor(cd, (int)c.pH, (int)c.pW, (int)c.sH, (int)c.sW, (int)c.dH, (int)c.dW,
                                             DCDNN_XCORR, DCDNN_FLOAT) == 0;
        g_dcdnn_desc[i * 4] = xd; g_dcdnn_desc[i * 4 + 1] = yd; g_dcdnn_desc[i * 4 + 2] = wd; g_dcdnn_desc[i * 4 + 3] = cd;
    }
    g_dcdnn_good[i] = ok;
    return ok ? i : -1;
}

// One convolution-side direction: F y = conv(x, w); BD x = from dy; BF w = from x and dy.
static int dcdnn_dir_(int plan, int dir, unsigned long x, unsigned long w, unsigned long y) {
    float one = (float)1.0;
    float zero = (float)0.0;
    unsafe {
        void* xd = g_dcdnn_desc[plan * 4]; void* yd = g_dcdnn_desc[plan * 4 + 1];
        void* wd = g_dcdnn_desc[plan * 4 + 2]; void* cd = g_dcdnn_desc[plan * 4 + 3];
        int slot = plan * 3 + dir;
        if (g_dcdnn_algo[slot] == -2) return 0;
        if (g_dcdnn_algo[slot] == -1) {
            // 16-bit data takes tensor cores; fp32 uses TF32 only when gpu_cuda.h allows it
            int math = (g_dcdnn_key[plan * DCDNN_KEY + 16] != 0UL || cuda_tf32_enabled()) ? DCDNN_DEFAULT_MATH : DCDNN_FMA_MATH;
            int perf[DCDNN_PERF_INTS * DCDNN_PERF_COUNT];
            int got = 0;
            int ok = cudnnSetConvolutionMathType(cd, math) == 0;
            if (ok && dir == DCDNN_F) ok = cudnnFindConvolutionForwardAlgorithm(g_dcdnn, xd, wd, cd, yd, DCDNN_PERF_COUNT, &got, (void*)&perf[0]) == 0;
            else if (ok && dir == DCDNN_BD) ok = cudnnFindConvolutionBackwardDataAlgorithm(g_dcdnn, wd, yd, cd, xd, DCDNN_PERF_COUNT, &got, (void*)&perf[0]) == 0;
            else if (ok) ok = cudnnFindConvolutionBackwardFilterAlgorithm(g_dcdnn, xd, yd, cd, wd, DCDNN_PERF_COUNT, &got, (void*)&perf[0]) == 0;
            int algo = -1;
            int r = 0;
            while (ok && algo < 0 && r < got) {
                int* p = (int*)&perf[r * DCDNN_PERF_INTS];
                unsigned long mem = *(unsigned long*)(void*)&p[4];
                if (p[1] == 0 && mem <= DCDNN_MAX_WS && dcdnn_ws_grow_(mem == 0UL ? 4UL : mem)) {
                    algo = p[0];
                    g_dcdnn_math[slot] = p[7];
                }
                r = r + 1;
            }
            if (algo < 0) { g_dcdnn_algo[slot] = -2; return 0; }
            g_dcdnn_algo[slot] = algo;
        }
        if (cudnnSetConvolutionMathType(cd, g_dcdnn_math[slot]) != 0) return 0;
        int algo = g_dcdnn_algo[slot];
        void* ws = (void*)g_dcdnn_ws;
        if (dir == DCDNN_F)
            return cudnnConvolutionForward(g_dcdnn, (const void*)&one, xd, (const void*)x, wd, (const void*)w, cd, algo, ws,
                                           g_dcdnn_wsBytes, (const void*)&zero, yd, (void*)y) == 0;
        if (dir == DCDNN_BD)
            return cudnnConvolutionBackwardData(g_dcdnn, (const void*)&one, wd, (const void*)w, yd, (const void*)y, cd, algo,
                                                ws, g_dcdnn_wsBytes, (const void*)&zero, xd, (void*)x) == 0;
        return cudnnConvolutionBackwardFilter(g_dcdnn, (const void*)&one, xd, (const void*)x, yd, (const void*)y, cd, algo,
                                              ws, g_dcdnn_wsBytes, (const void*)&zero, wd, (void*)w) == 0;
    }
}

static int dcdnn_run_(const &DevConv c) {
    if (!dcdnn_on_()) return 0;
    int plan = dcdnn_plan_(c);
    if (plan < 0) return 0;
    if (!c.transpose) {
        if (c.op == DEV_CONV_FWD) return dcdnn_dir_(plan, DCDNN_F, c.x, c.w, c.y);
        if (c.op == DEV_CONV_BWD_DATA) return dcdnn_dir_(plan, DCDNN_BD, c.x, c.w, c.y);
        return dcdnn_dir_(plan, DCDNN_BF, c.x, c.w, c.y);
    }
    // transposed: the layer's input is the convolution's y side, its output the x side
    if (c.op == DEV_CONV_FWD) return dcdnn_dir_(plan, DCDNN_BD, c.y, c.w, c.x);
    if (c.op == DEV_CONV_BWD_DATA) return dcdnn_dir_(plan, DCDNN_F, c.y, c.w, c.x);
    return dcdnn_dir_(plan, DCDNN_BF, c.y, c.w, c.x);
}

struct CudaBackend {
    int unused_;

    static int available();
    static unsigned long alloc(unsigned long bytes);
    static void release(unsigned long handle);
    static int upload(unsigned long dst, const &float src, unsigned long n);
    static int download(&float dst, unsigned long src, unsigned long n);
    static int copy(unsigned long dst, unsigned long src, unsigned long n);
    static int sync();
    static int gemm(int ta, int tb, unsigned long m, unsigned long n, unsigned long k, float alpha,
                    unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb,
                    float beta, unsigned long c, unsigned long ldc);
    static int gemm_batched(const &DevGemm g);
    static int conv(const &DevConv c);
    static int storage16();
    static int launch(int kernel, unsigned long n, const &DevArgs args);
    static unsigned long offset(unsigned long handle, unsigned long bytes);
};

int CudaBackend::available() { return dcuda_ensure_(); }

unsigned long CudaBackend::alloc(unsigned long bytes) {
    if (!dcuda_ensure_()) return 0UL;
    unsigned long long p = 0ULL;
    unsafe { if (cuMemAlloc_v2(&p, bytes) != 0) return 0UL; }
    return (unsigned long)p;
}

void CudaBackend::release(unsigned long handle) {
    if (handle == 0UL || !dcuda_ensure_()) return;
    unsafe { cuMemFree_v2((unsigned long long)handle); }
}

int CudaBackend::upload(unsigned long dst, const &float src, unsigned long n) {
    if (n == 0UL) return 1;
    if (!dcuda_ensure_()) return 0;
    unsafe { return cuMemcpyHtoDAsync_v2((unsigned long long)dst, (const void*)src, n * 4UL, g_dcuda_stream) == 0; }
}

int CudaBackend::download(&float dst, unsigned long src, unsigned long n) {
    if (n == 0UL) return 1;
    if (!dcuda_ensure_()) return 0;
    unsafe {
        if (cuMemcpyDtoHAsync_v2((void*)dst, (unsigned long long)src, n * 4UL, g_dcuda_stream) != 0) return 0;
        return cuStreamSynchronize(g_dcuda_stream) == 0;
    }
}

int CudaBackend::copy(unsigned long dst, unsigned long src, unsigned long n) {
    if (n == 0UL) return 1;
    if (!dcuda_ensure_()) return 0;
    unsafe { return cuMemcpyDtoDAsync_v2((unsigned long long)dst, (unsigned long long)src, n * 4UL, g_dcuda_stream) == 0; }
}

int CudaBackend::sync() {
    if (!dcuda_ensure_()) return 0;
    unsafe { return cuStreamSynchronize(g_dcuda_stream) == 0; }
}

// Row-major C = alpha op(A) op(B) + beta C is column-major C^T = op(B)^T op(A)^T.
int CudaBackend::gemm(int ta, int tb, unsigned long m, unsigned long n, unsigned long k, float alpha,
                      unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb,
                      float beta, unsigned long c, unsigned long ldc) {
    if (!dcuda_ensure_()) return 0;
    if (m == 0UL || n == 0UL) return 1;
    unsafe {
        return cublasSgemm_v2(g_dcuda_blas, tb ? DCUDA_T : DCUDA_N, ta ? DCUDA_T : DCUDA_N,
                              (int)n, (int)m, (int)k, (const float*)&alpha, (const float*)b, (int)ldb,
                              (const float*)a, (int)lda, (const float*)&beta, (float*)c, (int)ldc) == 0;
    }
}

int CudaBackend::gemm_batched(const &DevGemm g) {
    if (!dcuda_ensure_()) return 0;
    if (g.dtype != DEVICE_FP32) {
        int t = g.dtype == DEVICE_FP16 ? DCUDA_R_16F : DCUDA_R_16BF;
        int ct = g.ctype == DEVICE_FP16 ? DCUDA_R_16F : (g.ctype == DEVICE_BF16 ? DCUDA_R_16BF : DCUDA_R_32F);
        unsafe {
            return cublasGemmStridedBatchedEx(g_dcuda_blas, g.tb ? DCUDA_T : DCUDA_N, g.ta ? DCUDA_T : DCUDA_N,
                                              (int)g.n, (int)g.m, (int)g.k, (const void*)&g.alpha,
                                              (const void*)g.b, t, (int)g.ldb, (long long)g.sb,
                                              (const void*)g.a, t, (int)g.lda, (long long)g.sa, (const void*)&g.beta,
                                              (void*)g.c, ct, (int)g.ldc, (long long)g.sc, (int)g.batch,
                                              DCUDA_COMPUTE_32F, DCUDA_GEMM_DEFAULT) == 0;
        }
    }
    unsafe {
        return cublasSgemmStridedBatched(g_dcuda_blas, g.tb ? DCUDA_T : DCUDA_N, g.ta ? DCUDA_T : DCUDA_N,
                                         (int)g.n, (int)g.m, (int)g.k, (const float*)&g.alpha,
                                         (const float*)g.b, (int)g.ldb, (long long)g.sb,
                                         (const float*)g.a, (int)g.lda, (long long)g.sa, (const float*)&g.beta,
                                         (float*)g.c, (int)g.ldc, (long long)g.sc, (int)g.batch) == 0;
    }
}

int CudaBackend::conv(const &DevConv c) { return dcdnn_run_(c); }

int CudaBackend::storage16() { return 1; }
int CudaBackend::launch(int kernel, unsigned long n, const &DevArgs args) {
    if (!dcuda_ensure_() || kernel < 0 || kernel >= DK_COUNT) return 0;
    unsigned long blocks = (n + DCUDA_BLOCK - 1UL) / DCUDA_BLOCK;
    if (blocks > DCUDA_MAX_BLOCKS) blocks = DCUDA_MAX_BLOCKS;
    if (blocks == 0UL) blocks = 1UL;
    struct DevArgs a = *args;
    unsafe {
        void* kfn = g_dcuda_fn[kernel];
        int i = 0;
        while (i < DEVK_PTRS) {
            if (dev_tag_(a.p[i]) != 0UL) {
                if (g_dcuda_state16 == 0) g_dcuda_state16 = dcuda_module_(1, (&void)(void*)&g_dcuda_fn16[0]) ? 1 : -1;
                if (g_dcuda_state16 != 1) return 0;
                kfn = g_dcuda_fn16[kernel];
                i = DEVK_PTRS;
            }
            i = i + 1;
        }
        void* params[1];
        params[0] = (void*)&a;
        return cuLaunchKernel(kfn, (unsigned int)blocks, 1U, 1U, (unsigned int)DCUDA_BLOCK, 1U, 1U, 0U,
                              g_dcuda_stream, &params[0], (void**)0) == 0;
    }
}

unsigned long CudaBackend::offset(unsigned long handle, unsigned long bytes) { return handle + bytes; }

void device_cuda_register_() {
    struct CudaBackend tag;
    tag.unused_ = 0;
    device_register_backend_(DEVICE_CUDA, &tag);
}

} // namespace std
