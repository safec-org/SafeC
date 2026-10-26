// SafeC Standard Library — the ROCm device backend (see device.h).
// RocmBackend: device memory from hipMalloc, copies and kernels queued on
// gpu_rocm.h's stream (so they're ordered with every other gpu_rocm op,
// MIOpen and hipFFT), GEMMs through rocBLAS, and device_kernels.cu compiled
// once by hiprtc for the GPU at hand. A program enables it with its rocm
// feature; link -lamdhip64 -lhiprtc -lrocblas (and -lMIOpen for MIOpen's
// convolutions).
// safeguard: feature rocm
// safeguard: link linux -L${ROCM_PATH:-/opt/rocm}/lib -lamdhip64 -lhiprtc -lrocblas
#pragma once
#include <std/stddef.h>
#include <std/ml/device.h>
#include <std/ml/device_kernels.h>
#include <std/ml/gpu_rocm.h>

namespace std {

extern int hipMalloc(void** ptr, unsigned long size);
extern int hipFree(void* ptr);
extern int hipMemcpyAsync(void* dst, const void* src, unsigned long size, int kind, void* stream);
extern int hipStreamSynchronize(void* stream);
extern int hipModuleLoadData(void** module, const void* image);
extern int hipModuleGetFunction(void** func, void* module, const char* name);
extern int hipModuleLaunchKernel(void* func, unsigned int gx, unsigned int gy, unsigned int gz,
                                 unsigned int bx, unsigned int by, unsigned int bz,
                                 unsigned int shared, void* stream, void** params, void** extra);
extern int hiprtcCreateProgram(void** prog, const char* src, const char* name, int numHeaders,
                               const char** headers, const char** includeNames);
extern int hiprtcCompileProgram(void* prog, int numOptions, const char** options);
extern int hiprtcGetCodeSize(void* prog, unsigned long* size);
extern int hiprtcGetCode(void* prog, char* code);
extern int hiprtcGetProgramLogSize(void* prog, unsigned long* size);
extern int hiprtcGetProgramLog(void* prog, char* log);
extern int hiprtcDestroyProgram(void** prog);
extern int rocblas_create_handle(void** handle);
extern int rocblas_set_stream(void* handle, void* stream);
extern int rocblas_sgemm(void* handle, int transA, int transB, int m, int n, int k,
                         const float* alpha, const float* a, int lda, const float* b, int ldb,
                         const float* beta, float* c, int ldc);
extern int rocblas_sgemm_strided_batched(void* handle, int transA, int transB, int m, int n, int k,
                                         const float* alpha, const float* a, int lda, long long sa,
                                         const float* b, int ldb, long long sb, const float* beta,
                                         float* c, int ldc, long long sc, int batch);
extern int rocblas_gemm_strided_batched_ex(void* handle, int transA, int transB, int m, int n, int k,
                                           const void* alpha, const void* a, int aType, int lda, long long sa,
                                           const void* b, int bType, int ldb, long long sb, const void* beta,
                                           const void* c, int cType, int ldc, long long sc,
                                           void* d, int dType, int ldd, long long sd, int batch,
                                           int computeType, int algo, int solution, unsigned int flags);
#define DROCM_BLAS_F16  150   // rocblas_datatype_f16_r
#define DROCM_BLAS_F32  151   // rocblas_datatype_f32_r
#define DROCM_BLAS_BF16 168   // rocblas_datatype_bf16_r
extern unsigned char* malloc(size_t size);
extern void  free(unsigned char* ptr);

#define DROCM_H2D 1
#define DROCM_D2H 2
#define DROCM_D2D 3
#define DROCM_N   111
#define DROCM_T   112
#define DROCM_BLOCK 256UL
#define DROCM_MAX_BLOCKS 4096UL

static int   g_drocm_state = 0;      // 0 = not yet, 1 = ready, -1 = unusable
static void* g_drocm_stream = (void*)0;
static void* g_drocm_blas = (void*)0;
static void* g_drocm_fn[DK_COUNT];
static void* g_drocm_fn16[DK_COUNT];   // the DK_TAGGED build (16-bit arguments), compiled on first use
static int   g_drocm_state16 = 0;

// Compiles device_kernels.cu (with DK_TAGGED when tagged) into fns.
static int drocm_module_(int tagged, &void fns) {
    unsafe {
        void* prog = (void*)0;
        if (hiprtcCreateProgram(&prog, device_kernel_source_(), "safec_device.hip", 0,
                                (const char**)0, (const char**)0) != 0) return 0;
        const char* opts[2];
        opts[0] = "-O3";
        opts[1] = "-DDK_TAGGED";
        if (hiprtcCompileProgram(prog, tagged ? 2 : 1, &opts[0]) != 0) { hiprtcDestroyProgram(&prog); return 0; }
        unsigned long size = 0UL;
        hiprtcGetCodeSize(prog, &size);
        char* code = (char*)malloc(size + 1UL);
        if (code == (char*)0) { hiprtcDestroyProgram(&prog); return 0; }
        hiprtcGetCode(prog, code);
        hiprtcDestroyProgram(&prog);
        void* module = (void*)0;
        int loaded = hipModuleLoadData(&module, (const void*)code);
        free((unsigned char*)code);
        if (loaded != 0) return 0;
        void** f = (void**)fns;
        int k = 0;
        while (k < DK_COUNT) {
            if (hipModuleGetFunction(&f[k], module, device_kernel_name_(k)) != 0) return 0;
            k = k + 1;
        }
    }
    return 1;
}

static int drocm_ensure_() {
    if (g_drocm_state != 0) return g_drocm_state == 1;
    g_drocm_state = -1;
    if (!rocm_available()) return 0;
    unsafe {
        match (rocm_stream_handle()) {
            case some(s): { g_drocm_stream = (void*)s; }
            case null: { return 0; }
        }
        if (rocblas_create_handle(&g_drocm_blas) != 0) return 0;
        rocblas_set_stream(g_drocm_blas, g_drocm_stream);
        if (!drocm_module_(0, (&void)(void*)&g_drocm_fn[0])) return 0;
    }
    g_drocm_state = 1;
    return 1;
}

// ── Convolutions through MIOpen (optional) ───────────────────────────────────
// Weak references: a program linked with -lMIOpen gets MIOpen's convolution
// kernels (implicit GEMM, Winograd, matrix cores); without it these resolve
// to null and RocmBackend::conv declines (the tensor op runs im2col + GEMM).
// MIOpen queues on this backend's stream. Each shape gets a plan: its
// descriptors, and per direction the algorithm miopenFind picked the first
// time that direction ran.
extern int miopenCreate(void** handle) __attribute__((weak));
extern int miopenSetStream(void* handle, void* stream) __attribute__((weak));
extern int miopenCreateTensorDescriptor(void** desc) __attribute__((weak));
extern int miopenSet4dTensorDescriptor(void* desc, int dataType, int n, int c, int h, int w) __attribute__((weak));
extern int miopenCreateConvolutionDescriptor(void** desc) __attribute__((weak));
extern int miopenInitConvolutionDescriptor(void* desc, int mode, int padH, int padW, int strideH, int strideW,
                                           int dilationH, int dilationW) __attribute__((weak));
extern int miopenSetTransposeConvOutputPadding(void* desc, int adjH, int adjW) __attribute__((weak));
extern int miopenConvolutionForwardGetWorkSpaceSize(void* handle, void* wDesc, void* xDesc, void* convDesc,
                                                    void* yDesc, unsigned long* bytes) __attribute__((weak));
extern int miopenConvolutionBackwardDataGetWorkSpaceSize(void* handle, void* dyDesc, void* wDesc, void* convDesc,
                                                         void* dxDesc, unsigned long* bytes) __attribute__((weak));
extern int miopenConvolutionBackwardWeightsGetWorkSpaceSize(void* handle, void* dyDesc, void* xDesc, void* convDesc,
                                                            void* dwDesc, unsigned long* bytes) __attribute__((weak));
extern int miopenFindConvolutionForwardAlgorithm(void* handle, void* xDesc, const void* x, void* wDesc, const void* w,
                                                 void* convDesc, void* yDesc, void* y, int request, int* returned,
                                                 void* perf, void* ws, unsigned long wsBytes, int exhaustive)
    __attribute__((weak));
extern int miopenFindConvolutionBackwardDataAlgorithm(void* handle, void* dyDesc, const void* dy, void* wDesc,
                                                      const void* w, void* convDesc, void* dxDesc, void* dx,
                                                      int request, int* returned, void* perf, void* ws,
                                                      unsigned long wsBytes, int exhaustive) __attribute__((weak));
extern int miopenFindConvolutionBackwardWeightsAlgorithm(void* handle, void* dyDesc, const void* dy, void* xDesc,
                                                         const void* x, void* convDesc, void* dwDesc, void* dw,
                                                         int request, int* returned, void* perf, void* ws,
                                                         unsigned long wsBytes, int exhaustive) __attribute__((weak));
extern int miopenConvolutionForward(void* handle, const void* alpha, void* xDesc, const void* x, void* wDesc,
                                    const void* w, void* convDesc, int algo, const void* beta, void* yDesc, void* y,
                                    void* ws, unsigned long wsBytes) __attribute__((weak));
extern int miopenConvolutionBackwardData(void* handle, const void* alpha, void* dyDesc, const void* dy, void* wDesc,
                                         const void* w, void* convDesc, int algo, const void* beta, void* dxDesc,
                                         void* dx, void* ws, unsigned long wsBytes) __attribute__((weak));
extern int miopenConvolutionBackwardWeights(void* handle, const void* alpha, void* dyDesc, const void* dy,
                                            void* xDesc, const void* x, void* convDesc, int algo, const void* beta,
                                            void* dwDesc, void* dw, void* ws, unsigned long wsBytes)
    __attribute__((weak));

// miopenConvAlgoPerf_t
struct DrocmMioPerf_ {
    int           algo;
    float         time;
    unsigned long memory;
};

#define DROCM_MIO_PLANS 256
#define DROCM_MIO_KEY   17

static int   g_dmio_state = 0;          // 0 untried, 1 ready, -1 MIOpen not linked or failed
static void* g_dmio_handle = (void*)0;
static unsigned long g_dmio_ws = 0UL;   // workspace
static unsigned long g_dmio_wsBytes = 0UL;
static unsigned long g_dmio_key[DROCM_MIO_PLANS * DROCM_MIO_KEY];
static void* g_dmio_desc[DROCM_MIO_PLANS * 4];   // x, y, w, conv
static int   g_dmio_algo[DROCM_MIO_PLANS * 3];   // per op; -1 not found yet, -2 MIOpen can't
static int   g_dmio_good[DROCM_MIO_PLANS];
static int   g_dmio_count = 0;

static int dmio_on_() {
    if (g_dmio_state == 0) {
        g_dmio_state = -1;
        unsafe {
            if (drocm_ensure_() && (void*)miopenCreate != (void*)0 && miopenCreate(&g_dmio_handle) == 0 &&
                miopenSetStream(g_dmio_handle, g_drocm_stream) == 0) g_dmio_state = 1;
        }
    }
    return g_dmio_state == 1;
}

static void dmio_fill_key_(const &DevConv c, &unsigned long k) {
    unsafe {
        unsigned long* v = (unsigned long*)k;
        v[0] = (unsigned long)c.transpose; v[1] = c.batch; v[2] = c.inC; v[3] = c.inH; v[4] = c.inW;
        v[5] = c.outC; v[6] = c.outH; v[7] = c.outW; v[8] = c.kH; v[9] = c.kW; v[10] = c.sH; v[11] = c.sW;
        v[12] = c.pH; v[13] = c.pW; v[14] = c.dH; v[15] = c.dW; v[16] = (unsigned long)c.dtype;
    }
}

static int dmio_ws_grow_(unsigned long bytes) {
    if (bytes <= g_dmio_wsBytes) return 1;
    unsafe {
        if (g_dmio_ws != 0UL) { hipStreamSynchronize(g_drocm_stream); hipFree((void*)g_dmio_ws); }
        g_dmio_ws = 0UL;
        g_dmio_wsBytes = 0UL;
        void* p = (void*)0;
        if (hipMalloc(&p, bytes) != 0) return 0;
        g_dmio_ws = (unsigned long)p;
        g_dmio_wsBytes = bytes;
    }
    return 1;
}

// The plan for c's shape, made now if new; -1 when MIOpen can't take it.
static int dmio_plan_(const &DevConv c) {
    unsigned long key[DROCM_MIO_KEY];
    dmio_fill_key_(c, &key[0]);
    int i = 0;
    while (i < g_dmio_count) {
        int same = 1;
        int j = 0;
        while (same && j < DROCM_MIO_KEY) { if (g_dmio_key[i * DROCM_MIO_KEY + j] != key[j]) same = 0; j = j + 1; }
        if (same) return g_dmio_good[i] ? i : -1;
        i = i + 1;
    }
    if (g_dmio_count >= DROCM_MIO_PLANS) return -1;
    i = g_dmio_count;
    g_dmio_count = g_dmio_count + 1;
    unsafe { dmio_fill_key_(c, (&unsigned long)&g_dmio_key[i * DROCM_MIO_KEY]); }
    g_dmio_algo[i * 3] = -1; g_dmio_algo[i * 3 + 1] = -1; g_dmio_algo[i * 3 + 2] = -1;
    int ok = 1;
    unsafe {
        void* xd = (void*)0; void* yd = (void*)0; void* wd = (void*)0; void* cd = (void*)0;
        ok = miopenCreateTensorDescriptor(&xd) == 0 && miopenCreateTensorDescriptor(&yd) == 0 &&
             miopenCreateTensorDescriptor(&wd) == 0 && miopenCreateConvolutionDescriptor(&cd) == 0;
        int dt = c.dtype == DEVICE_FP16 ? 0 : (c.dtype == DEVICE_BF16 ? 5 : 1);   // miopenHalf, miopenBFloat16, miopenFloat
        ok = ok && miopenSet4dTensorDescriptor(xd, dt, (int)c.batch, (int)c.inC, (int)c.inH, (int)c.inW) == 0 &&
             miopenSet4dTensorDescriptor(yd, dt, (int)c.batch, (int)c.outC, (int)c.outH, (int)c.outW) == 0;
        if (c.transpose) ok = ok && miopenSet4dTensorDescriptor(wd, dt, (int)c.inC, (int)c.outC, (int)c.kH, (int)c.kW) == 0;
        else ok = ok && miopenSet4dTensorDescriptor(wd, dt, (int)c.outC, (int)c.inC, (int)c.kH, (int)c.kW) == 0;
        ok = ok && miopenInitConvolutionDescriptor(cd, c.transpose ? 1 : 0, (int)c.pH, (int)c.pW, (int)c.sH, (int)c.sW,
                                                   (int)c.dH, (int)c.dW) == 0;
        if (ok && c.transpose) {
            long adjH = (long)c.outH - ((long)(c.inH - 1UL) * (long)c.sH - 2L * (long)c.pH + (long)(c.dH * (c.kH - 1UL)) + 1L);
            long adjW = (long)c.outW - ((long)(c.inW - 1UL) * (long)c.sW - 2L * (long)c.pW + (long)(c.dW * (c.kW - 1UL)) + 1L);
            if (adjH < 0L || adjW < 0L) ok = 0;
            else if (adjH > 0L || adjW > 0L) ok = miopenSetTransposeConvOutputPadding(cd, (int)adjH, (int)adjW) == 0;
        }
        g_dmio_desc[i * 4] = xd; g_dmio_desc[i * 4 + 1] = yd; g_dmio_desc[i * 4 + 2] = wd; g_dmio_desc[i * 4 + 3] = cd;
    }
    g_dmio_good[i] = ok;
    return ok ? i : -1;
}

static int dmio_run_(const &DevConv c) {
    if (!dmio_on_()) return 0;
    int plan = dmio_plan_(c);
    if (plan < 0) return 0;
    float one = (float)1.0;
    float zero = (float)0.0;
    unsafe {
        void* xd = g_dmio_desc[plan * 4]; void* yd = g_dmio_desc[plan * 4 + 1];
        void* wd = g_dmio_desc[plan * 4 + 2]; void* cd = g_dmio_desc[plan * 4 + 3];
        int slot = plan * 3 + c.op;
        if (g_dmio_algo[slot] == -2) return 0;
        if (g_dmio_algo[slot] == -1) {
            // the first run of this direction: find its algorithm
            unsigned long need = 0UL;
            int ok;
            if (c.op == DEV_CONV_FWD) ok = miopenConvolutionForwardGetWorkSpaceSize(g_dmio_handle, wd, xd, cd, yd, &need) == 0;
            else if (c.op == DEV_CONV_BWD_DATA) ok = miopenConvolutionBackwardDataGetWorkSpaceSize(g_dmio_handle, yd, wd, cd, xd, &need) == 0;
            else ok = miopenConvolutionBackwardWeightsGetWorkSpaceSize(g_dmio_handle, yd, xd, cd, wd, &need) == 0;
            ok = ok && dmio_ws_grow_(need == 0UL ? 4UL : need);
            struct DrocmMioPerf_ pf;
            int got = 0;
            void* ws = (void*)g_dmio_ws;
            if (ok && c.op == DEV_CONV_FWD)
                ok = miopenFindConvolutionForwardAlgorithm(g_dmio_handle, xd, (const void*)c.x, wd, (const void*)c.w, cd, yd,
                                                           (void*)c.y, 1, &got, (void*)&pf, ws, g_dmio_wsBytes, 0) == 0;
            else if (ok && c.op == DEV_CONV_BWD_DATA)
                ok = miopenFindConvolutionBackwardDataAlgorithm(g_dmio_handle, yd, (const void*)c.y, wd, (const void*)c.w, cd,
                                                                xd, (void*)c.x, 1, &got, (void*)&pf, ws, g_dmio_wsBytes, 0) == 0;
            else if (ok)
                ok = miopenFindConvolutionBackwardWeightsAlgorithm(g_dmio_handle, yd, (const void*)c.y, xd, (const void*)c.x,
                                                                   cd, wd, (void*)c.w, 1, &got, (void*)&pf, ws,
                                                                   g_dmio_wsBytes, 0) == 0;
            if (!ok || got < 1) { g_dmio_algo[slot] = -2; return 0; }
            g_dmio_algo[slot] = pf.algo;
            if (pf.memory > g_dmio_wsBytes && !dmio_ws_grow_(pf.memory)) { g_dmio_algo[slot] = -2; return 0; }
        }
        int algo = g_dmio_algo[slot];
        void* ws = (void*)g_dmio_ws;
        if (c.op == DEV_CONV_FWD)
            return miopenConvolutionForward(g_dmio_handle, (const void*)&one, xd, (const void*)c.x, wd, (const void*)c.w, cd,
                                            algo, (const void*)&zero, yd, (void*)c.y, ws, g_dmio_wsBytes) == 0;
        if (c.op == DEV_CONV_BWD_DATA)
            return miopenConvolutionBackwardData(g_dmio_handle, (const void*)&one, yd, (const void*)c.y, wd, (const void*)c.w,
                                                 cd, algo, (const void*)&zero, xd, (void*)c.x, ws, g_dmio_wsBytes) == 0;
        return miopenConvolutionBackwardWeights(g_dmio_handle, (const void*)&one, yd, (const void*)c.y, xd, (const void*)c.x,
                                                cd, algo, (const void*)&zero, wd, (void*)c.w, ws, g_dmio_wsBytes) == 0;
    }
}

struct RocmBackend {
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

int RocmBackend::available() { return drocm_ensure_(); }

unsigned long RocmBackend::alloc(unsigned long bytes) {
    if (!drocm_ensure_()) return 0UL;
    void* p = (void*)0;
    unsafe { if (hipMalloc(&p, bytes) != 0) return 0UL; }
    return (unsigned long)p;
}

void RocmBackend::release(unsigned long handle) {
    if (handle == 0UL) return;
    unsafe { hipFree((void*)handle); }
}

int RocmBackend::upload(unsigned long dst, const &float src, unsigned long n) {
    if (n == 0UL) return 1;
    unsafe { return hipMemcpyAsync((void*)dst, (const void*)src, n * 4UL, DROCM_H2D, g_drocm_stream) == 0; }
}

int RocmBackend::download(&float dst, unsigned long src, unsigned long n) {
    if (n == 0UL) return 1;
    unsafe {
        if (hipMemcpyAsync((void*)dst, (const void*)src, n * 4UL, DROCM_D2H, g_drocm_stream) != 0) return 0;
        return hipStreamSynchronize(g_drocm_stream) == 0;
    }
}

int RocmBackend::copy(unsigned long dst, unsigned long src, unsigned long n) {
    if (n == 0UL) return 1;
    unsafe { return hipMemcpyAsync((void*)dst, (const void*)src, n * 4UL, DROCM_D2D, g_drocm_stream) == 0; }
}

int RocmBackend::sync() {
    if (!drocm_ensure_()) return 0;
    unsafe { return hipStreamSynchronize(g_drocm_stream) == 0; }
}

// Row-major C = alpha op(A) op(B) + beta C is column-major C^T = op(B)^T op(A)^T.
int RocmBackend::gemm(int ta, int tb, unsigned long m, unsigned long n, unsigned long k, float alpha,
                      unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb,
                      float beta, unsigned long c, unsigned long ldc) {
    if (!drocm_ensure_()) return 0;
    if (m == 0UL || n == 0UL) return 1;
    unsafe {
        return rocblas_sgemm(g_drocm_blas, tb ? DROCM_T : DROCM_N, ta ? DROCM_T : DROCM_N,
                             (int)n, (int)m, (int)k, (const float*)&alpha, (const float*)b, (int)ldb,
                             (const float*)a, (int)lda, (const float*)&beta, (float*)c, (int)ldc) == 0;
    }
}

int RocmBackend::gemm_batched(const &DevGemm g) {
    if (!drocm_ensure_()) return 0;
    if (g.dtype != DEVICE_FP32) {
        int t = g.dtype == DEVICE_FP16 ? DROCM_BLAS_F16 : DROCM_BLAS_BF16;
        int ct = g.ctype == DEVICE_FP16 ? DROCM_BLAS_F16 : (g.ctype == DEVICE_BF16 ? DROCM_BLAS_BF16 : DROCM_BLAS_F32);
        unsafe {
            return rocblas_gemm_strided_batched_ex(g_drocm_blas, g.tb ? DROCM_T : DROCM_N, g.ta ? DROCM_T : DROCM_N,
                                                   (int)g.n, (int)g.m, (int)g.k, (const void*)&g.alpha,
                                                   (const void*)g.b, t, (int)g.ldb, (long long)g.sb,
                                                   (const void*)g.a, t, (int)g.lda, (long long)g.sa, (const void*)&g.beta,
                                                   (const void*)g.c, ct, (int)g.ldc, (long long)g.sc,
                                                   (void*)g.c, ct, (int)g.ldc, (long long)g.sc, (int)g.batch,
                                                   DROCM_BLAS_F32, 0, 0, 0U) == 0;
        }
    }
    unsafe {
        return rocblas_sgemm_strided_batched(g_drocm_blas, g.tb ? DROCM_T : DROCM_N, g.ta ? DROCM_T : DROCM_N,
                                             (int)g.n, (int)g.m, (int)g.k, (const float*)&g.alpha,
                                             (const float*)g.b, (int)g.ldb, (long long)g.sb,
                                             (const float*)g.a, (int)g.lda, (long long)g.sa, (const float*)&g.beta,
                                             (float*)g.c, (int)g.ldc, (long long)g.sc, (int)g.batch) == 0;
    }
}

extern char* getenv(const char* n);
int RocmBackend::conv(const &DevConv c) { unsafe { if (getenv("TMP_NOLIB") != (char*)0) return 0; } return dmio_run_(c); }

int RocmBackend::storage16() { return 1; }
int RocmBackend::launch(int kernel, unsigned long n, const &DevArgs args) {
    if (!drocm_ensure_() || kernel < 0 || kernel >= DK_COUNT) return 0;
    unsigned long blocks = (n + DROCM_BLOCK - 1UL) / DROCM_BLOCK;
    if (blocks > DROCM_MAX_BLOCKS) blocks = DROCM_MAX_BLOCKS;
    if (blocks == 0UL) blocks = 1UL;
    struct DevArgs a = *args;
    unsafe {
        void* kfn = g_drocm_fn[kernel];
        int i = 0;
        while (i < DEVK_PTRS) {
            if (dev_tag_(a.p[i]) != 0UL) {
                if (g_drocm_state16 == 0) g_drocm_state16 = drocm_module_(1, (&void)(void*)&g_drocm_fn16[0]) ? 1 : -1;
                if (g_drocm_state16 != 1) return 0;
                kfn = g_drocm_fn16[kernel];
                i = DEVK_PTRS;
            }
            i = i + 1;
        }
        void* params[1];
        params[0] = (void*)&a;
        return hipModuleLaunchKernel(kfn, (unsigned int)blocks, 1U, 1U, (unsigned int)DROCM_BLOCK,
                                     1U, 1U, 0U, g_drocm_stream, &params[0], (void**)0) == 0;
    }
}

unsigned long RocmBackend::offset(unsigned long handle, unsigned long bytes) { return handle + bytes; }

void device_rocm_register_() {
    struct RocmBackend tag;
    tag.unused_ = 0;
    device_register_backend_(DEVICE_ROCM, &tag);
}

} // namespace std
