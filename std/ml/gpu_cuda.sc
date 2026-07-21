// SafeC Standard Library — CUDA GPU backend implementation (see
// gpu_cuda.h). UNVERIFIED — no NVIDIA GPU/CUDA toolkit in this sandbox;
// see that header's warning. Written against the real Driver API ABI.
// Every kernel below is hand-written PTX 7.0 (sm_50), the same technique
// cuda_add_f32 already used and the same reasoning gpu_mps.sc's runtime-
// compiled MSL uses: PTX is NVIDIA's stable, textual IR, so embedding it
// as a string avoids needing nvcc/nvrtc as a build dependency. None of
// this has been run through ptxas or any other real CUDA toolchain here —
// register counts and instruction sequences were written by hand,
// mirroring add_kernel's already-existing structure as closely as
// possible per kernel to minimize the chance of a transcription error, but
// this needs real hardware verification before being trusted.
#pragma once
#include <std/ml/gpu_cuda.h>
#include <std/ml/float16.h>
#include <std/mem.h>

namespace std {

// ── Driver API (cuda.h) — hand-matched signatures ───────────────────────────
extern int cuInit(unsigned int flags);
extern int cuDeviceGetCount(int* count);
extern int cuDeviceGet(int* device, int ordinal);
extern int cuCtxCreate_v2(void** pctx, unsigned int flags, int dev);
extern int cuCtxDestroy_v2(void* ctx);
extern int cuModuleLoadData(void** module, const void* image);
extern int cuModuleGetFunction(void** hfunc, void* hmod, const char* name);
extern int cuMemAlloc_v2(unsigned long long* dptr, unsigned long bytesize);
extern int cuMemFree_v2(unsigned long long dptr);
extern int cuMemcpyHtoD_v2(unsigned long long dstDevice, const void* srcHost, unsigned long byteCount);
extern int cuMemcpyDtoH_v2(void* dstHost, unsigned long long srcDevice, unsigned long byteCount);
extern int cuLaunchKernel(void* f,
                           unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                           unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                           unsigned int sharedMemBytes, void* hStream,
                           void** kernelParams, void** extra);
extern int cuCtxSynchronize();

// ── cuBLAS (cublas_v2.h) — hand-matched signatures ──────────────────────────
// cublasHandle_t is an opaque pointer (void*), same treatment as every
// other CUDA/Metal/Vulkan opaque handle in this codebase. cublasStatus_t
// is a plain C enum (int); CUBLAS_STATUS_SUCCESS == 0, matching
// CUDA_SUCCESS above. cublasOperation_t: CUBLAS_OP_N=0 (no transpose),
// CUBLAS_OP_T=1 (transpose) — only these two are used here.
extern int cublasCreate_v2(void** handle);
extern int cublasDestroy_v2(void* handle);
// alpha/beta are host pointers under cuBLAS's default pointer mode
// (CUBLAS_POINTER_MODE_HOST) — a plain address-of a local float works,
// no device allocation needed for the two scalars.
extern int cublasSgemm_v2(void* handle, int transa, int transb,
                           int m, int n, int k,
                           const float* alpha, const float* A, int lda,
                           const float* B, int ldb,
                           const float* beta, float* C, int ldc);

#define CUDA_SUCCESS 0
#define CUBLAS_OP_N 0

int cuda_available() {
    if (cuInit(0U) != CUDA_SUCCESS) return 0;
    int count = 0;
    unsafe { if (cuDeviceGetCount(&count) != CUDA_SUCCESS) return 0; }
    return count > 0;
}

// Shared context/module/launch/teardown for every 'out[i] = f(a[i], b[i])'
// kernel below — the only thing that varies between sub/mul/div/pow (and
// add, refactored to use this too) is the PTX source and kernel name.
static int __cuda_run_binary_kernel(const char* ptxSrc, const char* kernelName,
                                     const float* a, const float* b, float* out, unsigned long n) {
    unsafe {
        if (cuInit(0U) != CUDA_SUCCESS) return 0;
        int device;
        if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) return 0;
        void* ctx = (void*)0;
        if (cuCtxCreate_v2(&ctx, 0U, device) != CUDA_SUCCESS) return 0;

        void* module = (void*)0;
        if (cuModuleLoadData(&module, (const void*)ptxSrc) != CUDA_SUCCESS) {
            cuCtxDestroy_v2(ctx);
            return 0;
        }
        void* kernel = (void*)0;
        if (cuModuleGetFunction(&kernel, module, kernelName) != CUDA_SUCCESS) {
            cuCtxDestroy_v2(ctx);
            return 0;
        }

        unsigned long bytes = n * sizeof(float);
        unsigned long long devA = 0ULL; unsigned long long devB = 0ULL; unsigned long long devOut = 0ULL;
        cuMemAlloc_v2(&devA, bytes);
        cuMemAlloc_v2(&devB, bytes);
        cuMemAlloc_v2(&devOut, bytes);
        cuMemcpyHtoD_v2(devA, (const void*)a, bytes);
        cuMemcpyHtoD_v2(devB, (const void*)b, bytes);

        unsigned int nParam = (unsigned int)n;
        void* params[4];
        params[0] = (void*)&devA;
        params[1] = (void*)&devB;
        params[2] = (void*)&devOut;
        params[3] = (void*)&nParam;

        unsigned int blockSize = 256U;
        unsigned int gridSize = (unsigned int)((n + (unsigned long)blockSize - 1UL) / (unsigned long)blockSize);
        int launchOk = cuLaunchKernel(kernel, gridSize, 1U, 1U, blockSize, 1U, 1U,
                                       0U, (void*)0, params, (void**)0) == CUDA_SUCCESS;
        cuCtxSynchronize();
        if (launchOk) cuMemcpyDtoH_v2((void*)out, devOut, bytes);

        cuMemFree_v2(devA);
        cuMemFree_v2(devB);
        cuMemFree_v2(devOut);
        cuCtxDestroy_v2(ctx);
        return launchOk ? 1 : 0;
    }
}

// Same as __cuda_run_binary_kernel but for a single-input kernel (relu,
// log, exp, sqrt) — one input buffer instead of two.
static int __cuda_run_unary_kernel(const char* ptxSrc, const char* kernelName,
                                    const float* a, float* out, unsigned long n) {
    unsafe {
        if (cuInit(0U) != CUDA_SUCCESS) return 0;
        int device;
        if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) return 0;
        void* ctx = (void*)0;
        if (cuCtxCreate_v2(&ctx, 0U, device) != CUDA_SUCCESS) return 0;

        void* module = (void*)0;
        if (cuModuleLoadData(&module, (const void*)ptxSrc) != CUDA_SUCCESS) {
            cuCtxDestroy_v2(ctx);
            return 0;
        }
        void* kernel = (void*)0;
        if (cuModuleGetFunction(&kernel, module, kernelName) != CUDA_SUCCESS) {
            cuCtxDestroy_v2(ctx);
            return 0;
        }

        unsigned long bytes = n * sizeof(float);
        unsigned long long devA = 0ULL; unsigned long long devOut = 0ULL;
        cuMemAlloc_v2(&devA, bytes);
        cuMemAlloc_v2(&devOut, bytes);
        cuMemcpyHtoD_v2(devA, (const void*)a, bytes);

        unsigned int nParam = (unsigned int)n;
        void* params[3];
        params[0] = (void*)&devA;
        params[1] = (void*)&devOut;
        params[2] = (void*)&nParam;

        unsigned int blockSize = 256U;
        unsigned int gridSize = (unsigned int)((n + (unsigned long)blockSize - 1UL) / (unsigned long)blockSize);
        int launchOk = cuLaunchKernel(kernel, gridSize, 1U, 1U, blockSize, 1U, 1U,
                                       0U, (void*)0, params, (void**)0) == CUDA_SUCCESS;
        cuCtxSynchronize();
        if (launchOk) cuMemcpyDtoH_v2((void*)out, devOut, bytes);

        cuMemFree_v2(devA);
        cuMemFree_v2(devOut);
        cuCtxDestroy_v2(ctx);
        return launchOk ? 1 : 0;
    }
}

int cuda_add_f32(const float* a, const float* b, float* out, unsigned long n) {
    const char* ptxSrc =
        ".version 7.0\n"
        ".target sm_50\n"
        ".address_size 64\n"
        ".visible .entry add_kernel(\n"
        "    .param .u64 add_kernel_param_0,\n"
        "    .param .u64 add_kernel_param_1,\n"
        "    .param .u64 add_kernel_param_2,\n"
        "    .param .u32 add_kernel_param_3\n"
        ") {\n"
        "    .reg .pred %p<2>;\n"
        "    .reg .f32 %f<4>;\n"
        "    .reg .b32 %r<6>;\n"
        "    .reg .b64 %rd<11>;\n"
        "    ld.param.u64 %rd1, [add_kernel_param_0];\n"
        "    ld.param.u64 %rd2, [add_kernel_param_1];\n"
        "    ld.param.u64 %rd3, [add_kernel_param_2];\n"
        "    ld.param.u32 %r2, [add_kernel_param_3];\n"
        "    mov.u32 %r3, %ctaid.x;\n"
        "    mov.u32 %r4, %ntid.x;\n"
        "    mov.u32 %r5, %tid.x;\n"
        "    mad.lo.s32 %r1, %r3, %r4, %r5;\n"
        "    setp.ge.s32 %p1, %r1, %r2;\n"
        "    @%p1 bra DONE;\n"
        "    cvta.to.global.u64 %rd4, %rd1;\n"
        "    mul.wide.s32 %rd5, %r1, 4;\n"
        "    add.s64 %rd6, %rd4, %rd5;\n"
        "    cvta.to.global.u64 %rd7, %rd2;\n"
        "    add.s64 %rd8, %rd7, %rd5;\n"
        "    ld.global.f32 %f1, [%rd6];\n"
        "    ld.global.f32 %f2, [%rd8];\n"
        "    add.f32 %f3, %f1, %f2;\n"
        "    cvta.to.global.u64 %rd9, %rd3;\n"
        "    add.s64 %rd10, %rd9, %rd5;\n"
        "    st.global.f32 [%rd10], %f3;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    return __cuda_run_binary_kernel(ptxSrc, "add_kernel", a, b, out, n);
}

int cuda_sub_f32(const float* a, const float* b, float* out, unsigned long n) {
    const char* ptxSrc =
        ".version 7.0\n"
        ".target sm_50\n"
        ".address_size 64\n"
        ".visible .entry sub_kernel(\n"
        "    .param .u64 sub_kernel_param_0,\n"
        "    .param .u64 sub_kernel_param_1,\n"
        "    .param .u64 sub_kernel_param_2,\n"
        "    .param .u32 sub_kernel_param_3\n"
        ") {\n"
        "    .reg .pred %p<2>;\n"
        "    .reg .f32 %f<4>;\n"
        "    .reg .b32 %r<6>;\n"
        "    .reg .b64 %rd<11>;\n"
        "    ld.param.u64 %rd1, [sub_kernel_param_0];\n"
        "    ld.param.u64 %rd2, [sub_kernel_param_1];\n"
        "    ld.param.u64 %rd3, [sub_kernel_param_2];\n"
        "    ld.param.u32 %r2, [sub_kernel_param_3];\n"
        "    mov.u32 %r3, %ctaid.x;\n"
        "    mov.u32 %r4, %ntid.x;\n"
        "    mov.u32 %r5, %tid.x;\n"
        "    mad.lo.s32 %r1, %r3, %r4, %r5;\n"
        "    setp.ge.s32 %p1, %r1, %r2;\n"
        "    @%p1 bra DONE;\n"
        "    cvta.to.global.u64 %rd4, %rd1;\n"
        "    mul.wide.s32 %rd5, %r1, 4;\n"
        "    add.s64 %rd6, %rd4, %rd5;\n"
        "    cvta.to.global.u64 %rd7, %rd2;\n"
        "    add.s64 %rd8, %rd7, %rd5;\n"
        "    ld.global.f32 %f1, [%rd6];\n"
        "    ld.global.f32 %f2, [%rd8];\n"
        "    sub.f32 %f3, %f1, %f2;\n"
        "    cvta.to.global.u64 %rd9, %rd3;\n"
        "    add.s64 %rd10, %rd9, %rd5;\n"
        "    st.global.f32 [%rd10], %f3;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    return __cuda_run_binary_kernel(ptxSrc, "sub_kernel", a, b, out, n);
}

int cuda_mul_f32(const float* a, const float* b, float* out, unsigned long n) {
    const char* ptxSrc =
        ".version 7.0\n"
        ".target sm_50\n"
        ".address_size 64\n"
        ".visible .entry mul_kernel(\n"
        "    .param .u64 mul_kernel_param_0,\n"
        "    .param .u64 mul_kernel_param_1,\n"
        "    .param .u64 mul_kernel_param_2,\n"
        "    .param .u32 mul_kernel_param_3\n"
        ") {\n"
        "    .reg .pred %p<2>;\n"
        "    .reg .f32 %f<4>;\n"
        "    .reg .b32 %r<6>;\n"
        "    .reg .b64 %rd<11>;\n"
        "    ld.param.u64 %rd1, [mul_kernel_param_0];\n"
        "    ld.param.u64 %rd2, [mul_kernel_param_1];\n"
        "    ld.param.u64 %rd3, [mul_kernel_param_2];\n"
        "    ld.param.u32 %r2, [mul_kernel_param_3];\n"
        "    mov.u32 %r3, %ctaid.x;\n"
        "    mov.u32 %r4, %ntid.x;\n"
        "    mov.u32 %r5, %tid.x;\n"
        "    mad.lo.s32 %r1, %r3, %r4, %r5;\n"
        "    setp.ge.s32 %p1, %r1, %r2;\n"
        "    @%p1 bra DONE;\n"
        "    cvta.to.global.u64 %rd4, %rd1;\n"
        "    mul.wide.s32 %rd5, %r1, 4;\n"
        "    add.s64 %rd6, %rd4, %rd5;\n"
        "    cvta.to.global.u64 %rd7, %rd2;\n"
        "    add.s64 %rd8, %rd7, %rd5;\n"
        "    ld.global.f32 %f1, [%rd6];\n"
        "    ld.global.f32 %f2, [%rd8];\n"
        "    mul.f32 %f3, %f1, %f2;\n"
        "    cvta.to.global.u64 %rd9, %rd3;\n"
        "    add.s64 %rd10, %rd9, %rd5;\n"
        "    st.global.f32 [%rd10], %f3;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    return __cuda_run_binary_kernel(ptxSrc, "mul_kernel", a, b, out, n);
}

int cuda_div_f32(const float* a, const float* b, float* out, unsigned long n) {
    const char* ptxSrc =
        ".version 7.0\n"
        ".target sm_50\n"
        ".address_size 64\n"
        ".visible .entry div_kernel(\n"
        "    .param .u64 div_kernel_param_0,\n"
        "    .param .u64 div_kernel_param_1,\n"
        "    .param .u64 div_kernel_param_2,\n"
        "    .param .u32 div_kernel_param_3\n"
        ") {\n"
        "    .reg .pred %p<2>;\n"
        "    .reg .f32 %f<4>;\n"
        "    .reg .b32 %r<6>;\n"
        "    .reg .b64 %rd<11>;\n"
        "    ld.param.u64 %rd1, [div_kernel_param_0];\n"
        "    ld.param.u64 %rd2, [div_kernel_param_1];\n"
        "    ld.param.u64 %rd3, [div_kernel_param_2];\n"
        "    ld.param.u32 %r2, [div_kernel_param_3];\n"
        "    mov.u32 %r3, %ctaid.x;\n"
        "    mov.u32 %r4, %ntid.x;\n"
        "    mov.u32 %r5, %tid.x;\n"
        "    mad.lo.s32 %r1, %r3, %r4, %r5;\n"
        "    setp.ge.s32 %p1, %r1, %r2;\n"
        "    @%p1 bra DONE;\n"
        "    cvta.to.global.u64 %rd4, %rd1;\n"
        "    mul.wide.s32 %rd5, %r1, 4;\n"
        "    add.s64 %rd6, %rd4, %rd5;\n"
        "    cvta.to.global.u64 %rd7, %rd2;\n"
        "    add.s64 %rd8, %rd7, %rd5;\n"
        "    ld.global.f32 %f1, [%rd6];\n"
        "    ld.global.f32 %f2, [%rd8];\n"
        "    div.rn.f32 %f3, %f1, %f2;\n"
        "    cvta.to.global.u64 %rd9, %rd3;\n"
        "    add.s64 %rd10, %rd9, %rd5;\n"
        "    st.global.f32 [%rd10], %f3;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    return __cuda_run_binary_kernel(ptxSrc, "div_kernel", a, b, out, n);
}

// out[i] = a[i]^b[i], computed as 2^(b[i] * log2(a[i])) via PTX's
// approximate lg2/ex2 special-function instructions -- there is no native
// PTX 'pow' instruction; this is the same exp2(y*log2(x)) decomposition a
// real compiler uses to lower powf() for CUDA. Approximate, not
// correctly-rounded (matches mps_pow_f32's precision tier, which uses
// Metal's native pow() and is more precise -- this one is intentionally
// the cheaper approximate-instruction version, not a claim of equal
// accuracy).
int cuda_pow_f32(const float* a, const float* b, float* out, unsigned long n) {
    const char* ptxSrc =
        ".version 7.0\n"
        ".target sm_50\n"
        ".address_size 64\n"
        ".visible .entry pow_kernel(\n"
        "    .param .u64 pow_kernel_param_0,\n"
        "    .param .u64 pow_kernel_param_1,\n"
        "    .param .u64 pow_kernel_param_2,\n"
        "    .param .u32 pow_kernel_param_3\n"
        ") {\n"
        "    .reg .pred %p<2>;\n"
        "    .reg .f32 %f<6>;\n"
        "    .reg .b32 %r<6>;\n"
        "    .reg .b64 %rd<11>;\n"
        "    ld.param.u64 %rd1, [pow_kernel_param_0];\n"
        "    ld.param.u64 %rd2, [pow_kernel_param_1];\n"
        "    ld.param.u64 %rd3, [pow_kernel_param_2];\n"
        "    ld.param.u32 %r2, [pow_kernel_param_3];\n"
        "    mov.u32 %r3, %ctaid.x;\n"
        "    mov.u32 %r4, %ntid.x;\n"
        "    mov.u32 %r5, %tid.x;\n"
        "    mad.lo.s32 %r1, %r3, %r4, %r5;\n"
        "    setp.ge.s32 %p1, %r1, %r2;\n"
        "    @%p1 bra DONE;\n"
        "    cvta.to.global.u64 %rd4, %rd1;\n"
        "    mul.wide.s32 %rd5, %r1, 4;\n"
        "    add.s64 %rd6, %rd4, %rd5;\n"
        "    cvta.to.global.u64 %rd7, %rd2;\n"
        "    add.s64 %rd8, %rd7, %rd5;\n"
        "    ld.global.f32 %f1, [%rd6];\n"
        "    ld.global.f32 %f2, [%rd8];\n"
        "    lg2.approx.f32 %f3, %f1;\n"
        "    mul.f32 %f4, %f2, %f3;\n"
        "    ex2.approx.f32 %f5, %f4;\n"
        "    cvta.to.global.u64 %rd9, %rd3;\n"
        "    add.s64 %rd10, %rd9, %rd5;\n"
        "    st.global.f32 [%rd10], %f5;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    return __cuda_run_binary_kernel(ptxSrc, "pow_kernel", a, b, out, n);
}

int cuda_relu_f32(const float* a, float* out, unsigned long n) {
    const char* ptxSrc =
        ".version 7.0\n"
        ".target sm_50\n"
        ".address_size 64\n"
        ".visible .entry relu_kernel(\n"
        "    .param .u64 relu_kernel_param_0,\n"
        "    .param .u64 relu_kernel_param_1,\n"
        "    .param .u32 relu_kernel_param_2\n"
        ") {\n"
        "    .reg .pred %p<2>;\n"
        "    .reg .f32 %f<4>;\n"
        "    .reg .b32 %r<6>;\n"
        "    .reg .b64 %rd<9>;\n"
        "    ld.param.u64 %rd1, [relu_kernel_param_0];\n"
        "    ld.param.u64 %rd2, [relu_kernel_param_1];\n"
        "    ld.param.u32 %r2, [relu_kernel_param_2];\n"
        "    mov.u32 %r3, %ctaid.x;\n"
        "    mov.u32 %r4, %ntid.x;\n"
        "    mov.u32 %r5, %tid.x;\n"
        "    mad.lo.s32 %r1, %r3, %r4, %r5;\n"
        "    setp.ge.s32 %p1, %r1, %r2;\n"
        "    @%p1 bra DONE;\n"
        "    cvta.to.global.u64 %rd4, %rd1;\n"
        "    mul.wide.s32 %rd5, %r1, 4;\n"
        "    add.s64 %rd6, %rd4, %rd5;\n"
        "    ld.global.f32 %f1, [%rd6];\n"
        "    max.f32 %f2, %f1, 0f00000000;\n"
        "    cvta.to.global.u64 %rd7, %rd2;\n"
        "    add.s64 %rd8, %rd7, %rd5;\n"
        "    st.global.f32 [%rd8], %f2;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    return __cuda_run_unary_kernel(ptxSrc, "relu_kernel", a, out, n);
}

int cuda_sqrt_f32(const float* a, float* out, unsigned long n) {
    const char* ptxSrc =
        ".version 7.0\n"
        ".target sm_50\n"
        ".address_size 64\n"
        ".visible .entry sqrt_kernel(\n"
        "    .param .u64 sqrt_kernel_param_0,\n"
        "    .param .u64 sqrt_kernel_param_1,\n"
        "    .param .u32 sqrt_kernel_param_2\n"
        ") {\n"
        "    .reg .pred %p<2>;\n"
        "    .reg .f32 %f<4>;\n"
        "    .reg .b32 %r<6>;\n"
        "    .reg .b64 %rd<9>;\n"
        "    ld.param.u64 %rd1, [sqrt_kernel_param_0];\n"
        "    ld.param.u64 %rd2, [sqrt_kernel_param_1];\n"
        "    ld.param.u32 %r2, [sqrt_kernel_param_2];\n"
        "    mov.u32 %r3, %ctaid.x;\n"
        "    mov.u32 %r4, %ntid.x;\n"
        "    mov.u32 %r5, %tid.x;\n"
        "    mad.lo.s32 %r1, %r3, %r4, %r5;\n"
        "    setp.ge.s32 %p1, %r1, %r2;\n"
        "    @%p1 bra DONE;\n"
        "    cvta.to.global.u64 %rd4, %rd1;\n"
        "    mul.wide.s32 %rd5, %r1, 4;\n"
        "    add.s64 %rd6, %rd4, %rd5;\n"
        "    ld.global.f32 %f1, [%rd6];\n"
        "    sqrt.rn.f32 %f2, %f1;\n"
        "    cvta.to.global.u64 %rd7, %rd2;\n"
        "    add.s64 %rd8, %rd7, %rd5;\n"
        "    st.global.f32 [%rd8], %f2;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    return __cuda_run_unary_kernel(ptxSrc, "sqrt_kernel", a, out, n);
}

// Natural log via PTX's approximate 'lg2' (log base 2) instruction --
// there is no native PTX ln instruction -- times ln(2): ln(x) = log2(x) *
// ln(2). 0f3F317218 is ln(2) (0.69314718...) as an IEEE-754 float32 hex
// literal, the standard constant real compilers embed for exactly this
// decomposition.
int cuda_log_f32(const float* a, float* out, unsigned long n) {
    const char* ptxSrc =
        ".version 7.0\n"
        ".target sm_50\n"
        ".address_size 64\n"
        ".visible .entry log_kernel(\n"
        "    .param .u64 log_kernel_param_0,\n"
        "    .param .u64 log_kernel_param_1,\n"
        "    .param .u32 log_kernel_param_2\n"
        ") {\n"
        "    .reg .pred %p<2>;\n"
        "    .reg .f32 %f<5>;\n"
        "    .reg .b32 %r<6>;\n"
        "    .reg .b64 %rd<9>;\n"
        "    ld.param.u64 %rd1, [log_kernel_param_0];\n"
        "    ld.param.u64 %rd2, [log_kernel_param_1];\n"
        "    ld.param.u32 %r2, [log_kernel_param_2];\n"
        "    mov.u32 %r3, %ctaid.x;\n"
        "    mov.u32 %r4, %ntid.x;\n"
        "    mov.u32 %r5, %tid.x;\n"
        "    mad.lo.s32 %r1, %r3, %r4, %r5;\n"
        "    setp.ge.s32 %p1, %r1, %r2;\n"
        "    @%p1 bra DONE;\n"
        "    cvta.to.global.u64 %rd4, %rd1;\n"
        "    mul.wide.s32 %rd5, %r1, 4;\n"
        "    add.s64 %rd6, %rd4, %rd5;\n"
        "    ld.global.f32 %f1, [%rd6];\n"
        "    lg2.approx.f32 %f2, %f1;\n"
        "    mul.f32 %f3, %f2, 0f3F317218;\n"
        "    cvta.to.global.u64 %rd7, %rd2;\n"
        "    add.s64 %rd8, %rd7, %rd5;\n"
        "    st.global.f32 [%rd8], %f3;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    return __cuda_run_unary_kernel(ptxSrc, "log_kernel", a, out, n);
}

// Natural exp via PTX's approximate 'ex2' (2^x) instruction -- there is no
// native PTX 'ex' instruction -- after scaling by log2(e): e^x =
// 2^(x*log2(e)). 0f3FB8AA3B is log2(e) (1.44269504...) as an IEEE-754
// float32 hex literal, same standard-constant reasoning as cuda_log_f32.
int cuda_exp_f32(const float* a, float* out, unsigned long n) {
    const char* ptxSrc =
        ".version 7.0\n"
        ".target sm_50\n"
        ".address_size 64\n"
        ".visible .entry exp_kernel(\n"
        "    .param .u64 exp_kernel_param_0,\n"
        "    .param .u64 exp_kernel_param_1,\n"
        "    .param .u32 exp_kernel_param_2\n"
        ") {\n"
        "    .reg .pred %p<2>;\n"
        "    .reg .f32 %f<5>;\n"
        "    .reg .b32 %r<6>;\n"
        "    .reg .b64 %rd<9>;\n"
        "    ld.param.u64 %rd1, [exp_kernel_param_0];\n"
        "    ld.param.u64 %rd2, [exp_kernel_param_1];\n"
        "    ld.param.u32 %r2, [exp_kernel_param_2];\n"
        "    mov.u32 %r3, %ctaid.x;\n"
        "    mov.u32 %r4, %ntid.x;\n"
        "    mov.u32 %r5, %tid.x;\n"
        "    mad.lo.s32 %r1, %r3, %r4, %r5;\n"
        "    setp.ge.s32 %p1, %r1, %r2;\n"
        "    @%p1 bra DONE;\n"
        "    cvta.to.global.u64 %rd4, %rd1;\n"
        "    mul.wide.s32 %rd5, %r1, 4;\n"
        "    add.s64 %rd6, %rd4, %rd5;\n"
        "    ld.global.f32 %f1, [%rd6];\n"
        "    mul.f32 %f2, %f1, 0f3FB8AA3B;\n"
        "    ex2.approx.f32 %f3, %f2;\n"
        "    cvta.to.global.u64 %rd7, %rd2;\n"
        "    add.s64 %rd8, %rd7, %rd5;\n"
        "    st.global.f32 [%rd8], %f3;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    return __cuda_run_unary_kernel(ptxSrc, "exp_kernel", a, out, n);
}

// out[i] = a[i] * k for a runtime scalar k -- 4 params like the binary
// kernels, but the 2nd param is a scalar f32 passed by value (.param .f32)
// instead of a 2nd input buffer.
int cuda_scale_f32(const float* a, float k, float* out, unsigned long n) {
    const char* ptxSrc =
        ".version 7.0\n"
        ".target sm_50\n"
        ".address_size 64\n"
        ".visible .entry scale_kernel(\n"
        "    .param .u64 scale_kernel_param_0,\n"
        "    .param .u64 scale_kernel_param_1,\n"
        "    .param .f32 scale_kernel_param_2,\n"
        "    .param .u32 scale_kernel_param_3\n"
        ") {\n"
        "    .reg .pred %p<2>;\n"
        "    .reg .f32 %f<4>;\n"
        "    .reg .b32 %r<6>;\n"
        "    .reg .b64 %rd<9>;\n"
        "    ld.param.u64 %rd1, [scale_kernel_param_0];\n"
        "    ld.param.u64 %rd2, [scale_kernel_param_1];\n"
        "    ld.param.f32 %f3, [scale_kernel_param_2];\n"
        "    ld.param.u32 %r2, [scale_kernel_param_3];\n"
        "    mov.u32 %r3, %ctaid.x;\n"
        "    mov.u32 %r4, %ntid.x;\n"
        "    mov.u32 %r5, %tid.x;\n"
        "    mad.lo.s32 %r1, %r3, %r4, %r5;\n"
        "    setp.ge.s32 %p1, %r1, %r2;\n"
        "    @%p1 bra DONE;\n"
        "    cvta.to.global.u64 %rd4, %rd1;\n"
        "    mul.wide.s32 %rd5, %r1, 4;\n"
        "    add.s64 %rd6, %rd4, %rd5;\n"
        "    ld.global.f32 %f1, [%rd6];\n"
        "    mul.f32 %f2, %f1, %f3;\n"
        "    cvta.to.global.u64 %rd7, %rd2;\n"
        "    add.s64 %rd8, %rd7, %rd5;\n"
        "    st.global.f32 [%rd8], %f2;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    unsafe {
        if (cuInit(0U) != CUDA_SUCCESS) return 0;
        int device;
        if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) return 0;
        void* ctx = (void*)0;
        if (cuCtxCreate_v2(&ctx, 0U, device) != CUDA_SUCCESS) return 0;

        void* module = (void*)0;
        if (cuModuleLoadData(&module, (const void*)ptxSrc) != CUDA_SUCCESS) {
            cuCtxDestroy_v2(ctx);
            return 0;
        }
        void* kernel = (void*)0;
        if (cuModuleGetFunction(&kernel, module, "scale_kernel") != CUDA_SUCCESS) {
            cuCtxDestroy_v2(ctx);
            return 0;
        }

        unsigned long bytes = n * sizeof(float);
        unsigned long long devA = 0ULL; unsigned long long devOut = 0ULL;
        cuMemAlloc_v2(&devA, bytes);
        cuMemAlloc_v2(&devOut, bytes);
        cuMemcpyHtoD_v2(devA, (const void*)a, bytes);

        float kParam = k;
        unsigned int nParam = (unsigned int)n;
        void* params[4];
        params[0] = (void*)&devA;
        params[1] = (void*)&devOut;
        params[2] = (void*)&kParam;
        params[3] = (void*)&nParam;

        unsigned int blockSize = 256U;
        unsigned int gridSize = (unsigned int)((n + (unsigned long)blockSize - 1UL) / (unsigned long)blockSize);
        int launchOk = cuLaunchKernel(kernel, gridSize, 1U, 1U, blockSize, 1U, 1U,
                                       0U, (void*)0, params, (void**)0) == CUDA_SUCCESS;
        cuCtxSynchronize();
        if (launchOk) cuMemcpyDtoH_v2((void*)out, devOut, bytes);

        cuMemFree_v2(devA);
        cuMemFree_v2(devOut);
        cuCtxDestroy_v2(ctx);
        return launchOk ? 1 : 0;
    }
}

// out[0] = sum(a[0..n)) -- single-thread serial reduction (thread 0 of
// block 0 only; every other thread returns immediately), same "prove the
// dispatch shape works, not a tuned parallel reduction" spirit as
// mps_sum_f32.
int cuda_sum_f32(const float* a, float* out, unsigned long n) {
    const char* ptxSrc =
        ".version 7.0\n"
        ".target sm_50\n"
        ".address_size 64\n"
        ".visible .entry sum_kernel(\n"
        "    .param .u64 sum_kernel_param_0,\n"
        "    .param .u64 sum_kernel_param_1,\n"
        "    .param .u32 sum_kernel_param_2\n"
        ") {\n"
        "    .reg .pred %p<3>;\n"
        "    .reg .f32 %f<4>;\n"
        "    .reg .b32 %r<8>;\n"
        "    .reg .b64 %rd<8>;\n"
        "    ld.param.u64 %rd1, [sum_kernel_param_0];\n"
        "    ld.param.u64 %rd2, [sum_kernel_param_1];\n"
        "    ld.param.u32 %r1, [sum_kernel_param_2];\n"
        "    mov.u32 %r2, %ctaid.x;\n"
        "    mov.u32 %r3, %tid.x;\n"
        "    or.b32 %r4, %r2, %r3;\n"
        "    setp.ne.s32 %p1, %r4, 0;\n"
        "    @%p1 bra DONE;\n"
        "    cvta.to.global.u64 %rd3, %rd1;\n"
        "    mov.f32 %f1, 0f00000000;\n"
        "    mov.u32 %r5, 0;\n"
        "LOOP:\n"
        "    setp.ge.s32 %p2, %r5, %r1;\n"
        "    @%p2 bra STORE;\n"
        "    mul.wide.s32 %rd4, %r5, 4;\n"
        "    add.s64 %rd5, %rd3, %rd4;\n"
        "    ld.global.f32 %f2, [%rd5];\n"
        "    add.f32 %f1, %f1, %f2;\n"
        "    add.s32 %r5, %r5, 1;\n"
        "    bra.uni LOOP;\n"
        "STORE:\n"
        "    cvta.to.global.u64 %rd6, %rd2;\n"
        "    st.global.f32 [%rd6], %f1;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    unsafe {
        if (cuInit(0U) != CUDA_SUCCESS) return 0;
        int device;
        if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) return 0;
        void* ctx = (void*)0;
        if (cuCtxCreate_v2(&ctx, 0U, device) != CUDA_SUCCESS) return 0;

        void* module = (void*)0;
        if (cuModuleLoadData(&module, (const void*)ptxSrc) != CUDA_SUCCESS) {
            cuCtxDestroy_v2(ctx);
            return 0;
        }
        void* kernel = (void*)0;
        if (cuModuleGetFunction(&kernel, module, "sum_kernel") != CUDA_SUCCESS) {
            cuCtxDestroy_v2(ctx);
            return 0;
        }

        unsigned long bytes = n * sizeof(float);
        unsigned long long devA = 0ULL; unsigned long long devOut = 0ULL;
        cuMemAlloc_v2(&devA, bytes);
        cuMemAlloc_v2(&devOut, 4UL);
        cuMemcpyHtoD_v2(devA, (const void*)a, bytes);

        unsigned int nParam = (unsigned int)n;
        void* params[3];
        params[0] = (void*)&devA;
        params[1] = (void*)&devOut;
        params[2] = (void*)&nParam;

        int launchOk = cuLaunchKernel(kernel, 1U, 1U, 1U, 1U, 1U, 1U,
                                       0U, (void*)0, params, (void**)0) == CUDA_SUCCESS;
        cuCtxSynchronize();
        if (launchOk) cuMemcpyDtoH_v2((void*)out, devOut, 4UL);

        cuMemFree_v2(devA);
        cuMemFree_v2(devOut);
        cuCtxDestroy_v2(ctx);
        return launchOk ? 1 : 0;
    }
}

// out[M,N] = a[M,K] . b[K,N] -- naive one-thread-per-output-element kernel
// over a 2D grid (blockIdx/threadIdx.x -> column, .y -> row), no
// threadgroup/shared-memory tiling, same tradeoff as mps_matmul_f32.
int cuda_matmul_f32(const float* a, const float* b, float* out,
                     unsigned long M, unsigned long K, unsigned long N) {
    const char* ptxSrc =
        ".version 7.0\n"
        ".target sm_50\n"
        ".address_size 64\n"
        ".visible .entry matmul_kernel(\n"
        "    .param .u64 matmul_kernel_param_0,\n"
        "    .param .u64 matmul_kernel_param_1,\n"
        "    .param .u64 matmul_kernel_param_2,\n"
        "    .param .u32 matmul_kernel_param_3,\n"
        "    .param .u32 matmul_kernel_param_4,\n"
        "    .param .u32 matmul_kernel_param_5\n"
        ") {\n"
        "    .reg .pred %p<4>;\n"
        "    .reg .f32 %f<6>;\n"
        "    .reg .b32 %r<20>;\n"
        "    .reg .b64 %rd<20>;\n"
        "    ld.param.u64 %rd1, [matmul_kernel_param_0];\n"
        "    ld.param.u64 %rd2, [matmul_kernel_param_1];\n"
        "    ld.param.u64 %rd3, [matmul_kernel_param_2];\n"
        "    ld.param.u32 %r1, [matmul_kernel_param_3];\n"
        "    ld.param.u32 %r2, [matmul_kernel_param_4];\n"
        "    ld.param.u32 %r3, [matmul_kernel_param_5];\n"
        "    mov.u32 %r4, %ctaid.x;\n"
        "    mov.u32 %r5, %ntid.x;\n"
        "    mov.u32 %r6, %tid.x;\n"
        "    mad.lo.s32 %r7, %r4, %r5, %r6;\n"
        "    mov.u32 %r8, %ctaid.y;\n"
        "    mov.u32 %r9, %ntid.y;\n"
        "    mov.u32 %r10, %tid.y;\n"
        "    mad.lo.s32 %r11, %r8, %r9, %r10;\n"
        "    setp.ge.s32 %p1, %r7, %r3;\n"
        "    setp.ge.s32 %p2, %r11, %r1;\n"
        "    or.pred %p3, %p1, %p2;\n"
        "    @%p3 bra DONE;\n"
        "    cvta.to.global.u64 %rd4, %rd1;\n"
        "    cvta.to.global.u64 %rd5, %rd2;\n"
        "    mov.f32 %f1, 0f00000000;\n"
        "    mov.u32 %r12, 0;\n"
        "LOOP:\n"
        "    setp.ge.s32 %p1, %r12, %r2;\n"
        "    @%p1 bra STORE;\n"
        "    mul.lo.s32 %r13, %r11, %r2;\n"
        "    add.s32 %r13, %r13, %r12;\n"
        "    mul.wide.s32 %rd6, %r13, 4;\n"
        "    add.s64 %rd7, %rd4, %rd6;\n"
        "    ld.global.f32 %f2, [%rd7];\n"
        "    mul.lo.s32 %r14, %r12, %r3;\n"
        "    add.s32 %r14, %r14, %r7;\n"
        "    mul.wide.s32 %rd8, %r14, 4;\n"
        "    add.s64 %rd9, %rd5, %rd8;\n"
        "    ld.global.f32 %f3, [%rd9];\n"
        "    fma.rn.f32 %f1, %f2, %f3, %f1;\n"
        "    add.s32 %r12, %r12, 1;\n"
        "    bra.uni LOOP;\n"
        "STORE:\n"
        "    mul.lo.s32 %r15, %r11, %r3;\n"
        "    add.s32 %r15, %r15, %r7;\n"
        "    mul.wide.s32 %rd10, %r15, 4;\n"
        "    cvta.to.global.u64 %rd11, %rd3;\n"
        "    add.s64 %rd12, %rd11, %rd10;\n"
        "    st.global.f32 [%rd12], %f1;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    unsafe {
        if (cuInit(0U) != CUDA_SUCCESS) return 0;
        int device;
        if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) return 0;
        void* ctx = (void*)0;
        if (cuCtxCreate_v2(&ctx, 0U, device) != CUDA_SUCCESS) return 0;

        void* module = (void*)0;
        if (cuModuleLoadData(&module, (const void*)ptxSrc) != CUDA_SUCCESS) {
            cuCtxDestroy_v2(ctx);
            return 0;
        }
        void* kernel = (void*)0;
        if (cuModuleGetFunction(&kernel, module, "matmul_kernel") != CUDA_SUCCESS) {
            cuCtxDestroy_v2(ctx);
            return 0;
        }

        unsigned long bytesA = M * K * sizeof(float);
        unsigned long bytesB = K * N * sizeof(float);
        unsigned long bytesOut = M * N * sizeof(float);
        unsigned long long devA = 0ULL; unsigned long long devB = 0ULL; unsigned long long devOut = 0ULL;
        cuMemAlloc_v2(&devA, bytesA);
        cuMemAlloc_v2(&devB, bytesB);
        cuMemAlloc_v2(&devOut, bytesOut);
        cuMemcpyHtoD_v2(devA, (const void*)a, bytesA);
        cuMemcpyHtoD_v2(devB, (const void*)b, bytesB);

        unsigned int Mu = (unsigned int)M;
        unsigned int Ku = (unsigned int)K;
        unsigned int Nu = (unsigned int)N;
        void* params[6];
        params[0] = (void*)&devA;
        params[1] = (void*)&devB;
        params[2] = (void*)&devOut;
        params[3] = (void*)&Mu;
        params[4] = (void*)&Ku;
        params[5] = (void*)&Nu;

        unsigned int tgW = 16U;
        unsigned int tgH = 16U;
        unsigned int gridX = (unsigned int)((N + (unsigned long)tgW - 1UL) / (unsigned long)tgW);
        unsigned int gridY = (unsigned int)((M + (unsigned long)tgH - 1UL) / (unsigned long)tgH);
        int launchOk = cuLaunchKernel(kernel, gridX, gridY, 1U, tgW, tgH, 1U,
                                       0U, (void*)0, params, (void**)0) == CUDA_SUCCESS;
        cuCtxSynchronize();
        if (launchOk) cuMemcpyDtoH_v2((void*)out, devOut, bytesOut);

        cuMemFree_v2(devA);
        cuMemFree_v2(devB);
        cuMemFree_v2(devOut);
        cuCtxDestroy_v2(ctx);
        return launchOk ? 1 : 0;
    }
}

// out[M,N] = a[M,K] . b[K,N] via cublasSgemm instead of the hand-written
// PTX kernel above -- same device setup (init/context/memory), different
// compute dispatch.
//
// cuBLAS follows Fortran/BLAS convention: column-major, and it computes
// C = alpha*op(A)*op(B) + beta*C where A is m-by-k, B is k-by-n, C is
// m-by-n, all COLUMN-major. SafeC's Tensor is row-major throughout. The
// standard trick for calling a column-major GEMM on row-major data
// without physically transposing anything: a row-major M*K matrix IS a
// column-major K*M matrix over the same bytes (and vice versa), so
// instead of computing row-major C(M,N) = A(M,K)*B(K,N) directly, this
// computes column-major C^T(N,M) = B^T(N,K) * A^T(K,M) -- which is the
// exact same bytes as row-major C(M,N), just asking cuBLAS to compute
// "B times A" (operands swapped) with M/N swapped too, both still
// CUBLAS_OP_N since no operand is actually transposed, only reinterpreted.
// (Same trick tensor_blas.sc's Accelerate path does NOT need, since
// Apple's CBLAS accepts an explicit row-major order flag directly.)
int cuda_matmul_f32_blas(const float* a, const float* b, float* out,
                          unsigned long M, unsigned long K, unsigned long N) {
    unsafe {
        if (cuInit(0U) != CUDA_SUCCESS) return 0;
        int device;
        if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) return 0;
        void* ctx = (void*)0;
        if (cuCtxCreate_v2(&ctx, 0U, device) != CUDA_SUCCESS) return 0;

        void* handle = (void*)0;
        if (cublasCreate_v2(&handle) != CUDA_SUCCESS) {
            cuCtxDestroy_v2(ctx);
            return 0;
        }

        unsigned long bytesA = M * K * sizeof(float);
        unsigned long bytesB = K * N * sizeof(float);
        unsigned long bytesOut = M * N * sizeof(float);
        unsigned long long devA = 0ULL; unsigned long long devB = 0ULL; unsigned long long devOut = 0ULL;
        cuMemAlloc_v2(&devA, bytesA);
        cuMemAlloc_v2(&devB, bytesB);
        cuMemAlloc_v2(&devOut, bytesOut);
        cuMemcpyHtoD_v2(devA, (const void*)a, bytesA);
        cuMemcpyHtoD_v2(devB, (const void*)b, bytesB);

        float alpha = 1.0f;
        float beta = 0.0f;
        int Mi = (int)M; int Ki = (int)K; int Ni = (int)N;
        int status = cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            Ni, Mi, Ki,
            (const float*)&alpha,
            (const float*)(unsigned long)devB, Ni,
            (const float*)(unsigned long)devA, Ki,
            (const float*)&beta,
            (float*)(unsigned long)devOut, Ni);
        int ok = (status == CUDA_SUCCESS);
        if (ok) cuMemcpyDtoH_v2((void*)out, devOut, bytesOut);

        cuMemFree_v2(devA);
        cuMemFree_v2(devB);
        cuMemFree_v2(devOut);
        cublasDestroy_v2(handle);
        cuCtxDestroy_v2(ctx);
        return ok ? 1 : 0;
    }
}

// ── Persistent-context / GPU-resident tier (see gpu_cuda.h) ─────────────────
static void* __cuda_persistent_ctx_ = (void*)0;
static int __cuda_persistent_failed_ = 0;
static void* __cuda_matmul_kernel_ = (void*)0;
static void* __cuda_sgd_kernel_ = (void*)0;

// Creates the shared CUcontext once and reuses it forever after -- the
// direct fix for the "full cuCtxCreate_v2/cuCtxDestroy_v2 every call"
// pattern every function above this section pays.
static int __cuda_ensure_context() {
    if (__cuda_persistent_ctx_ != (void*)0) return 1;
    if (__cuda_persistent_failed_) return 0;
    unsafe {
        if (cuInit(0U) != CUDA_SUCCESS) { __cuda_persistent_failed_ = 1; return 0; }
        int device;
        if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) { __cuda_persistent_failed_ = 1; return 0; }
        void* ctx = (void*)0;
        if (cuCtxCreate_v2(&ctx, 0U, device) != CUDA_SUCCESS) { __cuda_persistent_failed_ = 1; return 0; }
        __cuda_persistent_ctx_ = ctx;
        return 1;
    }
}

int cuda_persistent_available() {
    return __cuda_ensure_context();
}

void* cuda_upload_persistent(const float* data, unsigned long bytes) {
    if (!__cuda_ensure_context()) return (void*)0;
    unsafe {
        unsigned long long devPtr = 0ULL;
        if (cuMemAlloc_v2(&devPtr, bytes) != CUDA_SUCCESS) return (void*)0;
        if (cuMemcpyHtoD_v2(devPtr, (const void*)data, bytes) != CUDA_SUCCESS) {
            cuMemFree_v2(devPtr);
            return (void*)0;
        }
        return (void*)(unsigned long)devPtr;
    }
}

void cuda_release_persistent(void* devPtr) {
    unsafe { cuMemFree_v2((unsigned long long)(unsigned long)devPtr); }
}

int cuda_matmul_f32_persistent(void* devA, void* devB, float* out,
                                unsigned long M, unsigned long K, unsigned long N) {
    if (!__cuda_ensure_context()) return 0;
    // Same naive one-thread-per-output-element kernel as cuda_matmul_f32
    // -- duplicated here (not shared via a helper) rather than
    // refactoring cuda_matmul_f32 itself, to keep this tier purely
    // additive and leave the already-written, already-reviewed function
    // above untouched.
    const char* ptxSrc =
        ".version 7.0\n"
        ".target sm_50\n"
        ".address_size 64\n"
        ".visible .entry matmul_kernel(\n"
        "    .param .u64 matmul_kernel_param_0,\n"
        "    .param .u64 matmul_kernel_param_1,\n"
        "    .param .u64 matmul_kernel_param_2,\n"
        "    .param .u32 matmul_kernel_param_3,\n"
        "    .param .u32 matmul_kernel_param_4,\n"
        "    .param .u32 matmul_kernel_param_5\n"
        ") {\n"
        "    .reg .pred %p<4>;\n"
        "    .reg .f32 %f<6>;\n"
        "    .reg .b32 %r<20>;\n"
        "    .reg .b64 %rd<20>;\n"
        "    ld.param.u64 %rd1, [matmul_kernel_param_0];\n"
        "    ld.param.u64 %rd2, [matmul_kernel_param_1];\n"
        "    ld.param.u64 %rd3, [matmul_kernel_param_2];\n"
        "    ld.param.u32 %r1, [matmul_kernel_param_3];\n"
        "    ld.param.u32 %r2, [matmul_kernel_param_4];\n"
        "    ld.param.u32 %r3, [matmul_kernel_param_5];\n"
        "    mov.u32 %r4, %ctaid.x;\n"
        "    mov.u32 %r5, %ntid.x;\n"
        "    mov.u32 %r6, %tid.x;\n"
        "    mad.lo.s32 %r7, %r4, %r5, %r6;\n"
        "    mov.u32 %r8, %ctaid.y;\n"
        "    mov.u32 %r9, %ntid.y;\n"
        "    mov.u32 %r10, %tid.y;\n"
        "    mad.lo.s32 %r11, %r8, %r9, %r10;\n"
        "    setp.ge.s32 %p1, %r7, %r3;\n"
        "    setp.ge.s32 %p2, %r11, %r1;\n"
        "    or.pred %p3, %p1, %p2;\n"
        "    @%p3 bra DONE;\n"
        "    cvta.to.global.u64 %rd4, %rd1;\n"
        "    cvta.to.global.u64 %rd5, %rd2;\n"
        "    mov.f32 %f1, 0f00000000;\n"
        "    mov.u32 %r12, 0;\n"
        "LOOP:\n"
        "    setp.ge.s32 %p1, %r12, %r2;\n"
        "    @%p1 bra STORE;\n"
        "    mul.lo.s32 %r13, %r11, %r2;\n"
        "    add.s32 %r13, %r13, %r12;\n"
        "    mul.wide.s32 %rd6, %r13, 4;\n"
        "    add.s64 %rd7, %rd4, %rd6;\n"
        "    ld.global.f32 %f2, [%rd7];\n"
        "    mul.lo.s32 %r14, %r12, %r3;\n"
        "    add.s32 %r14, %r14, %r7;\n"
        "    mul.wide.s32 %rd8, %r14, 4;\n"
        "    add.s64 %rd9, %rd5, %rd8;\n"
        "    ld.global.f32 %f3, [%rd9];\n"
        "    fma.rn.f32 %f1, %f2, %f3, %f1;\n"
        "    add.s32 %r12, %r12, 1;\n"
        "    bra.uni LOOP;\n"
        "STORE:\n"
        "    mul.lo.s32 %r15, %r11, %r3;\n"
        "    add.s32 %r15, %r15, %r7;\n"
        "    mul.wide.s32 %rd10, %r15, 4;\n"
        "    cvta.to.global.u64 %rd11, %rd3;\n"
        "    add.s64 %rd12, %rd11, %rd10;\n"
        "    st.global.f32 [%rd12], %f1;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    unsafe {
        if (__cuda_matmul_kernel_ == (void*)0) {
            void* module = (void*)0;
            if (cuModuleLoadData(&module, (const void*)ptxSrc) != CUDA_SUCCESS) return 0;
            void* kernel = (void*)0;
            if (cuModuleGetFunction(&kernel, module, "matmul_kernel") != CUDA_SUCCESS) return 0;
            __cuda_matmul_kernel_ = kernel;
        }

        unsigned long long devA64 = (unsigned long long)(unsigned long)devA;
        unsigned long long devB64 = (unsigned long long)(unsigned long)devB;
        unsigned long bytesOut = M * N * sizeof(float);
        unsigned long long devOut = 0ULL;
        if (cuMemAlloc_v2(&devOut, bytesOut) != CUDA_SUCCESS) return 0;

        unsigned int Mu = (unsigned int)M;
        unsigned int Ku = (unsigned int)K;
        unsigned int Nu = (unsigned int)N;
        void* params[6];
        params[0] = (void*)&devA64;
        params[1] = (void*)&devB64;
        params[2] = (void*)&devOut;
        params[3] = (void*)&Mu;
        params[4] = (void*)&Ku;
        params[5] = (void*)&Nu;

        unsigned int tgW = 16U;
        unsigned int tgH = 16U;
        unsigned int gridX = (unsigned int)((N + (unsigned long)tgW - 1UL) / (unsigned long)tgW);
        unsigned int gridY = (unsigned int)((M + (unsigned long)tgH - 1UL) / (unsigned long)tgH);
        int launchOk = cuLaunchKernel(__cuda_matmul_kernel_, gridX, gridY, 1U, tgW, tgH, 1U,
                                       0U, (void*)0, params, (void**)0) == CUDA_SUCCESS;
        cuCtxSynchronize();
        if (launchOk) cuMemcpyDtoH_v2((void*)out, devOut, bytesOut);
        cuMemFree_v2(devOut);
        return launchOk ? 1 : 0;
    }
}

// w[i] -= lr*grad[i], in place. fma.rn.f32 computes d = a*b+c, so this
// negates lr once (neg.f32) and folds the update into a single
// fused-multiply-add: w_new = grad*(-lr) + w.
int cuda_sgd_update_f32(void* devW, void* devGrad, float lr, unsigned long n) {
    if (!__cuda_ensure_context()) return 0;
    const char* ptxSrc =
        ".version 7.0\n"
        ".target sm_50\n"
        ".address_size 64\n"
        ".visible .entry sgd_update_kernel(\n"
        "    .param .u64 sgd_update_kernel_param_0,\n"
        "    .param .u64 sgd_update_kernel_param_1,\n"
        "    .param .f32 sgd_update_kernel_param_2,\n"
        "    .param .u32 sgd_update_kernel_param_3\n"
        ") {\n"
        "    .reg .pred %p<2>;\n"
        "    .reg .f32 %f<6>;\n"
        "    .reg .b32 %r<6>;\n"
        "    .reg .b64 %rd<8>;\n"
        "    ld.param.u64 %rd1, [sgd_update_kernel_param_0];\n"
        "    ld.param.u64 %rd2, [sgd_update_kernel_param_1];\n"
        "    ld.param.f32 %f3, [sgd_update_kernel_param_2];\n"
        "    ld.param.u32 %r2, [sgd_update_kernel_param_3];\n"
        "    mov.u32 %r3, %ctaid.x;\n"
        "    mov.u32 %r4, %ntid.x;\n"
        "    mov.u32 %r5, %tid.x;\n"
        "    mad.lo.s32 %r1, %r3, %r4, %r5;\n"
        "    setp.ge.s32 %p1, %r1, %r2;\n"
        "    @%p1 bra DONE;\n"
        "    cvta.to.global.u64 %rd3, %rd1;\n"
        "    mul.wide.s32 %rd4, %r1, 4;\n"
        "    add.s64 %rd5, %rd3, %rd4;\n"
        "    ld.global.f32 %f1, [%rd5];\n"
        "    cvta.to.global.u64 %rd6, %rd2;\n"
        "    add.s64 %rd7, %rd6, %rd4;\n"
        "    ld.global.f32 %f2, [%rd7];\n"
        "    neg.f32 %f4, %f3;\n"
        "    fma.rn.f32 %f5, %f2, %f4, %f1;\n"
        "    st.global.f32 [%rd5], %f5;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    unsafe {
        if (__cuda_sgd_kernel_ == (void*)0) {
            void* module = (void*)0;
            if (cuModuleLoadData(&module, (const void*)ptxSrc) != CUDA_SUCCESS) return 0;
            void* kernel = (void*)0;
            if (cuModuleGetFunction(&kernel, module, "sgd_update_kernel") != CUDA_SUCCESS) return 0;
            __cuda_sgd_kernel_ = kernel;
        }

        unsigned long long devW64 = (unsigned long long)(unsigned long)devW;
        unsigned long long devGrad64 = (unsigned long long)(unsigned long)devGrad;
        float lrVal = lr;
        unsigned int nParam = (unsigned int)n;
        void* params[4];
        params[0] = (void*)&devW64;
        params[1] = (void*)&devGrad64;
        params[2] = (void*)&lrVal;
        params[3] = (void*)&nParam;

        unsigned int blockSize = 256U;
        unsigned int gridSize = (unsigned int)((n + (unsigned long)blockSize - 1UL) / (unsigned long)blockSize);
        int launchOk = cuLaunchKernel(__cuda_sgd_kernel_, gridSize, 1U, 1U, blockSize, 1U, 1U,
                                       0U, (void*)0, params, (void**)0) == CUDA_SUCCESS;
        cuCtxSynchronize();
        return launchOk ? 1 : 0;
    }
}

extern void* malloc(unsigned long size);
extern void  free(void* ptr);

// ── fp16 / bf16 tier (see gpu_cuda.h) ────────────────────────────────────────
void* cuda_upload_f16_persistent(const float* data, unsigned long n) {
    unsafe {
        unsigned short* tmp = (unsigned short*)malloc(n * 2UL);
        if (tmp == (void*)0) return (void*)0;
        f32_to_fp16_bulk(data, tmp, n);
        void* buf = cuda_upload_persistent((const float*)tmp, n * 2UL);
        free((void*)tmp);
        return buf;
    }
}

void* cuda_upload_bf16_persistent(const float* data, unsigned long n) {
    unsafe {
        unsigned short* tmp = (unsigned short*)malloc(n * 2UL);
        if (tmp == (void*)0) return (void*)0;
        f32_to_bf16_bulk(data, tmp, n);
        void* buf = cuda_upload_persistent((const float*)tmp, n * 2UL);
        free((void*)tmp);
        return buf;
    }
}

static void* __cuda_matmul_f16_kernel_ = (void*)0;
static void* __cuda_matmul_bf16_kernel_ = (void*)0;
static void* __cuda_sgd_f16_kernel_ = (void*)0;
static void* __cuda_sgd_bf16_kernel_ = (void*)0;

// out = devA[M,K] . devB[K,N] -- same naive one-thread-per-output-element
// shape as cuda_matmul_f32_persistent, but 'devA'/'devB' hold fp16 bits
// and the kernel promotes each loaded operand to f32 with 'cvt.f32.f16'
// before the fma, converting the f32 accumulator back with
// 'cvt.rn.f16.f32' only once, at the final store -- the same storage-
// half/compute-float split gpu_mps_kernels.metal's matmul_kernel_f16
// uses. Targets sm_60 (Pascal+): PTX's f16 storage/cvt instructions need
// real f16 hardware datapaths, unlike this file's f32 kernels' sm_50.
int cuda_matmul_f16_persistent(void* devA, void* devB, unsigned short* out,
                                unsigned long M, unsigned long K, unsigned long N) {
    if (!__cuda_ensure_context()) return 0;
    const char* ptxSrc =
        ".version 7.0\n"
        ".target sm_60\n"
        ".address_size 64\n"
        ".visible .entry matmul_kernel_f16(\n"
        "    .param .u64 matmul_kernel_f16_param_0,\n"
        "    .param .u64 matmul_kernel_f16_param_1,\n"
        "    .param .u64 matmul_kernel_f16_param_2,\n"
        "    .param .u32 matmul_kernel_f16_param_3,\n"
        "    .param .u32 matmul_kernel_f16_param_4,\n"
        "    .param .u32 matmul_kernel_f16_param_5\n"
        ") {\n"
        "    .reg .pred %p<4>;\n"
        "    .reg .f16 %h<4>;\n"
        "    .reg .f32 %f<6>;\n"
        "    .reg .b32 %r<20>;\n"
        "    .reg .b64 %rd<20>;\n"
        "    ld.param.u64 %rd1, [matmul_kernel_f16_param_0];\n"
        "    ld.param.u64 %rd2, [matmul_kernel_f16_param_1];\n"
        "    ld.param.u64 %rd3, [matmul_kernel_f16_param_2];\n"
        "    ld.param.u32 %r1, [matmul_kernel_f16_param_3];\n"
        "    ld.param.u32 %r2, [matmul_kernel_f16_param_4];\n"
        "    ld.param.u32 %r3, [matmul_kernel_f16_param_5];\n"
        "    mov.u32 %r4, %ctaid.x;\n"
        "    mov.u32 %r5, %ntid.x;\n"
        "    mov.u32 %r6, %tid.x;\n"
        "    mad.lo.s32 %r7, %r4, %r5, %r6;\n"
        "    mov.u32 %r8, %ctaid.y;\n"
        "    mov.u32 %r9, %ntid.y;\n"
        "    mov.u32 %r10, %tid.y;\n"
        "    mad.lo.s32 %r11, %r8, %r9, %r10;\n"
        "    setp.ge.s32 %p1, %r7, %r3;\n"
        "    setp.ge.s32 %p2, %r11, %r1;\n"
        "    or.pred %p3, %p1, %p2;\n"
        "    @%p3 bra DONE;\n"
        "    cvta.to.global.u64 %rd4, %rd1;\n"
        "    cvta.to.global.u64 %rd5, %rd2;\n"
        "    mov.f32 %f1, 0f00000000;\n"
        "    mov.u32 %r12, 0;\n"
        "LOOP:\n"
        "    setp.ge.s32 %p1, %r12, %r2;\n"
        "    @%p1 bra STORE;\n"
        "    mul.lo.s32 %r13, %r11, %r2;\n"
        "    add.s32 %r13, %r13, %r12;\n"
        "    mul.wide.s32 %rd6, %r13, 2;\n"
        "    add.s64 %rd7, %rd4, %rd6;\n"
        "    ld.global.f16 %h1, [%rd7];\n"
        "    cvt.f32.f16 %f2, %h1;\n"
        "    mul.lo.s32 %r14, %r12, %r3;\n"
        "    add.s32 %r14, %r14, %r7;\n"
        "    mul.wide.s32 %rd8, %r14, 2;\n"
        "    add.s64 %rd9, %rd5, %rd8;\n"
        "    ld.global.f16 %h2, [%rd9];\n"
        "    cvt.f32.f16 %f3, %h2;\n"
        "    fma.rn.f32 %f1, %f2, %f3, %f1;\n"
        "    add.s32 %r12, %r12, 1;\n"
        "    bra.uni LOOP;\n"
        "STORE:\n"
        "    mul.lo.s32 %r15, %r11, %r3;\n"
        "    add.s32 %r15, %r15, %r7;\n"
        "    mul.wide.s32 %rd10, %r15, 2;\n"
        "    cvta.to.global.u64 %rd11, %rd3;\n"
        "    add.s64 %rd12, %rd11, %rd10;\n"
        "    cvt.rn.f16.f32 %h3, %f1;\n"
        "    st.global.f16 [%rd12], %h3;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    unsafe {
        if (__cuda_matmul_f16_kernel_ == (void*)0) {
            void* module = (void*)0;
            if (cuModuleLoadData(&module, (const void*)ptxSrc) != CUDA_SUCCESS) return 0;
            void* kernel = (void*)0;
            if (cuModuleGetFunction(&kernel, module, "matmul_kernel_f16") != CUDA_SUCCESS) return 0;
            __cuda_matmul_f16_kernel_ = kernel;
        }

        unsigned long long devA64 = (unsigned long long)(unsigned long)devA;
        unsigned long long devB64 = (unsigned long long)(unsigned long)devB;
        unsigned long bytesOut = M * N * 2UL;
        unsigned long long devOut = 0ULL;
        if (cuMemAlloc_v2(&devOut, bytesOut) != CUDA_SUCCESS) return 0;

        unsigned int Mu = (unsigned int)M;
        unsigned int Ku = (unsigned int)K;
        unsigned int Nu = (unsigned int)N;
        void* params[6];
        params[0] = (void*)&devA64;
        params[1] = (void*)&devB64;
        params[2] = (void*)&devOut;
        params[3] = (void*)&Mu;
        params[4] = (void*)&Ku;
        params[5] = (void*)&Nu;

        unsigned int tgW = 16U;
        unsigned int tgH = 16U;
        unsigned int gridX = (unsigned int)((N + (unsigned long)tgW - 1UL) / (unsigned long)tgW);
        unsigned int gridY = (unsigned int)((M + (unsigned long)tgH - 1UL) / (unsigned long)tgH);
        int launchOk = cuLaunchKernel(__cuda_matmul_f16_kernel_, gridX, gridY, 1U, tgW, tgH, 1U,
                                       0U, (void*)0, params, (void**)0) == CUDA_SUCCESS;
        cuCtxSynchronize();
        if (launchOk) cuMemcpyDtoH_v2((void*)out, devOut, bytesOut);
        cuMemFree_v2(devOut);
        return launchOk ? 1 : 0;
    }
}

// Same shape as cuda_matmul_f16_persistent, storage in '.bf16' instead of
// '.f16'. Targets sm_80 (Ampere+): bf16 is genuinely a newer NVIDIA GPU
// feature than f16 (introduced with the Ampere architecture in 2020),
// not an arbitrary stricter choice -- PTX's bf16 storage/cvt
// instructions need PTX ISA 7.8, hence the '.version 7.8' bump too
// (every other kernel in this file, including matmul_kernel_f16 above,
// only needs 7.0).
int cuda_matmul_bf16_persistent(void* devA, void* devB, unsigned short* out,
                                 unsigned long M, unsigned long K, unsigned long N) {
    if (!__cuda_ensure_context()) return 0;
    const char* ptxSrc =
        ".version 7.8\n"
        ".target sm_80\n"
        ".address_size 64\n"
        ".visible .entry matmul_kernel_bf16(\n"
        "    .param .u64 matmul_kernel_bf16_param_0,\n"
        "    .param .u64 matmul_kernel_bf16_param_1,\n"
        "    .param .u64 matmul_kernel_bf16_param_2,\n"
        "    .param .u32 matmul_kernel_bf16_param_3,\n"
        "    .param .u32 matmul_kernel_bf16_param_4,\n"
        "    .param .u32 matmul_kernel_bf16_param_5\n"
        ") {\n"
        "    .reg .pred %p<4>;\n"
        "    .reg .bf16 %b<4>;\n"
        "    .reg .f32 %f<6>;\n"
        "    .reg .b32 %r<20>;\n"
        "    .reg .b64 %rd<20>;\n"
        "    ld.param.u64 %rd1, [matmul_kernel_bf16_param_0];\n"
        "    ld.param.u64 %rd2, [matmul_kernel_bf16_param_1];\n"
        "    ld.param.u64 %rd3, [matmul_kernel_bf16_param_2];\n"
        "    ld.param.u32 %r1, [matmul_kernel_bf16_param_3];\n"
        "    ld.param.u32 %r2, [matmul_kernel_bf16_param_4];\n"
        "    ld.param.u32 %r3, [matmul_kernel_bf16_param_5];\n"
        "    mov.u32 %r4, %ctaid.x;\n"
        "    mov.u32 %r5, %ntid.x;\n"
        "    mov.u32 %r6, %tid.x;\n"
        "    mad.lo.s32 %r7, %r4, %r5, %r6;\n"
        "    mov.u32 %r8, %ctaid.y;\n"
        "    mov.u32 %r9, %ntid.y;\n"
        "    mov.u32 %r10, %tid.y;\n"
        "    mad.lo.s32 %r11, %r8, %r9, %r10;\n"
        "    setp.ge.s32 %p1, %r7, %r3;\n"
        "    setp.ge.s32 %p2, %r11, %r1;\n"
        "    or.pred %p3, %p1, %p2;\n"
        "    @%p3 bra DONE;\n"
        "    cvta.to.global.u64 %rd4, %rd1;\n"
        "    cvta.to.global.u64 %rd5, %rd2;\n"
        "    mov.f32 %f1, 0f00000000;\n"
        "    mov.u32 %r12, 0;\n"
        "LOOP:\n"
        "    setp.ge.s32 %p1, %r12, %r2;\n"
        "    @%p1 bra STORE;\n"
        "    mul.lo.s32 %r13, %r11, %r2;\n"
        "    add.s32 %r13, %r13, %r12;\n"
        "    mul.wide.s32 %rd6, %r13, 2;\n"
        "    add.s64 %rd7, %rd4, %rd6;\n"
        "    ld.global.bf16 %b1, [%rd7];\n"
        "    cvt.f32.bf16 %f2, %b1;\n"
        "    mul.lo.s32 %r14, %r12, %r3;\n"
        "    add.s32 %r14, %r14, %r7;\n"
        "    mul.wide.s32 %rd8, %r14, 2;\n"
        "    add.s64 %rd9, %rd5, %rd8;\n"
        "    ld.global.bf16 %b2, [%rd9];\n"
        "    cvt.f32.bf16 %f3, %b2;\n"
        "    fma.rn.f32 %f1, %f2, %f3, %f1;\n"
        "    add.s32 %r12, %r12, 1;\n"
        "    bra.uni LOOP;\n"
        "STORE:\n"
        "    mul.lo.s32 %r15, %r11, %r3;\n"
        "    add.s32 %r15, %r15, %r7;\n"
        "    mul.wide.s32 %rd10, %r15, 2;\n"
        "    cvta.to.global.u64 %rd11, %rd3;\n"
        "    add.s64 %rd12, %rd11, %rd10;\n"
        "    cvt.rn.bf16.f32 %b3, %f1;\n"
        "    st.global.bf16 [%rd12], %b3;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    unsafe {
        if (__cuda_matmul_bf16_kernel_ == (void*)0) {
            void* module = (void*)0;
            if (cuModuleLoadData(&module, (const void*)ptxSrc) != CUDA_SUCCESS) return 0;
            void* kernel = (void*)0;
            if (cuModuleGetFunction(&kernel, module, "matmul_kernel_bf16") != CUDA_SUCCESS) return 0;
            __cuda_matmul_bf16_kernel_ = kernel;
        }

        unsigned long long devA64 = (unsigned long long)(unsigned long)devA;
        unsigned long long devB64 = (unsigned long long)(unsigned long)devB;
        unsigned long bytesOut = M * N * 2UL;
        unsigned long long devOut = 0ULL;
        if (cuMemAlloc_v2(&devOut, bytesOut) != CUDA_SUCCESS) return 0;

        unsigned int Mu = (unsigned int)M;
        unsigned int Ku = (unsigned int)K;
        unsigned int Nu = (unsigned int)N;
        void* params[6];
        params[0] = (void*)&devA64;
        params[1] = (void*)&devB64;
        params[2] = (void*)&devOut;
        params[3] = (void*)&Mu;
        params[4] = (void*)&Ku;
        params[5] = (void*)&Nu;

        unsigned int tgW = 16U;
        unsigned int tgH = 16U;
        unsigned int gridX = (unsigned int)((N + (unsigned long)tgW - 1UL) / (unsigned long)tgW);
        unsigned int gridY = (unsigned int)((M + (unsigned long)tgH - 1UL) / (unsigned long)tgH);
        int launchOk = cuLaunchKernel(__cuda_matmul_bf16_kernel_, gridX, gridY, 1U, tgW, tgH, 1U,
                                       0U, (void*)0, params, (void**)0) == CUDA_SUCCESS;
        cuCtxSynchronize();
        if (launchOk) cuMemcpyDtoH_v2((void*)out, devOut, bytesOut);
        cuMemFree_v2(devOut);
        return launchOk ? 1 : 0;
    }
}

// w[i] -= lr*grad[i], in place, fp16 storage -- same cvt.f32.f16/
// cvt.rn.f16.f32 promote-compute-demote shape as the matmul kernel above,
// applied to cuda_sgd_update_f32's fused-multiply-add structure.
int cuda_sgd_update_f16(void* devW, void* devGrad, float lr, unsigned long n) {
    if (!__cuda_ensure_context()) return 0;
    const char* ptxSrc =
        ".version 7.0\n"
        ".target sm_60\n"
        ".address_size 64\n"
        ".visible .entry sgd_update_kernel_f16(\n"
        "    .param .u64 sgd_update_kernel_f16_param_0,\n"
        "    .param .u64 sgd_update_kernel_f16_param_1,\n"
        "    .param .f32 sgd_update_kernel_f16_param_2,\n"
        "    .param .u32 sgd_update_kernel_f16_param_3\n"
        ") {\n"
        "    .reg .pred %p<2>;\n"
        "    .reg .f16 %h<4>;\n"
        "    .reg .f32 %f<6>;\n"
        "    .reg .b32 %r<6>;\n"
        "    .reg .b64 %rd<8>;\n"
        "    ld.param.u64 %rd1, [sgd_update_kernel_f16_param_0];\n"
        "    ld.param.u64 %rd2, [sgd_update_kernel_f16_param_1];\n"
        "    ld.param.f32 %f3, [sgd_update_kernel_f16_param_2];\n"
        "    ld.param.u32 %r2, [sgd_update_kernel_f16_param_3];\n"
        "    mov.u32 %r3, %ctaid.x;\n"
        "    mov.u32 %r4, %ntid.x;\n"
        "    mov.u32 %r5, %tid.x;\n"
        "    mad.lo.s32 %r1, %r3, %r4, %r5;\n"
        "    setp.ge.s32 %p1, %r1, %r2;\n"
        "    @%p1 bra DONE;\n"
        "    cvta.to.global.u64 %rd3, %rd1;\n"
        "    mul.wide.s32 %rd4, %r1, 2;\n"
        "    add.s64 %rd5, %rd3, %rd4;\n"
        "    ld.global.f16 %h1, [%rd5];\n"
        "    cvt.f32.f16 %f1, %h1;\n"
        "    cvta.to.global.u64 %rd6, %rd2;\n"
        "    add.s64 %rd7, %rd6, %rd4;\n"
        "    ld.global.f16 %h2, [%rd7];\n"
        "    cvt.f32.f16 %f2, %h2;\n"
        "    neg.f32 %f4, %f3;\n"
        "    fma.rn.f32 %f5, %f2, %f4, %f1;\n"
        "    cvt.rn.f16.f32 %h3, %f5;\n"
        "    st.global.f16 [%rd5], %h3;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    unsafe {
        if (__cuda_sgd_f16_kernel_ == (void*)0) {
            void* module = (void*)0;
            if (cuModuleLoadData(&module, (const void*)ptxSrc) != CUDA_SUCCESS) return 0;
            void* kernel = (void*)0;
            if (cuModuleGetFunction(&kernel, module, "sgd_update_kernel_f16") != CUDA_SUCCESS) return 0;
            __cuda_sgd_f16_kernel_ = kernel;
        }

        unsigned long long devW64 = (unsigned long long)(unsigned long)devW;
        unsigned long long devGrad64 = (unsigned long long)(unsigned long)devGrad;
        float lrVal = lr;
        unsigned int nParam = (unsigned int)n;
        void* params[4];
        params[0] = (void*)&devW64;
        params[1] = (void*)&devGrad64;
        params[2] = (void*)&lrVal;
        params[3] = (void*)&nParam;

        unsigned int blockSize = 256U;
        unsigned int gridSize = (unsigned int)((n + (unsigned long)blockSize - 1UL) / (unsigned long)blockSize);
        int launchOk = cuLaunchKernel(__cuda_sgd_f16_kernel_, gridSize, 1U, 1U, blockSize, 1U, 1U,
                                       0U, (void*)0, params, (void**)0) == CUDA_SUCCESS;
        cuCtxSynchronize();
        return launchOk ? 1 : 0;
    }
}

// Same shape as cuda_sgd_update_f16, '.bf16' storage, sm_80/PTX 7.8 (see
// cuda_matmul_bf16_persistent's comment).
int cuda_sgd_update_bf16(void* devW, void* devGrad, float lr, unsigned long n) {
    if (!__cuda_ensure_context()) return 0;
    const char* ptxSrc =
        ".version 7.8\n"
        ".target sm_80\n"
        ".address_size 64\n"
        ".visible .entry sgd_update_kernel_bf16(\n"
        "    .param .u64 sgd_update_kernel_bf16_param_0,\n"
        "    .param .u64 sgd_update_kernel_bf16_param_1,\n"
        "    .param .f32 sgd_update_kernel_bf16_param_2,\n"
        "    .param .u32 sgd_update_kernel_bf16_param_3\n"
        ") {\n"
        "    .reg .pred %p<2>;\n"
        "    .reg .bf16 %b<4>;\n"
        "    .reg .f32 %f<6>;\n"
        "    .reg .b32 %r<6>;\n"
        "    .reg .b64 %rd<8>;\n"
        "    ld.param.u64 %rd1, [sgd_update_kernel_bf16_param_0];\n"
        "    ld.param.u64 %rd2, [sgd_update_kernel_bf16_param_1];\n"
        "    ld.param.f32 %f3, [sgd_update_kernel_bf16_param_2];\n"
        "    ld.param.u32 %r2, [sgd_update_kernel_bf16_param_3];\n"
        "    mov.u32 %r3, %ctaid.x;\n"
        "    mov.u32 %r4, %ntid.x;\n"
        "    mov.u32 %r5, %tid.x;\n"
        "    mad.lo.s32 %r1, %r3, %r4, %r5;\n"
        "    setp.ge.s32 %p1, %r1, %r2;\n"
        "    @%p1 bra DONE;\n"
        "    cvta.to.global.u64 %rd3, %rd1;\n"
        "    mul.wide.s32 %rd4, %r1, 2;\n"
        "    add.s64 %rd5, %rd3, %rd4;\n"
        "    ld.global.bf16 %b1, [%rd5];\n"
        "    cvt.f32.bf16 %f1, %b1;\n"
        "    cvta.to.global.u64 %rd6, %rd2;\n"
        "    add.s64 %rd7, %rd6, %rd4;\n"
        "    ld.global.bf16 %b2, [%rd7];\n"
        "    cvt.f32.bf16 %f2, %b2;\n"
        "    neg.f32 %f4, %f3;\n"
        "    fma.rn.f32 %f5, %f2, %f4, %f1;\n"
        "    cvt.rn.bf16.f32 %b3, %f5;\n"
        "    st.global.bf16 [%rd5], %b3;\n"
        "DONE:\n"
        "    ret;\n"
        "}\n";
    unsafe {
        if (__cuda_sgd_bf16_kernel_ == (void*)0) {
            void* module = (void*)0;
            if (cuModuleLoadData(&module, (const void*)ptxSrc) != CUDA_SUCCESS) return 0;
            void* kernel = (void*)0;
            if (cuModuleGetFunction(&kernel, module, "sgd_update_kernel_bf16") != CUDA_SUCCESS) return 0;
            __cuda_sgd_bf16_kernel_ = kernel;
        }

        unsigned long long devW64 = (unsigned long long)(unsigned long)devW;
        unsigned long long devGrad64 = (unsigned long long)(unsigned long)devGrad;
        float lrVal = lr;
        unsigned int nParam = (unsigned int)n;
        void* params[4];
        params[0] = (void*)&devW64;
        params[1] = (void*)&devGrad64;
        params[2] = (void*)&lrVal;
        params[3] = (void*)&nParam;

        unsigned int blockSize = 256U;
        unsigned int gridSize = (unsigned int)((n + (unsigned long)blockSize - 1UL) / (unsigned long)blockSize);
        int launchOk = cuLaunchKernel(__cuda_sgd_bf16_kernel_, gridSize, 1U, 1U, blockSize, 1U, 1U,
                                       0U, (void*)0, params, (void**)0) == CUDA_SUCCESS;
        cuCtxSynchronize();
        return launchOk ? 1 : 0;
    }
}

} // namespace std
