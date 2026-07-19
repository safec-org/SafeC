// SafeC Standard Library — CUDA GPU backend implementation (see
// gpu_cuda.h). UNVERIFIED — no NVIDIA GPU/CUDA toolkit in this sandbox;
// see that header's warning. Written against the real Driver API ABI.
#pragma once
#include <std/ml/gpu_cuda.h>
#include <std/mem.sc>

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

#define CUDA_SUCCESS 0

int cuda_available() {
    if (cuInit(0U) != CUDA_SUCCESS) return 0;
    int count = 0;
    unsafe { if (cuDeviceGetCount(&count) != CUDA_SUCCESS) return 0; }
    return count > 0;
}

int cuda_add_f32(const float* a, const float* b, float* out, unsigned long n) {
    // PTX for a trivial 'out[i] = a[i] + b[i]' kernel — the CUDA analogue
    // of gpu_mps.sc's inline MSL source (see gpu_cuda.h's header comment
    // for why PTX rather than requiring nvcc as a build dependency).
    // Hand-written, targeting a conservative compute capability (sm_50)
    // for broad compatibility; loaded at runtime via cuModuleLoadData.
    // Local (not file-scope) because SafeC's global-initializer constant
    // folder doesn't fold adjacent-string-literal concatenation (same
    // reason gpu_mps.sc's kernel source is a local, not a static).
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
        if (cuModuleGetFunction(&kernel, module, "add_kernel") != CUDA_SUCCESS) {
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

} // namespace std
