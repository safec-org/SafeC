// SafeC Standard Library — ROCm/HIP GPU backend implementation (see
// gpu_rocm.h). UNVERIFIED — no AMD GPU/ROCm toolkit in this sandbox.
// Written against the real HIP Runtime API ABI. Every op below has the
// same HSACO gap rocm_add_f32 already documents: the HIP Runtime API
// plumbing (device init, buffer alloc/copy, module/function lookup,
// launch, sync, teardown) is real and type-checked, but hipModuleLoadData
// needs a real compiled kernel binary this sandbox has no toolchain to
// produce, so every function here always returns 0. A real deployment
// would replace each 'kernelImage = (const void*)0' with bytes vendored
// from 'hipcc --genco' run against the matching .hip source for that op.
#pragma once
#include <std/ml/gpu_rocm.h>
#include <std/mem.sc>

namespace std {

// ── HIP Runtime API (hip_runtime_api.h) — hand-matched signatures ──────────
extern int hipInit(unsigned int flags);
extern int hipGetDeviceCount(int* count);
extern int hipSetDevice(int deviceId);
extern int hipModuleLoadData(void** module, const void* image);
extern int hipModuleGetFunction(void** hfunc, void* hmod, const char* name);
extern int hipMalloc(void** ptr, unsigned long size);
extern int hipFree(void* ptr);
extern int hipMemcpyHtoD(void* dst, void* src, unsigned long sizeBytes);
extern int hipMemcpyDtoH(void* dst, void* src, unsigned long sizeBytes);
extern int hipModuleLaunchKernel(void* f,
                                  unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                  unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                  unsigned int sharedMemBytes, void* stream,
                                  void** kernelParams, void** extra);
extern int hipDeviceSynchronize();

#define HIP_SUCCESS 0

int rocm_available() {
    if (hipInit(0U) != HIP_SUCCESS) return 0;
    int count = 0;
    unsafe { if (hipGetDeviceCount(&count) != HIP_SUCCESS) return 0; }
    return count > 0;
}

int rocm_add_f32(const float* a, const float* b, float* out, unsigned long n) {
    // See gpu_rocm.h's header comment: hipModuleLoadData needs a real
    // HSACO binary (offline-compiled by hipcc from a .hip/.cpp kernel
    // source against a specific GPU ISA target, e.g. gfx1100) — unlike
    // CUDA's PTX, that's not a text format this file can embed as a
    // string literal, and there's no ROCm toolchain available in this
    // sandbox to produce one. Everything below this point (device
    // selection, buffer alloc/copy, launch, sync, teardown) is real,
    // correctly-typed HIP Runtime API usage — the one missing piece is
    // the 'kernelImage' byte array a real deployment would supply (e.g.
    // vendored from a build step that runs 'hipcc --genco' against
    // add_kernel.hip and embeds the resulting .co file's bytes here).
    const void* kernelImage = (const void*)0;
    if (kernelImage == (const void*)0) return 0;

    unsafe {
        if (hipInit(0U) != HIP_SUCCESS) return 0;
        if (hipSetDevice(0) != HIP_SUCCESS) return 0;

        void* module = (void*)0;
        if (hipModuleLoadData(&module, kernelImage) != HIP_SUCCESS) return 0;
        void* kernel = (void*)0;
        if (hipModuleGetFunction(&kernel, module, "add_kernel") != HIP_SUCCESS) return 0;

        unsigned long bytes = n * sizeof(float);
        void* devA = (void*)0; void* devB = (void*)0; void* devOut = (void*)0;
        hipMalloc(&devA, bytes);
        hipMalloc(&devB, bytes);
        hipMalloc(&devOut, bytes);
        hipMemcpyHtoD(devA, (void*)a, bytes);
        hipMemcpyHtoD(devB, (void*)b, bytes);

        unsigned int nParam = (unsigned int)n;
        void* params[4];
        params[0] = (void*)&devA;
        params[1] = (void*)&devB;
        params[2] = (void*)&devOut;
        params[3] = (void*)&nParam;

        unsigned int blockSize = 256U;
        unsigned int gridSize = (unsigned int)((n + (unsigned long)blockSize - 1UL) / (unsigned long)blockSize);
        int launchOk = hipModuleLaunchKernel(kernel, gridSize, 1U, 1U, blockSize, 1U, 1U,
                                              0U, (void*)0, params, (void**)0) == HIP_SUCCESS;
        hipDeviceSynchronize();
        if (launchOk) hipMemcpyDtoH((void*)out, devOut, bytes);

        hipFree(devA);
        hipFree(devB);
        hipFree(devOut);
        return launchOk ? 1 : 0;
    }
}

int rocm_sub_f32(const float* a, const float* b, float* out, unsigned long n) {
    const void* kernelImage = (const void*)0; // would be sub_kernel.hip's compiled HSACO
    if (kernelImage == (const void*)0) return 0;
    unsafe {
        if (hipInit(0U) != HIP_SUCCESS) return 0;
        if (hipSetDevice(0) != HIP_SUCCESS) return 0;
        void* module = (void*)0;
        if (hipModuleLoadData(&module, kernelImage) != HIP_SUCCESS) return 0;
        void* kernel = (void*)0;
        if (hipModuleGetFunction(&kernel, module, "sub_kernel") != HIP_SUCCESS) return 0;
        unsigned long bytes = n * sizeof(float);
        void* devA = (void*)0; void* devB = (void*)0; void* devOut = (void*)0;
        hipMalloc(&devA, bytes); hipMalloc(&devB, bytes); hipMalloc(&devOut, bytes);
        hipMemcpyHtoD(devA, (void*)a, bytes);
        hipMemcpyHtoD(devB, (void*)b, bytes);
        unsigned int nParam = (unsigned int)n;
        void* params[4];
        params[0] = (void*)&devA; params[1] = (void*)&devB; params[2] = (void*)&devOut; params[3] = (void*)&nParam;
        unsigned int blockSize = 256U;
        unsigned int gridSize = (unsigned int)((n + (unsigned long)blockSize - 1UL) / (unsigned long)blockSize);
        int launchOk = hipModuleLaunchKernel(kernel, gridSize, 1U, 1U, blockSize, 1U, 1U,
                                              0U, (void*)0, params, (void**)0) == HIP_SUCCESS;
        hipDeviceSynchronize();
        if (launchOk) hipMemcpyDtoH((void*)out, devOut, bytes);
        hipFree(devA); hipFree(devB); hipFree(devOut);
        return launchOk ? 1 : 0;
    }
}

int rocm_mul_f32(const float* a, const float* b, float* out, unsigned long n) {
    const void* kernelImage = (const void*)0; // would be mul_kernel.hip's compiled HSACO
    if (kernelImage == (const void*)0) return 0;
    unsafe {
        if (hipInit(0U) != HIP_SUCCESS) return 0;
        if (hipSetDevice(0) != HIP_SUCCESS) return 0;
        void* module = (void*)0;
        if (hipModuleLoadData(&module, kernelImage) != HIP_SUCCESS) return 0;
        void* kernel = (void*)0;
        if (hipModuleGetFunction(&kernel, module, "mul_kernel") != HIP_SUCCESS) return 0;
        unsigned long bytes = n * sizeof(float);
        void* devA = (void*)0; void* devB = (void*)0; void* devOut = (void*)0;
        hipMalloc(&devA, bytes); hipMalloc(&devB, bytes); hipMalloc(&devOut, bytes);
        hipMemcpyHtoD(devA, (void*)a, bytes);
        hipMemcpyHtoD(devB, (void*)b, bytes);
        unsigned int nParam = (unsigned int)n;
        void* params[4];
        params[0] = (void*)&devA; params[1] = (void*)&devB; params[2] = (void*)&devOut; params[3] = (void*)&nParam;
        unsigned int blockSize = 256U;
        unsigned int gridSize = (unsigned int)((n + (unsigned long)blockSize - 1UL) / (unsigned long)blockSize);
        int launchOk = hipModuleLaunchKernel(kernel, gridSize, 1U, 1U, blockSize, 1U, 1U,
                                              0U, (void*)0, params, (void**)0) == HIP_SUCCESS;
        hipDeviceSynchronize();
        if (launchOk) hipMemcpyDtoH((void*)out, devOut, bytes);
        hipFree(devA); hipFree(devB); hipFree(devOut);
        return launchOk ? 1 : 0;
    }
}

int rocm_div_f32(const float* a, const float* b, float* out, unsigned long n) {
    const void* kernelImage = (const void*)0; // would be div_kernel.hip's compiled HSACO
    if (kernelImage == (const void*)0) return 0;
    unsafe {
        if (hipInit(0U) != HIP_SUCCESS) return 0;
        if (hipSetDevice(0) != HIP_SUCCESS) return 0;
        void* module = (void*)0;
        if (hipModuleLoadData(&module, kernelImage) != HIP_SUCCESS) return 0;
        void* kernel = (void*)0;
        if (hipModuleGetFunction(&kernel, module, "div_kernel") != HIP_SUCCESS) return 0;
        unsigned long bytes = n * sizeof(float);
        void* devA = (void*)0; void* devB = (void*)0; void* devOut = (void*)0;
        hipMalloc(&devA, bytes); hipMalloc(&devB, bytes); hipMalloc(&devOut, bytes);
        hipMemcpyHtoD(devA, (void*)a, bytes);
        hipMemcpyHtoD(devB, (void*)b, bytes);
        unsigned int nParam = (unsigned int)n;
        void* params[4];
        params[0] = (void*)&devA; params[1] = (void*)&devB; params[2] = (void*)&devOut; params[3] = (void*)&nParam;
        unsigned int blockSize = 256U;
        unsigned int gridSize = (unsigned int)((n + (unsigned long)blockSize - 1UL) / (unsigned long)blockSize);
        int launchOk = hipModuleLaunchKernel(kernel, gridSize, 1U, 1U, blockSize, 1U, 1U,
                                              0U, (void*)0, params, (void**)0) == HIP_SUCCESS;
        hipDeviceSynchronize();
        if (launchOk) hipMemcpyDtoH((void*)out, devOut, bytes);
        hipFree(devA); hipFree(devB); hipFree(devOut);
        return launchOk ? 1 : 0;
    }
}

int rocm_pow_f32(const float* a, const float* b, float* out, unsigned long n) {
    const void* kernelImage = (const void*)0; // would be pow_kernel.hip's compiled HSACO
    if (kernelImage == (const void*)0) return 0;
    unsafe {
        if (hipInit(0U) != HIP_SUCCESS) return 0;
        if (hipSetDevice(0) != HIP_SUCCESS) return 0;
        void* module = (void*)0;
        if (hipModuleLoadData(&module, kernelImage) != HIP_SUCCESS) return 0;
        void* kernel = (void*)0;
        if (hipModuleGetFunction(&kernel, module, "pow_kernel") != HIP_SUCCESS) return 0;
        unsigned long bytes = n * sizeof(float);
        void* devA = (void*)0; void* devB = (void*)0; void* devOut = (void*)0;
        hipMalloc(&devA, bytes); hipMalloc(&devB, bytes); hipMalloc(&devOut, bytes);
        hipMemcpyHtoD(devA, (void*)a, bytes);
        hipMemcpyHtoD(devB, (void*)b, bytes);
        unsigned int nParam = (unsigned int)n;
        void* params[4];
        params[0] = (void*)&devA; params[1] = (void*)&devB; params[2] = (void*)&devOut; params[3] = (void*)&nParam;
        unsigned int blockSize = 256U;
        unsigned int gridSize = (unsigned int)((n + (unsigned long)blockSize - 1UL) / (unsigned long)blockSize);
        int launchOk = hipModuleLaunchKernel(kernel, gridSize, 1U, 1U, blockSize, 1U, 1U,
                                              0U, (void*)0, params, (void**)0) == HIP_SUCCESS;
        hipDeviceSynchronize();
        if (launchOk) hipMemcpyDtoH((void*)out, devOut, bytes);
        hipFree(devA); hipFree(devB); hipFree(devOut);
        return launchOk ? 1 : 0;
    }
}

int rocm_relu_f32(const float* a, float* out, unsigned long n) {
    const void* kernelImage = (const void*)0; // would be relu_kernel.hip's compiled HSACO
    if (kernelImage == (const void*)0) return 0;
    unsafe {
        if (hipInit(0U) != HIP_SUCCESS) return 0;
        if (hipSetDevice(0) != HIP_SUCCESS) return 0;
        void* module = (void*)0;
        if (hipModuleLoadData(&module, kernelImage) != HIP_SUCCESS) return 0;
        void* kernel = (void*)0;
        if (hipModuleGetFunction(&kernel, module, "relu_kernel") != HIP_SUCCESS) return 0;
        unsigned long bytes = n * sizeof(float);
        void* devA = (void*)0; void* devOut = (void*)0;
        hipMalloc(&devA, bytes); hipMalloc(&devOut, bytes);
        hipMemcpyHtoD(devA, (void*)a, bytes);
        unsigned int nParam = (unsigned int)n;
        void* params[3];
        params[0] = (void*)&devA; params[1] = (void*)&devOut; params[2] = (void*)&nParam;
        unsigned int blockSize = 256U;
        unsigned int gridSize = (unsigned int)((n + (unsigned long)blockSize - 1UL) / (unsigned long)blockSize);
        int launchOk = hipModuleLaunchKernel(kernel, gridSize, 1U, 1U, blockSize, 1U, 1U,
                                              0U, (void*)0, params, (void**)0) == HIP_SUCCESS;
        hipDeviceSynchronize();
        if (launchOk) hipMemcpyDtoH((void*)out, devOut, bytes);
        hipFree(devA); hipFree(devOut);
        return launchOk ? 1 : 0;
    }
}

int rocm_log_f32(const float* a, float* out, unsigned long n) {
    const void* kernelImage = (const void*)0; // would be log_kernel.hip's compiled HSACO
    if (kernelImage == (const void*)0) return 0;
    unsafe {
        if (hipInit(0U) != HIP_SUCCESS) return 0;
        if (hipSetDevice(0) != HIP_SUCCESS) return 0;
        void* module = (void*)0;
        if (hipModuleLoadData(&module, kernelImage) != HIP_SUCCESS) return 0;
        void* kernel = (void*)0;
        if (hipModuleGetFunction(&kernel, module, "log_kernel") != HIP_SUCCESS) return 0;
        unsigned long bytes = n * sizeof(float);
        void* devA = (void*)0; void* devOut = (void*)0;
        hipMalloc(&devA, bytes); hipMalloc(&devOut, bytes);
        hipMemcpyHtoD(devA, (void*)a, bytes);
        unsigned int nParam = (unsigned int)n;
        void* params[3];
        params[0] = (void*)&devA; params[1] = (void*)&devOut; params[2] = (void*)&nParam;
        unsigned int blockSize = 256U;
        unsigned int gridSize = (unsigned int)((n + (unsigned long)blockSize - 1UL) / (unsigned long)blockSize);
        int launchOk = hipModuleLaunchKernel(kernel, gridSize, 1U, 1U, blockSize, 1U, 1U,
                                              0U, (void*)0, params, (void**)0) == HIP_SUCCESS;
        hipDeviceSynchronize();
        if (launchOk) hipMemcpyDtoH((void*)out, devOut, bytes);
        hipFree(devA); hipFree(devOut);
        return launchOk ? 1 : 0;
    }
}

int rocm_exp_f32(const float* a, float* out, unsigned long n) {
    const void* kernelImage = (const void*)0; // would be exp_kernel.hip's compiled HSACO
    if (kernelImage == (const void*)0) return 0;
    unsafe {
        if (hipInit(0U) != HIP_SUCCESS) return 0;
        if (hipSetDevice(0) != HIP_SUCCESS) return 0;
        void* module = (void*)0;
        if (hipModuleLoadData(&module, kernelImage) != HIP_SUCCESS) return 0;
        void* kernel = (void*)0;
        if (hipModuleGetFunction(&kernel, module, "exp_kernel") != HIP_SUCCESS) return 0;
        unsigned long bytes = n * sizeof(float);
        void* devA = (void*)0; void* devOut = (void*)0;
        hipMalloc(&devA, bytes); hipMalloc(&devOut, bytes);
        hipMemcpyHtoD(devA, (void*)a, bytes);
        unsigned int nParam = (unsigned int)n;
        void* params[3];
        params[0] = (void*)&devA; params[1] = (void*)&devOut; params[2] = (void*)&nParam;
        unsigned int blockSize = 256U;
        unsigned int gridSize = (unsigned int)((n + (unsigned long)blockSize - 1UL) / (unsigned long)blockSize);
        int launchOk = hipModuleLaunchKernel(kernel, gridSize, 1U, 1U, blockSize, 1U, 1U,
                                              0U, (void*)0, params, (void**)0) == HIP_SUCCESS;
        hipDeviceSynchronize();
        if (launchOk) hipMemcpyDtoH((void*)out, devOut, bytes);
        hipFree(devA); hipFree(devOut);
        return launchOk ? 1 : 0;
    }
}

int rocm_sqrt_f32(const float* a, float* out, unsigned long n) {
    const void* kernelImage = (const void*)0; // would be sqrt_kernel.hip's compiled HSACO
    if (kernelImage == (const void*)0) return 0;
    unsafe {
        if (hipInit(0U) != HIP_SUCCESS) return 0;
        if (hipSetDevice(0) != HIP_SUCCESS) return 0;
        void* module = (void*)0;
        if (hipModuleLoadData(&module, kernelImage) != HIP_SUCCESS) return 0;
        void* kernel = (void*)0;
        if (hipModuleGetFunction(&kernel, module, "sqrt_kernel") != HIP_SUCCESS) return 0;
        unsigned long bytes = n * sizeof(float);
        void* devA = (void*)0; void* devOut = (void*)0;
        hipMalloc(&devA, bytes); hipMalloc(&devOut, bytes);
        hipMemcpyHtoD(devA, (void*)a, bytes);
        unsigned int nParam = (unsigned int)n;
        void* params[3];
        params[0] = (void*)&devA; params[1] = (void*)&devOut; params[2] = (void*)&nParam;
        unsigned int blockSize = 256U;
        unsigned int gridSize = (unsigned int)((n + (unsigned long)blockSize - 1UL) / (unsigned long)blockSize);
        int launchOk = hipModuleLaunchKernel(kernel, gridSize, 1U, 1U, blockSize, 1U, 1U,
                                              0U, (void*)0, params, (void**)0) == HIP_SUCCESS;
        hipDeviceSynchronize();
        if (launchOk) hipMemcpyDtoH((void*)out, devOut, bytes);
        hipFree(devA); hipFree(devOut);
        return launchOk ? 1 : 0;
    }
}

int rocm_scale_f32(const float* a, float k, float* out, unsigned long n) {
    const void* kernelImage = (const void*)0; // would be scale_kernel.hip's compiled HSACO
    if (kernelImage == (const void*)0) return 0;
    unsafe {
        if (hipInit(0U) != HIP_SUCCESS) return 0;
        if (hipSetDevice(0) != HIP_SUCCESS) return 0;
        void* module = (void*)0;
        if (hipModuleLoadData(&module, kernelImage) != HIP_SUCCESS) return 0;
        void* kernel = (void*)0;
        if (hipModuleGetFunction(&kernel, module, "scale_kernel") != HIP_SUCCESS) return 0;
        unsigned long bytes = n * sizeof(float);
        void* devA = (void*)0; void* devOut = (void*)0;
        hipMalloc(&devA, bytes); hipMalloc(&devOut, bytes);
        hipMemcpyHtoD(devA, (void*)a, bytes);
        float kParam = k;
        unsigned int nParam = (unsigned int)n;
        void* params[4];
        params[0] = (void*)&devA; params[1] = (void*)&devOut; params[2] = (void*)&kParam; params[3] = (void*)&nParam;
        unsigned int blockSize = 256U;
        unsigned int gridSize = (unsigned int)((n + (unsigned long)blockSize - 1UL) / (unsigned long)blockSize);
        int launchOk = hipModuleLaunchKernel(kernel, gridSize, 1U, 1U, blockSize, 1U, 1U,
                                              0U, (void*)0, params, (void**)0) == HIP_SUCCESS;
        hipDeviceSynchronize();
        if (launchOk) hipMemcpyDtoH((void*)out, devOut, bytes);
        hipFree(devA); hipFree(devOut);
        return launchOk ? 1 : 0;
    }
}

int rocm_sum_f32(const float* a, float* out, unsigned long n) {
    const void* kernelImage = (const void*)0; // would be sum_kernel.hip's compiled HSACO
    if (kernelImage == (const void*)0) return 0;
    unsafe {
        if (hipInit(0U) != HIP_SUCCESS) return 0;
        if (hipSetDevice(0) != HIP_SUCCESS) return 0;
        void* module = (void*)0;
        if (hipModuleLoadData(&module, kernelImage) != HIP_SUCCESS) return 0;
        void* kernel = (void*)0;
        if (hipModuleGetFunction(&kernel, module, "sum_kernel") != HIP_SUCCESS) return 0;
        unsigned long bytes = n * sizeof(float);
        void* devA = (void*)0; void* devOut = (void*)0;
        hipMalloc(&devA, bytes); hipMalloc(&devOut, 4UL);
        hipMemcpyHtoD(devA, (void*)a, bytes);
        unsigned int nParam = (unsigned int)n;
        void* params[3];
        params[0] = (void*)&devA; params[1] = (void*)&devOut; params[2] = (void*)&nParam;
        int launchOk = hipModuleLaunchKernel(kernel, 1U, 1U, 1U, 1U, 1U, 1U,
                                              0U, (void*)0, params, (void**)0) == HIP_SUCCESS;
        hipDeviceSynchronize();
        if (launchOk) hipMemcpyDtoH((void*)out, devOut, 4UL);
        hipFree(devA); hipFree(devOut);
        return launchOk ? 1 : 0;
    }
}

int rocm_matmul_f32(const float* a, const float* b, float* out,
                     unsigned long M, unsigned long K, unsigned long N) {
    const void* kernelImage = (const void*)0; // would be matmul_kernel.hip's compiled HSACO
    if (kernelImage == (const void*)0) return 0;
    unsafe {
        if (hipInit(0U) != HIP_SUCCESS) return 0;
        if (hipSetDevice(0) != HIP_SUCCESS) return 0;
        void* module = (void*)0;
        if (hipModuleLoadData(&module, kernelImage) != HIP_SUCCESS) return 0;
        void* kernel = (void*)0;
        if (hipModuleGetFunction(&kernel, module, "matmul_kernel") != HIP_SUCCESS) return 0;

        unsigned long bytesA = M * K * sizeof(float);
        unsigned long bytesB = K * N * sizeof(float);
        unsigned long bytesOut = M * N * sizeof(float);
        void* devA = (void*)0; void* devB = (void*)0; void* devOut = (void*)0;
        hipMalloc(&devA, bytesA); hipMalloc(&devB, bytesB); hipMalloc(&devOut, bytesOut);
        hipMemcpyHtoD(devA, (void*)a, bytesA);
        hipMemcpyHtoD(devB, (void*)b, bytesB);

        unsigned int Mu = (unsigned int)M;
        unsigned int Ku = (unsigned int)K;
        unsigned int Nu = (unsigned int)N;
        void* params[6];
        params[0] = (void*)&devA; params[1] = (void*)&devB; params[2] = (void*)&devOut;
        params[3] = (void*)&Mu; params[4] = (void*)&Ku; params[5] = (void*)&Nu;

        unsigned int tgW = 16U;
        unsigned int tgH = 16U;
        unsigned int gridX = (unsigned int)((N + (unsigned long)tgW - 1UL) / (unsigned long)tgW);
        unsigned int gridY = (unsigned int)((M + (unsigned long)tgH - 1UL) / (unsigned long)tgH);
        int launchOk = hipModuleLaunchKernel(kernel, gridX, gridY, 1U, tgW, tgH, 1U,
                                              0U, (void*)0, params, (void**)0) == HIP_SUCCESS;
        hipDeviceSynchronize();
        if (launchOk) hipMemcpyDtoH((void*)out, devOut, bytesOut);

        hipFree(devA); hipFree(devB); hipFree(devOut);
        return launchOk ? 1 : 0;
    }
}

} // namespace std
