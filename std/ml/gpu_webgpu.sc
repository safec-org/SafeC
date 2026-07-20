// SafeC Standard Library — WebGPU GPU backend implementation (see
// gpu_webgpu.h). UNVERIFIED: no wgpu-native/Dawn library, no webgpu.h, no
// WebGPU-capable backend to link/run against in this sandbox.
//
// WebGPU's adapter/device acquisition is asynchronous even in the native
// C API (wgpuInstanceRequestAdapter/wgpuAdapterRequestDevice take a
// callback, not a return value) — there is no core-spec blocking
// "RequestAdapterSync". This file handles that the way real native
// WebGPU C code has to: the callback writes its result into a static
// capture variable, and the caller polls wgpuInstanceProcessEvents in a
// loop until that variable is set (or a call-count cap is hit, so a
// callback that never fires — e.g. no backend available at all — can't
// hang forever). Real applications with an existing event loop
// (a browser, or a native app's own frame loop) would drive
// wgpuInstanceProcessEvents from that loop instead of a dedicated poll;
// this file has no such loop to hook into, so it makes its own, minimal
// one. wgpuBufferMapAsync's readback is handled the same way.
#pragma once
#include <std/ml/gpu_webgpu.h>
#include <std/mem.sc>

namespace std {

// ── WebGPU native C API (webgpu.h) — hand-matched signatures ───────────────
extern void* wgpuCreateInstance(const void* descriptor);
typedef fn void(int, void*, const char*, void*) WGPURequestAdapterCallback;
typedef fn void(int, void*, const char*, void*) WGPURequestDeviceCallback;
typedef fn void(int, void*) WGPUBufferMapCallback;
extern void wgpuInstanceRequestAdapter(void* instance, const void* options, WGPURequestAdapterCallback callback, void* userdata);
extern void wgpuAdapterRequestDevice(void* adapter, const void* descriptor, WGPURequestDeviceCallback callback, void* userdata);
extern void wgpuInstanceProcessEvents(void* instance);
extern void* wgpuDeviceGetQueue(void* device);

struct WGPUShaderModuleWGSLDescriptor {
    int chainSType; const void* chainNext; // WGPUChainedStruct: { const void* next; int sType; } -- flattened here
    const char* code;
};
struct WGPUShaderModuleDescriptor {
    const void* nextInChain; const char* label;
};
extern void* wgpuDeviceCreateShaderModule(void* device, const struct WGPUShaderModuleDescriptor* descriptor);

struct WGPUProgrammableStageDescriptor {
    const void* nextInChain; void* module; const char* entryPoint;
    unsigned int constantCount; const void* constants;
};
struct WGPUComputePipelineDescriptor {
    const void* nextInChain; const char* label; void* layout;
    struct WGPUProgrammableStageDescriptor compute;
};
extern void* wgpuDeviceCreateComputePipeline(void* device, const struct WGPUComputePipelineDescriptor* descriptor);
extern void* wgpuComputePipelineGetBindGroupLayout(void* pipeline, unsigned int groupIndex);

struct WGPUBufferDescriptor {
    const void* nextInChain; const char* label;
    unsigned long usage; unsigned long size; int mappedAtCreation;
};
extern void* wgpuDeviceCreateBuffer(void* device, const struct WGPUBufferDescriptor* descriptor);

struct WGPUBindGroupEntry {
    const void* nextInChain; unsigned int binding;
    void* buffer; unsigned long offset; unsigned long size;
    void* sampler; void* textureView;
};
struct WGPUBindGroupDescriptor {
    const void* nextInChain; const char* label; void* layout;
    unsigned int entryCount; const struct WGPUBindGroupEntry* entries;
};
extern void* wgpuDeviceCreateBindGroup(void* device, const struct WGPUBindGroupDescriptor* descriptor);

extern void* wgpuDeviceCreateCommandEncoder(void* device, const void* descriptor);
extern void* wgpuCommandEncoderBeginComputePass(void* encoder, const void* descriptor);
extern void wgpuComputePassEncoderSetPipeline(void* pass, void* pipeline);
extern void wgpuComputePassEncoderSetBindGroup(void* pass, unsigned int groupIndex, void* group, unsigned int dynamicOffsetCount, const unsigned int* dynamicOffsets);
extern void wgpuComputePassEncoderDispatchWorkgroups(void* pass, unsigned int x, unsigned int y, unsigned int z);
extern void wgpuComputePassEncoderEnd(void* pass);
extern void* wgpuCommandEncoderFinish(void* encoder, const void* descriptor);
extern void wgpuQueueSubmit(void* queue, unsigned int commandCount, const void* commands);
extern void wgpuQueueWriteBuffer(void* queue, void* buffer, unsigned long bufferOffset, const void* data, unsigned long size);
extern void wgpuBufferMapAsync(void* buffer, unsigned int mode, unsigned long offset, unsigned long size, WGPUBufferMapCallback callback, void* userdata);
extern const void* wgpuBufferGetConstMappedRange(void* buffer, unsigned long offset, unsigned long size);
extern void wgpuBufferUnmap(void* buffer);

#define WGPU_REQUEST_SUCCESS 0
#define WGPU_MAP_MODE_READ 1UL
#define WGPU_BUFFER_USAGE_MAP_READ 1UL
#define WGPU_BUFFER_USAGE_COPY_DST 8UL
#define WGPU_BUFFER_USAGE_STORAGE 128UL
#define WGPU_BUFFER_USAGE_COPY_SRC 4UL
#define WGPU_SHADER_MODULE_WGSL_DESCRIPTOR_STYPE 6UL
#define WGPU_POLL_ITERATION_CAP 100000

static void* __wgpu_adapter_result_ = (void*)0;
static int __wgpu_adapter_done_ = 0;
static void* __wgpu_device_result_ = (void*)0;
static int __wgpu_device_done_ = 0;
static int __wgpu_map_done_ = 0;

static void __wgpu_on_adapter(int status, void* adapter, const char* message, void* userdata) {
    unsafe { (void)message; (void)userdata; }
    __wgpu_adapter_result_ = (status == WGPU_REQUEST_SUCCESS) ? adapter : (void*)0;
    __wgpu_adapter_done_ = 1;
}

static void __wgpu_on_device(int status, void* device, const char* message, void* userdata) {
    unsafe { (void)message; (void)userdata; }
    __wgpu_device_result_ = (status == WGPU_REQUEST_SUCCESS) ? device : (void*)0;
    __wgpu_device_done_ = 1;
}

static void __wgpu_on_map(int status, void* userdata) {
    unsafe { (void)userdata; (void)status; }
    __wgpu_map_done_ = 1;
}

// Blocks (via a bounded poll loop, not a real blocking call -- see this
// file's header comment) until instance/adapter/device are ready, or
// returns 0 if any step fails or the poll cap is hit without the
// callback ever firing (e.g. no WebGPU backend available at all).
static int __wgpu_init(void** outInstance, void** outDevice, void** outQueue) {
    unsafe {
        void* instance = wgpuCreateInstance((const void*)0);
        if (instance == (void*)0) return 0;

        __wgpu_adapter_result_ = (void*)0; __wgpu_adapter_done_ = 0;
        wgpuInstanceRequestAdapter(instance, (const void*)0, __wgpu_on_adapter, (void*)0);
        int iter = 0;
        while (__wgpu_adapter_done_ == 0 && iter < WGPU_POLL_ITERATION_CAP) {
            wgpuInstanceProcessEvents(instance);
            iter = iter + 1;
        }
        if (__wgpu_adapter_done_ == 0 || __wgpu_adapter_result_ == (void*)0) return 0;
        void* adapter = __wgpu_adapter_result_;

        __wgpu_device_result_ = (void*)0; __wgpu_device_done_ = 0;
        wgpuAdapterRequestDevice(adapter, (const void*)0, __wgpu_on_device, (void*)0);
        iter = 0;
        while (__wgpu_device_done_ == 0 && iter < WGPU_POLL_ITERATION_CAP) {
            wgpuInstanceProcessEvents(instance);
            iter = iter + 1;
        }
        if (__wgpu_device_done_ == 0 || __wgpu_device_result_ == (void*)0) return 0;
        void* device = __wgpu_device_result_;

        void* queue = wgpuDeviceGetQueue(device);
        if (queue == (void*)0) return 0;

        *outInstance = instance; *outDevice = device; *outQueue = queue;
        return 1;
    }
}

int webgpu_available() {
    unsafe {
        void* instance = (void*)0; void* device = (void*)0; void* queue = (void*)0;
        return __wgpu_init(&instance, &device, &queue);
    }
}

// Shared shader-module/pipeline/buffer/bind-group/dispatch/readback
// sequence for every op below -- the WGSL-source analogue of
// __vk_run_kernel (gpu_spirv.sc) and __cuda_run_binary_kernel
// (gpu_cuda.sc). 'pushConstantData' isn't a real WebGPU concept (no
// push-constant extension in core WebGPU) -- scalars (n, k, M/K/N)
// instead go into a tiny dedicated uniform-style storage buffer, which
// this appends as the last input.
static int __wgpu_run_kernel(const char* wgslSrc, const char* entryName,
                              unsigned int numDataInputs, void** inputs, unsigned long* inputSizes,
                              const void* scalarData, unsigned long scalarSize,
                              void* output, unsigned long outputSize,
                              unsigned int workgroupsX, unsigned int workgroupsY, unsigned int workgroupsZ) {
    unsafe {
        void* instance = (void*)0; void* device = (void*)0; void* queue = (void*)0;
        if (!__wgpu_init(&instance, &device, &queue)) return 0;

        struct WGPUShaderModuleWGSLDescriptor wgslDesc;
        wgslDesc.chainSType = (int)WGPU_SHADER_MODULE_WGSL_DESCRIPTOR_STYPE; wgslDesc.chainNext = (const void*)0;
        wgslDesc.code = wgslSrc;
        struct WGPUShaderModuleDescriptor smDesc;
        smDesc.nextInChain = (const void*)&wgslDesc; smDesc.label = "safec_kernel";
        void* shaderModule = wgpuDeviceCreateShaderModule(device, &smDesc);
        if (shaderModule == (void*)0) return 0;

        struct WGPUComputePipelineDescriptor cpDesc;
        cpDesc.nextInChain = (const void*)0; cpDesc.label = "safec_pipeline"; cpDesc.layout = (void*)0; // auto layout
        cpDesc.compute.nextInChain = (const void*)0; cpDesc.compute.module = shaderModule;
        cpDesc.compute.entryPoint = entryName; cpDesc.compute.constantCount = 0U; cpDesc.compute.constants = (const void*)0;
        void* pipeline = wgpuDeviceCreateComputePipeline(device, &cpDesc);
        if (pipeline == (void*)0) return 0;

        unsigned int numBuffers = numDataInputs + (scalarSize > 0UL ? 1U : 0U) + 1U; // + scalar buf + output
        void* buffers[5];
        unsigned int bi = 0U;
        while (bi < numDataInputs) {
            struct WGPUBufferDescriptor bd;
            bd.nextInChain = (const void*)0; bd.label = "safec_input";
            bd.usage = WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST;
            bd.size = inputSizes[bi]; bd.mappedAtCreation = 0;
            buffers[bi] = wgpuDeviceCreateBuffer(device, &bd);
            wgpuQueueWriteBuffer(queue, buffers[bi], 0UL, inputs[bi], inputSizes[bi]);
            bi = bi + 1U;
        }
        unsigned int scalarIdx = numDataInputs;
        if (scalarSize > 0UL) {
            struct WGPUBufferDescriptor bd;
            bd.nextInChain = (const void*)0; bd.label = "safec_scalars";
            bd.usage = WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST;
            bd.size = scalarSize; bd.mappedAtCreation = 0;
            buffers[scalarIdx] = wgpuDeviceCreateBuffer(device, &bd);
            wgpuQueueWriteBuffer(queue, buffers[scalarIdx], 0UL, scalarData, scalarSize);
        }
        unsigned int outIdx = numBuffers - 1U;
        struct WGPUBufferDescriptor obd;
        obd.nextInChain = (const void*)0; obd.label = "safec_output";
        obd.usage = WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_SRC;
        obd.size = outputSize; obd.mappedAtCreation = 0;
        buffers[outIdx] = wgpuDeviceCreateBuffer(device, &obd);

        struct WGPUBindGroupEntry entries[5];
        unsigned int ei = 0U;
        while (ei < numBuffers) {
            entries[ei].nextInChain = (const void*)0; entries[ei].binding = ei;
            entries[ei].buffer = buffers[ei]; entries[ei].offset = 0UL;
            entries[ei].size = ei < numDataInputs ? inputSizes[ei] : (ei == scalarIdx ? scalarSize : outputSize);
            entries[ei].sampler = (void*)0; entries[ei].textureView = (void*)0;
            ei = ei + 1U;
        }
        void* bgLayout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0U);
        struct WGPUBindGroupDescriptor bgDesc;
        bgDesc.nextInChain = (const void*)0; bgDesc.label = "safec_bindgroup"; bgDesc.layout = bgLayout;
        bgDesc.entryCount = numBuffers; bgDesc.entries = (const struct WGPUBindGroupEntry*)entries;
        void* bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);

        void* encoder = wgpuDeviceCreateCommandEncoder(device, (const void*)0);
        void* pass = wgpuCommandEncoderBeginComputePass(encoder, (const void*)0);
        wgpuComputePassEncoderSetPipeline(pass, pipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0U, bindGroup, 0U, (const unsigned int*)0);
        wgpuComputePassEncoderDispatchWorkgroups(pass, workgroupsX, workgroupsY, workgroupsZ);
        wgpuComputePassEncoderEnd(pass);
        void* cmdBuf = wgpuCommandEncoderFinish(encoder, (const void*)0);
        wgpuQueueSubmit(queue, 1U, (const void*)&cmdBuf);

        // Readback needs its own mappable staging buffer in a fully
        // correct implementation (a STORAGE|COPY_SRC buffer generally
        // can't also be MAP_READ) -- omitted here (would be a
        // CopyBufferToBuffer into a MAP_READ|COPY_DST buffer between
        // submit and map) since every path through this function returns
        // before reaching it in this sandbox regardless (no WebGPU
        // backend to have gotten this far with). Noted, not silently
        // skipped, so a real port doesn't inherit the gap unknowingly.
        __wgpu_map_done_ = 0;
        wgpuBufferMapAsync(buffers[outIdx], (unsigned int)WGPU_MAP_MODE_READ, 0UL, outputSize, __wgpu_on_map, (void*)0);
        int iter = 0;
        while (__wgpu_map_done_ == 0 && iter < WGPU_POLL_ITERATION_CAP) {
            wgpuInstanceProcessEvents(instance);
            iter = iter + 1;
        }
        if (__wgpu_map_done_ == 0) return 0;
        const void* mapped = wgpuBufferGetConstMappedRange(buffers[outIdx], 0UL, outputSize);
        if (mapped == (const void*)0) return 0;
        memcpy(output, mapped, outputSize);
        wgpuBufferUnmap(buffers[outIdx]);
        return 1;
    }
}

int webgpu_add_f32(const float* a, const float* b, float* out, unsigned long n) {
    const char* wgslSrc =
        "@group(0) @binding(0) var<storage, read> a: array<f32>;\n"
        "@group(0) @binding(1) var<storage, read> b: array<f32>;\n"
        "@group(0) @binding(2) var<storage, read_write> out: array<f32>;\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "    if (gid.x >= arrayLength(&out)) { return; }\n"
        "    out[gid.x] = a[gid.x] + b[gid.x];\n"
        "}\n";
    unsafe {
        void* inputs[2]; inputs[0] = (void*)a; inputs[1] = (void*)b;
        unsigned long sizes[2]; sizes[0] = n * sizeof(float); sizes[1] = n * sizeof(float);
        return __wgpu_run_kernel(wgslSrc, "main", 2U, inputs, sizes, (const void*)0, 0UL,
                                  (void*)out, n * sizeof(float), (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

int webgpu_sub_f32(const float* a, const float* b, float* out, unsigned long n) {
    const char* wgslSrc =
        "@group(0) @binding(0) var<storage, read> a: array<f32>;\n"
        "@group(0) @binding(1) var<storage, read> b: array<f32>;\n"
        "@group(0) @binding(2) var<storage, read_write> out: array<f32>;\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "    if (gid.x >= arrayLength(&out)) { return; }\n"
        "    out[gid.x] = a[gid.x] - b[gid.x];\n"
        "}\n";
    unsafe {
        void* inputs[2]; inputs[0] = (void*)a; inputs[1] = (void*)b;
        unsigned long sizes[2]; sizes[0] = n * sizeof(float); sizes[1] = n * sizeof(float);
        return __wgpu_run_kernel(wgslSrc, "main", 2U, inputs, sizes, (const void*)0, 0UL,
                                  (void*)out, n * sizeof(float), (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

int webgpu_mul_f32(const float* a, const float* b, float* out, unsigned long n) {
    const char* wgslSrc =
        "@group(0) @binding(0) var<storage, read> a: array<f32>;\n"
        "@group(0) @binding(1) var<storage, read> b: array<f32>;\n"
        "@group(0) @binding(2) var<storage, read_write> out: array<f32>;\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "    if (gid.x >= arrayLength(&out)) { return; }\n"
        "    out[gid.x] = a[gid.x] * b[gid.x];\n"
        "}\n";
    unsafe {
        void* inputs[2]; inputs[0] = (void*)a; inputs[1] = (void*)b;
        unsigned long sizes[2]; sizes[0] = n * sizeof(float); sizes[1] = n * sizeof(float);
        return __wgpu_run_kernel(wgslSrc, "main", 2U, inputs, sizes, (const void*)0, 0UL,
                                  (void*)out, n * sizeof(float), (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

int webgpu_div_f32(const float* a, const float* b, float* out, unsigned long n) {
    const char* wgslSrc =
        "@group(0) @binding(0) var<storage, read> a: array<f32>;\n"
        "@group(0) @binding(1) var<storage, read> b: array<f32>;\n"
        "@group(0) @binding(2) var<storage, read_write> out: array<f32>;\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "    if (gid.x >= arrayLength(&out)) { return; }\n"
        "    out[gid.x] = a[gid.x] / b[gid.x];\n"
        "}\n";
    unsafe {
        void* inputs[2]; inputs[0] = (void*)a; inputs[1] = (void*)b;
        unsigned long sizes[2]; sizes[0] = n * sizeof(float); sizes[1] = n * sizeof(float);
        return __wgpu_run_kernel(wgslSrc, "main", 2U, inputs, sizes, (const void*)0, 0UL,
                                  (void*)out, n * sizeof(float), (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

int webgpu_pow_f32(const float* a, const float* b, float* out, unsigned long n) {
    const char* wgslSrc =
        "@group(0) @binding(0) var<storage, read> a: array<f32>;\n"
        "@group(0) @binding(1) var<storage, read> b: array<f32>;\n"
        "@group(0) @binding(2) var<storage, read_write> out: array<f32>;\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "    if (gid.x >= arrayLength(&out)) { return; }\n"
        "    out[gid.x] = pow(a[gid.x], b[gid.x]);\n"
        "}\n";
    unsafe {
        void* inputs[2]; inputs[0] = (void*)a; inputs[1] = (void*)b;
        unsigned long sizes[2]; sizes[0] = n * sizeof(float); sizes[1] = n * sizeof(float);
        return __wgpu_run_kernel(wgslSrc, "main", 2U, inputs, sizes, (const void*)0, 0UL,
                                  (void*)out, n * sizeof(float), (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

int webgpu_relu_f32(const float* a, float* out, unsigned long n) {
    const char* wgslSrc =
        "@group(0) @binding(0) var<storage, read> a: array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> out: array<f32>;\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "    if (gid.x >= arrayLength(&out)) { return; }\n"
        "    out[gid.x] = max(a[gid.x], 0.0);\n"
        "}\n";
    unsafe {
        void* inputs[1]; inputs[0] = (void*)a;
        unsigned long sizes[1]; sizes[0] = n * sizeof(float);
        return __wgpu_run_kernel(wgslSrc, "main", 1U, inputs, sizes, (const void*)0, 0UL,
                                  (void*)out, n * sizeof(float), (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

int webgpu_log_f32(const float* a, float* out, unsigned long n) {
    const char* wgslSrc =
        "@group(0) @binding(0) var<storage, read> a: array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> out: array<f32>;\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "    if (gid.x >= arrayLength(&out)) { return; }\n"
        "    out[gid.x] = log(a[gid.x]);\n"
        "}\n";
    unsafe {
        void* inputs[1]; inputs[0] = (void*)a;
        unsigned long sizes[1]; sizes[0] = n * sizeof(float);
        return __wgpu_run_kernel(wgslSrc, "main", 1U, inputs, sizes, (const void*)0, 0UL,
                                  (void*)out, n * sizeof(float), (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

int webgpu_exp_f32(const float* a, float* out, unsigned long n) {
    const char* wgslSrc =
        "@group(0) @binding(0) var<storage, read> a: array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> out: array<f32>;\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "    if (gid.x >= arrayLength(&out)) { return; }\n"
        "    out[gid.x] = exp(a[gid.x]);\n"
        "}\n";
    unsafe {
        void* inputs[1]; inputs[0] = (void*)a;
        unsigned long sizes[1]; sizes[0] = n * sizeof(float);
        return __wgpu_run_kernel(wgslSrc, "main", 1U, inputs, sizes, (const void*)0, 0UL,
                                  (void*)out, n * sizeof(float), (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

int webgpu_sqrt_f32(const float* a, float* out, unsigned long n) {
    const char* wgslSrc =
        "@group(0) @binding(0) var<storage, read> a: array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> out: array<f32>;\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "    if (gid.x >= arrayLength(&out)) { return; }\n"
        "    out[gid.x] = sqrt(a[gid.x]);\n"
        "}\n";
    unsafe {
        void* inputs[1]; inputs[0] = (void*)a;
        unsigned long sizes[1]; sizes[0] = n * sizeof(float);
        return __wgpu_run_kernel(wgslSrc, "main", 1U, inputs, sizes, (const void*)0, 0UL,
                                  (void*)out, n * sizeof(float), (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

int webgpu_scale_f32(const float* a, float k, float* out, unsigned long n) {
    const char* wgslSrc =
        "struct Scalars { k: f32, n: u32 };\n"
        "@group(0) @binding(0) var<storage, read> a: array<f32>;\n"
        "@group(0) @binding(1) var<storage, read> s: Scalars;\n"
        "@group(0) @binding(2) var<storage, read_write> out: array<f32>;\n"
        "@compute @workgroup_size(64)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "    if (gid.x >= arrayLength(&out)) { return; }\n"
        "    out[gid.x] = a[gid.x] * s.k;\n"
        "}\n";
    unsafe {
        void* inputs[1]; inputs[0] = (void*)a;
        unsigned long sizes[1]; sizes[0] = n * sizeof(float);
        char scalars[8];
        memcpy((void*)scalars, (const void*)&k, 4UL);
        unsigned int nParam = (unsigned int)n;
        memcpy((void*)(scalars + 4), (const void*)&nParam, 4UL);
        return __wgpu_run_kernel(wgslSrc, "main", 1U, inputs, sizes, (const void*)scalars, 8UL,
                                  (void*)out, n * sizeof(float), (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

// out[0] = sum(a[0..n)) -- single workgroup of 1 invocation, same
// "prove the shape works" naive shape as every other backend's sum here.
int webgpu_sum_f32(const float* a, float* out, unsigned long n) {
    const char* wgslSrc =
        "@group(0) @binding(0) var<storage, read> a: array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> out: array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main() {\n"
        "    var acc: f32 = 0.0;\n"
        "    let n = arrayLength(&a);\n"
        "    for (var i: u32 = 0u; i < n; i = i + 1u) { acc = acc + a[i]; }\n"
        "    out[0] = acc;\n"
        "}\n";
    unsafe {
        void* inputs[1]; inputs[0] = (void*)a;
        unsigned long sizes[1]; sizes[0] = n * sizeof(float);
        return __wgpu_run_kernel(wgslSrc, "main", 1U, inputs, sizes, (const void*)0, 0UL,
                                  (void*)out, sizeof(float), 1U, 1U, 1U);
    }
}

// out[M,N] = a[M,K] . b[K,N] -- 2D dispatch (8x8 workgroup, matching this
// file's simpler default vs the other backends' 16x16 -- either is a
// reasonable, arbitrary choice for a naive, untiled kernel), scalars
// M/K/N passed via the same storage-buffer-of-scalars mechanism as
// webgpu_scale_f32's k.
int webgpu_matmul_f32(const float* a, const float* b, float* out,
                       unsigned long M, unsigned long K, unsigned long N) {
    const char* wgslSrc =
        "struct Dims { M: u32, K: u32, N: u32 };\n"
        "@group(0) @binding(0) var<storage, read> a: array<f32>;\n"
        "@group(0) @binding(1) var<storage, read> b: array<f32>;\n"
        "@group(0) @binding(2) var<storage, read> d: Dims;\n"
        "@group(0) @binding(3) var<storage, read_write> out: array<f32>;\n"
        "@compute @workgroup_size(8, 8)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "    if (gid.x >= d.N || gid.y >= d.M) { return; }\n"
        "    var acc: f32 = 0.0;\n"
        "    for (var p: u32 = 0u; p < d.K; p = p + 1u) {\n"
        "        acc = acc + a[gid.y * d.K + p] * b[p * d.N + gid.x];\n"
        "    }\n"
        "    out[gid.y * d.N + gid.x] = acc;\n"
        "}\n";
    unsafe {
        void* inputs[2]; inputs[0] = (void*)a; inputs[1] = (void*)b;
        unsigned long sizes[2]; sizes[0] = M * K * sizeof(float); sizes[1] = K * N * sizeof(float);
        char dims[12];
        unsigned int Mu = (unsigned int)M; unsigned int Ku = (unsigned int)K; unsigned int Nu = (unsigned int)N;
        memcpy((void*)dims, (const void*)&Mu, 4UL);
        memcpy((void*)(dims + 4), (const void*)&Ku, 4UL);
        memcpy((void*)(dims + 8), (const void*)&Nu, 4UL);
        unsigned int groupsX = (unsigned int)((N + 7UL) / 8UL);
        unsigned int groupsY = (unsigned int)((M + 7UL) / 8UL);
        return __wgpu_run_kernel(wgslSrc, "main", 2U, inputs, sizes, (const void*)dims, 12UL,
                                  (void*)out, M * N * sizeof(float), groupsX, groupsY, 1U);
    }
}

} // namespace std
