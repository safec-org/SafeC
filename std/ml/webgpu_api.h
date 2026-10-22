#pragma once
// SafeC Standard Library — the part of the WebGPU C API (webgpu.h, as
// wgpu-native and Dawn ship it) that device_webgpu.sc and gpu_webgpu.sc
// call, declared for SafeC. Layouts follow webgpu.h's current revision (the
// one with WGPUStringView labels, callback-info structs and WGPUFuture
// returns: wgpu-native v24 on, Dawn from 2024 on); tests/stdlib/webgpu
// checks them against the real header with a C compiler (layout_check.c).
//
// Handles (WGPUInstance, WGPUDevice, WGPUBuffer, ...) are opaque pointers
// here: void*. Enums are 32-bit, flags (WGPUBufferUsage, WGPUMapMode,
// WGPUShaderStage) 64-bit, WGPUBool 32-bit. Callbacks receive WGPUStringView
// by value; SafeC passes and receives structs by value with the platform's C
// ABI, so the callback types below take it as one.
//
// The one wgpu-native extension used (wgpu.h's wgpuDevicePoll) is declared
// at the end; see device_webgpu.sc's header comment for Dawn.
#include <std/stddef.h>

namespace std {

struct WGPUStringView {
    const char* data;
    unsigned long length;       // WGPU_STRLEN (all ones): data is null-terminated
};
#define WGPU_STRLEN_ 0xffffffffffffffffUL

struct WGPUChainedStruct {
    void* next;
    int sType;
};

struct WGPUFuture {
    unsigned long id;
};

// ── Enums and flags (the values used here) ───────────────────────────────────
#define WGPU_STYPE_SHADER_SOURCE_WGSL 2
#define WGPU_CALLBACK_MODE_WAIT_ANY_ONLY 1
#define WGPU_CALLBACK_MODE_ALLOW_PROCESS_EVENTS 2
#define WGPU_POWER_HIGH_PERFORMANCE 2
#define WGPU_REQUEST_STATUS_SUCCESS 1        // adapter, device, map, work done
#define WGPU_STATUS_SUCCESS 1
#define WGPU_FEATURE_SHADER_F16 11
#define WGPU_FEATURE_SUBGROUPS 18
#define WGPU_BUFFER_BINDING_UNIFORM 2
#define WGPU_BUFFER_BINDING_STORAGE 3
#define WGPU_BUFFER_BINDING_READ_ONLY_STORAGE 4
#define WGPU_BUFFER_USAGE_MAP_READ 1UL
#define WGPU_BUFFER_USAGE_MAP_WRITE 2UL
#define WGPU_BUFFER_USAGE_COPY_SRC 4UL
#define WGPU_BUFFER_USAGE_COPY_DST 8UL
#define WGPU_BUFFER_USAGE_UNIFORM 64UL
#define WGPU_BUFFER_USAGE_STORAGE 128UL
#define WGPU_MAP_MODE_READ 1UL
#define WGPU_SHADER_STAGE_COMPUTE 4UL
#define WGPU_WHOLE_SIZE_ 0xffffffffffffffffUL

// ── Callbacks ────────────────────────────────────────────────────────────────
typedef fn void(int, void*, struct WGPUStringView, void*, void*) WGPURequestAdapterCallback;   // status, adapter
typedef fn void(int, void*, struct WGPUStringView, void*, void*) WGPURequestDeviceCallback;    // status, device
typedef fn void(int, struct WGPUStringView, void*, void*) WGPUBufferMapCallback;               // status
typedef fn void(int, struct WGPUStringView, void*, void*) WGPUQueueWorkDoneCallback;           // status
typedef fn void(const void*, int, struct WGPUStringView, void*, void*) WGPUUncapturedErrorCallback;   // &device, type
typedef fn void(const void*, int, struct WGPUStringView, void*, void*) WGPUDeviceLostCallback;        // &device, reason

struct WGPURequestAdapterCallbackInfo {
    void* nextInChain;
    int mode;
    WGPURequestAdapterCallback callback;
    void* userdata1;
    void* userdata2;
};
struct WGPURequestDeviceCallbackInfo {
    void* nextInChain;
    int mode;
    WGPURequestDeviceCallback callback;
    void* userdata1;
    void* userdata2;
};
struct WGPUBufferMapCallbackInfo {
    void* nextInChain;
    int mode;
    WGPUBufferMapCallback callback;
    void* userdata1;
    void* userdata2;
};
struct WGPUQueueWorkDoneCallbackInfo {
    void* nextInChain;
    int mode;
    WGPUQueueWorkDoneCallback callback;
    void* userdata1;
    void* userdata2;
};
struct WGPUDeviceLostCallbackInfo {
    void* nextInChain;
    int mode;
    WGPUDeviceLostCallback callback;
    void* userdata1;
    void* userdata2;
};
struct WGPUUncapturedErrorCallbackInfo {
    void* nextInChain;
    WGPUUncapturedErrorCallback callback;
    void* userdata1;
    void* userdata2;
};

// ── Descriptors ──────────────────────────────────────────────────────────────
struct WGPUInstanceLimits {
    void* nextInChain;
    unsigned long timedWaitAnyMaxCount;
};
struct WGPUInstanceDescriptor {
    void* nextInChain;
    unsigned long requiredFeatureCount;
    const int* requiredFeatures;
    const struct WGPUInstanceLimits* requiredLimits;
};
struct WGPURequestAdapterOptions {
    void* nextInChain;
    int featureLevel;
    int powerPreference;
    unsigned int forceFallbackAdapter;
    int backendType;
    void* compatibleSurface;
};
struct WGPULimits {
    void* nextInChain;
    unsigned int maxTextureDimension1D;
    unsigned int maxTextureDimension2D;
    unsigned int maxTextureDimension3D;
    unsigned int maxTextureArrayLayers;
    unsigned int maxBindGroups;
    unsigned int maxBindGroupsPlusVertexBuffers;
    unsigned int maxBindingsPerBindGroup;
    unsigned int maxDynamicUniformBuffersPerPipelineLayout;
    unsigned int maxDynamicStorageBuffersPerPipelineLayout;
    unsigned int maxSampledTexturesPerShaderStage;
    unsigned int maxSamplersPerShaderStage;
    unsigned int maxStorageBuffersPerShaderStage;
    unsigned int maxStorageTexturesPerShaderStage;
    unsigned int maxUniformBuffersPerShaderStage;
    unsigned long maxUniformBufferBindingSize;
    unsigned long maxStorageBufferBindingSize;
    unsigned int minUniformBufferOffsetAlignment;
    unsigned int minStorageBufferOffsetAlignment;
    unsigned int maxVertexBuffers;
    unsigned long maxBufferSize;
    unsigned int maxVertexAttributes;
    unsigned int maxVertexBufferArrayStride;
    unsigned int maxInterStageShaderVariables;
    unsigned int maxColorAttachments;
    unsigned int maxColorAttachmentBytesPerSample;
    unsigned int maxComputeWorkgroupStorageSize;
    unsigned int maxComputeInvocationsPerWorkgroup;
    unsigned int maxComputeWorkgroupSizeX;
    unsigned int maxComputeWorkgroupSizeY;
    unsigned int maxComputeWorkgroupSizeZ;
    unsigned int maxComputeWorkgroupsPerDimension;
    unsigned int maxImmediateSize;
};
struct WGPUQueueDescriptor {
    void* nextInChain;
    struct WGPUStringView label;
};
struct WGPUDeviceDescriptor {
    void* nextInChain;
    struct WGPUStringView label;
    unsigned long requiredFeatureCount;
    const int* requiredFeatures;
    const struct WGPULimits* requiredLimits;
    struct WGPUQueueDescriptor defaultQueue;
    struct WGPUDeviceLostCallbackInfo deviceLostCallbackInfo;
    struct WGPUUncapturedErrorCallbackInfo uncapturedErrorCallbackInfo;
};
struct WGPUAdapterInfo {
    void* nextInChain;
    struct WGPUStringView vendor;
    struct WGPUStringView architecture;
    struct WGPUStringView device;
    struct WGPUStringView description;
    int backendType;
    int adapterType;
    unsigned int vendorID;
    unsigned int deviceID;
    unsigned int subgroupMinSize;
    unsigned int subgroupMaxSize;
};
struct WGPUBufferDescriptor {
    void* nextInChain;
    struct WGPUStringView label;
    unsigned long usage;
    unsigned long size;
    unsigned int mappedAtCreation;
};
struct WGPUShaderSourceWGSL {
    struct WGPUChainedStruct chain;
    struct WGPUStringView code;
};
struct WGPUShaderModuleDescriptor {
    void* nextInChain;
    struct WGPUStringView label;
};
struct WGPUBufferBindingLayout {
    void* nextInChain;
    int type;
    unsigned int hasDynamicOffset;
    unsigned long minBindingSize;
};
// A layout entry for a buffer binding; the sampler / texture / storage
// texture members (64 bytes) stay zero (BindingNotUsed).
struct WGPUBindGroupLayoutEntry {
    void* nextInChain;
    unsigned int binding;
    unsigned long visibility;
    unsigned int bindingArraySize;
    struct WGPUBufferBindingLayout buffer;
    unsigned long unusedBindings[8];
};
struct WGPUBindGroupLayoutDescriptor {
    void* nextInChain;
    struct WGPUStringView label;
    unsigned long entryCount;
    const struct WGPUBindGroupLayoutEntry* entries;
};
struct WGPUPipelineLayoutDescriptor {
    void* nextInChain;
    struct WGPUStringView label;
    unsigned long bindGroupLayoutCount;
    const void* const* bindGroupLayouts;
    unsigned int immediateSize;
};
struct WGPUBindGroupEntry {
    void* nextInChain;
    unsigned int binding;
    void* buffer;
    unsigned long offset;
    unsigned long size;
    void* sampler;
    void* textureView;
};
struct WGPUBindGroupDescriptor {
    void* nextInChain;
    struct WGPUStringView label;
    void* layout;
    unsigned long entryCount;
    const struct WGPUBindGroupEntry* entries;
};
struct WGPUConstantEntry {
    void* nextInChain;
    struct WGPUStringView key;
    double value;
};
struct WGPUComputeState {
    void* nextInChain;
    void* module;
    struct WGPUStringView entryPoint;
    unsigned long constantCount;
    const struct WGPUConstantEntry* constants;
};
struct WGPUComputePipelineDescriptor {
    void* nextInChain;
    struct WGPUStringView label;
    void* layout;
    struct WGPUComputeState compute;
};

// ── Functions ────────────────────────────────────────────────────────────────
extern void* wgpuCreateInstance(const struct WGPUInstanceDescriptor* descriptor);
extern struct WGPUFuture wgpuInstanceRequestAdapter(void* instance, const struct WGPURequestAdapterOptions* options,
                                                    struct WGPURequestAdapterCallbackInfo callbackInfo);
extern void wgpuInstanceProcessEvents(void* instance);
extern struct WGPUFuture wgpuAdapterRequestDevice(void* adapter, const struct WGPUDeviceDescriptor* descriptor,
                                                  struct WGPURequestDeviceCallbackInfo callbackInfo);
extern int wgpuAdapterGetLimits(void* adapter, struct WGPULimits* limits);
extern int wgpuAdapterGetInfo(void* adapter, struct WGPUAdapterInfo* info);
extern void wgpuAdapterInfoFreeMembers(struct WGPUAdapterInfo adapterInfo);
extern unsigned int wgpuAdapterHasFeature(void* adapter, int feature);
extern void* wgpuDeviceGetQueue(void* device);
extern void* wgpuDeviceCreateBuffer(void* device, const struct WGPUBufferDescriptor* descriptor);
extern void wgpuBufferRelease(void* buffer);
extern void wgpuBufferDestroy(void* buffer);
extern void* wgpuDeviceCreateShaderModule(void* device, const struct WGPUShaderModuleDescriptor* descriptor);
extern void* wgpuDeviceCreateBindGroupLayout(void* device, const struct WGPUBindGroupLayoutDescriptor* descriptor);
extern void* wgpuDeviceCreatePipelineLayout(void* device, const struct WGPUPipelineLayoutDescriptor* descriptor);
extern void* wgpuDeviceCreateComputePipeline(void* device, const struct WGPUComputePipelineDescriptor* descriptor);
extern void wgpuComputePipelineRelease(void* pipeline);
extern void* wgpuDeviceCreateBindGroup(void* device, const struct WGPUBindGroupDescriptor* descriptor);
extern void wgpuBindGroupRelease(void* bindGroup);
extern void* wgpuDeviceCreateCommandEncoder(void* device, const void* descriptor);
extern void* wgpuCommandEncoderBeginComputePass(void* encoder, const void* descriptor);
extern void wgpuComputePassEncoderSetPipeline(void* pass, void* pipeline);
extern void wgpuComputePassEncoderSetBindGroup(void* pass, unsigned int groupIndex, void* group,
                                               unsigned long dynamicOffsetCount, const unsigned int* dynamicOffsets);
extern void wgpuComputePassEncoderDispatchWorkgroups(void* pass, unsigned int x, unsigned int y, unsigned int z);
extern void wgpuComputePassEncoderEnd(void* pass);
extern void wgpuComputePassEncoderRelease(void* pass);
extern void wgpuCommandEncoderCopyBufferToBuffer(void* encoder, void* source, unsigned long sourceOffset,
                                                 void* destination, unsigned long destinationOffset, unsigned long size);
extern void* wgpuCommandEncoderFinish(void* encoder, const void* descriptor);
extern void wgpuCommandEncoderRelease(void* encoder);
extern void wgpuCommandBufferRelease(void* commandBuffer);
extern void wgpuQueueSubmit(void* queue, unsigned long commandCount, void* const* commands);
extern struct WGPUFuture wgpuQueueOnSubmittedWorkDone(void* queue, struct WGPUQueueWorkDoneCallbackInfo callbackInfo);
extern void wgpuQueueWriteBuffer(void* queue, void* buffer, unsigned long bufferOffset, const void* data,
                                 unsigned long size);
extern struct WGPUFuture wgpuBufferMapAsync(void* buffer, unsigned long mode, unsigned long offset, unsigned long size,
                                            struct WGPUBufferMapCallbackInfo callbackInfo);
extern const void* wgpuBufferGetConstMappedRange(void* buffer, unsigned long offset, unsigned long size);
extern void wgpuBufferUnmap(void* buffer);

// ── wgpu-native extensions (wgpu.h) ──────────────────────────────────────────
// Runs the device's maintenance (finished work, mapped buffers) so the
// instance's callbacks can fire; with wait, first blocks until submission
// *index (every submission when null) is done. 1 when the queue is empty.
extern unsigned int wgpuDevicePoll(void* device, unsigned int wait, const unsigned long* submissionIndex);

} // namespace std
