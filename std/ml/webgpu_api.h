#pragma once
// SafeC Standard Library — the part of the WebGPU C API (webgpu.h, as
// wgpu-native and Dawn ship it) that device_webgpu.sc and gpu_webgpu.sc
// call, declared for SafeC. Layouts follow webgpu.h's current revision (the
// one with WGPUStringView labels, callback-info structs and WGPUFuture
// returns: wgpu-native v24 on, Dawn from 2024 on); tests/stdlib/webgpu
// checks them against the real header with a C compiler (layout_check.c).
//
// Handles (WGPUInstance, WGPUDevice, WGPUBuffer, ...) are pointers to
// opaque marker structs here, one per kind. Enums are 32-bit, flags (WGPUBufferUsage, WGPUMapMode,
// WGPUShaderStage) 64-bit, WGPUBool 32-bit. Callbacks receive WGPUStringView
// by value; SafeC passes and receives structs by value with the platform's C
// ABI, so the callback types below take it as one.
//
// The two wgpu-native extensions used (wgpu.h's wgpuDevicePoll and
// wgpuDeviceCreateShaderModuleTrusted) are declared at the end; see
// device_webgpu.sc's header comment for Dawn.
#include <std/stddef.h>

namespace std {

struct WGPUStringView {
    const char* data;
    unsigned long length;       // WGPU_STRLEN (all ones): data is null-terminated
};
#define WGPU_STRLEN_ 0xffffffffffffffffUL

// ── Handles ──────────────────────────────────────────────────────────────────
// Never constructed or read; only the implementation looks behind them.
struct WGPUInstance_ { char opaque_; };
struct WGPUAdapter_ { char opaque_; };
struct WGPUDevice_ { char opaque_; };
struct WGPUQueue_ { char opaque_; };
struct WGPUBuffer_ { char opaque_; };
struct WGPUShaderModule_ { char opaque_; };
struct WGPUBindGroupLayout_ { char opaque_; };
struct WGPUPipelineLayout_ { char opaque_; };
struct WGPUComputePipeline_ { char opaque_; };
struct WGPUBindGroup_ { char opaque_; };
struct WGPUCommandEncoder_ { char opaque_; };
struct WGPUComputePassEncoder_ { char opaque_; };
struct WGPUCommandBuffer_ { char opaque_; };
struct WGPUSurface_ { char opaque_; };
struct WGPUSampler_ { char opaque_; };
struct WGPUTextureView_ { char opaque_; };
// Descriptors std never passes (always null).
struct WGPUCommandEncoderDescriptor_ { char opaque_; };
struct WGPUComputePassDescriptor_ { char opaque_; };
struct WGPUCommandBufferDescriptor_ { char opaque_; };
// A callback's userdata: whatever the callback info carried, handed back
// untouched (std passes a serial number in one, never dereferenced).
struct WGPUUserdata_ { char opaque_; };

struct WGPUChainedStruct {
    struct WGPUChainedStruct* next;
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
typedef fn void(int, WGPUAdapter_*, struct WGPUStringView, WGPUUserdata_*, WGPUUserdata_*) WGPURequestAdapterCallback;   // status, adapter
typedef fn void(int, WGPUDevice_*, struct WGPUStringView, WGPUUserdata_*, WGPUUserdata_*) WGPURequestDeviceCallback;    // status, device
typedef fn void(int, struct WGPUStringView, WGPUUserdata_*, WGPUUserdata_*) WGPUBufferMapCallback;               // status
typedef fn void(int, struct WGPUStringView, WGPUUserdata_*, WGPUUserdata_*) WGPUQueueWorkDoneCallback;           // status
typedef fn void(struct WGPUDevice_**, int, struct WGPUStringView, WGPUUserdata_*, WGPUUserdata_*) WGPUUncapturedErrorCallback;   // &device, type
typedef fn void(struct WGPUDevice_**, int, struct WGPUStringView, WGPUUserdata_*, WGPUUserdata_*) WGPUDeviceLostCallback;        // &device, reason

struct WGPURequestAdapterCallbackInfo {
    struct WGPUChainedStruct* nextInChain;
    int mode;
    WGPURequestAdapterCallback callback;
    WGPUUserdata_* userdata1;
    WGPUUserdata_* userdata2;
};
struct WGPURequestDeviceCallbackInfo {
    struct WGPUChainedStruct* nextInChain;
    int mode;
    WGPURequestDeviceCallback callback;
    WGPUUserdata_* userdata1;
    WGPUUserdata_* userdata2;
};
struct WGPUBufferMapCallbackInfo {
    struct WGPUChainedStruct* nextInChain;
    int mode;
    WGPUBufferMapCallback callback;
    WGPUUserdata_* userdata1;
    WGPUUserdata_* userdata2;
};
struct WGPUQueueWorkDoneCallbackInfo {
    struct WGPUChainedStruct* nextInChain;
    int mode;
    WGPUQueueWorkDoneCallback callback;
    WGPUUserdata_* userdata1;
    WGPUUserdata_* userdata2;
};
struct WGPUDeviceLostCallbackInfo {
    struct WGPUChainedStruct* nextInChain;
    int mode;
    WGPUDeviceLostCallback callback;
    WGPUUserdata_* userdata1;
    WGPUUserdata_* userdata2;
};
struct WGPUUncapturedErrorCallbackInfo {
    struct WGPUChainedStruct* nextInChain;
    WGPUUncapturedErrorCallback callback;
    WGPUUserdata_* userdata1;
    WGPUUserdata_* userdata2;
};

// ── Descriptors ──────────────────────────────────────────────────────────────
struct WGPUInstanceLimits {
    struct WGPUChainedStruct* nextInChain;
    unsigned long timedWaitAnyMaxCount;
};
struct WGPUInstanceDescriptor {
    struct WGPUChainedStruct* nextInChain;
    unsigned long requiredFeatureCount;
    const int* requiredFeatures;
    const struct WGPUInstanceLimits* requiredLimits;
};
struct WGPURequestAdapterOptions {
    struct WGPUChainedStruct* nextInChain;
    int featureLevel;
    int powerPreference;
    unsigned int forceFallbackAdapter;
    int backendType;
    WGPUSurface_* compatibleSurface;
};
struct WGPULimits {
    struct WGPUChainedStruct* nextInChain;
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
    struct WGPUChainedStruct* nextInChain;
    struct WGPUStringView label;
};
struct WGPUDeviceDescriptor {
    struct WGPUChainedStruct* nextInChain;
    struct WGPUStringView label;
    unsigned long requiredFeatureCount;
    const int* requiredFeatures;
    const struct WGPULimits* requiredLimits;
    struct WGPUQueueDescriptor defaultQueue;
    struct WGPUDeviceLostCallbackInfo deviceLostCallbackInfo;
    struct WGPUUncapturedErrorCallbackInfo uncapturedErrorCallbackInfo;
};
struct WGPUAdapterInfo {
    struct WGPUChainedStruct* nextInChain;
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
    struct WGPUChainedStruct* nextInChain;
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
    struct WGPUChainedStruct* nextInChain;
    struct WGPUStringView label;
};
struct WGPUBufferBindingLayout {
    struct WGPUChainedStruct* nextInChain;
    int type;
    unsigned int hasDynamicOffset;
    unsigned long minBindingSize;
};
// A layout entry for a buffer binding; the sampler / texture / storage
// texture members (64 bytes) stay zero (BindingNotUsed).
struct WGPUBindGroupLayoutEntry {
    struct WGPUChainedStruct* nextInChain;
    unsigned int binding;
    unsigned long visibility;
    unsigned int bindingArraySize;
    struct WGPUBufferBindingLayout buffer;
    unsigned long unusedBindings[8];
};
struct WGPUBindGroupLayoutDescriptor {
    struct WGPUChainedStruct* nextInChain;
    struct WGPUStringView label;
    unsigned long entryCount;
    const struct WGPUBindGroupLayoutEntry* entries;
};
struct WGPUPipelineLayoutDescriptor {
    struct WGPUChainedStruct* nextInChain;
    struct WGPUStringView label;
    unsigned long bindGroupLayoutCount;
    struct WGPUBindGroupLayout_** bindGroupLayouts;
    unsigned int immediateSize;
};
struct WGPUBindGroupEntry {
    struct WGPUChainedStruct* nextInChain;
    unsigned int binding;
    WGPUBuffer_* buffer;
    unsigned long offset;
    unsigned long size;
    WGPUSampler_* sampler;
    WGPUTextureView_* textureView;
};
struct WGPUBindGroupDescriptor {
    struct WGPUChainedStruct* nextInChain;
    struct WGPUStringView label;
    WGPUBindGroupLayout_* layout;
    unsigned long entryCount;
    const struct WGPUBindGroupEntry* entries;
};
struct WGPUConstantEntry {
    struct WGPUChainedStruct* nextInChain;
    struct WGPUStringView key;
    double value;
};
struct WGPUComputeState {
    struct WGPUChainedStruct* nextInChain;
    WGPUShaderModule_* module;
    struct WGPUStringView entryPoint;
    unsigned long constantCount;
    const struct WGPUConstantEntry* constants;
};
struct WGPUComputePipelineDescriptor {
    struct WGPUChainedStruct* nextInChain;
    struct WGPUStringView label;
    WGPUPipelineLayout_* layout;
    struct WGPUComputeState compute;
};

// ── Functions ────────────────────────────────────────────────────────────────
extern WGPUInstance_* wgpuCreateInstance(const struct WGPUInstanceDescriptor* descriptor);
extern struct WGPUFuture wgpuInstanceRequestAdapter(WGPUInstance_* instance, const struct WGPURequestAdapterOptions* options,
                                                    struct WGPURequestAdapterCallbackInfo callbackInfo);
extern void wgpuInstanceProcessEvents(WGPUInstance_* instance);
extern struct WGPUFuture wgpuAdapterRequestDevice(WGPUAdapter_* adapter, const struct WGPUDeviceDescriptor* descriptor,
                                                  struct WGPURequestDeviceCallbackInfo callbackInfo);
extern int wgpuAdapterGetLimits(WGPUAdapter_* adapter, struct WGPULimits* limits);
extern int wgpuAdapterGetInfo(WGPUAdapter_* adapter, struct WGPUAdapterInfo* info);
extern void wgpuAdapterInfoFreeMembers(struct WGPUAdapterInfo adapterInfo);
extern unsigned int wgpuAdapterHasFeature(WGPUAdapter_* adapter, int feature);
extern WGPUQueue_* wgpuDeviceGetQueue(WGPUDevice_* device);
extern WGPUBuffer_* wgpuDeviceCreateBuffer(WGPUDevice_* device, const struct WGPUBufferDescriptor* descriptor);
extern void wgpuBufferRelease(WGPUBuffer_* buffer);
extern void wgpuBufferDestroy(WGPUBuffer_* buffer);
extern WGPUShaderModule_* wgpuDeviceCreateShaderModule(WGPUDevice_* device, const struct WGPUShaderModuleDescriptor* descriptor);
extern WGPUBindGroupLayout_* wgpuDeviceCreateBindGroupLayout(WGPUDevice_* device, const struct WGPUBindGroupLayoutDescriptor* descriptor);
extern WGPUPipelineLayout_* wgpuDeviceCreatePipelineLayout(WGPUDevice_* device, const struct WGPUPipelineLayoutDescriptor* descriptor);
extern WGPUComputePipeline_* wgpuDeviceCreateComputePipeline(WGPUDevice_* device, const struct WGPUComputePipelineDescriptor* descriptor);
extern void wgpuComputePipelineRelease(WGPUComputePipeline_* pipeline);
extern WGPUBindGroup_* wgpuDeviceCreateBindGroup(WGPUDevice_* device, const struct WGPUBindGroupDescriptor* descriptor);
extern void wgpuBindGroupRelease(WGPUBindGroup_* bindGroup);
extern WGPUCommandEncoder_* wgpuDeviceCreateCommandEncoder(WGPUDevice_* device, const WGPUCommandEncoderDescriptor_* descriptor);
extern WGPUComputePassEncoder_* wgpuCommandEncoderBeginComputePass(WGPUCommandEncoder_* encoder, const WGPUComputePassDescriptor_* descriptor);
extern void wgpuComputePassEncoderSetPipeline(WGPUComputePassEncoder_* pass, WGPUComputePipeline_* pipeline);
extern void wgpuComputePassEncoderSetBindGroup(WGPUComputePassEncoder_* pass, unsigned int groupIndex, WGPUBindGroup_* group,
                                               unsigned long dynamicOffsetCount, const unsigned int* dynamicOffsets);
extern void wgpuComputePassEncoderDispatchWorkgroups(WGPUComputePassEncoder_* pass, unsigned int x, unsigned int y, unsigned int z);
extern void wgpuComputePassEncoderEnd(WGPUComputePassEncoder_* pass);
extern void wgpuComputePassEncoderRelease(WGPUComputePassEncoder_* pass);
extern void wgpuCommandEncoderCopyBufferToBuffer(WGPUCommandEncoder_* encoder, WGPUBuffer_* source, unsigned long sourceOffset,
                                                 WGPUBuffer_* destination, unsigned long destinationOffset, unsigned long size);
extern WGPUCommandBuffer_* wgpuCommandEncoderFinish(WGPUCommandEncoder_* encoder, const WGPUCommandBufferDescriptor_* descriptor);
extern void wgpuCommandEncoderRelease(WGPUCommandEncoder_* encoder);
extern void wgpuCommandBufferRelease(WGPUCommandBuffer_* commandBuffer);
extern void wgpuQueueSubmit(WGPUQueue_* queue, unsigned long commandCount, struct WGPUCommandBuffer_** commands);
extern struct WGPUFuture wgpuQueueOnSubmittedWorkDone(WGPUQueue_* queue, struct WGPUQueueWorkDoneCallbackInfo callbackInfo);
extern void wgpuQueueWriteBuffer(WGPUQueue_* queue, WGPUBuffer_* buffer, unsigned long bufferOffset, const unsigned char* data,
                                 unsigned long size);
extern struct WGPUFuture wgpuBufferMapAsync(WGPUBuffer_* buffer, unsigned long mode, unsigned long offset, unsigned long size,
                                            struct WGPUBufferMapCallbackInfo callbackInfo);
extern const unsigned char* wgpuBufferGetConstMappedRange(WGPUBuffer_* buffer, unsigned long offset, unsigned long size);
extern void wgpuBufferUnmap(WGPUBuffer_* buffer);

// ── wgpu-native extensions (wgpu.h) ──────────────────────────────────────────
// A shader module without the runtime checks wgpu adds to every shader
// (bounds checks on each buffer access, a counter in each loop); checks is
// a WGPUShaderRuntimeChecks mask, 0 for none.
#define WGPU_RUNTIME_CHECKS_BOUNDS 1UL
#define WGPU_RUNTIME_CHECKS_LOOP_BOUNDING 2UL
extern WGPUShaderModule_* wgpuDeviceCreateShaderModuleTrusted(WGPUDevice_* device, const struct WGPUShaderModuleDescriptor* descriptor,
                                                 unsigned long checks);
// Runs the device's maintenance (finished work, mapped buffers) so the
// instance's callbacks can fire; with wait, first blocks until submission
// *index (every submission when null) is done. 1 when the queue is empty.
extern unsigned int wgpuDevicePoll(WGPUDevice_* device, unsigned int wait, const unsigned long* submissionIndex);

} // namespace std
