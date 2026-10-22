// Generated alongside layout_check.sc: the layouts of the webgpu.h structs
// std/ml/webgpu_api.h declares, from the real header. run.sh diffs the two.
#include <stdio.h>
#include <stddef.h>
#include <webgpu/webgpu.h>

int main(void) {
    { WGPUStringView v; printf("WGPUStringView %zu", sizeof(v));
      printf(" data@%zu", offsetof(WGPUStringView, data));
      printf(" length@%zu", offsetof(WGPUStringView, length));
      printf("\n"); }
    { WGPUChainedStruct v; printf("WGPUChainedStruct %zu", sizeof(v));
      printf(" next@%zu", offsetof(WGPUChainedStruct, next));
      printf(" sType@%zu", offsetof(WGPUChainedStruct, sType));
      printf("\n"); }
    { WGPURequestAdapterCallbackInfo v; printf("WGPURequestAdapterCallbackInfo %zu", sizeof(v));
      printf(" mode@%zu", offsetof(WGPURequestAdapterCallbackInfo, mode));
      printf(" callback@%zu", offsetof(WGPURequestAdapterCallbackInfo, callback));
      printf(" userdata1@%zu", offsetof(WGPURequestAdapterCallbackInfo, userdata1));
      printf(" userdata2@%zu", offsetof(WGPURequestAdapterCallbackInfo, userdata2));
      printf("\n"); }
    { WGPURequestDeviceCallbackInfo v; printf("WGPURequestDeviceCallbackInfo %zu", sizeof(v));
      printf(" mode@%zu", offsetof(WGPURequestDeviceCallbackInfo, mode));
      printf(" callback@%zu", offsetof(WGPURequestDeviceCallbackInfo, callback));
      printf(" userdata2@%zu", offsetof(WGPURequestDeviceCallbackInfo, userdata2));
      printf("\n"); }
    { WGPUBufferMapCallbackInfo v; printf("WGPUBufferMapCallbackInfo %zu", sizeof(v));
      printf(" mode@%zu", offsetof(WGPUBufferMapCallbackInfo, mode));
      printf(" callback@%zu", offsetof(WGPUBufferMapCallbackInfo, callback));
      printf(" userdata2@%zu", offsetof(WGPUBufferMapCallbackInfo, userdata2));
      printf("\n"); }
    { WGPUQueueWorkDoneCallbackInfo v; printf("WGPUQueueWorkDoneCallbackInfo %zu", sizeof(v));
      printf(" mode@%zu", offsetof(WGPUQueueWorkDoneCallbackInfo, mode));
      printf(" callback@%zu", offsetof(WGPUQueueWorkDoneCallbackInfo, callback));
      printf(" userdata2@%zu", offsetof(WGPUQueueWorkDoneCallbackInfo, userdata2));
      printf("\n"); }
    { WGPUDeviceLostCallbackInfo v; printf("WGPUDeviceLostCallbackInfo %zu", sizeof(v));
      printf(" mode@%zu", offsetof(WGPUDeviceLostCallbackInfo, mode));
      printf(" callback@%zu", offsetof(WGPUDeviceLostCallbackInfo, callback));
      printf(" userdata2@%zu", offsetof(WGPUDeviceLostCallbackInfo, userdata2));
      printf("\n"); }
    { WGPUUncapturedErrorCallbackInfo v; printf("WGPUUncapturedErrorCallbackInfo %zu", sizeof(v));
      printf(" callback@%zu", offsetof(WGPUUncapturedErrorCallbackInfo, callback));
      printf(" userdata1@%zu", offsetof(WGPUUncapturedErrorCallbackInfo, userdata1));
      printf(" userdata2@%zu", offsetof(WGPUUncapturedErrorCallbackInfo, userdata2));
      printf("\n"); }
    { WGPUInstanceDescriptor v; printf("WGPUInstanceDescriptor %zu", sizeof(v));
      printf(" requiredFeatureCount@%zu", offsetof(WGPUInstanceDescriptor, requiredFeatureCount));
      printf(" requiredFeatures@%zu", offsetof(WGPUInstanceDescriptor, requiredFeatures));
      printf(" requiredLimits@%zu", offsetof(WGPUInstanceDescriptor, requiredLimits));
      printf("\n"); }
    { WGPURequestAdapterOptions v; printf("WGPURequestAdapterOptions %zu", sizeof(v));
      printf(" featureLevel@%zu", offsetof(WGPURequestAdapterOptions, featureLevel));
      printf(" powerPreference@%zu", offsetof(WGPURequestAdapterOptions, powerPreference));
      printf(" forceFallbackAdapter@%zu", offsetof(WGPURequestAdapterOptions, forceFallbackAdapter));
      printf(" backendType@%zu", offsetof(WGPURequestAdapterOptions, backendType));
      printf(" compatibleSurface@%zu", offsetof(WGPURequestAdapterOptions, compatibleSurface));
      printf("\n"); }
    { WGPULimits v; printf("WGPULimits %zu", sizeof(v));
      printf(" maxStorageBuffersPerShaderStage@%zu", offsetof(WGPULimits, maxStorageBuffersPerShaderStage));
      printf(" maxUniformBufferBindingSize@%zu", offsetof(WGPULimits, maxUniformBufferBindingSize));
      printf(" maxStorageBufferBindingSize@%zu", offsetof(WGPULimits, maxStorageBufferBindingSize));
      printf(" minUniformBufferOffsetAlignment@%zu", offsetof(WGPULimits, minUniformBufferOffsetAlignment));
      printf(" maxVertexBuffers@%zu", offsetof(WGPULimits, maxVertexBuffers));
      printf(" maxBufferSize@%zu", offsetof(WGPULimits, maxBufferSize));
      printf(" maxComputeWorkgroupStorageSize@%zu", offsetof(WGPULimits, maxComputeWorkgroupStorageSize));
      printf(" maxComputeWorkgroupsPerDimension@%zu", offsetof(WGPULimits, maxComputeWorkgroupsPerDimension));
      printf(" maxImmediateSize@%zu", offsetof(WGPULimits, maxImmediateSize));
      printf("\n"); }
    { WGPUDeviceDescriptor v; printf("WGPUDeviceDescriptor %zu", sizeof(v));
      printf(" label@%zu", offsetof(WGPUDeviceDescriptor, label));
      printf(" requiredFeatureCount@%zu", offsetof(WGPUDeviceDescriptor, requiredFeatureCount));
      printf(" requiredFeatures@%zu", offsetof(WGPUDeviceDescriptor, requiredFeatures));
      printf(" requiredLimits@%zu", offsetof(WGPUDeviceDescriptor, requiredLimits));
      printf(" defaultQueue@%zu", offsetof(WGPUDeviceDescriptor, defaultQueue));
      printf(" deviceLostCallbackInfo@%zu", offsetof(WGPUDeviceDescriptor, deviceLostCallbackInfo));
      printf(" uncapturedErrorCallbackInfo@%zu", offsetof(WGPUDeviceDescriptor, uncapturedErrorCallbackInfo));
      printf("\n"); }
    { WGPUAdapterInfo v; printf("WGPUAdapterInfo %zu", sizeof(v));
      printf(" device@%zu", offsetof(WGPUAdapterInfo, device));
      printf(" description@%zu", offsetof(WGPUAdapterInfo, description));
      printf(" backendType@%zu", offsetof(WGPUAdapterInfo, backendType));
      printf(" adapterType@%zu", offsetof(WGPUAdapterInfo, adapterType));
      printf(" subgroupMaxSize@%zu", offsetof(WGPUAdapterInfo, subgroupMaxSize));
      printf("\n"); }
    { WGPUBufferDescriptor v; printf("WGPUBufferDescriptor %zu", sizeof(v));
      printf(" label@%zu", offsetof(WGPUBufferDescriptor, label));
      printf(" usage@%zu", offsetof(WGPUBufferDescriptor, usage));
      printf(" size@%zu", offsetof(WGPUBufferDescriptor, size));
      printf(" mappedAtCreation@%zu", offsetof(WGPUBufferDescriptor, mappedAtCreation));
      printf("\n"); }
    { WGPUShaderSourceWGSL v; printf("WGPUShaderSourceWGSL %zu", sizeof(v));
      printf(" chain@%zu", offsetof(WGPUShaderSourceWGSL, chain));
      printf(" code@%zu", offsetof(WGPUShaderSourceWGSL, code));
      printf("\n"); }
    { WGPUShaderModuleDescriptor v; printf("WGPUShaderModuleDescriptor %zu", sizeof(v));
      printf(" label@%zu", offsetof(WGPUShaderModuleDescriptor, label));
      printf("\n"); }
    { WGPUBufferBindingLayout v; printf("WGPUBufferBindingLayout %zu", sizeof(v));
      printf(" type@%zu", offsetof(WGPUBufferBindingLayout, type));
      printf(" hasDynamicOffset@%zu", offsetof(WGPUBufferBindingLayout, hasDynamicOffset));
      printf(" minBindingSize@%zu", offsetof(WGPUBufferBindingLayout, minBindingSize));
      printf("\n"); }
    { WGPUBindGroupLayoutEntry v; printf("WGPUBindGroupLayoutEntry %zu", sizeof(v));
      printf(" binding@%zu", offsetof(WGPUBindGroupLayoutEntry, binding));
      printf(" visibility@%zu", offsetof(WGPUBindGroupLayoutEntry, visibility));
      printf(" bindingArraySize@%zu", offsetof(WGPUBindGroupLayoutEntry, bindingArraySize));
      printf(" buffer@%zu", offsetof(WGPUBindGroupLayoutEntry, buffer));
      printf("\n"); }
    { WGPUBindGroupLayoutDescriptor v; printf("WGPUBindGroupLayoutDescriptor %zu", sizeof(v));
      printf(" label@%zu", offsetof(WGPUBindGroupLayoutDescriptor, label));
      printf(" entryCount@%zu", offsetof(WGPUBindGroupLayoutDescriptor, entryCount));
      printf(" entries@%zu", offsetof(WGPUBindGroupLayoutDescriptor, entries));
      printf("\n"); }
    { WGPUPipelineLayoutDescriptor v; printf("WGPUPipelineLayoutDescriptor %zu", sizeof(v));
      printf(" label@%zu", offsetof(WGPUPipelineLayoutDescriptor, label));
      printf(" bindGroupLayoutCount@%zu", offsetof(WGPUPipelineLayoutDescriptor, bindGroupLayoutCount));
      printf(" bindGroupLayouts@%zu", offsetof(WGPUPipelineLayoutDescriptor, bindGroupLayouts));
      printf(" immediateSize@%zu", offsetof(WGPUPipelineLayoutDescriptor, immediateSize));
      printf("\n"); }
    { WGPUBindGroupEntry v; printf("WGPUBindGroupEntry %zu", sizeof(v));
      printf(" binding@%zu", offsetof(WGPUBindGroupEntry, binding));
      printf(" buffer@%zu", offsetof(WGPUBindGroupEntry, buffer));
      printf(" offset@%zu", offsetof(WGPUBindGroupEntry, offset));
      printf(" size@%zu", offsetof(WGPUBindGroupEntry, size));
      printf(" sampler@%zu", offsetof(WGPUBindGroupEntry, sampler));
      printf(" textureView@%zu", offsetof(WGPUBindGroupEntry, textureView));
      printf("\n"); }
    { WGPUBindGroupDescriptor v; printf("WGPUBindGroupDescriptor %zu", sizeof(v));
      printf(" label@%zu", offsetof(WGPUBindGroupDescriptor, label));
      printf(" layout@%zu", offsetof(WGPUBindGroupDescriptor, layout));
      printf(" entryCount@%zu", offsetof(WGPUBindGroupDescriptor, entryCount));
      printf(" entries@%zu", offsetof(WGPUBindGroupDescriptor, entries));
      printf("\n"); }
    { WGPUConstantEntry v; printf("WGPUConstantEntry %zu", sizeof(v));
      printf(" key@%zu", offsetof(WGPUConstantEntry, key));
      printf(" value@%zu", offsetof(WGPUConstantEntry, value));
      printf("\n"); }
    { WGPUComputeState v; printf("WGPUComputeState %zu", sizeof(v));
      printf(" module@%zu", offsetof(WGPUComputeState, module));
      printf(" entryPoint@%zu", offsetof(WGPUComputeState, entryPoint));
      printf(" constantCount@%zu", offsetof(WGPUComputeState, constantCount));
      printf(" constants@%zu", offsetof(WGPUComputeState, constants));
      printf("\n"); }
    { WGPUComputePipelineDescriptor v; printf("WGPUComputePipelineDescriptor %zu", sizeof(v));
      printf(" label@%zu", offsetof(WGPUComputePipelineDescriptor, label));
      printf(" layout@%zu", offsetof(WGPUComputePipelineDescriptor, layout));
      printf(" compute@%zu", offsetof(WGPUComputePipelineDescriptor, compute));
      printf("\n"); }
    return 0;
}
