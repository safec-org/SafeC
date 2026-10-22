// std/ml/webgpu_api.h's struct layouts, printed as layout_check.c prints the
// real webgpu.h's; run.sh diffs the two.
#include <std/ml/webgpu_api.h>

extern int printf(const char* fmt, ...);

int main() {
    unsafe {
        { struct WGPUStringView v; unsigned long b = (unsigned long)&v; printf("WGPUStringView %lu", (unsigned long)sizeof(struct WGPUStringView));
          printf(" data@%lu", (unsigned long)&v.data - b);
          printf(" length@%lu", (unsigned long)&v.length - b);
          printf("\n"); }
        { struct WGPUChainedStruct v; unsigned long b = (unsigned long)&v; printf("WGPUChainedStruct %lu", (unsigned long)sizeof(struct WGPUChainedStruct));
          printf(" next@%lu", (unsigned long)&v.next - b);
          printf(" sType@%lu", (unsigned long)&v.sType - b);
          printf("\n"); }
        { struct WGPURequestAdapterCallbackInfo v; unsigned long b = (unsigned long)&v; printf("WGPURequestAdapterCallbackInfo %lu", (unsigned long)sizeof(struct WGPURequestAdapterCallbackInfo));
          printf(" mode@%lu", (unsigned long)&v.mode - b);
          printf(" callback@%lu", (unsigned long)&v.callback - b);
          printf(" userdata1@%lu", (unsigned long)&v.userdata1 - b);
          printf(" userdata2@%lu", (unsigned long)&v.userdata2 - b);
          printf("\n"); }
        { struct WGPURequestDeviceCallbackInfo v; unsigned long b = (unsigned long)&v; printf("WGPURequestDeviceCallbackInfo %lu", (unsigned long)sizeof(struct WGPURequestDeviceCallbackInfo));
          printf(" mode@%lu", (unsigned long)&v.mode - b);
          printf(" callback@%lu", (unsigned long)&v.callback - b);
          printf(" userdata2@%lu", (unsigned long)&v.userdata2 - b);
          printf("\n"); }
        { struct WGPUBufferMapCallbackInfo v; unsigned long b = (unsigned long)&v; printf("WGPUBufferMapCallbackInfo %lu", (unsigned long)sizeof(struct WGPUBufferMapCallbackInfo));
          printf(" mode@%lu", (unsigned long)&v.mode - b);
          printf(" callback@%lu", (unsigned long)&v.callback - b);
          printf(" userdata2@%lu", (unsigned long)&v.userdata2 - b);
          printf("\n"); }
        { struct WGPUQueueWorkDoneCallbackInfo v; unsigned long b = (unsigned long)&v; printf("WGPUQueueWorkDoneCallbackInfo %lu", (unsigned long)sizeof(struct WGPUQueueWorkDoneCallbackInfo));
          printf(" mode@%lu", (unsigned long)&v.mode - b);
          printf(" callback@%lu", (unsigned long)&v.callback - b);
          printf(" userdata2@%lu", (unsigned long)&v.userdata2 - b);
          printf("\n"); }
        { struct WGPUDeviceLostCallbackInfo v; unsigned long b = (unsigned long)&v; printf("WGPUDeviceLostCallbackInfo %lu", (unsigned long)sizeof(struct WGPUDeviceLostCallbackInfo));
          printf(" mode@%lu", (unsigned long)&v.mode - b);
          printf(" callback@%lu", (unsigned long)&v.callback - b);
          printf(" userdata2@%lu", (unsigned long)&v.userdata2 - b);
          printf("\n"); }
        { struct WGPUUncapturedErrorCallbackInfo v; unsigned long b = (unsigned long)&v; printf("WGPUUncapturedErrorCallbackInfo %lu", (unsigned long)sizeof(struct WGPUUncapturedErrorCallbackInfo));
          printf(" callback@%lu", (unsigned long)&v.callback - b);
          printf(" userdata1@%lu", (unsigned long)&v.userdata1 - b);
          printf(" userdata2@%lu", (unsigned long)&v.userdata2 - b);
          printf("\n"); }
        { struct WGPUInstanceDescriptor v; unsigned long b = (unsigned long)&v; printf("WGPUInstanceDescriptor %lu", (unsigned long)sizeof(struct WGPUInstanceDescriptor));
          printf(" requiredFeatureCount@%lu", (unsigned long)&v.requiredFeatureCount - b);
          printf(" requiredFeatures@%lu", (unsigned long)&v.requiredFeatures - b);
          printf(" requiredLimits@%lu", (unsigned long)&v.requiredLimits - b);
          printf("\n"); }
        { struct WGPURequestAdapterOptions v; unsigned long b = (unsigned long)&v; printf("WGPURequestAdapterOptions %lu", (unsigned long)sizeof(struct WGPURequestAdapterOptions));
          printf(" featureLevel@%lu", (unsigned long)&v.featureLevel - b);
          printf(" powerPreference@%lu", (unsigned long)&v.powerPreference - b);
          printf(" forceFallbackAdapter@%lu", (unsigned long)&v.forceFallbackAdapter - b);
          printf(" backendType@%lu", (unsigned long)&v.backendType - b);
          printf(" compatibleSurface@%lu", (unsigned long)&v.compatibleSurface - b);
          printf("\n"); }
        { struct WGPULimits v; unsigned long b = (unsigned long)&v; printf("WGPULimits %lu", (unsigned long)sizeof(struct WGPULimits));
          printf(" maxStorageBuffersPerShaderStage@%lu", (unsigned long)&v.maxStorageBuffersPerShaderStage - b);
          printf(" maxUniformBufferBindingSize@%lu", (unsigned long)&v.maxUniformBufferBindingSize - b);
          printf(" maxStorageBufferBindingSize@%lu", (unsigned long)&v.maxStorageBufferBindingSize - b);
          printf(" minUniformBufferOffsetAlignment@%lu", (unsigned long)&v.minUniformBufferOffsetAlignment - b);
          printf(" maxVertexBuffers@%lu", (unsigned long)&v.maxVertexBuffers - b);
          printf(" maxBufferSize@%lu", (unsigned long)&v.maxBufferSize - b);
          printf(" maxComputeWorkgroupStorageSize@%lu", (unsigned long)&v.maxComputeWorkgroupStorageSize - b);
          printf(" maxComputeWorkgroupsPerDimension@%lu", (unsigned long)&v.maxComputeWorkgroupsPerDimension - b);
          printf(" maxImmediateSize@%lu", (unsigned long)&v.maxImmediateSize - b);
          printf("\n"); }
        { struct WGPUDeviceDescriptor v; unsigned long b = (unsigned long)&v; printf("WGPUDeviceDescriptor %lu", (unsigned long)sizeof(struct WGPUDeviceDescriptor));
          printf(" label@%lu", (unsigned long)&v.label - b);
          printf(" requiredFeatureCount@%lu", (unsigned long)&v.requiredFeatureCount - b);
          printf(" requiredFeatures@%lu", (unsigned long)&v.requiredFeatures - b);
          printf(" requiredLimits@%lu", (unsigned long)&v.requiredLimits - b);
          printf(" defaultQueue@%lu", (unsigned long)&v.defaultQueue - b);
          printf(" deviceLostCallbackInfo@%lu", (unsigned long)&v.deviceLostCallbackInfo - b);
          printf(" uncapturedErrorCallbackInfo@%lu", (unsigned long)&v.uncapturedErrorCallbackInfo - b);
          printf("\n"); }
        { struct WGPUAdapterInfo v; unsigned long b = (unsigned long)&v; printf("WGPUAdapterInfo %lu", (unsigned long)sizeof(struct WGPUAdapterInfo));
          printf(" device@%lu", (unsigned long)&v.device - b);
          printf(" description@%lu", (unsigned long)&v.description - b);
          printf(" backendType@%lu", (unsigned long)&v.backendType - b);
          printf(" adapterType@%lu", (unsigned long)&v.adapterType - b);
          printf(" subgroupMaxSize@%lu", (unsigned long)&v.subgroupMaxSize - b);
          printf("\n"); }
        { struct WGPUBufferDescriptor v; unsigned long b = (unsigned long)&v; printf("WGPUBufferDescriptor %lu", (unsigned long)sizeof(struct WGPUBufferDescriptor));
          printf(" label@%lu", (unsigned long)&v.label - b);
          printf(" usage@%lu", (unsigned long)&v.usage - b);
          printf(" size@%lu", (unsigned long)&v.size - b);
          printf(" mappedAtCreation@%lu", (unsigned long)&v.mappedAtCreation - b);
          printf("\n"); }
        { struct WGPUShaderSourceWGSL v; unsigned long b = (unsigned long)&v; printf("WGPUShaderSourceWGSL %lu", (unsigned long)sizeof(struct WGPUShaderSourceWGSL));
          printf(" chain@%lu", (unsigned long)&v.chain - b);
          printf(" code@%lu", (unsigned long)&v.code - b);
          printf("\n"); }
        { struct WGPUShaderModuleDescriptor v; unsigned long b = (unsigned long)&v; printf("WGPUShaderModuleDescriptor %lu", (unsigned long)sizeof(struct WGPUShaderModuleDescriptor));
          printf(" label@%lu", (unsigned long)&v.label - b);
          printf("\n"); }
        { struct WGPUBufferBindingLayout v; unsigned long b = (unsigned long)&v; printf("WGPUBufferBindingLayout %lu", (unsigned long)sizeof(struct WGPUBufferBindingLayout));
          printf(" type@%lu", (unsigned long)&v.type - b);
          printf(" hasDynamicOffset@%lu", (unsigned long)&v.hasDynamicOffset - b);
          printf(" minBindingSize@%lu", (unsigned long)&v.minBindingSize - b);
          printf("\n"); }
        { struct WGPUBindGroupLayoutEntry v; unsigned long b = (unsigned long)&v; printf("WGPUBindGroupLayoutEntry %lu", (unsigned long)sizeof(struct WGPUBindGroupLayoutEntry));
          printf(" binding@%lu", (unsigned long)&v.binding - b);
          printf(" visibility@%lu", (unsigned long)&v.visibility - b);
          printf(" bindingArraySize@%lu", (unsigned long)&v.bindingArraySize - b);
          printf(" buffer@%lu", (unsigned long)&v.buffer - b);
          printf("\n"); }
        { struct WGPUBindGroupLayoutDescriptor v; unsigned long b = (unsigned long)&v; printf("WGPUBindGroupLayoutDescriptor %lu", (unsigned long)sizeof(struct WGPUBindGroupLayoutDescriptor));
          printf(" label@%lu", (unsigned long)&v.label - b);
          printf(" entryCount@%lu", (unsigned long)&v.entryCount - b);
          printf(" entries@%lu", (unsigned long)&v.entries - b);
          printf("\n"); }
        { struct WGPUPipelineLayoutDescriptor v; unsigned long b = (unsigned long)&v; printf("WGPUPipelineLayoutDescriptor %lu", (unsigned long)sizeof(struct WGPUPipelineLayoutDescriptor));
          printf(" label@%lu", (unsigned long)&v.label - b);
          printf(" bindGroupLayoutCount@%lu", (unsigned long)&v.bindGroupLayoutCount - b);
          printf(" bindGroupLayouts@%lu", (unsigned long)&v.bindGroupLayouts - b);
          printf(" immediateSize@%lu", (unsigned long)&v.immediateSize - b);
          printf("\n"); }
        { struct WGPUBindGroupEntry v; unsigned long b = (unsigned long)&v; printf("WGPUBindGroupEntry %lu", (unsigned long)sizeof(struct WGPUBindGroupEntry));
          printf(" binding@%lu", (unsigned long)&v.binding - b);
          printf(" buffer@%lu", (unsigned long)&v.buffer - b);
          printf(" offset@%lu", (unsigned long)&v.offset - b);
          printf(" size@%lu", (unsigned long)&v.size - b);
          printf(" sampler@%lu", (unsigned long)&v.sampler - b);
          printf(" textureView@%lu", (unsigned long)&v.textureView - b);
          printf("\n"); }
        { struct WGPUBindGroupDescriptor v; unsigned long b = (unsigned long)&v; printf("WGPUBindGroupDescriptor %lu", (unsigned long)sizeof(struct WGPUBindGroupDescriptor));
          printf(" label@%lu", (unsigned long)&v.label - b);
          printf(" layout@%lu", (unsigned long)&v.layout - b);
          printf(" entryCount@%lu", (unsigned long)&v.entryCount - b);
          printf(" entries@%lu", (unsigned long)&v.entries - b);
          printf("\n"); }
        { struct WGPUConstantEntry v; unsigned long b = (unsigned long)&v; printf("WGPUConstantEntry %lu", (unsigned long)sizeof(struct WGPUConstantEntry));
          printf(" key@%lu", (unsigned long)&v.key - b);
          printf(" value@%lu", (unsigned long)&v.value - b);
          printf("\n"); }
        { struct WGPUComputeState v; unsigned long b = (unsigned long)&v; printf("WGPUComputeState %lu", (unsigned long)sizeof(struct WGPUComputeState));
          printf(" module@%lu", (unsigned long)&v.module - b);
          printf(" entryPoint@%lu", (unsigned long)&v.entryPoint - b);
          printf(" constantCount@%lu", (unsigned long)&v.constantCount - b);
          printf(" constants@%lu", (unsigned long)&v.constants - b);
          printf("\n"); }
        { struct WGPUComputePipelineDescriptor v; unsigned long b = (unsigned long)&v; printf("WGPUComputePipelineDescriptor %lu", (unsigned long)sizeof(struct WGPUComputePipelineDescriptor));
          printf(" label@%lu", (unsigned long)&v.label - b);
          printf(" layout@%lu", (unsigned long)&v.layout - b);
          printf(" compute@%lu", (unsigned long)&v.compute - b);
          printf("\n"); }
    }
    return 0;
}
