// SafeC Standard Library — Vulkan/SPIR-V GPU backend implementation (see
// gpu_spirv.h). UNVERIFIED, more so than gpu_cuda.sc/gpu_rocm.sc: no
// Vulkan SDK, driver, or SPIR-V toolchain in this sandbox to check any of
// this against (confirmed: no glslc/glslangValidator/spirv-as, no
// vulkan/ headers under the usual Homebrew include paths). Struct
// layouts and sType enum values below are written from memory against
// the Vulkan 1.0 core spec's structure ordering, not copy-checked
// against a real vulkan.h.
//
// Same shape as gpu_rocm.sc's HSACO gap: every op's SPIR-V module bytes
// ('spirvCode') are an explicit null placeholder, so every function
// returns 0 immediately, before any of the real (but unverified) Vulkan
// calls below that point would run. A real deployment replaces each
// null with bytes from an offline 'glslc kernel.comp -o kernel.spv' (or
// equivalent HLSL/WGSL-to-SPIR-V) build step.
#pragma once
#include <std/ml/gpu_spirv.h>
#include <std/mem.h>

namespace std {

extern void* memcpy(void* dst, const void* src, unsigned long n);

// ── Vulkan core structs (hand-matched to vulkan_core.h's field order) ──────
struct VkApplicationInfo {
    int sType; const void* pNext;
    const char* pApplicationName; unsigned int applicationVersion;
    const char* pEngineName; unsigned int engineVersion;
    unsigned int apiVersion;
};
struct VkInstanceCreateInfo {
    int sType; const void* pNext; unsigned int flags;
    const struct VkApplicationInfo* pApplicationInfo;
    unsigned int enabledLayerCount; const void* ppEnabledLayerNames;
    unsigned int enabledExtensionCount; const void* ppEnabledExtensionNames;
};
struct VkDeviceQueueCreateInfo {
    int sType; const void* pNext; unsigned int flags;
    unsigned int queueFamilyIndex; unsigned int queueCount;
    const float* pQueuePriorities;
};
struct VkDeviceCreateInfo {
    int sType; const void* pNext; unsigned int flags;
    unsigned int queueCreateInfoCount;
    const struct VkDeviceQueueCreateInfo* pQueueCreateInfos;
    unsigned int enabledLayerCount; const void* ppEnabledLayerNames;
    unsigned int enabledExtensionCount; const void* ppEnabledExtensionNames;
    const void* pEnabledFeatures;
};
struct VkShaderModuleCreateInfo {
    int sType; const void* pNext; unsigned int flags;
    unsigned long codeSize; const unsigned int* pCode;
};
struct VkDescriptorSetLayoutBinding {
    unsigned int binding; int descriptorType; unsigned int descriptorCount;
    unsigned int stageFlags; const void* pImmutableSamplers;
};
struct VkDescriptorSetLayoutCreateInfo {
    int sType; const void* pNext; unsigned int flags;
    unsigned int bindingCount; const struct VkDescriptorSetLayoutBinding* pBindings;
};
struct VkPushConstantRange { unsigned int stageFlags; unsigned int offset; unsigned int size; };
struct VkPipelineLayoutCreateInfo {
    int sType; const void* pNext; unsigned int flags;
    unsigned int setLayoutCount; const void* pSetLayouts;
    unsigned int pushConstantRangeCount; const struct VkPushConstantRange* pPushConstantRanges;
};
struct VkPipelineShaderStageCreateInfo {
    int sType; const void* pNext; unsigned int flags; unsigned int stage;
    void* module; const char* pName; const void* pSpecializationInfo;
};
struct VkComputePipelineCreateInfo {
    int sType; const void* pNext; unsigned int flags;
    struct VkPipelineShaderStageCreateInfo stage;
    void* layout; void* basePipelineHandle; int basePipelineIndex;
};
struct VkBufferCreateInfo {
    int sType; const void* pNext; unsigned int flags; unsigned long size;
    unsigned int usage; int sharingMode;
    unsigned int queueFamilyIndexCount; const unsigned int* pQueueFamilyIndices;
};
struct VkMemoryRequirements {
    unsigned long size; unsigned long alignment; unsigned int memoryTypeBits;
};
struct VkMemoryAllocateInfo {
    int sType; const void* pNext; unsigned long allocationSize; unsigned int memoryTypeIndex;
};
struct VkDescriptorPoolSize { int type; unsigned int descriptorCount; };
struct VkDescriptorPoolCreateInfo {
    int sType; const void* pNext; unsigned int flags; unsigned int maxSets;
    unsigned int poolSizeCount; const struct VkDescriptorPoolSize* pPoolSizes;
};
struct VkDescriptorSetAllocateInfo {
    int sType; const void* pNext; void* descriptorPool;
    unsigned int descriptorSetCount; const void* pSetLayouts;
};
struct VkDescriptorBufferInfo { void* buffer; unsigned long offset; unsigned long range; };
struct VkWriteDescriptorSet {
    int sType; const void* pNext; void* dstSet; unsigned int dstBinding;
    unsigned int dstArrayElement; unsigned int descriptorCount; int descriptorType;
    const void* pImageInfo; const struct VkDescriptorBufferInfo* pBufferInfo;
    const void* pTexelBufferView;
};
struct VkCommandPoolCreateInfo {
    int sType; const void* pNext; unsigned int flags; unsigned int queueFamilyIndex;
};
struct VkCommandBufferAllocateInfo {
    int sType; const void* pNext; void* commandPool; int level;
    unsigned int commandBufferCount;
};
struct VkCommandBufferBeginInfo { int sType; const void* pNext; unsigned int flags; const void* pInheritanceInfo; };
struct VkSubmitInfo {
    int sType; const void* pNext; unsigned int waitSemaphoreCount;
    const void* pWaitSemaphores; const unsigned int* pWaitDstStageMask;
    unsigned int commandBufferCount; const void* pCommandBuffers;
    unsigned int signalSemaphoreCount; const void* pSignalSemaphores;
};

// ── Vulkan core functions (hand-matched to vulkan_core.h) ───────────────────
extern int vkCreateInstance(const struct VkInstanceCreateInfo* pCreateInfo, const void* pAllocator, void** pInstance);
extern int vkEnumeratePhysicalDevices(void* instance, unsigned int* pPhysicalDeviceCount, void** pPhysicalDevices);
extern int vkCreateDevice(void* physicalDevice, const struct VkDeviceCreateInfo* pCreateInfo, const void* pAllocator, void** pDevice);
extern void vkGetDeviceQueue(void* device, unsigned int queueFamilyIndex, unsigned int queueIndex, void** pQueue);
extern int vkCreateShaderModule(void* device, const struct VkShaderModuleCreateInfo* pCreateInfo, const void* pAllocator, void** pShaderModule);
extern int vkCreateDescriptorSetLayout(void* device, const struct VkDescriptorSetLayoutCreateInfo* pCreateInfo, const void* pAllocator, void** pSetLayout);
extern int vkCreatePipelineLayout(void* device, const struct VkPipelineLayoutCreateInfo* pCreateInfo, const void* pAllocator, void** pPipelineLayout);
extern int vkCreateComputePipelines(void* device, void* pipelineCache, unsigned int createInfoCount, const struct VkComputePipelineCreateInfo* pCreateInfos, const void* pAllocator, void** pPipelines);
extern int vkCreateBuffer(void* device, const struct VkBufferCreateInfo* pCreateInfo, const void* pAllocator, void** pBuffer);
extern void vkGetBufferMemoryRequirements(void* device, void* buffer, struct VkMemoryRequirements* pMemoryRequirements);
extern int vkAllocateMemory(void* device, const struct VkMemoryAllocateInfo* pAllocateInfo, const void* pAllocator, void** pMemory);
extern int vkBindBufferMemory(void* device, void* buffer, void* memory, unsigned long memoryOffset);
extern int vkMapMemory(void* device, void* memory, unsigned long offset, unsigned long size, unsigned int flags, void** ppData);
extern void vkUnmapMemory(void* device, void* memory);
extern int vkCreateDescriptorPool(void* device, const struct VkDescriptorPoolCreateInfo* pCreateInfo, const void* pAllocator, void** pDescriptorPool);
extern int vkAllocateDescriptorSets(void* device, const struct VkDescriptorSetAllocateInfo* pAllocateInfo, void** pDescriptorSets);
extern void vkUpdateDescriptorSets(void* device, unsigned int descriptorWriteCount, const struct VkWriteDescriptorSet* pDescriptorWrites, unsigned int descriptorCopyCount, const void* pDescriptorCopies);
extern int vkCreateCommandPool(void* device, const struct VkCommandPoolCreateInfo* pCreateInfo, const void* pAllocator, void** pCommandPool);
extern int vkAllocateCommandBuffers(void* device, const struct VkCommandBufferAllocateInfo* pAllocateInfo, void** pCommandBuffers);
extern int vkBeginCommandBuffer(void* commandBuffer, const struct VkCommandBufferBeginInfo* pBeginInfo);
extern void vkCmdBindPipeline(void* commandBuffer, int pipelineBindPoint, void* pipeline);
extern void vkCmdBindDescriptorSets(void* commandBuffer, int pipelineBindPoint, void* layout, unsigned int firstSet, unsigned int descriptorSetCount, const void* pDescriptorSets, unsigned int dynamicOffsetCount, const unsigned int* pDynamicOffsets);
extern void vkCmdPushConstants(void* commandBuffer, void* layout, unsigned int stageFlags, unsigned int offset, unsigned int size, const void* pValues);
extern void vkCmdDispatch(void* commandBuffer, unsigned int groupCountX, unsigned int groupCountY, unsigned int groupCountZ);
extern int vkEndCommandBuffer(void* commandBuffer);
extern int vkQueueSubmit(void* queue, unsigned int submitCount, const struct VkSubmitInfo* pSubmits, void* fence);
extern int vkQueueWaitIdle(void* queue);
extern void vkDestroyBuffer(void* device, void* buffer, const void* pAllocator);
extern void vkFreeMemory(void* device, void* memory, const void* pAllocator);
extern void vkDestroyDevice(void* device, const void* pAllocator);
extern void vkDestroyInstance(void* instance, const void* pAllocator);

#define VK_SUCCESS 0
#define VK_STRUCTURE_TYPE_APPLICATION_INFO 0
#define VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO 1
#define VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO 2
#define VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO 3
#define VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO 16
#define VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO 32
#define VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO 30
#define VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO 18
#define VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO 29
#define VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO 12
#define VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO 5
#define VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO 33
#define VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO 34
#define VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET 35
#define VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO 39
#define VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO 40
#define VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO 42
#define VK_STRUCTURE_TYPE_SUBMIT_INFO 4
#define VK_SHADER_STAGE_COMPUTE_BIT 32
#define VK_DESCRIPTOR_TYPE_STORAGE_BUFFER 7
#define VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 32
#define VK_BUFFER_USAGE_TRANSFER_SRC_BIT 1
#define VK_BUFFER_USAGE_TRANSFER_DST_BIT 2
#define VK_SHARING_MODE_EXCLUSIVE 0
#define VK_PIPELINE_BIND_POINT_COMPUTE 1
#define VK_COMMAND_BUFFER_LEVEL_PRIMARY 0
#define VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT 2
#define VK_MEMORY_PROPERTY_HOST_COHERENT_BIT 4

int spirv_available() {
    unsafe {
        struct VkApplicationInfo appInfo;
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; appInfo.pNext = (const void*)0;
        appInfo.pApplicationName = "safec-tensor-gpu"; appInfo.applicationVersion = 1U;
        appInfo.pEngineName = "std::ml"; appInfo.engineVersion = 1U;
        appInfo.apiVersion = 4194304U; // VK_API_VERSION_1_0 = (1<<22)

        struct VkInstanceCreateInfo ci;
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ci.pNext = (const void*)0; ci.flags = 0U;
        ci.pApplicationInfo = (const struct VkApplicationInfo*)&appInfo;
        ci.enabledLayerCount = 0U; ci.ppEnabledLayerNames = (const void*)0;
        ci.enabledExtensionCount = 0U; ci.ppEnabledExtensionNames = (const void*)0;

        void* instance = (void*)0;
        if (vkCreateInstance(&ci, (const void*)0, &instance) != VK_SUCCESS) return 0;

        unsigned int deviceCount = 0U;
        int r = vkEnumeratePhysicalDevices(instance, &deviceCount, (void**)0);
        vkDestroyInstance(instance, (const void*)0);
        if (r != VK_SUCCESS) return 0;
        return deviceCount > 0U;
    }
}

// Shared instance/device/shader-module/pipeline/buffer/dispatch/readback
// sequence for every op below. Bails out at the SPIR-V-bytecode check
// (always null here, per this file's header comment) before any of the
// real-but-unverified Vulkan calls below that point would run. Handles
// both the 1-input and 2-input cases via 'numInputs'; 'pushConstants' is
// how per-call scalars (n, k, or M/K/N) reach the shader, the idiomatic
// Vulkan way to pass small uniform-like values without a dedicated buffer.
static int __vk_run_kernel(const unsigned int* spirvCode, unsigned long spirvWords, const char* entryName,
                            unsigned int numInputs, void** inputs, unsigned long* inputSizes,
                            void* output, unsigned long outputSize,
                            const void* pushConstants, unsigned int pushConstantSize,
                            unsigned int groupCountX, unsigned int groupCountY, unsigned int groupCountZ) {
    if (spirvCode == (const unsigned int*)0) return 0;

    unsafe {
        struct VkApplicationInfo appInfo;
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; appInfo.pNext = (const void*)0;
        appInfo.pApplicationName = "safec-tensor-gpu"; appInfo.applicationVersion = 1U;
        appInfo.pEngineName = "std::ml"; appInfo.engineVersion = 1U;
        appInfo.apiVersion = 4194304U;

        struct VkInstanceCreateInfo ici;
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pNext = (const void*)0; ici.flags = 0U;
        ici.pApplicationInfo = (const struct VkApplicationInfo*)&appInfo;
        ici.enabledLayerCount = 0U; ici.ppEnabledLayerNames = (const void*)0;
        ici.enabledExtensionCount = 0U; ici.ppEnabledExtensionNames = (const void*)0;

        void* instance = (void*)0;
        if (vkCreateInstance(&ici, (const void*)0, &instance) != VK_SUCCESS) return 0;

        unsigned int deviceCount = 0U;
        vkEnumeratePhysicalDevices(instance, &deviceCount, (void**)0);
        if (deviceCount == 0U) { vkDestroyInstance(instance, (const void*)0); return 0; }
        void* physicalDevices[1];
        deviceCount = 1U;
        vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices);
        void* physicalDevice = physicalDevices[0];

        float prio = 1.0f;
        struct VkDeviceQueueCreateInfo qci;
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; qci.pNext = (const void*)0; qci.flags = 0U;
        qci.queueFamilyIndex = 0U; qci.queueCount = 1U; qci.pQueuePriorities = (const float*)&prio;

        struct VkDeviceCreateInfo dci;
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO; dci.pNext = (const void*)0; dci.flags = 0U;
        dci.queueCreateInfoCount = 1U; dci.pQueueCreateInfos = (const struct VkDeviceQueueCreateInfo*)&qci;
        dci.enabledLayerCount = 0U; dci.ppEnabledLayerNames = (const void*)0;
        dci.enabledExtensionCount = 0U; dci.ppEnabledExtensionNames = (const void*)0;
        dci.pEnabledFeatures = (const void*)0;

        void* device = (void*)0;
        if (vkCreateDevice(physicalDevice, &dci, (const void*)0, &device) != VK_SUCCESS) {
            vkDestroyInstance(instance, (const void*)0);
            return 0;
        }
        void* queue = (void*)0;
        vkGetDeviceQueue(device, 0U, 0U, &queue);

        struct VkShaderModuleCreateInfo smci;
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO; smci.pNext = (const void*)0; smci.flags = 0U;
        smci.codeSize = spirvWords * 4UL; smci.pCode = spirvCode;
        void* shaderModule = (void*)0;
        if (vkCreateShaderModule(device, &smci, (const void*)0, &shaderModule) != VK_SUCCESS) {
            vkDestroyDevice(device, (const void*)0);
            vkDestroyInstance(instance, (const void*)0);
            return 0;
        }

        // Descriptor set layout: one binding per input buffer + one for
        // output, all VK_DESCRIPTOR_TYPE_STORAGE_BUFFER.
        unsigned int numBindings = numInputs + 1U;
        struct VkDescriptorSetLayoutBinding bindings[4]; // supports up to 3 inputs + 1 output
        unsigned int bi = 0U;
        while (bi < numBindings) {
            bindings[bi].binding = bi;
            bindings[bi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[bi].descriptorCount = 1U;
            bindings[bi].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[bi].pImmutableSamplers = (const void*)0;
            bi = bi + 1U;
        }
        struct VkDescriptorSetLayoutCreateInfo dslci;
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO; dslci.pNext = (const void*)0; dslci.flags = 0U;
        dslci.bindingCount = numBindings; dslci.pBindings = (const struct VkDescriptorSetLayoutBinding*)bindings;
        void* setLayout = (void*)0;
        vkCreateDescriptorSetLayout(device, &dslci, (const void*)0, &setLayout);

        struct VkPushConstantRange pcRange;
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pcRange.offset = 0U; pcRange.size = pushConstantSize;

        struct VkPipelineLayoutCreateInfo plci;
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO; plci.pNext = (const void*)0; plci.flags = 0U;
        plci.setLayoutCount = 1U; plci.pSetLayouts = (const void*)&setLayout;
        plci.pushConstantRangeCount = pushConstantSize > 0U ? 1U : 0U;
        plci.pPushConstantRanges = pushConstantSize > 0U ? (const struct VkPushConstantRange*)&pcRange : (const struct VkPushConstantRange*)0;
        void* pipelineLayout = (void*)0;
        vkCreatePipelineLayout(device, &plci, (const void*)0, &pipelineLayout);

        struct VkComputePipelineCreateInfo cpci;
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO; cpci.pNext = (const void*)0; cpci.flags = 0U;
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cpci.stage.pNext = (const void*)0;
        cpci.stage.flags = 0U; cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = shaderModule; cpci.stage.pName = entryName; cpci.stage.pSpecializationInfo = (const void*)0;
        cpci.layout = pipelineLayout; cpci.basePipelineHandle = (void*)0; cpci.basePipelineIndex = -1;
        void* pipeline = (void*)0;
        if (vkCreateComputePipelines(device, (void*)0, 1U, &cpci, (const void*)0, &pipeline) != VK_SUCCESS) {
            vkDestroyDevice(device, (const void*)0);
            vkDestroyInstance(instance, (const void*)0);
            return 0;
        }

        // Buffers: one per input (host-visible+coherent, so no explicit
        // flush/staging-buffer copy is needed for this small-scale case),
        // one for output.
        void* buffers[4];
        void* memories[4];
        unsigned int idx = 0U;
        while (idx < numInputs) {
            struct VkBufferCreateInfo bci;
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bci.pNext = (const void*)0; bci.flags = 0U;
            bci.size = inputSizes[idx];
            bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE; bci.queueFamilyIndexCount = 0U; bci.pQueueFamilyIndices = (const unsigned int*)0;
            vkCreateBuffer(device, &bci, (const void*)0, &buffers[idx]);
            struct VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(device, buffers[idx], &req);
            struct VkMemoryAllocateInfo mai;
            mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; mai.pNext = (const void*)0;
            mai.allocationSize = req.size; mai.memoryTypeIndex = 0U; // real code: pick a host-visible|coherent type from VkPhysicalDeviceMemoryProperties
            vkAllocateMemory(device, &mai, (const void*)0, &memories[idx]);
            vkBindBufferMemory(device, buffers[idx], memories[idx], 0UL);
            void* mapped = (void*)0;
            vkMapMemory(device, memories[idx], 0UL, req.size, 0U, &mapped);
            memcpy(mapped, inputs[idx], inputSizes[idx]);
            vkUnmapMemory(device, memories[idx]);
            idx = idx + 1U;
        }
        struct VkBufferCreateInfo obci;
        obci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; obci.pNext = (const void*)0; obci.flags = 0U;
        obci.size = outputSize;
        obci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        obci.sharingMode = VK_SHARING_MODE_EXCLUSIVE; obci.queueFamilyIndexCount = 0U; obci.pQueueFamilyIndices = (const unsigned int*)0;
        vkCreateBuffer(device, &obci, (const void*)0, &buffers[numInputs]);
        struct VkMemoryRequirements oreq;
        vkGetBufferMemoryRequirements(device, buffers[numInputs], &oreq);
        struct VkMemoryAllocateInfo omai;
        omai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; omai.pNext = (const void*)0;
        omai.allocationSize = oreq.size; omai.memoryTypeIndex = 0U;
        vkAllocateMemory(device, &omai, (const void*)0, &memories[numInputs]);
        vkBindBufferMemory(device, buffers[numInputs], memories[numInputs], 0UL);

        struct VkDescriptorPoolSize poolSize;
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; poolSize.descriptorCount = numBindings;
        struct VkDescriptorPoolCreateInfo dpci;
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; dpci.pNext = (const void*)0; dpci.flags = 0U;
        dpci.maxSets = 1U; dpci.poolSizeCount = 1U; dpci.pPoolSizes = (const struct VkDescriptorPoolSize*)&poolSize;
        void* descPool = (void*)0;
        vkCreateDescriptorPool(device, &dpci, (const void*)0, &descPool);

        struct VkDescriptorSetAllocateInfo dsai;
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; dsai.pNext = (const void*)0;
        dsai.descriptorPool = descPool; dsai.descriptorSetCount = 1U; dsai.pSetLayouts = (const void*)&setLayout;
        void* descSet = (void*)0;
        vkAllocateDescriptorSets(device, &dsai, &descSet);

        struct VkDescriptorBufferInfo bufInfos[4];
        struct VkWriteDescriptorSet writes[4];
        unsigned int wi = 0U;
        while (wi < numBindings) {
            bufInfos[wi].buffer = buffers[wi]; bufInfos[wi].offset = 0UL;
            bufInfos[wi].range = wi < numInputs ? inputSizes[wi] : outputSize;
            writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[wi].pNext = (const void*)0;
            writes[wi].dstSet = descSet; writes[wi].dstBinding = wi; writes[wi].dstArrayElement = 0U;
            writes[wi].descriptorCount = 1U; writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[wi].pImageInfo = (const void*)0; writes[wi].pBufferInfo = (const struct VkDescriptorBufferInfo*)&bufInfos[wi]; writes[wi].pTexelBufferView = (const void*)0;
            wi = wi + 1U;
        }
        vkUpdateDescriptorSets(device, numBindings, (const struct VkWriteDescriptorSet*)writes, 0U, (const void*)0);

        struct VkCommandPoolCreateInfo cpci2;
        cpci2.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; cpci2.pNext = (const void*)0;
        cpci2.flags = 0U; cpci2.queueFamilyIndex = 0U;
        void* cmdPool = (void*)0;
        vkCreateCommandPool(device, &cpci2, (const void*)0, &cmdPool);

        struct VkCommandBufferAllocateInfo cbai;
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; cbai.pNext = (const void*)0;
        cbai.commandPool = cmdPool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1U;
        void* cmdBuf = (void*)0;
        vkAllocateCommandBuffers(device, &cbai, &cmdBuf);

        struct VkCommandBufferBeginInfo cbbi;
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; cbbi.pNext = (const void*)0;
        cbbi.flags = 0U; cbbi.pInheritanceInfo = (const void*)0;
        vkBeginCommandBuffer(cmdBuf, &cbbi);
        vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0U, 1U, (const void*)&descSet, 0U, (const unsigned int*)0);
        if (pushConstantSize > 0U) {
            vkCmdPushConstants(cmdBuf, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0U, pushConstantSize, pushConstants);
        }
        vkCmdDispatch(cmdBuf, groupCountX, groupCountY, groupCountZ);
        vkEndCommandBuffer(cmdBuf);

        struct VkSubmitInfo si;
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.pNext = (const void*)0;
        si.waitSemaphoreCount = 0U; si.pWaitSemaphores = (const void*)0; si.pWaitDstStageMask = (const unsigned int*)0;
        si.commandBufferCount = 1U; si.pCommandBuffers = (const void*)&cmdBuf;
        si.signalSemaphoreCount = 0U; si.pSignalSemaphores = (const void*)0;
        int submitOk = vkQueueSubmit(queue, 1U, &si, (void*)0) == VK_SUCCESS;
        if (submitOk) vkQueueWaitIdle(queue);

        if (submitOk) {
            void* outMapped = (void*)0;
            vkMapMemory(device, memories[numInputs], 0UL, outputSize, 0U, &outMapped);
            memcpy(output, outMapped, outputSize);
            vkUnmapMemory(device, memories[numInputs]);
        }

        unsigned int fi = 0U;
        while (fi < numBindings) {
            vkDestroyBuffer(device, buffers[fi], (const void*)0);
            vkFreeMemory(device, memories[fi], (const void*)0);
            fi = fi + 1U;
        }
        vkDestroyDevice(device, (const void*)0);
        vkDestroyInstance(instance, (const void*)0);
        return submitOk ? 1 : 0;
    }
}

int spirv_add_f32(const float* a, const float* b, float* out, unsigned long n) {
    const unsigned int* spirvCode = (const unsigned int*)0; // would be add.comp -> glslc -> add.spv's words
    unsafe {
        void* inputs[2]; inputs[0] = (void*)a; inputs[1] = (void*)b;
        unsigned long sizes[2]; sizes[0] = n * sizeof(float); sizes[1] = n * sizeof(float);
        unsigned int nParam = (unsigned int)n;
        return __vk_run_kernel(spirvCode, 0UL, "main", 2U, inputs, sizes, (void*)out, n * sizeof(float),
                                (const void*)&nParam, 4U, (n + 255UL) > 0xFFFFFFFFUL ? 0xFFFFFFFFU : (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

int spirv_sub_f32(const float* a, const float* b, float* out, unsigned long n) {
    const unsigned int* spirvCode = (const unsigned int*)0; // would be sub.comp's compiled SPIR-V
    unsafe {
        void* inputs[2]; inputs[0] = (void*)a; inputs[1] = (void*)b;
        unsigned long sizes[2]; sizes[0] = n * sizeof(float); sizes[1] = n * sizeof(float);
        unsigned int nParam = (unsigned int)n;
        return __vk_run_kernel(spirvCode, 0UL, "main", 2U, inputs, sizes, (void*)out, n * sizeof(float),
                                (const void*)&nParam, 4U, (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

int spirv_mul_f32(const float* a, const float* b, float* out, unsigned long n) {
    const unsigned int* spirvCode = (const unsigned int*)0; // would be mul.comp's compiled SPIR-V
    unsafe {
        void* inputs[2]; inputs[0] = (void*)a; inputs[1] = (void*)b;
        unsigned long sizes[2]; sizes[0] = n * sizeof(float); sizes[1] = n * sizeof(float);
        unsigned int nParam = (unsigned int)n;
        return __vk_run_kernel(spirvCode, 0UL, "main", 2U, inputs, sizes, (void*)out, n * sizeof(float),
                                (const void*)&nParam, 4U, (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

int spirv_div_f32(const float* a, const float* b, float* out, unsigned long n) {
    const unsigned int* spirvCode = (const unsigned int*)0; // would be div.comp's compiled SPIR-V
    unsafe {
        void* inputs[2]; inputs[0] = (void*)a; inputs[1] = (void*)b;
        unsigned long sizes[2]; sizes[0] = n * sizeof(float); sizes[1] = n * sizeof(float);
        unsigned int nParam = (unsigned int)n;
        return __vk_run_kernel(spirvCode, 0UL, "main", 2U, inputs, sizes, (void*)out, n * sizeof(float),
                                (const void*)&nParam, 4U, (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

int spirv_pow_f32(const float* a, const float* b, float* out, unsigned long n) {
    const unsigned int* spirvCode = (const unsigned int*)0; // would be pow.comp (GLSL 'pow()' builtin) compiled SPIR-V
    unsafe {
        void* inputs[2]; inputs[0] = (void*)a; inputs[1] = (void*)b;
        unsigned long sizes[2]; sizes[0] = n * sizeof(float); sizes[1] = n * sizeof(float);
        unsigned int nParam = (unsigned int)n;
        return __vk_run_kernel(spirvCode, 0UL, "main", 2U, inputs, sizes, (void*)out, n * sizeof(float),
                                (const void*)&nParam, 4U, (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

int spirv_relu_f32(const float* a, float* out, unsigned long n) {
    const unsigned int* spirvCode = (const unsigned int*)0; // would be relu.comp (GLSL 'max(x,0.0)') compiled SPIR-V
    unsafe {
        void* inputs[1]; inputs[0] = (void*)a;
        unsigned long sizes[1]; sizes[0] = n * sizeof(float);
        unsigned int nParam = (unsigned int)n;
        return __vk_run_kernel(spirvCode, 0UL, "main", 1U, inputs, sizes, (void*)out, n * sizeof(float),
                                (const void*)&nParam, 4U, (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

int spirv_log_f32(const float* a, float* out, unsigned long n) {
    const unsigned int* spirvCode = (const unsigned int*)0; // would be log.comp (GLSL 'log()') compiled SPIR-V
    unsafe {
        void* inputs[1]; inputs[0] = (void*)a;
        unsigned long sizes[1]; sizes[0] = n * sizeof(float);
        unsigned int nParam = (unsigned int)n;
        return __vk_run_kernel(spirvCode, 0UL, "main", 1U, inputs, sizes, (void*)out, n * sizeof(float),
                                (const void*)&nParam, 4U, (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

int spirv_exp_f32(const float* a, float* out, unsigned long n) {
    const unsigned int* spirvCode = (const unsigned int*)0; // would be exp.comp (GLSL 'exp()') compiled SPIR-V
    unsafe {
        void* inputs[1]; inputs[0] = (void*)a;
        unsigned long sizes[1]; sizes[0] = n * sizeof(float);
        unsigned int nParam = (unsigned int)n;
        return __vk_run_kernel(spirvCode, 0UL, "main", 1U, inputs, sizes, (void*)out, n * sizeof(float),
                                (const void*)&nParam, 4U, (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

int spirv_sqrt_f32(const float* a, float* out, unsigned long n) {
    const unsigned int* spirvCode = (const unsigned int*)0; // would be sqrt.comp (GLSL 'sqrt()') compiled SPIR-V
    unsafe {
        void* inputs[1]; inputs[0] = (void*)a;
        unsigned long sizes[1]; sizes[0] = n * sizeof(float);
        unsigned int nParam = (unsigned int)n;
        return __vk_run_kernel(spirvCode, 0UL, "main", 1U, inputs, sizes, (void*)out, n * sizeof(float),
                                (const void*)&nParam, 4U, (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

int spirv_scale_f32(const float* a, float k, float* out, unsigned long n) {
    const unsigned int* spirvCode = (const unsigned int*)0; // would be scale.comp compiled SPIR-V
    unsafe {
        void* inputs[1]; inputs[0] = (void*)a;
        unsigned long sizes[1]; sizes[0] = n * sizeof(float);
        // Push-constant layout: { float k; uint n; } -- 8 bytes.
        char pc[8];
        memcpy((void*)pc, (const void*)&k, 4UL);
        unsigned int nParam = (unsigned int)n;
        memcpy((void*)(pc + 4), (const void*)&nParam, 4UL);
        return __vk_run_kernel(spirvCode, 0UL, "main", 1U, inputs, sizes, (void*)out, n * sizeof(float),
                                (const void*)pc, 8U, (unsigned int)((n + 63UL) / 64UL), 1U, 1U);
    }
}

// out[0] = sum(a[0..n)) -- single workgroup, matching mps_sum_f32/
// cuda_sum_f32's "prove the shape works, not a tuned reduction" spirit;
// a real .comp source would still need an in-shader loop or
// shared-memory reduction tree since this dispatches groupCount=(1,1,1).
int spirv_sum_f32(const float* a, float* out, unsigned long n) {
    const unsigned int* spirvCode = (const unsigned int*)0; // would be sum.comp compiled SPIR-V
    unsafe {
        void* inputs[1]; inputs[0] = (void*)a;
        unsigned long sizes[1]; sizes[0] = n * sizeof(float);
        unsigned int nParam = (unsigned int)n;
        return __vk_run_kernel(spirvCode, 0UL, "main", 1U, inputs, sizes, (void*)out, sizeof(float),
                                (const void*)&nParam, 4U, 1U, 1U, 1U);
    }
}

// out[M,N] = a[M,K] . b[K,N] -- 2D dispatch (groupCountX/Y over N/M),
// push-constant layout { uint M; uint K; uint N; } (12 bytes), same
// naive one-thread-per-output-element shape as every other backend's
// matmul kernel here.
int spirv_matmul_f32(const float* a, const float* b, float* out,
                      unsigned long M, unsigned long K, unsigned long N) {
    const unsigned int* spirvCode = (const unsigned int*)0; // would be matmul.comp compiled SPIR-V
    unsafe {
        void* inputs[2]; inputs[0] = (void*)a; inputs[1] = (void*)b;
        unsigned long sizes[2]; sizes[0] = M * K * sizeof(float); sizes[1] = K * N * sizeof(float);
        char pc[12];
        unsigned int Mu = (unsigned int)M; unsigned int Ku = (unsigned int)K; unsigned int Nu = (unsigned int)N;
        memcpy((void*)pc, (const void*)&Mu, 4UL);
        memcpy((void*)(pc + 4), (const void*)&Ku, 4UL);
        memcpy((void*)(pc + 8), (const void*)&Nu, 4UL);
        unsigned int groupsX = (unsigned int)((N + 15UL) / 16UL);
        unsigned int groupsY = (unsigned int)((M + 15UL) / 16UL);
        return __vk_run_kernel(spirvCode, 0UL, "main", 2U, inputs, sizes, (void*)out, M * N * sizeof(float),
                                (const void*)pc, 12U, groupsX, groupsY, 1U);
    }
}

} // namespace std
