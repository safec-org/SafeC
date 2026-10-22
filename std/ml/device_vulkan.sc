// SafeC Standard Library — the Vulkan device backend (see device.h).
// VulkanBackend: any Vulkan 1.2 GPU (NVIDIA, AMD, Intel, Apple through
// MoltenVK, ...). Device memory is carved from a few large VkBuffers (blocks
// of 64 MB or more) and named by buffer device addresses, the way ROCm and
// CUDA name theirs, so a handle plus a byte offset is an address and the
// kernels' raw-address arguments (DevArgs.p, the optimizer's tables of
// handles) work as they do there; a table sorted by address finds the piece
// behind one for the host. device_kernels.cu's
// kernels run as SPIR-V that gen_device_kernels.sh compiles ahead of time
// with Slang (device_kernels_spirv.h, checked in: users need no shader
// toolchain), each kernel its own module, per build: plain, DK_TAGGED
// (16-bit arguments) and DK_ATTN64 (16 x 16 attention tiles for head dims up
// to 64). The kernels take Metal's paths (32-wide subgroups for the row
// kernels and block reductions, 8 x 8 attention tiles) and Metal's GEMMs and
// convolutions, mk_gemm and mk_conv_* (implicit GEMMs, through conv()),
// their simdgroup matrices emulated as a register tile per lane.
//
// Memory is one that is device-local and host-visible where the GPU has it
// (unified memory: Apple, integrated GPUs, resizable BAR), mapped once: the
// host reads and writes it in place. Elsewhere (or with SAFEC_VULKAN_STAGING
// =1, which tests that path) it is device-local only, and uploads and
// downloads pass through a host-visible staging arena.
//
// Work is recorded into one command buffer at a time (a pipeline barrier
// after each dispatch, so the next sees its writes) and submitted every
// DVK_COMMIT_EVERY dispatches without waiting, with a fence per command
// buffer in a ring of DVK_RING. On MoltenVK the barriers between dispatches
// are left out: its compute encoders are serial (each dispatch starts once
// the one before is done) and it ends the encoder at every barrier, ~20 us
// a dispatch (SAFEC_VULKAN_BARRIERS=1 puts them back). The host waits only when it reads
// (download, sync), before releasing memory, and before an upload while
// work is queued that doesn't fit the staging arena (one that does is
// copied there and applied in order by a dispatch, mk_copy). A program
// enables it with its vulkan feature; safeguard then links the loader
// (-lvulkan, -lvulkan-1 on Windows; on macOS: brew install vulkan-loader
// molten-vk, found in /opt/homebrew/lib or $VULKAN_SDK/lib).
//
// Requirements (else available() is 0): Vulkan 1.2 with bufferDeviceAddress,
// shaderInt64, shaderInt16 and storageBuffer16BitAccess (16-bit values; fp16
// converts through GLSL.std.450's PackHalf2x16, so no shaderFloat16), float
// atomic adds on buffers (VK_EXT_shader_atomic_float), subgroup arithmetic
// and shuffles, and 32-wide subgroups (the device's size, or one it can
// require: not AMD's wave64-only GCN). MoltenVK's fast math is turned off
// (VK_EXT_layer_settings), so the kernels round as the other backends' do.
// The SPIR-V is regenerated only with Slang (gen_device_kernels.sh); using
// the backend needs nothing but the Vulkan loader and a driver.
//
// Limits, against ROCm / CUDA:
//   - no doubles in the kernels (as on Metal): the batch / group norm
//     statistics accumulate in fp32, two-pass;
//   - the fused attention kernels work in 8 x 8 tiles for head dims over 64
//     (32 KB of shared memory, Vulkan's common minimum on Apple GPUs);
//   - no tensor cores / simdgroup matrices: GEMMs and convolutions multiply
//     on the ALUs (Vulkan's cooperative matrices aren't used: MoltenVK
//     doesn't expose them);
//   - a kernel runs on the CPU (device_ref.sc, on the mapped memory; counted
//     in DeviceStats.cpuLaunches) only if its pipeline can't be built; one
//     with 16-bit arguments or a table of handles then declines instead, as
//     do GEMMs and copies, and so does every kernel in staging mode (the
//     tensor op computes on the CPU; also counted);
//   - pipelines are built on first use (MoltenVK translates each to Metal
//     then), so a program's first steps include their compile time.
// safeguard: feature vulkan
// safeguard: link linux -lvulkan
// safeguard: link macos -L${VULKAN_SDK}/lib -L/opt/homebrew/lib -L/usr/local/lib -lvulkan
// safeguard: link windows -L${VULKAN_SDK}/Lib -lvulkan-1
#pragma once
#include <std/stddef.h>
#include <std/ml/device.h>
#include <std/ml/device_kernels.h>
#include <std/ml/device_kernels_spirv.h>

#if DK_SPIRV_SOURCE_CRC != DK_SOURCE_CRC
#error "device_kernels_spirv.h is older than device_kernels.cu: run gen_device_kernels.sh with slangc"
#endif

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);
extern void* memcpy(void* dst, const void* src, size_t n);
extern long  write(int fd, const void* buf, unsigned long n);
extern char* getenv(const char* name);
extern int   strcmp(const char* a, const char* b);

// ── Vulkan (vulkan_core.h's layouts; every handle as a pointer) ──────────────
struct VkApplicationInfo {
    int sType; const void* pNext;
    const char* pApplicationName; unsigned int applicationVersion;
    const char* pEngineName; unsigned int engineVersion;
    unsigned int apiVersion;
};
struct VkLayerSettingEXT {
    const char* pLayerName; const char* pSettingName; int type; unsigned int valueCount; const void* pValues;
};
struct VkLayerSettingsCreateInfoEXT {
    int sType; const void* pNext; unsigned int settingCount; const struct VkLayerSettingEXT* pSettings;
};
struct VkInstanceCreateInfo {
    int sType; const void* pNext; unsigned int flags;
    const struct VkApplicationInfo* pApplicationInfo;
    unsigned int enabledLayerCount; const char** ppEnabledLayerNames;
    unsigned int enabledExtensionCount; const char** ppEnabledExtensionNames;
};
struct VkExtensionProperties { char extensionName[256]; unsigned int specVersion; };
// VkPhysicalDeviceProperties2, the limits as words (DVK_LIMIT_*)
struct VkPhysicalDeviceProperties2 {
    int sType; void* pNext;
    unsigned int apiVersion; unsigned int driverVersion; unsigned int vendorID; unsigned int deviceID;
    int deviceType; char deviceName[256]; unsigned char pipelineCacheUUID[16];
    unsigned int pad_;
    unsigned int limits[126];
    unsigned int sparse[5];
};
#define DVK_LIMIT_SHARED 54         // maxComputeSharedMemorySize
struct VkQueueFamilyProperties {
    unsigned int queueFlags; unsigned int queueCount; unsigned int timestampValidBits;
    unsigned int granularity[3];
};
struct VkPhysicalDeviceMemoryProperties {
    unsigned int memoryTypeCount;
    unsigned int memoryTypes[64];   // {propertyFlags, heapIndex} each
    unsigned int memoryHeapCount;
    unsigned long memoryHeaps[32];  // {size, flags} each
};
// VkPhysicalDeviceFeatures2: VkPhysicalDeviceFeatures as 55 VkBool32
struct VkPhysicalDeviceFeatures2 { int sType; void* pNext; unsigned int f[55]; };
#define DVK_F_INT64 40
#define DVK_F_INT16 41
// The extension feature structs (VkPhysicalDevice16BitStorageFeatures, ...)
// as their VkBool32s: a struct of the largest's size.
struct DvkFeat_ { int sType; void* pNext; unsigned int b[16]; };
// VkPhysicalDeviceSubgroupProperties / ...SubgroupSizeControlProperties
struct DvkProp_ { int sType; void* pNext; unsigned int v[4]; };
struct VkPhysicalDeviceDriverProperties {
    int sType; void* pNext; int driverID; char driverName[256]; char driverInfo[256]; unsigned char conformanceVersion[4];
};
#define DVK_DRIVER_MOLTENVK 14
struct VkDeviceQueueCreateInfo {
    int sType; const void* pNext; unsigned int flags;
    unsigned int queueFamilyIndex; unsigned int queueCount; const float* pQueuePriorities;
};
struct VkDeviceCreateInfo {
    int sType; const void* pNext; unsigned int flags;
    unsigned int queueCreateInfoCount; const struct VkDeviceQueueCreateInfo* pQueueCreateInfos;
    unsigned int enabledLayerCount; const char** ppEnabledLayerNames;
    unsigned int enabledExtensionCount; const char** ppEnabledExtensionNames;
    const void* pEnabledFeatures;
};
struct VkBufferCreateInfo {
    int sType; const void* pNext; unsigned int flags; unsigned long size;
    unsigned int usage; int sharingMode; unsigned int queueFamilyIndexCount; const unsigned int* pQueueFamilyIndices;
};
struct VkMemoryRequirements { unsigned long size; unsigned long alignment; unsigned int memoryTypeBits; };
struct VkMemoryAllocateInfo { int sType; const void* pNext; unsigned long allocationSize; unsigned int memoryTypeIndex; };
struct VkMemoryAllocateFlagsInfo { int sType; const void* pNext; unsigned int flags; unsigned int deviceMask; };
struct VkBufferDeviceAddressInfo { int sType; const void* pNext; void* buffer; };
struct VkShaderModuleCreateInfo { int sType; const void* pNext; unsigned int flags; unsigned long codeSize; const unsigned int* pCode; };
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
struct VkPipelineShaderStageRequiredSubgroupSizeCreateInfo { int sType; void* pNext; unsigned int requiredSubgroupSize; };
struct VkCommandPoolCreateInfo { int sType; const void* pNext; unsigned int flags; unsigned int queueFamilyIndex; };
struct VkCommandBufferAllocateInfo { int sType; const void* pNext; void* commandPool; int level; unsigned int commandBufferCount; };
struct VkCommandBufferBeginInfo { int sType; const void* pNext; unsigned int flags; const void* pInheritanceInfo; };
struct VkSubmitInfo {
    int sType; const void* pNext;
    unsigned int waitSemaphoreCount; const void* pWaitSemaphores; const unsigned int* pWaitDstStageMask;
    unsigned int commandBufferCount; void** pCommandBuffers;
    unsigned int signalSemaphoreCount; const void* pSignalSemaphores;
};
struct VkFenceCreateInfo { int sType; const void* pNext; unsigned int flags; };
struct VkMemoryBarrier { int sType; const void* pNext; unsigned int srcAccessMask; unsigned int dstAccessMask; };

extern int vkEnumerateInstanceExtensionProperties(const char* layer, unsigned int* count, struct VkExtensionProperties* props);
extern int vkCreateInstance(const struct VkInstanceCreateInfo* info, const void* alloc, void** instance);
extern int vkEnumeratePhysicalDevices(void* instance, unsigned int* count, void** devices);
extern void vkGetPhysicalDeviceProperties2(void* pd, struct VkPhysicalDeviceProperties2* props);
extern void vkGetPhysicalDeviceFeatures2(void* pd, struct VkPhysicalDeviceFeatures2* features);
extern void vkGetPhysicalDeviceQueueFamilyProperties(void* pd, unsigned int* count, struct VkQueueFamilyProperties* props);
extern void vkGetPhysicalDeviceMemoryProperties(void* pd, struct VkPhysicalDeviceMemoryProperties* props);
extern int vkEnumerateDeviceExtensionProperties(void* pd, const char* layer, unsigned int* count, struct VkExtensionProperties* props);
extern int vkCreateDevice(void* pd, const struct VkDeviceCreateInfo* info, const void* alloc, void** device);
extern void vkGetDeviceQueue(void* device, unsigned int family, unsigned int index, void** queue);
extern int vkCreateBuffer(void* device, const struct VkBufferCreateInfo* info, const void* alloc, void** buffer);
extern void vkDestroyBuffer(void* device, void* buffer, const void* alloc);
extern void vkGetBufferMemoryRequirements(void* device, void* buffer, struct VkMemoryRequirements* req);
extern int vkAllocateMemory(void* device, const struct VkMemoryAllocateInfo* info, const void* alloc, void** memory);
extern void vkFreeMemory(void* device, void* memory, const void* alloc);
extern int vkBindBufferMemory(void* device, void* buffer, void* memory, unsigned long offset);
extern int vkMapMemory(void* device, void* memory, unsigned long offset, unsigned long size, unsigned int flags, void** data);
extern unsigned long vkGetBufferDeviceAddress(void* device, const struct VkBufferDeviceAddressInfo* info);
extern int vkCreateShaderModule(void* device, const struct VkShaderModuleCreateInfo* info, const void* alloc, void** module);
extern void vkDestroyShaderModule(void* device, void* module, const void* alloc);
extern int vkCreatePipelineLayout(void* device, const struct VkPipelineLayoutCreateInfo* info, const void* alloc, void** layout);
extern int vkCreateComputePipelines(void* device, void* cache, unsigned int count, const struct VkComputePipelineCreateInfo* infos,
                                    const void* alloc, void** pipelines);
extern int vkCreateCommandPool(void* device, const struct VkCommandPoolCreateInfo* info, const void* alloc, void** pool);
extern int vkAllocateCommandBuffers(void* device, const struct VkCommandBufferAllocateInfo* info, void** buffers);
extern int vkBeginCommandBuffer(void* cb, const struct VkCommandBufferBeginInfo* info);
extern int vkEndCommandBuffer(void* cb);
extern int vkResetCommandBuffer(void* cb, unsigned int flags);
extern void vkCmdBindPipeline(void* cb, int bindPoint, void* pipeline);
extern void vkCmdPushConstants(void* cb, void* layout, unsigned int stages, unsigned int offset, unsigned int size, const void* values);
extern void vkCmdDispatch(void* cb, unsigned int x, unsigned int y, unsigned int z);
extern void vkCmdPipelineBarrier(void* cb, unsigned int srcStages, unsigned int dstStages, unsigned int deps,
                                 unsigned int memCount, const struct VkMemoryBarrier* mem, unsigned int bufCount, const void* bufs,
                                 unsigned int imgCount, const void* imgs);
extern int vkQueueSubmit(void* queue, unsigned int count, const struct VkSubmitInfo* submits, void* fence);
extern int vkCreateFence(void* device, const struct VkFenceCreateInfo* info, const void* alloc, void** fence);
extern int vkWaitForFences(void* device, unsigned int count, void** fences, unsigned int waitAll, unsigned long timeout);
extern int vkResetFences(void* device, unsigned int count, void** fences);

#define DVK_STYPE_APP 0
#define DVK_STYPE_INSTANCE 1
#define DVK_STYPE_DEVICE_QUEUE 2
#define DVK_STYPE_DEVICE 3
#define DVK_STYPE_SUBMIT 4
#define DVK_STYPE_MEMORY_ALLOCATE 5
#define DVK_STYPE_FENCE 8
#define DVK_STYPE_BUFFER 12
#define DVK_STYPE_SHADER_MODULE 16
#define DVK_STYPE_SHADER_STAGE 18
#define DVK_STYPE_COMPUTE_PIPELINE 29
#define DVK_STYPE_PIPELINE_LAYOUT 30
#define DVK_STYPE_COMMAND_POOL 39
#define DVK_STYPE_COMMAND_BUFFER_ALLOCATE 40
#define DVK_STYPE_COMMAND_BUFFER_BEGIN 42
#define DVK_STYPE_MEMORY_BARRIER 46
#define DVK_STYPE_FEATURES2 1000059000
#define DVK_STYPE_PROPERTIES2 1000059001
#define DVK_STYPE_MEMORY_ALLOCATE_FLAGS 1000060000
#define DVK_STYPE_16BIT_STORAGE 1000083000
#define DVK_STYPE_SUBGROUP_PROPERTIES 1000094000
#define DVK_STYPE_DRIVER_PROPERTIES 1000196000
#define DVK_STYPE_SUBGROUP_SIZE_PROPERTIES 1000225000
#define DVK_STYPE_REQUIRED_SUBGROUP_SIZE 1000225001
#define DVK_STYPE_SUBGROUP_SIZE_FEATURES 1000225002
#define DVK_STYPE_BDA_INFO 1000244001
#define DVK_STYPE_BDA_FEATURES 1000257000
#define DVK_STYPE_ATOMIC_FLOAT_FEATURES 1000260000
#define DVK_STYPE_LAYER_SETTINGS 1000496000
#define DVK_API_1_2 4202496U                   // VK_MAKE_API_VERSION(0, 1, 2, 0)
#define DVK_API_1_3 4206592U
#define DVK_PORTABILITY_ENUMERATE 1U          // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
#define DVK_QUEUE_COMPUTE 2U
#define DVK_MEM_DEVICE_LOCAL 1U
#define DVK_MEM_HOST_VISIBLE 2U
#define DVK_MEM_HOST_COHERENT 4U
#define DVK_MEM_HOST_CACHED 8U
#define DVK_BUFFER_USAGE 131107U               // storage | transfer src / dst | shader device address
#define DVK_ALLOCATE_DEVICE_ADDRESS 2U
#define DVK_STAGE_COMPUTE 32U                  // VK_SHADER_STAGE_COMPUTE_BIT
#define DVK_REQUIRE_FULL_SUBGROUPS 2U
#define DVK_SUBGROUP_NEEDS 21U                 // basic | arithmetic | shuffle
#define DVK_STAGE_SHADER 2048U                 // VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
#define DVK_STAGE_HOST 16384U
#define DVK_ACCESS_SHADER_WRITE 64U
#define DVK_ACCESS_SHADER 96U                  // shader read | write
#define DVK_ACCESS_HOST 24576U                 // host read | write
#define DVK_POOL_RESET 2U
#define DVK_ONE_TIME 1U
#define DVK_DEVICE_LOST (-4)

#define DVK_BLOCK 256UL
#define DVK_MAX_GROUPS 4096UL
#define DVK_ATTN_T 8UL            // device_kernels.cu's ATTN_T in Vulkan builds ...
#define DVK_ATTN64_T 16UL         // ... and in their DK_ATTN64 builds, for head dims up to 64
#define DVK_GEMM_FILL 32UL        // the bigger GEMM / conv tiles from this many tiles on
#define DVK_PUSH_BYTES 128U       // DevArgs (DkGemm and DkConv are smaller)
#define DVK_COMMIT_EVERY 64UL     // dispatches per command buffer
#define DVK_RING 16UL             // command buffers (and fences) in flight
#define DVK_STAGE_BYTES 16777216UL   // the staging arena
#define DVK_BLOCK_BYTES 67108864UL   // device memory is carved from VkBuffers of at least this
#define DVK_ALIGN 256UL              // ... in pieces aligned to this

// A VkBuffer and its memory: its address range, the host's view (null when not mapped).
struct DvkBuf_ {
    unsigned long addr;
    unsigned long bytes;
    void* buf;
    void* mem;
    unsigned char* cpu;
};

// A free range of a block, in bytes from its start.
struct DvkRange_ {
    unsigned long off;
    unsigned long bytes;
};

// A block device memory is carved from: its buffer, its free ranges (sorted, coalesced), its live pieces.
struct DvkBlock_ {
    struct DvkBuf_ b;
    struct DvkRange_* free;
    unsigned long nfree;
    unsigned long capfree;
    unsigned long live;
};

// A piece (a handle's memory): its address range, the host's view, its block's address.
struct DvkPiece_ {
    unsigned long addr;
    unsigned long bytes;
    unsigned char* cpu;
    unsigned long block;
};

// mk_gemm's DkGemm.
struct DvkGemm_ {
    unsigned long a; unsigned long b; unsigned long c;
    unsigned long lda; unsigned long ldb; unsigned long ldc;
    unsigned long sa; unsigned long sb; unsigned long sc;
    unsigned int m; unsigned int n; unsigned int k; unsigned int ta; unsigned int tb; unsigned int at; unsigned int ct;
    float alpha;
    float beta;
};

// mk_conv_*'s DkConv: a plain convolution I [B, Ci, Hi, Wi] -> O [B, Co, Ho, Wo]
// with W [Co, Ci, kH, kW].
struct DvkConv_ {
    unsigned long x; unsigned long w; unsigned long y;
    unsigned int B; unsigned int Ci; unsigned int Hi; unsigned int Wi; unsigned int Co; unsigned int Ho; unsigned int Wo;
    unsigned int kH; unsigned int kW; unsigned int sH; unsigned int sW; unsigned int pH; unsigned int pW;
    unsigned int dH; unsigned int dW;
    unsigned int kSplit;
};

static int   g_dvk_state = 0;          // 0 = not yet, 1 = ready, -1 = unusable
static void* g_dvk_instance = (void*)0;
static void* g_dvk_pd = (void*)0;
static void* g_dvk_dev = (void*)0;
static void* g_dvk_queue = (void*)0;
static void* g_dvk_layout = (void*)0;
static void* g_dvk_pool = (void*)0;
static unsigned int g_dvk_family = 0U;
static unsigned int g_dvk_memtype = 0U;     // device buffers
static unsigned int g_dvk_hosttype = 0U;    // the staging arena
static int g_dvk_mapped = 0;                // device buffers are host-visible
static int g_dvk_full_sg = 0;               // can require full subgroups
static int g_dvk_req_sg = 0;                // must require a subgroup size of 32
static int g_dvk_serial = 0;                // dispatches run in order without barriers (MoltenVK)
static int g_dvk_reported = 0;

// Pipelines: one per module of device_kernels_spirv.h (0 = not yet, -1 = can't)
static void* g_dvk_pipe[DK_SPIRV_MODULES];
static int   g_dvk_pipe_state[DK_SPIRV_MODULES];
static int   g_dvk_kmod[4 * DK_COUNT];      // the module of (build, DK_* kernel): 0 = not looked up, -1 = none, else index + 1

// Blocks, and the pieces handed out, sorted by address.
static struct DvkBlock_* g_dvk_blocks = (struct DvkBlock_*)0;
static unsigned long g_dvk_nblocks = 0UL;
static unsigned long g_dvk_capblocks = 0UL;
static struct DvkPiece_* g_dvk_pieces = (struct DvkPiece_*)0;
static unsigned long g_dvk_npieces = 0UL;
static unsigned long g_dvk_cappieces = 0UL;

// The command buffer ring: slot i's fence is pending while its submission may run.
static void* g_dvk_cb[DVK_RING];
static void* g_dvk_fence[DVK_RING];
static int   g_dvk_pending[DVK_RING];
static long  g_dvk_open = -1L;              // the slot being recorded, or -1
static unsigned long g_dvk_next = 0UL;      // the slot the next command buffer takes
static void* g_dvk_last_pipe = (void*)0;
static unsigned long g_dvk_count = 0UL;

// The staging arena (host-visible, coherent): bump-allocated, reset when the GPU is idle.
static struct DvkBuf_ g_dvk_stage;
static unsigned long g_dvk_stage_used = 0UL;

static void dvk_log_(const char* what, const char* detail) {
    unsafe {
        unsigned long n = 0UL;
        while (what[n] != (char)0) n = n + 1UL;
        write(2, (const void*)what, n);
        n = 0UL;
        while (detail[n] != (char)0) n = n + 1UL;
        write(2, (const void*)detail, n);
        write(2, (const void*)"\n", 1UL);
    }
}

static int dvk_streq_(const char* a, const char* b) {
    unsafe { return strcmp(a, b) == 0; }
}

// 1 if the extension list (n entries) names 'name'.
static int dvk_has_ext_(const struct VkExtensionProperties* exts, unsigned int n, const char* name) {
    unsafe {
        unsigned int i = 0U;
        while (i < n) {
            if (dvk_streq_((const char*)&exts[i].extensionName[0], name)) return 1;
            i = i + 1U;
        }
    }
    return 0;
}

// The extensions of physical device pd (malloc'd; *n entries), null on failure.
static struct VkExtensionProperties* dvk_device_exts_(void* pd, unsigned int* n) {
    unsafe {
        *n = 0U;
        if (vkEnumerateDeviceExtensionProperties(pd, (const char*)0, n, (struct VkExtensionProperties*)0) != 0) return (struct VkExtensionProperties*)0;
        struct VkExtensionProperties* e = (struct VkExtensionProperties*)malloc(sizeof(struct VkExtensionProperties) * ((unsigned long)*n + 1UL));
        if (e == (struct VkExtensionProperties*)0) return e;
        if (vkEnumerateDeviceExtensionProperties(pd, (const char*)0, n, e) != 0) { free((void*)e); return (struct VkExtensionProperties*)0; }
        return e;
    }
}

// The instance: Vulkan 1.2, portability devices enumerated (MoltenVK), and
// MoltenVK's fast math off where layer settings reach it.
static int dvk_instance_() {
    unsafe {
        unsigned int n = 0U;
        vkEnumerateInstanceExtensionProperties((const char*)0, &n, (struct VkExtensionProperties*)0);
        struct VkExtensionProperties* exts = (struct VkExtensionProperties*)malloc(sizeof(struct VkExtensionProperties) * ((unsigned long)n + 1UL));
        if (exts == (struct VkExtensionProperties*)0) return 0;
        if (vkEnumerateInstanceExtensionProperties((const char*)0, &n, exts) != 0) n = 0U;
        const char* names[2];
        unsigned int nn = 0U;
        unsigned int flags = 0U;
        if (dvk_has_ext_(exts, n, "VK_KHR_portability_enumeration")) {
            names[nn] = "VK_KHR_portability_enumeration"; nn = nn + 1U;
            flags = DVK_PORTABILITY_ENUMERATE;
        }
        int settings = dvk_has_ext_(exts, n, "VK_EXT_layer_settings");
        if (settings) { names[nn] = "VK_EXT_layer_settings"; nn = nn + 1U; }
        free((void*)exts);

        unsigned int never = 0U;   // MVK_CONFIG_FAST_MATH_NEVER
        struct VkLayerSettingEXT s;
        s.pLayerName = "MoltenVK"; s.pSettingName = "MVK_CONFIG_FAST_MATH_ENABLED";
        s.type = 3; s.valueCount = 1U; s.pValues = (const void*)&never;   // VK_LAYER_SETTING_TYPE_UINT32_EXT
        struct VkLayerSettingsCreateInfoEXT ls;
        ls.sType = DVK_STYPE_LAYER_SETTINGS; ls.pNext = (const void*)0; ls.settingCount = 1U; ls.pSettings = (const struct VkLayerSettingEXT*)&s;

        struct VkApplicationInfo app;
        app.sType = DVK_STYPE_APP; app.pNext = (const void*)0;
        app.pApplicationName = "safec"; app.applicationVersion = 1U;
        app.pEngineName = "std::ml"; app.engineVersion = 1U;
        app.apiVersion = DVK_API_1_3;
        struct VkInstanceCreateInfo ci;
        ci.sType = DVK_STYPE_INSTANCE; ci.pNext = settings ? (const void*)&ls : (const void*)0; ci.flags = flags;
        ci.pApplicationInfo = (const struct VkApplicationInfo*)&app;
        ci.enabledLayerCount = 0U; ci.ppEnabledLayerNames = (const char**)0;
        ci.enabledExtensionCount = nn; ci.ppEnabledExtensionNames = (const char**)&names[0];
        if (vkCreateInstance(&ci, (const void*)0, &g_dvk_instance) != 0) return 0;
    }
    return 1;
}

// What physical device pd offers; 0 if it lacks something the kernels need.
// Fills the feature structs to enable (chained from f2) when it qualifies.
static int dvk_qualifies_(void* pd, struct VkPhysicalDeviceFeatures2* f2, struct DvkFeat_* f16, struct DvkFeat_* bda,
                          struct DvkFeat_* atom, struct DvkFeat_* sgc) {
    unsafe {
        struct DvkProp_ sg;
        struct DvkProp_ sgs;
        sg.sType = DVK_STYPE_SUBGROUP_PROPERTIES; sg.pNext = (void*)&sgs;
        struct VkPhysicalDeviceDriverProperties drv;
        sgs.sType = DVK_STYPE_SUBGROUP_SIZE_PROPERTIES; sgs.pNext = (void*)&drv;
        drv.sType = DVK_STYPE_DRIVER_PROPERTIES; drv.pNext = (void*)0; drv.driverID = 0;
        sg.v[0] = 0U; sg.v[1] = 0U; sg.v[2] = 0U; sgs.v[0] = 0U; sgs.v[1] = 0U; sgs.v[3] = 0U;
        struct VkPhysicalDeviceProperties2 p;
        p.sType = DVK_STYPE_PROPERTIES2; p.pNext = (void*)&sg;
        vkGetPhysicalDeviceProperties2(pd, &p);
        if (p.apiVersion < DVK_API_1_2) return 0;
        // subgroups: arithmetic and shuffles in compute shaders, 32 wide
        if ((sg.v[1] & DVK_STAGE_COMPUTE) == 0U || (sg.v[2] & DVK_SUBGROUP_NEEDS) != DVK_SUBGROUP_NEEDS) return 0;

        unsigned int i = 0U;
        f16->sType = DVK_STYPE_16BIT_STORAGE; f16->pNext = (void*)bda;
        bda->sType = DVK_STYPE_BDA_FEATURES; bda->pNext = (void*)atom;
        atom->sType = DVK_STYPE_ATOMIC_FLOAT_FEATURES; atom->pNext = (void*)sgc;
        sgc->sType = DVK_STYPE_SUBGROUP_SIZE_FEATURES; sgc->pNext = (void*)0;
        while (i < 16U) { f16->b[i] = 0U; bda->b[i] = 0U; atom->b[i] = 0U; sgc->b[i] = 0U; i = i + 1U; }
        f2->sType = DVK_STYPE_FEATURES2; f2->pNext = (void*)f16;
        i = 0U;
        while (i < 55U) { f2->f[i] = 0U; i = i + 1U; }
        vkGetPhysicalDeviceFeatures2(pd, f2);
        // bufferDeviceAddress, shaderInt64, shaderBufferFloat32AtomicAdd, and the
        // 16-bit loads and stores of the 16-bit kernels (and the GEMMs', for any c)
        if (!bda->b[0] || !f2->f[DVK_F_INT64] || !atom->b[1] || !f16->b[0] || !f2->f[DVK_F_INT16]) return 0;
        int full = sgc->b[1] != 0U;                                          // computeFullSubgroups
        int need = 0;
        if (sg.v[0] != 32U) {
            // a size it can require for compute shaders
            if (!sgc->b[0] || sgs.v[0] > 32U || sgs.v[1] < 32U || (sgs.v[3] & DVK_STAGE_COMPUTE) == 0U || !full) return 0;
            need = 1;
        }
        // what to enable: those and the subgroup controls
        i = 0U;
        while (i < 55U) { f2->f[i] = 0U; i = i + 1U; }
        f2->f[DVK_F_INT64] = 1U;
        f2->f[DVK_F_INT16] = 1U;
        f16->b[0] = 1U; f16->b[1] = 0U; f16->b[2] = 0U; f16->b[3] = 0U;
        bda->b[0] = 1U; bda->b[1] = 0U; bda->b[2] = 0U;
        i = 0U;
        while (i < 12U) { atom->b[i] = i == 1U ? 1U : 0U; i = i + 1U; }
        sgc->b[0] = need ? 1U : 0U; sgc->b[1] = full ? 1U : 0U;
        g_dvk_full_sg = full;
        g_dvk_req_sg = need;
        // MoltenVK records compute dispatches into serial Metal encoders
        // (MTLDispatchTypeSerial: each starts once the one before it is done)
        // and ends the encoder at every pipeline barrier, which costs ~20 us
        // a dispatch; there the barriers between dispatches are redundant.
        char* forced = getenv("SAFEC_VULKAN_BARRIERS");
        g_dvk_serial = drv.driverID == DVK_DRIVER_MOLTENVK && (forced == (char*)0 || forced[0] != '1');
        return p.deviceType == 2 ? 3 : (p.deviceType == 1 ? 2 : 1);   // discrete, integrated, other
    }
}

// The memory type for 'want' among memTypeBits, preferring those with 'nice' too; -1 if none.
static int dvk_memtype_(const struct VkPhysicalDeviceMemoryProperties* mp, unsigned int bits, unsigned int want, unsigned int nice) {
    unsafe {
        int best = -1;
        unsigned int i = 0U;
        while (i < mp->memoryTypeCount) {
            unsigned int fl = mp->memoryTypes[2U * i];
            if ((bits & (1U << i)) != 0U && (fl & want) == want) {
                if (best < 0) best = (int)i;
                if ((fl & nice) == nice) return (int)i;
            }
            i = i + 1U;
        }
        return best;
    }
}

static unsigned long dvk_alloc_buf_(unsigned long bytes, unsigned int type, struct DvkBuf_* out);

static int dvk_ensure_() {
    if (g_dvk_state != 0) return g_dvk_state == 1;
    g_dvk_state = -1;
    if (!dvk_instance_()) return 0;
    unsafe {
        unsigned int n = 0U;
        if (vkEnumeratePhysicalDevices(g_dvk_instance, &n, (void**)0) != 0 || n == 0U) return 0;
        void** pds = (void**)malloc(sizeof(void*) * (unsigned long)n);
        if (pds == (void**)0) return 0;
        vkEnumeratePhysicalDevices(g_dvk_instance, &n, pds);
        // the best qualifying device: discrete, then integrated, then any
        struct VkPhysicalDeviceFeatures2 f2;
        struct DvkFeat_ f16; struct DvkFeat_ bda; struct DvkFeat_ atom; struct DvkFeat_ sgc;
        int bestRank = 0;
        unsigned int best = 0U;
        unsigned int i = 0U;
        while (i < n) {
            int r = dvk_qualifies_(pds[i], &f2, &f16, &bda, &atom, &sgc);
            if (r > bestRank) { bestRank = r; best = i; }
            i = i + 1U;
        }
        if (bestRank == 0) { free((void*)pds); return 0; }
        g_dvk_pd = pds[best];
        free((void*)pds);
        dvk_qualifies_(g_dvk_pd, &f2, &f16, &bda, &atom, &sgc);   // its features to enable

        // a queue family with compute
        unsigned int nq = 0U;
        vkGetPhysicalDeviceQueueFamilyProperties(g_dvk_pd, &nq, (struct VkQueueFamilyProperties*)0);
        struct VkQueueFamilyProperties qf[16];
        if (nq > 16U) nq = 16U;
        vkGetPhysicalDeviceQueueFamilyProperties(g_dvk_pd, &nq, &qf[0]);
        int fam = -1;
        i = 0U;
        while (i < nq && fam < 0) { if ((qf[i].queueFlags & DVK_QUEUE_COMPUTE) != 0U) fam = (int)i; i = i + 1U; }
        if (fam < 0) return 0;
        g_dvk_family = (unsigned int)fam;

        unsigned int ne = 0U;
        struct VkExtensionProperties* exts = dvk_device_exts_(g_dvk_pd, &ne);
        if (exts == (struct VkExtensionProperties*)0) return 0;
        const char* names[4];
        unsigned int nn = 0U;
        if (!dvk_has_ext_(exts, ne, "VK_EXT_shader_atomic_float")) { free((void*)exts); return 0; }
        names[nn] = "VK_EXT_shader_atomic_float"; nn = nn + 1U;
        if (dvk_has_ext_(exts, ne, "VK_EXT_subgroup_size_control")) { names[nn] = "VK_EXT_subgroup_size_control"; nn = nn + 1U; }
        if (dvk_has_ext_(exts, ne, "VK_KHR_portability_subset")) { names[nn] = "VK_KHR_portability_subset"; nn = nn + 1U; }
        free((void*)exts);

        float prio = 1.0f;
        struct VkDeviceQueueCreateInfo q;
        q.sType = DVK_STYPE_DEVICE_QUEUE; q.pNext = (const void*)0; q.flags = 0U;
        q.queueFamilyIndex = g_dvk_family; q.queueCount = 1U; q.pQueuePriorities = (const float*)&prio;
        struct VkDeviceCreateInfo dc;
        dc.sType = DVK_STYPE_DEVICE; dc.pNext = (const void*)&f2; dc.flags = 0U;
        dc.queueCreateInfoCount = 1U; dc.pQueueCreateInfos = (const struct VkDeviceQueueCreateInfo*)&q;
        dc.enabledLayerCount = 0U; dc.ppEnabledLayerNames = (const char**)0;
        dc.enabledExtensionCount = nn; dc.ppEnabledExtensionNames = (const char**)&names[0];
        dc.pEnabledFeatures = (const void*)0;
        if (vkCreateDevice(g_dvk_pd, &dc, (const void*)0, &g_dvk_dev) != 0) { dvk_log_("device_vulkan: ", "vkCreateDevice failed"); return 0; }
        vkGetDeviceQueue(g_dvk_dev, g_dvk_family, 0U, &g_dvk_queue);

        // memory: device-local and mapped where it can be (cached for the host's reads), else device-local
        struct VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(g_dvk_pd, &mp);
        char* env = getenv("SAFEC_VULKAN_STAGING");
        int staging = env != (char*)0 && env[0] == '1';
        int t = staging ? -1 : dvk_memtype_(&mp, 0xffffffffU, DVK_MEM_DEVICE_LOCAL | DVK_MEM_HOST_VISIBLE | DVK_MEM_HOST_COHERENT,
                                            DVK_MEM_DEVICE_LOCAL | DVK_MEM_HOST_VISIBLE | DVK_MEM_HOST_COHERENT | DVK_MEM_HOST_CACHED);
        if (t >= 0) {
            g_dvk_mapped = 1;
        } else {
            // device-local without host access (the staging path)
            unsigned int k = 0U;
            while (k < mp.memoryTypeCount && t < 0) {
                unsigned int fl = mp.memoryTypes[2U * k];
                if ((fl & DVK_MEM_DEVICE_LOCAL) != 0U && (fl & DVK_MEM_HOST_VISIBLE) == 0U) t = (int)k;
                k = k + 1U;
            }
            if (t < 0) t = dvk_memtype_(&mp, 0xffffffffU, DVK_MEM_DEVICE_LOCAL, DVK_MEM_DEVICE_LOCAL);
            if (t < 0) return 0;
        }
        g_dvk_memtype = (unsigned int)t;
        int h = dvk_memtype_(&mp, 0xffffffffU, DVK_MEM_HOST_VISIBLE | DVK_MEM_HOST_COHERENT,
                             DVK_MEM_HOST_VISIBLE | DVK_MEM_HOST_COHERENT | DVK_MEM_HOST_CACHED);
        if (h < 0) return 0;
        g_dvk_hosttype = (unsigned int)h;

        struct VkPushConstantRange pr;
        pr.stageFlags = DVK_STAGE_COMPUTE; pr.offset = 0U; pr.size = DVK_PUSH_BYTES;
        struct VkPipelineLayoutCreateInfo pl;
        pl.sType = DVK_STYPE_PIPELINE_LAYOUT; pl.pNext = (const void*)0; pl.flags = 0U;
        pl.setLayoutCount = 0U; pl.pSetLayouts = (const void*)0;
        pl.pushConstantRangeCount = 1U; pl.pPushConstantRanges = (const struct VkPushConstantRange*)&pr;
        if (vkCreatePipelineLayout(g_dvk_dev, &pl, (const void*)0, &g_dvk_layout) != 0) return 0;

        struct VkCommandPoolCreateInfo cp;
        cp.sType = DVK_STYPE_COMMAND_POOL; cp.pNext = (const void*)0; cp.flags = DVK_POOL_RESET; cp.queueFamilyIndex = g_dvk_family;
        if (vkCreateCommandPool(g_dvk_dev, &cp, (const void*)0, &g_dvk_pool) != 0) return 0;
        struct VkCommandBufferAllocateInfo ca;
        ca.sType = DVK_STYPE_COMMAND_BUFFER_ALLOCATE; ca.pNext = (const void*)0; ca.commandPool = g_dvk_pool;
        ca.level = 0; ca.commandBufferCount = (unsigned int)DVK_RING;
        if (vkAllocateCommandBuffers(g_dvk_dev, &ca, &g_dvk_cb[0]) != 0) return 0;
        struct VkFenceCreateInfo fc;
        fc.sType = DVK_STYPE_FENCE; fc.pNext = (const void*)0; fc.flags = 0U;
        unsigned long r = 0UL;
        while (r < DVK_RING) {
            if (vkCreateFence(g_dvk_dev, &fc, (const void*)0, &g_dvk_fence[r]) != 0) return 0;
            g_dvk_pending[r] = 0;
            r = r + 1UL;
        }
        if (dvk_alloc_buf_(DVK_STAGE_BYTES, g_dvk_hosttype, &g_dvk_stage) == 0UL) return 0;
    }
    g_dvk_state = 1;
    return 1;
}

// A buffer of 'bytes' from memory type 'type', mapped if host-visible: its address, 0 on failure.
static unsigned long dvk_alloc_buf_(unsigned long bytes, unsigned int type, struct DvkBuf_* out) {
    unsafe {
        struct VkBufferCreateInfo bc;
        bc.sType = DVK_STYPE_BUFFER; bc.pNext = (const void*)0; bc.flags = 0U; bc.size = bytes;
        bc.usage = DVK_BUFFER_USAGE; bc.sharingMode = 0; bc.queueFamilyIndexCount = 0U; bc.pQueueFamilyIndices = (const unsigned int*)0;
        void* buf = (void*)0;
        if (vkCreateBuffer(g_dvk_dev, &bc, (const void*)0, &buf) != 0) return 0UL;
        struct VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(g_dvk_dev, buf, &req);
        struct VkMemoryAllocateFlagsInfo af;
        af.sType = DVK_STYPE_MEMORY_ALLOCATE_FLAGS; af.pNext = (const void*)0; af.flags = DVK_ALLOCATE_DEVICE_ADDRESS; af.deviceMask = 0U;
        struct VkMemoryAllocateInfo ai;
        ai.sType = DVK_STYPE_MEMORY_ALLOCATE; ai.pNext = (const void*)&af; ai.allocationSize = req.size; ai.memoryTypeIndex = type;
        void* mem = (void*)0;
        if ((req.memoryTypeBits & (1U << type)) == 0U || vkAllocateMemory(g_dvk_dev, &ai, (const void*)0, &mem) != 0) {
            vkDestroyBuffer(g_dvk_dev, buf, (const void*)0);
            return 0UL;
        }
        void* cpu = (void*)0;
        int ok = vkBindBufferMemory(g_dvk_dev, buf, mem, 0UL) == 0;
        if (ok && (type == g_dvk_hosttype || g_dvk_mapped)) ok = vkMapMemory(g_dvk_dev, mem, 0UL, req.size, 0U, &cpu) == 0;
        struct VkBufferDeviceAddressInfo bi;
        bi.sType = DVK_STYPE_BDA_INFO; bi.pNext = (const void*)0; bi.buffer = buf;
        unsigned long addr = ok ? vkGetBufferDeviceAddress(g_dvk_dev, &bi) : 0UL;
        if (addr == 0UL || dev_tag_(addr) != 0UL) {
            vkDestroyBuffer(g_dvk_dev, buf, (const void*)0);
            vkFreeMemory(g_dvk_dev, mem, (const void*)0);
            return 0UL;
        }
        out->addr = addr; out->bytes = bytes; out->buf = buf; out->mem = mem; out->cpu = (unsigned char*)cpu;
        return addr;
    }
}

// ── Kernels: device_kernels_spirv.h's modules ───────────────────────────────
// The module of entry point 'name' in build b, or -1.
static int dvk_module_(int b, const char* name) {
    unsafe {
        int i = 0;
        while (i < DK_SPIRV_MODULES) {
            if (dk_spirv_index_[3 * i] == (unsigned int)b && dvk_streq_(dk_spirv_name_(i), name)) return i;
            i = i + 1;
        }
    }
    return -1;
}

// Module m's pipeline (built on first use), null if it can't be built.
static void* dvk_pipe_at_(int m) {
    if (m < 0) return (void*)0;
    unsafe {
        if (g_dvk_pipe_state[m] != 0) return g_dvk_pipe[m];
        g_dvk_pipe_state[m] = -1;
        struct VkShaderModuleCreateInfo sm;
        sm.sType = DVK_STYPE_SHADER_MODULE; sm.pNext = (const void*)0; sm.flags = 0U;
        sm.codeSize = 4UL * (unsigned long)dk_spirv_index_[3 * m + 2];
        sm.pCode = &dk_spirv_words_[dk_spirv_index_[3 * m + 1]];
        void* mod = (void*)0;
        if (vkCreateShaderModule(g_dvk_dev, &sm, (const void*)0, &mod) != 0) {
            dvk_log_("device_vulkan: a kernel's module failed: ", dk_spirv_name_(m));
            return (void*)0;
        }
        struct VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rs;
        rs.sType = DVK_STYPE_REQUIRED_SUBGROUP_SIZE; rs.pNext = (void*)0; rs.requiredSubgroupSize = 32U;
        struct VkComputePipelineCreateInfo pc;
        pc.sType = DVK_STYPE_COMPUTE_PIPELINE; pc.pNext = (const void*)0; pc.flags = 0U;
        pc.stage.sType = DVK_STYPE_SHADER_STAGE; pc.stage.pNext = g_dvk_req_sg ? (const void*)&rs : (const void*)0;
        pc.stage.flags = g_dvk_full_sg ? DVK_REQUIRE_FULL_SUBGROUPS : 0U;
        pc.stage.stage = DVK_STAGE_COMPUTE; pc.stage.module = mod; pc.stage.pName = dk_spirv_name_(m);
        pc.stage.pSpecializationInfo = (const void*)0;
        pc.layout = g_dvk_layout; pc.basePipelineHandle = (void*)0; pc.basePipelineIndex = -1;
        void* pipe = (void*)0;
        if (vkCreateComputePipelines(g_dvk_dev, (void*)0, 1U, &pc, (const void*)0, &pipe) != 0) {
            dvk_log_("device_vulkan: a kernel's pipeline failed: ", dk_spirv_name_(m));
            pipe = (void*)0;
        }
        vkDestroyShaderModule(g_dvk_dev, mod, (const void*)0);
        if (pipe == (void*)0) return pipe;
        g_dvk_pipe[m] = pipe;
        g_dvk_pipe_state[m] = 1;
        return pipe;
    }
}

// DK_* kernel k's pipeline in build b (bit 0 DK_TAGGED, bit 1 DK_ATTN64).
static void* dvk_pipe_(int b, int k) {
    unsafe {
        int slot = b * DK_COUNT + k;
        if (g_dvk_kmod[slot] == 0) {
            int m = dvk_module_(b, device_kernel_name_(k));
            g_dvk_kmod[slot] = m + 1 == 0 ? -1 : m + 1;
        }
        return g_dvk_kmod[slot] < 0 ? (void*)0 : dvk_pipe_at_(g_dvk_kmod[slot] - 1);
    }
}

// The pipeline of a named (mk_*) kernel, cached in *cache (0 = not looked up).
static void* dvk_named_pipe_(const char* name, int* cache) {
    unsafe {
        if (*cache == 0) {
            int m = dvk_module_(0, name);
            *cache = m < 0 ? -1 : m + 1;
        }
        return *cache < 0 ? (void*)0 : dvk_pipe_at_(*cache - 1);
    }
}

// The kernels with a simdgroup per row (device_kernels.cu's GSR): 32 threads for each of the n rows.
static int dvk_rows_(int kernel) {
    return kernel == DK_ROWSUM || kernel == DK_SOFTMAX_ROWS || kernel == DK_SOFTMAX_BWD || kernel == DK_CE_FWD ||
           kernel == DK_CE_BWD || kernel == DK_LAYERNORM || kernel == DK_LAYERNORM_BWD;
}

// ── Buffers ──────────────────────────────────────────────────────────────────
// Device memory is carved from a few large VkBuffers (blocks), as Vulkan
// drivers prefer: some cap allocations at 4096, and MoltenVK makes every
// buffer with a device address resident on each command encoder it opens.
// The piece holding address h, or -1.
static long dvk_find_(unsigned long h) {
    unsafe {
        long lo = 0L;
        long hi = (long)g_dvk_npieces - 1L;
        while (lo <= hi) {
            long mid = (lo + hi) / 2L;
            struct DvkPiece_* b = &g_dvk_pieces[mid];
            if (h < b->addr) hi = mid - 1L;
            else if (h >= b->addr + b->bytes) lo = mid + 1L;
            else return mid;
        }
    }
    return -1L;
}

// The host's view of device address h (null for an unknown or unmapped one).
static unsigned char* dvk_host_(unsigned long h) {
    h = dev_addr_(h);
    long i = dvk_find_(h);
    if (i < 0L) return (unsigned char*)0;
    unsafe {
        if (g_dvk_pieces[i].cpu == (unsigned char*)0) return (unsigned char*)0;
        return g_dvk_pieces[i].cpu + (h - g_dvk_pieces[i].addr);
    }
}

// Room for one more element in a malloc'd array (*arr of *cap elements of 'size' bytes, n used).
static int dvk_grow_(void** arr, unsigned long* cap, unsigned long n, unsigned long size) {
    unsafe {
        if (n < *cap) return 1;
        unsigned long nc = *cap == 0UL ? 16UL : *cap * 2UL;
        void* na = malloc(size * nc);
        if (na == (void*)0) return 0;
        if (n > 0UL) memcpy(na, (const void*)*arr, size * n);
        if (*arr != (void*)0) free(*arr);
        *arr = na;
        *cap = nc;
    }
    return 1;
}

// 'need' bytes from block k's free ranges (first fit): the piece's offset, or -1.
static long dvk_take_(unsigned long k, unsigned long need) {
    unsafe {
        struct DvkBlock_* bl = &g_dvk_blocks[k];
        unsigned long i = 0UL;
        while (i < bl->nfree) {
            struct DvkRange_* r = &bl->free[i];
            if (r->bytes >= need) {
                unsigned long off = r->off;
                r->off = r->off + need;
                r->bytes = r->bytes - need;
                if (r->bytes == 0UL) {
                    unsigned long j = i;
                    while (j + 1UL < bl->nfree) { bl->free[j] = bl->free[j + 1UL]; j = j + 1UL; }
                    bl->nfree = bl->nfree - 1UL;
                }
                bl->live = bl->live + 1UL;
                return (long)off;
            }
            i = i + 1UL;
        }
    }
    return -1L;
}

// Gives [off, off + bytes) back to block k's free ranges, merging neighbours.
static int dvk_give_(unsigned long k, unsigned long off, unsigned long bytes) {
    unsafe {
        struct DvkBlock_* bl = &g_dvk_blocks[k];
        unsigned long at = 0UL;
        while (at < bl->nfree && bl->free[at].off < off) at = at + 1UL;
        int prev = at > 0UL && bl->free[at - 1UL].off + bl->free[at - 1UL].bytes == off;
        int next = at < bl->nfree && off + bytes == bl->free[at].off;
        if (prev && next) {
            bl->free[at - 1UL].bytes = bl->free[at - 1UL].bytes + bytes + bl->free[at].bytes;
            unsigned long j = at;
            while (j + 1UL < bl->nfree) { bl->free[j] = bl->free[j + 1UL]; j = j + 1UL; }
            bl->nfree = bl->nfree - 1UL;
        } else if (prev) {
            bl->free[at - 1UL].bytes = bl->free[at - 1UL].bytes + bytes;
        } else if (next) {
            bl->free[at].off = off;
            bl->free[at].bytes = bl->free[at].bytes + bytes;
        } else {
            if (!dvk_grow_((void**)&bl->free, &bl->capfree, bl->nfree, sizeof(struct DvkRange_))) return 0;
            unsigned long j = bl->nfree;
            while (j > at) { bl->free[j] = bl->free[j - 1UL]; j = j - 1UL; }
            bl->free[at].off = off;
            bl->free[at].bytes = bytes;
            bl->nfree = bl->nfree + 1UL;
        }
        bl->live = bl->live - 1UL;
    }
    return 1;
}

// ── The command stream ───────────────────────────────────────────────────────
static int dvk_check_(int r) {
    if (r == 0) return 1;
    if (!g_dvk_reported) {
        g_dvk_reported = 1;
        dvk_log_("device_vulkan: ", r == DVK_DEVICE_LOST ? "the device was lost" : "a submission failed");
    }
    return 0;
}

// Waits for ring slot s's submission.
static int dvk_wait_slot_(unsigned long s) {
    if (!g_dvk_pending[s]) return 1;
    unsafe {
        int r = vkWaitForFences(g_dvk_dev, 1U, &g_dvk_fence[s], 1U, ~0UL);
        vkResetFences(g_dvk_dev, 1U, &g_dvk_fence[s]);
        g_dvk_pending[s] = 0;
        return dvk_check_(r);
    }
}

// The open command buffer, started (in the next ring slot) if there is none.
static int dvk_begin_() {
    if (g_dvk_open >= 0L) return 1;
    unsafe {
        unsigned long s = g_dvk_next;
        if (!dvk_wait_slot_(s)) return 0;
        vkResetCommandBuffer(g_dvk_cb[s], 0U);
        struct VkCommandBufferBeginInfo bi;
        bi.sType = DVK_STYPE_COMMAND_BUFFER_BEGIN; bi.pNext = (const void*)0; bi.flags = DVK_ONE_TIME; bi.pInheritanceInfo = (const void*)0;
        if (vkBeginCommandBuffer(g_dvk_cb[s], &bi) != 0) return 0;
        g_dvk_open = (long)s;
    }
    g_dvk_last_pipe = (void*)0;
    g_dvk_count = 0UL;
    return 1;
}

// Submits the open command buffer (no wait).
static int dvk_commit_() {
    if (g_dvk_open < 0L) return 1;
    unsafe {
        unsigned long s = (unsigned long)g_dvk_open;
        g_dvk_open = -1L;
        g_dvk_next = (s + 1UL) % DVK_RING;
        // the host may read what this command buffer wrote once its fence signals
        struct VkMemoryBarrier mb;
        mb.sType = DVK_STYPE_MEMORY_BARRIER; mb.pNext = (const void*)0;
        mb.srcAccessMask = DVK_ACCESS_SHADER_WRITE; mb.dstAccessMask = DVK_ACCESS_HOST;
        vkCmdPipelineBarrier(g_dvk_cb[s], DVK_STAGE_SHADER, DVK_STAGE_HOST, 0U, 1U, &mb, 0U, (const void*)0, 0U, (const void*)0);
        if (vkEndCommandBuffer(g_dvk_cb[s]) != 0) return dvk_check_(-1);
        struct VkSubmitInfo si;
        si.sType = DVK_STYPE_SUBMIT; si.pNext = (const void*)0;
        si.waitSemaphoreCount = 0U; si.pWaitSemaphores = (const void*)0; si.pWaitDstStageMask = (const unsigned int*)0;
        si.commandBufferCount = 1U; si.pCommandBuffers = &g_dvk_cb[s];
        si.signalSemaphoreCount = 0U; si.pSignalSemaphores = (const void*)0;
        int r = vkQueueSubmit(g_dvk_queue, 1U, &si, g_dvk_fence[s]);
        if (r == 0) g_dvk_pending[s] = 1;
        return dvk_check_(r);
    }
}

static int dvk_busy_() {
    if (g_dvk_open >= 0L) return 1;
    unsafe {
        unsigned long s = 0UL;
        while (s < DVK_RING) { if (g_dvk_pending[s]) return 1; s = s + 1UL; }
    }
    return 0;
}

// Submits and waits for everything queued; 0 if the GPU reported an error.
static int dvk_wait_() {
    int ok = dvk_commit_();
    unsafe {
        unsigned long s = 0UL;
        while (s < DVK_RING) { if (!dvk_wait_slot_(s)) ok = 0; s = s + 1UL; }
    }
    g_dvk_stage_used = 0UL;   // nothing reads the staging arena now
    return ok;
}

// One dispatch of (gx, gy, gz) workgroups with 'len' bytes of push
// constants, then a barrier, so the next one sees its writes (none where
// dispatches run in order: g_dvk_serial).
static int dvk_dispatch_(void* pipe, const void* push, unsigned long len, unsigned long gx, unsigned long gy, unsigned long gz) {
    if (!dvk_begin_()) return 0;
    unsafe {
        void* cb = g_dvk_cb[g_dvk_open];
        if (pipe != g_dvk_last_pipe) {
            vkCmdBindPipeline(cb, 1, pipe);   // VK_PIPELINE_BIND_POINT_COMPUTE
            g_dvk_last_pipe = pipe;
        }
        vkCmdPushConstants(cb, g_dvk_layout, DVK_STAGE_COMPUTE, 0U, (unsigned int)len, push);
        vkCmdDispatch(cb, (unsigned int)gx, (unsigned int)gy, (unsigned int)gz);
        if (!g_dvk_serial) {
            struct VkMemoryBarrier mb;
            mb.sType = DVK_STYPE_MEMORY_BARRIER; mb.pNext = (const void*)0;
            mb.srcAccessMask = DVK_ACCESS_SHADER_WRITE; mb.dstAccessMask = DVK_ACCESS_SHADER;
            vkCmdPipelineBarrier(cb, DVK_STAGE_SHADER, DVK_STAGE_SHADER, 0U, 1U, &mb, 0U, (const void*)0, 0U, (const void*)0);
        }
    }
    g_dvk_count = g_dvk_count + 1UL;
    if (g_dvk_count >= DVK_COMMIT_EVERY) return dvk_commit_();
    return 1;
}

static int g_dvk_copy_mod = 0;

// dst[i] = src[i] for n 32-bit words on the GPU (mk_copy), in order with the queued work.
static int dvk_copy_words_(unsigned long dst, unsigned long src, unsigned long n) {
    void* pipe = dvk_named_pipe_("mk_copy", &g_dvk_copy_mod);
    if (pipe == (void*)0) return 0;
    struct DevArgs a = dev_args_();
    unsafe {
        a.p[0] = src; a.p[2] = dst; a.u[7] = (unsigned int)n;
        unsigned long groups = (n + DVK_BLOCK - 1UL) / DVK_BLOCK;
        if (groups > DVK_MAX_GROUPS) groups = DVK_MAX_GROUPS;
        return dvk_dispatch_(pipe, (const void*)&a, sizeof(struct DevArgs), groups, 1UL, 1UL);
    }
}

// 'bytes' of the staging arena (16-byte aligned) for the queued work: its
// address, 0 if the arena is too full (waiting empties it) or too small.
static unsigned long dvk_stage_(unsigned long bytes) {
    unsigned long need = (bytes + 15UL) / 16UL * 16UL;
    if (need > DVK_STAGE_BYTES) return 0UL;
    if (g_dvk_stage_used + need > DVK_STAGE_BYTES && !dvk_wait_()) return 0UL;
    unsigned long at = g_dvk_stage.addr + g_dvk_stage_used;
    g_dvk_stage_used = g_dvk_stage_used + need;
    return at;
}

// GEMM pipelines (mk_gemm<tile>_<op(a)><op(b)>_<type>) and convolution ones (mk_conv_<op><shape>_<type>).
static int g_dvk_gemm_mod[24];
static int g_dvk_conv_mod[36];

// The GEMM kernel for g: 64 x 64 tiles when there are enough of them to fill
// the GPU (four times as many when b is transposed: its 16-deep tiles read
// b^T in short runs), else 32 x 32.
static int dvk_gemm_(const &DvkGemm_ g, unsigned long batch) {
    if (!dvk_ensure_()) return 0;
    if (g.m == 0U || g.n == 0U || batch == 0UL) return 1;
    unsigned long m = (unsigned long)g.m;
    unsigned long n = (unsigned long)g.n;
    unsigned long tile = 32UL;
    if (((m + 63UL) / 64UL) * ((n + 63UL) / 64UL) * batch >= (g.tb == 0U ? DVK_GEMM_FILL : 4UL * DVK_GEMM_FILL)) tile = 64UL;
    unsigned long slot = (tile == 64UL ? 12UL : 0UL) + (unsigned long)g.at * 4UL + (unsigned long)g.ta + 2UL * (unsigned long)g.tb;
    if (g.at > 2U || slot >= 24UL) return 0;
    unsafe {
        char name[32];
        const char* base = tile == 64UL ? "mk_gemm64_" : "mk_gemm32_";
        const char* sfx = g.at == 0U ? "f32" : (g.at == 1U ? "f16" : "bf16");
        unsigned long i = 0UL;
        while (base[i] != (char)0) { name[i] = base[i]; i = i + 1UL; }
        name[i] = g.ta ? 't' : 'n'; name[i + 1UL] = g.tb ? 't' : 'n'; name[i + 2UL] = '_';
        i = i + 3UL;
        unsigned long j = 0UL;
        while (sfx[j] != (char)0) { name[i] = sfx[j]; i = i + 1UL; j = j + 1UL; }
        name[i] = (char)0;
        void* pipe = dvk_named_pipe_((const char*)&name[0], &g_dvk_gemm_mod[slot]);
        if (pipe == (void*)0) { device_note_cpu_launch_(DEVICE_VULKAN); return 0; }   // the tensor op runs on the CPU
        return dvk_dispatch_(pipe, (const void*)g, sizeof(struct DvkGemm_), (n + tile - 1UL) / tile, (m + tile - 1UL) / tile, batch);
    }
}

// Conv kernel tile shapes (device_kernels.cu's mk_conv_*): rows x columns of the product per workgroup.
#define DVK_SHAPE_32 0   // 32 x 32
#define DVK_SHAPE_64 1   // 64 x 64
#define DVK_SHAPE_W  2   // 32 x 128
#define DVK_SHAPE_S  3   // 8 x 128: up to 8 rows

static unsigned long dvk_shape_rows_(int shape) { return shape == DVK_SHAPE_64 ? 64UL : (shape == DVK_SHAPE_S ? 8UL : 32UL); }
static unsigned long dvk_shape_cols_(int shape) { return shape == DVK_SHAPE_64 ? 64UL : (shape == DVK_SHAPE_32 ? 32UL : 128UL); }

// The shape for an m x n product (op 2: the weight gradient), as MpsBackend picks it.
static int dvk_conv_shape_(int op, unsigned long m, unsigned long n) {
    if (op == 2) return m >= 64UL ? DVK_SHAPE_64 : DVK_SHAPE_32;
    if (m <= 8UL && (n + 127UL) / 128UL >= DVK_GEMM_FILL) return DVK_SHAPE_S;
    if (((n + 127UL) / 128UL) * ((m + 31UL) / 32UL) >= DVK_GEMM_FILL) return DVK_SHAPE_W;
    return m >= 64UL && ((m + 63UL) / 64UL) * ((n + 63UL) / 64UL) >= DVK_GEMM_FILL ? DVK_SHAPE_64 : DVK_SHAPE_32;
}

// The pipeline for conv kernel op (0 fwd, 1 bwd_x, 2 bwd_w), a shape, type t.
static void* dvk_conv_pipe_(int op, int shape, int t) {
    unsigned long slot = ((unsigned long)op * 4UL + (unsigned long)shape) * 3UL + (unsigned long)t;
    unsafe {
        char name[40];
        const char* base = op == 0 ? "mk_conv_fwd" : (op == 1 ? "mk_conv_bwd_x" : "mk_conv_bwd_w");
        const char* sh = shape == DVK_SHAPE_32 ? "32_" : (shape == DVK_SHAPE_64 ? "64_" : (shape == DVK_SHAPE_W ? "w_" : "s_"));
        const char* sfx = t == 0 ? "f32" : (t == 1 ? "f16" : "bf16");
        unsigned long i = 0UL;
        while (base[i] != (char)0) { name[i] = base[i]; i = i + 1UL; }
        unsigned long j = 0UL;
        while (sh[j] != (char)0) { name[i] = sh[j]; i = i + 1UL; j = j + 1UL; }
        j = 0UL;
        while (sfx[j] != (char)0) { name[i] = sfx[j]; i = i + 1UL; j = j + 1UL; }
        name[i] = (char)0;
        return dvk_named_pipe_((const char*)&name[0], &g_dvk_conv_mod[slot]);
    }
}

// A kernel on the CPU (device_ref.sc) over the mapped memory, once the GPU is
// done; 0 (the op computes on the CPU through the tensor layer) for 16-bit
// arguments, a table of handles, or unmapped memory, which device_ref can't read.
static int dvk_cpu_launch_(int kernel, unsigned long n, const &DevArgs args, int tagged) {
    device_note_cpu_launch_(DEVICE_VULKAN);
    if (tagged || !g_dvk_mapped || kernel == DK_OPT_MULTI || kernel == DK_AMP_UNSCALE_MULTI) return 0;
    if (!dvk_wait_()) return 0;
    struct DevArgs h = *args;
    unsafe {
        unsigned long i = 0UL;
        while (i < (unsigned long)DEVK_PTRS) { h.p[i] = (unsigned long)dvk_host_(args.p[i]); i = i + 1UL; }
    }
    return device_ref_launch_(kernel, n, &h);
}

struct VulkanBackend {
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

int VulkanBackend::available() { return dvk_ensure_(); }

// A piece of a block with room (first fit), else of a new block.
unsigned long VulkanBackend::alloc(unsigned long bytes) {
    if (!dvk_ensure_()) return 0UL;
    unsigned long need = ((bytes == 0UL ? 1UL : bytes) + DVK_ALIGN - 1UL) / DVK_ALIGN * DVK_ALIGN;
    unsafe {
        if (!dvk_grow_((void**)&g_dvk_pieces, &g_dvk_cappieces, g_dvk_npieces, sizeof(struct DvkPiece_))) return 0UL;
        unsigned long k = 0UL;
        long off = -1L;
        while (k < g_dvk_nblocks && off < 0L) {
            off = dvk_take_(k, need);
            if (off < 0L) k = k + 1UL;
        }
        if (off < 0L) {
            if (!dvk_grow_((void**)&g_dvk_blocks, &g_dvk_capblocks, g_dvk_nblocks, sizeof(struct DvkBlock_))) return 0UL;
            struct DvkBlock_* bl = &g_dvk_blocks[g_dvk_nblocks];
            unsigned long size = need > DVK_BLOCK_BYTES ? need : DVK_BLOCK_BYTES;
            if (dvk_alloc_buf_(size, g_dvk_memtype, &bl->b) == 0UL) return 0UL;
            bl->free = (struct DvkRange_*)0; bl->nfree = 0UL; bl->capfree = 0UL; bl->live = 1UL;
            off = 0L;
            if (size > need) {
                if (!dvk_grow_((void**)&bl->free, &bl->capfree, 0UL, sizeof(struct DvkRange_))) return 0UL;
                bl->free[0].off = need; bl->free[0].bytes = size - need; bl->nfree = 1UL;
            }
            k = g_dvk_nblocks;
            g_dvk_nblocks = g_dvk_nblocks + 1UL;
        }
        struct DvkBlock_* b = &g_dvk_blocks[k];
        unsigned long addr = b->b.addr + (unsigned long)off;
        // insert the piece, keeping the table sorted by address
        unsigned long at = g_dvk_npieces;
        while (at > 0UL && g_dvk_pieces[at - 1UL].addr > addr) { g_dvk_pieces[at] = g_dvk_pieces[at - 1UL]; at = at - 1UL; }
        g_dvk_pieces[at].addr = addr;
        g_dvk_pieces[at].bytes = need;
        g_dvk_pieces[at].cpu = b->b.cpu == (unsigned char*)0 ? (unsigned char*)0 : b->b.cpu + (unsigned long)off;
        g_dvk_pieces[at].block = b->b.addr;
        g_dvk_npieces = g_dvk_npieces + 1UL;
        return addr;
    }
}

// The piece goes back to its block; a block left empty goes back to the
// driver (after the queued work, which may still use it).
void VulkanBackend::release(unsigned long handle) {
    if (handle == 0UL) return;
    long i = dvk_find_(handle);
    if (i < 0L) return;
    unsafe {
        struct DvkPiece_ pc = g_dvk_pieces[i];
        unsigned long j = (unsigned long)i;
        while (j + 1UL < g_dvk_npieces) { g_dvk_pieces[j] = g_dvk_pieces[j + 1UL]; j = j + 1UL; }
        g_dvk_npieces = g_dvk_npieces - 1UL;
        unsigned long k = 0UL;
        while (k < g_dvk_nblocks && g_dvk_blocks[k].b.addr != pc.block) k = k + 1UL;
        if (k == g_dvk_nblocks) return;
        dvk_give_(k, pc.addr - pc.block, pc.bytes);
        if (g_dvk_blocks[k].live > 0UL) return;
        if (dvk_busy_()) dvk_wait_();
        vkDestroyBuffer(g_dvk_dev, g_dvk_blocks[k].b.buf, (const void*)0);
        vkFreeMemory(g_dvk_dev, g_dvk_blocks[k].b.mem, (const void*)0);
        if (g_dvk_blocks[k].free != (struct DvkRange_*)0) free((void*)g_dvk_blocks[k].free);
        g_dvk_blocks[k] = g_dvk_blocks[g_dvk_nblocks - 1UL];
        g_dvk_nblocks = g_dvk_nblocks - 1UL;
    }
}

// Into idle mapped memory, a memcpy; while work is queued (or without
// mapped memory), through the staging arena and a copy in order with it,
// in pieces of at most half the arena.
int VulkanBackend::upload(unsigned long dst, const &float src, unsigned long n) {
    if (n == 0UL) return 1;
    if (!dvk_ensure_()) return 0;
    unsigned char* p = dvk_host_(dst);
    if (g_dvk_mapped && p == (unsigned char*)0) return 0;
    if (g_dvk_mapped && !dvk_busy_()) {
        unsafe { memcpy((void*)p, (const void*)src, n * 4UL); }
        return 1;
    }
    unsigned long done = 0UL;
    unsigned long piece = DVK_STAGE_BYTES / 8UL;   // floats: half the arena
    while (done < n) {
        unsigned long k = n - done < piece ? n - done : piece;
        unsigned long at = dvk_stage_(k * 4UL);
        if (at == 0UL) return 0;
        unsafe {
            memcpy((void*)(g_dvk_stage.cpu + (at - g_dvk_stage.addr)), (const void*)((const float*)src + done), k * 4UL);
        }
        if (!dvk_copy_words_(dst + done * 4UL, at, k)) return 0;
        done = done + k;
    }
    return 1;
}

int VulkanBackend::download(&float dst, unsigned long src, unsigned long n) {
    if (n == 0UL) return 1;
    if (!dvk_ensure_()) return 0;
    if (g_dvk_mapped) {
        unsigned char* p = dvk_host_(src);
        if (p == (unsigned char*)0 || !dvk_wait_()) return 0;
        unsafe { memcpy((void*)dst, (const void*)p, n * 4UL); }
        return 1;
    }
    // through the staging arena, a piece at a time
    unsigned long done = 0UL;
    unsigned long piece = DVK_STAGE_BYTES / 8UL;
    while (done < n) {
        unsigned long k = n - done < piece ? n - done : piece;
        unsigned long at = dvk_stage_(k * 4UL);
        if (at == 0UL || !dvk_copy_words_(at, src + done * 4UL, k) || !dvk_wait_()) return 0;
        unsafe { memcpy((void*)((float*)dst + done), (const void*)(g_dvk_stage.cpu + (at - g_dvk_stage.addr)), k * 4UL); }
        done = done + k;
    }
    return 1;
}

int VulkanBackend::copy(unsigned long dst, unsigned long src, unsigned long n) {
    if (n == 0UL) return 1;
    if (!dvk_ensure_()) return 0;
    if (dvk_copy_words_(dst, src, n)) return 1;
    device_note_cpu_launch_(DEVICE_VULKAN);
    return 0;
}

int VulkanBackend::sync() {
    if (!dvk_ensure_()) return 0;
    return dvk_wait_();
}

int VulkanBackend::gemm(int ta, int tb, unsigned long m, unsigned long n, unsigned long k, float alpha,
                        unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb,
                        float beta, unsigned long c, unsigned long ldc) {
    struct DvkGemm_ g;
    g.a = a; g.b = b; g.c = c; g.lda = lda; g.ldb = ldb; g.ldc = ldc; g.sa = 0UL; g.sb = 0UL; g.sc = 0UL;
    g.m = (unsigned int)m; g.n = (unsigned int)n; g.k = (unsigned int)k; g.ta = (unsigned int)ta; g.tb = (unsigned int)tb;
    g.at = 0U; g.ct = 0U; g.alpha = alpha; g.beta = beta;
    return dvk_gemm_(&g, 1UL);
}

int VulkanBackend::gemm_batched(const &DevGemm g) {
    struct DvkGemm_ r;
    r.a = g.a; r.b = g.b; r.c = g.c; r.lda = g.lda; r.ldb = g.ldb; r.ldc = g.ldc; r.sa = g.sa; r.sb = g.sb; r.sc = g.sc;
    r.m = (unsigned int)g.m; r.n = (unsigned int)g.n; r.k = (unsigned int)g.k;
    r.ta = (unsigned int)g.ta; r.tb = (unsigned int)g.tb; r.at = (unsigned int)g.dtype; r.ct = (unsigned int)g.ctype;
    r.alpha = g.alpha; r.beta = g.beta;
    return dvk_gemm_(&r, g.batch);
}

// Convolutions as implicit GEMMs (device_kernels.cu's mk_conv_*), as on
// Metal: the forward pass, the input gradient as a gather over the windows
// (no atomics), the weight gradient split over batch x pixels and added
// atomically into zeroed weights (fp32 only: a 16-bit one declines and
// device.sc runs it in fp32). A transposed convolution is the plain one with
// input and output swapped.
int VulkanBackend::conv(const &DevConv c) {
    if (!dvk_ensure_() || c.dtype < DEVICE_FP32 || c.dtype > DEVICE_BF16) return 0;
    struct DvkConv_ k;
    k.w = c.w; k.B = (unsigned int)c.batch;
    k.kH = (unsigned int)c.kH; k.kW = (unsigned int)c.kW; k.sH = (unsigned int)c.sH; k.sW = (unsigned int)c.sW;
    k.pH = (unsigned int)c.pH; k.pW = (unsigned int)c.pW; k.dH = (unsigned int)c.dH; k.dW = (unsigned int)c.dW;
    k.kSplit = 0U;
    if (!c.transpose) {
        k.x = c.x; k.y = c.y;
        k.Ci = (unsigned int)c.inC; k.Hi = (unsigned int)c.inH; k.Wi = (unsigned int)c.inW;
        k.Co = (unsigned int)c.outC; k.Ho = (unsigned int)c.outH; k.Wo = (unsigned int)c.outW;
    } else {
        k.x = c.y; k.y = c.x;
        k.Ci = (unsigned int)c.outC; k.Hi = (unsigned int)c.outH; k.Wi = (unsigned int)c.outW;
        k.Co = (unsigned int)c.inC; k.Ho = (unsigned int)c.inH; k.Wo = (unsigned int)c.inW;
    }
    int op = c.op == DEV_CONV_BWD_WEIGHTS ? 2 : ((c.op == DEV_CONV_FWD) == (c.transpose == 0) ? 0 : 1);
    if (op == 2 && c.dtype != DEVICE_FP32) return 0;
    unsigned long kk = (unsigned long)k.kH * (unsigned long)k.kW;
    unsigned long m = op == 1 ? (unsigned long)k.Ci : (unsigned long)k.Co;
    unsigned long n = op == 0 ? c.batch * (unsigned long)k.Ho * (unsigned long)k.Wo
                    : (op == 1 ? c.batch * (unsigned long)k.Hi * (unsigned long)k.Wi : (unsigned long)k.Ci * kk);
    if (m == 0UL || n == 0UL) return 1;
    // the input gradient runs per stride phase (without dilation): fewer pixels each
    unsigned long phases = 1UL;
    if (op == 1 && k.dH == 1U && k.dW == 1U) {
        phases = (unsigned long)k.sH * (unsigned long)k.sW;
        n = c.batch * (((unsigned long)k.Hi + (unsigned long)k.sH - 1UL) / (unsigned long)k.sH) *
            (((unsigned long)k.Wi + (unsigned long)k.sW - 1UL) / (unsigned long)k.sW);
    }
    int shape = dvk_conv_shape_(op, m, n * phases);
    void* pipe = dvk_conv_pipe_(op, shape, c.dtype);
    if (pipe == (void*)0) { device_note_cpu_launch_(DEVICE_VULKAN); return 0; }   // im2col + GEMM instead
    unsigned long groupsX = (n + dvk_shape_cols_(shape) - 1UL) / dvk_shape_cols_(shape);
    unsigned long groupsY = (m + dvk_shape_rows_(shape) - 1UL) / dvk_shape_rows_(shape);
    unsigned long splits = phases;
    if (op == 2) {
        // zero the weights, then enough slices of k to fill the GPU
        unsigned long nw = (unsigned long)k.Co * (unsigned long)k.Ci * kk;
        struct DevArgs z = dev_args_();
        unsafe { z.p[2] = c.w; z.u[0] = 5U; z.u[7] = (unsigned int)nw; }   // EW_FILL 0
        if (!VulkanBackend::launch(DK_EWOP_5, nw, &z)) return 0;
        unsigned long kTot = c.batch * (unsigned long)k.Ho * (unsigned long)k.Wo;
        splits = (128UL + groupsX * groupsY - 1UL) / (groupsX * groupsY);
        if (splits > kTot / 256UL) splits = kTot / 256UL;
        if (splits < 1UL) splits = 1UL;
        unsigned long per = (kTot + splits - 1UL) / splits;
        per = (per + 31UL) / 32UL * 32UL;
        k.kSplit = (unsigned int)per;
        splits = (kTot + per - 1UL) / per;
    }
    unsafe { return dvk_dispatch_(pipe, (const void*)&k, sizeof(struct DvkConv_), groupsX, groupsY, splits); }
}

int VulkanBackend::storage16() { return dvk_ensure_(); }

int VulkanBackend::launch(int kernel, unsigned long n, const &DevArgs args) {
    if (!dvk_ensure_() || kernel < 0 || kernel >= DK_COUNT) return 0;
    int tagged = 0;
    unsafe {
        unsigned long i = 0UL;
        while (i < (unsigned long)DEVK_PTRS) { if (dev_tag_(args.p[i]) != 0UL) tagged = 1; i = i + 1UL; }
    }
    int attn = kernel == DK_ATTN_FWD || kernel == DK_ATTN_BWD_Q || kernel == DK_ATTN_BWD_KV;
    unsigned long tile = DVK_ATTN_T;
    void* pipe = (void*)0;
    unsafe {
        if (attn && args.u[3] <= 64U && args.u[4] <= 64U) {
            pipe = dvk_pipe_(2 + tagged, kernel);
            tile = DVK_ATTN64_T;
        }
    }
    if (pipe == (void*)0) { pipe = dvk_pipe_(tagged, kernel); tile = DVK_ATTN_T; }
    if (pipe == (void*)0) return dvk_cpu_launch_(kernel, n, args, tagged);
    unsigned long groups = ((dvk_rows_(kernel) ? 32UL * n : n) + DVK_BLOCK - 1UL) / DVK_BLOCK;
    if (groups > DVK_MAX_GROUPS) groups = DVK_MAX_GROUPS;
    if (groups == 0UL) groups = 1UL;
    if (attn) {
        // a workgroup per (head, tile of rows): rows are keys for dk_attn_bwd_kv, else queries
        unsafe {
            unsigned long rows = kernel == DK_ATTN_BWD_KV ? (unsigned long)args.u[1] : (unsigned long)args.u[0];
            groups = (unsigned long)args.u[2] * ((rows + tile - 1UL) / tile);
        }
        if (groups > 65535UL) groups = 65535UL;
        if (groups == 0UL) groups = 1UL;
    }
    unsafe { return dvk_dispatch_(pipe, (const void*)args, sizeof(struct DevArgs), groups, 1UL, 1UL); }
}

unsigned long VulkanBackend::offset(unsigned long handle, unsigned long bytes) { return handle + bytes; }

void device_vulkan_register_() {
    struct VulkanBackend tag;
    tag.unused_ = 0;
    device_register_backend_(DEVICE_VULKAN, &tag);
}

} // namespace std
