// SafeC Standard Library — the WebGPU device backend (see device.h).
// WebGpuBackend: std::ml's tensor ops through the WebGPU C API (webgpu.h),
// on whatever the implementation runs on (Metal, Vulkan, D3D12). WGSL has no
// pointers, no 64-bit integers and no float atomics, so device_kernels.cu
// can't run here; device_kernels.wgsl implements the same DK_* kernels with
// the same arguments and semantics (gen_device_kernels_wgsl.sh checks the
// two against each other and embeds the WGSL in device_kernels_wgsl.h),
// plus what rocBLAS / MIOpen do on ROCm: GEMMs (wk_gemm1 for small products,
// wk_gemm and wk_gemm8 in tiles), convolutions as implicit GEMMs (wk_conv,
// through conv()) and copies (wk_copy).
//
// Memory is one storage buffer, the heap, bound to every kernel twice (as
// words and as atomic words: bind group 0, set once per compute pass). A
// handle is a byte offset into it, so a handle plus a byte offset is
// another handle and the optimizer's tables of handles work as they do on
// ROCm. A kernel's DevArgs travel as 32 words (offsets, their element types,
// u, f) at a 256-byte aligned dynamic offset of a uniform ring (bind group
// 1: rebinding only it per dispatch costs the GPU no more than not
// rebinding). The heap is a buddy allocator over power-of-two blocks from
// 256 bytes (device.sc's cache asks for those sizes anyway); it starts at
// 64 MiB and doubles when full, the GPU copying the old contents over, so
// handles stay valid.
//
// Work is encoded into one compute pass at a time and submitted every
// DWG_BATCH dispatches without waiting (WebGPU orders a pass's dispatches
// and makes each one's writes visible to the next); the host waits only
// when it reads (download, sync) and when DWG_FLIGHT command buffers are in
// flight. Uploads go straight to the queue (wgpuQueueWriteBuffer, ordered
// after the work submitted before them) unless the open command buffer has
// work: then a small one travels with it, in a ring of host data written
// once per submit, and a dispatch (wk_upload) writes it in order, as on
// Metal; a large one submits first. Downloads copy into a mappable buffer
// and map it.
//
// Linking: the webgpu feature (SAFEC_FEATURE_WEBGPU) registers the backend;
// the program links a WebGPU implementation, for wgpu-native (a release from
// https://github.com/gfx-rs/wgpu-native/releases, v24 or later: the webgpu.h
// revision with WGPUStringView and callback-info structs; webgpu_api.h's
// layouts are checked against v29), unpacked anywhere as <wgpu-native>:
//   -L<wgpu-native>/lib -lwgpu_native            the shared library, or
//   <wgpu-native>/lib/libwgpu_native.a -framework Metal -framework QuartzCore
//     -framework Foundation -lobjc               the static one (macOS; on
//                                                Linux add -ldl -lm -lpthread)
// In a safeguard project, the webgpu feature links -lwgpu_native, from
// $WGPU_HOME/lib when that is set (with an rpath to it); otherwise add
// lib_dirs = ["<wgpu-native>/lib"] under [build]. The shared
// library's install name is @rpath/libwgpu_native.dylib (libwgpu_native.so
// on Linux): the program finds it through an rpath (-Wl,-rpath,<dir>) or
// DYLD_LIBRARY_PATH / LD_LIBRARY_PATH. Nothing here hard-codes where it is
// installed; tests/stdlib/webgpu/run.sh takes WGPU_NATIVE_DIR.
//
// The host waits by running the instance's callbacks (work done, buffer
// mapped) with the device polled without blocking (wgpuDevicePoll: a
// blocking poll sleeps about a millisecond per wait on Metal). That and
// the kernels' module without wgpu's runtime checks
// (wgpuDeviceCreateShaderModuleTrusted, see dwg_module_) are the two
// wgpu-native extensions used. Dawn exports neither and checks the device
// in wgpuInstanceProcessEvents itself: build with -DSAFEC_WEBGPU_DAWN=1 and
// link -lwebgpu_dawn (not run here: no Dawn build on this machine).
//
// Limits, against ROCm / CUDA:
//   - Device memory is at most one storage binding: the largest power of two
//     within maxStorageBufferBindingSize and maxBufferSize, and 2 GiB (4 GiB
//     on an Apple GPU's Metal: 2 GiB is used). An allocation beyond it fails
//     (device.sc then frees its cache and retries; the tensor op runs on the
//     CPU if it still fails).
//   - No doubles and no float atomics in WGSL: batch / group norm statistics
//     accumulate in fp32 (two-pass, as on Metal) and atomic float adds are
//     compare-exchange loops (gradient scatters, col2im, pooling backward).
//   - No 16-bit stores: a kernel writing 16-bit storage writes element pairs
//     as whole words, or an element by two atomic operations on its word.
//   - The fused attention kernels work in 8 x 8 tiles for head dims over 64
//     (16 x 16 up to 64), as on Metal: 32 KB of workgroup memory.
//   - GEMMs and convolutions multiply on scalar FMAs: WGSL can't reach an
//     Apple GPU's simdgroup matrices (Metal's mk_gemm) or subgroups, so they
//     run at about half Metal's throughput (1.4 against 2.9 TFLOP/s for an
//     fp32 1024^3 GEMM on an M1 Pro), and 16-bit operands only save memory.
//   - A kernel runs on the CPU only if its pipeline can't be built; the
//     launch then declines and the tensor op computes on the CPU (counted in
//     DeviceStats.cpuLaunches), as do GEMMs whose grid exceeds 65535
//     workgroups a side. The heap's single binding means no CPU run of a
//     device_ref.sc kernel over device memory, unlike Metal's shared memory.
// safeguard: feature webgpu
// safeguard: implements <std/ml/device.h> (device_webgpu_register_, which device.h's link hook calls)
// safeguard: link hosted -L${WGPU_HOME}/lib -lwgpu_native
#pragma once
#include <std/stddef.h>
#include <std/ml/device.h>
#include <std/ml/device_kernels.h>
#include <std/ml/device_kernels_wgsl.h>
#include <std/ml/device_webgpu.h>
#include <std/ml/webgpu_api.h>

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);
extern void* memcpy(void* dst, const void* src, size_t n);
extern void* memset(void* dst, int c, size_t n);
extern long  write(int fd, const void* buf, unsigned long n);
extern char* getenv(const char* name);

#define DWG_BLOCK 256UL
#define DWG_MAX_GROUPS 4096UL
#define DWG_MAX_DIM 65535UL
#define DWG_ARG_WORDS 32UL
#define DWG_ARG_STRIDE 256UL          // the uniform buffer's dynamic offset alignment (WebGPU's largest)
#define DWG_BATCH 128UL               // dispatches per command buffer
#define DWG_FLIGHT 4UL                // command buffers in flight before the host waits for the oldest
#define DWG_UNIT_SHIFT 8UL            // heap blocks: powers of two from 256 bytes
#define DWG_HEAP_START (1UL << 26)    // 64 MiB at first
#define DWG_HEAP_LIMIT (1UL << 31)    // offsets stay below 2^31
#define DWG_DATA_BYTES (1UL << 20)    // per command buffer: the host data its uploads carry
#define DWG_INLINE_BYTES (1UL << 16)  // uploads up to this size travel with the open command buffer
#define DWG_ATTN_T 8UL                // device_kernels.wgsl's attention tile for head dims up to 128 ...
#define DWG_ATTN16_T 16UL             // ... and up to 64
#define DWG_GEMM_FILL 32UL            // the 128 x 128 GEMM tile from this many tiles on (two per GPU core)
#define DWG_POLL_CAP 1000000

static int   g_dwg_state = 0;          // 0 = not yet, 1 = ready, -1 = unusable
static void* g_dwg_inst = (void*)0;
static void* g_dwg_adapter = (void*)0;
static void* g_dwg_dev = (void*)0;
static void* g_dwg_queue = (void*)0;
static void* g_dwg_module = (void*)0;
static void* g_dwg_bgl[2];               // group 0: the heap; group 1: the arguments
static void* g_dwg_layout = (void*)0;
static void* g_dwg_bg = (void*)0;        // the heap's bind group (remade when it grows)
static void* g_dwg_abg = (void*)0;       // the arguments' (one dynamic offset per dispatch)
static unsigned long g_dwg_errors = 0UL;   // errors the device reported
static int g_dwg_lost = 0;                 // the device is lost: nothing more will finish
static int g_dwg_req_done = 0;
static void* g_dwg_req_result = (void*)0;

// The heap and its buddy allocator: per 256-byte unit, 0x80 | order for the
// first unit of an allocated block, 0x40 | order for a free one, else 0; a
// stack of free blocks per order (entries are checked against the tags when
// popped, so merged-away ones are skipped).
static void* g_dwg_heap = (void*)0;
static unsigned long g_dwg_heap_bytes = 0UL;
static unsigned long g_dwg_heap_max = 0UL;
static unsigned char* g_dwg_tag = (unsigned char*)0;
static unsigned long g_dwg_top = 0UL;            // the heap is 2^top units
static unsigned long* g_dwg_free[32];
static unsigned long g_dwg_nfree[32];
static unsigned long g_dwg_capfree[32];

// The command stream: the open encoder and compute pass, the arguments of
// its dispatches (host copy of its slot of the uniform ring), its uploads' data.
static void* g_dwg_args = (void*)0;              // uniform: DWG_FLIGHT slots of DWG_BATCH arguments
static unsigned char* g_dwg_args_host = (unsigned char*)0;
static void* g_dwg_data = (void*)0;              // DWG_FLIGHT slots of DWG_DATA_BYTES (group 0's D)
static unsigned char* g_dwg_data_host = (unsigned char*)0;
static void* g_dwg_rb = (void*)0;                // mappable readback buffer
static unsigned long g_dwg_rb_bytes = 0UL;
static void* g_dwg_enc = (void*)0;
static void* g_dwg_pass = (void*)0;
static void* g_dwg_last_pipe = (void*)0;
static unsigned long g_dwg_nargs = 0UL;          // dispatches in the open command buffer
static unsigned long g_dwg_pending = 0UL;        // commands (dispatches, copies) in it
static unsigned long g_dwg_data_used = 0UL;      // bytes of the open command buffer's data slot
static unsigned long g_dwg_slot = 0UL;
static unsigned long g_dwg_sub[DWG_FLIGHT];      // per slot: the serial of its last submission, 0 = none
static unsigned long g_dwg_serial = 0UL;         // the last submission's serial (from 1)
static unsigned long g_dwg_done = 0UL;           // every submission up to this serial is done
static int g_dwg_map_done = 0;
static int g_dwg_map_status = 0;

// Pipelines: the DK_* kernels (plain and TAGGED), the attention kernels'
// 16 x 16 builds, GEMMs per transposes and element types, the rest.
static void* g_dwg_pipe[2 * DK_COUNT];
static int   g_dwg_pipe_state[2 * DK_COUNT];     // 0 = not yet, 1 = built, -1 = can't
static void* g_dwg_apipe[6];
static int   g_dwg_apipe_state[6];
static void* g_dwg_lpipe[2 * DK_COUNT];          // the row kernels with 32 lanes a row
static int   g_dwg_lpipe_state[2 * DK_COUNT];
static void* g_dwg_gemm[108];            // wk_gemm / wk_gemm8 / wk_gemm1 per transposes and element types
static int   g_dwg_gemm_state[108];
static void* g_dwg_conv[27];             // wk_conv per op, tile shape and element type
static int   g_dwg_conv_state[27];
static void* g_dwg_copy = (void*)0;
static void* g_dwg_upload = (void*)0;
static void* g_dwg_map = (void*)0;

static unsigned long dwg_strlen_(const char* s) {
    unsigned long n = 0UL;
    unsafe { while (s[n] != (char)0) n = n + 1UL; }
    return n;
}

static void dwg_log_(const char* what, const char* msg, unsigned long len) {
    unsafe {
        write(2, (const void*)what, dwg_strlen_(what));
        if (msg != (const char*)0) {
            if (len == WGPU_STRLEN_) len = dwg_strlen_(msg);
            write(2, (const void*)msg, len);
        }
        write(2, (const void*)"\n", 1UL);
    }
}

static struct WGPUStringView dwg_sv_(const char* s) {
    struct WGPUStringView v;
    v.data = s;
    v.length = WGPU_STRLEN_;
    return v;
}

// ── Callbacks ────────────────────────────────────────────────────────────────
static void dwg_on_adapter_(int status, void* adapter, struct WGPUStringView msg, void* u1, void* u2) {
    g_dwg_req_result = status == WGPU_REQUEST_STATUS_SUCCESS ? adapter : (void*)0;
    g_dwg_req_done = 1;
}
static void dwg_on_device_(int status, void* device, struct WGPUStringView msg, void* u1, void* u2) {
    g_dwg_req_result = status == WGPU_REQUEST_STATUS_SUCCESS ? device : (void*)0;
    if (status != WGPU_REQUEST_STATUS_SUCCESS) dwg_log_("device_webgpu: no device: ", msg.data, msg.length);
    g_dwg_req_done = 1;
}
static void dwg_on_error_(const void* device, int type, struct WGPUStringView msg, void* u1, void* u2) {
    g_dwg_errors = g_dwg_errors + 1UL;
    dwg_log_("device_webgpu: ", msg.data, msg.length);
}
static void dwg_on_lost_(const void* device, int reason, struct WGPUStringView msg, void* u1, void* u2) {
    g_dwg_lost = 1;
    dwg_log_("device_webgpu: device lost: ", msg.data, msg.length);
}
static void dwg_on_map_(int status, struct WGPUStringView msg, void* u1, void* u2) {
    g_dwg_map_status = status;
    g_dwg_map_done = 1;
}

// ── Waiting ──────────────────────────────────────────────────────────────────
// The host waits by running the instance's callbacks until a work-done or
// map callback has fired, polling the device without blocking: a blocking
// wgpuDevicePoll sleeps in steps of about a millisecond on Metal, against
// a quarter of one for a readback this way.
static void dwg_on_done_(int status, struct WGPUStringView msg, void* u1, void* u2) {
    unsigned long s = (unsigned long)u1;
    if (s > g_dwg_done) g_dwg_done = s;
}

// One round of the instance's callbacks. wgpu-native first needs the device
// polled (wgpu.h's wgpuDevicePoll, without blocking); Dawn's
// wgpuInstanceProcessEvents checks the device itself.
static void dwg_tick_() {
    unsafe {
#if !defined(SAFEC_WEBGPU_DAWN)
        wgpuDevicePoll(g_dwg_dev, 0U, (const unsigned long*)0);
#endif
        wgpuInstanceProcessEvents(g_dwg_inst);
    }
}

// Submits a command buffer; returns its serial, which dwg_wait_for_ waits on.
static unsigned long dwg_queue_submit_(void* cb) {
    g_dwg_serial = g_dwg_serial + 1UL;
    unsafe {
        wgpuQueueSubmit(g_dwg_queue, 1UL, (void* const*)&cb);
        struct WGPUQueueWorkDoneCallbackInfo info;
        memset((void*)&info, 0, sizeof(struct WGPUQueueWorkDoneCallbackInfo));
        info.mode = WGPU_CALLBACK_MODE_ALLOW_PROCESS_EVENTS;
        info.callback = dwg_on_done_;
        info.userdata1 = (void*)g_dwg_serial;
        wgpuQueueOnSubmittedWorkDone(g_dwg_queue, info);
    }
    return g_dwg_serial;
}

// Returns once submission s (and every one before it) is done, or the
// device is lost.
static void dwg_wait_for_(unsigned long s) {
    while (g_dwg_done < s && !g_dwg_lost) dwg_tick_();
}

// Maps the readback buffer's first 'bytes' and waits for it; 1 on success.
static int dwg_map_readback_(unsigned long bytes) {
    g_dwg_map_done = 0;
    g_dwg_map_status = 0;
    unsafe {
        struct WGPUBufferMapCallbackInfo info;
        memset((void*)&info, 0, sizeof(struct WGPUBufferMapCallbackInfo));
        info.mode = WGPU_CALLBACK_MODE_ALLOW_PROCESS_EVENTS;
        info.callback = dwg_on_map_;
        wgpuBufferMapAsync(g_dwg_rb, WGPU_MAP_MODE_READ, 0UL, bytes, info);
    }
    while (!g_dwg_map_done && !g_dwg_lost) dwg_tick_();
    return g_dwg_map_done && g_dwg_map_status == WGPU_REQUEST_STATUS_SUCCESS;
}

// Runs the instance's callbacks until a request's has fired.
static void* dwg_await_request_() {
    int iter = 0;
    unsafe {
        while (!g_dwg_req_done && iter < DWG_POLL_CAP) {
            wgpuInstanceProcessEvents(g_dwg_inst);
            iter = iter + 1;
        }
    }
    return g_dwg_req_done ? g_dwg_req_result : (void*)0;
}

// ── Buffers ──────────────────────────────────────────────────────────────────
static void* dwg_buffer_(unsigned long bytes, unsigned long usage) {
    unsafe {
        struct WGPUBufferDescriptor d;
        memset((void*)&d, 0, sizeof(struct WGPUBufferDescriptor));
        d.label = dwg_sv_("safec");
        d.usage = usage;
        d.size = bytes;
        unsigned long before = g_dwg_errors;
        void* b = wgpuDeviceCreateBuffer(g_dwg_dev, &d);
        if (b != (void*)0 && g_dwg_errors != before) { wgpuBufferRelease(b); return (void*)0; }
        return b;
    }
}

// (Re)binds the heap: group 0 of every kernel, as words and as atomic words,
// with the uploads' data ring.
static int dwg_bind_() {
    unsafe {
        struct WGPUBindGroupEntry e[3];
        memset((void*)&e[0], 0, sizeof(struct WGPUBindGroupEntry) * 3UL);
        e[0].binding = 0U; e[0].buffer = g_dwg_heap; e[0].offset = 0UL; e[0].size = g_dwg_heap_bytes;
        e[1].binding = 1U; e[1].buffer = g_dwg_heap; e[1].offset = 0UL; e[1].size = g_dwg_heap_bytes;
        e[2].binding = 2U; e[2].buffer = g_dwg_data; e[2].offset = 0UL; e[2].size = DWG_FLIGHT * DWG_DATA_BYTES;
        struct WGPUBindGroupDescriptor d;
        memset((void*)&d, 0, sizeof(struct WGPUBindGroupDescriptor));
        d.label = dwg_sv_("safec heap");
        d.layout = g_dwg_bgl[0];
        d.entryCount = 3UL;
        d.entries = (const struct WGPUBindGroupEntry*)&e[0];
        unsigned long before = g_dwg_errors;
        void* bg = wgpuDeviceCreateBindGroup(g_dwg_dev, &d);
        if (bg == (void*)0 || g_dwg_errors != before) return 0;
        if (g_dwg_bg != (void*)0) wgpuBindGroupRelease(g_dwg_bg);
        g_dwg_bg = bg;
    }
    return 1;
}

// Binds the arguments' ring: group 1, one dynamic offset per dispatch (only
// this group changes between dispatches).
static int dwg_bind_args_() {
    unsafe {
        struct WGPUBindGroupEntry e;
        memset((void*)&e, 0, sizeof(struct WGPUBindGroupEntry));
        e.binding = 0U; e.buffer = g_dwg_args; e.offset = 0UL; e.size = DWG_ARG_WORDS * 4UL;
        struct WGPUBindGroupDescriptor d;
        memset((void*)&d, 0, sizeof(struct WGPUBindGroupDescriptor));
        d.label = dwg_sv_("safec arguments");
        d.layout = g_dwg_bgl[1];
        d.entryCount = 1UL;
        d.entries = (const struct WGPUBindGroupEntry*)&e;
        unsigned long before = g_dwg_errors;
        g_dwg_abg = wgpuDeviceCreateBindGroup(g_dwg_dev, &d);
        return g_dwg_abg != (void*)0 && g_dwg_errors == before;
    }
}

static void dwg_push_free_(unsigned long order, unsigned long unit) {
    unsafe {
        if (g_dwg_nfree[order] == g_dwg_capfree[order]) {
            unsigned long cap = g_dwg_capfree[order] == 0UL ? 64UL : g_dwg_capfree[order] * 2UL;
            unsigned long* a = (unsigned long*)malloc(sizeof(unsigned long) * cap);
            if (a == (unsigned long*)0) return;   // the block is lost to the allocator, not to correctness
            if (g_dwg_nfree[order] > 0UL) memcpy((void*)a, (const void*)g_dwg_free[order], sizeof(unsigned long) * g_dwg_nfree[order]);
            if (g_dwg_free[order] != (unsigned long*)0) free((void*)g_dwg_free[order]);
            g_dwg_free[order] = a;
            g_dwg_capfree[order] = cap;
        }
        g_dwg_free[order][g_dwg_nfree[order]] = unit;
        g_dwg_nfree[order] = g_dwg_nfree[order] + 1UL;
        g_dwg_tag[unit] = (unsigned char)(0x40UL | order);
    }
}

// A free block of exactly 'order', split from a larger one if needed; -1 if none.
static long dwg_take_(unsigned long order) {
    unsigned long k = order;
    unsafe {
        while (k <= g_dwg_top) {
            while (g_dwg_nfree[k] > 0UL) {
                g_dwg_nfree[k] = g_dwg_nfree[k] - 1UL;
                unsigned long u = g_dwg_free[k][g_dwg_nfree[k]];
                if (g_dwg_tag[u] != (unsigned char)(0x40UL | k)) continue;   // merged or taken since
                // split down to 'order', freeing the upper halves
                while (k > order) {
                    k = k - 1UL;
                    dwg_push_free_(k, u + (1UL << k));
                }
                g_dwg_tag[u] = (unsigned char)(0x80UL | order);
                return (long)u;
            }
            k = k + 1UL;
        }
    }
    return -1L;
}

static void dwg_put_(unsigned long unit) {
    unsafe {
        unsigned long order = (unsigned long)g_dwg_tag[unit] & 0x3fUL;
        g_dwg_tag[unit] = (unsigned char)0;
        while (order < g_dwg_top) {
            unsigned long buddy = unit ^ (1UL << order);
            if (g_dwg_tag[buddy] != (unsigned char)(0x40UL | order)) break;
            g_dwg_tag[buddy] = (unsigned char)0;
            if (buddy < unit) unit = buddy;
            order = order + 1UL;
        }
        dwg_push_free_(order, unit);
    }
}

// ── The command stream ───────────────────────────────────────────────────────
static int dwg_encoder_() {
    if (g_dwg_enc != (void*)0) return 1;
    unsafe { g_dwg_enc = wgpuDeviceCreateCommandEncoder(g_dwg_dev, (const void*)0); }
    return g_dwg_enc != (void*)0;
}

static void dwg_end_pass_() {
    if (g_dwg_pass == (void*)0) return;
    unsafe {
        wgpuComputePassEncoderEnd(g_dwg_pass);
        wgpuComputePassEncoderRelease(g_dwg_pass);
    }
    g_dwg_pass = (void*)0;
}

static int dwg_pass_() {
    if (g_dwg_pass != (void*)0) return 1;
    if (!dwg_encoder_()) return 0;
    unsafe {
        g_dwg_pass = wgpuCommandEncoderBeginComputePass(g_dwg_enc, (const void*)0);
        if (g_dwg_pass == (void*)0) return 0;
        wgpuComputePassEncoderSetBindGroup(g_dwg_pass, 0U, g_dwg_bg, 0UL, (const unsigned int*)0);
    }
    g_dwg_last_pipe = (void*)0;
    return 1;
}

// Submits the open command buffer (no wait); the next one takes the next
// slot of the argument and data rings, once that slot's last command
// buffer is done (which bounds how far the host runs ahead).
static void dwg_submit_() {
    if (g_dwg_enc == (void*)0) return;
    dwg_end_pass_();
    unsafe {
        void* cb = wgpuCommandEncoderFinish(g_dwg_enc, (const void*)0);
        wgpuCommandEncoderRelease(g_dwg_enc);
        g_dwg_enc = (void*)0;
        if (g_dwg_nargs > 0UL) {
            unsigned long off = g_dwg_slot * DWG_BATCH * DWG_ARG_STRIDE;
            wgpuQueueWriteBuffer(g_dwg_queue, g_dwg_args, off, (const void*)(g_dwg_args_host + off), g_dwg_nargs * DWG_ARG_STRIDE);
        }
        if (g_dwg_data_used > 0UL) {
            unsigned long off = g_dwg_slot * DWG_DATA_BYTES;
            wgpuQueueWriteBuffer(g_dwg_queue, g_dwg_data, off, (const void*)(g_dwg_data_host + off), g_dwg_data_used);
        }
        unsigned long s = dwg_queue_submit_(cb);
        wgpuCommandBufferRelease(cb);
        g_dwg_sub[g_dwg_slot] = s;
        g_dwg_slot = (g_dwg_slot + 1UL) % DWG_FLIGHT;
        g_dwg_nargs = 0UL;
        g_dwg_pending = 0UL;
        g_dwg_data_used = 0UL;
        if (g_dwg_sub[g_dwg_slot] != 0UL) {
            dwg_wait_for_(g_dwg_sub[g_dwg_slot]);
            g_dwg_sub[g_dwg_slot] = 0UL;
        }
    }
}

// Submits and waits for everything queued; 0 if the device reported an error.
static int dwg_wait_() {
    dwg_submit_();
    dwg_wait_for_(g_dwg_serial);
    return g_dwg_errors == 0UL && !g_dwg_lost;
}

// One dispatch of (gx, gy, gz) workgroups with 'words' (32) as its arguments.
static int dwg_dispatch_(void* pipe, const unsigned int* words, unsigned long gx, unsigned long gy, unsigned long gz) {
    if (gx == 0UL || gy == 0UL || gz == 0UL) return 1;
    if (!dwg_pass_()) return 0;
    unsafe {
        unsigned long off = (g_dwg_slot * DWG_BATCH + g_dwg_nargs) * DWG_ARG_STRIDE;
        memcpy((void*)(g_dwg_args_host + off), (const void*)words, DWG_ARG_WORDS * 4UL);
        if (pipe != g_dwg_last_pipe) {
            wgpuComputePassEncoderSetPipeline(g_dwg_pass, pipe);
            g_dwg_last_pipe = pipe;
        }
        unsigned int doff = (unsigned int)off;
        wgpuComputePassEncoderSetBindGroup(g_dwg_pass, 1U, g_dwg_abg, 1UL, (const unsigned int*)&doff);
        wgpuComputePassEncoderDispatchWorkgroups(g_dwg_pass, (unsigned int)gx, (unsigned int)gy, (unsigned int)gz);
    }
    g_dwg_nargs = g_dwg_nargs + 1UL;
    g_dwg_pending = g_dwg_pending + 1UL;
    if (g_dwg_nargs == DWG_BATCH) dwg_submit_();
    return 1;
}

// Records a buffer-to-buffer copy in order with the queued dispatches.
static int dwg_copy_buffers_(void* src, unsigned long so, void* dst, unsigned long dof, unsigned long bytes) {
    dwg_end_pass_();
    if (!dwg_encoder_()) return 0;
    unsafe { wgpuCommandEncoderCopyBufferToBuffer(g_dwg_enc, src, so, dst, dof, bytes); }
    g_dwg_pending = g_dwg_pending + 1UL;
    return 1;
}

// Doubles the heap (or more, to fit a block of 'order') up to its limit: a
// new buffer, the old contents copied over by the GPU in order with the
// queued work, the old buffer released once that is done.
static int dwg_grow_(unsigned long order) {
    unsigned long need = (1UL << order) << DWG_UNIT_SHIFT;
    unsigned long bytes = g_dwg_heap_bytes;
    while (bytes < g_dwg_heap_max && (bytes - g_dwg_heap_bytes < need || bytes == g_dwg_heap_bytes)) bytes = bytes * 2UL;
    if (bytes <= g_dwg_heap_bytes || bytes > g_dwg_heap_max) return 0;
    unsigned long units = bytes >> DWG_UNIT_SHIFT;
    unsafe {
        unsigned char* tag = (unsigned char*)malloc(units);
        if (tag == (unsigned char*)0) return 0;
        void* nh = dwg_buffer_(bytes, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_SRC | WGPU_BUFFER_USAGE_COPY_DST);
        if (nh == (void*)0) { free((void*)tag); return 0; }
        if (!dwg_copy_buffers_(g_dwg_heap, 0UL, nh, 0UL, g_dwg_heap_bytes)) {
            wgpuBufferRelease(nh);
            free((void*)tag);
            return 0;
        }
        dwg_submit_();   // what is queued still binds the old heap
        void* old = g_dwg_heap;
        unsigned long oldUnits = g_dwg_heap_bytes >> DWG_UNIT_SHIFT;
        g_dwg_heap = nh;
        g_dwg_heap_bytes = bytes;
        if (!dwg_bind_()) return 0;
        wgpuBufferRelease(old);   // kept alive by the GPU until the copy has run
        memcpy((void*)tag, (const void*)g_dwg_tag, oldUnits);
        memset((void*)(tag + oldUnits), 0, units - oldUnits);
        free((void*)g_dwg_tag);
        g_dwg_tag = tag;
        // the new space: free blocks of orders top, top + 1, ... above the old heap
        unsigned long top = g_dwg_top;
        while ((1UL << top) < units) {
            dwg_push_free_(top, 1UL << top);
            top = top + 1UL;
        }
        g_dwg_top = top;
    }
    return 1;
}

// ── Pipelines ────────────────────────────────────────────────────────────────
static struct WGPUConstantEntry dwg_const_(const char* key, double value) {
    struct WGPUConstantEntry c;
    unsafe { memset((void*)&c, 0, sizeof(struct WGPUConstantEntry)); }
    c.key = dwg_sv_(key);
    c.value = value;
    return c;
}

// The pipeline for entry point 'entry' with 'n' pipeline constants, null if
// it can't be built (logged).
static void* dwg_make_pipe_(const char* entry, const struct WGPUConstantEntry* consts, unsigned long n) {
    unsafe {
        struct WGPUComputePipelineDescriptor d;
        memset((void*)&d, 0, sizeof(struct WGPUComputePipelineDescriptor));
        d.label = dwg_sv_(entry);
        d.layout = g_dwg_layout;
        d.compute.module = g_dwg_module;
        d.compute.entryPoint = dwg_sv_(entry);
        d.compute.constantCount = n;
        d.compute.constants = consts;
        unsigned long before = g_dwg_errors;
        void* p = wgpuDeviceCreateComputePipeline(g_dwg_dev, &d);
        if (g_dwg_errors != before || p == (void*)0) {
            // an invalid pipeline must never reach a pass: its submit would fail
            if (p != (void*)0) wgpuComputePipelineRelease(p);
            g_dwg_errors = before;
            dwg_log_("device_webgpu: can't build the pipeline for ", entry, WGPU_STRLEN_);
            return (void*)0;
        }
        return p;
    }
}

static int dwg_is_attn_(int kernel) { return kernel == DK_ATTN_FWD || kernel == DK_ATTN_BWD_Q || kernel == DK_ATTN_BWD_KV; }

// The kernels with a row per RL lanes (device_kernels.wgsl): u0 is the row length.
static int dwg_rows_(int kernel) {
    return kernel == DK_ROWSUM || kernel == DK_SOFTMAX_ROWS || kernel == DK_SOFTMAX_BWD || kernel == DK_CE_FWD ||
           kernel == DK_CE_BWD || kernel == DK_LAYERNORM || kernel == DK_LAYERNORM_BWD;
}

// DK_* kernel 'kernel' (TAGGED when 'tagged'), its 16 x 16 attention build
// when 'wide', with 32 lanes a row when 'lanes'; built on first use.
static void* dwg_pipe_(int kernel, int tagged, int wide, int lanes) {
    unsafe {
        void** cache = &g_dwg_pipe[0];
        int* state = &g_dwg_pipe_state[0];
        unsigned long slot = (unsigned long)tagged * (unsigned long)DK_COUNT + (unsigned long)kernel;
        if (lanes) { cache = &g_dwg_lpipe[0]; state = &g_dwg_lpipe_state[0]; }
        else if (wide) {
            cache = &g_dwg_apipe[0];
            state = &g_dwg_apipe_state[0];
            slot = (unsigned long)tagged * 3UL + (unsigned long)(kernel - DK_ATTN_FWD);
        }
        if (state[slot] != 0) return cache[slot];
        struct WGPUConstantEntry c[4];
        unsigned long n = 0UL;
        if (tagged) { c[n] = dwg_const_("TAGGED", 1.0); n = n + 1UL; }
        int opc = device_kernel_wgsl_opc_(kernel);
        if (opc >= 0) { c[n] = dwg_const_("OPC", (double)opc); n = n + 1UL; }
        if (lanes) { c[n] = dwg_const_("RL", 32.0); n = n + 1UL; }
        if (wide) {
            c[n] = dwg_const_("ATT_T", (double)DWG_ATTN16_T); n = n + 1UL;
            c[n] = dwg_const_("ATT_MAXD", 64.0); n = n + 1UL;
        }
        cache[slot] = dwg_make_pipe_(device_kernel_wgsl_entry_(kernel), (const struct WGPUConstantEntry*)&c[0], n);
        state[slot] = cache[slot] != (void*)0 ? 1 : -1;
        return cache[slot];
    }
}

static void* dwg_gemm_pipe_(int big, int ta, int tb, int at, int ct) {
    unsigned long slot = ((((unsigned long)big * 2UL + (unsigned long)ta) * 2UL + (unsigned long)tb) * 3UL + (unsigned long)at) * 3UL +
                         (unsigned long)ct;
    unsafe {
        if (g_dwg_gemm_state[slot] == 0) {
            struct WGPUConstantEntry c[4];
            c[0] = dwg_const_("GTA", (double)ta);
            c[1] = dwg_const_("GTB", (double)tb);
            c[2] = dwg_const_("GAT", (double)at);
            c[3] = dwg_const_("GCT", (double)ct);
            g_dwg_gemm[slot] = dwg_make_pipe_(big == 2 ? "wk_gemm1" : (big ? "wk_gemm8" : "wk_gemm"),
                                              (const struct WGPUConstantEntry*)&c[0], 4UL);
            g_dwg_gemm_state[slot] = g_dwg_gemm[slot] != (void*)0 ? 1 : -1;
        }
        return g_dwg_gemm[slot];
    }
}

// wk_conv's tile shapes: rows x columns of the product.
#define DWG_CONV_64 0     // 64 x 64
#define DWG_CONV_32 1     // 32 x 128
#define DWG_CONV_16 2     // 16 x 256
static unsigned long dwg_conv_rows_(int shape) { return shape == DWG_CONV_64 ? 64UL : (shape == DWG_CONV_32 ? 32UL : 16UL); }
static unsigned long dwg_conv_cols_(int shape) { return 4096UL / dwg_conv_rows_(shape); }

// wk_conv for op (0 forward, 1 input gradient, 2 weight gradient), a tile
// shape and element type t.
static void* dwg_conv_pipe_(int op, int shape, int t) {
    unsigned long slot = ((unsigned long)op * 3UL + (unsigned long)shape) * 3UL + (unsigned long)t;
    unsafe {
        if (g_dwg_conv_state[slot] == 0) {
            struct WGPUConstantEntry c[3];
            c[0] = dwg_const_("COP", (double)op);
            c[1] = dwg_const_("CBM", (double)dwg_conv_rows_(shape));
            c[2] = dwg_const_("CT", (double)t);
            g_dwg_conv[slot] = dwg_make_pipe_("wk_conv", (const struct WGPUConstantEntry*)&c[0], 3UL);
            g_dwg_conv_state[slot] = g_dwg_conv[slot] != (void*)0 ? 1 : -1;
        }
        return g_dwg_conv[slot];
    }
}

// The kernels' module. wgpu adds bounds checks to every buffer access and a
// counter to every loop of a shader (a web sandbox's guard against reads
// past a binding and hung GPUs; CUDA, HIP and Metal kernels have neither):
// a third of a small kernel's time on an M1 Pro. The kernels index only
// the heap and terminate, so on wgpu-native the module is created without
// them (its extension wgpuDeviceCreateShaderModuleTrusted);
// SAFEC_WEBGPU_CHECKS=1 in the environment keeps them, for debugging a
// kernel. Dawn builds keep them (its toggle to drop them,
// disable_robustness, would go in a DawnTogglesDescriptor here).
static void* dwg_module_(const struct WGPUShaderModuleDescriptor* md) {
    unsafe {
#if defined(SAFEC_WEBGPU_DAWN)
        return wgpuDeviceCreateShaderModule(g_dwg_dev, md);
#else
        char* keep = getenv("SAFEC_WEBGPU_CHECKS");
        if (keep != (char*)0 && keep[0] == '1') return wgpuDeviceCreateShaderModule(g_dwg_dev, md);
        return wgpuDeviceCreateShaderModuleTrusted(g_dwg_dev, md, 0UL);
#endif
    }
}

// ── Setup ────────────────────────────────────────────────────────────────────
static int dwg_ensure_() {
    if (g_dwg_state != 0) return g_dwg_state == 1;
    g_dwg_state = -1;
    unsafe {
        g_dwg_inst = wgpuCreateInstance((const struct WGPUInstanceDescriptor*)0);
        if (g_dwg_inst == (void*)0) return 0;
        struct WGPURequestAdapterOptions opts;
        memset((void*)&opts, 0, sizeof(struct WGPURequestAdapterOptions));
        opts.powerPreference = WGPU_POWER_HIGH_PERFORMANCE;
        struct WGPURequestAdapterCallbackInfo ai;
        memset((void*)&ai, 0, sizeof(struct WGPURequestAdapterCallbackInfo));
        ai.mode = WGPU_CALLBACK_MODE_ALLOW_PROCESS_EVENTS;
        ai.callback = dwg_on_adapter_;
        g_dwg_req_done = 0;
        wgpuInstanceRequestAdapter(g_dwg_inst, &opts, ai);
        g_dwg_adapter = dwg_await_request_();
        if (g_dwg_adapter == (void*)0) return 0;

        // every limit the adapter offers (the attention tiles need 17 KB of
        // workgroup memory, the heap a large storage binding)
        struct WGPULimits lim;
        memset((void*)&lim, 0, sizeof(struct WGPULimits));
        if (wgpuAdapterGetLimits(g_dwg_adapter, &lim) != WGPU_STATUS_SUCCESS) return 0;
        if (lim.minUniformBufferOffsetAlignment > (unsigned int)DWG_ARG_STRIDE || lim.maxStorageBuffersPerShaderStage < 2U) return 0;
        struct WGPUDeviceDescriptor dd;
        memset((void*)&dd, 0, sizeof(struct WGPUDeviceDescriptor));
        dd.label = dwg_sv_("safec");
        dd.requiredLimits = (const struct WGPULimits*)&lim;
        dd.uncapturedErrorCallbackInfo.callback = dwg_on_error_;
        dd.deviceLostCallbackInfo.mode = WGPU_CALLBACK_MODE_ALLOW_PROCESS_EVENTS;
        dd.deviceLostCallbackInfo.callback = dwg_on_lost_;
        struct WGPURequestDeviceCallbackInfo di;
        memset((void*)&di, 0, sizeof(struct WGPURequestDeviceCallbackInfo));
        di.mode = WGPU_CALLBACK_MODE_ALLOW_PROCESS_EVENTS;
        di.callback = dwg_on_device_;
        g_dwg_req_done = 0;
        wgpuAdapterRequestDevice(g_dwg_adapter, &dd, di);
        g_dwg_dev = dwg_await_request_();
        if (g_dwg_dev == (void*)0) return 0;
        g_dwg_queue = wgpuDeviceGetQueue(g_dwg_dev);
        if (g_dwg_queue == (void*)0) return 0;

        // the kernels
        struct WGPUShaderSourceWGSL src;
        memset((void*)&src, 0, sizeof(struct WGPUShaderSourceWGSL));
        src.chain.sType = WGPU_STYPE_SHADER_SOURCE_WGSL;
        src.code = dwg_sv_(device_kernel_wgsl_source_());
        struct WGPUShaderModuleDescriptor md;
        memset((void*)&md, 0, sizeof(struct WGPUShaderModuleDescriptor));
        md.nextInChain = (void*)&src;
        md.label = dwg_sv_("device_kernels.wgsl");
        g_dwg_module = dwg_module_(&md);
        if (g_dwg_module == (void*)0 || g_dwg_errors != 0UL) return 0;

        // group 0: the heap as words and as atomics, the uploads' data; group 1: the arguments (dynamic offset)
        struct WGPUBindGroupLayoutEntry le[4];
        memset((void*)&le[0], 0, sizeof(struct WGPUBindGroupLayoutEntry) * 4UL);
        unsigned long i = 0UL;
        while (i < 4UL) {
            le[i].binding = i == 3UL ? 0U : (unsigned int)i;
            le[i].visibility = WGPU_SHADER_STAGE_COMPUTE;
            le[i].buffer.type = i == 3UL ? WGPU_BUFFER_BINDING_UNIFORM : (i == 2UL ? WGPU_BUFFER_BINDING_READ_ONLY_STORAGE : WGPU_BUFFER_BINDING_STORAGE);
            i = i + 1UL;
        }
        le[3].buffer.hasDynamicOffset = 1U;
        le[3].buffer.minBindingSize = DWG_ARG_WORDS * 4UL;
        struct WGPUBindGroupLayoutDescriptor ld;
        memset((void*)&ld, 0, sizeof(struct WGPUBindGroupLayoutDescriptor));
        ld.entryCount = 3UL;
        ld.entries = (const struct WGPUBindGroupLayoutEntry*)&le[0];
        g_dwg_bgl[0] = wgpuDeviceCreateBindGroupLayout(g_dwg_dev, &ld);
        ld.entryCount = 1UL;
        ld.entries = (const struct WGPUBindGroupLayoutEntry*)&le[3];
        g_dwg_bgl[1] = wgpuDeviceCreateBindGroupLayout(g_dwg_dev, &ld);
        struct WGPUPipelineLayoutDescriptor pd;
        memset((void*)&pd, 0, sizeof(struct WGPUPipelineLayoutDescriptor));
        pd.bindGroupLayoutCount = 2UL;
        pd.bindGroupLayouts = (const void* const*)&g_dwg_bgl[0];
        g_dwg_layout = wgpuDeviceCreatePipelineLayout(g_dwg_dev, &pd);
        if (g_dwg_bgl[0] == (void*)0 || g_dwg_bgl[1] == (void*)0 || g_dwg_layout == (void*)0 || g_dwg_errors != 0UL) return 0;

        // the heap: the largest power of two the bindings allow
        unsigned long cap = lim.maxStorageBufferBindingSize < lim.maxBufferSize ? lim.maxStorageBufferBindingSize : lim.maxBufferSize;
        if (cap > DWG_HEAP_LIMIT) cap = DWG_HEAP_LIMIT;
        g_dwg_heap_max = 1UL << DWG_UNIT_SHIFT;
        while (g_dwg_heap_max * 2UL <= cap) g_dwg_heap_max = g_dwg_heap_max * 2UL;
        g_dwg_heap_bytes = DWG_HEAP_START < g_dwg_heap_max ? DWG_HEAP_START : g_dwg_heap_max;
        g_dwg_heap = dwg_buffer_(g_dwg_heap_bytes, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_SRC | WGPU_BUFFER_USAGE_COPY_DST);
        g_dwg_args = dwg_buffer_(DWG_FLIGHT * DWG_BATCH * DWG_ARG_STRIDE, WGPU_BUFFER_USAGE_UNIFORM | WGPU_BUFFER_USAGE_COPY_DST);
        g_dwg_data = dwg_buffer_(DWG_FLIGHT * DWG_DATA_BYTES, WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_DST);
        g_dwg_args_host = (unsigned char*)malloc(DWG_FLIGHT * DWG_BATCH * DWG_ARG_STRIDE);
        g_dwg_data_host = (unsigned char*)malloc(DWG_FLIGHT * DWG_DATA_BYTES);
        unsigned long units = g_dwg_heap_bytes >> DWG_UNIT_SHIFT;
        g_dwg_tag = (unsigned char*)malloc(units);
        if (g_dwg_heap == (void*)0 || g_dwg_args == (void*)0 || g_dwg_data == (void*)0 ||
            g_dwg_args_host == (unsigned char*)0 || g_dwg_data_host == (unsigned char*)0 || g_dwg_tag == (unsigned char*)0)
            return 0;
        memset((void*)g_dwg_args_host, 0, DWG_FLIGHT * DWG_BATCH * DWG_ARG_STRIDE);
        memset((void*)g_dwg_tag, 0, units);
        if (!dwg_bind_() || !dwg_bind_args_()) return 0;
        g_dwg_top = 0UL;
        while ((1UL << g_dwg_top) < units) g_dwg_top = g_dwg_top + 1UL;
        dwg_push_free_(g_dwg_top, 0UL);
        // unit 0 is never handed out: handle 0 means none
        if (dwg_take_(0UL) != 0L) return 0;
    }
    g_dwg_state = 1;
    return 1;
}

// ── The backend ──────────────────────────────────────────────────────────────
int WebGpuBackend::available() { return dwg_ensure_(); }

unsigned long WebGpuBackend::alloc(unsigned long bytes) {
    if (!dwg_ensure_()) return 0UL;
    unsigned long order = 0UL;
    while (((1UL << order) << DWG_UNIT_SHIFT) < bytes) order = order + 1UL;
    if (order >= 32UL) return 0UL;
    long u = dwg_take_(order);
    while (u < 0L) {
        if (!dwg_grow_(order)) return 0UL;
        u = dwg_take_(order);
    }
    return (unsigned long)u << DWG_UNIT_SHIFT;
}

// No wait: the queue runs in order, so work that still reads the block runs
// before anything a new owner queues.
void WebGpuBackend::release(unsigned long handle) {
    handle = dev_addr_(handle);
    if (handle == 0UL || g_dwg_state != 1 || handle >= g_dwg_heap_bytes) return;
    unsigned long unit = handle >> DWG_UNIT_SHIFT;
    unsafe { if (((unsigned long)g_dwg_tag[unit] & 0x80UL) == 0UL || (handle & ((1UL << DWG_UNIT_SHIFT) - 1UL)) != 0UL) return; }
    dwg_put_(unit);
}

int WebGpuBackend::upload_bytes(unsigned long dst, const void* src, unsigned long bytes) {
    if (bytes == 0UL) return 1;
    if (!dwg_ensure_()) return 0;
    dst = dev_addr_(dst);
    if ((dst & 3UL) != 0UL || (bytes & 3UL) != 0UL || dst + bytes > g_dwg_heap_bytes) return 0;
    unsafe {
        if (g_dwg_pending > 0UL && bytes <= DWG_INLINE_BYTES && g_dwg_data_used + bytes <= DWG_DATA_BYTES) {
            // behind the open command buffer's work: the bytes travel in its
            // data slot and wk_upload writes them in order
            if (g_dwg_upload == (void*)0) g_dwg_upload = dwg_make_pipe_("wk_upload", (const struct WGPUConstantEntry*)0, 0UL);
            if (g_dwg_upload != (void*)0) {
                unsigned long so = g_dwg_slot * DWG_DATA_BYTES + g_dwg_data_used;
                memcpy((void*)(g_dwg_data_host + so), src, bytes);
                g_dwg_data_used = g_dwg_data_used + bytes;
                unsigned int w[DWG_ARG_WORDS];
                memset((void*)&w[0], 0, DWG_ARG_WORDS * 4UL);
                w[0] = (unsigned int)so; w[2] = (unsigned int)dst; w[23] = (unsigned int)(bytes / 4UL);
                unsigned long groups = (bytes / 4UL + DWG_BLOCK - 1UL) / DWG_BLOCK;
                return dwg_dispatch_(g_dwg_upload, (const unsigned int*)&w[0], groups, 1UL, 1UL);
            }
        }
        if (g_dwg_pending > 0UL) dwg_submit_();   // queued work first
        wgpuQueueWriteBuffer(g_dwg_queue, g_dwg_heap, dst, src, bytes);
    }
    return 1;
}

int WebGpuBackend::upload(unsigned long dst, const &float src, unsigned long n) {
    unsafe { return WebGpuBackend::upload_bytes(dst, (const void*)src, n * 4UL); }
}

int WebGpuBackend::download_bytes(void* dst, unsigned long src, unsigned long bytes) {
    if (bytes == 0UL) return 1;
    if (!dwg_ensure_()) return 0;
    src = dev_addr_(src);
    // whole words from a 4-byte boundary (a 16-bit view may start mid-word)
    unsigned long start = src & ~3UL;
    unsigned long lead = src - start;
    unsigned long total = (lead + bytes + 3UL) & ~3UL;
    if (start + total > g_dwg_heap_bytes) return 0;
    unsafe {
        if (g_dwg_rb_bytes < total) {
            unsigned long rb = 1UL << 16;
            while (rb < total) rb = rb * 2UL;
            void* b = dwg_buffer_(rb, WGPU_BUFFER_USAGE_MAP_READ | WGPU_BUFFER_USAGE_COPY_DST);
            if (b == (void*)0) return 0;
            if (g_dwg_rb != (void*)0) wgpuBufferRelease(g_dwg_rb);
            g_dwg_rb = b;
            g_dwg_rb_bytes = rb;
        }
        if (!dwg_copy_buffers_(g_dwg_heap, start, g_dwg_rb, 0UL, total)) return 0;
        if (!dwg_wait_()) return 0;
        if (!dwg_map_readback_(total)) return 0;
        const unsigned char* p = (const unsigned char*)wgpuBufferGetConstMappedRange(g_dwg_rb, 0UL, total);
        if (p == (const unsigned char*)0) { wgpuBufferUnmap(g_dwg_rb); return 0; }
        memcpy(dst, (const void*)(p + lead), bytes);
        wgpuBufferUnmap(g_dwg_rb);
    }
    return 1;
}

int WebGpuBackend::download(&float dst, unsigned long src, unsigned long n) {
    unsafe { return WebGpuBackend::download_bytes((void*)dst, src, n * 4UL); }
}

// n 32-bit words, bit-exact (wk_copy: WebGPU copies within one buffer only
// through a kernel)
int WebGpuBackend::copy(unsigned long dst, unsigned long src, unsigned long n) {
    if (n == 0UL) return 1;
    if (!dwg_ensure_()) return 0;
    if (g_dwg_copy == (void*)0) g_dwg_copy = dwg_make_pipe_("wk_copy", (const struct WGPUConstantEntry*)0, 0UL);
    if (g_dwg_copy == (void*)0) { device_note_cpu_launch_(DEVICE_WEBGPU); return 0; }
    unsigned int w[DWG_ARG_WORDS];
    unsafe {
        memset((void*)&w[0], 0, DWG_ARG_WORDS * 4UL);
        w[0] = (unsigned int)dev_addr_(src);
        w[2] = (unsigned int)dev_addr_(dst);
        w[23] = (unsigned int)n;
    }
    unsigned long groups = (n + DWG_BLOCK - 1UL) / DWG_BLOCK;
    if (groups > DWG_MAX_GROUPS) groups = DWG_MAX_GROUPS;
    unsafe { return dwg_dispatch_(g_dwg_copy, (const unsigned int*)&w[0], groups, 1UL, 1UL); }
}

int WebGpuBackend::sync() {
    if (!dwg_ensure_()) return 0;
    return dwg_wait_();
}

// A strided batch: small products (up to 16384 elements of c, k up to 512)
// on wk_gemm1, a thread per element (no tiles: the lowest latency); the
// rest on wk_gemm (64 x 64 tiles) or, with enough tiles to fill the GPU,
// wk_gemm8 (128 x 128). A pipeline per kernel, transposes and element
// types; the batch split into grids of at most 65535. A single fp32
// product with too few 64 x 64 tiles and a long k splits k over the grid
// (wk_gemm adds the slices atomically), c first scaled by beta.
static int dwg_gemm_(int ta, int tb, int at, int ct, unsigned long m, unsigned long n, unsigned long k, float alpha,
                     unsigned long a, unsigned long lda, unsigned long sa, unsigned long b, unsigned long ldb,
                     unsigned long sb, float beta, unsigned long c, unsigned long ldc, unsigned long sc,
                     unsigned long batch) {
    if (!dwg_ensure_()) return 0;
    if (m == 0UL || n == 0UL || batch == 0UL) return 1;
    if (at < 0 || at > 2 || ct < 0 || ct > 2) return 0;
    int big = ((m + 127UL) / 128UL) * ((n + 127UL) / 128UL) * batch >= DWG_GEMM_FILL;
    if (m * n * batch <= 16384UL && k <= 512UL) big = 2;
    if (big == 2) {
        void* p1 = dwg_gemm_pipe_(2, ta != 0, tb != 0, at, ct);
        if (p1 == (void*)0) return 0;
        unsigned int w1[DWG_ARG_WORDS];
        unsafe {
            memset((void*)&w1[0], 0, DWG_ARG_WORDS * 4UL);
            w1[0] = (unsigned int)dev_addr_(a); w1[1] = (unsigned int)dev_addr_(b); w1[2] = (unsigned int)dev_addr_(c);
            w1[3] = (unsigned int)m; w1[4] = (unsigned int)n; w1[5] = (unsigned int)k;
            w1[6] = (unsigned int)lda; w1[7] = (unsigned int)ldb; w1[8] = (unsigned int)ldc;
            w1[9] = (unsigned int)sa; w1[10] = (unsigned int)sb; w1[11] = (unsigned int)sc;
            memcpy((void*)&w1[12], (const void*)&alpha, 4UL);
            memcpy((void*)&w1[13], (const void*)&beta, 4UL);
            w1[17] = (unsigned int)batch;
            unsigned long g = (m * n * batch + 255UL) / 256UL;
            if (g > DWG_MAX_GROUPS) g = DWG_MAX_GROUPS;
            return dwg_dispatch_(p1, (const unsigned int*)&w1[0], g, 1UL, 1UL);
        }
    }
    unsigned long tile = big ? 128UL : 64UL;
    unsigned long gx = (n + tile - 1UL) / tile;
    unsigned long gy = (m + tile - 1UL) / tile;
    void* pipe = dwg_gemm_pipe_(big, ta != 0, tb != 0, at, ct);
    if (pipe == (void*)0 || gx > DWG_MAX_DIM || gy > DWG_MAX_DIM) { device_note_cpu_launch_(DEVICE_WEBGPU); return 0; }
    // split-K
    unsigned long S = 1UL;
    unsigned long kSlice = k;
    if (!big && ct == 0 && batch == 1UL && ldc == n && k >= 256UL && gx * gy < DWG_GEMM_FILL) {
        S = (DWG_GEMM_FILL + gx * gy - 1UL) / (gx * gy);
        if (S > k / 128UL) S = k / 128UL;
        kSlice = ((k + S - 1UL) / S + 15UL) / 16UL * 16UL;
        S = (k + kSlice - 1UL) / kSlice;
    }
    if (S > 1UL && beta != (float)1.0) {
        struct DevArgs e = dev_args_();
        unsafe {
            e.p[0] = c; e.p[2] = c; e.u[7] = (unsigned int)(m * n);
            e.u[0] = beta == (float)0.0 ? 5U : 3U;   // EW_FILL 0 / EW_SCALE beta
            e.f[0] = beta == (float)0.0 ? (float)0.0 : beta;
        }
        if (!WebGpuBackend::launch(beta == (float)0.0 ? DK_EWOP_5 : DK_EWOP_3, m * n, &e)) return 0;
    }
    unsigned int w[DWG_ARG_WORDS];
    unsafe {
        memset((void*)&w[0], 0, DWG_ARG_WORDS * 4UL);
        w[0] = (unsigned int)dev_addr_(a); w[1] = (unsigned int)dev_addr_(b); w[2] = (unsigned int)dev_addr_(c);
        w[3] = (unsigned int)m; w[4] = (unsigned int)n; w[5] = (unsigned int)k;
        w[6] = (unsigned int)lda; w[7] = (unsigned int)ldb; w[8] = (unsigned int)ldc;
        w[9] = (unsigned int)sa; w[10] = (unsigned int)sb; w[11] = (unsigned int)sc;
        memcpy((void*)&w[12], (const void*)&alpha, 4UL);
        memcpy((void*)&w[13], (const void*)&beta, 4UL);
        w[15] = (unsigned int)S; w[16] = (unsigned int)kSlice;
    }
    unsigned long z = 0UL;
    while (z < batch) {
        unsigned long gz = batch - z;
        if (gz > DWG_MAX_DIM) gz = DWG_MAX_DIM;
        unsafe {
            w[14] = (unsigned int)z;
            if (!dwg_dispatch_(pipe, (const unsigned int*)&w[0], gx, gy, gz * S)) return 0;
        }
        z = z + gz;
    }
    return 1;
}

int WebGpuBackend::gemm(int ta, int tb, unsigned long m, unsigned long n, unsigned long k, float alpha,
                        unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb,
                        float beta, unsigned long c, unsigned long ldc) {
    return dwg_gemm_(ta, tb, 0, 0, m, n, k, alpha, a, lda, 0UL, b, ldb, 0UL, beta, c, ldc, 0UL, 1UL);
}

int WebGpuBackend::gemm_batched(const &DevGemm g) {
    return dwg_gemm_(g.ta, g.tb, g.dtype, g.ctype, g.m, g.n, g.k, g.alpha, g.a, g.lda, g.sa, g.b, g.ldb, g.sb, g.beta,
                     g.c, g.ldc, g.sc, g.batch);
}

// Convolutions as implicit GEMMs (device_kernels.wgsl's wk_conv): the
// forward pass, the input gradient as a gather over the windows (no
// atomics), the weight gradient split over batch x pixels and added
// atomically into zeroed weights (fp32 only: a 16-bit one declines and
// device.sc runs it in fp32). A transposed convolution is the plain one with
// input and output swapped, so its forward pass is the input-gradient kernel
// and its input gradient the forward one; the input gradient runs a grid
// slice per stride phase, as Metal's does. The tile, by the product's rows
// (output channels, or input channels for the input gradient): 64 x 64
// from 48, 32 x 128 from 17, 16 x 256 below.
int WebGpuBackend::conv(const &DevConv c) {
    if (!dwg_ensure_() || c.dtype < DEVICE_FP32 || c.dtype > DEVICE_BF16) return 0;
    unsigned long x = c.x; unsigned long y = c.y;
    unsigned long Ci = c.inC; unsigned long Hi = c.inH; unsigned long Wi = c.inW;
    unsigned long Co = c.outC; unsigned long Ho = c.outH; unsigned long Wo = c.outW;
    if (c.transpose) {
        x = c.y; y = c.x;
        Ci = c.outC; Hi = c.outH; Wi = c.outW;
        Co = c.inC; Ho = c.inH; Wo = c.inW;
    }
    int op = c.op == DEV_CONV_BWD_WEIGHTS ? 2 : ((c.op == DEV_CONV_FWD) == (c.transpose == 0) ? 0 : 1);
    if (op == 2 && c.dtype != DEVICE_FP32) return 0;
    unsigned long kk = c.kH * c.kW;
    // the product's m and n
    unsigned long m = op == 1 ? Ci : Co;
    unsigned long n = op == 0 ? c.batch * Ho * Wo : (op == 1 ? c.batch * Hi * Wi : Ci * kk);
    if (m == 0UL || n == 0UL) return 1;
    // the input gradient runs per stride phase (without dilation): fewer pixels each
    unsigned long phases = 1UL;
    if (op == 1 && c.dH == 1UL && c.dW == 1UL) {
        phases = c.sH * c.sW;
        n = c.batch * ((Hi + c.sH - 1UL) / c.sH) * ((Wi + c.sW - 1UL) / c.sW);
    }
    int shape = m >= 48UL ? DWG_CONV_64 : (m > 16UL ? DWG_CONV_32 : DWG_CONV_16);
    unsigned long rows = dwg_conv_rows_(shape);
    unsigned long cols = dwg_conv_cols_(shape);
    void* pipe = dwg_conv_pipe_(op, shape, c.dtype);
    unsigned long gx = (n + cols - 1UL) / cols;
    unsigned long gy = (m + rows - 1UL) / rows;
    if (pipe == (void*)0 || gx > DWG_MAX_DIM || gy > DWG_MAX_DIM) return 0;   // im2col + GEMM instead
    unsigned int w[DWG_ARG_WORDS];
    unsafe {
        memset((void*)&w[0], 0, DWG_ARG_WORDS * 4UL);
        w[0] = (unsigned int)dev_addr_(x); w[1] = (unsigned int)dev_addr_(c.w); w[2] = (unsigned int)dev_addr_(y);
        w[3] = (unsigned int)c.batch; w[4] = (unsigned int)Ci; w[5] = (unsigned int)Hi; w[6] = (unsigned int)Wi;
        w[7] = (unsigned int)Co; w[8] = (unsigned int)Ho; w[9] = (unsigned int)Wo;
        w[10] = (unsigned int)c.kH; w[11] = (unsigned int)c.kW; w[12] = (unsigned int)c.sH; w[13] = (unsigned int)c.sW;
        w[14] = (unsigned int)c.pH; w[15] = (unsigned int)c.pW; w[16] = (unsigned int)c.dH; w[17] = (unsigned int)c.dW;
    }
    unsigned long splits = phases;
    if (op == 2) {
        // zero the weights, then enough slices of k to fill the GPU
        unsigned long nw = Co * Ci * kk;
        struct DevArgs z = dev_args_();
        unsafe { z.p[2] = c.w; z.u[0] = 5U; z.u[7] = (unsigned int)nw; }   // EW_FILL 0
        if (!WebGpuBackend::launch(DK_EWOP_5, nw, &z)) return 0;
        unsigned long kTot = c.batch * Ho * Wo;
        splits = (128UL + gx * gy - 1UL) / (gx * gy);
        if (splits > kTot / 256UL) splits = kTot / 256UL;
        if (splits < 1UL) splits = 1UL;
        unsigned long per = (kTot + splits - 1UL) / splits;
        per = (per + 15UL) / 16UL * 16UL;
        splits = (kTot + per - 1UL) / per;
        unsafe { w[18] = (unsigned int)per; }
    }
    unsafe { return dwg_dispatch_(pipe, (const unsigned int*)&w[0], gx, gy, splits); }
}

int WebGpuBackend::storage16() { return 1; }

int WebGpuBackend::launch(int kernel, unsigned long n, const &DevArgs args) {
    if (!dwg_ensure_() || kernel < 0 || kernel >= DK_COUNT) return 0;
    unsigned int w[DWG_ARG_WORDS];
    int tagged = 0;
    unsafe {
        unsigned long i = 0UL;
        while (i < 8UL) {
            w[i] = (unsigned int)dev_addr_(args.p[i]);
            w[8UL + i] = (unsigned int)dev_tag_(args.p[i]);
            if (args.p[i] != 0UL && dev_tag_(args.p[i]) != 0UL) tagged = 1;
            w[16UL + i] = args.u[i];
            memcpy((void*)&w[24UL + i], (const void*)&args.f[i], 4UL);
            i = i + 1UL;
        }
    }
    int attn = dwg_is_attn_(kernel);
    int wide = 0;
    unsafe { wide = attn && args.u[3] <= 64U && args.u[4] <= 64U; }
    int lanes = 0;
    unsafe { lanes = dwg_rows_(kernel) && args.u[0] >= 32U; }
    void* pipe = dwg_pipe_(kernel, tagged, wide, lanes);
    if (pipe == (void*)0 && (wide || lanes)) { wide = 0; lanes = 0; pipe = dwg_pipe_(kernel, tagged, 0, 0); }
    if (pipe == (void*)0) { device_note_cpu_launch_(DEVICE_WEBGPU); return 0; }   // the tensor op runs on the CPU
    unsigned long groups = ((lanes ? 32UL * n : n) + DWG_BLOCK - 1UL) / DWG_BLOCK;
    if (groups > DWG_MAX_GROUPS) groups = DWG_MAX_GROUPS;
    if (attn) {
        // a workgroup per (head, tile of rows): rows are keys for dk_attn_bwd_kv, else queries
        unsigned long tile = wide ? DWG_ATTN16_T : DWG_ATTN_T;
        unsafe {
            unsigned long rows = kernel == DK_ATTN_BWD_KV ? (unsigned long)args.u[1] : (unsigned long)args.u[0];
            groups = (unsigned long)args.u[2] * ((rows + tile - 1UL) / tile);
        }
        if (groups > DWG_MAX_DIM) groups = DWG_MAX_DIM;
    }
    if (groups == 0UL) groups = 1UL;
    unsafe { return dwg_dispatch_(pipe, (const unsigned int*)&w[0], groups, 1UL, 1UL); }
}

unsigned long WebGpuBackend::offset(unsigned long handle, unsigned long bytes) { return handle + bytes; }

int WebGpuBackend::map(int op, unsigned long a, unsigned long b, unsigned long out, unsigned long n) {
    if (n == 0UL) return 1;
    if (!dwg_ensure_()) return 0;
    if (g_dwg_map == (void*)0) g_dwg_map = dwg_make_pipe_("wk_map", (const struct WGPUConstantEntry*)0, 0UL);
    if (g_dwg_map == (void*)0) return 0;
    unsigned int w[DWG_ARG_WORDS];
    unsafe {
        memset((void*)&w[0], 0, DWG_ARG_WORDS * 4UL);
        w[0] = (unsigned int)dev_addr_(a); w[1] = (unsigned int)dev_addr_(b); w[2] = (unsigned int)dev_addr_(out);
        w[16] = (unsigned int)op;
        w[23] = (unsigned int)n;
    }
    unsigned long groups = (n + DWG_BLOCK - 1UL) / DWG_BLOCK;
    if (groups > DWG_MAX_GROUPS) groups = DWG_MAX_GROUPS;
    unsafe { return dwg_dispatch_(g_dwg_map, (const unsigned int*)&w[0], groups, 1UL, 1UL); }
}

void device_webgpu_register_() {
    struct WebGpuBackend tag;
    tag.unused_ = 0;
    device_register_backend_(DEVICE_WEBGPU, &tag);
}

} // namespace std
