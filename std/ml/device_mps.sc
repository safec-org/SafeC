// SafeC Standard Library — the Metal (Apple Silicon) device backend (see device.h).
// MpsBackend: device memory is MTLBuffers in shared storage, named by their
// GPU addresses (MTLBuffer.gpuAddress) the way ROCm and CUDA name theirs, so
// a handle plus a byte offset is an address and the kernels' raw-address
// arguments (DevArgs.p, the optimizer's tables of handles) work as they do
// there. Every buffer joins one MTLResidencySet attached to gpu_mps.h's
// queue, which keeps it resident for kernels that reach it only through an
// address. device_kernels.cu is compiled at run time by newLibraryWithSource
// (plain, and with DK_TAGGED on first use), and its Metal-only kernels run
// the GEMMs (mk_gemm, simdgroup matrices) and copies (mk_copy).
//
// Work is encoded into one command buffer at a time (one compute encoder, a
// memory barrier after each dispatch) and committed every DMPS_COMMIT_EVERY
// dispatches without waiting; each command buffer waits for an MTLEvent its
// predecessor signals, so they run in order although the GPU sees their
// buffers only through addresses. The host waits only when it reads
// (download, sync), and before an upload into or release of memory the
// queued work may still use. A program enables it with its mps feature;
// link -framework Metal -framework Foundation -lobjc.
//
// Limits, against ROCm / CUDA:
//   - Metal has no doubles: the batch / group norm statistics accumulate in
//     fp32 (two-pass, so they hold device_check's tolerance).
//   - The fused attention kernels work in 8 x 8 tiles (64 threads), not
//     16 x 16: four 16-row tiles of 128 floats overflow 32 KB of
//     threadgroup memory.
//   - Convolutions run im2col + GEMM (conv() declines): there is no MIOpen /
//     cuDNN to call.
//   - A kernel runs on the CPU (device_ref.sc, on the shared memory; counted
//     in DeviceStats.cpuLaunches) only if its pipeline can't be built; one
//     with 16-bit arguments or a table of handles then declines instead.
//   - gpu_mps.h's batches and this backend share the queue: an open batch
//     is submitted before this backend's next command buffer starts, but
//     work this backend has encoded and not yet committed waits for
//     device_sync(DEVICE_MPS) before gpu_mps.h ops that should follow it.
#pragma once
#include <std/stddef.h>
#include <std/ml/device.h>
#include <std/ml/device_kernels.h>
#include <std/ml/gpu_mps.h>

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);
extern void* memcpy(void* dst, const void* src, size_t n);
extern long  write(int fd, const void* buf, unsigned long n);

#if defined(__APPLE__)
extern void* objc_getClass(const char* name);
extern void* sel_registerName(const char* name);
extern void* objc_msgSend(void* recv, void* op);
extern void* objc_autoreleasePoolPush();
extern void  objc_autoreleasePoolPop(void* pool);
#else
// No Metal elsewhere (gpu_mps.sc's stand-ins): mps_device_handle() is null,
// so the backend is never available.
static void* objc_getClass(const char* name) { return (void*)0; }
static void* sel_registerName(const char* name) { return (void*)0; }
static void* objc_msgSend(void* recv, void* op) { return (void*)0; }
static void* objc_autoreleasePoolPush() { return (void*)0; }
static void  objc_autoreleasePoolPop(void* pool) { }
#endif

// Message shapes (objc_msgSend cast per shape, assigned before the call: see
// gpu_mps.sc's header comment).
struct DmpsSize_ {
    unsigned long width;
    unsigned long height;
    unsigned long depth;
};
typedef fn void*(void*, void*) DmpsMsg0;
typedef fn void*(void*, void*, void*) DmpsMsg1;
typedef fn void*(void*, void*, void*, void*) DmpsMsg2;
typedef fn void*(void*, void*, void*, void*, void*) DmpsMsg3;
typedef fn void*(void*, void*, unsigned long) DmpsMsgPU;
typedef fn void*(void*, void*, unsigned long, unsigned long) DmpsMsgPUU;
typedef fn unsigned long(void*, void*) DmpsMsgU;
typedef fn int(void*, void*, void*) DmpsMsgB1;
typedef fn void(void*, void*) DmpsMsgV;
typedef fn void(void*, void*, void*) DmpsMsgV1;
typedef fn void(void*, void*, unsigned long) DmpsMsgVU;
typedef fn void(void*, void*, unsigned long, unsigned long) DmpsMsgVUU;
typedef fn void(void*, void*, void*, unsigned long) DmpsMsgV1U;
typedef fn void(void*, void*, const void*, unsigned long, unsigned long) DmpsMsgBytes;
typedef fn void(void*, void*, struct DmpsSize_, struct DmpsSize_) DmpsMsgDispatch;

#define DMPS_BLOCK 256UL
#define DMPS_MAX_GROUPS 4096UL
#define DMPS_ATTN_T 8UL            // device_kernels.cu's ATTN_T on Metal
#define DMPS_GEMM_TILE 64UL        // mk_gemm's tile (MK_BM, MK_BN), 128 threads
#define DMPS_RED_BYTES 128UL       // dk_red: a float per simdgroup
#define DMPS_COMMIT_EVERY 64UL     // dispatches per command buffer
#define DMPS_FLIGHT 32UL           // committed command buffers not yet known done
#define DMPS_SHARED_UNTRACKED 256UL   // MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked
#define DMPS_CONCURRENT 1UL        // MTLDispatchTypeConcurrent
#define DMPS_SCOPE_BUFFERS 1UL     // MTLBarrierScopeBuffers
#define DMPS_STATUS_ERROR 5UL      // MTLCommandBufferStatusError

// A device buffer: its GPU address range, the MTLBuffer, its CPU view.
struct DmpsBuf_ {
    unsigned long addr;
    unsigned long bytes;
    void* buf;
    unsigned char* cpu;
};

// mk_gemm's DkGemm.
struct DmpsGemm_ {
    unsigned long a; unsigned long b; unsigned long c;
    unsigned long lda; unsigned long ldb; unsigned long ldc;
    unsigned long sa; unsigned long sb; unsigned long sc;
    unsigned int m; unsigned int n; unsigned int k; unsigned int ta; unsigned int tb; unsigned int at; unsigned int ct;
    float alpha;
    float beta;
};

static int   g_dmps_state = 0;          // 0 = not yet, 1 = ready, -1 = unusable
static void* g_dmps_dev = (void*)0;
static void* g_dmps_queue = (void*)0;
static void* g_dmps_rset = (void*)0;    // the residency set (null before macOS 15: useResources instead)
static void* g_dmps_event = (void*)0;
static void* g_dmps_lib[2];             // plain, DK_TAGGED
static int   g_dmps_lib_state[2];
static void* g_dmps_pipe[2 * DK_COUNT];
static int   g_dmps_pipe_state[2 * DK_COUNT];   // 0 = not yet, 1 = built, -1 = can't
static void* g_dmps_gemm = (void*)0;
static void* g_dmps_copy = (void*)0;

// Buffers sorted by address.
static struct DmpsBuf_* g_dmps_bufs = (struct DmpsBuf_*)0;
static unsigned long g_dmps_nbufs = 0UL;
static unsigned long g_dmps_capbufs = 0UL;

// The open command buffer and its encoder (retained; null when none), and
// the committed ones not yet known to be done.
static void* g_dmps_cb = (void*)0;
static void* g_dmps_enc = (void*)0;
static void* g_dmps_last_pipe = (void*)0;
static unsigned long g_dmps_count = 0UL;
static unsigned long g_dmps_serial = 0UL;   // what the last committed command buffer signals
static void* g_dmps_flight[DMPS_FLIGHT];
static unsigned long g_dmps_nflight = 0UL;
static int g_dmps_reported = 0;

// Selectors on the encoding path, resolved once.
static void* s_dmps_setPipe = (void*)0;
static void* s_dmps_setBytes = (void*)0;
static void* s_dmps_dispatch = (void*)0;
static void* s_dmps_barrier = (void*)0;
static void* s_dmps_status = (void*)0;
static void* s_dmps_release = (void*)0;

static void dmps_log_(const char* what, void* nserror) {
    unsafe {
        const char* msg = "";
        if (nserror != (void*)0) {
            DmpsMsg0 m0 = (DmpsMsg0)objc_msgSend;
            void* d = m0(nserror, sel_registerName("localizedDescription"));
            if (d != (void*)0) msg = (const char*)m0(d, sel_registerName("UTF8String"));
        }
        unsigned long n = 0UL;
        while (what[n] != (char)0) n = n + 1UL;
        write(2, (const void*)what, n);
        n = 0UL;
        while (msg[n] != (char)0) n = n + 1UL;
        write(2, (const void*)msg, n);
        write(2, (const void*)"\n", 1UL);
    }
}

static void dmps_release_obj_(void* o) {
    if (o == (void*)0) return;
    unsafe {
        DmpsMsgV mv = (DmpsMsgV)objc_msgSend;
        mv(o, s_dmps_release);
    }
}

static int dmps_ensure_() {
    if (g_dmps_state != 0) return g_dmps_state == 1;
    g_dmps_state = -1;
    unsafe {
        match (mps_device_handle()) {
            case some(d): { g_dmps_dev = (void*)d; }
            case null: { return 0; }
        }
        match (mps_queue_handle()) {
            case some(q): { g_dmps_queue = (void*)q; }
            case null: { return 0; }
        }
        s_dmps_setPipe = sel_registerName("setComputePipelineState:");
        s_dmps_setBytes = sel_registerName("setBytes:length:atIndex:");
        s_dmps_dispatch = sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:");
        s_dmps_barrier = sel_registerName("memoryBarrierWithScope:");
        s_dmps_status = sel_registerName("status");
        s_dmps_release = sel_registerName("release");
        DmpsMsg0 m0 = (DmpsMsg0)objc_msgSend;
        DmpsMsg2 m2 = (DmpsMsg2)objc_msgSend;
        DmpsMsgB1 responds = (DmpsMsgB1)objc_msgSend;
        g_dmps_event = m0(g_dmps_dev, sel_registerName("newEvent"));
        if (g_dmps_event == (void*)0) return 0;
        void* resSel = sel_registerName("newResidencySetWithDescriptor:error:");
        if ((responds(g_dmps_dev, sel_registerName("respondsToSelector:"), resSel) & 255) != 0) {
            void* desc = m0(objc_getClass("MTLResidencySetDescriptor"), sel_registerName("new"));
            DmpsMsgVU setCap = (DmpsMsgVU)objc_msgSend;
            setCap(desc, sel_registerName("setInitialCapacity:"), 256UL);
            void* err = (void*)0;
            g_dmps_rset = m2(g_dmps_dev, resSel, desc, (void*)&err);
            dmps_release_obj_(desc);
            if (g_dmps_rset != (void*)0) {
                DmpsMsgV1 m1v = (DmpsMsgV1)objc_msgSend;
                m1v(g_dmps_queue, sel_registerName("addResidencySet:"), g_dmps_rset);
            }
        }
    }
    g_dmps_state = 1;
    return 1;
}

// ── Kernels: device_kernels.cu as Metal Shading Language ─────────────────────
// Compiled on first use with IEEE math (no fast-math, precise functions), as
// hiprtc and NVRTC compile it.
static int dmps_library_(int tagged) {
    if (g_dmps_lib_state[tagged] != 0) return g_dmps_lib_state[tagged] == 1;
    g_dmps_lib_state[tagged] = -1;
    unsafe {
        const char* src = device_kernel_source_();
        char* text = (char*)0;
        if (tagged) {
            const char* pre = "#define DK_TAGGED 1\n";
            unsigned long np = 0UL;
            while (pre[np] != (char)0) np = np + 1UL;
            unsigned long ns = 0UL;
            while (src[ns] != (char)0) ns = ns + 1UL;
            text = (char*)malloc(np + ns + 1UL);
            if (text == (char*)0) return 0;
            memcpy((void*)text, (const void*)pre, np);
            memcpy((void*)(text + np), (const void*)src, ns + 1UL);
            src = (const char*)text;
        }
        void* pool = objc_autoreleasePoolPush();
        DmpsMsg0 m0 = (DmpsMsg0)objc_msgSend;
        DmpsMsg1 m1 = (DmpsMsg1)objc_msgSend;
        DmpsMsg3 m3 = (DmpsMsg3)objc_msgSend;
        DmpsMsgVU mvu = (DmpsMsgVU)objc_msgSend;
        DmpsMsgB1 responds = (DmpsMsgB1)objc_msgSend;
        void* str = m1(m0(objc_getClass("NSString"), sel_registerName("alloc")), sel_registerName("initWithUTF8String:"),
                       (void*)src);
        void* opts = m0(m0(objc_getClass("MTLCompileOptions"), sel_registerName("alloc")), sel_registerName("init"));
        void* mathMode = sel_registerName("setMathMode:");
        if ((responds(opts, sel_registerName("respondsToSelector:"), mathMode) & 255) != 0) {
            mvu(opts, mathMode, 0UL);                                                  // MTLMathModeSafe
            mvu(opts, sel_registerName("setMathFloatingPointFunctions:"), 1UL);        // ...Precise
        } else {
            mvu(opts, sel_registerName("setFastMathEnabled:"), 0UL);
        }
        void* err = (void*)0;
        void* lib = m3(g_dmps_dev, sel_registerName("newLibraryWithSource:options:error:"), str, opts, (void*)&err);
        if (lib == (void*)0) dmps_log_("device_mps: the kernels don't compile: ", err);
        dmps_release_obj_(str);
        dmps_release_obj_(opts);
        objc_autoreleasePoolPop(pool);
        if (text != (char*)0) free((void*)text);
        if (lib == (void*)0) return 0;
        g_dmps_lib[tagged] = lib;
    }
    g_dmps_lib_state[tagged] = 1;
    return 1;
}

// The pipeline for function 'name' of a library, null when it can't be built
// for 'threads' threads per threadgroup.
static void* dmps_make_pipe_(int tagged, const char* name, unsigned long threads) {
    if (!dmps_library_(tagged)) return (void*)0;
    unsafe {
        void* pool = objc_autoreleasePoolPush();
        DmpsMsg0 m0 = (DmpsMsg0)objc_msgSend;
        DmpsMsg1 m1 = (DmpsMsg1)objc_msgSend;
        DmpsMsg2 m2 = (DmpsMsg2)objc_msgSend;
        void* str = m1(m0(objc_getClass("NSString"), sel_registerName("alloc")), sel_registerName("initWithUTF8String:"),
                       (void*)name);
        void* f = m1(g_dmps_lib[tagged], sel_registerName("newFunctionWithName:"), str);
        dmps_release_obj_(str);
        void* pipe = (void*)0;
        if (f != (void*)0) {
            void* err = (void*)0;
            pipe = m2(g_dmps_dev, sel_registerName("newComputePipelineStateWithFunction:error:"), f, (void*)&err);
            if (pipe == (void*)0) dmps_log_("device_mps: a kernel's pipeline failed: ", err);
            dmps_release_obj_(f);
        }
        if (pipe != (void*)0) {
            DmpsMsgU mu = (DmpsMsgU)objc_msgSend;
            if (mu(pipe, sel_registerName("maxTotalThreadsPerThreadgroup")) < threads) {
                dmps_release_obj_(pipe);
                pipe = (void*)0;
            }
        }
        objc_autoreleasePoolPop(pool);
        return pipe;
    }
}

// Threads per threadgroup for a DK_* kernel.
static unsigned long dmps_block_(int kernel) {
    if (kernel == DK_ATTN_FWD || kernel == DK_ATTN_BWD_Q || kernel == DK_ATTN_BWD_KV) return DMPS_ATTN_T * DMPS_ATTN_T;
    return DMPS_BLOCK;
}

static void* dmps_pipe_(int kernel, int tagged) {
    unsigned long slot = (unsigned long)tagged * (unsigned long)DK_COUNT + (unsigned long)kernel;
    unsafe {
        if (g_dmps_pipe_state[slot] == 0) {
            g_dmps_pipe[slot] = dmps_make_pipe_(tagged, device_kernel_name_(kernel), dmps_block_(kernel));
            g_dmps_pipe_state[slot] = g_dmps_pipe[slot] != (void*)0 ? 1 : -1;
        }
        return g_dmps_pipe[slot];
    }
}

// ── Buffers ──────────────────────────────────────────────────────────────────
// The buffer holding address h, or -1.
static long dmps_find_(unsigned long h) {
    unsafe {
        long lo = 0L;
        long hi = (long)g_dmps_nbufs - 1L;
        while (lo <= hi) {
            long mid = (lo + hi) / 2L;
            struct DmpsBuf_* b = &g_dmps_bufs[mid];
            if (h < b->addr) hi = mid - 1L;
            else if (h >= b->addr + b->bytes) lo = mid + 1L;
            else return mid;
        }
    }
    return -1L;
}

// The CPU's view of device address h (null for an unknown one).
static unsigned char* dmps_host_(unsigned long h) {
    h = dev_addr_(h);
    long i = dmps_find_(h);
    if (i < 0L) return (unsigned char*)0;
    unsafe { return g_dmps_bufs[i].cpu + (h - g_dmps_bufs[i].addr); }
}

// ── The command stream ───────────────────────────────────────────────────────
// Makes resident every buffer for the open encoder (no residency set).
static void dmps_use_all_() {
    if (g_dmps_nbufs == 0UL) return;
    unsafe {
        void** list = (void**)malloc(sizeof(void*) * g_dmps_nbufs);
        if (list == (void**)0) return;
        unsigned long i = 0UL;
        while (i < g_dmps_nbufs) { list[i] = g_dmps_bufs[i].buf; i = i + 1UL; }
        DmpsMsgBytes use = (DmpsMsgBytes)objc_msgSend;
        use(g_dmps_enc, sel_registerName("useResources:count:usage:"), (const void*)list, g_dmps_nbufs, 3UL);
        free((void*)list);
    }
}

// The open encoder, started (on a new command buffer) if there is none.
static int dmps_begin_() {
    if (g_dmps_enc != (void*)0) return 1;
    // gpu_mps.h work in an open batch runs first
    if (mps_batch_is_active()) mps_batch_submit();
    unsafe {
        void* pool = objc_autoreleasePoolPush();
        DmpsMsg0 m0 = (DmpsMsg0)objc_msgSend;
        DmpsMsgPU mpu = (DmpsMsgPU)objc_msgSend;
        void* retain = sel_registerName("retain");
        g_dmps_cb = m0(m0(g_dmps_queue, sel_registerName("commandBufferWithUnretainedReferences")), retain);
        if (g_dmps_cb != (void*)0 && g_dmps_serial > 0UL) {
            DmpsMsgV1U wait = (DmpsMsgV1U)objc_msgSend;
            wait(g_dmps_cb, sel_registerName("encodeWaitForEvent:value:"), g_dmps_event, g_dmps_serial);
        }
        if (g_dmps_cb != (void*)0)
            g_dmps_enc = m0(mpu(g_dmps_cb, sel_registerName("computeCommandEncoderWithDispatchType:"), DMPS_CONCURRENT), retain);
        objc_autoreleasePoolPop(pool);
        if (g_dmps_enc == (void*)0) {
            dmps_release_obj_(g_dmps_cb);
            g_dmps_cb = (void*)0;
            return 0;
        }
        DmpsMsgVUU tgMem = (DmpsMsgVUU)objc_msgSend;
        tgMem(g_dmps_enc, sel_registerName("setThreadgroupMemoryLength:atIndex:"), DMPS_RED_BYTES, 0UL);
        if (g_dmps_rset == (void*)0) dmps_use_all_();
    }
    g_dmps_last_pipe = (void*)0;
    g_dmps_count = 0UL;
    return 1;
}

// Releases the oldest committed command buffers that have finished.
static void dmps_retire_() {
    unsafe {
        DmpsMsgU mu = (DmpsMsgU)objc_msgSend;
        unsigned long done = 0UL;
        while (done < g_dmps_nflight && mu(g_dmps_flight[done], s_dmps_status) >= 4UL) done = done + 1UL;
        if (done == 0UL) return;
        unsigned long i = 0UL;
        while (i < done) { dmps_release_obj_(g_dmps_flight[i]); i = i + 1UL; }
        i = done;
        while (i < g_dmps_nflight) { g_dmps_flight[i - done] = g_dmps_flight[i]; i = i + 1UL; }
        g_dmps_nflight = g_dmps_nflight - done;
    }
}

// Commits the open command buffer (no wait).
static void dmps_commit_() {
    if (g_dmps_cb == (void*)0) return;
    unsafe {
        DmpsMsgV mv = (DmpsMsgV)objc_msgSend;
        mv(g_dmps_enc, sel_registerName("endEncoding"));
        dmps_release_obj_(g_dmps_enc);
        g_dmps_enc = (void*)0;
        g_dmps_serial = g_dmps_serial + 1UL;
        DmpsMsgV1U signal = (DmpsMsgV1U)objc_msgSend;
        signal(g_dmps_cb, sel_registerName("encodeSignalEvent:value:"), g_dmps_event, g_dmps_serial);
        mv(g_dmps_cb, sel_registerName("commit"));
        dmps_retire_();
        if (g_dmps_nflight == DMPS_FLIGHT) {
            mv(g_dmps_flight[0], sel_registerName("waitUntilCompleted"));
            dmps_retire_();
        }
        g_dmps_flight[g_dmps_nflight] = g_dmps_cb;
        g_dmps_nflight = g_dmps_nflight + 1UL;
        g_dmps_cb = (void*)0;
    }
}

static int dmps_busy_() { return g_dmps_cb != (void*)0 || g_dmps_nflight > 0UL; }

// Commits and waits for everything queued; 0 if the GPU reported an error.
static int dmps_wait_() {
    dmps_commit_();
    if (g_dmps_nflight == 0UL) return 1;
    int ok = 1;
    unsafe {
        void* pool = objc_autoreleasePoolPush();
        DmpsMsgV mv = (DmpsMsgV)objc_msgSend;
        DmpsMsgU mu = (DmpsMsgU)objc_msgSend;
        DmpsMsg0 m0 = (DmpsMsg0)objc_msgSend;
        // each waits for its predecessor's event: the last one done is all done
        mv(g_dmps_flight[g_dmps_nflight - 1UL], sel_registerName("waitUntilCompleted"));
        unsigned long i = 0UL;
        while (i < g_dmps_nflight) {
            if (mu(g_dmps_flight[i], s_dmps_status) == DMPS_STATUS_ERROR) {
                ok = 0;
                if (!g_dmps_reported) {
                    g_dmps_reported = 1;
                    dmps_log_("device_mps: a command buffer failed: ", m0(g_dmps_flight[i], sel_registerName("error")));
                }
            }
            dmps_release_obj_(g_dmps_flight[i]);
            i = i + 1UL;
        }
        g_dmps_nflight = 0UL;
        objc_autoreleasePoolPop(pool);
    }
    return ok;
}

// One dispatch of 'groups' threadgroups of 'threads' on the open encoder,
// 'args' (len bytes) at buffer 0; then a barrier, so the next one sees its writes.
static int dmps_dispatch_(void* pipe, const void* args, unsigned long len, unsigned long groupsX, unsigned long groupsY,
                          unsigned long groupsZ, unsigned long threads) {
    if (!dmps_begin_()) return 0;
    unsafe {
        if (pipe != g_dmps_last_pipe) {
            DmpsMsgV1 setPipe = (DmpsMsgV1)objc_msgSend;
            setPipe(g_dmps_enc, s_dmps_setPipe, pipe);
            g_dmps_last_pipe = pipe;
        }
        DmpsMsgBytes setBytes = (DmpsMsgBytes)objc_msgSend;
        setBytes(g_dmps_enc, s_dmps_setBytes, args, len, 0UL);
        struct DmpsSize_ grid;
        grid.width = groupsX; grid.height = groupsY; grid.depth = groupsZ;
        struct DmpsSize_ tg;
        tg.width = threads; tg.height = 1UL; tg.depth = 1UL;
        DmpsMsgDispatch dispatch = (DmpsMsgDispatch)objc_msgSend;
        dispatch(g_dmps_enc, s_dmps_dispatch, grid, tg);
        DmpsMsgVU barrier = (DmpsMsgVU)objc_msgSend;
        barrier(g_dmps_enc, s_dmps_barrier, DMPS_SCOPE_BUFFERS);
    }
    g_dmps_count = g_dmps_count + 1UL;
    if (g_dmps_count >= DMPS_COMMIT_EVERY) dmps_commit_();
    return 1;
}

static int dmps_gemm_(const &DmpsGemm_ g, unsigned long batch) {
    if (!dmps_ensure_()) return 0;
    if (g.m == 0U || g.n == 0U || batch == 0UL) return 1;
    if (g_dmps_gemm == (void*)0) g_dmps_gemm = dmps_make_pipe_(0, "mk_gemm", 128UL);
    if (g_dmps_gemm == (void*)0) return 0;
    unsafe {
        return dmps_dispatch_(g_dmps_gemm, (const void*)g, sizeof(struct DmpsGemm_),
                              ((unsigned long)g.n + DMPS_GEMM_TILE - 1UL) / DMPS_GEMM_TILE,
                              ((unsigned long)g.m + DMPS_GEMM_TILE - 1UL) / DMPS_GEMM_TILE, batch, 128UL);
    }
}

// A kernel on the CPU (device_ref.sc) over the shared memory, once the GPU is
// done; 0 (the op computes on the CPU through the tensor layer) for 16-bit
// arguments or a table of handles, which device_ref can't read.
static int dmps_cpu_launch_(int kernel, unsigned long n, const &DevArgs args, int tagged) {
    device_note_cpu_launch_(DEVICE_MPS);
    if (tagged || kernel == DK_OPT_MULTI || kernel == DK_AMP_UNSCALE_MULTI) return 0;
    if (!dmps_wait_()) return 0;
    struct DevArgs h = *args;
    unsafe {
        unsigned long i = 0UL;
        while (i < (unsigned long)DEVK_PTRS) { h.p[i] = (unsigned long)dmps_host_(args.p[i]); i = i + 1UL; }
    }
    return device_ref_launch_(kernel, n, &h);
}

struct MpsBackend {
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

int MpsBackend::available() { return dmps_ensure_(); }

unsigned long MpsBackend::alloc(unsigned long bytes) {
    if (!dmps_ensure_()) return 0UL;
    unsafe {
        if (g_dmps_nbufs == g_dmps_capbufs) {
            unsigned long cap = g_dmps_capbufs == 0UL ? 256UL : g_dmps_capbufs * 2UL;
            struct DmpsBuf_* bufs = (struct DmpsBuf_*)malloc(sizeof(struct DmpsBuf_) * cap);
            if (bufs == (struct DmpsBuf_*)0) return 0UL;
            if (g_dmps_nbufs > 0UL) memcpy((void*)bufs, (const void*)g_dmps_bufs, sizeof(struct DmpsBuf_) * g_dmps_nbufs);
            if (g_dmps_bufs != (struct DmpsBuf_*)0) free((void*)g_dmps_bufs);
            g_dmps_bufs = bufs;
            g_dmps_capbufs = cap;
        }
        DmpsMsgPUU newBuf = (DmpsMsgPUU)objc_msgSend;
        void* buf = newBuf(g_dmps_dev, sel_registerName("newBufferWithLength:options:"), bytes, DMPS_SHARED_UNTRACKED);
        if (buf == (void*)0) return 0UL;
        DmpsMsgU mu = (DmpsMsgU)objc_msgSend;
        DmpsMsg0 m0 = (DmpsMsg0)objc_msgSend;
        unsigned long addr = mu(buf, sel_registerName("gpuAddress"));
        unsigned char* cpu = (unsigned char*)m0(buf, sel_registerName("contents"));
        if (addr == 0UL || dev_tag_(addr) != 0UL || cpu == (unsigned char*)0) { dmps_release_obj_(buf); return 0UL; }
        if (g_dmps_rset != (void*)0) {
            DmpsMsgV1 m1v = (DmpsMsgV1)objc_msgSend;
            DmpsMsgV mv = (DmpsMsgV)objc_msgSend;
            m1v(g_dmps_rset, sel_registerName("addAllocation:"), buf);
            mv(g_dmps_rset, sel_registerName("commit"));
        } else if (g_dmps_enc != (void*)0) {
            DmpsMsgV1U use = (DmpsMsgV1U)objc_msgSend;
            use(g_dmps_enc, sel_registerName("useResource:usage:"), buf, 3UL);
        }
        // insert, keeping the table sorted by address
        unsigned long at = g_dmps_nbufs;
        while (at > 0UL && g_dmps_bufs[at - 1UL].addr > addr) { g_dmps_bufs[at] = g_dmps_bufs[at - 1UL]; at = at - 1UL; }
        g_dmps_bufs[at].addr = addr;
        g_dmps_bufs[at].bytes = bytes;
        g_dmps_bufs[at].buf = buf;
        g_dmps_bufs[at].cpu = cpu;
        g_dmps_nbufs = g_dmps_nbufs + 1UL;
        return addr;
    }
}

void MpsBackend::release(unsigned long handle) {
    if (handle == 0UL) return;
    long i = dmps_find_(handle);
    if (i < 0L) return;
    if (dmps_busy_()) dmps_wait_();
    unsafe {
        void* buf = g_dmps_bufs[i].buf;
        if (g_dmps_rset != (void*)0) {
            DmpsMsgV1 m1v = (DmpsMsgV1)objc_msgSend;
            DmpsMsgV mv = (DmpsMsgV)objc_msgSend;
            m1v(g_dmps_rset, sel_registerName("removeAllocation:"), buf);
            mv(g_dmps_rset, sel_registerName("commit"));
        }
        dmps_release_obj_(buf);
        unsigned long j = (unsigned long)i;
        while (j + 1UL < g_dmps_nbufs) { g_dmps_bufs[j] = g_dmps_bufs[j + 1UL]; j = j + 1UL; }
        g_dmps_nbufs = g_dmps_nbufs - 1UL;
    }
}

int MpsBackend::upload(unsigned long dst, const &float src, unsigned long n) {
    if (n == 0UL) return 1;
    unsigned char* p = dmps_host_(dst);
    if (p == (unsigned char*)0) return 0;
    if (dmps_busy_() && !dmps_wait_()) return 0;   // queued work may still read or write dst
    unsafe { memcpy((void*)p, (const void*)src, n * 4UL); }
    return 1;
}

int MpsBackend::download(&float dst, unsigned long src, unsigned long n) {
    if (n == 0UL) return 1;
    unsigned char* p = dmps_host_(src);
    if (p == (unsigned char*)0 || !dmps_wait_()) return 0;
    unsafe { memcpy((void*)dst, (const void*)p, n * 4UL); }
    return 1;
}

int MpsBackend::copy(unsigned long dst, unsigned long src, unsigned long n) {
    if (n == 0UL) return 1;
    if (!dmps_ensure_()) return 0;
    if (g_dmps_copy == (void*)0) g_dmps_copy = dmps_make_pipe_(0, "mk_copy", DMPS_BLOCK);
    if (g_dmps_copy == (void*)0) return 0;
    struct DevArgs a = dev_args_();
    unsafe {
        a.p[0] = src; a.p[2] = dst; a.u[7] = (unsigned int)n;
        return dmps_dispatch_(g_dmps_copy, (const void*)&a, sizeof(struct DevArgs), (n + DMPS_BLOCK - 1UL) / DMPS_BLOCK, 1UL,
                              1UL, DMPS_BLOCK);
    }
}

int MpsBackend::sync() {
    if (!dmps_ensure_()) return 0;
    return dmps_wait_();
}

int MpsBackend::gemm(int ta, int tb, unsigned long m, unsigned long n, unsigned long k, float alpha,
                     unsigned long a, unsigned long lda, unsigned long b, unsigned long ldb,
                     float beta, unsigned long c, unsigned long ldc) {
    struct DmpsGemm_ g;
    g.a = a; g.b = b; g.c = c; g.lda = lda; g.ldb = ldb; g.ldc = ldc; g.sa = 0UL; g.sb = 0UL; g.sc = 0UL;
    g.m = (unsigned int)m; g.n = (unsigned int)n; g.k = (unsigned int)k; g.ta = (unsigned int)ta; g.tb = (unsigned int)tb;
    g.at = 0U; g.ct = 0U; g.alpha = alpha; g.beta = beta;
    return dmps_gemm_(&g, 1UL);
}

int MpsBackend::gemm_batched(const &DevGemm g) {
    struct DmpsGemm_ r;
    r.a = g.a; r.b = g.b; r.c = g.c; r.lda = g.lda; r.ldb = g.ldb; r.ldc = g.ldc; r.sa = g.sa; r.sb = g.sb; r.sc = g.sc;
    r.m = (unsigned int)g.m; r.n = (unsigned int)g.n; r.k = (unsigned int)g.k;
    r.ta = (unsigned int)g.ta; r.tb = (unsigned int)g.tb; r.at = (unsigned int)g.dtype; r.ct = (unsigned int)g.ctype;
    r.alpha = g.alpha; r.beta = g.beta;
    return dmps_gemm_(&r, g.batch);
}

int MpsBackend::conv(const &DevConv c) { return 0; }
int MpsBackend::storage16() { return 1; }

int MpsBackend::launch(int kernel, unsigned long n, const &DevArgs args) {
    if (!dmps_ensure_() || kernel < 0 || kernel >= DK_COUNT) return 0;
    int tagged = 0;
    unsafe {
        unsigned long i = 0UL;
        while (i < (unsigned long)DEVK_PTRS) { if (dev_tag_(args.p[i]) != 0UL) tagged = 1; i = i + 1UL; }
    }
    void* pipe = dmps_pipe_(kernel, tagged);
    if (pipe == (void*)0) return dmps_cpu_launch_(kernel, n, args, tagged);
    unsigned long threads = dmps_block_(kernel);
    unsigned long groups = (n + DMPS_BLOCK - 1UL) / DMPS_BLOCK;
    if (groups > DMPS_MAX_GROUPS) groups = DMPS_MAX_GROUPS;
    if (groups == 0UL) groups = 1UL;
    if (threads != DMPS_BLOCK) {
        // a block per (head, tile of 8 rows): rows are keys for dk_attn_bwd_kv, else queries
        unsafe {
            unsigned long rows = kernel == DK_ATTN_BWD_KV ? (unsigned long)args.u[1] : (unsigned long)args.u[0];
            groups = (unsigned long)args.u[2] * ((rows + DMPS_ATTN_T - 1UL) / DMPS_ATTN_T);
        }
        if (groups > 65535UL) groups = 65535UL;
        if (groups == 0UL) groups = 1UL;
    }
    unsafe { return dmps_dispatch_(pipe, (const void*)args, sizeof(struct DevArgs), groups, 1UL, 1UL, threads); }
}

unsigned long MpsBackend::offset(unsigned long handle, unsigned long bytes) { return handle + bytes; }

void device_mps_register_() {
    struct MpsBackend tag;
    tag.unused_ = 0;
    device_register_backend_(DEVICE_MPS, &tag);
}

} // namespace std
