// SafeC Standard Library — the Metal (Apple Silicon) device backend (see device.h).
// MpsBackend: device memory is MTLBuffers in shared storage, named by their
// GPU addresses (MTLBuffer.gpuAddress) the way ROCm and CUDA name theirs, so
// a handle plus a byte offset is an address and the kernels' raw-address
// arguments (DevArgs.p, the optimizer's tables of handles) work as they do
// there. Every buffer joins one MTLResidencySet attached to gpu_mps.h's
// queue, which keeps it resident for kernels that reach it only through an
// address. device_kernels.cu is compiled at run time by newLibraryWithSource
// (plain, and with DK_TAGGED or DK_ATTN64 on first use), and its Metal-only
// kernels run what rocBLAS and MIOpen do on ROCm: GEMMs (mk_gemm, simdgroup
// matrices), convolutions as implicit GEMMs (mk_conv_*, through conv()) and
// copies (mk_copy, mk_upload).
//
// Work is encoded into one command buffer at a time (one compute encoder, a
// memory barrier after each dispatch) and committed every DMPS_COMMIT_EVERY
// dispatches without waiting; each command buffer waits for an MTLEvent its
// predecessor signals, so they run in order although the GPU sees their
// buffers only through addresses. The host waits only when it reads
// (download, sync), before releasing memory, and before an upload too large
// to travel with a dispatch while work is queued. A program enables it with
// its mps feature; link -framework Metal -framework Foundation -lobjc. The
// kernels need float atomics (Apple GPU family 7, M1 on; elsewhere the
// source doesn't compile and they fall back as below), residency sets need
// macOS 15 (before it, each encoder makes every buffer resident with
// useResources).
//
// Limits, against ROCm / CUDA:
//   - Metal has no doubles: the batch / group norm statistics accumulate in
//     fp32 (two-pass, so they hold device_check's tolerance).
//   - The fused attention kernels work in 8 x 8 tiles (64 threads), not
//     16 x 16, for head dims over 64: four 16-row tiles of 128 floats
//     overflow 32 KB of threadgroup memory (a second build, DK_ATTN64, has
//     16 x 16 tiles for head dims up to 64).
//   - A kernel runs on the CPU (device_ref.sc, on the shared memory; counted
//     in DeviceStats.cpuLaunches) only if its pipeline can't be built; one
//     with 16-bit arguments or a table of handles then declines instead, as
//     do GEMMs and copies (the tensor op computes on the CPU; also counted).
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
#define DMPS_ATTN_T 8UL            // device_kernels.cu's ATTN_T on Metal ...
#define DMPS_ATTN64_T 16UL         // ... and in its DK_ATTN64 build, for head dims up to 64
#define DMPS_GEMM_FILL 32UL        // the bigger GEMM / conv tiles from this many tiles on (two per GPU core)
#define DMPS_RED_BYTES 128UL       // dk_red: a float per simdgroup
#define DMPS_INLINE_BYTES 4096UL   // uploads up to this size travel with a dispatch (setBytes)
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
static void* g_dmps_lib[4];             // builds: plain, DK_TAGGED, and both with DK_ATTN64
static int   g_dmps_lib_state[4];
static void* g_dmps_pipe[2 * DK_COUNT];
static int   g_dmps_pipe_state[2 * DK_COUNT];   // 0 = not yet, 1 = built, -1 = can't
static void* g_dmps_apipe[6];           // the attention kernels of the DK_ATTN64 builds
static int   g_dmps_apipe_state[6];
static void* g_dmps_gemm[24];           // mk_gemm<tile>_<op(a)><op(b)>_<type>
static int   g_dmps_gemm_state[24];
static void* g_dmps_conv[36];           // mk_conv_<op><shape>_<type>
static int   g_dmps_conv_state[36];
static void* g_dmps_copy = (void*)0;
static void* g_dmps_upload = (void*)0;

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
// hiprtc and NVRTC compile it. Build b: bit 0 DK_TAGGED (16-bit arguments),
// bit 1 DK_ATTN64 (16 x 16 attention tiles for head dims up to 64).
static int dmps_library_(int b) {
    if (g_dmps_lib_state[b] != 0) return g_dmps_lib_state[b] == 1;
    g_dmps_lib_state[b] = -1;
    unsafe {
        const char* src = device_kernel_source_();
        char* text = (char*)0;
        if (b != 0) {
            const char* pre = b == 1 ? "#define DK_TAGGED 1\n"
                            : (b == 2 ? "#define DK_ATTN64 1\n" : "#define DK_TAGGED 1\n#define DK_ATTN64 1\n");
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
        g_dmps_lib[b] = lib;
    }
    g_dmps_lib_state[b] = 1;
    return 1;
}

// The pipeline for function 'name' of build b, null when it can't be built
// for 'threads' threads per threadgroup.
static void* dmps_make_pipe_(int b, const char* name, unsigned long threads) {
    if (!dmps_library_(b)) return (void*)0;
    unsafe {
        void* pool = objc_autoreleasePoolPush();
        DmpsMsg0 m0 = (DmpsMsg0)objc_msgSend;
        DmpsMsg1 m1 = (DmpsMsg1)objc_msgSend;
        DmpsMsg2 m2 = (DmpsMsg2)objc_msgSend;
        void* str = m1(m0(objc_getClass("NSString"), sel_registerName("alloc")), sel_registerName("initWithUTF8String:"),
                       (void*)name);
        void* f = m1(g_dmps_lib[b], sel_registerName("newFunctionWithName:"), str);
        dmps_release_obj_(str);
        void* pipe = (void*)0;
        if (f != (void*)0) {
            void* err = (void*)0;
            pipe = m2(g_dmps_dev, sel_registerName("newComputePipelineStateWithFunction:error:"), f, (void*)&err);
            if (pipe == (void*)0) dmps_log_("device_mps: a kernel's pipeline failed: ", err);
            dmps_release_obj_(f);
        } else {
            unsigned long n = 0UL;
            while (name[n] != (char)0) n = n + 1UL;
            write(2, (const void*)"device_mps: no such kernel: ", 28UL);
            write(2, (const void*)name, n);
            write(2, (const void*)"\n", 1UL);
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

// The kernels with a simdgroup per row on Metal (device_kernels.cu's GSR):
// 32 threads for each of the n rows.
static int dmps_rows_(int kernel) {
    return kernel == DK_ROWSUM || kernel == DK_SOFTMAX_ROWS || kernel == DK_SOFTMAX_BWD || kernel == DK_CE_FWD ||
           kernel == DK_CE_BWD || kernel == DK_LAYERNORM || kernel == DK_LAYERNORM_BWD;
}

// Threads per threadgroup for a DK_* kernel of the first two builds.
static unsigned long dmps_block_(int kernel) {
    if (kernel == DK_ATTN_FWD || kernel == DK_ATTN_BWD_Q || kernel == DK_ATTN_BWD_KV) return DMPS_ATTN_T * DMPS_ATTN_T;
    return DMPS_BLOCK;
}

// The fused attention kernels with 16 x 16 tiles (the DK_ATTN64 builds), when
// the head dims allow; null otherwise.
static void* dmps_attn64_pipe_(int kernel, int tagged, const &DevArgs args) {
    if (kernel != DK_ATTN_FWD && kernel != DK_ATTN_BWD_Q && kernel != DK_ATTN_BWD_KV) return (void*)0;
    unsafe {
        if (args.u[3] > 64U || args.u[4] > 64U) return (void*)0;
        unsigned long slot = (unsigned long)tagged * 3UL + (unsigned long)(kernel - DK_ATTN_FWD);
        if (g_dmps_apipe_state[slot] == 0) {
            g_dmps_apipe[slot] = dmps_make_pipe_(2 + tagged, device_kernel_name_(kernel), DMPS_ATTN64_T * DMPS_ATTN64_T);
            g_dmps_apipe_state[slot] = g_dmps_apipe[slot] != (void*)0 ? 1 : -1;
        }
        return g_dmps_apipe[slot];
    }
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
// 'args' (len bytes) at buffer 0 and 'extra' (extraLen bytes, if any) at
// buffer 1; then a barrier, so the next one sees its writes.
static int dmps_dispatch_ex_(void* pipe, const void* args, unsigned long len, const void* extra, unsigned long extraLen,
                             unsigned long groupsX, unsigned long groupsY, unsigned long groupsZ, unsigned long threads) {
    if (!dmps_begin_()) return 0;
    unsafe {
        if (pipe != g_dmps_last_pipe) {
            DmpsMsgV1 setPipe = (DmpsMsgV1)objc_msgSend;
            setPipe(g_dmps_enc, s_dmps_setPipe, pipe);
            g_dmps_last_pipe = pipe;
        }
        DmpsMsgBytes setBytes = (DmpsMsgBytes)objc_msgSend;
        setBytes(g_dmps_enc, s_dmps_setBytes, args, len, 0UL);
        if (extraLen > 0UL) setBytes(g_dmps_enc, s_dmps_setBytes, extra, extraLen, 1UL);
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

static int dmps_dispatch_(void* pipe, const void* args, unsigned long len, unsigned long groupsX, unsigned long groupsY,
                          unsigned long groupsZ, unsigned long threads) {
    unsafe { return dmps_dispatch_ex_(pipe, args, len, (const void*)0, 0UL, groupsX, groupsY, groupsZ, threads); }
}

// The GEMM kernel for g: 64 x 64 tiles when there are enough of them to fill
// the GPU and b isn't transposed (its 16-deep tiles read b^T in short runs),
// else 32 x 32; one pipeline per tile, transposes and element type.
static int dmps_gemm_(const &DmpsGemm_ g, unsigned long batch) {
    if (!dmps_ensure_()) return 0;
    if (g.m == 0U || g.n == 0U || batch == 0UL) return 1;
    unsigned long m = (unsigned long)g.m;
    unsigned long n = (unsigned long)g.n;
    unsigned long tile = 32UL;
    if (g.tb == 0U && ((m + 63UL) / 64UL) * ((n + 63UL) / 64UL) * batch >= DMPS_GEMM_FILL) tile = 64UL;
    unsigned long slot = (tile == 64UL ? 12UL : 0UL) + (unsigned long)g.at * 4UL + (unsigned long)g.ta + 2UL * (unsigned long)g.tb;
    if (g.at > 2U || slot >= 24UL) return 0;
    unsafe {
        if (g_dmps_gemm_state[slot] == 0) {
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
            g_dmps_gemm[slot] = dmps_make_pipe_(0, (const char*)&name[0], 128UL);
            g_dmps_gemm_state[slot] = g_dmps_gemm[slot] != (void*)0 ? 1 : -1;
        }
        if (g_dmps_gemm[slot] == (void*)0) { device_note_cpu_launch_(DEVICE_MPS); return 0; }   // the tensor op runs on the CPU
        return dmps_dispatch_(g_dmps_gemm[slot], (const void*)g, sizeof(struct DmpsGemm_), (n + tile - 1UL) / tile,
                              (m + tile - 1UL) / tile, batch, 128UL);
    }
}

// mk_conv_*'s DkConv: a plain convolution I [B, Ci, Hi, Wi] -> O [B, Co, Ho, Wo]
// with W [Co, Ci, kH, kW].
struct DmpsConv_ {
    unsigned long x; unsigned long w; unsigned long y;
    unsigned int B; unsigned int Ci; unsigned int Hi; unsigned int Wi; unsigned int Co; unsigned int Ho; unsigned int Wo;
    unsigned int kH; unsigned int kW; unsigned int sH; unsigned int sW; unsigned int pH; unsigned int pW;
    unsigned int dH; unsigned int dW;
    unsigned int kSplit;
};

// Conv kernel tile shapes (device_kernels.cu's mk_conv_*): rows x columns
// of the product per threadgroup.
#define DMPS_SHAPE_32 0   // 32 x 32
#define DMPS_SHAPE_64 1   // 64 x 64
#define DMPS_SHAPE_W  2   // 32 x 128
#define DMPS_SHAPE_S  3   // 8 x 128: up to 8 rows

static unsigned long dmps_shape_rows_(int shape) { return shape == DMPS_SHAPE_64 ? 64UL : (shape == DMPS_SHAPE_S ? 8UL : 32UL); }
static unsigned long dmps_shape_cols_(int shape) { return shape == DMPS_SHAPE_64 ? 64UL : (shape == DMPS_SHAPE_32 ? 32UL : 128UL); }

// The shape for an m x n product (op 2: the weight gradient): 32 x 128 is
// the fastest wherever it leaves the GPU enough threadgroups (2 TFLOP/s
// on an M1 Pro), 8 x 128 for one to eight rows; the weight gradient's n
// is short, so it takes 64 x 64 from 64 rows on, else 32 x 32.
static int dmps_conv_shape_(int op, unsigned long m, unsigned long n) {
    if (op == 2) return m >= 64UL ? DMPS_SHAPE_64 : DMPS_SHAPE_32;
    if (m <= 8UL && (n + 127UL) / 128UL >= DMPS_GEMM_FILL) return DMPS_SHAPE_S;
    if (((n + 127UL) / 128UL) * ((m + 31UL) / 32UL) >= DMPS_GEMM_FILL) return DMPS_SHAPE_W;
    return m >= 64UL && ((m + 63UL) / 64UL) * ((n + 63UL) / 64UL) >= DMPS_GEMM_FILL ? DMPS_SHAPE_64 : DMPS_SHAPE_32;
}

// The pipeline for conv kernel op (0 fwd, 1 bwd_x, 2 bwd_w), a shape, type t.
static void* dmps_conv_pipe_(int op, int shape, int t) {
    unsigned long slot = ((unsigned long)op * 4UL + (unsigned long)shape) * 3UL + (unsigned long)t;
    unsafe {
        if (g_dmps_conv_state[slot] == 0) {
            char name[40];
            const char* base = op == 0 ? "mk_conv_fwd" : (op == 1 ? "mk_conv_bwd_x" : "mk_conv_bwd_w");
            const char* sh = shape == DMPS_SHAPE_32 ? "32_" : (shape == DMPS_SHAPE_64 ? "64_" : (shape == DMPS_SHAPE_W ? "w_" : "s_"));
            const char* sfx = t == 0 ? "f32" : (t == 1 ? "f16" : "bf16");
            unsigned long i = 0UL;
            while (base[i] != (char)0) { name[i] = base[i]; i = i + 1UL; }
            unsigned long j = 0UL;
            while (sh[j] != (char)0) { name[i] = sh[j]; i = i + 1UL; j = j + 1UL; }
            j = 0UL;
            while (sfx[j] != (char)0) { name[i] = sfx[j]; i = i + 1UL; j = j + 1UL; }
            name[i] = (char)0;
            g_dmps_conv[slot] = dmps_make_pipe_(0, (const char*)&name[0], 128UL);
            g_dmps_conv_state[slot] = g_dmps_conv[slot] != (void*)0 ? 1 : -1;
        }
        return g_dmps_conv[slot];
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

// Into idle memory, a memcpy; while work is queued, a small upload (a
// batch's row indices, a seed gradient) travels with a dispatch that
// writes it in order, and a large one waits for the queue first.
int MpsBackend::upload(unsigned long dst, const &float src, unsigned long n) {
    if (n == 0UL) return 1;
    unsigned char* p = dmps_host_(dst);
    if (p == (unsigned char*)0) return 0;
    if (dmps_busy_() && n * 4UL <= DMPS_INLINE_BYTES) {
        if (g_dmps_upload == (void*)0) g_dmps_upload = dmps_make_pipe_(0, "mk_upload", DMPS_BLOCK);
        if (g_dmps_upload != (void*)0) {
            struct DevArgs a = dev_args_();
            unsafe {
                a.p[2] = dst; a.u[7] = (unsigned int)n;
                return dmps_dispatch_ex_(g_dmps_upload, (const void*)&a, sizeof(struct DevArgs), (const void*)src, n * 4UL,
                                         (n + DMPS_BLOCK - 1UL) / DMPS_BLOCK, 1UL, 1UL, DMPS_BLOCK);
            }
        }
    }
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
    if (g_dmps_copy == (void*)0) { device_note_cpu_launch_(DEVICE_MPS); return 0; }
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

// Convolutions as implicit GEMMs (device_kernels.cu's mk_conv_*): the
// forward pass, the input gradient as a gather over the windows (no
// atomics), the weight gradient split over batch x pixels and added
// atomically into zeroed weights (fp32 only: a 16-bit one declines and
// device.sc runs it in fp32). A transposed convolution is the plain one with
// input and output swapped, so its forward pass is the input-gradient kernel
// and its input gradient the forward one.
int MpsBackend::conv(const &DevConv c) {
    if (!dmps_ensure_() || c.dtype < DEVICE_FP32 || c.dtype > DEVICE_BF16) return 0;
    struct DmpsConv_ k;
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
    // the product's m and n
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
    int shape = dmps_conv_shape_(op, m, n * phases);
    void* pipe = dmps_conv_pipe_(op, shape, c.dtype);
    if (pipe == (void*)0) { device_note_cpu_launch_(DEVICE_MPS); return 0; }   // im2col + GEMM instead
    unsigned long groupsX = (n + dmps_shape_cols_(shape) - 1UL) / dmps_shape_cols_(shape);
    unsigned long groupsY = (m + dmps_shape_rows_(shape) - 1UL) / dmps_shape_rows_(shape);
    unsigned long splits = phases;
    if (op == 2) {
        // zero the weights, then enough slices of k to fill the GPU
        unsigned long nw = (unsigned long)k.Co * (unsigned long)k.Ci * kk;
        struct DevArgs z = dev_args_();
        unsafe { z.p[2] = c.w; z.u[0] = 5U; z.u[7] = (unsigned int)nw; }   // EW_FILL 0
        if (!MpsBackend::launch(DK_EWOP_5, nw, &z)) return 0;
        unsigned long kTot = c.batch * (unsigned long)k.Ho * (unsigned long)k.Wo;
        splits = (128UL + groupsX * groupsY - 1UL) / (groupsX * groupsY);
        if (splits > kTot / 256UL) splits = kTot / 256UL;
        if (splits < 1UL) splits = 1UL;
        unsigned long per = (kTot + splits - 1UL) / splits;
        per = (per + 31UL) / 32UL * 32UL;
        k.kSplit = (unsigned int)per;
        splits = (kTot + per - 1UL) / per;
    }
    unsafe { return dmps_dispatch_(pipe, (const void*)&k, sizeof(struct DmpsConv_), groupsX, groupsY, splits, 128UL); }
}
int MpsBackend::storage16() { return 1; }

int MpsBackend::launch(int kernel, unsigned long n, const &DevArgs args) {
    if (!dmps_ensure_() || kernel < 0 || kernel >= DK_COUNT) return 0;
    int tagged = 0;
    unsafe {
        unsigned long i = 0UL;
        while (i < (unsigned long)DEVK_PTRS) { if (dev_tag_(args.p[i]) != 0UL) tagged = 1; i = i + 1UL; }
    }
    unsigned long tile = DMPS_ATTN64_T;
    void* pipe = dmps_attn64_pipe_(kernel, tagged, args);
    if (pipe == (void*)0) { pipe = dmps_pipe_(kernel, tagged); tile = DMPS_ATTN_T; }
    if (pipe == (void*)0) return dmps_cpu_launch_(kernel, n, args, tagged);
    int attn = kernel == DK_ATTN_FWD || kernel == DK_ATTN_BWD_Q || kernel == DK_ATTN_BWD_KV;
    unsigned long threads = attn ? tile * tile : DMPS_BLOCK;
    unsigned long groups = ((dmps_rows_(kernel) ? 32UL * n : n) + DMPS_BLOCK - 1UL) / DMPS_BLOCK;
    if (groups > DMPS_MAX_GROUPS) groups = DMPS_MAX_GROUPS;
    if (groups == 0UL) groups = 1UL;
    if (attn) {
        // a block per (head, tile of rows): rows are keys for dk_attn_bwd_kv, else queries
        unsafe {
            unsigned long rows = kernel == DK_ATTN_BWD_KV ? (unsigned long)args.u[1] : (unsigned long)args.u[0];
            groups = (unsigned long)args.u[2] * ((rows + tile - 1UL) / tile);
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
