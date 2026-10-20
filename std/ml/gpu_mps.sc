// SafeC Standard Library — Metal GPU backend implementation (see
// gpu_mps.h). See std/gui/gui_cocoa.sc's header comment for the general
// "no native Objective-C support" technique this reuses. Every
// objc_msgSend reinterpret-cast is assigned to a named variable before
// being called (rather than called inline as '((SomeMsgSend)objc_msgSend)
// (...)') — the inline form compiles but a real link/run against Metal
// showed it silently keeps objc_msgSend's *original* 2-argument type for
// the call site instead of the cast's, producing a wrong-arg-count LLVM
// verifier failure; the assign-then-call form (matching gui_cocoa.sc's
// existing convention throughout) reliably picks up the cast type.
//
// Three things this file used to do at runtime, on every process or every
// dispatch, that are now either done once or not at all:
//   1. Kernel compilation. Every mps_*_f32 call used to hand a Metal
//      Shading Language *source string* to newLibraryWithSource:options:
//      error:, a real MSL-to-AIR compile. gpu_mps_kernels.metal is now
//      compiled ahead of time by gen_mps_metallib.sh into a .metallib,
//      embedded as a byte array (gpu_mps_metallib.h), and loaded once via
//      newLibraryWithData:error: — no MSL compiler runs in-process.
//   2. Selector lookup. Every objc_msgSend call used to re-resolve its
//      selector via sel_registerName(name) on every single dispatch,
//      resolved once now (__mps_init_selectors) into static SEL variables.
//   3. Buffer allocation — the one that actually mattered. Every op used
//      to call newBufferWithBytes:length:options:/newBufferWithLength:
//      options: fresh, on *every single dispatch*, and never released
//      them: a real Metal buffer allocation (IOSurface/IOAccelerator
//      resource creation, not a cheap malloc) on every op call, growing
//      without bound over a training loop. Measured directly: this, not
//      shader compilation, was the actual dominant per-call cost once
//      pipelines were already cached — fixing (1)/(2) alone left the
//      bigger-model training loop's GPU time unchanged, because neither
//      addressed the thing actually being paid for on every op.
//      PyTorch's MPSAllocator and MLX's Metal buffer cache both solve
//      this the same way: a caching allocator over MTLBuffer objects,
//      keyed by size, so a same-shape op reuses a buffer instead of
//      asking the driver for a new one — see the size-class buffer pool
//      below (__mps_buf_get/__mps_buf_put), which applies that exact
//      pattern here, the same size-class-cache design std::alloc/dealloc
//      already uses on the CPU side (see std/mem.sc).
#pragma once
#include <std/stddef.h>
#include <std/ml/gpu_mps.h>
#include <std/ml/gpu_mps_metallib.h>
#include <std/ml/float16.h>
#include <std/mem.h>

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);

extern void* objc_getClass(const char* name);
extern void* sel_registerName(const char* name);
extern void* objc_msgSend(void* recv, void* op);
extern void* memcpy(void* dst, const void* src, size_t n);
extern void* memset(void* dst, int c, size_t n);
// The real Metal framework entry point — a plain C function (not a
// message send): returns the default MTLDevice, or NULL if this machine
// has no Metal-capable GPU.
extern void* MTLCreateSystemDefaultDevice();
// libdispatch, part of libSystem on every macOS target (no extra link
// flag needed). Wraps the embedded metallib bytes for
// newLibraryWithData:error: below. queue=NULL and destructor=NULL together
// mean DISPATCH_DATA_DESTRUCTOR_DEFAULT: dispatch_data_create copies the
// buffer into memory it manages, so no block/destructor callback is ever
// invoked — safe to call from a language with no Objective-C block
// literal support.
extern void* dispatch_data_create(const void* buffer, unsigned long size, void* queue, void* destructor);

struct MTLSize {
    unsigned long width;
    unsigned long height;
    unsigned long depth;
};

typedef fn void*(void*, void*) MMsg0;
typedef fn void*(void*, void*, void*) MMsg1;
typedef fn void*(void*, void*, void*, void*) MMsg2;
typedef fn void*(void*, void*, unsigned long, unsigned long) MMsgNewBufferLen;
typedef fn void(void*, void*, void*) MMsgVoid1;
typedef fn void(void*, void*, void*, unsigned long, unsigned long) MMsgSetBuffer;
typedef fn void(void*, void*, struct MTLSize, struct MTLSize) MMsgDispatch;
typedef fn void(void*, void*) MMsgVoid0;

// ── Cached selectors ──────────────────────────────────────────────────────
// Filled once by __mps_init_selectors (called from __mps_ensure_device_queue
// the first time any op runs), then read directly everywhere below instead
// of calling sel_registerName(name) per dispatch.
static void* __sel_alloc = (void*)0;
static void* __sel_initWithUTF8String = (void*)0;
static void* __sel_newCommandQueue = (void*)0;
static void* __sel_newLibraryWithData = (void*)0;
static void* __sel_newFunctionWithName = (void*)0;
static void* __sel_newComputePipelineState = (void*)0;
static void* __sel_newBufferWithLength = (void*)0;
static void* __sel_commandBuffer = (void*)0;
static void* __sel_computeCommandEncoder = (void*)0;
static void* __sel_setComputePipelineState = (void*)0;
static void* __sel_setBuffer = (void*)0;
static void* __sel_setBytes = (void*)0;
static void* __sel_dispatchThreadgroups = (void*)0;
static void* __sel_endEncoding = (void*)0;
static void* __sel_commit = (void*)0;
static void* __sel_waitUntilCompleted = (void*)0;
static void* __sel_contents = (void*)0;
static void* __sel_release = (void*)0;

static void __mps_init_selectors() {
    unsafe {
        __sel_alloc = sel_registerName("alloc");
        __sel_initWithUTF8String = sel_registerName("initWithUTF8String:");
        __sel_newCommandQueue = sel_registerName("newCommandQueue");
        __sel_newLibraryWithData = sel_registerName("newLibraryWithData:error:");
        __sel_newFunctionWithName = sel_registerName("newFunctionWithName:");
        __sel_newComputePipelineState = sel_registerName("newComputePipelineStateWithFunction:error:");
        __sel_newBufferWithLength = sel_registerName("newBufferWithLength:options:");
        __sel_commandBuffer = sel_registerName("commandBuffer");
        __sel_computeCommandEncoder = sel_registerName("computeCommandEncoder");
        __sel_setComputePipelineState = sel_registerName("setComputePipelineState:");
        __sel_setBuffer = sel_registerName("setBuffer:offset:atIndex:");
        __sel_setBytes = sel_registerName("setBytes:length:atIndex:");
        __sel_dispatchThreadgroups = sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:");
        __sel_endEncoding = sel_registerName("endEncoding");
        __sel_commit = sel_registerName("commit");
        __sel_waitUntilCompleted = sel_registerName("waitUntilCompleted");
        __sel_contents = sel_registerName("contents");
        __sel_release = sel_registerName("release");
    }
}

static void* __mps_nsstring(const char* cstr) {
    unsafe {
        void* cls = objc_getClass("NSString");
        MMsg0 allocFn = (MMsg0)objc_msgSend;
        void* obj = allocFn(cls, __sel_alloc);
        MMsg1 initFn = (MMsg1)objc_msgSend;
        return initFn(obj, __sel_initWithUTF8String, (void*)cstr);
    }
}

int mps_available() {
    void* dev;
    unsafe { dev = MTLCreateSystemDefaultDevice(); }
    return dev != (void*)0;
}

// Cached device/queue/library, shared by every op in this file. Lazily
// created on whichever op runs first, then reused forever.
static void* __mps_device = (void*)0;
static void* __mps_queue = (void*)0;
static void* __mps_library = (void*)0;

static int __mps_ensure_device_queue() {
    if (__mps_device != (void*)0) return 1;
    unsafe {
        void* dev = MTLCreateSystemDefaultDevice();
        if (dev == (void*)0) return 0;
        __mps_init_selectors();
        MMsg0 msg0init = (MMsg0)objc_msgSend;
        void* q = msg0init(dev, __sel_newCommandQueue);
        if (q == (void*)0) return 0;
        __mps_device = dev;
        __mps_queue = q;
        return 1;
    }
}

// Loads the whole kernel set (gpu_mps_kernels.metal, precompiled to AIR
// bytecode by gen_mps_metallib.sh and embedded in gpu_mps_metallib.h) as
// one MTLLibrary, once. Every op below then just looks up its own function
// by name from this shared library — no MSL source, no runtime compile.
static int __mps_ensure_library() {
    if (__mps_library != (void*)0) return 1;
    if (!__mps_ensure_device_queue()) return 0;
    unsafe {
        void* data = dispatch_data_create((const void*)__mps_metallib_bytes, __mps_metallib_size,
                                           (void*)0, (void*)0);
        if (data == (void*)0) return 0;
        void* errPtr = (void*)0;
        MMsg2 newLibFn = (MMsg2)objc_msgSend;
        void* library = newLibFn(__mps_device, __sel_newLibraryWithData, data, (void*)&errPtr);
        if (library == (void*)0) return 0;
        __mps_library = library;
        return 1;
    }
}

// Looks up 'kernelName' in the shared library and builds a compute
// pipeline for it. Shared by every op's pipeline-cache-fill branch below.
static void* __mps_make_pipeline(const char* kernelName) {
    unsafe {
        if (!__mps_ensure_library()) return (void*)0;
        void* fnNameStr = __mps_nsstring(kernelName);
        MMsg1 msg1 = (MMsg1)objc_msgSend;
        void* kernelFn = msg1(__mps_library, __sel_newFunctionWithName, fnNameStr);
        if (kernelFn == (void*)0) return (void*)0;
        void* errPtr = (void*)0;
        MMsg2 newPipelineFn = (MMsg2)objc_msgSend;
        void* pipeline = newPipelineFn(__mps_device, __sel_newComputePipelineState, kernelFn, (void*)&errPtr);
        return pipeline;
    }
}

// ── Size-class MTLBuffer pool ────────────────────────────────────────────────
// Same idea as std::alloc/dealloc's size-class cache (std/mem.sc), applied
// to Metal buffers instead of heap blocks: a freed buffer goes back into a
// thread-local(-ish; this file was never thread-safe — see __mps_device's
// own lack of synchronization) free list bucketed by power-of-two size
// class, and the next request for that class is satisfied from there
// instead of asking the Metal driver for a brand new buffer. Classes run
// 4KB to 8MB (12 classes) — small enough to cover every buffer this file's
// actual op set ever allocates (a batch of matmul/relu/elementwise calls
// over a training-loop-sized MLP, not arbitrary user data), large enough
// that a single class comfortably holds this bigger benchmark's biggest
// tensor (W1 at 512x1024 doubles-converted-to-float32 = 2MB). Anything
// larger just allocates directly, uncached, same as std::alloc's oversize
// path. Per-class capacity is memory-budget-scaled (roughly 128MB worth of
// buffers per class, floored at 2 slots, capped at 128 slots) rather than a
// flat count, so a class of huge buffers doesn't retain as many as a class
// of small ones — same reasoning as alloc_class_cap_ in std/mem.sc. Raised
// from an original 8-slot/16MB cap: with many independent ops sharing one
// batch (see train_gpu.sc's inference loop — dozens of matmul/relu passes
// batched together purely to amortize one command-buffer submit + wait
// across all of them), every one of those ops needs its own buffer of the
// same handful of sizes (the fixed X/W1/W2/activation shapes), and NONE of
// them can return to the pool until the whole batch's wait completes — a
// small pool that's fine for one training step's ~9 ops was measured
// forcing hundreds of real newBufferWithLength allocations (not just pool
// misses) once a single batch's own concurrent demand for one size class
// exceeded 8, which cost far more than the extra command-buffer submits it
// was meant to avoid. A slot that overflows its class's cap gets an
// explicit 'release' instead of being silently dropped (leaked): Metal
// buffers are real, non-trivial GPU resources, not a few bytes of heap, so
// an unbounded leak here is worth avoiding even in
// the overflow case std::alloc's own quarantine ring tolerates for plain
// memory.
#define MPS_BUF_MIN_CLASS_   ((unsigned long)4096)
#define MPS_BUF_NUM_CLASSES_ ((unsigned long)12)
#define MPS_BUF_MAX_SLOTS_   ((unsigned long)128)

static void*         __mps_bufSlot[12][128];
static unsigned long __mps_bufCount[12];

static long __mps_buf_class_for(unsigned long bytes) {
    unsigned long clsBytes = MPS_BUF_MIN_CLASS_;
    long idx = 0L;
    while (clsBytes < bytes) {
        if (idx >= (long)(MPS_BUF_NUM_CLASSES_ - (unsigned long)1)) { return -1L; }
        clsBytes = clsBytes << 1UL;
        idx = idx + 1L;
    }
    return idx;
}

static unsigned long __mps_buf_class_bytes(unsigned long idx) {
    return MPS_BUF_MIN_CLASS_ << idx;
}

// ~128MB budget per class, floored at 2 slots, capped at the array's
// physical width (MPS_BUF_MAX_SLOTS_) -- same shift-based, division-free
// approach as std/mem.sc's alloc_class_cap_.
static unsigned long __mps_buf_class_cap(unsigned long cls) {
    unsigned long classBytes = __mps_buf_class_bytes(cls);
    unsigned long cap = ((unsigned long)128 * (unsigned long)1024 * (unsigned long)1024) / classBytes;
    if (cap > MPS_BUF_MAX_SLOTS_) { cap = MPS_BUF_MAX_SLOTS_; }
    if (cap < (unsigned long)2) { cap = (unsigned long)2; }
    return cap;
}

// Returns an MTLBuffer of at least 'bytes', either popped from its size
// class's free list or freshly allocated (at the class's rounded-up size,
// so it's reusable by any future request that rounds to the same class).
static void* __mps_buf_get(unsigned long bytes) {
    unsafe {
        long clsSigned = __mps_buf_class_for(bytes);
        MMsgNewBufferLen newBufLenFn = (MMsgNewBufferLen)objc_msgSend;
        if (clsSigned >= 0L) {
            unsigned long cls = (unsigned long)clsSigned;
            if (__mps_bufCount[cls] > 0UL) {
                __mps_bufCount[cls] = __mps_bufCount[cls] - 1UL;
                return __mps_bufSlot[cls][__mps_bufCount[cls]];
            }
            return newBufLenFn(__mps_device, __sel_newBufferWithLength, __mps_buf_class_bytes(cls), 0UL);
        }
        return newBufLenFn(__mps_device, __sel_newBufferWithLength, bytes, 0UL);
    }
}

// Same as __mps_buf_get, but also copies 'src' into the buffer's
// CPU-visible 'contents' (storage mode 0 = MTLResourceStorageModeShared,
// set on every allocation above) -- replaces the old
// newBufferWithBytes:length:options: call, which allocated and copied in
// one step but could never be satisfied from the pool.
static void* __mps_buf_get_with_bytes(const void* src, unsigned long bytes) {
    unsafe {
        void* buf = __mps_buf_get(bytes);
        if (buf == (void*)0) return (void*)0;
        MMsg0 msg0 = (MMsg0)objc_msgSend;
        void* contents = msg0(buf, __sel_contents);
        if (contents != (void*)0) { memcpy(contents, src, bytes); }
        return buf;
    }
}

// Returns 'buf' (of size 'bytes') to its size class's free list, or
// releases it outright if that class is already at capacity. Only valid
// to call once the GPU is provably done with the buffer -- i.e. after the
// command buffer that used it has completed (waitUntilCompleted returned,
// whether that wait happened here or, for batched dispatches, in
// mps_batch_end()).
static void __mps_buf_put(void* buf, unsigned long bytes) {
    if (buf == (void*)0) return;
    unsafe {
        long clsSigned = __mps_buf_class_for(bytes);
        if (clsSigned >= 0L) {
            unsigned long cls = (unsigned long)clsSigned;
            if (__mps_bufCount[cls] < __mps_buf_class_cap(cls)) {
                __mps_bufSlot[cls][__mps_bufCount[cls]] = buf;
                __mps_bufCount[cls] = __mps_bufCount[cls] + 1UL;
                return;
            }
        }
        MMsgVoid0 releaseFn = (MMsgVoid0)objc_msgSend;
        releaseFn(buf, __sel_release);
    }
}

// Shared setup/dispatch/readback for every elementwise binary kernel
// (add/sub/mul/div/pow) — the only thing that varies between them is which
// compiled kernel function they bind. Previously, the
// dispatchThreadgroups:threadsPerThreadgroup: call inside here segfaulted
// on real Apple Silicon hardware: it passes *two* struct arguments (each
// 'struct MTLSize', 3x unsigned long = 24 bytes) by value through an
// objc_msgSend cast. Root-caused and fixed in CodeGen.cpp's genCall:
// indirect calls (through a function-pointer cast, exactly this pattern)
// now lower non-HFA struct-by-value arguments over 16 bytes to the real
// AAPCS64 form (caller copies to a stack temporary, passes a plain
// pointer) instead of a raw aggregate value.
//
// 'pipelineCache' is a pointer to the CALLER's own static void* (one per
// op — see e.g. __mps_add_pipeline below): each op needs its own compiled
// MTLComputePipelineState, but the compile-once-reuse-forever pattern is
// identical for all of them.
static int __mps_run_binary_kernel(void** pipelineCache, const char* kernelName,
                                    const float* a, const float* b, float* out, unsigned long n) {
    unsafe {
        if (!__mps_ensure_device_queue()) return 0;
        void* device = __mps_device;
        void* queue = __mps_queue;
        MMsg0 msg0 = (MMsg0)objc_msgSend;

        if (*pipelineCache == (void*)0) {
            void* pipeline = __mps_make_pipeline(kernelName);
            if (pipeline == (void*)0) return 0;
            *pipelineCache = pipeline;
        }
        void* pipeline = *pipelineCache;

        unsigned long bytes = n * sizeof(float);
        void* bufA = __mps_buf_get_with_bytes((const void*)a, bytes);
        void* bufB = __mps_buf_get_with_bytes((const void*)b, bytes);
        void* bufOut = __mps_buf_get(bytes);
        if (bufA == (void*)0 || bufB == (void*)0 || bufOut == (void*)0) return 0;

        void* cmdBuf = msg0(queue, __sel_commandBuffer);
        void* encoder = msg0(cmdBuf, __sel_computeCommandEncoder);

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(encoder, __sel_setBuffer, bufB, 0UL, 1UL);
        setBufFn(encoder, __sel_setBuffer, bufOut, 0UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);
        MMsgVoid0 msgVoid0 = (MMsgVoid0)objc_msgSend;
        msgVoid0(encoder, __sel_endEncoding);
        msgVoid0(cmdBuf, __sel_commit);
        msgVoid0(cmdBuf, __sel_waitUntilCompleted);

        void* outPtr = msg0(bufOut, __sel_contents);
        if (outPtr == (void*)0) return 0;
        memcpy((void*)out, outPtr, bytes);
        __mps_buf_put(bufA, bytes);
        __mps_buf_put(bufB, bytes);
        __mps_buf_put(bufOut, bytes);
        return 1;
    }
}

// Same as __mps_run_binary_kernel but for a single-input elementwise kernel
// (log/exp/sqrt) — one input buffer instead of two, otherwise identical
// setup/dispatch/readback.
static int __mps_run_unary_kernel(void** pipelineCache, const char* kernelName,
                                   const float* a, float* out, unsigned long n) {
    unsafe {
        if (!__mps_ensure_device_queue()) return 0;
        void* device = __mps_device;
        void* queue = __mps_queue;
        MMsg0 msg0 = (MMsg0)objc_msgSend;

        if (*pipelineCache == (void*)0) {
            void* pipeline = __mps_make_pipeline(kernelName);
            if (pipeline == (void*)0) return 0;
            *pipelineCache = pipeline;
        }
        void* pipeline = *pipelineCache;

        unsigned long bytes = n * sizeof(float);
        void* bufA = __mps_buf_get_with_bytes((const void*)a, bytes);
        void* bufOut = __mps_buf_get(bytes);
        if (bufA == (void*)0 || bufOut == (void*)0) return 0;

        void* cmdBuf = msg0(queue, __sel_commandBuffer);
        void* encoder = msg0(cmdBuf, __sel_computeCommandEncoder);

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(encoder, __sel_setBuffer, bufOut, 0UL, 1UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);
        MMsgVoid0 msgVoid0 = (MMsgVoid0)objc_msgSend;
        msgVoid0(encoder, __sel_endEncoding);
        msgVoid0(cmdBuf, __sel_commit);
        msgVoid0(cmdBuf, __sel_waitUntilCompleted);

        void* outPtr = msg0(bufOut, __sel_contents);
        if (outPtr == (void*)0) return 0;
        memcpy((void*)out, outPtr, bytes);
        __mps_buf_put(bufA, bytes);
        __mps_buf_put(bufOut, bytes);
        return 1;
    }
}

// One compiled-pipeline cache per elementwise op, each fed to
// __mps_run_binary_kernel/__mps_run_unary_kernel by address so it's filled
// (once) and reused (every call after) independently of every other op's
// cache.
static void* __mps_add_pipeline  = (void*)0;
static void* __mps_sub_pipeline  = (void*)0;
static void* __mps_mul_pipeline  = (void*)0;
static void* __mps_div_pipeline  = (void*)0;
static void* __mps_pow_pipeline  = (void*)0;
static void* __mps_log_pipeline  = (void*)0;
static void* __mps_exp_pipeline  = (void*)0;
static void* __mps_sqrt_pipeline = (void*)0;
static void* __mps_relu_backward_pipeline = (void*)0;

int mps_add_f32(const &float a, const &float b, &float out, unsigned long n) {
    return __mps_run_binary_kernel(&__mps_add_pipeline, "add_kernel", a, b, out, n);
}

// mps_relu_backward_f32's own definition (batch-aware, not routed through
// __mps_run_binary_kernel) lives further down, next to the matmul-backward
// pair it's used alongside in a real backward pass -- see that section's
// header comment.

// mps_sub_f32's own definition (batch-aware, not routed through
// __mps_run_binary_kernel) lives further down, next to mps_matmul_f32 and
// the batching primitives it needs to reference -- same reasoning as
// mps_relu_f32/mps_relu_backward_f32 above (see mps_relu_f32's comment).

int mps_mul_f32(const &float a, const &float b, &float out, unsigned long n) {
    return __mps_run_binary_kernel(&__mps_mul_pipeline, "mul_kernel", a, b, out, n);
}

int mps_div_f32(const &float a, const &float b, &float out, unsigned long n) {
    return __mps_run_binary_kernel(&__mps_div_pipeline, "div_kernel", a, b, out, n);
}

int mps_pow_f32(const &float a, const &float b, &float out, unsigned long n) {
    return __mps_run_binary_kernel(&__mps_pow_pipeline, "pow_kernel", a, b, out, n);
}

int mps_log_f32(const &float a, &float out, unsigned long n) {
    return __mps_run_unary_kernel(&__mps_log_pipeline, "log_kernel", a, out, n);
}

int mps_exp_f32(const &float a, &float out, unsigned long n) {
    return __mps_run_unary_kernel(&__mps_exp_pipeline, "exp_kernel", a, out, n);
}

int mps_sqrt_f32(const &float a, &float out, unsigned long n) {
    return __mps_run_unary_kernel(&__mps_sqrt_pipeline, "sqrt_kernel", a, out, n);
}

// mps_relu_f32's own definition lives further down, alongside
// mps_matmul_f32 and the batching primitives it (like matmul) needs to
// check — see the comment there for why it's not routed through
// __mps_run_unary_kernel like log/exp/sqrt above it.

// out[i] = a[i] * k for a compile-time-unknown scalar k -- same shape as
// __mps_run_unary_kernel but with one extra setBytes:length:atIndex: call
// to pass k into buffer index 1 (out moves to index 2) before dispatch --
// kept as its own function (rather than folded into __mps_run_unary_kernel)
// because of that extra setBytes call.
static void* __mps_scale_pipeline = (void*)0;
static void* __mps_gated_gelu_pipeline = (void*)0;

int mps_scale_f32(const &float a, float k, &float out, unsigned long n) {
    unsafe {
        if (!__mps_ensure_device_queue()) return 0;
        void* device = __mps_device;
        void* queue = __mps_queue;
        MMsg0 msg0 = (MMsg0)objc_msgSend;

        if (__mps_scale_pipeline == (void*)0) {
            void* pipeline = __mps_make_pipeline("scale_kernel");
            if (pipeline == (void*)0) return 0;
            __mps_scale_pipeline = pipeline;
        }
        void* pipeline = __mps_scale_pipeline;

        unsigned long bytes = n * sizeof(float);
        void* bufA = __mps_buf_get_with_bytes((const void*)a, bytes);
        void* bufOut = __mps_buf_get(bytes);
        if (bufA == (void*)0 || bufOut == (void*)0) return 0;

        void* cmdBuf = msg0(queue, __sel_commandBuffer);
        void* encoder = msg0(cmdBuf, __sel_computeCommandEncoder);

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        // setBytes:length:atIndex: shares setBuffer:offset:atIndex:'s
        // (recv, sel, ptr, NSUInteger, NSUInteger) shape -- see
        // mps_matmul_f32's comment on reusing MMsgSetBuffer for it.
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        float kVal = k;
        setBytesFn(encoder, __sel_setBytes, (void*)&kVal, 4UL, 1UL);
        setBufFn(encoder, __sel_setBuffer, bufOut, 0UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);
        MMsgVoid0 msgVoid0 = (MMsgVoid0)objc_msgSend;
        msgVoid0(encoder, __sel_endEncoding);
        msgVoid0(cmdBuf, __sel_commit);
        msgVoid0(cmdBuf, __sel_waitUntilCompleted);

        void* outPtr = msg0(bufOut, __sel_contents);
        if (outPtr == (void*)0) return 0;
        memcpy((void*)out, outPtr, bytes);
        __mps_buf_put(bufA, bytes);
        __mps_buf_put(bufOut, bytes);
        return 1;
    }
}

// out[0] = sum(a[0..n)) -- two-stage parallel reduction (see
// gpu_mps_kernels.metal's sum_kernel comment for the full design): a
// bounded number of threadgroups each grid-stride-accumulate then
// tree-reduce their own share of the input in threadgroup memory
// (O(log SUM_TG_SIZE) span instead of the old single-thread version's
// O(n)), emitting one partial sum per threadgroup; this function combines
// those (at most SUM_MAX_GROUPS of them, however large n is) on the CPU
// after one readback, which is cheap enough not to need a second GPU pass.
#define SUM_TG_SIZE     ((unsigned long)256)
#define SUM_MAX_GROUPS  ((unsigned long)256)

static void* __mps_sum_pipeline = (void*)0;

// mps_sum_f32's own definition (batch-aware, deferred-finalize partial-sum
// combine) lives further down, next to mps_matmul_f32/mps_sub_f32 and the
// batching primitives it needs to reference -- same reasoning as
// mps_relu_f32's comment above.

// ── Batched dispatch (see gpu_mps.h's mps_batch_begin comment for the
// full contract) ───────────────────────────────────────────────────────────
// __mps_batch_active/_cmdBuf/_encoder track one in-flight shared command
// buffer. __mps_pending_* is a fixed-size array of "GPU buffer -> CPU
// pointer" raw-readback entries collected while a batch is open — originally
// sized for a single training step's op count (a handful), raised to cover
// many independent ops sharing one batch too (e.g. dozens of inference
// passes batched together purely to amortize one command-buffer submit +
// wait across all of them -- see train_gpu.sc's inference loop). A real
// growable Vec would work too, but this file doesn't otherwise depend on
// std/collections, and a bounds-checked fixed array covers every actual
// use here without adding that dependency. __mps_finalize_* is the
// parallel list of caller-supplied post-readback callbacks (e.g.
// tensor_gpu.sc's float32->float64 conversion). __mps_poolbuf_* is a
// third, separate list: every buffer __mps_buf_get/_with_bytes hands out
// *while a batch is open* stays in flight until the whole batch's command
// buffer completes (the GPU may still be reading or writing it right up
// until waitUntilCompleted returns in mps_batch_end() below), so it can't
// go back to the pool at the point it's used the way the non-batched path
// does -- it's tracked here instead and returned in bulk once the wait is
// done. A buffer obtained via a _chained call's 'bufA' parameter (i.e.
// someone else's still-pending output, not a fresh __mps_buf_get) is never
// double-tracked, because tracking only happens at the __mps_buf_get call
// site itself, and chained functions never call it for their input.
#define MPS_BATCH_MAX_PENDING 1024
#define MPS_BATCH_MAX_FINALIZE 64
#define MPS_BATCH_MAX_POOLBUFS 1024

static int           __mps_batch_active = 0;
static void*         __mps_batch_cmdBuf = (void*)0;
static void*         __mps_batch_encoder = (void*)0;
static void*         __mps_pending_gpuBuf[MPS_BATCH_MAX_PENDING];
static void*         __mps_pending_cpuOut[MPS_BATCH_MAX_PENDING];
static unsigned long __mps_pending_bytes[MPS_BATCH_MAX_PENDING];
static unsigned long __mps_pending_count = 0UL;
static MpsFinalizeFn __mps_finalize_fn[MPS_BATCH_MAX_FINALIZE];
static void*         __mps_finalize_ctx[MPS_BATCH_MAX_FINALIZE];
static unsigned long __mps_finalize_count = 0UL;
static void*         __mps_poolbuf_buf[MPS_BATCH_MAX_POOLBUFS];
static unsigned long __mps_poolbuf_bytes[MPS_BATCH_MAX_POOLBUFS];
static unsigned long __mps_poolbuf_count = 0UL;

static void __mps_batch_track_pool_buf(void* buf, unsigned long bytes) {
    if (buf == (void*)0) return;
    if (__mps_poolbuf_count < (unsigned long)MPS_BATCH_MAX_POOLBUFS) {
        __mps_poolbuf_buf[__mps_poolbuf_count] = buf;
        __mps_poolbuf_bytes[__mps_poolbuf_count] = bytes;
        __mps_poolbuf_count = __mps_poolbuf_count + 1UL;
    }
}

// Batch-scoped versions of __mps_buf_get/_with_bytes: same allocation, but
// also register the buffer for deferred pool-return in mps_batch_end().
static void* __mps_buf_get_batched(unsigned long bytes) {
    void* buf = __mps_buf_get(bytes);
    __mps_batch_track_pool_buf(buf, bytes);
    return buf;
}

static void* __mps_buf_get_with_bytes_batched(const void* src, unsigned long bytes) {
    void* buf = __mps_buf_get_with_bytes(src, bytes);
    __mps_batch_track_pool_buf(buf, bytes);
    return buf;
}

int mps_batch_is_active() { return __mps_batch_active; }

void mps_batch_begin() {
    unsafe {
        if (!__mps_ensure_device_queue()) return;
        MMsg0 msg0 = (MMsg0)objc_msgSend;
        __mps_batch_cmdBuf  = msg0(__mps_queue, __sel_commandBuffer);
        __mps_batch_encoder = msg0(__mps_batch_cmdBuf, __sel_computeCommandEncoder);
        __mps_pending_count = 0UL;
        __mps_finalize_count = 0UL;
        __mps_poolbuf_count = 0UL;
        __mps_batch_active = 1;
    }
}

void mps_batch_register_finalize_raw_(MpsFinalizeFn finalizeFn, void* ctx) {
    if (__mps_finalize_count >= (unsigned long)MPS_BATCH_MAX_FINALIZE) return;
    __mps_finalize_fn[__mps_finalize_count] = finalizeFn;
    __mps_finalize_ctx[__mps_finalize_count] = ctx;
    __mps_finalize_count = __mps_finalize_count + 1UL;
}

?&void mps_batch_last_output_buffer() {
    if (!__mps_batch_active || __mps_pending_count == 0UL) return (void*)0;
    return __mps_pending_gpuBuf[__mps_pending_count - 1UL];
}

void mps_batch_end() {
    if (!__mps_batch_active) return;
    unsafe {
        MMsgVoid0 msgVoid0 = (MMsgVoid0)objc_msgSend;
        msgVoid0(__mps_batch_encoder, __sel_endEncoding);
        msgVoid0(__mps_batch_cmdBuf, __sel_commit);
        msgVoid0(__mps_batch_cmdBuf, __sel_waitUntilCompleted);

        MMsg0 msg0 = (MMsg0)objc_msgSend;
        unsigned long i = 0UL;
        while (i < __mps_pending_count) {
            void* outPtr = msg0(__mps_pending_gpuBuf[i], __sel_contents);
            if (outPtr != (void*)0) {
                memcpy(__mps_pending_cpuOut[i], outPtr, __mps_pending_bytes[i]);
            }
            i = i + 1UL;
        }
        __mps_pending_count = 0UL;

        // Every buffer used in this batch is now provably done (the GPU
        // finished reading/writing it before waitUntilCompleted returned)
        // -- safe to return the whole set to the pool at once.
        i = 0UL;
        while (i < __mps_poolbuf_count) {
            __mps_buf_put(__mps_poolbuf_buf[i], __mps_poolbuf_bytes[i]);
            i = i + 1UL;
        }
        __mps_poolbuf_count = 0UL;

        // Deactivate before running finalizers: a finalize callback must
        // never itself try to enqueue more batched work against a buffer
        // that's already been committed.
        __mps_batch_active = 0;
        __mps_batch_cmdBuf = (void*)0;
        __mps_batch_encoder = (void*)0;

        i = 0UL;
        while (i < __mps_finalize_count) {
            __mps_finalize_fn[i](__mps_finalize_ctx[i]);
            i = i + 1UL;
        }
        __mps_finalize_count = 0UL;
    }
}

// mps_sub_f32: own dedicated dispatch (not routed through
// __mps_run_binary_kernel like mul/div/pow) -- same reasoning as
// mps_relu_f32 below (see its comment): batch-aware, so tensor_sub_gpu can
// encode it onto a shared batch's command buffer for the fused
// forward+loss+backward training-step path instead of forcing its own
// commit+wait mid-batch.
int mps_sub_f32(const &float a, const &float b, &float out, unsigned long n) {
    unsafe {
        if (!__mps_ensure_device_queue()) return 0;
        void* device = __mps_device;
        void* queue = __mps_queue;
        MMsg0 msg0 = (MMsg0)objc_msgSend;

        if (__mps_sub_pipeline == (void*)0) {
            void* pipeline = __mps_make_pipeline("sub_kernel");
            if (pipeline == (void*)0) return 0;
            __mps_sub_pipeline = pipeline;
        }
        void* pipeline = __mps_sub_pipeline;

        unsigned long bytes = n * sizeof(float);
        int batched = __mps_batch_active;
        void* bufA = batched ? __mps_buf_get_with_bytes_batched((const void*)a, bytes)
                              : __mps_buf_get_with_bytes((const void*)a, bytes);
        void* bufB = batched ? __mps_buf_get_with_bytes_batched((const void*)b, bytes)
                              : __mps_buf_get_with_bytes((const void*)b, bytes);
        void* bufOut = batched ? __mps_buf_get_batched(bytes) : __mps_buf_get(bytes);
        if (bufA == (void*)0 || bufB == (void*)0 || bufOut == (void*)0) return 0;

        void* cmdBuf = (void*)0;
        void* encoder;
        if (batched) {
            encoder = __mps_batch_encoder;
        } else {
            cmdBuf = msg0(queue, __sel_commandBuffer);
            encoder = msg0(cmdBuf, __sel_computeCommandEncoder);
        }

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(encoder, __sel_setBuffer, bufB, 0UL, 1UL);
        setBufFn(encoder, __sel_setBuffer, bufOut, 0UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (batched) {
            if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
                __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
                __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
                __mps_pending_bytes[__mps_pending_count] = bytes;
                __mps_pending_count = __mps_pending_count + 1UL;
            }
            return 1;
        }

        MMsgVoid0 msgVoid0 = (MMsgVoid0)objc_msgSend;
        msgVoid0(encoder, __sel_endEncoding);
        msgVoid0(cmdBuf, __sel_commit);
        msgVoid0(cmdBuf, __sel_waitUntilCompleted);

        void* outPtr = msg0(bufOut, __sel_contents);
        if (outPtr == (void*)0) return 0;
        memcpy((void*)out, outPtr, bytes);
        __mps_buf_put(bufA, bytes);
        __mps_buf_put(bufB, bytes);
        __mps_buf_put(bufOut, bytes);
        return 1;
    }
}

// mps_sum_f32: batch-aware. Under an open batch, the CPU-side "combine the
// partial-sums array" step (the while-loop below) can't run inline the way
// the non-batched path does -- bufOut isn't readable until mps_batch_end()'s
// single wait, same reasoning as tensor_gpu.sc's __MpsGradAccumCtx/
// __mps_grad_accum_finalize. Same fix: register a finalize callback instead,
// pointed at a small scratch copy of the partials (populated by
// mps_batch_end()'s own generic pending-readback, same as any other batched
// op's output) rather than reading bufOut's Metal contents directly.
struct __MpsSumFinalizeCtx {
    float* partials;
    float* out;
    unsigned long numGroups;
};

static void __mps_sum_finalize(void* ctxPtr) {
    unsafe {
        struct __MpsSumFinalizeCtx* ctx = (struct __MpsSumFinalizeCtx*)ctxPtr;
        float acc = 0.0f;
        unsigned long g = 0UL;
        while (g < ctx->numGroups) { acc = acc + ctx->partials[g]; g = g + 1UL; }
        *(ctx->out) = acc;
        dealloc_raw_((void*)ctx->partials);
        dealloc_raw_((void*)ctx);
    }
}

int mps_sum_f32(const &float a, &float out, unsigned long n) {
    unsafe {
        if (!__mps_ensure_device_queue()) return 0;
        void* device = __mps_device;
        void* queue = __mps_queue;
        MMsg0 msg0 = (MMsg0)objc_msgSend;

        if (__mps_sum_pipeline == (void*)0) {
            void* pipeline = __mps_make_pipeline("sum_kernel");
            if (pipeline == (void*)0) return 0;
            __mps_sum_pipeline = pipeline;
        }
        void* pipeline = __mps_sum_pipeline;

        unsigned long bytes = n * sizeof(float);
        unsigned long numGroups = (n + SUM_TG_SIZE - 1UL) / SUM_TG_SIZE;
        if (numGroups > SUM_MAX_GROUPS) numGroups = SUM_MAX_GROUPS;
        if (numGroups == 0UL) numGroups = 1UL;
        unsigned long partialBytes = numGroups * sizeof(float);

        int batched = __mps_batch_active;
        void* bufA = batched ? __mps_buf_get_with_bytes_batched((const void*)a, bytes)
                              : __mps_buf_get_with_bytes((const void*)a, bytes);
        void* bufOut = batched ? __mps_buf_get_batched(partialBytes) : __mps_buf_get(partialBytes);
        if (bufA == (void*)0 || bufOut == (void*)0) return 0;

        void* cmdBuf = (void*)0;
        void* encoder;
        if (batched) {
            encoder = __mps_batch_encoder;
        } else {
            cmdBuf = msg0(queue, __sel_commandBuffer);
            encoder = msg0(cmdBuf, __sel_computeCommandEncoder);
        }

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(encoder, __sel_setBuffer, bufOut, 0UL, 1UL);
        unsigned int nu = (unsigned int)n;
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(encoder, __sel_setBytes, (void*)&nu, 4UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = numGroups; gridSize.height = 1UL; gridSize.depth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = SUM_TG_SIZE; tgSize.height = 1UL; tgSize.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __sel_dispatchThreadgroups, gridSize, tgSize);

        if (batched) {
            float* partials = (float*)alloc_raw_(sizeof(float) * numGroups);
            if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
                __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
                __mps_pending_cpuOut[__mps_pending_count] = (void*)partials;
                __mps_pending_bytes[__mps_pending_count] = partialBytes;
                __mps_pending_count = __mps_pending_count + 1UL;
            }
            struct __MpsSumFinalizeCtx* ctx = (struct __MpsSumFinalizeCtx*)alloc_raw_(sizeof(struct __MpsSumFinalizeCtx));
            ctx->partials = partials; ctx->out = out; ctx->numGroups = numGroups;
            mps_batch_register_finalize_raw_((MpsFinalizeFn)__mps_sum_finalize, (void*)ctx);
            return 1;
        }

        MMsgVoid0 msgVoid0 = (MMsgVoid0)objc_msgSend;
        msgVoid0(encoder, __sel_endEncoding);
        msgVoid0(cmdBuf, __sel_commit);
        msgVoid0(cmdBuf, __sel_waitUntilCompleted);

        void* outPtr = msg0(bufOut, __sel_contents);
        if (outPtr == (void*)0) return 0;
        float* partials = (float*)outPtr;
        float acc = 0.0f;
        unsigned long g = 0UL;
        while (g < numGroups) { acc = acc + partials[g]; g = g + 1UL; }
        *out = acc;
        __mps_buf_put(bufA, bytes);
        __mps_buf_put(bufOut, partialBytes);
        return 1;
    }
}

int mps_sum_f32_chained(&void bufA, &float out, unsigned long n) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (__mps_sum_pipeline == (void*)0) {
            void* pipeline = __mps_make_pipeline("sum_kernel");
            if (pipeline == (void*)0) return 0;
            __mps_sum_pipeline = pipeline;
        }

        unsigned long numGroups = (n + SUM_TG_SIZE - 1UL) / SUM_TG_SIZE;
        if (numGroups > SUM_MAX_GROUPS) numGroups = SUM_MAX_GROUPS;
        if (numGroups == 0UL) numGroups = 1UL;
        unsigned long partialBytes = numGroups * sizeof(float);

        void* bufOut = __mps_buf_get_batched(partialBytes);
        if (bufOut == (void*)0) return 0;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, __mps_sum_pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 1UL);
        unsigned int nu = (unsigned int)n;
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&nu, 4UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = numGroups; gridSize.height = 1UL; gridSize.depth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = SUM_TG_SIZE; tgSize.height = 1UL; tgSize.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, gridSize, tgSize);

        float* partials = (float*)alloc_raw_(sizeof(float) * numGroups);
        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)partials;
            __mps_pending_bytes[__mps_pending_count] = partialBytes;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        struct __MpsSumFinalizeCtx* ctx = (struct __MpsSumFinalizeCtx*)alloc_raw_(sizeof(struct __MpsSumFinalizeCtx));
        ctx->partials = partials; ctx->out = out; ctx->numGroups = numGroups;
        mps_batch_register_finalize_raw_((MpsFinalizeFn)__mps_sum_finalize, (void*)ctx);
        return 1;
    }
}

// out[M,N] = a[M,K] . b[K,N] -- naive one-thread-per-output-element kernel
// (no threadgroup-memory tiling/blocking), dispatched over a 2D grid. Not
// competitive with a tuned GEMM (MPSMatrixMultiplication, or a tiled kernel
// that reuses each loaded value across multiple output elements via
// threadgroup memory) -- this is the GPU-dispatch-pipeline-works version,
// same spirit as mps_add_f32 originally being "the simplest possible real
// GPU-executed op" before anything fancier. Metal has no 'double' type at
// all (confirmed: 'double' is a hard compile error in MSL), so this is
// float32-only; std::ml::Tensor stores 'double' data, so the GPU-backed
// Tensor path (tensor_matmul_gpu in tensor.sc) converts in and out of
// float32 around this call, trading precision for the ability to run on
// the GPU at all.
static void* __mps_matmul_pipeline = (void*)0;
static void* __mps_matmul_smma_pipeline = (void*)0;
static void* __mps_matmul_multi_pipeline = (void*)0;
static void* __mps_relu_pipeline = (void*)0;

int mps_matmul_f32(const &float a, const &float b, &float out,
                    unsigned long M, unsigned long K, unsigned long N) {
    unsafe {
        if (!__mps_ensure_device_queue()) return 0;
        void* device = __mps_device;
        void* queue = __mps_queue;
        MMsg0 msg0 = (MMsg0)objc_msgSend;

        // Prefer the simdgroup_matrix (hardware matrix unit) kernel
        // whenever the shape allows it -- see gpu_mps_kernels.metal's
        // comment on matmul_kernel_smma for why it can't handle
        // non-8-aligned shapes at all. Every other detail below (buffer
        // setup, batching, readback) is identical between the two paths;
        // only the pipeline object and the threadgroup/grid sizing differ.
        int useMulti = (M % 32UL == 0UL) && (K % 32UL == 0UL) && (N % 32UL == 0UL);
        int useSmma = (M % 8UL == 0UL) && (K % 8UL == 0UL) && (N % 8UL == 0UL);
        void* pipeline;
        if (useMulti) {
            if (__mps_matmul_multi_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_kernel_smma_multi");
                if (p == (void*)0) { useMulti = 0; } else { __mps_matmul_multi_pipeline = p; }
            }
            if (useMulti) pipeline = __mps_matmul_multi_pipeline;
        }
        if (!useMulti && useSmma) {
            if (__mps_matmul_smma_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_kernel_smma");
                if (p == (void*)0) { useSmma = 0; } else { __mps_matmul_smma_pipeline = p; }
            }
            if (useSmma) pipeline = __mps_matmul_smma_pipeline;
        }
        if (!useMulti && !useSmma) {
            if (__mps_matmul_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_kernel");
                if (p == (void*)0) return 0;
                __mps_matmul_pipeline = p;
            }
            pipeline = __mps_matmul_pipeline;
        }

        unsigned long bytesA = M * K * sizeof(float);
        unsigned long bytesB = K * N * sizeof(float);
        unsigned long bytesOut = M * N * sizeof(float);

        // Batched: reuse the shared command buffer/encoder that
        // mps_batch_begin() already opened instead of creating our own
        // (and don't end/commit/wait/readback here — mps_batch_end()
        // does all of that once, for every op encoded since begin()).
        // Buffers obtained while batched are tracked for deferred
        // pool-return there too, instead of being put back immediately.
        int batched = __mps_batch_active;
        void* bufA = batched ? __mps_buf_get_with_bytes_batched((const void*)a, bytesA)
                              : __mps_buf_get_with_bytes((const void*)a, bytesA);
        void* bufB = batched ? __mps_buf_get_with_bytes_batched((const void*)b, bytesB)
                              : __mps_buf_get_with_bytes((const void*)b, bytesB);
        void* bufOut = batched ? __mps_buf_get_batched(bytesOut) : __mps_buf_get(bytesOut);
        if (bufA == (void*)0 || bufB == (void*)0 || bufOut == (void*)0) return 0;

        unsigned int Mu = (unsigned int)M;
        unsigned int Ku = (unsigned int)K;
        unsigned int Nu = (unsigned int)N;

        void* cmdBuf = (void*)0;
        void* encoder;
        if (batched) {
            encoder = __mps_batch_encoder;
        } else {
            cmdBuf = msg0(queue, __sel_commandBuffer);
            encoder = msg0(cmdBuf, __sel_computeCommandEncoder);
        }

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(encoder, __sel_setBuffer, bufB, 0UL, 1UL);
        setBufFn(encoder, __sel_setBuffer, bufOut, 0UL, 2UL);
        // setBytes:length:atIndex: has the exact same (recv, sel, ptr,
        // NSUInteger, NSUInteger) shape as setBuffer:offset:atIndex:, so
        // MMsgSetBuffer's typedef works for it too -- only the selector
        // and what the middle argument means (raw bytes vs a buffer
        // object) differ.
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(encoder, __sel_setBytes, (void*)&Mu, 4UL, 3UL);
        setBytesFn(encoder, __sel_setBytes, (void*)&Ku, 4UL, 4UL);
        setBytesFn(encoder, __sel_setBytes, (void*)&Nu, 4UL, 5UL);

        // Dispatch: one threadgroup per 32x32 output tile (128 threads, 4
        // simdgroups) for the multi path; one threadgroup per 8x8 output
        // tile (32 threads, one simdgroup) for the single-simdgroup smma
        // path; TILE_SIZE(16) x TILE_SIZE(16) threadgroups for the tiled
        // fallback, matching matmul_kernel's threadgroup-memory tiling
        // exactly -- see mps_matmul_f32's header comment (above the
        // function) for why the tiled dispatch can't shrink for small
        // matrices.
        struct MTLSize tgSize;
        struct MTLSize numThreadgroups;
        if (useMulti) {
            tgSize.width = 128UL; tgSize.height = 1UL; tgSize.depth = 1UL;
            numThreadgroups.width = N / 32UL; numThreadgroups.height = M / 32UL; numThreadgroups.depth = 1UL;
        } else if (useSmma) {
            tgSize.width = 32UL; tgSize.height = 1UL; tgSize.depth = 1UL;
            numThreadgroups.width = N / 8UL; numThreadgroups.height = M / 8UL; numThreadgroups.depth = 1UL;
        } else {
            unsigned long tgW = 16UL;
            unsigned long tgH = 16UL;
            tgSize.width = tgW; tgSize.height = tgH; tgSize.depth = 1UL;
            unsigned long groupsX = (N + tgW - 1UL) / tgW;
            unsigned long groupsY = (M + tgH - 1UL) / tgH;
            numThreadgroups.width = groupsX; numThreadgroups.height = groupsY; numThreadgroups.depth = 1UL;
        }
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (batched) {
            if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
                __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
                __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
                __mps_pending_bytes[__mps_pending_count] = bytesOut;
                __mps_pending_count = __mps_pending_count + 1UL;
            }
            return 1; // deferred success -- 'out' not valid until mps_batch_end()
        }

        MMsgVoid0 msgVoid0 = (MMsgVoid0)objc_msgSend;
        msgVoid0(encoder, __sel_endEncoding);
        msgVoid0(cmdBuf, __sel_commit);
        msgVoid0(cmdBuf, __sel_waitUntilCompleted);

        void* outPtr = msg0(bufOut, __sel_contents);
        if (outPtr == (void*)0) return 0;
        memcpy((void*)out, outPtr, bytesOut);
        __mps_buf_put(bufA, bytesA);
        __mps_buf_put(bufB, bytesB);
        __mps_buf_put(bufOut, bytesOut);
        return 1;
    }
}

// ── matmul backward (see gpu_mps.h's own comment on these two) ───────────────
// Same dispatch shape as mps_matmul_f32 (2D grid, cached pipeline, pooled
// buffers) but against matmul_abt_kernel/matmul_atb_kernel instead of
// matmul_kernel -- not routed through mps_matmul_f32 itself since the
// output shape and which operand reads transposed differ between the two,
// and neither needs batching (tensor_backward() always runs after any
// forward-pass batch has already ended).
static void* __mps_matmul_abt_pipeline = (void*)0;
static void* __mps_matmul_abt_smma_pipeline = (void*)0;
static void* __mps_matmul_abt_multi_pipeline = (void*)0;

int mps_matmul_abt_f32(const &float a, const &float b, &float out,
                        unsigned long M, unsigned long N, unsigned long K) {
    unsafe {
        if (!__mps_ensure_device_queue()) return 0;
        void* device = __mps_device;
        void* queue = __mps_queue;
        MMsg0 msg0 = (MMsg0)objc_msgSend;

        // See mps_matmul_f32's comment on the smma/tiled choice.
        int useMulti = (M % 32UL == 0UL) && (N % 32UL == 0UL) && (K % 32UL == 0UL);
        int useSmma = (M % 8UL == 0UL) && (N % 8UL == 0UL) && (K % 8UL == 0UL);
        void* pipeline;
        if (useMulti) {
            if (__mps_matmul_abt_multi_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_abt_kernel_smma_multi");
                if (p == (void*)0) { useMulti = 0; } else { __mps_matmul_abt_multi_pipeline = p; }
            }
            if (useMulti) pipeline = __mps_matmul_abt_multi_pipeline;
        }
        if (!useMulti && useSmma) {
            if (__mps_matmul_abt_smma_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_abt_kernel_smma");
                if (p == (void*)0) { useSmma = 0; } else { __mps_matmul_abt_smma_pipeline = p; }
            }
            if (useSmma) pipeline = __mps_matmul_abt_smma_pipeline;
        }
        if (!useMulti && !useSmma) {
            if (__mps_matmul_abt_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_abt_kernel");
                if (p == (void*)0) return 0;
                __mps_matmul_abt_pipeline = p;
            }
            pipeline = __mps_matmul_abt_pipeline;
        }

        unsigned long bytesA = M * N * sizeof(float);
        unsigned long bytesB = K * N * sizeof(float);
        unsigned long bytesOut = M * K * sizeof(float);

        // Batched: same "reuse the shared encoder, defer commit/wait/
        // readback to mps_batch_end()" pattern as mps_matmul_f32 -- see
        // its comment.
        int batched = __mps_batch_active;
        void* bufA = batched ? __mps_buf_get_with_bytes_batched((const void*)a, bytesA)
                              : __mps_buf_get_with_bytes((const void*)a, bytesA);
        void* bufB = batched ? __mps_buf_get_with_bytes_batched((const void*)b, bytesB)
                              : __mps_buf_get_with_bytes((const void*)b, bytesB);
        void* bufOut = batched ? __mps_buf_get_batched(bytesOut) : __mps_buf_get(bytesOut);
        if (bufA == (void*)0 || bufB == (void*)0 || bufOut == (void*)0) return 0;

        unsigned int Mu = (unsigned int)M;
        unsigned int Nu = (unsigned int)N;
        unsigned int Ku = (unsigned int)K;

        void* cmdBuf = (void*)0;
        void* encoder;
        if (batched) {
            encoder = __mps_batch_encoder;
        } else {
            cmdBuf = msg0(queue, __sel_commandBuffer);
            encoder = msg0(cmdBuf, __sel_computeCommandEncoder);
        }

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(encoder, __sel_setBuffer, bufB, 0UL, 1UL);
        setBufFn(encoder, __sel_setBuffer, bufOut, 0UL, 2UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(encoder, __sel_setBytes, (void*)&Mu, 4UL, 3UL);
        setBytesFn(encoder, __sel_setBytes, (void*)&Nu, 4UL, 4UL);
        setBytesFn(encoder, __sel_setBytes, (void*)&Ku, 4UL, 5UL);

        // See mps_matmul_f32's comment on the smma/tiled dispatch shapes.
        struct MTLSize tgSize;
        struct MTLSize numThreadgroups;
        if (useMulti) {
            tgSize.width = 128UL; tgSize.height = 1UL; tgSize.depth = 1UL;
            numThreadgroups.width = K / 32UL; numThreadgroups.height = M / 32UL; numThreadgroups.depth = 1UL;
        } else if (useSmma) {
            tgSize.width = 32UL; tgSize.height = 1UL; tgSize.depth = 1UL;
            numThreadgroups.width = K / 8UL; numThreadgroups.height = M / 8UL; numThreadgroups.depth = 1UL;
        } else {
            unsigned long tgW = 16UL;
            unsigned long tgH = 16UL;
            tgSize.width = tgW; tgSize.height = tgH; tgSize.depth = 1UL;
            unsigned long groupsX = (K + tgW - 1UL) / tgW;
            unsigned long groupsY = (M + tgH - 1UL) / tgH;
            numThreadgroups.width = groupsX; numThreadgroups.height = groupsY; numThreadgroups.depth = 1UL;
        }
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (batched) {
            if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
                __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
                __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
                __mps_pending_bytes[__mps_pending_count] = bytesOut;
                __mps_pending_count = __mps_pending_count + 1UL;
            }
            return 1;
        }

        MMsgVoid0 msgVoid0 = (MMsgVoid0)objc_msgSend;
        msgVoid0(encoder, __sel_endEncoding);
        msgVoid0(cmdBuf, __sel_commit);
        msgVoid0(cmdBuf, __sel_waitUntilCompleted);

        void* outPtr = msg0(bufOut, __sel_contents);
        if (outPtr == (void*)0) return 0;
        memcpy((void*)out, outPtr, bytesOut);
        __mps_buf_put(bufA, bytesA);
        __mps_buf_put(bufB, bytesB);
        __mps_buf_put(bufOut, bytesOut);
        return 1;
    }
}

static void* __mps_matmul_atb_pipeline = (void*)0;
static void* __mps_matmul_atb_smma_pipeline = (void*)0;
static void* __mps_matmul_atb_multi_pipeline = (void*)0;

int mps_matmul_atb_f32(const &float a, const &float b, &float out,
                        unsigned long M, unsigned long K, unsigned long N) {
    unsafe {
        if (!__mps_ensure_device_queue()) return 0;
        void* device = __mps_device;
        void* queue = __mps_queue;
        MMsg0 msg0 = (MMsg0)objc_msgSend;

        // See mps_matmul_f32's comment on the smma/tiled choice.
        int useMulti = (M % 32UL == 0UL) && (K % 32UL == 0UL) && (N % 32UL == 0UL);
        int useSmma = (M % 8UL == 0UL) && (K % 8UL == 0UL) && (N % 8UL == 0UL);
        void* pipeline;
        if (useMulti) {
            if (__mps_matmul_atb_multi_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_atb_kernel_smma_multi");
                if (p == (void*)0) { useMulti = 0; } else { __mps_matmul_atb_multi_pipeline = p; }
            }
            if (useMulti) pipeline = __mps_matmul_atb_multi_pipeline;
        }
        if (!useMulti && useSmma) {
            if (__mps_matmul_atb_smma_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_atb_kernel_smma");
                if (p == (void*)0) { useSmma = 0; } else { __mps_matmul_atb_smma_pipeline = p; }
            }
            if (useSmma) pipeline = __mps_matmul_atb_smma_pipeline;
        }
        if (!useMulti && !useSmma) {
            if (__mps_matmul_atb_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_atb_kernel");
                if (p == (void*)0) return 0;
                __mps_matmul_atb_pipeline = p;
            }
            pipeline = __mps_matmul_atb_pipeline;
        }

        unsigned long bytesA = M * K * sizeof(float);
        unsigned long bytesB = M * N * sizeof(float);
        unsigned long bytesOut = K * N * sizeof(float);

        int batched = __mps_batch_active;
        void* bufA = batched ? __mps_buf_get_with_bytes_batched((const void*)a, bytesA)
                              : __mps_buf_get_with_bytes((const void*)a, bytesA);
        void* bufB = batched ? __mps_buf_get_with_bytes_batched((const void*)b, bytesB)
                              : __mps_buf_get_with_bytes((const void*)b, bytesB);
        void* bufOut = batched ? __mps_buf_get_batched(bytesOut) : __mps_buf_get(bytesOut);
        if (bufA == (void*)0 || bufB == (void*)0 || bufOut == (void*)0) return 0;

        unsigned int Mu = (unsigned int)M;
        unsigned int Ku = (unsigned int)K;
        unsigned int Nu = (unsigned int)N;

        void* cmdBuf = (void*)0;
        void* encoder;
        if (batched) {
            encoder = __mps_batch_encoder;
        } else {
            cmdBuf = msg0(queue, __sel_commandBuffer);
            encoder = msg0(cmdBuf, __sel_computeCommandEncoder);
        }

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(encoder, __sel_setBuffer, bufB, 0UL, 1UL);
        setBufFn(encoder, __sel_setBuffer, bufOut, 0UL, 2UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(encoder, __sel_setBytes, (void*)&Mu, 4UL, 3UL);
        setBytesFn(encoder, __sel_setBytes, (void*)&Ku, 4UL, 4UL);
        setBytesFn(encoder, __sel_setBytes, (void*)&Nu, 4UL, 5UL);

        // See mps_matmul_f32's comment on the smma/tiled dispatch shapes.
        struct MTLSize tgSize;
        struct MTLSize numThreadgroups;
        if (useMulti) {
            tgSize.width = 128UL; tgSize.height = 1UL; tgSize.depth = 1UL;
            numThreadgroups.width = N / 32UL; numThreadgroups.height = K / 32UL; numThreadgroups.depth = 1UL;
        } else if (useSmma) {
            tgSize.width = 32UL; tgSize.height = 1UL; tgSize.depth = 1UL;
            numThreadgroups.width = N / 8UL; numThreadgroups.height = K / 8UL; numThreadgroups.depth = 1UL;
        } else {
            unsigned long tgW = 16UL;
            unsigned long tgH = 16UL;
            tgSize.width = tgW; tgSize.height = tgH; tgSize.depth = 1UL;
            unsigned long groupsX = (N + tgW - 1UL) / tgW;
            unsigned long groupsY = (K + tgH - 1UL) / tgH;
            numThreadgroups.width = groupsX; numThreadgroups.height = groupsY; numThreadgroups.depth = 1UL;
        }
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (batched) {
            if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
                __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
                __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
                __mps_pending_bytes[__mps_pending_count] = bytesOut;
                __mps_pending_count = __mps_pending_count + 1UL;
            }
            return 1;
        }

        MMsgVoid0 msgVoid0 = (MMsgVoid0)objc_msgSend;
        msgVoid0(encoder, __sel_endEncoding);
        msgVoid0(cmdBuf, __sel_commit);
        msgVoid0(cmdBuf, __sel_waitUntilCompleted);

        void* outPtr = msg0(bufOut, __sel_contents);
        if (outPtr == (void*)0) return 0;
        memcpy((void*)out, outPtr, bytesOut);
        __mps_buf_put(bufA, bytesA);
        __mps_buf_put(bufB, bytesB);
        __mps_buf_put(bufOut, bytesOut);
        return 1;
    }
}

// mps_relu_backward_f32: own dedicated dispatch (not routed through
// __mps_run_binary_kernel, same reasoning as mps_relu_f32 above it in the
// file -- batching needs to skip the always-commit-and-wait path that
// helper takes). Same 1D-grid dispatch shape as mps_relu_f32/mps_add_f32.
int mps_relu_backward_f32(const &float a, const &float selfGrad, &float out, unsigned long n) {
    unsafe {
        if (!__mps_ensure_device_queue()) return 0;
        void* device = __mps_device;
        void* queue = __mps_queue;
        MMsg0 msg0 = (MMsg0)objc_msgSend;

        if (__mps_relu_backward_pipeline == (void*)0) {
            void* pipeline = __mps_make_pipeline("relu_backward_kernel");
            if (pipeline == (void*)0) return 0;
            __mps_relu_backward_pipeline = pipeline;
        }
        void* pipeline = __mps_relu_backward_pipeline;

        unsigned long bytes = n * sizeof(float);
        int batched = __mps_batch_active;
        void* bufA = batched ? __mps_buf_get_with_bytes_batched((const void*)a, bytes)
                              : __mps_buf_get_with_bytes((const void*)a, bytes);
        void* bufSelfGrad = batched ? __mps_buf_get_with_bytes_batched((const void*)selfGrad, bytes)
                                     : __mps_buf_get_with_bytes((const void*)selfGrad, bytes);
        void* bufOut = batched ? __mps_buf_get_batched(bytes) : __mps_buf_get(bytes);
        if (bufA == (void*)0 || bufSelfGrad == (void*)0 || bufOut == (void*)0) return 0;

        void* cmdBuf = (void*)0;
        void* encoder;
        if (batched) {
            encoder = __mps_batch_encoder;
        } else {
            cmdBuf = msg0(queue, __sel_commandBuffer);
            encoder = msg0(cmdBuf, __sel_computeCommandEncoder);
        }

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(encoder, __sel_setBuffer, bufSelfGrad, 0UL, 1UL);
        setBufFn(encoder, __sel_setBuffer, bufOut, 0UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (batched) {
            if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
                __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
                __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
                __mps_pending_bytes[__mps_pending_count] = bytes;
                __mps_pending_count = __mps_pending_count + 1UL;
            }
            return 1;
        }

        MMsgVoid0 msgVoid0 = (MMsgVoid0)objc_msgSend;
        msgVoid0(encoder, __sel_endEncoding);
        msgVoid0(cmdBuf, __sel_commit);
        msgVoid0(cmdBuf, __sel_waitUntilCompleted);

        void* outPtr = msg0(bufOut, __sel_contents);
        if (outPtr == (void*)0) return 0;
        memcpy((void*)out, outPtr, bytes);
        __mps_buf_put(bufA, bytes);
        __mps_buf_put(bufSelfGrad, bytes);
        __mps_buf_put(bufOut, bytes);
        return 1;
    }
}

// out[i] = max(a[i], 0) -- cached device/queue/pipeline and batching
// support, same pattern as mps_matmul_f32 above (not routed through
// __mps_run_unary_kernel like log/exp/sqrt: that helper always creates
// its own command buffer and blocks on waitUntilCompleted, which is
// exactly what a batched relu — encoded onto a matmul's shared command
// buffer, e.g. for a matmul -> relu -> matmul forward pass with no CPU
// step in between — needs to skip).
int mps_relu_f32(const &float a, &float out, unsigned long n) {
    unsafe {
        if (!__mps_ensure_device_queue()) return 0;
        void* device = __mps_device;
        void* queue = __mps_queue;
        MMsg0 msg0 = (MMsg0)objc_msgSend;

        if (__mps_relu_pipeline == (void*)0) {
            void* pipeline = __mps_make_pipeline("relu_kernel");
            if (pipeline == (void*)0) return 0;
            __mps_relu_pipeline = pipeline;
        }
        void* pipeline = __mps_relu_pipeline;

        unsigned long bytes = n * sizeof(float);
        int batched = __mps_batch_active;
        void* bufA = batched ? __mps_buf_get_with_bytes_batched((const void*)a, bytes)
                              : __mps_buf_get_with_bytes((const void*)a, bytes);
        void* bufOut = batched ? __mps_buf_get_batched(bytes) : __mps_buf_get(bytes);
        if (bufA == (void*)0 || bufOut == (void*)0) return 0;

        void* cmdBuf = (void*)0;
        void* encoder;
        if (batched) {
            encoder = __mps_batch_encoder;
        } else {
            cmdBuf = msg0(queue, __sel_commandBuffer);
            encoder = msg0(cmdBuf, __sel_computeCommandEncoder);
        }

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(encoder, __sel_setBuffer, bufOut, 0UL, 1UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (batched) {
            if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
                __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
                __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
                __mps_pending_bytes[__mps_pending_count] = bytes;
                __mps_pending_count = __mps_pending_count + 1UL;
            }
            return 1;
        }

        MMsgVoid0 msgVoid0 = (MMsgVoid0)objc_msgSend;
        msgVoid0(encoder, __sel_endEncoding);
        msgVoid0(cmdBuf, __sel_commit);
        msgVoid0(cmdBuf, __sel_waitUntilCompleted);

        void* outPtr = msg0(bufOut, __sel_contents);
        if (outPtr == (void*)0) return 0;
        memcpy((void*)out, outPtr, bytes);
        __mps_buf_put(bufA, bytes);
        __mps_buf_put(bufOut, bytes);
        return 1;
    }
}

// ── GPU-buffer-chained variants (see gpu_mps.h's own comment on these) ───────
// Only meaningful while a batch is active: 'bufA' is used directly as
// buffer(0) instead of uploading from a CPU float array, so the dispatch
// this encodes reads whatever a PREVIOUS dispatch on the same command
// buffer already wrote there -- correct without any CPU round trip
// specifically because Metal executes dispatches encoded on the same
// encoder in encoding order. 'bufA' itself is never pool-tracked here: it
// was already tracked by whichever op produced it as its own bufOut.
int mps_matmul_f32_chained(&void bufA, const &float b, &float out,
                            unsigned long M, unsigned long K, unsigned long N) {
    if (!__mps_batch_active) return 0;
    unsafe {
        // See mps_matmul_f32's comment on the smma/tiled choice.
        int useMulti = (M % 32UL == 0UL) && (K % 32UL == 0UL) && (N % 32UL == 0UL);
        int useSmma = (M % 8UL == 0UL) && (K % 8UL == 0UL) && (N % 8UL == 0UL);
        void* pipeline;
        if (useMulti) {
            if (__mps_matmul_multi_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_kernel_smma_multi");
                if (p == (void*)0) { useMulti = 0; } else { __mps_matmul_multi_pipeline = p; }
            }
            if (useMulti) pipeline = __mps_matmul_multi_pipeline;
        }
        if (!useMulti && useSmma) {
            if (__mps_matmul_smma_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_kernel_smma");
                if (p == (void*)0) { useSmma = 0; } else { __mps_matmul_smma_pipeline = p; }
            }
            if (useSmma) pipeline = __mps_matmul_smma_pipeline;
        }
        if (!useMulti && !useSmma) {
            if (__mps_matmul_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_kernel");
                if (p == (void*)0) return 0;
                __mps_matmul_pipeline = p;
            }
            pipeline = __mps_matmul_pipeline;
        }

        unsigned long bytesB = K * N * sizeof(float);
        unsigned long bytesOut = M * N * sizeof(float);
        void* bufB = __mps_buf_get_with_bytes_batched((const void*)b, bytesB);
        void* bufOut = __mps_buf_get_batched(bytesOut);
        if (bufB == (void*)0 || bufOut == (void*)0) return 0;

        unsigned int Mu = (unsigned int)M;
        unsigned int Ku = (unsigned int)K;
        unsigned int Nu = (unsigned int)N;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufB, 0UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Mu, 4UL, 3UL);
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Ku, 4UL, 4UL);
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Nu, 4UL, 5UL);

        struct MTLSize tgSize;
        struct MTLSize numThreadgroups;
        if (useMulti) {
            tgSize.width = 128UL; tgSize.height = 1UL; tgSize.depth = 1UL;
            numThreadgroups.width = N / 32UL; numThreadgroups.height = M / 32UL; numThreadgroups.depth = 1UL;
        } else if (useSmma) {
            tgSize.width = 32UL; tgSize.height = 1UL; tgSize.depth = 1UL;
            numThreadgroups.width = N / 8UL; numThreadgroups.height = M / 8UL; numThreadgroups.depth = 1UL;
        } else {
            unsigned long tgW = 16UL;
            unsigned long tgH = 16UL;
            tgSize.width = tgW; tgSize.height = tgH; tgSize.depth = 1UL;
            unsigned long groupsX = (N + tgW - 1UL) / tgW;
            unsigned long groupsY = (M + tgH - 1UL) / tgH;
            numThreadgroups.width = groupsX; numThreadgroups.height = groupsY; numThreadgroups.depth = 1UL;
        }
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytesOut;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

// out[M,N] = bufA[M,K] . bufB[K,N] -- same computation as mps_matmul_f32_
// chained, but 'bufB' is ALSO a pre-existing device buffer (typically from
// mps_upload_persistent below) instead of a plain CPU array to upload
// fresh. Needed for e.g. an inference loop that runs the same matmul
// against the same weight matrix hundreds of times: mps_matmul_f32_chained
// would re-upload (memcpy) that weight matrix from CPU on every single
// call, even though it never changes between calls. 'bufA' can be either
// another persistent buffer or a previous op's still-pending chained
// output (mps_batch_last_output_buffer()) -- this function doesn't care
// which, both are just "already a device buffer".
int mps_matmul_f32_persistent(&void bufA, &void bufB, &float out,
                               unsigned long M, unsigned long K, unsigned long N) {
    if (!__mps_batch_active) return 0;
    unsafe {
        int useMulti = (M % 32UL == 0UL) && (K % 32UL == 0UL) && (N % 32UL == 0UL);
        int useSmma = (M % 8UL == 0UL) && (K % 8UL == 0UL) && (N % 8UL == 0UL);
        void* pipeline;
        if (useMulti) {
            if (__mps_matmul_multi_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_kernel_smma_multi");
                if (p == (void*)0) { useMulti = 0; } else { __mps_matmul_multi_pipeline = p; }
            }
            if (useMulti) pipeline = __mps_matmul_multi_pipeline;
        }
        if (!useMulti && useSmma) {
            if (__mps_matmul_smma_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_kernel_smma");
                if (p == (void*)0) { useSmma = 0; } else { __mps_matmul_smma_pipeline = p; }
            }
            if (useSmma) pipeline = __mps_matmul_smma_pipeline;
        }
        if (!useMulti && !useSmma) {
            if (__mps_matmul_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_kernel");
                if (p == (void*)0) return 0;
                __mps_matmul_pipeline = p;
            }
            pipeline = __mps_matmul_pipeline;
        }

        unsigned long bytesOut = M * N * sizeof(float);
        void* bufOut = __mps_buf_get_batched(bytesOut);
        if (bufOut == (void*)0) return 0;

        unsigned int Mu = (unsigned int)M;
        unsigned int Ku = (unsigned int)K;
        unsigned int Nu = (unsigned int)N;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufB, 0UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Mu, 4UL, 3UL);
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Ku, 4UL, 4UL);
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Nu, 4UL, 5UL);

        struct MTLSize tgSize;
        struct MTLSize numThreadgroups;
        if (useMulti) {
            tgSize.width = 128UL; tgSize.height = 1UL; tgSize.depth = 1UL;
            numThreadgroups.width = N / 32UL; numThreadgroups.height = M / 32UL; numThreadgroups.depth = 1UL;
        } else if (useSmma) {
            tgSize.width = 32UL; tgSize.height = 1UL; tgSize.depth = 1UL;
            numThreadgroups.width = N / 8UL; numThreadgroups.height = M / 8UL; numThreadgroups.depth = 1UL;
        } else {
            unsigned long tgW = 16UL;
            unsigned long tgH = 16UL;
            tgSize.width = tgW; tgSize.height = tgH; tgSize.depth = 1UL;
            unsigned long groupsX = (N + tgW - 1UL) / tgW;
            unsigned long groupsY = (M + tgH - 1UL) / tgH;
            numThreadgroups.width = groupsX; numThreadgroups.height = groupsY; numThreadgroups.depth = 1UL;
        }
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytesOut;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

// Uploads 'data' to a device buffer ONCE and hands back a handle the
// caller keeps and reuses directly (as 'bufA'/'bufB' above) across as many
// dispatches and batches as it wants, instead of it being re-uploaded
// fresh on every single call the way an ordinary (non-persistent) Tensor
// operand is. Safe to call whether or not a batch is currently open (it's
// just a pool allocation + memcpy, no command buffer involved) -- but only
// safe to call when no GPU work that might still be reading the reused
// pool slot's previous contents is in flight, i.e. after any previous
// batch's mps_batch_end() has already returned. NOT tracked by the
// batching machinery the way __mps_buf_get_with_bytes_batched's return
// value is -- the caller owns this buffer's lifetime and must release it
// with mps_release_persistent below when done, exactly once.
//
// __mps_ensure_device_queue() first, unlike a bare __mps_buf_get_with_
// bytes call: every OTHER public entry point in this file only ever runs
// after mps_matmul_f32 or similar has already lazily initialized __mps_
// device as a side effect of an earlier call, so nothing else needed this
// check explicitly. This one doesn't get that for free -- it's meant to
// be usable as the FIRST GPU call in a whole program (e.g. uploading a
// training loop's weights before any forward pass has run at all), and
// without this, __mps_buf_get would silently newBufferWithLength: a nil
// device (Objective-C's "message to nil returns nil" behavior, not a
// crash) and hand back a nil buffer that reads back as all zeros forever
// after -- found by a real correctness test that returned exactly zero
// for every output element, not by inspection.
?&void mps_upload_persistent(const &float data, unsigned long bytes) {
    if (!__mps_ensure_device_queue()) return (void*)0;
    return __mps_buf_get_with_bytes(data, bytes);
}

// Returns a buffer obtained from mps_upload_persistent to the size-class
// pool (see __mps_buf_put's own comment) -- call once, after every batch
// that might still be reading it has completed.
void mps_release_persistent(&void buf, unsigned long bytes) {
    __mps_buf_put(buf, bytes);
}

?&void mps_alloc_persistent(unsigned long bytes) {
    if (!__mps_ensure_device_queue()) return (void*)0;
    return __mps_buf_get(bytes);
}

int mps_read_persistent(&void buf, &void dst, unsigned long bytes) {
    unsafe {
        MMsg0 msg0 = (MMsg0)objc_msgSend;
        void* src = msg0((void*)buf, __sel_contents);
        if (src == (void*)0) return 0;
        memcpy((void*)dst, src, bytes);
        return 1;
    }
}

int mps_relu_f32_chained(&void bufA, &float out, unsigned long n) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (__mps_relu_pipeline == (void*)0) {
            void* pipeline = __mps_make_pipeline("relu_kernel");
            if (pipeline == (void*)0) return 0;
            __mps_relu_pipeline = pipeline;
        }

        unsigned long bytes = n * sizeof(float);
        void* bufOut = __mps_buf_get_batched(bytes);
        if (bufOut == (void*)0) return 0;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, __mps_relu_pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 1UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytes;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

// out[i] = bufA[i] - b[i] -- see gpu_mps.h's comment.
int mps_sub_f32_chained(&void bufA, const &float b, &float out, unsigned long n) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (__mps_sub_pipeline == (void*)0) {
            void* pipeline = __mps_make_pipeline("sub_kernel");
            if (pipeline == (void*)0) return 0;
            __mps_sub_pipeline = pipeline;
        }

        unsigned long bytes = n * sizeof(float);
        void* bufB = __mps_buf_get_with_bytes_batched((const void*)b, bytes);
        void* bufOut = __mps_buf_get_batched(bytes);
        if (bufB == (void*)0 || bufOut == (void*)0) return 0;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, __mps_sub_pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufB, 0UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytes;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

// out[i] = bufA[i] * b[i] -- see gpu_mps.h's comment.
int mps_mul_f32_chained(&void bufA, const &float b, &float out, unsigned long n) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (__mps_mul_pipeline == (void*)0) {
            void* pipeline = __mps_make_pipeline("mul_kernel");
            if (pipeline == (void*)0) return 0;
            __mps_mul_pipeline = pipeline;
        }

        unsigned long bytes = n * sizeof(float);
        void* bufB = __mps_buf_get_with_bytes_batched((const void*)b, bytes);
        void* bufOut = __mps_buf_get_batched(bytes);
        if (bufB == (void*)0 || bufOut == (void*)0) return 0;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, __mps_mul_pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufB, 0UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytes;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

// out[i] = buf[i] * buf[i] -- see gpu_mps.h's comment. Same mul_kernel as
// mps_mul_f32_chained above, just bound to buffer indices 0 AND 1 with the
// same buffer -- reading the same device memory location from two bound
// slots is safe (both reads, no write), so this needs no dedicated kernel.
int mps_square_f32_chained(&void buf, &float out, unsigned long n) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (__mps_mul_pipeline == (void*)0) {
            void* pipeline = __mps_make_pipeline("mul_kernel");
            if (pipeline == (void*)0) return 0;
            __mps_mul_pipeline = pipeline;
        }

        unsigned long bytes = n * sizeof(float);
        void* bufOut = __mps_buf_get_batched(bytes);
        if (bufOut == (void*)0) return 0;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, __mps_mul_pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, buf, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, buf, 0UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytes;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

// out[i] = buf[i] * k -- see gpu_mps.h's comment.
int mps_scale_f32_chained(&void buf, float k, &float out, unsigned long n) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (__mps_scale_pipeline == (void*)0) {
            void* pipeline = __mps_make_pipeline("scale_kernel");
            if (pipeline == (void*)0) return 0;
            __mps_scale_pipeline = pipeline;
        }

        unsigned long bytes = n * sizeof(float);
        void* bufOut = __mps_buf_get_batched(bytes);
        if (bufOut == (void*)0) return 0;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, __mps_scale_pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, buf, 0UL, 0UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        float kVal = k;
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&kVal, 4UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytes;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

// See gpu_mps.h's comment. Same dispatch shape as mps_scale_f32_chained
// above (one buffer input, one scalar constant, one buffer output), just
// with the scalar as a 4-byte 'uint' (matching the kernel's 'constant
// uint&') instead of a float, and a fused gate/up-split + GELU + multiply
// kernel body instead of a plain multiply-by-scalar. Shared by the f32 and
// bf16 variants; 'elemBytes' is the storage width (4 or 2).
static int __mps_gated_gelu_chained(const char* kernelName, void** pipelineSlot,
                                     void* bufRaw, void* out, unsigned long rows,
                                     unsigned long intermediateSize, unsigned long elemBytes) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (*pipelineSlot == (void*)0) {
            void* p = __mps_make_pipeline(kernelName);
            if (p == (void*)0) return 0;
            *pipelineSlot = p;
        }

        unsigned long n = rows * intermediateSize;
        unsigned long bytes = n * elemBytes;
        void* bufOut = __mps_buf_get_batched(bytes);
        if (bufOut == (void*)0) return 0;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, *pipelineSlot);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufRaw, 0UL, 0UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        unsigned int iSizeVal = (unsigned int)intermediateSize;
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&iSizeVal, 4UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = out;
            __mps_pending_bytes[__mps_pending_count] = bytes;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

int mps_gated_gelu_f32_chained(&void bufRaw, &float out, unsigned long rows, unsigned long intermediateSize) {
    unsafe {
        return __mps_gated_gelu_chained("gated_gelu_kernel", &__mps_gated_gelu_pipeline,
                                        bufRaw, (void*)out, rows, intermediateSize, sizeof(float));
    }
}

static void* __mps_gated_gelu_bf16_pipeline = (void*)0;
int mps_gated_gelu_bf16_chained(&void bufRaw, &unsigned short out, unsigned long rows, unsigned long intermediateSize) {
    unsafe {
        return __mps_gated_gelu_chained("gated_gelu_kernel_bf16", &__mps_gated_gelu_bf16_pipeline,
                                        bufRaw, (void*)out, rows, intermediateSize, 2UL);
    }
}

// ── Backward chained variants (see gpu_mps.h's own comment on these) ─────────
// Same "read a previous batched op's still-pending GPU buffer directly,
// no CPU round trip" idea as mps_matmul_f32_chained/mps_relu_f32_chained
// above, applied to the backward chain instead of the forward one.
int mps_matmul_abt_f32_chained(&void bufDC, const &float b, &float out,
                                unsigned long M, unsigned long N, unsigned long K) {
    if (!__mps_batch_active) return 0;
    unsafe {
        int useMulti = (M % 32UL == 0UL) && (N % 32UL == 0UL) && (K % 32UL == 0UL);
        int useSmma = (M % 8UL == 0UL) && (N % 8UL == 0UL) && (K % 8UL == 0UL);
        void* pipeline;
        if (useMulti) {
            if (__mps_matmul_abt_multi_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_abt_kernel_smma_multi");
                if (p == (void*)0) { useMulti = 0; } else { __mps_matmul_abt_multi_pipeline = p; }
            }
            if (useMulti) pipeline = __mps_matmul_abt_multi_pipeline;
        }
        if (!useMulti && useSmma) {
            if (__mps_matmul_abt_smma_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_abt_kernel_smma");
                if (p == (void*)0) { useSmma = 0; } else { __mps_matmul_abt_smma_pipeline = p; }
            }
            if (useSmma) pipeline = __mps_matmul_abt_smma_pipeline;
        }
        if (!useMulti && !useSmma) {
            if (__mps_matmul_abt_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_abt_kernel");
                if (p == (void*)0) return 0;
                __mps_matmul_abt_pipeline = p;
            }
            pipeline = __mps_matmul_abt_pipeline;
        }

        unsigned long bytesB = K * N * sizeof(float);
        unsigned long bytesOut = M * K * sizeof(float);
        void* bufB = __mps_buf_get_with_bytes_batched((const void*)b, bytesB);
        void* bufOut = __mps_buf_get_batched(bytesOut);
        if (bufB == (void*)0 || bufOut == (void*)0) return 0;

        unsigned int Mu = (unsigned int)M;
        unsigned int Nu = (unsigned int)N;
        unsigned int Ku = (unsigned int)K;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufDC, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufB, 0UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Mu, 4UL, 3UL);
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Nu, 4UL, 4UL);
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Ku, 4UL, 5UL);

        struct MTLSize tgSize;
        struct MTLSize numThreadgroups;
        if (useMulti) {
            tgSize.width = 128UL; tgSize.height = 1UL; tgSize.depth = 1UL;
            numThreadgroups.width = K / 32UL; numThreadgroups.height = M / 32UL; numThreadgroups.depth = 1UL;
        } else if (useSmma) {
            tgSize.width = 32UL; tgSize.height = 1UL; tgSize.depth = 1UL;
            numThreadgroups.width = K / 8UL; numThreadgroups.height = M / 8UL; numThreadgroups.depth = 1UL;
        } else {
            unsigned long tgW = 16UL;
            unsigned long tgH = 16UL;
            tgSize.width = tgW; tgSize.height = tgH; tgSize.depth = 1UL;
            unsigned long groupsX = (K + tgW - 1UL) / tgW;
            unsigned long groupsY = (M + tgH - 1UL) / tgH;
            numThreadgroups.width = groupsX; numThreadgroups.height = groupsY; numThreadgroups.depth = 1UL;
        }
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytesOut;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

// Same as mps_matmul_abt_f32_chained above, but 'b' is ALSO a pre-existing
// device buffer (typically mps_upload_persistent, e.g. a weight matrix
// that never leaves the GPU across a whole training run) instead of a
// plain CPU array to upload fresh -- same reasoning as mps_matmul_f32_
// persistent. Needed for dH = dY . W2^T when W2 stays GPU-resident and
// is updated in place by sgd_update_kernel every step instead of ever
// being read back to a CPU array.
int mps_matmul_abt_f32_persistent(&void bufDC, &void bufB, &float out,
                                   unsigned long M, unsigned long N, unsigned long K) {
    if (!__mps_batch_active) return 0;
    unsafe {
        int useMulti = (M % 32UL == 0UL) && (N % 32UL == 0UL) && (K % 32UL == 0UL);
        int useSmma = (M % 8UL == 0UL) && (N % 8UL == 0UL) && (K % 8UL == 0UL);
        void* pipeline;
        if (useMulti) {
            if (__mps_matmul_abt_multi_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_abt_kernel_smma_multi");
                if (p == (void*)0) { useMulti = 0; } else { __mps_matmul_abt_multi_pipeline = p; }
            }
            if (useMulti) pipeline = __mps_matmul_abt_multi_pipeline;
        }
        if (!useMulti && useSmma) {
            if (__mps_matmul_abt_smma_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_abt_kernel_smma");
                if (p == (void*)0) { useSmma = 0; } else { __mps_matmul_abt_smma_pipeline = p; }
            }
            if (useSmma) pipeline = __mps_matmul_abt_smma_pipeline;
        }
        if (!useMulti && !useSmma) {
            if (__mps_matmul_abt_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_abt_kernel");
                if (p == (void*)0) return 0;
                __mps_matmul_abt_pipeline = p;
            }
            pipeline = __mps_matmul_abt_pipeline;
        }

        unsigned long bytesOut = M * K * sizeof(float);
        void* bufOut = __mps_buf_get_batched(bytesOut);
        if (bufOut == (void*)0) return 0;

        unsigned int Mu = (unsigned int)M;
        unsigned int Nu = (unsigned int)N;
        unsigned int Ku = (unsigned int)K;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufDC, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufB, 0UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Mu, 4UL, 3UL);
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Nu, 4UL, 4UL);
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Ku, 4UL, 5UL);

        struct MTLSize tgSize;
        struct MTLSize numThreadgroups;
        if (useMulti) {
            tgSize.width = 128UL; tgSize.height = 1UL; tgSize.depth = 1UL;
            numThreadgroups.width = K / 32UL; numThreadgroups.height = M / 32UL; numThreadgroups.depth = 1UL;
        } else if (useSmma) {
            tgSize.width = 32UL; tgSize.height = 1UL; tgSize.depth = 1UL;
            numThreadgroups.width = K / 8UL; numThreadgroups.height = M / 8UL; numThreadgroups.depth = 1UL;
        } else {
            unsigned long tgW = 16UL;
            unsigned long tgH = 16UL;
            tgSize.width = tgW; tgSize.height = tgH; tgSize.depth = 1UL;
            unsigned long groupsX = (K + tgW - 1UL) / tgW;
            unsigned long groupsY = (M + tgH - 1UL) / tgH;
            numThreadgroups.width = groupsX; numThreadgroups.height = groupsY; numThreadgroups.depth = 1UL;
        }
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytesOut;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

int mps_matmul_atb_f32_chained(const &float a, &void bufDC, &float out,
                                unsigned long M, unsigned long K, unsigned long N) {
    if (!__mps_batch_active) return 0;
    unsafe {
        int useMulti = (M % 32UL == 0UL) && (K % 32UL == 0UL) && (N % 32UL == 0UL);
        int useSmma = (M % 8UL == 0UL) && (K % 8UL == 0UL) && (N % 8UL == 0UL);
        void* pipeline;
        if (useMulti) {
            if (__mps_matmul_atb_multi_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_atb_kernel_smma_multi");
                if (p == (void*)0) { useMulti = 0; } else { __mps_matmul_atb_multi_pipeline = p; }
            }
            if (useMulti) pipeline = __mps_matmul_atb_multi_pipeline;
        }
        if (!useMulti && useSmma) {
            if (__mps_matmul_atb_smma_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_atb_kernel_smma");
                if (p == (void*)0) { useSmma = 0; } else { __mps_matmul_atb_smma_pipeline = p; }
            }
            if (useSmma) pipeline = __mps_matmul_atb_smma_pipeline;
        }
        if (!useMulti && !useSmma) {
            if (__mps_matmul_atb_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_atb_kernel");
                if (p == (void*)0) return 0;
                __mps_matmul_atb_pipeline = p;
            }
            pipeline = __mps_matmul_atb_pipeline;
        }

        unsigned long bytesA = M * K * sizeof(float);
        unsigned long bytesOut = K * N * sizeof(float);
        void* bufA = __mps_buf_get_with_bytes_batched((const void*)a, bytesA);
        void* bufOut = __mps_buf_get_batched(bytesOut);
        if (bufA == (void*)0 || bufOut == (void*)0) return 0;

        unsigned int Mu = (unsigned int)M;
        unsigned int Ku = (unsigned int)K;
        unsigned int Nu = (unsigned int)N;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufDC, 0UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Mu, 4UL, 3UL);
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Ku, 4UL, 4UL);
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Nu, 4UL, 5UL);

        struct MTLSize tgSize;
        struct MTLSize numThreadgroups;
        if (useMulti) {
            tgSize.width = 128UL; tgSize.height = 1UL; tgSize.depth = 1UL;
            numThreadgroups.width = N / 32UL; numThreadgroups.height = K / 32UL; numThreadgroups.depth = 1UL;
        } else if (useSmma) {
            tgSize.width = 32UL; tgSize.height = 1UL; tgSize.depth = 1UL;
            numThreadgroups.width = N / 8UL; numThreadgroups.height = K / 8UL; numThreadgroups.depth = 1UL;
        } else {
            unsigned long tgW = 16UL;
            unsigned long tgH = 16UL;
            tgSize.width = tgW; tgSize.height = tgH; tgSize.depth = 1UL;
            unsigned long groupsX = (N + tgW - 1UL) / tgW;
            unsigned long groupsY = (K + tgH - 1UL) / tgH;
            numThreadgroups.width = groupsX; numThreadgroups.height = groupsY; numThreadgroups.depth = 1UL;
        }
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytesOut;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

// Same as mps_matmul_atb_f32_chained above, but 'a' is ALSO a still-pending
// GPU buffer instead of a plain CPU array. Needed for the fused
// forward+loss+backward training step: dW2 = H^T . dY needs BOTH H (an
// activation computed earlier in the SAME still-open batch by
// tensor_relu_gpu, not yet read back to CPU) and dY (matmul2-backward's
// own incoming gradient, also still-pending) read straight from GPU
// memory -- mps_matmul_atb_f32_chained's plain-CPU-array 'a' parameter
// can't express that. (Doesn't arise for matmul1's own backward: its 'a'
// is X, a real input tensor, always CPU-resident already.)
int mps_matmul_atb_f32_chained_ab(&void bufA, &void bufDC, &float out,
                                   unsigned long M, unsigned long K, unsigned long N) {
    if (!__mps_batch_active) return 0;
    unsafe {
        int useMulti = (M % 32UL == 0UL) && (K % 32UL == 0UL) && (N % 32UL == 0UL);
        int useSmma = (M % 8UL == 0UL) && (K % 8UL == 0UL) && (N % 8UL == 0UL);
        void* pipeline;
        if (useMulti) {
            if (__mps_matmul_atb_multi_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_atb_kernel_smma_multi");
                if (p == (void*)0) { useMulti = 0; } else { __mps_matmul_atb_multi_pipeline = p; }
            }
            if (useMulti) pipeline = __mps_matmul_atb_multi_pipeline;
        }
        if (!useMulti && useSmma) {
            if (__mps_matmul_atb_smma_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_atb_kernel_smma");
                if (p == (void*)0) { useSmma = 0; } else { __mps_matmul_atb_smma_pipeline = p; }
            }
            if (useSmma) pipeline = __mps_matmul_atb_smma_pipeline;
        }
        if (!useMulti && !useSmma) {
            if (__mps_matmul_atb_pipeline == (void*)0) {
                void* p = __mps_make_pipeline("matmul_atb_kernel");
                if (p == (void*)0) return 0;
                __mps_matmul_atb_pipeline = p;
            }
            pipeline = __mps_matmul_atb_pipeline;
        }

        unsigned long bytesOut = K * N * sizeof(float);
        void* bufOut = __mps_buf_get_batched(bytesOut);
        if (bufOut == (void*)0) return 0;

        unsigned int Mu = (unsigned int)M;
        unsigned int Ku = (unsigned int)K;
        unsigned int Nu = (unsigned int)N;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufDC, 0UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Mu, 4UL, 3UL);
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Ku, 4UL, 4UL);
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Nu, 4UL, 5UL);

        struct MTLSize tgSize;
        struct MTLSize numThreadgroups;
        if (useMulti) {
            tgSize.width = 128UL; tgSize.height = 1UL; tgSize.depth = 1UL;
            numThreadgroups.width = N / 32UL; numThreadgroups.height = K / 32UL; numThreadgroups.depth = 1UL;
        } else if (useSmma) {
            tgSize.width = 32UL; tgSize.height = 1UL; tgSize.depth = 1UL;
            numThreadgroups.width = N / 8UL; numThreadgroups.height = K / 8UL; numThreadgroups.depth = 1UL;
        } else {
            unsigned long tgW = 16UL;
            unsigned long tgH = 16UL;
            tgSize.width = tgW; tgSize.height = tgH; tgSize.depth = 1UL;
            unsigned long groupsX = (N + tgW - 1UL) / tgW;
            unsigned long groupsY = (K + tgH - 1UL) / tgH;
            numThreadgroups.width = groupsX; numThreadgroups.height = groupsY; numThreadgroups.depth = 1UL;
        }
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytesOut;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

int mps_relu_backward_f32_chained(const &float a, &void bufSelfGrad, &float out, unsigned long n) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (__mps_relu_backward_pipeline == (void*)0) {
            void* pipeline = __mps_make_pipeline("relu_backward_kernel");
            if (pipeline == (void*)0) return 0;
            __mps_relu_backward_pipeline = pipeline;
        }

        unsigned long bytes = n * sizeof(float);
        void* bufA = __mps_buf_get_with_bytes_batched((const void*)a, bytes);
        void* bufOut = __mps_buf_get_batched(bytes);
        if (bufA == (void*)0 || bufOut == (void*)0) return 0;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, __mps_relu_backward_pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufSelfGrad, 0UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytes;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

// Same as mps_relu_backward_f32_chained above, but 'a' is ALSO a
// still-pending GPU buffer instead of a plain CPU array -- same reasoning
// as mps_matmul_atb_f32_chained_ab (see its comment): in the fused
// forward+loss+backward training step, relu's own input (Z1, matmul1's
// output) hasn't been read back to CPU yet either when relu-backward runs.
int mps_relu_backward_f32_chained_ab(&void bufA, &void bufSelfGrad, &float out, unsigned long n) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (__mps_relu_backward_pipeline == (void*)0) {
            void* pipeline = __mps_make_pipeline("relu_backward_kernel");
            if (pipeline == (void*)0) return 0;
            __mps_relu_backward_pipeline = pipeline;
        }

        unsigned long bytes = n * sizeof(float);
        void* bufOut = __mps_buf_get_batched(bytes);
        if (bufOut == (void*)0) return 0;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, __mps_relu_backward_pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufSelfGrad, 0UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytes;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

// In-place SGD step against a persistent device buffer (see gpu_mps_
// kernels.metal's sgd_update_kernel): w[i] -= lr*grad[i], both already
// device buffers -- 'bufW' typically from mps_upload_persistent (a weight
// matrix uploaded once at the start of training and never read back to a
// CPU array until the caller is actually done with it), 'bufGrad'
// typically another op's still-pending chained output from the SAME
// batch's backward pass. No 'out'/pending-readback parameter: unlike
// every other batched op here, this one's result is never meant to reach
// CPU memory as part of the batch ending -- that's the entire point of
// keeping a weight GPU-resident across a training run instead of
// re-uploading it (mps_matmul_f32/_persistent) every step.
static void* __mps_sgd_update_pipeline = (void*)0;

int mps_sgd_update_f32_chained(&void bufW, &void bufGrad, float lr, unsigned long n) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (__mps_sgd_update_pipeline == (void*)0) {
            void* pipeline = __mps_make_pipeline("sgd_update_kernel");
            if (pipeline == (void*)0) return 0;
            __mps_sgd_update_pipeline = pipeline;
        }

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, __mps_sgd_update_pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufW, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufGrad, 0UL, 1UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        float lrVal = lr;
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&lrVal, 4UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        return 1;
    }
}

// ── fp16 / bf16 tier (see gpu_mps.h) ─────────────────────────────────────────
static void* __mps_matmul_f16_pipeline = (void*)0;
static void* __mps_matmul_bf16_pipeline = (void*)0;
static void* __mps_relu_f16_pipeline = (void*)0;
static void* __mps_relu_bf16_pipeline = (void*)0;

int mps_matmul_f16(const &float a, const &float b, &float out,
                    unsigned long M, unsigned long K, unsigned long N) {
    unsafe {
        if (!__mps_ensure_device_queue()) return 0;
        void* queue = __mps_queue;
        MMsg0 msg0 = (MMsg0)objc_msgSend;

        if (__mps_matmul_f16_pipeline == (void*)0) {
            void* p = __mps_make_pipeline("matmul_kernel_f16");
            if (p == (void*)0) return 0;
            __mps_matmul_f16_pipeline = p;
        }

        unsigned long nA = M * K;
        unsigned long nB = K * N;
        unsigned long nOut = M * N;
        unsigned short* aHalf = (unsigned short*)malloc(nA * 2UL);
        unsigned short* bHalf = (unsigned short*)malloc(nB * 2UL);
        unsigned short* outHalf = (unsigned short*)malloc(nOut * 2UL);
        if (aHalf == (void*)0 || bHalf == (void*)0 || outHalf == (void*)0) {
            if (aHalf != (void*)0) free((void*)aHalf);
            if (bHalf != (void*)0) free((void*)bHalf);
            if (outHalf != (void*)0) free((void*)outHalf);
            return 0;
        }
        f32_to_fp16_bulk(a, aHalf, nA);
        f32_to_fp16_bulk(b, bHalf, nB);

        unsigned long bytesA = nA * 2UL;
        unsigned long bytesB = nB * 2UL;
        unsigned long bytesOut = nOut * 2UL;
        void* bufA = __mps_buf_get_with_bytes((const void*)aHalf, bytesA);
        void* bufB = __mps_buf_get_with_bytes((const void*)bHalf, bytesB);
        void* bufOut = __mps_buf_get(bytesOut);
        free((void*)aHalf);
        free((void*)bHalf);
        if (bufA == (void*)0 || bufB == (void*)0 || bufOut == (void*)0) {
            free((void*)outHalf);
            return 0;
        }

        unsigned int Mu = (unsigned int)M;
        unsigned int Ku = (unsigned int)K;
        unsigned int Nu = (unsigned int)N;

        void* cmdBuf = msg0(queue, __sel_commandBuffer);
        void* encoder = msg0(cmdBuf, __sel_computeCommandEncoder);

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __sel_setComputePipelineState, __mps_matmul_f16_pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(encoder, __sel_setBuffer, bufB, 0UL, 1UL);
        setBufFn(encoder, __sel_setBuffer, bufOut, 0UL, 2UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(encoder, __sel_setBytes, (void*)&Mu, 4UL, 3UL);
        setBytesFn(encoder, __sel_setBytes, (void*)&Ku, 4UL, 4UL);
        setBytesFn(encoder, __sel_setBytes, (void*)&Nu, 4UL, 5UL);

        unsigned long tgW = 16UL;
        unsigned long tgH = 16UL;
        struct MTLSize tgSize;
        tgSize.width = tgW; tgSize.height = tgH; tgSize.depth = 1UL;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = (N + tgW - 1UL) / tgW;
        numThreadgroups.height = (M + tgH - 1UL) / tgH;
        numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        MMsgVoid0 msgVoid0 = (MMsgVoid0)objc_msgSend;
        msgVoid0(encoder, __sel_endEncoding);
        msgVoid0(cmdBuf, __sel_commit);
        msgVoid0(cmdBuf, __sel_waitUntilCompleted);

        void* outPtr = msg0(bufOut, __sel_contents);
        if (outPtr == (void*)0) {
            free((void*)outHalf);
            __mps_buf_put(bufA, bytesA);
            __mps_buf_put(bufB, bytesB);
            __mps_buf_put(bufOut, bytesOut);
            return 0;
        }
        memcpy((void*)outHalf, outPtr, bytesOut);
        fp16_to_f32_bulk(outHalf, out, nOut);
        free((void*)outHalf);

        __mps_buf_put(bufA, bytesA);
        __mps_buf_put(bufB, bytesB);
        __mps_buf_put(bufOut, bytesOut);
        return 1;
    }
}

int mps_matmul_bf16(const &float a, const &float b, &float out,
                     unsigned long M, unsigned long K, unsigned long N) {
    unsafe {
        if (!__mps_ensure_device_queue()) return 0;
        void* queue = __mps_queue;
        MMsg0 msg0 = (MMsg0)objc_msgSend;

        if (__mps_matmul_bf16_pipeline == (void*)0) {
            void* p = __mps_make_pipeline("matmul_kernel_bf16");
            if (p == (void*)0) return 0;
            __mps_matmul_bf16_pipeline = p;
        }

        unsigned long nA = M * K;
        unsigned long nB = K * N;
        unsigned long nOut = M * N;
        unsigned short* aBf = (unsigned short*)malloc(nA * 2UL);
        unsigned short* bBf = (unsigned short*)malloc(nB * 2UL);
        unsigned short* outBf = (unsigned short*)malloc(nOut * 2UL);
        if (aBf == (void*)0 || bBf == (void*)0 || outBf == (void*)0) {
            if (aBf != (void*)0) free((void*)aBf);
            if (bBf != (void*)0) free((void*)bBf);
            if (outBf != (void*)0) free((void*)outBf);
            return 0;
        }
        f32_to_bf16_bulk(a, aBf, nA);
        f32_to_bf16_bulk(b, bBf, nB);

        unsigned long bytesA = nA * 2UL;
        unsigned long bytesB = nB * 2UL;
        unsigned long bytesOut = nOut * 2UL;
        void* bufA = __mps_buf_get_with_bytes((const void*)aBf, bytesA);
        void* bufB = __mps_buf_get_with_bytes((const void*)bBf, bytesB);
        void* bufOut = __mps_buf_get(bytesOut);
        free((void*)aBf);
        free((void*)bBf);
        if (bufA == (void*)0 || bufB == (void*)0 || bufOut == (void*)0) {
            free((void*)outBf);
            return 0;
        }

        unsigned int Mu = (unsigned int)M;
        unsigned int Ku = (unsigned int)K;
        unsigned int Nu = (unsigned int)N;

        void* cmdBuf = msg0(queue, __sel_commandBuffer);
        void* encoder = msg0(cmdBuf, __sel_computeCommandEncoder);

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __sel_setComputePipelineState, __mps_matmul_bf16_pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(encoder, __sel_setBuffer, bufB, 0UL, 1UL);
        setBufFn(encoder, __sel_setBuffer, bufOut, 0UL, 2UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(encoder, __sel_setBytes, (void*)&Mu, 4UL, 3UL);
        setBytesFn(encoder, __sel_setBytes, (void*)&Ku, 4UL, 4UL);
        setBytesFn(encoder, __sel_setBytes, (void*)&Nu, 4UL, 5UL);

        unsigned long tgW = 16UL;
        unsigned long tgH = 16UL;
        struct MTLSize tgSize;
        tgSize.width = tgW; tgSize.height = tgH; tgSize.depth = 1UL;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = (N + tgW - 1UL) / tgW;
        numThreadgroups.height = (M + tgH - 1UL) / tgH;
        numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        MMsgVoid0 msgVoid0 = (MMsgVoid0)objc_msgSend;
        msgVoid0(encoder, __sel_endEncoding);
        msgVoid0(cmdBuf, __sel_commit);
        msgVoid0(cmdBuf, __sel_waitUntilCompleted);

        void* outPtr = msg0(bufOut, __sel_contents);
        if (outPtr == (void*)0) {
            free((void*)outBf);
            __mps_buf_put(bufA, bytesA);
            __mps_buf_put(bufB, bytesB);
            __mps_buf_put(bufOut, bytesOut);
            return 0;
        }
        memcpy((void*)outBf, outPtr, bytesOut);
        bf16_to_f32_bulk(outBf, out, nOut);
        free((void*)outBf);

        __mps_buf_put(bufA, bytesA);
        __mps_buf_put(bufB, bytesB);
        __mps_buf_put(bufOut, bytesOut);
        return 1;
    }
}

// Shared elementwise-relu dispatch body for mps_relu_f16/mps_relu_bf16 --
// only the pipeline name and the conversion functions differ, so this
// takes both as parameters rather than duplicating the whole
// upload/dispatch/readback sequence a third and fourth time.
typedef fn void(const float*, unsigned short*, unsigned long) __MpsToReducedFn;
typedef fn void(const unsigned short*, float*, unsigned long) __MpsToFloatFn;

static int __mps_relu_reduced_precision(const char* kernelName, void** pipelineSlot,
                                         const float* a, float* out, unsigned long n,
                                         __MpsToReducedFn toReduced, __MpsToFloatFn toFloat) {
    unsafe {
        if (!__mps_ensure_device_queue()) return 0;
        void* queue = __mps_queue;
        MMsg0 msg0 = (MMsg0)objc_msgSend;

        if (*pipelineSlot == (void*)0) {
            void* p = __mps_make_pipeline(kernelName);
            if (p == (void*)0) return 0;
            *pipelineSlot = p;
        }

        unsigned short* aRed = (unsigned short*)malloc(n * 2UL);
        unsigned short* outRed = (unsigned short*)malloc(n * 2UL);
        if (aRed == (void*)0 || outRed == (void*)0) {
            if (aRed != (void*)0) free((void*)aRed);
            if (outRed != (void*)0) free((void*)outRed);
            return 0;
        }
        toReduced(a, aRed, n);

        unsigned long bytes = n * 2UL;
        void* bufA = __mps_buf_get_with_bytes((const void*)aRed, bytes);
        void* bufOut = __mps_buf_get(bytes);
        free((void*)aRed);
        if (bufA == (void*)0 || bufOut == (void*)0) {
            free((void*)outRed);
            return 0;
        }

        void* cmdBuf = msg0(queue, __sel_commandBuffer);
        void* encoder = msg0(cmdBuf, __sel_computeCommandEncoder);

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __sel_setComputePipelineState, *pipelineSlot);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(encoder, __sel_setBuffer, bufOut, 0UL, 1UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        unsigned int nParam = (unsigned int)n;
        setBytesFn(encoder, __sel_setBytes, (void*)&nParam, 4UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 256UL ? n : 256UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = (n + tgWidth - 1UL) / tgWidth;
        numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        MMsgVoid0 msgVoid0 = (MMsgVoid0)objc_msgSend;
        msgVoid0(encoder, __sel_endEncoding);
        msgVoid0(cmdBuf, __sel_commit);
        msgVoid0(cmdBuf, __sel_waitUntilCompleted);

        void* outPtr = msg0(bufOut, __sel_contents);
        if (outPtr == (void*)0) {
            free((void*)outRed);
            __mps_buf_put(bufA, bytes);
            __mps_buf_put(bufOut, bytes);
            return 0;
        }
        memcpy((void*)outRed, outPtr, bytes);
        toFloat(outRed, out, n);
        free((void*)outRed);

        __mps_buf_put(bufA, bytes);
        __mps_buf_put(bufOut, bytes);
        return 1;
    }
}

int mps_relu_f16(const &float a, &float out, unsigned long n) {
    return __mps_relu_reduced_precision("relu_kernel_f16", &__mps_relu_f16_pipeline, a, out, n,
                                         f32_to_fp16_bulk, fp16_to_f32_bulk);
}

int mps_relu_bf16(const &float a, &float out, unsigned long n) {
    return __mps_relu_reduced_precision("relu_kernel_bf16", &__mps_relu_bf16_pipeline, a, out, n,
                                         f32_to_bf16_bulk, bf16_to_f32_bulk);
}

// ── fp16 / bf16 persistent-buffer / batched tier (see gpu_mps.h) ────────────
?&void mps_upload_f16_persistent(const &float data, unsigned long n) {
    unsafe {
        unsigned short* tmp = (unsigned short*)malloc(n * 2UL);
        if (tmp == (void*)0) return (void*)0;
        f32_to_fp16_bulk(data, tmp, n);
        void* buf = mps_upload_persistent((const float*)tmp, n * 2UL);
        free((void*)tmp);
        return buf;
    }
}

?&void mps_upload_bf16_persistent(const &float data, unsigned long n) {
    unsafe {
        unsigned short* tmp = (unsigned short*)malloc(n * 2UL);
        if (tmp == (void*)0) return (void*)0;
        f32_to_bf16_bulk(data, tmp, n);
        void* buf = mps_upload_persistent((const float*)tmp, n * 2UL);
        free((void*)tmp);
        return buf;
    }
}

// Shared dispatch body for every matmul-shaped fp16/bf16 chained op
// (matmul_kernel_f16/bf16, matmul_atb_kernel_f16/bf16, matmul_abt_kernel_
// f16/bf16 all take the same (bufA, bufB, bufOut, uint, uint, uint)
// binding shape from the host side -- only the kernel name and what the
// three uint constants/output shape mean differ between the three ops).
static int __mps_matmul3_chained(const char* kernelName, void** pipelineSlot,
                                  void* bufA, void* bufB, unsigned short* out,
                                  unsigned int p0, unsigned int p1, unsigned int p2,
                                  unsigned long outRows, unsigned long outCols) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (*pipelineSlot == (void*)0) {
            void* p = __mps_make_pipeline(kernelName);
            if (p == (void*)0) return 0;
            *pipelineSlot = p;
        }
        unsigned long bytesOut = outRows * outCols * 2UL;
        void* bufOut = __mps_buf_get_batched(bytesOut);
        if (bufOut == (void*)0) return 0;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, *pipelineSlot);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufB, 0UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&p0, 4UL, 3UL);
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&p1, 4UL, 4UL);
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&p2, 4UL, 5UL);

        unsigned long tgW = 16UL;
        unsigned long tgH = 16UL;
        struct MTLSize tgSize;
        tgSize.width = tgW; tgSize.height = tgH; tgSize.depth = 1UL;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = (outCols + tgW - 1UL) / tgW;
        numThreadgroups.height = (outRows + tgH - 1UL) / tgH;
        numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytesOut;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

static void* __mps_matmul_f16_persistent_pipeline = (void*)0;
int mps_matmul_f16_persistent(&void bufA, &void bufB, &unsigned short out,
                               unsigned long M, unsigned long K, unsigned long N) {
    return __mps_matmul3_chained("matmul_kernel_f16", &__mps_matmul_f16_persistent_pipeline,
                                  bufA, bufB, out, (unsigned int)M, (unsigned int)K, (unsigned int)N, M, N);
}

static void* __mps_matmul_bf16_persistent_pipeline = (void*)0;
int mps_matmul_bf16_persistent(&void bufA, &void bufB, &unsigned short out,
                                unsigned long M, unsigned long K, unsigned long N) {
    return __mps_matmul3_chained("matmul_kernel_bf16", &__mps_matmul_bf16_persistent_pipeline,
                                  bufA, bufB, out, (unsigned int)M, (unsigned int)K, (unsigned int)N, M, N);
}

static void* __mps_matmul_atb_f16_pipeline = (void*)0;
int mps_matmul_atb_f16_chained_ab(&void bufA, &void bufDC, &unsigned short out,
                                   unsigned long M, unsigned long K, unsigned long N) {
    return __mps_matmul3_chained("matmul_atb_kernel_f16", &__mps_matmul_atb_f16_pipeline,
                                  bufA, bufDC, out, (unsigned int)M, (unsigned int)K, (unsigned int)N, K, N);
}

static void* __mps_matmul_atb_bf16_pipeline = (void*)0;
int mps_matmul_atb_bf16_chained_ab(&void bufA, &void bufDC, &unsigned short out,
                                    unsigned long M, unsigned long K, unsigned long N) {
    return __mps_matmul3_chained("matmul_atb_kernel_bf16", &__mps_matmul_atb_bf16_pipeline,
                                  bufA, bufDC, out, (unsigned int)M, (unsigned int)K, (unsigned int)N, K, N);
}

static void* __mps_matmul_abt_f16_pipeline = (void*)0;
int mps_matmul_abt_f16_persistent(&void bufDC, &void bufB, &unsigned short out,
                                   unsigned long M, unsigned long N, unsigned long K) {
    return __mps_matmul3_chained("matmul_abt_kernel_f16", &__mps_matmul_abt_f16_pipeline,
                                  bufDC, bufB, out, (unsigned int)M, (unsigned int)N, (unsigned int)K, M, K);
}

// M==1 GEMV path for mps_matmul_abt_bf16_persistent -- see matmul_abt_
// kernel_bf16_gemv's comment in gpu_mps_kernels.metal. Dispatch is 1D over
// K, one output per simdgroup, GEMV_SIMDGROUPS (must match the kernel's)
// per threadgroup (buffer(3)=N, buffer(4)=K -- no M buffer, since this
// path only exists for M==1).
#define GEMV_SIMDGROUPS ((unsigned long)4)
static void* __mps_matmul_abt_bf16_gemv_pipeline = (void*)0;
static int __mps_matmul_abt_bf16_gemv_chained(void* bufA, void* bufB, unsigned short* out,
                                               unsigned int N, unsigned int K) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (__mps_matmul_abt_bf16_gemv_pipeline == (void*)0) {
            void* p = __mps_make_pipeline("matmul_abt_kernel_bf16_gemv");
            if (p == (void*)0) return 0;
            __mps_matmul_abt_bf16_gemv_pipeline = p;
        }
        unsigned long bytesOut = (unsigned long)K * 2UL;
        void* bufOut = __mps_buf_get_batched(bytesOut);
        if (bufOut == (void*)0) return 0;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, __mps_matmul_abt_bf16_gemv_pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufB, 0UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&N, 4UL, 3UL);
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&K, 4UL, 4UL);

        struct MTLSize tgSize;
        tgSize.width = GEMV_SIMDGROUPS * 32UL; tgSize.height = 1UL; tgSize.depth = 1UL;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = ((unsigned long)K + GEMV_SIMDGROUPS - 1UL) / GEMV_SIMDGROUPS;
        numThreadgroups.height = 1UL;
        numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytesOut;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

static void* __mps_matmul_abt_bf16_pipeline = (void*)0;
int mps_matmul_abt_bf16_persistent(&void bufDC, &void bufB, &unsigned short out,
                                    unsigned long M, unsigned long N, unsigned long K) {
    // M==1 (a decode step's projections) routes to the dedicated GEMV
    // kernel; the tiled kernel would compute and discard 15 of every 16
    // rows. See gpu_mps_kernels.metal for why a simdgroup_matrix bf16 tier
    // isn't used for M>1 either.
    if (M == 1UL) {
        int rc = __mps_matmul_abt_bf16_gemv_chained(bufDC, bufB, out, (unsigned int)N, (unsigned int)K);
        if (rc) { return rc; }
    }
    return __mps_matmul3_chained("matmul_abt_kernel_bf16", &__mps_matmul_abt_bf16_pipeline,
                                  bufDC, bufB, out, (unsigned int)M, (unsigned int)N, (unsigned int)K, M, K);
}

// Shared dispatch body for a single-device-input elementwise op (relu).
static int __mps_unary_chained(const char* kernelName, void** pipelineSlot,
                                void* bufA, unsigned short* out, unsigned long n) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (*pipelineSlot == (void*)0) {
            void* p = __mps_make_pipeline(kernelName);
            if (p == (void*)0) return 0;
            *pipelineSlot = p;
        }
        unsigned long bytes = n * 2UL;
        void* bufOut = __mps_buf_get_batched(bytes);
        if (bufOut == (void*)0) return 0;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, *pipelineSlot);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 1UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytes;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

static void* __mps_relu_f16_chained_pipeline = (void*)0;
int mps_relu_f16_chained(&void bufA, &unsigned short out, unsigned long n) {
    return __mps_unary_chained("relu_kernel_f16", &__mps_relu_f16_chained_pipeline, bufA, out, n);
}

static void* __mps_relu_bf16_chained_pipeline = (void*)0;
int mps_relu_bf16_chained(&void bufA, &unsigned short out, unsigned long n) {
    return __mps_unary_chained("relu_kernel_bf16", &__mps_relu_bf16_chained_pipeline, bufA, out, n);
}

// Shared dispatch body for a two-device-input elementwise op (relu
// backward: dZ1 = relu'(Z1) * dH, both operands already GPU-pending).
static int __mps_binary_dev_chained(const char* kernelName, void** pipelineSlot,
                                     void* bufA, void* bufB, unsigned short* out, unsigned long n) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (*pipelineSlot == (void*)0) {
            void* p = __mps_make_pipeline(kernelName);
            if (p == (void*)0) return 0;
            *pipelineSlot = p;
        }
        unsigned long bytes = n * 2UL;
        void* bufOut = __mps_buf_get_batched(bytes);
        if (bufOut == (void*)0) return 0;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, *pipelineSlot);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufB, 0UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytes;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

static void* __mps_relu_backward_f16_pipeline = (void*)0;
int mps_relu_backward_f16_chained_ab(&void bufA, &void bufSelfGrad, &unsigned short out, unsigned long n) {
    return __mps_binary_dev_chained("relu_backward_kernel_f16", &__mps_relu_backward_f16_pipeline, bufA, bufSelfGrad, out, n);
}

static void* __mps_relu_backward_bf16_pipeline = (void*)0;
int mps_relu_backward_bf16_chained_ab(&void bufA, &void bufSelfGrad, &unsigned short out, unsigned long n) {
    return __mps_binary_dev_chained("relu_backward_kernel_bf16", &__mps_relu_backward_bf16_pipeline, bufA, bufSelfGrad, out, n);
}

// out[i] = bufA[i] - bHost[i] -- 'bHost' is a plain CPU array of raw
// fp16/bf16 bits, uploaded fresh each call (see gpu_mps.h's comment).
static int __mps_sub_chained(const char* kernelName, void** pipelineSlot,
                              void* bufA, const unsigned short* bHost, unsigned short* out, unsigned long n) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (*pipelineSlot == (void*)0) {
            void* p = __mps_make_pipeline(kernelName);
            if (p == (void*)0) return 0;
            *pipelineSlot = p;
        }
        unsigned long bytes = n * 2UL;
        void* bufB = __mps_buf_get_with_bytes_batched((const void*)bHost, bytes);
        void* bufOut = __mps_buf_get_batched(bytes);
        if (bufB == (void*)0 || bufOut == (void*)0) return 0;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, *pipelineSlot);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufB, 0UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytes;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

static void* __mps_sub_f16_pipeline = (void*)0;
int mps_sub_f16_chained(&void bufA, const &unsigned short bHalf, &unsigned short out, unsigned long n) {
    return __mps_sub_chained("sub_kernel_f16", &__mps_sub_f16_pipeline, bufA, bHalf, out, n);
}

static void* __mps_sub_bf16_pipeline = (void*)0;
int mps_sub_bf16_chained(&void bufA, const &unsigned short bBf, &unsigned short out, unsigned long n) {
    return __mps_sub_chained("sub_kernel_bf16", &__mps_sub_bf16_pipeline, bufA, bBf, out, n);
}

// out[i] = buf[i] * k -- 'k' stays a plain float scalar (see gpu_mps_
// kernels.metal's scale_kernel_f16/bf16 comment).
static int __mps_scale_chained(const char* kernelName, void** pipelineSlot,
                                void* buf, float k, unsigned short* out, unsigned long n) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (*pipelineSlot == (void*)0) {
            void* p = __mps_make_pipeline(kernelName);
            if (p == (void*)0) return 0;
            *pipelineSlot = p;
        }
        unsigned long bytes = n * 2UL;
        void* bufOut = __mps_buf_get_batched(bytes);
        if (bufOut == (void*)0) return 0;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, *pipelineSlot);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, buf, 0UL, 0UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        float kVal = k;
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&kVal, 4UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytes;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

static void* __mps_scale_f16_pipeline = (void*)0;
int mps_scale_f16_chained(&void buf, float k, &unsigned short out, unsigned long n) {
    return __mps_scale_chained("scale_kernel_f16", &__mps_scale_f16_pipeline, buf, k, out, n);
}

static void* __mps_scale_bf16_pipeline = (void*)0;
int mps_scale_bf16_chained(&void buf, float k, &unsigned short out, unsigned long n) {
    return __mps_scale_chained("scale_kernel_bf16", &__mps_scale_bf16_pipeline, buf, k, out, n);
}

// Both operands device buffers -- same binding shape as relu-backward's
// __mps_binary_dev_chained above.
static void* __mps_add_bf16_pipeline = (void*)0;
int mps_add_bf16_chained(&void bufA, &void bufB, &unsigned short out, unsigned long n) {
    return __mps_binary_dev_chained("add_kernel_bf16", &__mps_add_bf16_pipeline, bufA, bufB, out, n);
}

static void* __mps_gelu_mul_bf16_pipeline = (void*)0;
int mps_gelu_mul_bf16_chained(&void bufA, &void bufB, &unsigned short out, unsigned long n) {
    return __mps_binary_dev_chained("gelu_mul_kernel_bf16", &__mps_gelu_mul_bf16_pipeline, bufA, bufB, out, n);
}

// Row-wise RMSNorm -- see rmsnorm_kernel_bf16's comment in gpu_mps_
// kernels.metal. One RMSNORM_TG-thread threadgroup per row; RMSNORM_TG
// must match the kernel's.
#define RMSNORM_TG ((unsigned long)256)
static void* __mps_rmsnorm_bf16_pipeline = (void*)0;
int mps_rmsnorm_bf16_chained(&void bufX, &void weightBuf, &unsigned short out,
                              unsigned long rows, unsigned long cols, float eps) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (__mps_rmsnorm_bf16_pipeline == (void*)0) {
            void* p = __mps_make_pipeline("rmsnorm_kernel_bf16");
            if (p == (void*)0) return 0;
            __mps_rmsnorm_bf16_pipeline = p;
        }
        unsigned long bytes = rows * cols * 2UL;
        void* bufOut = __mps_buf_get_batched(bytes);
        if (bufOut == (void*)0) return 0;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, __mps_rmsnorm_bf16_pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufX, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, weightBuf, 0UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        unsigned int colsVal = (unsigned int)cols;
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&colsVal, 4UL, 3UL);
        float epsVal = eps;
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&epsVal, 4UL, 4UL);

        struct MTLSize tgSize;
        tgSize.width = RMSNORM_TG; tgSize.height = 1UL; tgSize.depth = 1UL;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = rows; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)out;
            __mps_pending_bytes[__mps_pending_count] = bytes;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        return 1;
    }
}

// In-place SGD step against a persistent fp16/bf16 device buffer -- no
// 'out'/pending-readback registration, result stays on the GPU (see
// mps_sgd_update_f32_chained's comment).
static int __mps_sgd_chained(const char* kernelName, void** pipelineSlot,
                              void* bufW, void* bufGrad, float lr, unsigned long n) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (*pipelineSlot == (void*)0) {
            void* p = __mps_make_pipeline(kernelName);
            if (p == (void*)0) return 0;
            *pipelineSlot = p;
        }
        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, *pipelineSlot);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufW, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufGrad, 0UL, 1UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        float lrVal = lr;
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&lrVal, 4UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(__mps_batch_encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        return 1;
    }
}

static void* __mps_sgd_update_f16_pipeline = (void*)0;
int mps_sgd_update_f16_chained(&void bufW, &void bufGrad, float lr, unsigned long n) {
    return __mps_sgd_chained("sgd_update_kernel_f16", &__mps_sgd_update_f16_pipeline, bufW, bufGrad, lr, n);
}

static void* __mps_sgd_update_bf16_pipeline = (void*)0;
int mps_sgd_update_bf16_chained(&void bufW, &void bufGrad, float lr, unsigned long n) {
    return __mps_sgd_chained("sgd_update_kernel_bf16", &__mps_sgd_update_bf16_pipeline, bufW, bufGrad, lr, n);
}

// ── Batched multi-head attention (bf16) ─────────────────────────────────────
// See attn_batch_bf16_kernel's comment in gpu_mps_kernels.metal. The kernel
// is instantiated per head dim (attn_batch_bf16_kernel_d<headDim>, one
// cached pipeline each) and runs one threadgroup of ATTN_SIMDGROUPS(D)
// simdgroups per (batch, head) -- __mps_attn_batch_simdgroups mirrors the
// kernel's formula. Other
// head dims, or a head count that isn't a whole multiple of the KV head
// count, are refused up front so callers fall back to their CPU path.
static unsigned long __mps_attn_batch_simdgroups(unsigned long headDim) {
    unsigned long sg = 4096UL / headDim;
    return sg < 32UL ? sg : 32UL;
}

// The attention kernels below (decode, flash, linear) are all templated
// on head dim and instantiated for 64/128/256/512, as
// '<kernel>_d<headDim>'; each op keeps one cached pipeline per head dim.
// Slot for 'headDim', or -1 if no instantiation exists.
static long __mps_head_dim_slot(unsigned long headDim) {
    if (headDim == 64UL) return 0L;
    if (headDim == 128UL) return 1L;
    if (headDim == 256UL) return 2L;
    if (headDim == 512UL) return 3L;
    return -1L;
}

static int __mps_attn_shape_ok(unsigned long numHeads, unsigned long numKvHeads, unsigned long headDim) {
    return __mps_head_dim_slot(headDim) >= 0L && numKvHeads != 0UL && numHeads % numKvHeads == 0UL;
}

// Cached pipeline for a head-dim-templated kernel, given its four
// instantiations' names. Callers have already checked __mps_attn_shape_ok.
static void* __mps_head_dim_pipeline(void** slots, unsigned long headDim, const char* name64,
                                      const char* name128, const char* name256, const char* name512) {
    unsafe {
        long slot = __mps_head_dim_slot(headDim);
        if (slots[slot] == (void*)0) {
            const char* name = name64;
            if (slot == 1L) name = name128;
            if (slot == 2L) name = name256;
            if (slot == 3L) name = name512;
            slots[slot] = __mps_make_pipeline(name);
        }
        return slots[slot];
    }
}

// Binds 'count' 4-byte constants to consecutive buffer indices starting at
// 'firstIndex'.
static void __mps_set_uints(void* encoder, unsigned int* params, unsigned long count, unsigned long firstIndex) {
    unsafe {
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        unsigned long i = 0UL;
        while (i < count) {
            setBytesFn(encoder, __sel_setBytes, (void*)&params[i], 4UL, firstIndex + i);
            i = i + 1UL;
        }
    }
}

// Registers 'bufOut' as a chained op's pending readback into 'out' -- the
// same bookkeeping every *_chained op above does inline.
static void __mps_pending_add(void* bufOut, void* out, unsigned long bytes) {
    if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
        __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
        __mps_pending_cpuOut[__mps_pending_count] = out;
        __mps_pending_bytes[__mps_pending_count] = bytes;
        __mps_pending_count = __mps_pending_count + 1UL;
    }
}

static void* __mps_attn_batch_pipelines[4];

// Binds the pipeline, buffers 0-4 and the five uint constants, then
// dispatches -- shared by the one-shot and chained entry points, which
// differ only in which encoder they target and how 'bufOut' is read back.
// Callers have already checked __mps_attn_shape_ok.
static int __mps_attn_batch_encode(void* encoder, void* bufQ, void* bufK, void* bufV, void* bufCacheLens,
                                    void* bufOut, unsigned long batchSize, unsigned long numHeads,
                                    unsigned long numKvHeads, unsigned long headDim, unsigned long maxKvLen,
                                    int isSliding, unsigned long slidingWindow) {
    unsafe {
        void* pipeline = __mps_head_dim_pipeline(__mps_attn_batch_pipelines, headDim,
                                                 "attn_batch_bf16_kernel_d64", "attn_batch_bf16_kernel_d128",
                                                 "attn_batch_bf16_kernel_d256", "attn_batch_bf16_kernel_d512");
        if (pipeline == (void*)0) return 0;
        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __sel_setBuffer, bufQ, 0UL, 0UL);
        setBufFn(encoder, __sel_setBuffer, bufK, 0UL, 1UL);
        setBufFn(encoder, __sel_setBuffer, bufV, 0UL, 2UL);
        setBufFn(encoder, __sel_setBuffer, bufCacheLens, 0UL, 3UL);
        setBufFn(encoder, __sel_setBuffer, bufOut, 0UL, 4UL);

        unsigned int params[5];
        params[0] = (unsigned int)numHeads;
        params[1] = (unsigned int)numKvHeads;
        params[2] = (unsigned int)maxKvLen;
        params[3] = isSliding ? 1U : 0U;
        params[4] = (unsigned int)slidingWindow;
        __mps_set_uints(encoder, params, 5UL, 5UL);

        struct MTLSize tgSize;
        tgSize.width = __mps_attn_batch_simdgroups(headDim) * 32UL; tgSize.height = 1UL; tgSize.depth = 1UL;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = batchSize; numThreadgroups.height = numHeads; numThreadgroups.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);
        return 1;
    }
}

// Standalone one-shot dispatch (own commit+wait+readback, same shape as
// __mps_run_binary_kernel above) -- every input changes each decode step,
// so nothing here benefits from staying GPU-resident across calls.
int mps_attn_batch_bf16(const &unsigned short q, const &unsigned short kAll, const &unsigned short vAll,
                          const &unsigned int cacheLens, &unsigned short out,
                          unsigned long batchSize, unsigned long numHeads, unsigned long numKvHeads,
                          unsigned long headDim, unsigned long maxKvLen,
                          int isSliding, unsigned long slidingWindow) {
    if (!__mps_attn_shape_ok(numHeads, numKvHeads, headDim)) return 0;
    unsafe {
        if (!__mps_ensure_device_queue()) return 0;
        MMsg0 msg0 = (MMsg0)objc_msgSend;

        unsigned long qBytes = batchSize * numHeads * headDim * 2UL;
        unsigned long kvBytes = batchSize * maxKvLen * numKvHeads * headDim * 2UL;
        unsigned long cacheLensBytes = batchSize * 4UL;

        void* bufQ = __mps_buf_get_with_bytes((const void*)q, qBytes);
        void* bufK = __mps_buf_get_with_bytes((const void*)kAll, kvBytes);
        void* bufV = __mps_buf_get_with_bytes((const void*)vAll, kvBytes);
        void* bufCacheLens = __mps_buf_get_with_bytes((const void*)cacheLens, cacheLensBytes);
        void* bufOut = __mps_buf_get(qBytes);

        int ok = bufQ != (void*)0 && bufK != (void*)0 && bufV != (void*)0 &&
                 bufCacheLens != (void*)0 && bufOut != (void*)0;
        if (ok) {
            void* cmdBuf = msg0(__mps_queue, __sel_commandBuffer);
            void* encoder = msg0(cmdBuf, __sel_computeCommandEncoder);
            ok = __mps_attn_batch_encode(encoder, bufQ, bufK, bufV, bufCacheLens, bufOut,
                                         batchSize, numHeads, numKvHeads, headDim, maxKvLen,
                                         isSliding, slidingWindow);
            MMsgVoid0 msgVoid0 = (MMsgVoid0)objc_msgSend;
            msgVoid0(encoder, __sel_endEncoding);
            if (ok) {
                msgVoid0(cmdBuf, __sel_commit);
                msgVoid0(cmdBuf, __sel_waitUntilCompleted);
                void* outPtr = msg0(bufOut, __sel_contents);
                ok = outPtr != (void*)0;
                if (ok) { memcpy((void*)out, outPtr, qBytes); }
            }
        }

        if (bufQ != (void*)0) { __mps_buf_put(bufQ, qBytes); }
        if (bufK != (void*)0) { __mps_buf_put(bufK, kvBytes); }
        if (bufV != (void*)0) { __mps_buf_put(bufV, kvBytes); }
        if (bufCacheLens != (void*)0) { __mps_buf_put(bufCacheLens, cacheLensBytes); }
        if (bufOut != (void*)0) { __mps_buf_put(bufOut, qBytes); }
        return ok;
    }
}

// Chained variant -- see gpu_mps.h's comment. Encodes onto the shared batch
// encoder and registers 'out' as the pending readback target like every
// other *_chained op, leaving the result as mps_batch_last_output_buffer().
int mps_attn_batch_bf16_chained(&void bufQ, &void bufK, &void bufV, &void bufCacheLens,
                                  &unsigned short out,
                                  unsigned long batchSize, unsigned long numHeads, unsigned long numKvHeads,
                                  unsigned long headDim, unsigned long maxKvLen,
                                  int isSliding, unsigned long slidingWindow) {
    if (!__mps_batch_active) return 0;
    if (!__mps_attn_shape_ok(numHeads, numKvHeads, headDim)) return 0;
    unsafe {
        unsigned long outBytes = batchSize * numHeads * headDim * 2UL;
        void* bufOut = __mps_buf_get_batched(outBytes);
        if (bufOut == (void*)0) return 0;
        if (!__mps_attn_batch_encode(__mps_batch_encoder, bufQ, bufK, bufV, bufCacheLens, bufOut,
                                     batchSize, numHeads, numKvHeads, headDim, maxKvLen,
                                     isSliding, slidingWindow)) return 0;
        __mps_pending_add(bufOut, (void*)out, outBytes);
        return 1;
    }
}

// ── Flash and linear attention ──────────────────────────────────────────────
// See the "Flash and linear attention" kernels in gpu_mps_kernels.metal.
// Every call fills one MpsAttnParams (the kernels' AttnParams, field for
// field) with the problem size and element strides, so the head-major
// [batch][heads][seq][dim] API (mps_*_attention_ex) and the token-major
// [seq][heads*dim] bf16 API (mps_*_attn_bf16) share the same kernels.
struct MpsAttnParams {
    unsigned int seqLen;
    unsigned int kvLen;
    unsigned int heads;
    unsigned int kvHeads;
    unsigned int causal;
    unsigned int window;
    unsigned int hasState;
    unsigned int seg;
    float scale;
    unsigned int qRow;
    unsigned int qHead;
    unsigned int qBatch;
    unsigned int kvRow;
    unsigned int kvHead;
    unsigned int kvBatch;
};

// FLASH_SG / FLASH_BQ / LIN_SG must match gpu_mps_kernels.metal: the matrix
// kernel (head dim <= 256) runs FLASH_SG simdgroups of 8 query rows, the
// head-dim-512 kernel FLASH_BQ simdgroups of one row, linear attention
// LIN_SG simdgroups of LIN_COLS(headDim) output columns.
#define FLASH_SG ((unsigned long)4)
#define FLASH_BQ ((unsigned long)16)
#define LIN_SG   ((unsigned long)4)

int mps_attention_head_dim_ok(unsigned long headDim) {
    return headDim == 32UL || headDim == 64UL || headDim == 96UL || headDim == 128UL ||
           headDim == 256UL || headDim == 512UL;
}

static unsigned long __mps_attn_dim_index(unsigned long headDim) {
    if (headDim == 32UL) return 0UL;
    if (headDim == 64UL) return 1UL;
    if (headDim == 96UL) return 2UL;
    if (headDim == 128UL) return 3UL;
    if (headDim == 256UL) return 4UL;
    return 5UL;
}

// One cached pipeline per (dtype, head dim): slot dtype * 6 + dim index.
static void* __mps_flash_pipelines[18];
static void* __mps_linear_pipelines[18];
static void* __mps_linear_z_pipelines[3];

// "<prefix><f32|f16|bf16>_d<headDim>" into 'buf' (>= 48 bytes); headDim 0
// leaves off the "_d" suffix.
static void __mps_attn_kernel_name(&char buf, const char* prefix, int dtype, unsigned long headDim) {
    unsafe {
        char* out = (char*)buf;
        unsigned long n = 0UL;
        while (prefix[n] != (char)0) { out[n] = prefix[n]; n = n + 1UL; }
        const char* dt = dtype == MPS_FP16 ? "f16" : (dtype == MPS_BF16 ? "bf16" : "f32");
        unsigned long i = 0UL;
        while (dt[i] != (char)0) { out[n] = dt[i]; n = n + 1UL; i = i + 1UL; }
        if (headDim != 0UL) {
            out[n] = '_'; out[n + 1UL] = 'd'; n = n + 2UL;
            char digits[8];
            unsigned long nd = 0UL;
            unsigned long x = headDim;
            while (x != 0UL) { digits[nd] = (char)(48UL + x % 10UL); nd = nd + 1UL; x = x / 10UL; }
            while (nd != 0UL) { nd = nd - 1UL; out[n] = digits[nd]; n = n + 1UL; }
        }
        out[n] = (char)0;
    }
}

static void* __mps_attn_pipeline(void** slots, unsigned long slot, const char* prefix, int dtype,
                                 unsigned long headDim) {
    unsafe {
        if (slots[slot] == (void*)0) {
            char name[48];
            __mps_attn_kernel_name(&name[0], prefix, dtype, headDim);
            slots[slot] = __mps_make_pipeline(&name[0]);
        }
        return slots[slot];
    }
}

// Shared shape checks: supported head dim, kvHeads dividing heads, a
// non-empty problem, and kvLen >= seqLen whenever positions matter.
static int __mps_attn_check(struct MpsAttnShape s, int dtype, int causal) {
    if (dtype != 0 && dtype != MPS_FP16 && dtype != MPS_BF16) return 0;
    if (!mps_attention_head_dim_ok(s.headDim)) return 0;
    if (s.batch == 0UL || s.seqLen == 0UL || s.kvLen == 0UL) return 0;
    if (s.kvHeads == 0UL || s.heads % s.kvHeads != 0UL) return 0;
    if ((causal || s.window != 0UL) && s.kvLen < s.seqLen) return 0;
    return 1;
}

// Head-major strides: q/out [batch][heads][seqLen][D], k/v [batch][kvHeads][kvLen][D].
static struct MpsAttnParams __mps_attn_params_head_major(struct MpsAttnShape s, int causal) {
    struct MpsAttnParams p;
    p.seqLen = (unsigned int)s.seqLen; p.kvLen = (unsigned int)s.kvLen;
    p.heads = (unsigned int)s.heads; p.kvHeads = (unsigned int)s.kvHeads;
    p.causal = causal ? 1U : 0U; p.window = (unsigned int)s.window; p.hasState = 0U; p.seg = 0U;
    p.scale = (float)0.0;
    p.qRow = (unsigned int)s.headDim;
    p.qHead = (unsigned int)(s.seqLen * s.headDim);
    p.qBatch = (unsigned int)(s.heads * s.seqLen * s.headDim);
    p.kvRow = (unsigned int)s.headDim;
    p.kvHead = (unsigned int)(s.kvLen * s.headDim);
    p.kvBatch = (unsigned int)(s.kvHeads * s.kvLen * s.headDim);
    return p;
}

// Token-major strides (batch 1): q/out [seqLen][heads*D], k/v [kvLen][kvHeads*D].
static struct MpsAttnParams __mps_attn_params_token_major(struct MpsAttnShape s, int causal) {
    struct MpsAttnParams p = __mps_attn_params_head_major(s, causal);
    p.qRow = (unsigned int)(s.heads * s.headDim);
    p.qHead = (unsigned int)s.headDim;
    p.kvRow = (unsigned int)(s.kvHeads * s.headDim);
    p.kvHead = (unsigned int)s.headDim;
    return p;
}

// Encodes flash attention onto 'encoder' ('p.scale' <= 0: the kernel uses 1/sqrt(headDim)).
static int __mps_flash_encode(void* encoder, void* bufQ, void* bufK, void* bufV, void* bufOut,
                              struct MpsAttnShape s, int dtype, struct MpsAttnParams p) {
    unsafe {
        int wide = s.headDim > 256UL;
        unsigned long slot = (unsigned long)dtype * 6UL + __mps_attn_dim_index(s.headDim);
        void* pipeline = __mps_attn_pipeline(__mps_flash_pipelines, slot,
                                             wide ? "flash_attn_simd_" : "flash_attn_mma_", dtype, s.headDim);
        if (pipeline == (void*)0) return 0;
        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __sel_setBuffer, bufQ, 0UL, 0UL);
        setBufFn(encoder, __sel_setBuffer, bufK, 0UL, 1UL);
        setBufFn(encoder, __sel_setBuffer, bufV, 0UL, 2UL);
        setBufFn(encoder, __sel_setBuffer, bufOut, 0UL, 3UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(encoder, __sel_setBytes, (void*)&p, sizeof(struct MpsAttnParams), 4UL);

        unsigned long rowsPerGroup = wide ? FLASH_BQ : FLASH_SG * 8UL;
        struct MTLSize tgSize;
        tgSize.width = (wide ? FLASH_BQ : FLASH_SG) * 32UL; tgSize.height = 1UL; tgSize.depth = 1UL;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = (s.seqLen + rowsPerGroup - 1UL) / rowsPerGroup;
        numThreadgroups.height = s.heads;
        numThreadgroups.depth = s.batch;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);
        return 1;
    }
}

// Chunk-parallel linear attention (see linear_chunk_state in
// gpu_mps_kernels.metal): head dims 64 and 128, whole sequences only.
#define LINC_ROWS ((unsigned long)64)   // must match the kernel source

static void* __mps_linc_state_pipelines[6];
static void* __mps_linc_out_pipelines[12];   // [causal][dtype][dim]
static void* __mps_linc_prefix_pipelines[2];

// The per-segment states live in one scratch buffer this library keeps and
// grows. Calls in one batch may share it: the serial encoder orders their
// dispatches. A buffer replaced mid-batch is released when the batch ends.
static void* __mps_linc_scratch = (void*)0;
static unsigned long __mps_linc_scratch_bytes = 0UL;

static void* __mps_linc_scratch_get(unsigned long bytes) {
    if (bytes <= __mps_linc_scratch_bytes) return __mps_linc_scratch;
    if (__mps_linc_scratch != (void*)0) {
        if (__mps_batch_active) { __mps_batch_track_pool_buf(__mps_linc_scratch, __mps_linc_scratch_bytes); }
        else { __mps_buf_put(__mps_linc_scratch, __mps_linc_scratch_bytes); }
    }
    __mps_linc_scratch = __mps_buf_get(bytes);
    __mps_linc_scratch_bytes = __mps_linc_scratch == (void*)0 ? 0UL : bytes;
    return __mps_linc_scratch;
}

static int __mps_linear_chunked_encode(void* encoder, void* bufQ, void* bufK, void* bufV, void* bufOut,
                                       struct MpsAttnShape s, int dtype, struct MpsAttnParams p) {
    unsafe {
        unsigned long di = s.headDim == 64UL ? 0UL : 1UL;
        unsigned long slot = (unsigned long)dtype * 2UL + di;
        void* stateP = __mps_attn_pipeline(__mps_linc_state_pipelines, slot, "linear_chunk_state_", dtype, s.headDim);
        void* outP = p.causal ? __mps_attn_pipeline(__mps_linc_out_pipelines, 6UL + slot, "linear_chunk_out_causal_", dtype, s.headDim)
                              : __mps_attn_pipeline(__mps_linc_out_pipelines, slot, "linear_chunk_out_", dtype, s.headDim);
        if (__mps_linc_prefix_pipelines[di] == (void*)0) {
            __mps_linc_prefix_pipelines[di] = __mps_make_pipeline(di == 0UL ? "linear_chunk_prefix_d64"
                                                                            : "linear_chunk_prefix_d128");
        }
        void* prefixP = __mps_linc_prefix_pipelines[di];
        if (stateP == (void*)0 || outP == (void*)0 || prefixP == (void*)0) return 0;

        // Causal needs a state per LINC_ROWS-row chunk; non-causal only a
        // total, summed from at most 16 segments.
        unsigned long seg = LINC_ROWS;
        if (!p.causal) {
            unsigned long per = (s.kvLen + 15UL) / 16UL;
            per = (per + LINC_ROWS - 1UL) / LINC_ROWS * LINC_ROWS;
            if (per > seg) seg = per;
        }
        p.seg = (unsigned int)seg;
        unsigned long nSeg = (s.kvLen + seg - 1UL) / seg;
        unsigned long elems = s.headDim * s.headDim + s.headDim;
        void* states = __mps_linc_scratch_get(s.batch * s.heads * nSeg * elems * 4UL);
        if (states == (void*)0) return 0;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        unsigned long slabs = s.headDim / 64UL;
        struct MTLSize tgSize;
        struct MTLSize grid;
        tgSize.height = 1UL; tgSize.depth = 1UL;

        msgVoid1(encoder, __sel_setComputePipelineState, stateP);
        setBufFn(encoder, __sel_setBuffer, bufK, 0UL, 0UL);
        setBufFn(encoder, __sel_setBuffer, bufV, 0UL, 1UL);
        setBufFn(encoder, __sel_setBuffer, states, 0UL, 2UL);
        setBytesFn(encoder, __sel_setBytes, (void*)&p, sizeof(struct MpsAttnParams), 3UL);
        tgSize.width = s.headDim * 2UL;
        grid.width = nSeg; grid.height = s.heads * slabs; grid.depth = s.batch;
        dispatchFn(encoder, __sel_dispatchThreadgroups, grid, tgSize);

        msgVoid1(encoder, __sel_setComputePipelineState, prefixP);
        setBufFn(encoder, __sel_setBuffer, states, 0UL, 0UL);
        setBytesFn(encoder, __sel_setBytes, (void*)&p, sizeof(struct MpsAttnParams), 1UL);
        tgSize.width = 256UL;
        grid.width = (elems + 255UL) / 256UL; grid.height = s.batch * s.heads; grid.depth = 1UL;
        dispatchFn(encoder, __sel_dispatchThreadgroups, grid, tgSize);

        msgVoid1(encoder, __sel_setComputePipelineState, outP);
        setBufFn(encoder, __sel_setBuffer, bufQ, 0UL, 0UL);
        setBufFn(encoder, __sel_setBuffer, bufK, 0UL, 1UL);
        setBufFn(encoder, __sel_setBuffer, bufV, 0UL, 2UL);
        setBufFn(encoder, __sel_setBuffer, bufOut, 0UL, 3UL);
        setBufFn(encoder, __sel_setBuffer, states, 0UL, 4UL);
        setBytesFn(encoder, __sel_setBytes, (void*)&p, sizeof(struct MpsAttnParams), 5UL);
        tgSize.width = 256UL;
        grid.width = (s.seqLen + LINC_ROWS - 1UL) / LINC_ROWS; grid.height = s.heads * slabs; grid.depth = s.batch;
        dispatchFn(encoder, __sel_dispatchThreadgroups, grid, tgSize);
        return 1;
    }
}

// Encodes linear attention (and, with a state, the z update after it).
static int __mps_linear_encode(void* encoder, void* bufQ, void* bufK, void* bufV, void* bufOut, void* bufState,
                               struct MpsAttnShape s, int dtype, struct MpsAttnParams p) {
    if (bufState == (void*)0 && (s.headDim == 64UL || s.headDim == 128UL) && (!p.causal || s.kvLen == s.seqLen)) {
        return __mps_linear_chunked_encode(encoder, bufQ, bufK, bufV, bufOut, s, dtype, p);
    }
    unsafe {
        unsigned long slot = (unsigned long)dtype * 6UL + __mps_attn_dim_index(s.headDim);
        void* pipeline = __mps_attn_pipeline(__mps_linear_pipelines, slot, "linear_attn_", dtype, s.headDim);
        if (pipeline == (void*)0) return 0;
        int hasState = bufState != (void*)0;
        void* zPipeline = (void*)0;
        if (hasState) {
            zPipeline = __mps_attn_pipeline(__mps_linear_z_pipelines, (unsigned long)dtype,
                                            "linear_attn_z_update_", dtype, 0UL);
            if (zPipeline == (void*)0) return 0;
        }
        p.hasState = hasState ? 1U : 0U;
        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __sel_setBuffer, bufQ, 0UL, 0UL);
        setBufFn(encoder, __sel_setBuffer, bufK, 0UL, 1UL);
        setBufFn(encoder, __sel_setBuffer, bufV, 0UL, 2UL);
        setBufFn(encoder, __sel_setBuffer, bufOut, 0UL, 3UL);
        // The kernel never touches buffer(4) without a state; any buffer
        // satisfies the binding.
        setBufFn(encoder, __sel_setBuffer, hasState ? bufState : bufOut, 0UL, 4UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(encoder, __sel_setBytes, (void*)&p, sizeof(struct MpsAttnParams), 5UL);

        unsigned long cols = 1UL;   // LIN_COLS(D) in the kernel source
        if (s.headDim <= 256UL) cols = 2UL;
        if (s.headDim <= 128UL) cols = 4UL;
        if (s.headDim <= 64UL) cols = 8UL;
        if (s.headDim <= 32UL) cols = 16UL;
        unsigned long groups = s.headDim / cols;
        struct MTLSize tgSize;
        tgSize.width = LIN_SG * 32UL; tgSize.height = 1UL; tgSize.depth = 1UL;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = (groups + LIN_SG - 1UL) / LIN_SG;
        numThreadgroups.height = s.heads;
        numThreadgroups.depth = s.batch;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);

        if (hasState) {
            msgVoid1(encoder, __sel_setComputePipelineState, zPipeline);
            setBufFn(encoder, __sel_setBuffer, bufK, 0UL, 0UL);
            setBufFn(encoder, __sel_setBuffer, bufState, 0UL, 1UL);
            setBytesFn(encoder, __sel_setBytes, (void*)&p, sizeof(struct MpsAttnParams), 2UL);
            unsigned int dim = (unsigned int)s.headDim;
            setBytesFn(encoder, __sel_setBytes, (void*)&dim, 4UL, 3UL);
            unsigned long n = s.heads * s.headDim;
            tgSize.width = 64UL;
            numThreadgroups.width = (n + 63UL) / 64UL;
            numThreadgroups.height = 1UL;
            dispatchFn(encoder, __sel_dispatchThreadgroups, numThreadgroups, tgSize);
        }
        return 1;
    }
}

// Runs 'lin' (0 flash, 1 linear) into the open batch, or in a one-op batch
// that is waited for when none is open.
static int __mps_attn_run(int lin, void* bufQ, void* bufK, void* bufV, void* bufOut, struct MpsAttnShape s,
                          int dtype, struct MpsAttnParams p) {
    int ownBatch = !__mps_batch_active;
    if (ownBatch) {
        mps_batch_begin();
        if (!__mps_batch_active) return 0;
    }
    int ok = lin ? __mps_linear_encode(__mps_batch_encoder, bufQ, bufK, bufV, bufOut, (void*)0, s, dtype, p)
                 : __mps_flash_encode(__mps_batch_encoder, bufQ, bufK, bufV, bufOut, s, dtype, p);
    if (ownBatch) { mps_batch_end(); }
    return ok;
}

int mps_flash_attention_ex(&void q, &void k, &void v, &void out, struct MpsAttnShape s, int dtype,
                           float scale, int causal) {
    if (!__mps_attn_check(s, dtype, causal)) return 0;
    struct MpsAttnParams p = __mps_attn_params_head_major(s, causal);
    p.scale = scale;
    unsafe { return __mps_attn_run(0, (void*)q, (void*)k, (void*)v, (void*)out, s, dtype, p); }
}

int mps_linear_attention_ex(&void q, &void k, &void v, &void out, struct MpsAttnShape s, int dtype, int causal) {
    if (s.window != 0UL || !__mps_attn_check(s, dtype, causal)) return 0;
    struct MpsAttnParams p = __mps_attn_params_head_major(s, causal);
    unsafe { return __mps_attn_run(1, (void*)q, (void*)k, (void*)v, (void*)out, s, dtype, p); }
}

static struct MpsAttnShape __mps_attn_shape_flat(unsigned long heads, unsigned long seqLen, unsigned long kvLen,
                                                 unsigned long headDim) {
    struct MpsAttnShape s;
    s.batch = 1UL; s.heads = heads; s.kvHeads = heads; s.seqLen = seqLen; s.kvLen = kvLen;
    s.headDim = headDim; s.window = 0UL;
    return s;
}

int mps_flash_attention(&void q, &void k, &void v, &void out, unsigned long heads, unsigned long seqLen,
                        unsigned long kvLen, unsigned long headDim, float scale, int causal) {
    return mps_flash_attention_ex(q, k, v, out, __mps_attn_shape_flat(heads, seqLen, kvLen, headDim), 0, scale, causal);
}

int mps_linear_attention(&void q, &void k, &void v, &void out, unsigned long heads, unsigned long seqLen,
                         unsigned long kvLen, unsigned long headDim, int causal) {
    return mps_linear_attention_ex(q, k, v, out, __mps_attn_shape_flat(heads, seqLen, kvLen, headDim), 0, causal);
}

// Host-array forms: upload, run, read back. Inside an open batch, 'out' is
// filled at mps_batch_end() like every other batched op.
static int __mps_attn_host_f32(int lin, const &float q, const &float k, const &float v, &float out,
                               unsigned long heads, unsigned long seqLen, unsigned long kvLen,
                               unsigned long headDim, float scale, int causal) {
    struct MpsAttnShape s = __mps_attn_shape_flat(heads, seqLen, kvLen, headDim);
    if (!__mps_attn_check(s, 0, causal)) return 0;
    struct MpsAttnParams p = __mps_attn_params_head_major(s, causal);
    p.scale = scale;
    unsafe {
        int ownBatch = !__mps_batch_active;
        if (ownBatch) {
            mps_batch_begin();
            if (!__mps_batch_active) return 0;
        }
        unsigned long qBytes = heads * seqLen * headDim * 4UL;
        unsigned long kvBytes = heads * kvLen * headDim * 4UL;
        void* bufQ = __mps_buf_get_with_bytes_batched((const void*)q, qBytes);
        void* bufK = __mps_buf_get_with_bytes_batched((const void*)k, kvBytes);
        void* bufV = __mps_buf_get_with_bytes_batched((const void*)v, kvBytes);
        void* bufOut = __mps_buf_get_batched(qBytes);
        int ok = bufQ != (void*)0 && bufK != (void*)0 && bufV != (void*)0 && bufOut != (void*)0;
        if (ok) {
            ok = lin ? __mps_linear_encode(__mps_batch_encoder, bufQ, bufK, bufV, bufOut, (void*)0, s, 0, p)
                     : __mps_flash_encode(__mps_batch_encoder, bufQ, bufK, bufV, bufOut, s, 0, p);
        }
        if (ok) { __mps_pending_add(bufOut, (void*)out, qBytes); }
        if (ownBatch) { mps_batch_end(); }
        return ok;
    }
}

int mps_flash_attention_f32(const &float q, const &float k, const &float v, &float out,
                            unsigned long heads, unsigned long seqLen, unsigned long kvLen,
                            unsigned long headDim, float scale, int causal) {
    return __mps_attn_host_f32(0, q, k, v, out, heads, seqLen, kvLen, headDim, scale, causal);
}

int mps_linear_attention_f32(const &float q, const &float k, const &float v, &float out,
                             unsigned long heads, unsigned long seqLen, unsigned long kvLen,
                             unsigned long headDim, int causal) {
    return __mps_attn_host_f32(1, q, k, v, out, heads, seqLen, kvLen, headDim, (float)0.0, causal);
}

// ── Token-major bf16 API (flash) ────────────────────────────────────────────
static struct MpsAttnShape __mps_attn_shape_tm(unsigned long seqLen, unsigned long kvLen, unsigned long numHeads,
                                               unsigned long numKvHeads, unsigned long headDim,
                                               unsigned long window) {
    struct MpsAttnShape s;
    s.batch = 1UL; s.heads = numHeads; s.kvHeads = numKvHeads; s.seqLen = seqLen; s.kvLen = kvLen;
    s.headDim = headDim; s.window = window;
    return s;
}

int mps_flash_attn_bf16_chained(&void bufQ, &void bufK, &void bufV, &unsigned short out,
                                  unsigned long seqLen, unsigned long kvLen,
                                  unsigned long numHeads, unsigned long numKvHeads, unsigned long headDim,
                                  float scale, int causal, int isSliding, unsigned long slidingWindow) {
    if (!__mps_batch_active) return 0;
    struct MpsAttnShape s = __mps_attn_shape_tm(seqLen, kvLen, numHeads, numKvHeads, headDim,
                                                isSliding ? slidingWindow : 0UL);
    if (!__mps_attn_check(s, MPS_BF16, causal)) return 0;
    struct MpsAttnParams p = __mps_attn_params_token_major(s, causal);
    p.scale = scale;
    unsafe {
        unsigned long outBytes = seqLen * numHeads * headDim * 2UL;
        void* bufOut = __mps_buf_get_batched(outBytes);
        if (bufOut == (void*)0) return 0;
        if (!__mps_flash_encode(__mps_batch_encoder, bufQ, bufK, bufV, bufOut, s, MPS_BF16, p)) return 0;
        __mps_pending_add(bufOut, (void*)out, outBytes);
        return 1;
    }
}

// Batch-aware like mps_sub_f32: inside an open batch it encodes there and
// 'out' is filled at that batch's mps_batch_end(); otherwise it runs its
// own one-op batch and 'out' is filled on return.
int mps_flash_attn_bf16(const &unsigned short q, const &unsigned short k, const &unsigned short v,
                          &unsigned short out,
                          unsigned long seqLen, unsigned long kvLen,
                          unsigned long numHeads, unsigned long numKvHeads, unsigned long headDim,
                          float scale, int causal, int isSliding, unsigned long slidingWindow) {
    unsafe {
        int ownBatch = !__mps_batch_active;
        if (ownBatch) {
            mps_batch_begin();
            if (!__mps_batch_active) return 0;
        }
        unsigned long qBytes = seqLen * numHeads * headDim * 2UL;
        unsigned long kvBytes = kvLen * numKvHeads * headDim * 2UL;
        void* bufQ = __mps_buf_get_with_bytes_batched((const void*)q, qBytes);
        void* bufK = __mps_buf_get_with_bytes_batched((const void*)k, kvBytes);
        void* bufV = __mps_buf_get_with_bytes_batched((const void*)v, kvBytes);
        int ok = bufQ != (void*)0 && bufK != (void*)0 && bufV != (void*)0 &&
                 mps_flash_attn_bf16_chained(bufQ, bufK, bufV, out, seqLen, kvLen, numHeads, numKvHeads,
                                             headDim, scale, causal, isSliding, slidingWindow);
        if (ownBatch) { mps_batch_end(); }
        return ok;
    }
}

// ── Token-major bf16 API (linear) ───────────────────────────────────────────
unsigned long mps_linear_attn_state_bytes(unsigned long numHeads, unsigned long headDim) {
    return numHeads * (headDim * headDim + headDim) * 4UL;
}

?&void mps_linear_attn_state_new(unsigned long numHeads, unsigned long headDim) {
    unsafe {
        if (!__mps_ensure_device_queue()) return (void*)0;
        unsigned long bytes = mps_linear_attn_state_bytes(numHeads, headDim);
        void* buf = __mps_buf_get(bytes);
        if (buf == (void*)0) return (void*)0;
        MMsg0 msg0 = (MMsg0)objc_msgSend;
        void* ptr = msg0(buf, __sel_contents);
        if (ptr == (void*)0) { __mps_buf_put(buf, bytes); return (void*)0; }
        memset(ptr, 0, bytes);
        return buf;
    }
}

// Encodes token-major linear attention onto the open batch; 'bufState'
// NULL means a zero start state that is discarded.
static int __mps_linear_tm_encode(void* bufQ, void* bufK, void* bufV, void* bufState, void* out,
                                  unsigned long seqLen, unsigned long kvLen,
                                  unsigned long numHeads, unsigned long numKvHeads, unsigned long headDim,
                                  int causal) {
    struct MpsAttnShape s = __mps_attn_shape_tm(seqLen, kvLen, numHeads, numKvHeads, headDim, 0UL);
    if (!__mps_attn_check(s, MPS_BF16, causal)) return 0;
    struct MpsAttnParams p = __mps_attn_params_token_major(s, causal);
    unsafe {
        unsigned long outBytes = seqLen * numHeads * headDim * 2UL;
        void* bufOut = __mps_buf_get_batched(outBytes);
        if (bufOut == (void*)0) return 0;
        if (!__mps_linear_encode(__mps_batch_encoder, bufQ, bufK, bufV, bufOut, bufState, s, MPS_BF16, p)) return 0;
        __mps_pending_add(bufOut, out, outBytes);
        return 1;
    }
}

int mps_linear_attn_bf16_chained(&void bufQ, &void bufK, &void bufV, &void bufState, &unsigned short out,
                                   unsigned long seqLen,
                                   unsigned long numHeads, unsigned long numKvHeads, unsigned long headDim) {
    if (!__mps_batch_active) return 0;
    unsafe {
        return __mps_linear_tm_encode(bufQ, bufK, bufV, bufState, (void*)out,
                                      seqLen, seqLen, numHeads, numKvHeads, headDim, 1);
    }
}

// Batch-aware like mps_flash_attn_bf16 above.
int mps_linear_attn_bf16(const &unsigned short q, const &unsigned short k, const &unsigned short v,
                           &unsigned short out,
                           unsigned long seqLen, unsigned long kvLen,
                           unsigned long numHeads, unsigned long numKvHeads, unsigned long headDim,
                           int causal) {
    if (causal && kvLen != seqLen) return 0;
    unsafe {
        int ownBatch = !__mps_batch_active;
        if (ownBatch) {
            mps_batch_begin();
            if (!__mps_batch_active) return 0;
        }
        unsigned long qBytes = seqLen * numHeads * headDim * 2UL;
        unsigned long kvBytes = kvLen * numKvHeads * headDim * 2UL;
        void* bufQ = __mps_buf_get_with_bytes_batched((const void*)q, qBytes);
        void* bufK = __mps_buf_get_with_bytes_batched((const void*)k, kvBytes);
        void* bufV = __mps_buf_get_with_bytes_batched((const void*)v, kvBytes);
        int ok = bufQ != (void*)0 && bufK != (void*)0 && bufV != (void*)0 &&
                 __mps_linear_tm_encode(bufQ, bufK, bufV, (void*)0, (void*)out,
                                        seqLen, kvLen, numHeads, numKvHeads, headDim, causal);
        if (ownBatch) { mps_batch_end(); }
        return ok;
    }
}

} // namespace std
