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
#include <std/ml/gpu_mps.h>
#include <std/ml/gpu_mps_metallib.h>
#include <std/ml/float16.h>
#include <std/mem.h>

namespace std {

extern void* malloc(unsigned long size);
extern void  free(void* ptr);

extern void* objc_getClass(const char* name);
extern void* sel_registerName(const char* name);
extern void* objc_msgSend(void* recv, void* op);
extern void* memcpy(void* dst, const void* src, unsigned long n);
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

int mps_add_f32(const float* a, const float* b, float* out, unsigned long n) {
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

int mps_mul_f32(const float* a, const float* b, float* out, unsigned long n) {
    return __mps_run_binary_kernel(&__mps_mul_pipeline, "mul_kernel", a, b, out, n);
}

int mps_div_f32(const float* a, const float* b, float* out, unsigned long n) {
    return __mps_run_binary_kernel(&__mps_div_pipeline, "div_kernel", a, b, out, n);
}

int mps_pow_f32(const float* a, const float* b, float* out, unsigned long n) {
    return __mps_run_binary_kernel(&__mps_pow_pipeline, "pow_kernel", a, b, out, n);
}

int mps_log_f32(const float* a, float* out, unsigned long n) {
    return __mps_run_unary_kernel(&__mps_log_pipeline, "log_kernel", a, out, n);
}

int mps_exp_f32(const float* a, float* out, unsigned long n) {
    return __mps_run_unary_kernel(&__mps_exp_pipeline, "exp_kernel", a, out, n);
}

int mps_sqrt_f32(const float* a, float* out, unsigned long n) {
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

int mps_scale_f32(const float* a, float k, float* out, unsigned long n) {
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

void mps_batch_register_finalize(MpsFinalizeFn finalizeFn, void* ctx) {
    if (__mps_finalize_count >= (unsigned long)MPS_BATCH_MAX_FINALIZE) return;
    __mps_finalize_fn[__mps_finalize_count] = finalizeFn;
    __mps_finalize_ctx[__mps_finalize_count] = ctx;
    __mps_finalize_count = __mps_finalize_count + 1UL;
}

void* mps_batch_last_output_buffer() {
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
int mps_sub_f32(const float* a, const float* b, float* out, unsigned long n) {
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
        dealloc((void*)ctx->partials);
        dealloc((void*)ctx);
    }
}

int mps_sum_f32(const float* a, float* out, unsigned long n) {
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
            float* partials = (float*)alloc(sizeof(float) * numGroups);
            if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
                __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
                __mps_pending_cpuOut[__mps_pending_count] = (void*)partials;
                __mps_pending_bytes[__mps_pending_count] = partialBytes;
                __mps_pending_count = __mps_pending_count + 1UL;
            }
            struct __MpsSumFinalizeCtx* ctx = (struct __MpsSumFinalizeCtx*)alloc(sizeof(struct __MpsSumFinalizeCtx));
            ctx->partials = partials; ctx->out = out; ctx->numGroups = numGroups;
            mps_batch_register_finalize((MpsFinalizeFn)__mps_sum_finalize, (void*)ctx);
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

int mps_sum_f32_chained(void* bufA, float* out, unsigned long n) {
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

        float* partials = (float*)alloc(sizeof(float) * numGroups);
        if (__mps_pending_count < (unsigned long)MPS_BATCH_MAX_PENDING) {
            __mps_pending_gpuBuf[__mps_pending_count] = bufOut;
            __mps_pending_cpuOut[__mps_pending_count] = (void*)partials;
            __mps_pending_bytes[__mps_pending_count] = partialBytes;
            __mps_pending_count = __mps_pending_count + 1UL;
        }
        struct __MpsSumFinalizeCtx* ctx = (struct __MpsSumFinalizeCtx*)alloc(sizeof(struct __MpsSumFinalizeCtx));
        ctx->partials = partials; ctx->out = out; ctx->numGroups = numGroups;
        mps_batch_register_finalize((MpsFinalizeFn)__mps_sum_finalize, (void*)ctx);
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

int mps_matmul_f32(const float* a, const float* b, float* out,
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

int mps_matmul_abt_f32(const float* a, const float* b, float* out,
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

int mps_matmul_atb_f32(const float* a, const float* b, float* out,
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
int mps_relu_backward_f32(const float* a, const float* selfGrad, float* out, unsigned long n) {
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
int mps_relu_f32(const float* a, float* out, unsigned long n) {
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
int mps_matmul_f32_chained(void* bufA, const float* b, float* out,
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
int mps_matmul_f32_persistent(void* bufA, void* bufB, float* out,
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
void* mps_upload_persistent(const float* data, unsigned long bytes) {
    if (!__mps_ensure_device_queue()) return (void*)0;
    return __mps_buf_get_with_bytes((const void*)data, bytes);
}

// Returns a buffer obtained from mps_upload_persistent to the size-class
// pool (see __mps_buf_put's own comment) -- call once, after every batch
// that might still be reading it has completed.
void mps_release_persistent(void* buf, unsigned long bytes) {
    __mps_buf_put(buf, bytes);
}

int mps_relu_f32_chained(void* bufA, float* out, unsigned long n) {
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
int mps_sub_f32_chained(void* bufA, const float* b, float* out, unsigned long n) {
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
int mps_mul_f32_chained(void* bufA, const float* b, float* out, unsigned long n) {
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
int mps_square_f32_chained(void* buf, float* out, unsigned long n) {
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
int mps_scale_f32_chained(void* buf, float k, float* out, unsigned long n) {
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

// ── Backward chained variants (see gpu_mps.h's own comment on these) ─────────
// Same "read a previous batched op's still-pending GPU buffer directly,
// no CPU round trip" idea as mps_matmul_f32_chained/mps_relu_f32_chained
// above, applied to the backward chain instead of the forward one.
int mps_matmul_abt_f32_chained(void* bufDC, const float* b, float* out,
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
int mps_matmul_abt_f32_persistent(void* bufDC, void* bufB, float* out,
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

int mps_matmul_atb_f32_chained(const float* a, void* bufDC, float* out,
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
int mps_matmul_atb_f32_chained_ab(void* bufA, void* bufDC, float* out,
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

int mps_relu_backward_f32_chained(const float* a, void* bufSelfGrad, float* out, unsigned long n) {
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
int mps_relu_backward_f32_chained_ab(void* bufA, void* bufSelfGrad, float* out, unsigned long n) {
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

int mps_sgd_update_f32_chained(void* bufW, void* bufGrad, float lr, unsigned long n) {
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

int mps_matmul_f16(const float* a, const float* b, float* out,
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

int mps_matmul_bf16(const float* a, const float* b, float* out,
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

int mps_relu_f16(const float* a, float* out, unsigned long n) {
    return __mps_relu_reduced_precision("relu_kernel_f16", &__mps_relu_f16_pipeline, a, out, n,
                                         f32_to_fp16_bulk, fp16_to_f32_bulk);
}

int mps_relu_bf16(const float* a, float* out, unsigned long n) {
    return __mps_relu_reduced_precision("relu_kernel_bf16", &__mps_relu_bf16_pipeline, a, out, n,
                                         f32_to_bf16_bulk, bf16_to_f32_bulk);
}

// ── fp16 / bf16 persistent-buffer / batched tier (see gpu_mps.h) ────────────
void* mps_upload_f16_persistent(const float* data, unsigned long n) {
    unsafe {
        unsigned short* tmp = (unsigned short*)malloc(n * 2UL);
        if (tmp == (void*)0) return (void*)0;
        f32_to_fp16_bulk(data, tmp, n);
        void* buf = mps_upload_persistent((const float*)tmp, n * 2UL);
        free((void*)tmp);
        return buf;
    }
}

void* mps_upload_bf16_persistent(const float* data, unsigned long n) {
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
int mps_matmul_f16_persistent(void* bufA, void* bufB, unsigned short* out,
                               unsigned long M, unsigned long K, unsigned long N) {
    return __mps_matmul3_chained("matmul_kernel_f16", &__mps_matmul_f16_persistent_pipeline,
                                  bufA, bufB, out, (unsigned int)M, (unsigned int)K, (unsigned int)N, M, N);
}

static void* __mps_matmul_bf16_persistent_pipeline = (void*)0;
int mps_matmul_bf16_persistent(void* bufA, void* bufB, unsigned short* out,
                                unsigned long M, unsigned long K, unsigned long N) {
    return __mps_matmul3_chained("matmul_kernel_bf16", &__mps_matmul_bf16_persistent_pipeline,
                                  bufA, bufB, out, (unsigned int)M, (unsigned int)K, (unsigned int)N, M, N);
}

static void* __mps_matmul_atb_f16_pipeline = (void*)0;
int mps_matmul_atb_f16_chained_ab(void* bufA, void* bufDC, unsigned short* out,
                                   unsigned long M, unsigned long K, unsigned long N) {
    return __mps_matmul3_chained("matmul_atb_kernel_f16", &__mps_matmul_atb_f16_pipeline,
                                  bufA, bufDC, out, (unsigned int)M, (unsigned int)K, (unsigned int)N, K, N);
}

static void* __mps_matmul_atb_bf16_pipeline = (void*)0;
int mps_matmul_atb_bf16_chained_ab(void* bufA, void* bufDC, unsigned short* out,
                                    unsigned long M, unsigned long K, unsigned long N) {
    return __mps_matmul3_chained("matmul_atb_kernel_bf16", &__mps_matmul_atb_bf16_pipeline,
                                  bufA, bufDC, out, (unsigned int)M, (unsigned int)K, (unsigned int)N, K, N);
}

static void* __mps_matmul_abt_f16_pipeline = (void*)0;
int mps_matmul_abt_f16_persistent(void* bufDC, void* bufB, unsigned short* out,
                                   unsigned long M, unsigned long N, unsigned long K) {
    return __mps_matmul3_chained("matmul_abt_kernel_f16", &__mps_matmul_abt_f16_pipeline,
                                  bufDC, bufB, out, (unsigned int)M, (unsigned int)N, (unsigned int)K, M, K);
}

static void* __mps_matmul_abt_bf16_pipeline = (void*)0;
int mps_matmul_abt_bf16_persistent(void* bufDC, void* bufB, unsigned short* out,
                                    unsigned long M, unsigned long N, unsigned long K) {
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
int mps_relu_f16_chained(void* bufA, unsigned short* out, unsigned long n) {
    return __mps_unary_chained("relu_kernel_f16", &__mps_relu_f16_chained_pipeline, bufA, out, n);
}

static void* __mps_relu_bf16_chained_pipeline = (void*)0;
int mps_relu_bf16_chained(void* bufA, unsigned short* out, unsigned long n) {
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
int mps_relu_backward_f16_chained_ab(void* bufA, void* bufSelfGrad, unsigned short* out, unsigned long n) {
    return __mps_binary_dev_chained("relu_backward_kernel_f16", &__mps_relu_backward_f16_pipeline, bufA, bufSelfGrad, out, n);
}

static void* __mps_relu_backward_bf16_pipeline = (void*)0;
int mps_relu_backward_bf16_chained_ab(void* bufA, void* bufSelfGrad, unsigned short* out, unsigned long n) {
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
int mps_sub_f16_chained(void* bufA, const unsigned short* bHalf, unsigned short* out, unsigned long n) {
    return __mps_sub_chained("sub_kernel_f16", &__mps_sub_f16_pipeline, bufA, bHalf, out, n);
}

static void* __mps_sub_bf16_pipeline = (void*)0;
int mps_sub_bf16_chained(void* bufA, const unsigned short* bBf, unsigned short* out, unsigned long n) {
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
int mps_scale_f16_chained(void* buf, float k, unsigned short* out, unsigned long n) {
    return __mps_scale_chained("scale_kernel_f16", &__mps_scale_f16_pipeline, buf, k, out, n);
}

static void* __mps_scale_bf16_pipeline = (void*)0;
int mps_scale_bf16_chained(void* buf, float k, unsigned short* out, unsigned long n) {
    return __mps_scale_chained("scale_kernel_bf16", &__mps_scale_bf16_pipeline, buf, k, out, n);
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
int mps_sgd_update_f16_chained(void* bufW, void* bufGrad, float lr, unsigned long n) {
    return __mps_sgd_chained("sgd_update_kernel_f16", &__mps_sgd_update_f16_pipeline, bufW, bufGrad, lr, n);
}

static void* __mps_sgd_update_bf16_pipeline = (void*)0;
int mps_sgd_update_bf16_chained(void* bufW, void* bufGrad, float lr, unsigned long n) {
    return __mps_sgd_chained("sgd_update_kernel_bf16", &__mps_sgd_update_bf16_pipeline, bufW, bufGrad, lr, n);
}

} // namespace std
