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
// Two things this file used to do at runtime, on every process, that are
// now done once, offline, at build time instead:
//   1. Kernel compilation. Every mps_*_f32 call used to hand a Metal
//      Shading Language *source string* to newLibraryWithSource:options:
//      error:, a real MSL-to-AIR compile, the most expensive single step
//      in the whole dispatch (cached after the first call per op, but
//      still paid once per process per op). gpu_mps_kernels.metal is now
//      compiled ahead of time by gen_mps_metallib.sh into a .metallib,
//      embedded as a byte array (gpu_mps_metallib.h), and loaded directly
//      via newLibraryWithData:error: — no MSL compiler runs in-process at
//      all anymore.
//   2. Selector lookup. Every objc_msgSend call used to re-resolve its
//      selector via sel_registerName(name) — a string lookup — on every
//      single dispatch. The full set this file ever sends is fixed and
//      known ahead of time, so they're resolved once (in
//      __mps_init_selectors, called from __mps_ensure_device_queue) into
//      static SEL variables instead.
#pragma once
#include <std/ml/gpu_mps.h>
#include <std/ml/gpu_mps_metallib.h>
#include <std/mem.h>

namespace std {

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
typedef fn void*(void*, void*, const void*, unsigned long, unsigned long) MMsgNewBufferBytes;
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
static void* __sel_newBufferWithBytes = (void*)0;
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

static void __mps_init_selectors() {
    unsafe {
        __sel_alloc = sel_registerName("alloc");
        __sel_initWithUTF8String = sel_registerName("initWithUTF8String:");
        __sel_newCommandQueue = sel_registerName("newCommandQueue");
        __sel_newLibraryWithData = sel_registerName("newLibraryWithData:error:");
        __sel_newFunctionWithName = sel_registerName("newFunctionWithName:");
        __sel_newComputePipelineState = sel_registerName("newComputePipelineStateWithFunction:error:");
        __sel_newBufferWithBytes = sel_registerName("newBufferWithBytes:length:options:");
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
        // storage mode 0 = MTLResourceStorageModeShared — the unified-
        // memory mode: 'contents' below is directly CPU-readable with no
        // explicit sync step needed after waitUntilCompleted.
        MMsgNewBufferBytes newBufBytesFn = (MMsgNewBufferBytes)objc_msgSend;
        void* bufA = newBufBytesFn(device, __sel_newBufferWithBytes, (const void*)a, bytes, 0UL);
        void* bufB = newBufBytesFn(device, __sel_newBufferWithBytes, (const void*)b, bytes, 0UL);
        MMsgNewBufferLen newBufLenFn = (MMsgNewBufferLen)objc_msgSend;
        void* bufOut = newBufLenFn(device, __sel_newBufferWithLength, bytes, 0UL);
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
        MMsgNewBufferBytes newBufBytesFn = (MMsgNewBufferBytes)objc_msgSend;
        void* bufA = newBufBytesFn(device, __sel_newBufferWithBytes, (const void*)a, bytes, 0UL);
        MMsgNewBufferLen newBufLenFn = (MMsgNewBufferLen)objc_msgSend;
        void* bufOut = newBufLenFn(device, __sel_newBufferWithLength, bytes, 0UL);
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

int mps_add_f32(const float* a, const float* b, float* out, unsigned long n) {
    return __mps_run_binary_kernel(&__mps_add_pipeline, "add_kernel", a, b, out, n);
}

int mps_sub_f32(const float* a, const float* b, float* out, unsigned long n) {
    return __mps_run_binary_kernel(&__mps_sub_pipeline, "sub_kernel", a, b, out, n);
}

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
        MMsgNewBufferBytes newBufBytesFn = (MMsgNewBufferBytes)objc_msgSend;
        void* bufA = newBufBytesFn(device, __sel_newBufferWithBytes, (const void*)a, bytes, 0UL);
        MMsgNewBufferLen newBufLenFn = (MMsgNewBufferLen)objc_msgSend;
        void* bufOut = newBufLenFn(device, __sel_newBufferWithLength, bytes, 0UL);
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
        return 1;
    }
}

// out[0] = sum(a[0..n)) -- a single GPU thread walks the whole array and
// accumulates serially. Correct, and enough to prove the reduction-shaped
// dispatch works end to end, but not a real parallel reduction (no
// threadgroup-memory tree, no multiple-threadgroups-then-combine pass) --
// same "simplest real version first" spirit as mps_matmul_f32's lack of
// tiling. A single GPU thread doing a serial O(n) sum is, unsurprisingly,
// not where GPU dispatch is going to win; this exists for completeness of
// the op set (tensor_sum is used for the loss in the training benchmark)
// more than as a performance claim.
static void* __mps_sum_pipeline = (void*)0;

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
        MMsgNewBufferBytes newBufBytesFn = (MMsgNewBufferBytes)objc_msgSend;
        void* bufA = newBufBytesFn(device, __sel_newBufferWithBytes, (const void*)a, bytes, 0UL);
        MMsgNewBufferLen newBufLenFn = (MMsgNewBufferLen)objc_msgSend;
        void* bufOut = newBufLenFn(device, __sel_newBufferWithLength, 4UL, 0UL);
        if (bufA == (void*)0 || bufOut == (void*)0) return 0;

        void* cmdBuf = msg0(queue, __sel_commandBuffer);
        void* encoder = msg0(cmdBuf, __sel_computeCommandEncoder);

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __sel_setComputePipelineState, pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(encoder, __sel_setBuffer, bufOut, 0UL, 1UL);
        unsigned int nu = (unsigned int)n;
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(encoder, __sel_setBytes, (void*)&nu, 4UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = 1UL; gridSize.height = 1UL; gridSize.depth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = 1UL; tgSize.height = 1UL; tgSize.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __sel_dispatchThreadgroups, gridSize, tgSize);
        MMsgVoid0 msgVoid0 = (MMsgVoid0)objc_msgSend;
        msgVoid0(encoder, __sel_endEncoding);
        msgVoid0(cmdBuf, __sel_commit);
        msgVoid0(cmdBuf, __sel_waitUntilCompleted);

        void* outPtr = msg0(bufOut, __sel_contents);
        if (outPtr == (void*)0) return 0;
        memcpy((void*)out, outPtr, 4UL);
        return 1;
    }
}

// ── Batched dispatch (see gpu_mps.h's mps_batch_begin comment for the
// full contract) ───────────────────────────────────────────────────────────
// __mps_batch_active/_cmdBuf/_encoder track one in-flight shared command
// buffer. __mps_pending_* is a fixed-size array of "GPU buffer -> CPU
// pointer" raw-readback entries collected while a batch is open — sized
// generously for any realistic single training step's op count (a real
// growable Vec would work too, but this file doesn't otherwise depend on
// std/collections, and a bounds-checked fixed array covers every actual
// use here without adding that dependency). __mps_finalize_* is the
// parallel list of caller-supplied post-readback callbacks (e.g.
// tensor_gpu.sc's float32->float64 conversion).
#define MPS_BATCH_MAX_PENDING 64
#define MPS_BATCH_MAX_FINALIZE 64

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

int mps_batch_is_active() { return __mps_batch_active; }

void mps_batch_begin() {
    unsafe {
        if (!__mps_ensure_device_queue()) return;
        MMsg0 msg0 = (MMsg0)objc_msgSend;
        __mps_batch_cmdBuf  = msg0(__mps_queue, __sel_commandBuffer);
        __mps_batch_encoder = msg0(__mps_batch_cmdBuf, __sel_computeCommandEncoder);
        __mps_pending_count = 0UL;
        __mps_finalize_count = 0UL;
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
static void* __mps_relu_pipeline = (void*)0;

int mps_matmul_f32(const float* a, const float* b, float* out,
                    unsigned long M, unsigned long K, unsigned long N) {
    unsafe {
        if (!__mps_ensure_device_queue()) return 0;
        void* device = __mps_device;
        void* queue = __mps_queue;
        MMsg0 msg0 = (MMsg0)objc_msgSend;

        if (__mps_matmul_pipeline == (void*)0) {
            void* pipeline = __mps_make_pipeline("matmul_kernel");
            if (pipeline == (void*)0) return 0;
            __mps_matmul_pipeline = pipeline;
        }
        void* pipeline = __mps_matmul_pipeline;

        unsigned long bytesA = M * K * sizeof(float);
        unsigned long bytesB = K * N * sizeof(float);
        unsigned long bytesOut = M * N * sizeof(float);
        MMsgNewBufferBytes newBufBytesFn = (MMsgNewBufferBytes)objc_msgSend;
        void* bufA = newBufBytesFn(device, __sel_newBufferWithBytes, (const void*)a, bytesA, 0UL);
        void* bufB = newBufBytesFn(device, __sel_newBufferWithBytes, (const void*)b, bytesB, 0UL);
        MMsgNewBufferLen newBufLenFn = (MMsgNewBufferLen)objc_msgSend;
        void* bufOut = newBufLenFn(device, __sel_newBufferWithLength, bytesOut, 0UL);
        if (bufA == (void*)0 || bufB == (void*)0 || bufOut == (void*)0) return 0;

        unsigned int Mu = (unsigned int)M;
        unsigned int Ku = (unsigned int)K;
        unsigned int Nu = (unsigned int)N;

        // Batched: reuse the shared command buffer/encoder that
        // mps_batch_begin() already opened instead of creating our own
        // (and don't end/commit/wait/readback here — mps_batch_end()
        // does all of that once, for every op encoded since begin()).
        int batched = __mps_batch_active;
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

        // 2D dispatch: one thread per output element, 16x16 threadgroups
        // (a conventional default for a 2D elementwise-per-output kernel;
        // no tiling/threadgroup-memory reuse here to tune against).
        struct MTLSize gridSize;
        gridSize.width = N; gridSize.height = M; gridSize.depth = 1UL;
        unsigned long tgW = N < 16UL ? N : 16UL;
        unsigned long tgH = M < 16UL ? M : 16UL;
        if (tgW == 0UL) tgW = 1UL;
        if (tgH == 0UL) tgH = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgW; tgSize.height = tgH; tgSize.depth = 1UL;
        unsigned long groupsX = (N + tgW - 1UL) / tgW;
        unsigned long groupsY = (M + tgH - 1UL) / tgH;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = groupsX; numThreadgroups.height = groupsY; numThreadgroups.depth = 1UL;
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
        MMsgNewBufferBytes newBufBytesFn = (MMsgNewBufferBytes)objc_msgSend;
        void* bufA = newBufBytesFn(device, __sel_newBufferWithBytes, (const void*)a, bytes, 0UL);
        MMsgNewBufferLen newBufLenFn = (MMsgNewBufferLen)objc_msgSend;
        void* bufOut = newBufLenFn(device, __sel_newBufferWithLength, bytes, 0UL);
        if (bufA == (void*)0 || bufOut == (void*)0) return 0;

        int batched = __mps_batch_active;
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
        return 1;
    }
}

// ── GPU-buffer-chained variants (see gpu_mps.h's own comment on these) ───────
// Only meaningful while a batch is active: 'bufA' is used directly as
// buffer(0) instead of uploading from a CPU float array, so the dispatch
// this encodes reads whatever a PREVIOUS dispatch on the same command
// buffer already wrote there -- correct without any CPU round trip
// specifically because Metal executes dispatches encoded on the same
// encoder in encoding order.
int mps_matmul_f32_chained(void* bufA, const float* b, float* out,
                            unsigned long M, unsigned long K, unsigned long N) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (__mps_matmul_pipeline == (void*)0) {
            void* pipeline = __mps_make_pipeline("matmul_kernel");
            if (pipeline == (void*)0) return 0;
            __mps_matmul_pipeline = pipeline;
        }

        unsigned long bytesB = K * N * sizeof(float);
        unsigned long bytesOut = M * N * sizeof(float);
        MMsgNewBufferBytes newBufBytesFn = (MMsgNewBufferBytes)objc_msgSend;
        void* bufB = newBufBytesFn(__mps_device, __sel_newBufferWithBytes, (const void*)b, bytesB, 0UL);
        MMsgNewBufferLen newBufLenFn = (MMsgNewBufferLen)objc_msgSend;
        void* bufOut = newBufLenFn(__mps_device, __sel_newBufferWithLength, bytesOut, 0UL);
        if (bufB == (void*)0 || bufOut == (void*)0) return 0;

        unsigned int Mu = (unsigned int)M;
        unsigned int Ku = (unsigned int)K;
        unsigned int Nu = (unsigned int)N;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __sel_setComputePipelineState, __mps_matmul_pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufA, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufB, 0UL, 1UL);
        setBufFn(__mps_batch_encoder, __sel_setBuffer, bufOut, 0UL, 2UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Mu, 4UL, 3UL);
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Ku, 4UL, 4UL);
        setBytesFn(__mps_batch_encoder, __sel_setBytes, (void*)&Nu, 4UL, 5UL);

        struct MTLSize gridSize;
        gridSize.width = N; gridSize.height = M; gridSize.depth = 1UL;
        unsigned long tgW = N < 16UL ? N : 16UL;
        unsigned long tgH = M < 16UL ? M : 16UL;
        if (tgW == 0UL) tgW = 1UL;
        if (tgH == 0UL) tgH = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgW; tgSize.height = tgH; tgSize.depth = 1UL;
        unsigned long groupsX = (N + tgW - 1UL) / tgW;
        unsigned long groupsY = (M + tgH - 1UL) / tgH;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = groupsX; numThreadgroups.height = groupsY; numThreadgroups.depth = 1UL;
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

int mps_relu_f32_chained(void* bufA, float* out, unsigned long n) {
    if (!__mps_batch_active) return 0;
    unsafe {
        if (__mps_relu_pipeline == (void*)0) {
            void* pipeline = __mps_make_pipeline("relu_kernel");
            if (pipeline == (void*)0) return 0;
            __mps_relu_pipeline = pipeline;
        }

        unsigned long bytes = n * sizeof(float);
        MMsgNewBufferLen newBufLenFn = (MMsgNewBufferLen)objc_msgSend;
        void* bufOut = newBufLenFn(__mps_device, __sel_newBufferWithLength, bytes, 0UL);
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

} // namespace std
