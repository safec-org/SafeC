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
#pragma once
#include <std/ml/gpu_mps.h>
#include <std/mem.sc>

namespace std {

extern void* objc_getClass(const char* name);
extern void* sel_registerName(const char* name);
extern void* objc_msgSend(void* recv, void* op);
// The real Metal framework entry point — a plain C function (not a
// message send): returns the default MTLDevice, or NULL if this machine
// has no Metal-capable GPU.
extern void* MTLCreateSystemDefaultDevice();

struct MTLSize {
    unsigned long width;
    unsigned long height;
    unsigned long depth;
};

typedef fn void*(void*, void*) MMsg0;
typedef fn void*(void*, void*, void*) MMsg1;
typedef fn void*(void*, void*, void*, void*) MMsg2;
typedef fn void*(void*, void*, void*, void*, void*) MMsg3Lib; // source, options, error
typedef fn void*(void*, void*, const void*, unsigned long, unsigned long) MMsgNewBufferBytes;
typedef fn void*(void*, void*, unsigned long, unsigned long) MMsgNewBufferLen;
typedef fn void(void*, void*, void*) MMsgVoid1;
typedef fn void(void*, void*, void*, unsigned long, unsigned long) MMsgSetBuffer;
typedef fn void(void*, void*, struct MTLSize, struct MTLSize) MMsgDispatch;
typedef fn void(void*, void*) MMsgVoid0;

static void* __mps_sel(const char* name) {
    unsafe { return sel_registerName(name); }
}

static void* __mps_nsstring(const char* cstr) {
    unsafe {
        void* cls = objc_getClass("NSString");
        MMsg0 allocFn = (MMsg0)objc_msgSend;
        void* obj = allocFn(cls, __mps_sel("alloc"));
        MMsg1 initFn = (MMsg1)objc_msgSend;
        return initFn(obj, __mps_sel("initWithUTF8String:"), (void*)cstr);
    }
}

int mps_available() {
    void* dev;
    unsafe { dev = MTLCreateSystemDefaultDevice(); }
    return dev != (void*)0;
}

int mps_add_f32(const float* a, const float* b, float* out, unsigned long n) {
    // A single-buffer elementwise-add kernel — deliberately the simplest
    // possible real GPU-executed op (compiles, dispatches, and reads back
    // through the shared-storage 'contents' pointer) rather than a large
    // kernel library, since the point here is proving the whole
    // SafeC -> objc_msgSend -> Metal pipeline works end to end, not
    // building out a full shader library. gpu_mps.h's header comment
    // covers the unified-memory story this demonstrates. Local (not a
    // file-scope global) because SafeC's global-initializer constant
    // folder doesn't fold adjacent-string-literal concatenation.
    const char* kernelSrc =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void add_kernel(device const float* a [[buffer(0)]],\n"
        "                        device const float* b [[buffer(1)]],\n"
        "                        device float* out [[buffer(2)]],\n"
        "                        uint id [[thread_position_in_grid]]) {\n"
        "    out[id] = a[id] + b[id];\n"
        "}\n";

    unsafe {
        void* device = MTLCreateSystemDefaultDevice();
        if (device == (void*)0) return 0;
        MMsg0 msg0 = (MMsg0)objc_msgSend;
        void* queue = msg0(device, __mps_sel("newCommandQueue"));
        if (queue == (void*)0) return 0;

        void* srcStr = __mps_nsstring(kernelSrc);
        void* errPtr = (void*)0;
        MMsg3Lib newLibFn = (MMsg3Lib)objc_msgSend;
        void* library = newLibFn(device, __mps_sel("newLibraryWithSource:options:error:"),
                                  srcStr, (void*)0, (void*)&errPtr);
        if (library == (void*)0) return 0;

        void* fnNameStr = __mps_nsstring("add_kernel");
        MMsg1 msg1 = (MMsg1)objc_msgSend;
        void* kernelFn = msg1(library, __mps_sel("newFunctionWithName:"), fnNameStr);
        if (kernelFn == (void*)0) return 0;

        MMsg2 newPipelineFn = (MMsg2)objc_msgSend;
        void* pipeline = newPipelineFn(device, __mps_sel("newComputePipelineStateWithFunction:error:"),
                                        kernelFn, (void*)&errPtr);
        if (pipeline == (void*)0) return 0;

        unsigned long bytes = n * sizeof(float);
        // storage mode 0 = MTLResourceStorageModeShared — the unified-
        // memory mode: 'contents' below is directly CPU-readable with no
        // explicit sync step needed after waitUntilCompleted.
        MMsgNewBufferBytes newBufBytesFn = (MMsgNewBufferBytes)objc_msgSend;
        void* bufA = newBufBytesFn(device, __mps_sel("newBufferWithBytes:length:options:"),
                                    (const void*)a, bytes, 0UL);
        void* bufB = newBufBytesFn(device, __mps_sel("newBufferWithBytes:length:options:"),
                                    (const void*)b, bytes, 0UL);
        MMsgNewBufferLen newBufLenFn = (MMsgNewBufferLen)objc_msgSend;
        void* bufOut = newBufLenFn(device, __mps_sel("newBufferWithLength:options:"), bytes, 0UL);
        if (bufA == (void*)0 || bufB == (void*)0 || bufOut == (void*)0) return 0;

        void* cmdBuf = msg0(queue, __mps_sel("commandBuffer"));
        void* encoder = msg0(cmdBuf, __mps_sel("computeCommandEncoder"));

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __mps_sel("setComputePipelineState:"), pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __mps_sel("setBuffer:offset:atIndex:"), bufA, 0UL, 0UL);
        setBufFn(encoder, __mps_sel("setBuffer:offset:atIndex:"), bufB, 0UL, 1UL);
        setBufFn(encoder, __mps_sel("setBuffer:offset:atIndex:"), bufOut, 0UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = n; gridSize.height = 1UL; gridSize.depth = 1UL;
        unsigned long tgWidth = n < 64UL ? n : 64UL;
        if (tgWidth == 0UL) tgWidth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = tgWidth; tgSize.height = 1UL; tgSize.depth = 1UL;
        unsigned long numGroups = (n + tgWidth - 1UL) / tgWidth;
        struct MTLSize numThreadgroups;
        numThreadgroups.width = numGroups; numThreadgroups.height = 1UL; numThreadgroups.depth = 1UL;
        // KNOWN ISSUE, confirmed on real Apple Silicon hardware: this
        // specific call — the one selector here passing *two* struct
        // arguments (each 'struct MTLSize', 3x unsigned long = 24 bytes)
        // by value through an objc_msgSend cast — segfaults. Every call
        // above it (device/queue creation, a real runtime MSL shader
        // *compile*, pipeline-state creation, buffer allocation, encoder
        // creation, setComputePipelineState:/setBuffer:offset:atIndex:,
        // each passing at most *one* non-scalar or scalar argument) was
        // individually verified working via debug-instrumented bisection.
        // std/gui/gui_cocoa.sc's single-struct-by-value case (NSRect, 4
        // doubles) works fine, so this looks like a SafeC codegen gap
        // specific to *multiple consecutive* struct-by-value objc_msgSend
        // arguments (ARM64 AAPCS64 stack-argument layout for back-to-back
        // aggregates > 16 bytes each) — not yet root-caused or fixed.
        // mps_available() above this function is fully verified; this
        // function is not yet safe to call outside of further debugging.
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __mps_sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                   numThreadgroups, tgSize);
        MMsgVoid0 msgVoid0 = (MMsgVoid0)objc_msgSend;
        msgVoid0(encoder, __mps_sel("endEncoding"));
        msgVoid0(cmdBuf, __mps_sel("commit"));
        msgVoid0(cmdBuf, __mps_sel("waitUntilCompleted"));

        void* outPtr = msg0(bufOut, __mps_sel("contents"));
        if (outPtr == (void*)0) return 0;
        memcpy((void*)out, outPtr, bytes);
        return 1;
    }
}

} // namespace std
