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

// Shared setup/dispatch/readback for every elementwise binary kernel below
// (add/sub/mul/div/pow) — the only thing that varies between them is the
// one-line kernel body, so that's the only thing each op passes in.
// Previously, the dispatchThreadgroups:threadsPerThreadgroup: call inside
// here segfaulted on real Apple Silicon hardware: it passes *two* struct
// arguments (each 'struct MTLSize', 3x unsigned long = 24 bytes) by value
// through an objc_msgSend cast. Root-caused and fixed in CodeGen.cpp's
// genCall: indirect calls (through a function-pointer cast, exactly this
// pattern) now lower non-HFA struct-by-value arguments over 16 bytes to the
// real AAPCS64 form (caller copies to a stack temporary, passes a plain
// pointer) instead of a raw aggregate value — the raw-value form is self-
// consistent for SafeC-to-SafeC calls (which is why every earlier single-
// struct-argument call in this function, and std/gui/gui_cocoa.sc's NSRect
// case, worked fine already) but wasn't what a real, externally-compiled
// function's argument registers actually expect. Verified end to end: this
// now returns correct results from a real GPU dispatch for every op below.
static int __mps_run_binary_kernel(const char* kernelSrc, const char* kernelName,
                                    const float* a, const float* b, float* out, unsigned long n) {
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

        void* fnNameStr = __mps_nsstring(kernelName);
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

// Same as __mps_run_binary_kernel but for a single-input elementwise kernel
// (log/exp/sqrt) — one input buffer instead of two, otherwise identical
// setup/dispatch/readback.
static int __mps_run_unary_kernel(const char* kernelSrc, const char* kernelName,
                                   const float* a, float* out, unsigned long n) {
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

        void* fnNameStr = __mps_nsstring(kernelName);
        MMsg1 msg1 = (MMsg1)objc_msgSend;
        void* kernelFn = msg1(library, __mps_sel("newFunctionWithName:"), fnNameStr);
        if (kernelFn == (void*)0) return 0;

        MMsg2 newPipelineFn = (MMsg2)objc_msgSend;
        void* pipeline = newPipelineFn(device, __mps_sel("newComputePipelineStateWithFunction:error:"),
                                        kernelFn, (void*)&errPtr);
        if (pipeline == (void*)0) return 0;

        unsigned long bytes = n * sizeof(float);
        MMsgNewBufferBytes newBufBytesFn = (MMsgNewBufferBytes)objc_msgSend;
        void* bufA = newBufBytesFn(device, __mps_sel("newBufferWithBytes:length:options:"),
                                    (const void*)a, bytes, 0UL);
        MMsgNewBufferLen newBufLenFn = (MMsgNewBufferLen)objc_msgSend;
        void* bufOut = newBufLenFn(device, __mps_sel("newBufferWithLength:options:"), bytes, 0UL);
        if (bufA == (void*)0 || bufOut == (void*)0) return 0;

        void* cmdBuf = msg0(queue, __mps_sel("commandBuffer"));
        void* encoder = msg0(cmdBuf, __mps_sel("computeCommandEncoder"));

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __mps_sel("setComputePipelineState:"), pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __mps_sel("setBuffer:offset:atIndex:"), bufA, 0UL, 0UL);
        setBufFn(encoder, __mps_sel("setBuffer:offset:atIndex:"), bufOut, 0UL, 1UL);

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

int mps_add_f32(const float* a, const float* b, float* out, unsigned long n) {
    const char* kernelSrc =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void add_kernel(device const float* a [[buffer(0)]],\n"
        "                        device const float* b [[buffer(1)]],\n"
        "                        device float* out [[buffer(2)]],\n"
        "                        uint id [[thread_position_in_grid]]) {\n"
        "    out[id] = a[id] + b[id];\n"
        "}\n";
    return __mps_run_binary_kernel(kernelSrc, "add_kernel", a, b, out, n);
}

int mps_sub_f32(const float* a, const float* b, float* out, unsigned long n) {
    const char* kernelSrc =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void sub_kernel(device const float* a [[buffer(0)]],\n"
        "                        device const float* b [[buffer(1)]],\n"
        "                        device float* out [[buffer(2)]],\n"
        "                        uint id [[thread_position_in_grid]]) {\n"
        "    out[id] = a[id] - b[id];\n"
        "}\n";
    return __mps_run_binary_kernel(kernelSrc, "sub_kernel", a, b, out, n);
}

int mps_mul_f32(const float* a, const float* b, float* out, unsigned long n) {
    const char* kernelSrc =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void mul_kernel(device const float* a [[buffer(0)]],\n"
        "                        device const float* b [[buffer(1)]],\n"
        "                        device float* out [[buffer(2)]],\n"
        "                        uint id [[thread_position_in_grid]]) {\n"
        "    out[id] = a[id] * b[id];\n"
        "}\n";
    return __mps_run_binary_kernel(kernelSrc, "mul_kernel", a, b, out, n);
}

int mps_div_f32(const float* a, const float* b, float* out, unsigned long n) {
    const char* kernelSrc =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void div_kernel(device const float* a [[buffer(0)]],\n"
        "                        device const float* b [[buffer(1)]],\n"
        "                        device float* out [[buffer(2)]],\n"
        "                        uint id [[thread_position_in_grid]]) {\n"
        "    out[id] = a[id] / b[id];\n"
        "}\n";
    return __mps_run_binary_kernel(kernelSrc, "div_kernel", a, b, out, n);
}

int mps_pow_f32(const float* a, const float* b, float* out, unsigned long n) {
    const char* kernelSrc =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void pow_kernel(device const float* a [[buffer(0)]],\n"
        "                        device const float* b [[buffer(1)]],\n"
        "                        device float* out [[buffer(2)]],\n"
        "                        uint id [[thread_position_in_grid]]) {\n"
        "    out[id] = pow(a[id], b[id]);\n"
        "}\n";
    return __mps_run_binary_kernel(kernelSrc, "pow_kernel", a, b, out, n);
}

int mps_log_f32(const float* a, float* out, unsigned long n) {
    const char* kernelSrc =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void log_kernel(device const float* a [[buffer(0)]],\n"
        "                        device float* out [[buffer(1)]],\n"
        "                        uint id [[thread_position_in_grid]]) {\n"
        "    out[id] = log(a[id]);\n"
        "}\n";
    return __mps_run_unary_kernel(kernelSrc, "log_kernel", a, out, n);
}

int mps_exp_f32(const float* a, float* out, unsigned long n) {
    const char* kernelSrc =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void exp_kernel(device const float* a [[buffer(0)]],\n"
        "                        device float* out [[buffer(1)]],\n"
        "                        uint id [[thread_position_in_grid]]) {\n"
        "    out[id] = exp(a[id]);\n"
        "}\n";
    return __mps_run_unary_kernel(kernelSrc, "exp_kernel", a, out, n);
}

int mps_sqrt_f32(const float* a, float* out, unsigned long n) {
    const char* kernelSrc =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void sqrt_kernel(device const float* a [[buffer(0)]],\n"
        "                        device float* out [[buffer(1)]],\n"
        "                        uint id [[thread_position_in_grid]]) {\n"
        "    out[id] = sqrt(a[id]);\n"
        "}\n";
    return __mps_run_unary_kernel(kernelSrc, "sqrt_kernel", a, out, n);
}

// mps_relu_f32's own definition lives further down, alongside
// mps_matmul_f32 and the batching primitives it (like matmul) needs to
// check — see the comment there for why it's not routed through
// __mps_run_unary_kernel like log/exp/sqrt above it.

// out[i] = a[i] * k for a compile-time-unknown scalar k -- same shape as
// __mps_run_unary_kernel but with one extra setBytes:length:atIndex: call
// to pass k into buffer index 1 (out moves to index 2) before dispatch.
int mps_scale_f32(const float* a, float k, float* out, unsigned long n) {
    const char* kernelSrc =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void scale_kernel(device const float* a [[buffer(0)]],\n"
        "                          constant float& k [[buffer(1)]],\n"
        "                          device float* out [[buffer(2)]],\n"
        "                          uint id [[thread_position_in_grid]]) {\n"
        "    out[id] = a[id] * k;\n"
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

        void* fnNameStr = __mps_nsstring("scale_kernel");
        MMsg1 msg1 = (MMsg1)objc_msgSend;
        void* kernelFn = msg1(library, __mps_sel("newFunctionWithName:"), fnNameStr);
        if (kernelFn == (void*)0) return 0;

        MMsg2 newPipelineFn = (MMsg2)objc_msgSend;
        void* pipeline = newPipelineFn(device, __mps_sel("newComputePipelineStateWithFunction:error:"),
                                        kernelFn, (void*)&errPtr);
        if (pipeline == (void*)0) return 0;

        unsigned long bytes = n * sizeof(float);
        MMsgNewBufferBytes newBufBytesFn = (MMsgNewBufferBytes)objc_msgSend;
        void* bufA = newBufBytesFn(device, __mps_sel("newBufferWithBytes:length:options:"),
                                    (const void*)a, bytes, 0UL);
        MMsgNewBufferLen newBufLenFn = (MMsgNewBufferLen)objc_msgSend;
        void* bufOut = newBufLenFn(device, __mps_sel("newBufferWithLength:options:"), bytes, 0UL);
        if (bufA == (void*)0 || bufOut == (void*)0) return 0;

        void* cmdBuf = msg0(queue, __mps_sel("commandBuffer"));
        void* encoder = msg0(cmdBuf, __mps_sel("computeCommandEncoder"));

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __mps_sel("setComputePipelineState:"), pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __mps_sel("setBuffer:offset:atIndex:"), bufA, 0UL, 0UL);
        // setBytes:length:atIndex: shares setBuffer:offset:atIndex:'s
        // (recv, sel, ptr, NSUInteger, NSUInteger) shape -- see
        // mps_matmul_f32's comment on reusing MMsgSetBuffer for it.
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        float kVal = k;
        setBytesFn(encoder, __mps_sel("setBytes:length:atIndex:"), (void*)&kVal, 4UL, 1UL);
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

// out[0] = sum(a[0..n)) -- a single GPU thread walks the whole array and
// accumulates serially. Correct, and enough to prove the reduction-shaped
// dispatch works end to end, but not a real parallel reduction (no
// threadgroup-memory tree, no multiple-threadgroups-then-combine pass) --
// same "simplest real version first" spirit as mps_matmul_f32's lack of
// tiling. A single GPU thread doing a serial O(n) sum is, unsurprisingly,
// not where GPU dispatch is going to win; this exists for completeness of
// the op set (tensor_sum is used for the loss in the training benchmark)
// more than as a performance claim.
int mps_sum_f32(const float* a, float* out, unsigned long n) {
    const char* kernelSrc =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void sum_kernel(device const float* a [[buffer(0)]],\n"
        "                        device float* out [[buffer(1)]],\n"
        "                        constant uint& n [[buffer(2)]],\n"
        "                        uint id [[thread_position_in_grid]]) {\n"
        "    if (id != 0) return;\n"
        "    float acc = 0.0;\n"
        "    for (uint i = 0; i < n; i++) { acc += a[i]; }\n"
        "    out[0] = acc;\n"
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

        void* fnNameStr = __mps_nsstring("sum_kernel");
        MMsg1 msg1 = (MMsg1)objc_msgSend;
        void* kernelFn = msg1(library, __mps_sel("newFunctionWithName:"), fnNameStr);
        if (kernelFn == (void*)0) return 0;

        MMsg2 newPipelineFn = (MMsg2)objc_msgSend;
        void* pipeline = newPipelineFn(device, __mps_sel("newComputePipelineStateWithFunction:error:"),
                                        kernelFn, (void*)&errPtr);
        if (pipeline == (void*)0) return 0;

        unsigned long bytes = n * sizeof(float);
        MMsgNewBufferBytes newBufBytesFn = (MMsgNewBufferBytes)objc_msgSend;
        void* bufA = newBufBytesFn(device, __mps_sel("newBufferWithBytes:length:options:"),
                                    (const void*)a, bytes, 0UL);
        MMsgNewBufferLen newBufLenFn = (MMsgNewBufferLen)objc_msgSend;
        void* bufOut = newBufLenFn(device, __mps_sel("newBufferWithLength:options:"), 4UL, 0UL);
        if (bufA == (void*)0 || bufOut == (void*)0) return 0;

        void* cmdBuf = msg0(queue, __mps_sel("commandBuffer"));
        void* encoder = msg0(cmdBuf, __mps_sel("computeCommandEncoder"));

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __mps_sel("setComputePipelineState:"), pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __mps_sel("setBuffer:offset:atIndex:"), bufA, 0UL, 0UL);
        setBufFn(encoder, __mps_sel("setBuffer:offset:atIndex:"), bufOut, 0UL, 1UL);
        unsigned int nu = (unsigned int)n;
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(encoder, __mps_sel("setBytes:length:atIndex:"), (void*)&nu, 4UL, 2UL);

        struct MTLSize gridSize;
        gridSize.width = 1UL; gridSize.height = 1UL; gridSize.depth = 1UL;
        struct MTLSize tgSize;
        tgSize.width = 1UL; tgSize.height = 1UL; tgSize.depth = 1UL;
        MMsgDispatch dispatchFn = (MMsgDispatch)objc_msgSend;
        dispatchFn(encoder, __mps_sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                   gridSize, tgSize);
        MMsgVoid0 msgVoid0 = (MMsgVoid0)objc_msgSend;
        msgVoid0(encoder, __mps_sel("endEncoding"));
        msgVoid0(cmdBuf, __mps_sel("commit"));
        msgVoid0(cmdBuf, __mps_sel("waitUntilCompleted"));

        void* outPtr = msg0(bufOut, __mps_sel("contents"));
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
        if (__mps_device == (void*)0) {
            void* dev = MTLCreateSystemDefaultDevice();
            if (dev == (void*)0) return;
            MMsg0 msg0init = (MMsg0)objc_msgSend;
            void* q = msg0init(dev, __mps_sel("newCommandQueue"));
            if (q == (void*)0) return;
            __mps_device = dev;
            __mps_queue = q;
        }
        MMsg0 msg0 = (MMsg0)objc_msgSend;
        __mps_batch_cmdBuf  = msg0(__mps_queue, __mps_sel("commandBuffer"));
        __mps_batch_encoder = msg0(__mps_batch_cmdBuf, __mps_sel("computeCommandEncoder"));
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
        msgVoid0(__mps_batch_encoder, __mps_sel("endEncoding"));
        msgVoid0(__mps_batch_cmdBuf, __mps_sel("commit"));
        msgVoid0(__mps_batch_cmdBuf, __mps_sel("waitUntilCompleted"));

        MMsg0 msg0 = (MMsg0)objc_msgSend;
        unsigned long i = 0UL;
        while (i < __mps_pending_count) {
            void* outPtr = msg0(__mps_pending_gpuBuf[i], __mps_sel("contents"));
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
// Cached device/queue (shared with every other cached op in this file)
// and this op's own compiled pipeline state, lazily created once on the
// first call and reused forever after. Before this caching existed,
// EVERY single mps_matmul_f32 call recompiled 'kernelSrc' from Metal
// Shading Language source via newLibraryWithSource:options:error: (a
// real runtime shader compile) and recreated the device/queue/pipeline
// from scratch -- runtime shader compilation is, by a wide margin, the
// most expensive single step in this whole function, dwarfing the
// dispatch+readback that follows it for any matmul small enough that
// dispatch overhead was already the concern (see this file's own
// mps_matmul_f32 comment on not being a tuned GEMM). Metal objects
// created via '...alloc'/'new...' methods are already +1 retained with
// no matching release anywhere else in this file (consistent existing
// convention) -- caching them in file-scope statics instead of a local
// is exactly that same lifetime, just reused across calls instead of
// leaked once per call.
static void* __mps_device = (void*)0;
static void* __mps_queue = (void*)0;
static void* __mps_matmul_pipeline = (void*)0;
static void* __mps_relu_pipeline = (void*)0;

int mps_matmul_f32(const float* a, const float* b, float* out,
                    unsigned long M, unsigned long K, unsigned long N) {
    unsafe {
        if (__mps_device == (void*)0) {
            void* dev = MTLCreateSystemDefaultDevice();
            if (dev == (void*)0) return 0;
            MMsg0 msg0init = (MMsg0)objc_msgSend;
            void* q = msg0init(dev, __mps_sel("newCommandQueue"));
            if (q == (void*)0) return 0;
            __mps_device = dev;
            __mps_queue = q;
        }
        void* device = __mps_device;
        void* queue = __mps_queue;
        MMsg0 msg0 = (MMsg0)objc_msgSend;

        if (__mps_matmul_pipeline == (void*)0) {
            const char* kernelSrc =
                "#include <metal_stdlib>\n"
                "using namespace metal;\n"
                "kernel void matmul_kernel(device const float* a [[buffer(0)]],\n"
                "                           device const float* b [[buffer(1)]],\n"
                "                           device float* out [[buffer(2)]],\n"
                "                           constant uint& M [[buffer(3)]],\n"
                "                           constant uint& K [[buffer(4)]],\n"
                "                           constant uint& N [[buffer(5)]],\n"
                "                           uint2 gid [[thread_position_in_grid]]) {\n"
                "    if (gid.x >= N || gid.y >= M) return;\n"
                "    float acc = 0.0;\n"
                "    for (uint p = 0; p < K; p++) {\n"
                "        acc += a[gid.y * K + p] * b[p * N + gid.x];\n"
                "    }\n"
                "    out[gid.y * N + gid.x] = acc;\n"
                "}\n";
            void* srcStr = __mps_nsstring(kernelSrc);
            void* errPtr = (void*)0;
            MMsg3Lib newLibFn = (MMsg3Lib)objc_msgSend;
            void* library = newLibFn(device, __mps_sel("newLibraryWithSource:options:error:"),
                                      srcStr, (void*)0, (void*)&errPtr);
            if (library == (void*)0) return 0;

            void* fnNameStr = __mps_nsstring("matmul_kernel");
            MMsg1 msg1 = (MMsg1)objc_msgSend;
            void* kernelFn = msg1(library, __mps_sel("newFunctionWithName:"), fnNameStr);
            if (kernelFn == (void*)0) return 0;

            MMsg2 newPipelineFn = (MMsg2)objc_msgSend;
            void* pipeline = newPipelineFn(device, __mps_sel("newComputePipelineStateWithFunction:error:"),
                                            kernelFn, (void*)&errPtr);
            if (pipeline == (void*)0) return 0;
            __mps_matmul_pipeline = pipeline;
        }
        void* pipeline = __mps_matmul_pipeline;

        unsigned long bytesA = M * K * sizeof(float);
        unsigned long bytesB = K * N * sizeof(float);
        unsigned long bytesOut = M * N * sizeof(float);
        MMsgNewBufferBytes newBufBytesFn = (MMsgNewBufferBytes)objc_msgSend;
        void* bufA = newBufBytesFn(device, __mps_sel("newBufferWithBytes:length:options:"),
                                    (const void*)a, bytesA, 0UL);
        void* bufB = newBufBytesFn(device, __mps_sel("newBufferWithBytes:length:options:"),
                                    (const void*)b, bytesB, 0UL);
        MMsgNewBufferLen newBufLenFn = (MMsgNewBufferLen)objc_msgSend;
        void* bufOut = newBufLenFn(device, __mps_sel("newBufferWithLength:options:"), bytesOut, 0UL);
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
            cmdBuf = msg0(queue, __mps_sel("commandBuffer"));
            encoder = msg0(cmdBuf, __mps_sel("computeCommandEncoder"));
        }

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __mps_sel("setComputePipelineState:"), pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __mps_sel("setBuffer:offset:atIndex:"), bufA, 0UL, 0UL);
        setBufFn(encoder, __mps_sel("setBuffer:offset:atIndex:"), bufB, 0UL, 1UL);
        setBufFn(encoder, __mps_sel("setBuffer:offset:atIndex:"), bufOut, 0UL, 2UL);
        // setBytes:length:atIndex: has the exact same (recv, sel, ptr,
        // NSUInteger, NSUInteger) shape as setBuffer:offset:atIndex:, so
        // MMsgSetBuffer's typedef works for it too -- only the selector
        // and what the middle argument means (raw bytes vs a buffer
        // object) differ.
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(encoder, __mps_sel("setBytes:length:atIndex:"), (void*)&Mu, 4UL, 3UL);
        setBytesFn(encoder, __mps_sel("setBytes:length:atIndex:"), (void*)&Ku, 4UL, 4UL);
        setBytesFn(encoder, __mps_sel("setBytes:length:atIndex:"), (void*)&Nu, 4UL, 5UL);

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
        dispatchFn(encoder, __mps_sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                   numThreadgroups, tgSize);

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
        msgVoid0(encoder, __mps_sel("endEncoding"));
        msgVoid0(cmdBuf, __mps_sel("commit"));
        msgVoid0(cmdBuf, __mps_sel("waitUntilCompleted"));

        void* outPtr = msg0(bufOut, __mps_sel("contents"));
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
        if (__mps_device == (void*)0) {
            void* dev = MTLCreateSystemDefaultDevice();
            if (dev == (void*)0) return 0;
            MMsg0 msg0init = (MMsg0)objc_msgSend;
            void* q = msg0init(dev, __mps_sel("newCommandQueue"));
            if (q == (void*)0) return 0;
            __mps_device = dev;
            __mps_queue = q;
        }
        void* device = __mps_device;
        void* queue = __mps_queue;
        MMsg0 msg0 = (MMsg0)objc_msgSend;

        if (__mps_relu_pipeline == (void*)0) {
            const char* kernelSrc =
                "#include <metal_stdlib>\n"
                "using namespace metal;\n"
                "kernel void relu_kernel(device const float* a [[buffer(0)]],\n"
                "                        device float* out [[buffer(1)]],\n"
                "                        uint id [[thread_position_in_grid]]) {\n"
                "    out[id] = max(a[id], 0.0f);\n"
                "}\n";
            void* srcStr = __mps_nsstring(kernelSrc);
            void* errPtr = (void*)0;
            MMsg3Lib newLibFn = (MMsg3Lib)objc_msgSend;
            void* library = newLibFn(device, __mps_sel("newLibraryWithSource:options:error:"),
                                      srcStr, (void*)0, (void*)&errPtr);
            if (library == (void*)0) return 0;

            void* fnNameStr = __mps_nsstring("relu_kernel");
            MMsg1 msg1 = (MMsg1)objc_msgSend;
            void* kernelFn = msg1(library, __mps_sel("newFunctionWithName:"), fnNameStr);
            if (kernelFn == (void*)0) return 0;

            MMsg2 newPipelineFn = (MMsg2)objc_msgSend;
            void* pipeline = newPipelineFn(device, __mps_sel("newComputePipelineStateWithFunction:error:"),
                                            kernelFn, (void*)&errPtr);
            if (pipeline == (void*)0) return 0;
            __mps_relu_pipeline = pipeline;
        }
        void* pipeline = __mps_relu_pipeline;

        unsigned long bytes = n * sizeof(float);
        MMsgNewBufferBytes newBufBytesFn = (MMsgNewBufferBytes)objc_msgSend;
        void* bufA = newBufBytesFn(device, __mps_sel("newBufferWithBytes:length:options:"),
                                    (const void*)a, bytes, 0UL);
        MMsgNewBufferLen newBufLenFn = (MMsgNewBufferLen)objc_msgSend;
        void* bufOut = newBufLenFn(device, __mps_sel("newBufferWithLength:options:"), bytes, 0UL);
        if (bufA == (void*)0 || bufOut == (void*)0) return 0;

        int batched = __mps_batch_active;
        void* cmdBuf = (void*)0;
        void* encoder;
        if (batched) {
            encoder = __mps_batch_encoder;
        } else {
            cmdBuf = msg0(queue, __mps_sel("commandBuffer"));
            encoder = msg0(cmdBuf, __mps_sel("computeCommandEncoder"));
        }

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(encoder, __mps_sel("setComputePipelineState:"), pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(encoder, __mps_sel("setBuffer:offset:atIndex:"), bufA, 0UL, 0UL);
        setBufFn(encoder, __mps_sel("setBuffer:offset:atIndex:"), bufOut, 0UL, 1UL);

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
        dispatchFn(encoder, __mps_sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                   numThreadgroups, tgSize);

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
        msgVoid0(encoder, __mps_sel("endEncoding"));
        msgVoid0(cmdBuf, __mps_sel("commit"));
        msgVoid0(cmdBuf, __mps_sel("waitUntilCompleted"));

        void* outPtr = msg0(bufOut, __mps_sel("contents"));
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
            const char* kernelSrc =
                "#include <metal_stdlib>\n"
                "using namespace metal;\n"
                "kernel void matmul_kernel(device const float* a [[buffer(0)]],\n"
                "                           device const float* b [[buffer(1)]],\n"
                "                           device float* out [[buffer(2)]],\n"
                "                           constant uint& M [[buffer(3)]],\n"
                "                           constant uint& K [[buffer(4)]],\n"
                "                           constant uint& N [[buffer(5)]],\n"
                "                           uint2 gid [[thread_position_in_grid]]) {\n"
                "    if (gid.x >= N || gid.y >= M) return;\n"
                "    float acc = 0.0;\n"
                "    for (uint p = 0; p < K; p++) {\n"
                "        acc += a[gid.y * K + p] * b[p * N + gid.x];\n"
                "    }\n"
                "    out[gid.y * N + gid.x] = acc;\n"
                "}\n";
            void* srcStr = __mps_nsstring(kernelSrc);
            void* errPtr = (void*)0;
            MMsg3Lib newLibFn = (MMsg3Lib)objc_msgSend;
            void* library = newLibFn(__mps_device, __mps_sel("newLibraryWithSource:options:error:"),
                                      srcStr, (void*)0, (void*)&errPtr);
            if (library == (void*)0) return 0;
            void* fnNameStr = __mps_nsstring("matmul_kernel");
            MMsg1 msg1 = (MMsg1)objc_msgSend;
            void* kernelFn = msg1(library, __mps_sel("newFunctionWithName:"), fnNameStr);
            if (kernelFn == (void*)0) return 0;
            MMsg2 newPipelineFn = (MMsg2)objc_msgSend;
            void* pipeline = newPipelineFn(__mps_device, __mps_sel("newComputePipelineStateWithFunction:error:"),
                                            kernelFn, (void*)&errPtr);
            if (pipeline == (void*)0) return 0;
            __mps_matmul_pipeline = pipeline;
        }

        unsigned long bytesB = K * N * sizeof(float);
        unsigned long bytesOut = M * N * sizeof(float);
        MMsgNewBufferBytes newBufBytesFn = (MMsgNewBufferBytes)objc_msgSend;
        void* bufB = newBufBytesFn(__mps_device, __mps_sel("newBufferWithBytes:length:options:"),
                                    (const void*)b, bytesB, 0UL);
        MMsgNewBufferLen newBufLenFn = (MMsgNewBufferLen)objc_msgSend;
        void* bufOut = newBufLenFn(__mps_device, __mps_sel("newBufferWithLength:options:"), bytesOut, 0UL);
        if (bufB == (void*)0 || bufOut == (void*)0) return 0;

        unsigned int Mu = (unsigned int)M;
        unsigned int Ku = (unsigned int)K;
        unsigned int Nu = (unsigned int)N;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __mps_sel("setComputePipelineState:"), __mps_matmul_pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __mps_sel("setBuffer:offset:atIndex:"), bufA, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __mps_sel("setBuffer:offset:atIndex:"), bufB, 0UL, 1UL);
        setBufFn(__mps_batch_encoder, __mps_sel("setBuffer:offset:atIndex:"), bufOut, 0UL, 2UL);
        MMsgSetBuffer setBytesFn = (MMsgSetBuffer)objc_msgSend;
        setBytesFn(__mps_batch_encoder, __mps_sel("setBytes:length:atIndex:"), (void*)&Mu, 4UL, 3UL);
        setBytesFn(__mps_batch_encoder, __mps_sel("setBytes:length:atIndex:"), (void*)&Ku, 4UL, 4UL);
        setBytesFn(__mps_batch_encoder, __mps_sel("setBytes:length:atIndex:"), (void*)&Nu, 4UL, 5UL);

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
        dispatchFn(__mps_batch_encoder, __mps_sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                   numThreadgroups, tgSize);

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
            const char* kernelSrc =
                "#include <metal_stdlib>\n"
                "using namespace metal;\n"
                "kernel void relu_kernel(device const float* a [[buffer(0)]],\n"
                "                        device float* out [[buffer(1)]],\n"
                "                        uint id [[thread_position_in_grid]]) {\n"
                "    out[id] = max(a[id], 0.0f);\n"
                "}\n";
            void* srcStr = __mps_nsstring(kernelSrc);
            void* errPtr = (void*)0;
            MMsg3Lib newLibFn = (MMsg3Lib)objc_msgSend;
            void* library = newLibFn(__mps_device, __mps_sel("newLibraryWithSource:options:error:"),
                                      srcStr, (void*)0, (void*)&errPtr);
            if (library == (void*)0) return 0;
            void* fnNameStr = __mps_nsstring("relu_kernel");
            MMsg1 msg1 = (MMsg1)objc_msgSend;
            void* kernelFn = msg1(library, __mps_sel("newFunctionWithName:"), fnNameStr);
            if (kernelFn == (void*)0) return 0;
            MMsg2 newPipelineFn = (MMsg2)objc_msgSend;
            void* pipeline = newPipelineFn(__mps_device, __mps_sel("newComputePipelineStateWithFunction:error:"),
                                            kernelFn, (void*)&errPtr);
            if (pipeline == (void*)0) return 0;
            __mps_relu_pipeline = pipeline;
        }

        unsigned long bytes = n * sizeof(float);
        MMsgNewBufferLen newBufLenFn = (MMsgNewBufferLen)objc_msgSend;
        void* bufOut = newBufLenFn(__mps_device, __mps_sel("newBufferWithLength:options:"), bytes, 0UL);
        if (bufOut == (void*)0) return 0;

        MMsgVoid1 msgVoid1 = (MMsgVoid1)objc_msgSend;
        msgVoid1(__mps_batch_encoder, __mps_sel("setComputePipelineState:"), __mps_relu_pipeline);
        MMsgSetBuffer setBufFn = (MMsgSetBuffer)objc_msgSend;
        setBufFn(__mps_batch_encoder, __mps_sel("setBuffer:offset:atIndex:"), bufA, 0UL, 0UL);
        setBufFn(__mps_batch_encoder, __mps_sel("setBuffer:offset:atIndex:"), bufOut, 0UL, 1UL);

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
        dispatchFn(__mps_batch_encoder, __mps_sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                   numThreadgroups, tgSize);

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
