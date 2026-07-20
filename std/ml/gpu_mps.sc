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

int mps_relu_f32(const float* a, float* out, unsigned long n) {
    const char* kernelSrc =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void relu_kernel(device const float* a [[buffer(0)]],\n"
        "                        device float* out [[buffer(1)]],\n"
        "                        uint id [[thread_position_in_grid]]) {\n"
        "    out[id] = max(a[id], 0.0f);\n"
        "}\n";
    return __mps_run_unary_kernel(kernelSrc, "relu_kernel", a, out, n);
}

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
int mps_matmul_f32(const float* a, const float* b, float* out,
                    unsigned long M, unsigned long K, unsigned long N) {
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

        void* fnNameStr = __mps_nsstring("matmul_kernel");
        MMsg1 msg1 = (MMsg1)objc_msgSend;
        void* kernelFn = msg1(library, __mps_sel("newFunctionWithName:"), fnNameStr);
        if (kernelFn == (void*)0) return 0;

        MMsg2 newPipelineFn = (MMsg2)objc_msgSend;
        void* pipeline = newPipelineFn(device, __mps_sel("newComputePipelineStateWithFunction:error:"),
                                        kernelFn, (void*)&errPtr);
        if (pipeline == (void*)0) return 0;

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

        void* cmdBuf = msg0(queue, __mps_sel("commandBuffer"));
        void* encoder = msg0(cmdBuf, __mps_sel("computeCommandEncoder"));

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

} // namespace std
