// SafeC Standard Library — cuDNN convolution, pooling and U-Net (see gpu_cudnn.h).
//
// Convolutions go through cuDNN's graph (backend) API on gpu_cuda's stream:
// each shape gets a cached plan for the fused graph conv -> + bias (-> relu),
// built from the engine configs cuDNN's heuristics suggest and timed on the
// spot (wall clock, so per-call host cost counts as well as kernel time);
// the fastest one is kept with its own workspace and a variant pack that is
// rebuilt only when the buffers change. cuDNN 9's legacy entry points
// (cudnnConvolutionForward & co.) re-derive their engine on every call,
// which costs ~40 us of CPU per convolution on a Xeon 8570 — longer than the
// kernel itself at batch 1. The legacy descriptor API (algorithm from
// cudnnFindConvolutionForwardAlgorithm, conv + cudnnAddTensor) stays as the
// fallback for shapes no graph engine takes.
// Workspaces are per plan rather than shared, so a CudaGraph that captured a
// convolution keeps a valid workspace however many shapes come later.
#pragma once
#include <std/stddef.h>
#include <std/ml/gpu_cudnn.h>
#include <std/ml/gpu_cuda.h>
#include <std/ml/cnn.h>
#include <std/ml/unet.h>
#include <std/mem.h>
#include <std/time.h>

namespace std {

// ── cuDNN 9 C ABI (cudnn_graph.h, cudnn_ops.h, cudnn_cnn.h) ──────────────────
extern int cudnnCreate(void** handle);
extern int cudnnSetStream(void* handle, void* stream);
extern int cudnnCreateTensorDescriptor(void** desc);
extern int cudnnDestroyTensorDescriptor(void* desc);
extern int cudnnSetTensor4dDescriptor(void* desc, int format, int dataType, int n, int c, int h, int w);
extern int cudnnCreateFilterDescriptor(void** desc);
extern int cudnnDestroyFilterDescriptor(void* desc);
extern int cudnnSetFilter4dDescriptor(void* desc, int dataType, int format, int k, int c, int h, int w);
extern int cudnnCreateConvolutionDescriptor(void** desc);
extern int cudnnDestroyConvolutionDescriptor(void* desc);
extern int cudnnSetConvolution2dDescriptor(void* desc, int padH, int padW, int u, int v,
                                           int dilationH, int dilationW, int mode, int computeType);
extern int cudnnSetConvolutionMathType(void* desc, int mathType);
extern int cudnnFindConvolutionForwardAlgorithm(void* handle, void* xDesc, void* wDesc, void* convDesc,
                                                void* yDesc, int requested, int* returned, void* perfResults);
extern int cudnnConvolutionForward(void* handle, const void* alpha, void* xDesc, const void* x,
                                   void* wDesc, const void* w, void* convDesc, int algo,
                                   void* workspace, size_t wsBytes, const void* beta, void* yDesc, void* y);
extern int cudnnConvolutionBiasActivationForward(void* handle, const void* alpha1, void* xDesc, const void* x,
                                                 void* wDesc, const void* w, void* convDesc, int algo,
                                                 void* workspace, size_t wsBytes, const void* alpha2,
                                                 void* zDesc, const void* z, void* biasDesc, const void* bias,
                                                 void* actDesc, void* yDesc, void* y);
extern int cudnnAddTensor(void* handle, const void* alpha, void* aDesc, const void* a,
                          const void* beta, void* cDesc, void* c);
extern int cudnnCreateActivationDescriptor(void** desc);
extern int cudnnSetActivationDescriptor(void* desc, int mode, int nanOpt, double coef);
extern int cudnnCreatePoolingDescriptor(void** desc);
extern int cudnnSetPooling2dDescriptor(void* desc, int mode, int nanOpt, int windowH, int windowW,
                                       int padH, int padW, int strideH, int strideW);
extern int cudnnPoolingForward(void* handle, void* poolDesc, const void* alpha, void* xDesc, const void* x,
                               const void* beta, void* yDesc, void* y);

// cudnn_graph.h backend API
extern int cudnnBackendCreateDescriptor(int type, void** desc);
extern int cudnnBackendDestroyDescriptor(void* desc);
extern int cudnnBackendFinalize(void* desc);
extern int cudnnBackendSetAttribute(void* desc, int name, int type, long count, const void* values);
extern int cudnnBackendGetAttribute(void* desc, int name, int type, long requested, long* count, void* values);
extern int cudnnBackendExecute(void* handle, void* plan, void* variantPack);

extern int cuMemAlloc_v2(unsigned long long* dptr, size_t bytes);
extern int cuMemFree_v2(unsigned long long dptr);
extern int cuStreamSynchronize(void* stream);
extern int cuEventCreate(void** event, unsigned int flags);
extern int cuEventRecord(void* event, void* stream);
extern int cuEventSynchronize(void* event);
extern int cuEventElapsedTime(float* ms, void* start, void* end);
extern int cuEventDestroy_v2(void* event);

#define DNN_TENSOR_NCHW        0    // CUDNN_TENSOR_NCHW
#define DNN_DATA_FLOAT         0    // CUDNN_DATA_FLOAT
#define DNN_DATA_HALF          2    // CUDNN_DATA_HALF
#define DNN_DATA_BFLOAT16      9    // CUDNN_DATA_BFLOAT16
#define DNN_TENSOR_OP_MATH     1    // CUDNN_TENSOR_OP_MATH
#define DNN_CROSS_CORRELATION  1    // CUDNN_CROSS_CORRELATION (no filter flip, like cnn.sc)
#define DNN_DEFAULT_MATH       0    // CUDNN_DEFAULT_MATH (fp32 data may use TF32 tensor cores)
#define DNN_FMA_MATH           3    // CUDNN_FMA_MATH (plain fp32 FMA)
#define DNN_POOL_MAX           0    // CUDNN_POOLING_MAX
#define DNN_POOL_AVG           2    // CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
#define DNN_NOT_PROPAGATE_NAN  0    // CUDNN_NOT_PROPAGATE_NAN
#define DNN_ACT_RELU           1    // CUDNN_ACTIVATION_RELU
#define DNN_PERF_INTS          12   // sizeof(cudnnConvolutionFwdAlgoPerf_t) / 4 (48 bytes)
#define DNN_PERF_COUNT         8
#define DNN_PLANS              32
// cudnn_graph.h enum values
#define DNB_POINTWISE          0    // CUDNN_BACKEND_POINTWISE_DESCRIPTOR
#define DNB_CONV               1    // CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR
#define DNB_ENGINE             2    // CUDNN_BACKEND_ENGINE_DESCRIPTOR
#define DNB_ENGINECFG          3    // CUDNN_BACKEND_ENGINECFG_DESCRIPTOR
#define DNB_ENGINEHEUR         4    // CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR
#define DNB_PLAN               5    // CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR
#define DNB_OP_CONV_FWD        10   // CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR
#define DNB_OP_POINTWISE       13   // CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR
#define DNB_OPGRAPH            15   // CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR
#define DNB_VARIANT_PACK       16   // CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR
#define DNB_TENSOR             17   // CUDNN_BACKEND_TENSOR_DESCRIPTOR
#define DNB_T_HANDLE           0    // CUDNN_TYPE_HANDLE
#define DNB_T_DATA_TYPE        1
#define DNB_T_BOOLEAN          2
#define DNB_T_INT64            3
#define DNB_T_FLOAT            4
#define DNB_T_VOID_PTR         6
#define DNB_T_CONV_MODE        7
#define DNB_T_HEUR_MODE        8
#define DNB_T_NUMERICAL_NOTE   11
#define DNB_T_POINTWISE_MODE   14
#define DNB_T_DESCRIPTOR       15
#define DNB_A_PW_MODE          0    // CUDNN_ATTR_POINTWISE_MODE
#define DNB_A_PW_MATH_PREC     1
#define DNB_A_CONV_COMP_TYPE   100
#define DNB_A_CONV_MODE        101
#define DNB_A_CONV_DILATIONS   102
#define DNB_A_CONV_STRIDES     103
#define DNB_A_CONV_POST_PAD    104
#define DNB_A_CONV_PRE_PAD     105
#define DNB_A_CONV_DIMS        106
#define DNB_A_HEUR_MODE        200
#define DNB_A_HEUR_GRAPH       201
#define DNB_A_HEUR_RESULTS     202
#define DNB_A_CFG_ENGINE       300
#define DNB_A_PLAN_HANDLE      400
#define DNB_A_PLAN_CFG         401
#define DNB_A_PLAN_WS_SIZE     402
#define DNB_A_FWD_ALPHA        700
#define DNB_A_FWD_BETA         701
#define DNB_A_FWD_CONV         702
#define DNB_A_FWD_W            703
#define DNB_A_FWD_X            704
#define DNB_A_FWD_Y            705
#define DNB_A_PWOP_DESC        750
#define DNB_A_PWOP_X           751
#define DNB_A_PWOP_B           752
#define DNB_A_PWOP_Y           753
#define DNB_A_GRAPH_HANDLE     800
#define DNB_A_GRAPH_OPS        801
#define DNB_A_TENSOR_ALIGN     900
#define DNB_A_TENSOR_TYPE      901
#define DNB_A_TENSOR_DIMS      902
#define DNB_A_TENSOR_STRIDES   903
#define DNB_A_TENSOR_UID       906
#define DNB_A_TENSOR_VIRTUAL   907
#define DNB_A_VP_UIDS          1000
#define DNB_A_VP_PTRS          1001
#define DNB_A_VP_WORKSPACE     1003
#define DNB_A_ENGINE_NOTES     1303
#define DNB_PW_ADD             0    // CUDNN_POINTWISE_ADD
#define DNB_PW_RELU            100  // CUDNN_POINTWISE_RELU_FWD
#define DNB_HEUR_A             3    // CUDNN_HEUR_MODE_A
#define DNB_NOTE_TENSOR_CORE   0    // CUDNN_NUMERICAL_NOTE_TENSOR_CORE
#define DNB_NOTE_DOWN_CONVERT  1    // ..._DOWN_CONVERT_INPUTS
#define DNB_NOTE_REDUCED_PREC  2    // ..._REDUCED_PRECISION_REDUCTION
#define DNB_DESCS              14   // graph descriptors a backend plan keeps alive
#define DNB_MAX_CONFIGS        24   // heuristic configs timed per shape
#define DNN_MAX_WORKSPACE      268435456UL   // 256 MiB: skip algorithms that want more

// ── State ────────────────────────────────────────────────────────────────────
static int   g_dnn_state = 0;        // 0 = not tried, 1 = ready, -1 = unavailable
static void* g_dnn = (void*)0;
static void* g_dnn_relu = (void*)0;
static void* g_dnn_pool = (void*)0;
static void* g_dnn_pool_x = (void*)0;
static void* g_dnn_pool_y = (void*)0;

// Plan cache. g_dp_key[s * 8 ..] = batch, inC, H, W, outC, kH, kW,
// stride | padding << 16 | tf32 << 32 | relu << 33; g_dp_used[s] = 0 for a
// free slot, 1 for a legacy plan (g_dp_x .. g_dp_algo), 2 for a graph plan
// (g_dp_bd: the descriptors it keeps alive, [0] the execution plan;
// g_dp_vp: variant pack for the buffers in g_dp_ptr; g_dp_gws: workspace).
// While a plan is built both kinds may be present; the slower is freed.
static unsigned long g_dp_key[DNN_PLANS * 8];
static int   g_dp_used[DNN_PLANS];
static void* g_dp_bd[DNN_PLANS * DNB_DESCS];
static void* g_dp_vp[DNN_PLANS];
static unsigned long g_dp_ptr[DNN_PLANS * 4];
static unsigned long long g_dp_gws[DNN_PLANS];
static void* g_dp_x[DNN_PLANS];
static void* g_dp_w[DNN_PLANS];
static void* g_dp_conv[DNN_PLANS];
static void* g_dp_y[DNN_PLANS];
static void* g_dp_b[DNN_PLANS];
static int   g_dp_algo[DNN_PLANS];
static unsigned long long g_dp_ws[DNN_PLANS];
static unsigned long g_dp_wsBytes[DNN_PLANS];
static int   g_dp_evict = 0;
// Data type of a graph plan's input / output tensors while one is built:
// DNN_DATA_FLOAT, or DNN_DATA_HALF / DNN_DATA_BFLOAT16 for the 16-bit
// convolutions (virtual tensors and compute stay fp32).
static int   g_dnb_io = 0;

static int cudnn_ensure_() {
    if (g_dnn_state == 1) return cuda_available();
    if (g_dnn_state != 0) return 0;
    g_dnn_state = -1;
    if (!cuda_available()) return 0;
    unsafe {
        if (cudnnCreate(&g_dnn) != 0) return 0;
        if (cudnnSetStream(g_dnn, cuda_stream_handle()) != 0) return 0;
        if (cudnnCreateActivationDescriptor(&g_dnn_relu) != 0 ||
            cudnnSetActivationDescriptor(g_dnn_relu, DNN_ACT_RELU, DNN_NOT_PROPAGATE_NAN, 0.0) != 0) return 0;
        if (cudnnCreatePoolingDescriptor(&g_dnn_pool) != 0 ||
            cudnnCreateTensorDescriptor(&g_dnn_pool_x) != 0 ||
            cudnnCreateTensorDescriptor(&g_dnn_pool_y) != 0) return 0;
        int s = 0;
        while (s < DNN_PLANS) {
            g_dp_used[s] = 0;
            g_dp_x[s] = (void*)0;
            g_dp_bd[s * DNB_DESCS] = (void*)0;
            s = s + 1;
        }
    }
    g_dnn_state = 1;
    return 1;
}

unsigned long cuda_conv_out_dim(unsigned long dim, unsigned long kernel,
                                unsigned long stride, unsigned long padding) {
    if (stride == 0UL || dim + 2UL * padding < kernel) return 0UL;
    return (dim + 2UL * padding - kernel) / stride + 1UL;
}

static int dnn_int_ok_(unsigned long v) { return v > 0UL && v <= 2147483647UL; }

// Frees a slot's graph plan (descriptors, variant pack, workspace), if any.
static void dnb_free_(int s) {
    unsafe {
        if (g_dp_bd[s * DNB_DESCS] == (void*)0) return;
        cuStreamSynchronize(cuda_stream_handle());   // queued work may still use the workspace
        if (g_dp_vp[s] != (void*)0) cudnnBackendDestroyDescriptor(g_dp_vp[s]);
        int i = 0;
        while (i < DNB_DESCS) {
            if (g_dp_bd[s * DNB_DESCS + i] != (void*)0) cudnnBackendDestroyDescriptor(g_dp_bd[s * DNB_DESCS + i]);
            g_dp_bd[s * DNB_DESCS + i] = (void*)0;
            i = i + 1;
        }
        if (g_dp_gws[s] != 0ULL) cuMemFree_v2(g_dp_gws[s]);
        g_dp_vp[s] = (void*)0;
        g_dp_gws[s] = 0ULL;
    }
}

// Frees a slot's legacy plan (descriptors, workspace), if any.
static void dnn_legacy_free_(int s) {
    unsafe {
        if (g_dp_x[s] == (void*)0) return;
        cudnnDestroyTensorDescriptor(g_dp_x[s]);
        cudnnDestroyFilterDescriptor(g_dp_w[s]);
        cudnnDestroyConvolutionDescriptor(g_dp_conv[s]);
        cudnnDestroyTensorDescriptor(g_dp_y[s]);
        cudnnDestroyTensorDescriptor(g_dp_b[s]);
        if (g_dp_ws[s] != 0ULL) {
            cuStreamSynchronize(cuda_stream_handle());
            cuMemFree_v2(g_dp_ws[s]);
        }
        g_dp_x[s] = (void*)0;
        g_dp_ws[s] = 0ULL;
    }
}

static void dnn_plan_release_(int s) {
    if (g_dp_used[s] == 0) return;
    dnb_free_(s);
    dnn_legacy_free_(s);
    g_dp_used[s] = 0;
}

// Cost in ms per call of the slot's graph plan (kind 2) or legacy plan
// (kind 1) on x -> y, for back-to-back calls: the larger of the host time
// to enqueue a call and the GPU time per call (between two stream events),
// which is what a queue of such calls takes on an idle GPU. Each is the
// best of several rounds, so a burst of other work on a shared GPU doesn't
// decide. Negative if a call fails.
static int dnn_run_(int s, int kind, const &CudaConv2D layer, unsigned long x, unsigned long y, int relu);

static double dnn_time_(int s, int kind, const &CudaConv2D layer, unsigned long x, unsigned long y, int relu) {
    void* stream = cuda_stream_handle();
    double bestHost = 1e30;
    double bestGpu = 1e30;
    unsafe {
        if (!dnn_run_(s, kind, layer, x, y, relu) || cuStreamSynchronize(stream) != 0) return -1.0;
        void* e0 = (void*)0;
        void* e1 = (void*)0;
        int ok = cuEventCreate(&e0, 0U) == 0 && cuEventCreate(&e1, 0U) == 0;
        int reps = 4;
        int round = 0;
        while (ok && round < 5) {
            ok = cuEventRecord(e0, stream) == 0;
            long long t0 = time_mono_ns();
            int r = 0;
            while (ok && r < reps) { ok = dnn_run_(s, kind, layer, x, y, relu); r = r + 1; }
            double host = time_ns_to_ms(time_mono_ns() - t0) / (double)reps;
            float gpu = 0.0f;
            ok = ok && cuEventRecord(e1, stream) == 0 && cuEventSynchronize(e1) == 0 &&
                 cuEventElapsedTime(&gpu, e0, e1) == 0;
            if (ok) {
                if (host < bestHost) bestHost = host;
                double g = (double)gpu / (double)reps;
                if (g < bestGpu) bestGpu = g;
                if ((double)gpu < 0.2 && reps < 64) reps = reps * 4;   // too short to time well
            }
            round = round + 1;
        }
        if (e0 != (void*)0) cuEventDestroy_v2(e0);
        if (e1 != (void*)0) cuEventDestroy_v2(e1);
        if (!ok) return -1.0;
    }
    return bestHost > bestGpu ? bestHost : bestGpu;
}

// ── Graph (backend) plans ────────────────────────────────────────────────────

// Finalized NCHW tensor descriptor (fp32 if virtual, else g_dnb_io), or null.
static void* dnb_tensor_(long uid, unsigned long n, unsigned long c, unsigned long h, unsigned long w, int isVirtual) {
    unsafe {
        void* d = (void*)0;
        if (cudnnBackendCreateDescriptor(DNB_TENSOR, &d) != 0) return (void*)0;
        long dims[4];
        long strides[4];
        dims[0] = (long)n; dims[1] = (long)c; dims[2] = (long)h; dims[3] = (long)w;
        strides[0] = (long)(c * h * w); strides[1] = (long)(h * w); strides[2] = (long)w; strides[3] = 1L;
        int dt = isVirtual ? DNN_DATA_FLOAT : g_dnb_io;
        long align = dt == DNN_DATA_FLOAT ? 4L : 16L;
        unsigned char virt = isVirtual ? (unsigned char)1 : (unsigned char)0;
        int ok = cudnnBackendSetAttribute(d, DNB_A_TENSOR_TYPE, DNB_T_DATA_TYPE, 1L, (const void*)&dt) == 0 &&
                 cudnnBackendSetAttribute(d, DNB_A_TENSOR_DIMS, DNB_T_INT64, 4L, (const void*)&dims[0]) == 0 &&
                 cudnnBackendSetAttribute(d, DNB_A_TENSOR_STRIDES, DNB_T_INT64, 4L, (const void*)&strides[0]) == 0 &&
                 cudnnBackendSetAttribute(d, DNB_A_TENSOR_UID, DNB_T_INT64, 1L, (const void*)&uid) == 0 &&
                 cudnnBackendSetAttribute(d, DNB_A_TENSOR_ALIGN, DNB_T_INT64, 1L, (const void*)&align) == 0 &&
                 cudnnBackendSetAttribute(d, DNB_A_TENSOR_VIRTUAL, DNB_T_BOOLEAN, 1L, (const void*)&virt) == 0 &&
                 cudnnBackendFinalize(d) == 0;
        if (!ok) { cudnnBackendDestroyDescriptor(d); return (void*)0; }
        return d;
    }
}

// Finalized pointwise op y = mode(x[, b]) in fp32; its pointwise descriptor
// goes to *pwOut (both kept alive by the plan). Null on failure.
static void* dnb_pointwise_(int mode, void* x, void* b, void* y, &void* pwOut) {
    unsafe {
        void* pw = (void*)0;
        void* op = (void*)0;
        int dt = DNN_DATA_FLOAT;
        int ok = cudnnBackendCreateDescriptor(DNB_POINTWISE, &pw) == 0;
        ok = ok && cudnnBackendSetAttribute(pw, DNB_A_PW_MODE, DNB_T_POINTWISE_MODE, 1L, (const void*)&mode) == 0 &&
             cudnnBackendSetAttribute(pw, DNB_A_PW_MATH_PREC, DNB_T_DATA_TYPE, 1L, (const void*)&dt) == 0 &&
             cudnnBackendFinalize(pw) == 0;
        ok = ok && cudnnBackendCreateDescriptor(DNB_OP_POINTWISE, &op) == 0;
        ok = ok && cudnnBackendSetAttribute(op, DNB_A_PWOP_DESC, DNB_T_DESCRIPTOR, 1L, (const void*)&pw) == 0 &&
             cudnnBackendSetAttribute(op, DNB_A_PWOP_X, DNB_T_DESCRIPTOR, 1L, (const void*)&x) == 0 &&
             (b == (void*)0 || cudnnBackendSetAttribute(op, DNB_A_PWOP_B, DNB_T_DESCRIPTOR, 1L, (const void*)&b) == 0) &&
             cudnnBackendSetAttribute(op, DNB_A_PWOP_Y, DNB_T_DESCRIPTOR, 1L, (const void*)&y) == 0 &&
             cudnnBackendFinalize(op) == 0;
        *pwOut = pw;
        if (!ok) {
            if (op != (void*)0) cudnnBackendDestroyDescriptor(op);
            return (void*)0;
        }
        return op;
    }
}

// Variant pack binding x, w, b, y (uids 1..4) and the workspace; null on failure.
static void* dnb_variant_pack_(unsigned long x, unsigned long w, unsigned long b, unsigned long y,
                               unsigned long long ws) {
    unsafe {
        void* vp = (void*)0;
        if (cudnnBackendCreateDescriptor(DNB_VARIANT_PACK, &vp) != 0) return (void*)0;
        long uids[4];
        void* ptrs[4];
        uids[0] = 1L; uids[1] = 2L; uids[2] = 3L; uids[3] = 4L;
        ptrs[0] = (void*)x; ptrs[1] = (void*)w; ptrs[2] = (void*)b; ptrs[3] = (void*)y;
        void* wsp = (void*)(unsigned long)ws;
        int ok = cudnnBackendSetAttribute(vp, DNB_A_VP_UIDS, DNB_T_INT64, 4L, (const void*)&uids[0]) == 0 &&
                 cudnnBackendSetAttribute(vp, DNB_A_VP_PTRS, DNB_T_VOID_PTR, 4L, (const void*)&ptrs[0]) == 0 &&
                 cudnnBackendSetAttribute(vp, DNB_A_VP_WORKSPACE, DNB_T_VOID_PTR, 1L, (const void*)&wsp) == 0 &&
                 cudnnBackendFinalize(vp) == 0;
        if (!ok) { cudnnBackendDestroyDescriptor(vp); return (void*)0; }
        return vp;
    }
}

// 1 if the engine behind 'cfg' is allowed in the current precision mode:
// without TF32, no tensor-core / down-converting / reduced-precision engines.
static int dnb_cfg_allowed_(void* cfg) {
    if (cuda_tf32_enabled() || g_dnb_io != DNN_DATA_FLOAT) return 1;   // 16-bit data: tensor cores are the point
    unsafe {
        void* eng = (void*)0;
        if (cudnnBackendCreateDescriptor(DNB_ENGINE, &eng) != 0) return 0;
        long got = 0L;
        int notes[16];
        long nn = 0L;
        int ok = cudnnBackendGetAttribute(cfg, DNB_A_CFG_ENGINE, DNB_T_DESCRIPTOR, 1L, &got, (void*)&eng) == 0 &&
                 cudnnBackendGetAttribute(eng, DNB_A_ENGINE_NOTES, DNB_T_NUMERICAL_NOTE, 16L, &nn,
                                          (void*)&notes[0]) == 0;
        cudnnBackendDestroyDescriptor(eng);
        if (!ok) return 0;
        long i = 0L;
        while (i < nn && i < 16L) {
            int n = notes[i];
            if (n == DNB_NOTE_TENSOR_CORE || n == DNB_NOTE_DOWN_CONVERT || n == DNB_NOTE_REDUCED_PREC) return 0;
            i = i + 1L;
        }
    }
    return 1;
}

// Builds a graph plan into slot 'slot' (descriptors in g_dp_bd, workspace,
// variant pack for these buffers): conv -> + bias (-> relu). Every engine
// config the heuristics return (up to DNB_MAX_CONFIGS) that suits the
// precision mode and workspace limit is finalized and timed on x/y; the
// fastest is kept. Returns its ms per call, or -1 if no config works
// (slot's graph half left empty).
static double dnb_build_(int slot, const &CudaConv2D layer, unsigned long batch, unsigned long height,
                      unsigned long width, unsigned long outH, unsigned long outW, int relu,
                      unsigned long x, unsigned long y) {
    unsafe {
        void* d[DNB_DESCS];
        int i = 0;
        while (i < DNB_DESCS) { d[i] = (void*)0; i = i + 1; }
        // d[0] plan (filled last), 1 x, 2 w, 3 b, 4 y, 5 conv out, 6 biased,
        // 7 conv desc, 8 conv op, 9/10 add pw/op, 11/12 relu pw/op, 13 graph
        d[1] = dnb_tensor_(1L, batch, layer.inChannels, height, width, 0);
        d[2] = dnb_tensor_(2L, layer.outChannels, layer.inChannels, layer.kH, layer.kW, 0);
        d[3] = dnb_tensor_(3L, 1UL, layer.outChannels, 1UL, 1UL, 0);
        d[4] = dnb_tensor_(4L, batch, layer.outChannels, outH, outW, 0);
        d[5] = dnb_tensor_(5L, batch, layer.outChannels, outH, outW, 1);
        if (relu) d[6] = dnb_tensor_(6L, batch, layer.outChannels, outH, outW, 1);
        int ok = d[1] != (void*)0 && d[2] != (void*)0 && d[3] != (void*)0 && d[4] != (void*)0 &&
                 d[5] != (void*)0 && (!relu || d[6] != (void*)0);
        long two = 2L;
        long pad[2];
        long stride[2];
        long dil[2];
        pad[0] = (long)layer.padding; pad[1] = (long)layer.padding;
        stride[0] = (long)layer.stride; stride[1] = (long)layer.stride;
        dil[0] = 1L; dil[1] = 1L;
        int ct = DNN_DATA_FLOAT;
        int cm = DNN_CROSS_CORRELATION;
        ok = ok && cudnnBackendCreateDescriptor(DNB_CONV, &d[7]) == 0;
        ok = ok && cudnnBackendSetAttribute(d[7], DNB_A_CONV_DIMS, DNB_T_INT64, 1L, (const void*)&two) == 0 &&
             cudnnBackendSetAttribute(d[7], DNB_A_CONV_COMP_TYPE, DNB_T_DATA_TYPE, 1L, (const void*)&ct) == 0 &&
             cudnnBackendSetAttribute(d[7], DNB_A_CONV_MODE, DNB_T_CONV_MODE, 1L, (const void*)&cm) == 0 &&
             cudnnBackendSetAttribute(d[7], DNB_A_CONV_PRE_PAD, DNB_T_INT64, 2L, (const void*)&pad[0]) == 0 &&
             cudnnBackendSetAttribute(d[7], DNB_A_CONV_POST_PAD, DNB_T_INT64, 2L, (const void*)&pad[0]) == 0 &&
             cudnnBackendSetAttribute(d[7], DNB_A_CONV_DILATIONS, DNB_T_INT64, 2L, (const void*)&dil[0]) == 0 &&
             cudnnBackendSetAttribute(d[7], DNB_A_CONV_STRIDES, DNB_T_INT64, 2L, (const void*)&stride[0]) == 0 &&
             cudnnBackendFinalize(d[7]) == 0;
        float one = 1.0f;
        float zero = 0.0f;
        ok = ok && cudnnBackendCreateDescriptor(DNB_OP_CONV_FWD, &d[8]) == 0;
        ok = ok && cudnnBackendSetAttribute(d[8], DNB_A_FWD_ALPHA, DNB_T_FLOAT, 1L, (const void*)&one) == 0 &&
             cudnnBackendSetAttribute(d[8], DNB_A_FWD_BETA, DNB_T_FLOAT, 1L, (const void*)&zero) == 0 &&
             cudnnBackendSetAttribute(d[8], DNB_A_FWD_CONV, DNB_T_DESCRIPTOR, 1L, (const void*)&d[7]) == 0 &&
             cudnnBackendSetAttribute(d[8], DNB_A_FWD_X, DNB_T_DESCRIPTOR, 1L, (const void*)&d[1]) == 0 &&
             cudnnBackendSetAttribute(d[8], DNB_A_FWD_W, DNB_T_DESCRIPTOR, 1L, (const void*)&d[2]) == 0 &&
             cudnnBackendSetAttribute(d[8], DNB_A_FWD_Y, DNB_T_DESCRIPTOR, 1L, (const void*)&d[5]) == 0 &&
             cudnnBackendFinalize(d[8]) == 0;
        if (ok) {
            d[10] = dnb_pointwise_(DNB_PW_ADD, d[5], d[3], relu ? d[6] : d[4], &d[9]);
            ok = d[10] != (void*)0;
        }
        if (ok && relu) {
            d[12] = dnb_pointwise_(DNB_PW_RELU, d[6], (void*)0, d[4], &d[11]);
            ok = d[12] != (void*)0;
        }
        void* ops[3];
        ops[0] = d[8]; ops[1] = d[10]; ops[2] = d[12];
        long nops = relu ? 3L : 2L;
        ok = ok && cudnnBackendCreateDescriptor(DNB_OPGRAPH, &d[13]) == 0;
        ok = ok && cudnnBackendSetAttribute(d[13], DNB_A_GRAPH_OPS, DNB_T_DESCRIPTOR, nops, (const void*)&ops[0]) == 0 &&
             cudnnBackendSetAttribute(d[13], DNB_A_GRAPH_HANDLE, DNB_T_HANDLE, 1L, (const void*)&g_dnn) == 0 &&
             cudnnBackendFinalize(d[13]) == 0;

        // Heuristic engine configs, best guess first.
        void* heur = (void*)0;
        void* cfgs[DNB_MAX_CONFIGS];
        long ncfg = 0L;
        i = 0;
        while (i < DNB_MAX_CONFIGS) { cfgs[i] = (void*)0; i = i + 1; }
        int hm = DNB_HEUR_A;
        ok = ok && cudnnBackendCreateDescriptor(DNB_ENGINEHEUR, &heur) == 0;
        ok = ok && cudnnBackendSetAttribute(heur, DNB_A_HEUR_GRAPH, DNB_T_DESCRIPTOR, 1L, (const void*)&d[13]) == 0 &&
             cudnnBackendSetAttribute(heur, DNB_A_HEUR_MODE, DNB_T_HEUR_MODE, 1L, (const void*)&hm) == 0 &&
             cudnnBackendFinalize(heur) == 0;
        i = 0;
        while (ok && i < DNB_MAX_CONFIGS) {
            if (cudnnBackendCreateDescriptor(DNB_ENGINECFG, &cfgs[i]) != 0) ok = 0;
            i = i + 1;
        }
        ok = ok && cudnnBackendGetAttribute(heur, DNB_A_HEUR_RESULTS, DNB_T_DESCRIPTOR, (long)DNB_MAX_CONFIGS,
                                            &ncfg, (void*)&cfgs[0]) == 0;
        if (ncfg > (long)DNB_MAX_CONFIGS) ncfg = (long)DNB_MAX_CONFIGS;

        // Time each usable config; keep the fastest plan and its workspace.
        // (dnn_run_ then finds the candidate's variant pack already bound.)
        g_dp_ptr[slot * 4] = x;
        g_dp_ptr[slot * 4 + 1] = layer.weight.dev;
        g_dp_ptr[slot * 4 + 2] = layer.bias.dev;
        g_dp_ptr[slot * 4 + 3] = y;
        void* best = (void*)0;
        unsigned long long bestWs = 0ULL;
        double bestMs = 1e30;
        long c = 0L;
        while (ok && c < ncfg) {
            void* plan = (void*)0;
            long wsSize = 0L;
            long got = 0L;
            int usable = dnb_cfg_allowed_(cfgs[c]) && cudnnBackendCreateDescriptor(DNB_PLAN, &plan) == 0;
            usable = usable && cudnnBackendSetAttribute(plan, DNB_A_PLAN_HANDLE, DNB_T_HANDLE, 1L, (const void*)&g_dnn) == 0 &&
                     cudnnBackendSetAttribute(plan, DNB_A_PLAN_CFG, DNB_T_DESCRIPTOR, 1L, (const void*)&cfgs[c]) == 0 &&
                     cudnnBackendFinalize(plan) == 0 &&
                     cudnnBackendGetAttribute(plan, DNB_A_PLAN_WS_SIZE, DNB_T_INT64, 1L, &got, (void*)&wsSize) == 0 &&
                     wsSize >= 0L && (unsigned long)wsSize <= DNN_MAX_WORKSPACE;
            unsigned long long ws = 0ULL;
            if (usable && wsSize > 0L) usable = cuMemAlloc_v2(&ws, (size_t)wsSize) == 0;
            void* vp = (void*)0;
            if (usable) {
                vp = dnb_variant_pack_(x, layer.weight.dev, layer.bias.dev, y, ws);
                usable = vp != (void*)0;
            }
            if (usable) {
                g_dp_bd[slot * DNB_DESCS] = plan;   // dnn_run_ executes the slot's plan
                g_dp_vp[slot] = vp;
                double ms = dnn_time_(slot, 2, layer, x, y, relu);
                g_dp_bd[slot * DNB_DESCS] = (void*)0;
                g_dp_vp[slot] = (void*)0;
                if (ms >= 0.0 && ms < bestMs) {
                    if (best != (void*)0) cudnnBackendDestroyDescriptor(best);
                    if (bestWs != 0ULL) cuMemFree_v2(bestWs);
                    best = plan;
                    bestWs = ws;
                    bestMs = ms;
                    plan = (void*)0;
                    ws = 0ULL;
                }
            }
            if (vp != (void*)0) cudnnBackendDestroyDescriptor(vp);
            if (plan != (void*)0) cudnnBackendDestroyDescriptor(plan);
            if (ws != 0ULL) cuMemFree_v2(ws);
            c = c + 1L;
        }
        if (heur != (void*)0) cudnnBackendDestroyDescriptor(heur);
        i = 0;
        while (i < DNB_MAX_CONFIGS) {
            if (cfgs[i] != (void*)0) cudnnBackendDestroyDescriptor(cfgs[i]);
            i = i + 1;
        }
        void* vp = (void*)0;
        if (best != (void*)0) vp = dnb_variant_pack_(x, layer.weight.dev, layer.bias.dev, y, bestWs);
        if (vp == (void*)0) {
            if (best != (void*)0) cudnnBackendDestroyDescriptor(best);
            if (bestWs != 0ULL) cuMemFree_v2(bestWs);
            i = 0;
            while (i < DNB_DESCS) { if (d[i] != (void*)0) cudnnBackendDestroyDescriptor(d[i]); i = i + 1; }
            return -1.0;
        }
        d[0] = best;
        i = 0;
        while (i < DNB_DESCS) { g_dp_bd[slot * DNB_DESCS + i] = d[i]; i = i + 1; }
        g_dp_vp[slot] = vp;
        g_dp_gws[slot] = bestWs;
        return bestMs;
    }
}

// ── Legacy plans ─────────────────────────────────────────────────────────────

// Builds a legacy-API plan into 'slot'; 1 on success.
static int dnn_legacy_build_(int slot, const &CudaConv2D layer, unsigned long batch, unsigned long height,
                             unsigned long width, unsigned long outH, unsigned long outW) {
    unsafe {
        void* xd = (void*)0;
        void* wd = (void*)0;
        void* cd = (void*)0;
        void* yd = (void*)0;
        void* bd = (void*)0;
        int ok = cudnnCreateTensorDescriptor(&xd) == 0 && cudnnCreateFilterDescriptor(&wd) == 0 &&
                 cudnnCreateConvolutionDescriptor(&cd) == 0 && cudnnCreateTensorDescriptor(&yd) == 0 &&
                 cudnnCreateTensorDescriptor(&bd) == 0;
        ok = ok && cudnnSetTensor4dDescriptor(xd, DNN_TENSOR_NCHW, DNN_DATA_FLOAT, (int)batch,
                                              (int)layer.inChannels, (int)height, (int)width) == 0;
        ok = ok && cudnnSetFilter4dDescriptor(wd, DNN_DATA_FLOAT, DNN_TENSOR_NCHW, (int)layer.outChannels,
                                              (int)layer.inChannels, (int)layer.kH, (int)layer.kW) == 0;
        ok = ok && cudnnSetConvolution2dDescriptor(cd, (int)layer.padding, (int)layer.padding,
                                                   (int)layer.stride, (int)layer.stride, 1, 1,
                                                   DNN_CROSS_CORRELATION, DNN_DATA_FLOAT) == 0;
        int math = cuda_tf32_enabled() ? DNN_DEFAULT_MATH : DNN_FMA_MATH;
        ok = ok && cudnnSetConvolutionMathType(cd, math) == 0;
        ok = ok && cudnnSetTensor4dDescriptor(yd, DNN_TENSOR_NCHW, DNN_DATA_FLOAT, (int)batch,
                                              (int)layer.outChannels, (int)outH, (int)outW) == 0;
        ok = ok && cudnnSetTensor4dDescriptor(bd, DNN_TENSOR_NCHW, DNN_DATA_FLOAT, 1,
                                              (int)layer.outChannels, 1, 1) == 0;
        // Time every algorithm (results come back fastest first) and keep
        // the fastest that succeeded within the workspace limit.
        int perf[DNN_PERF_INTS * DNN_PERF_COUNT];
        int returned = 0;
        int algo = -1;
        int algoMath = math;
        unsigned long wsBytes = 0UL;
        ok = ok && cudnnFindConvolutionForwardAlgorithm(g_dnn, xd, wd, cd, yd, DNN_PERF_COUNT, &returned,
                                                        (void*)&perf[0]) == 0;
        int r = 0;
        while (ok && algo < 0 && r < returned) {
            int* p = (int*)&perf[r * DNN_PERF_INTS];
            unsigned long mem = *(unsigned long*)(void*)&p[4];
            if (p[1] == 0 && mem <= DNN_MAX_WORKSPACE) {
                algo = p[0];
                wsBytes = mem;
                algoMath = p[7];
            }
            r = r + 1;
        }
        // With DEFAULT_MATH cuDNN reports whether the winner uses tensor
        // cores; set that on the descriptor, as cuDNN asks. With FMA_MATH
        // it can only report FMA.
        ok = ok && algo >= 0 && cudnnSetConvolutionMathType(cd, algoMath) == 0;
        unsigned long long ws = 0ULL;
        if (ok && wsBytes > 0UL) ok = cuMemAlloc_v2(&ws, wsBytes) == 0;
        if (!ok) {
            if (xd != (void*)0) cudnnDestroyTensorDescriptor(xd);
            if (wd != (void*)0) cudnnDestroyFilterDescriptor(wd);
            if (cd != (void*)0) cudnnDestroyConvolutionDescriptor(cd);
            if (yd != (void*)0) cudnnDestroyTensorDescriptor(yd);
            if (bd != (void*)0) cudnnDestroyTensorDescriptor(bd);
            return 0;
        }
        g_dp_x[slot] = xd;
        g_dp_w[slot] = wd;
        g_dp_conv[slot] = cd;
        g_dp_y[slot] = yd;
        g_dp_b[slot] = bd;
        g_dp_algo[slot] = algo;
        g_dp_ws[slot] = ws;
        g_dp_wsBytes[slot] = wsBytes;
    }
    return 1;
}


// Finds or builds the plan for this convolution shape (relu: the fused
// conv + bias + relu graph); x and y are the buffers a new plan is timed
// on. dt is the tensors' data type: fp32 builds a graph and a legacy plan
// and keeps the faster; 16-bit builds only a graph plan, and a shape no
// graph engine takes is remembered as g_dp_used = 3 (the caller then uses
// its own legacy path). -1 on failure.
static int dnn_plan_(const &CudaConv2D layer, unsigned long batch, unsigned long height, unsigned long width,
                     int relu, unsigned long x, unsigned long y, int dt) {
    unsigned long outH = cuda_conv_out_dim(height, layer.kH, layer.stride, layer.padding);
    unsigned long outW = cuda_conv_out_dim(width, layer.kW, layer.stride, layer.padding);
    if (!dnn_int_ok_(batch) || !dnn_int_ok_(height) || !dnn_int_ok_(width) || !dnn_int_ok_(outH) ||
        !dnn_int_ok_(outW) || !dnn_int_ok_(layer.inChannels) || !dnn_int_ok_(layer.outChannels) ||
        !dnn_int_ok_(layer.kH) || !dnn_int_ok_(layer.kW) || layer.stride > 65535UL || layer.padding > 65535UL)
        return -1;
    unsigned long key[8];
    key[0] = batch; key[1] = layer.inChannels; key[2] = height; key[3] = width;
    key[4] = layer.outChannels; key[5] = layer.kH; key[6] = layer.kW;
    key[7] = layer.stride | (layer.padding << 16) | ((unsigned long)cuda_tf32_enabled() << 32) |
             ((unsigned long)(relu ? 1 : 0) << 33) | ((unsigned long)dt << 34);
    int slot = -1;
    int s = 0;
    while (s < DNN_PLANS) {
        if (g_dp_used[s] != 0) {
            int same = 1;
            int i = 0;
            while (i < 8) { if (g_dp_key[s * 8 + i] != key[i]) same = 0; i = i + 1; }
            if (same) return s;
        } else if (slot < 0) {
            slot = s;
        }
        s = s + 1;
    }
    if (slot < 0) {
        slot = g_dp_evict;
        g_dp_evict = (g_dp_evict + 1) % DNN_PLANS;
        dnn_plan_release_(slot);
    }
    if (dt != DNN_DATA_FLOAT) {
        g_dnb_io = dt;
        double ms = dnb_build_(slot, layer, batch, height, width, outH, outW, relu, x, y);
        g_dnb_io = DNN_DATA_FLOAT;
        g_dp_used[slot] = ms >= 0.0 ? 2 : 3;
        int k = 0;
        while (k < 8) { g_dp_key[slot * 8 + k] = key[k]; k = k + 1; }
        return slot;
    }
    // Build both kinds, time them on these buffers, keep the faster.
    double graphMs = dnb_build_(slot, layer, batch, height, width, outH, outW, relu, x, y);
    double legacyMs = -1.0;
    if (dnn_legacy_build_(slot, layer, batch, height, width, outH, outW)) {
        legacyMs = dnn_time_(slot, 1, layer, x, y, relu);
        if (legacyMs < 0.0) dnn_legacy_free_(slot);
    }
    if (graphMs < 0.0 && legacyMs < 0.0) return -1;
    if (graphMs >= 0.0 && (legacyMs < 0.0 || graphMs <= legacyMs)) {
        dnn_legacy_free_(slot);
        g_dp_used[slot] = 2;
    } else {
        dnb_free_(slot);
        g_dp_used[slot] = 1;
    }
    int i = 0;
    while (i < 8) { g_dp_key[slot * 8 + i] = key[i]; i = i + 1; }
    return slot;
}

// One convolution with slot s's graph plan (kind 2) or legacy plan (kind 1).
static int dnn_run_(int s, int kind, const &CudaConv2D layer, unsigned long x, unsigned long y, int relu) {
    float one = 1.0f;
    float zero = 0.0f;
    unsafe {
        if (kind == 2) {
            // Rebind the variant pack only when the buffers changed.
            unsigned long* bound = (unsigned long*)&g_dp_ptr[s * 4];
            if (bound[0] != x || bound[1] != layer.weight.dev || bound[2] != layer.bias.dev || bound[3] != y) {
                void* vp = dnb_variant_pack_(x, layer.weight.dev, layer.bias.dev, y, g_dp_gws[s]);
                if (vp == (void*)0) return 0;
                if (g_dp_vp[s] != (void*)0) cudnnBackendDestroyDescriptor(g_dp_vp[s]);
                g_dp_vp[s] = vp;
                bound[0] = x; bound[1] = layer.weight.dev; bound[2] = layer.bias.dev; bound[3] = y;
            }
            return cudnnBackendExecute(g_dnn, g_dp_bd[s * DNB_DESCS], g_dp_vp[s]) == 0;
        }
        void* ws = (void*)(unsigned long)g_dp_ws[s];
        if (relu) {
            // y = relu(1 * conv(x) + 0 * z + bias), z aliased to y
            return cudnnConvolutionBiasActivationForward(g_dnn, (const void*)&one, g_dp_x[s], (const void*)x,
                       g_dp_w[s], (const void*)layer.weight.dev, g_dp_conv[s], g_dp_algo[s], ws, g_dp_wsBytes[s],
                       (const void*)&zero, g_dp_y[s], (const void*)y, g_dp_b[s], (const void*)layer.bias.dev,
                       g_dnn_relu, g_dp_y[s], (void*)y) == 0;
        }
        // (BiasActivation with an identity activation is limited to one
        // algorithm, so the plain case is convolution + broadcast bias add.)
        if (cudnnConvolutionForward(g_dnn, (const void*)&one, g_dp_x[s], (const void*)x,
                g_dp_w[s], (const void*)layer.weight.dev, g_dp_conv[s], g_dp_algo[s], ws, g_dp_wsBytes[s],
                (const void*)&zero, g_dp_y[s], (void*)y) != 0) return 0;
        return cudnnAddTensor(g_dnn, (const void*)&one, g_dp_b[s], (const void*)layer.bias.dev,
                              (const void*)&one, g_dp_y[s], (void*)y) == 0;
    }
}

static int dnn_conv_(const &CudaConv2D layer, struct CudaBuf x, struct CudaBuf y,
                     unsigned long batch, unsigned long height, unsigned long width, int relu) {
    if (!cudnn_ensure_() || layer.weight.dev == 0UL || layer.bias.dev == 0UL) return 0;
    unsigned long outH = cuda_conv_out_dim(height, layer.kH, layer.stride, layer.padding);
    unsigned long outW = cuda_conv_out_dim(width, layer.kW, layer.stride, layer.padding);
    if (x.dev == 0UL || y.dev == 0UL ||
        x.len < batch * layer.inChannels * height * width ||
        y.len < batch * layer.outChannels * outH * outW) return 0;
    int p = dnn_plan_(layer, batch, height, width, relu, x.dev, y.dev, DNN_DATA_FLOAT);
    if (p < 0) return 0;
    return dnn_run_(p, g_dp_used[p], layer, x.dev, y.dev, relu);
}

int cuda_conv2d(const &CudaConv2D layer, struct CudaBuf x, struct CudaBuf y,
                unsigned long batch, unsigned long height, unsigned long width) {
    return dnn_conv_(layer, x, y, batch, height, width, 0);
}

int cuda_conv2d_relu(const &CudaConv2D layer, struct CudaBuf x, struct CudaBuf y,
                     unsigned long batch, unsigned long height, unsigned long width) {
    return dnn_conv_(layer, x, y, batch, height, width, 1);
}

static int dnn_pool_(int mode, struct CudaBuf x, struct CudaBuf y, unsigned long batch, unsigned long channels,
                     unsigned long height, unsigned long width, unsigned long kernel, unsigned long stride) {
    if (!cudnn_ensure_()) return 0;
    unsigned long outH = cuda_conv_out_dim(height, kernel, stride, 0UL);
    unsigned long outW = cuda_conv_out_dim(width, kernel, stride, 0UL);
    if (!dnn_int_ok_(batch) || !dnn_int_ok_(channels) || !dnn_int_ok_(height) || !dnn_int_ok_(width) ||
        !dnn_int_ok_(kernel) || !dnn_int_ok_(stride) || outH == 0UL || outW == 0UL ||
        x.dev == 0UL || y.dev == 0UL ||
        x.len < batch * channels * height * width || y.len < batch * channels * outH * outW) return 0;
    float one = 1.0f;
    float zero = 0.0f;
    unsafe {
        if (cudnnSetPooling2dDescriptor(g_dnn_pool, mode, DNN_NOT_PROPAGATE_NAN, (int)kernel, (int)kernel,
                                        0, 0, (int)stride, (int)stride) != 0 ||
            cudnnSetTensor4dDescriptor(g_dnn_pool_x, DNN_TENSOR_NCHW, DNN_DATA_FLOAT, (int)batch,
                                       (int)channels, (int)height, (int)width) != 0 ||
            cudnnSetTensor4dDescriptor(g_dnn_pool_y, DNN_TENSOR_NCHW, DNN_DATA_FLOAT, (int)batch,
                                       (int)channels, (int)outH, (int)outW) != 0) return 0;
        return cudnnPoolingForward(g_dnn, g_dnn_pool, (const void*)&one, g_dnn_pool_x, (const void*)x.dev,
                                   (const void*)&zero, g_dnn_pool_y, (void*)y.dev) == 0;
    }
}

int cuda_maxpool2d(struct CudaBuf x, struct CudaBuf y, unsigned long batch, unsigned long channels,
                   unsigned long height, unsigned long width, unsigned long kernel, unsigned long stride) {
    return dnn_pool_(DNN_POOL_MAX, x, y, batch, channels, height, width, kernel, stride);
}

int cuda_avgpool2d(struct CudaBuf x, struct CudaBuf y, unsigned long batch, unsigned long channels,
                   unsigned long height, unsigned long width, unsigned long kernel, unsigned long stride) {
    return dnn_pool_(DNN_POOL_AVG, x, y, batch, channels, height, width, kernel, stride);
}

int cuda_concat_channels(struct CudaBuf a, unsigned long aChannels, struct CudaBuf c, unsigned long cChannels,
                         struct CudaBuf out, unsigned long batch, unsigned long plane) {
    unsigned long na = aChannels * plane;
    unsigned long nc = cChannels * plane;
    if (batch == 0UL || na == 0UL || nc == 0UL || a.len < batch * na || c.len < batch * nc ||
        out.len < batch * (na + nc)) return 0;
    // two strided copies (one kernel each), not one memcpy per image and input
    return cuda_copy_rows(a, 0UL, na, out, 0UL, na + nc, na, batch) &&
           cuda_copy_rows(c, 0UL, nc, out, na, na + nc, nc, batch);
}

// ── fp16 / bf16 ──────────────────────────────────────────────────────────────
// Legacy-API plans for 16-bit data with fp32 compute, kept apart from the
// fp32 plans above (keyed by shape and dtype; fastest algorithm cuDNN finds
// that fits the workspace limit). The bias and ReLU are gpu_cuda's fused
// 16-bit kernel.

static unsigned long g_dh_key[DNN_PLANS * 8];
static int   g_dh_used[DNN_PLANS];
static void* g_dh_x[DNN_PLANS];
static void* g_dh_w[DNN_PLANS];
static void* g_dh_conv[DNN_PLANS];
static void* g_dh_y[DNN_PLANS];
static int   g_dh_algo[DNN_PLANS];
static unsigned long long g_dh_ws[DNN_PLANS];
static unsigned long g_dh_wsBytes[DNN_PLANS];
static int   g_dh_evict = 0;

static void dnn_half_release_(int s) {
    unsafe {
        if (g_dh_used[s] == 0) return;
        cudnnDestroyTensorDescriptor(g_dh_x[s]);
        cudnnDestroyFilterDescriptor(g_dh_w[s]);
        cudnnDestroyConvolutionDescriptor(g_dh_conv[s]);
        cudnnDestroyTensorDescriptor(g_dh_y[s]);
        if (g_dh_ws[s] != 0ULL) {
            cuStreamSynchronize(cuda_stream_handle());
            cuMemFree_v2(g_dh_ws[s]);
        }
        g_dh_ws[s] = 0ULL;
        g_dh_used[s] = 0;
    }
}

static int dnn_half_plan_(const &CudaHalfConv2D layer, int dt, unsigned long batch, unsigned long height,
                          unsigned long width) {
    unsigned long outH = cuda_conv_out_dim(height, layer.kH, layer.stride, layer.padding);
    unsigned long outW = cuda_conv_out_dim(width, layer.kW, layer.stride, layer.padding);
    if (!dnn_int_ok_(batch) || !dnn_int_ok_(height) || !dnn_int_ok_(width) || !dnn_int_ok_(outH) ||
        !dnn_int_ok_(outW) || !dnn_int_ok_(layer.inChannels) || !dnn_int_ok_(layer.outChannels) ||
        !dnn_int_ok_(layer.kH) || !dnn_int_ok_(layer.kW) || layer.stride > 65535UL || layer.padding > 65535UL)
        return -1;
    unsigned long key[8];
    key[0] = batch; key[1] = layer.inChannels; key[2] = height; key[3] = width;
    key[4] = layer.outChannels; key[5] = layer.kH; key[6] = layer.kW;
    key[7] = layer.stride | (layer.padding << 16) | ((unsigned long)dt << 32);
    int slot = -1;
    int s = 0;
    while (s < DNN_PLANS) {
        if (g_dh_used[s] != 0) {
            int same = 1;
            int i = 0;
            while (i < 8) { if (g_dh_key[s * 8 + i] != key[i]) same = 0; i = i + 1; }
            if (same) return s;
        } else if (slot < 0) {
            slot = s;
        }
        s = s + 1;
    }
    if (slot < 0) {
        slot = g_dh_evict;
        g_dh_evict = (g_dh_evict + 1) % DNN_PLANS;
        dnn_half_release_(slot);
    }
    unsafe {
        void* xd = (void*)0;
        void* wd = (void*)0;
        void* cd = (void*)0;
        void* yd = (void*)0;
        int ok = cudnnCreateTensorDescriptor(&xd) == 0 && cudnnCreateFilterDescriptor(&wd) == 0 &&
                 cudnnCreateConvolutionDescriptor(&cd) == 0 && cudnnCreateTensorDescriptor(&yd) == 0;
        ok = ok && cudnnSetTensor4dDescriptor(xd, DNN_TENSOR_NCHW, dt, (int)batch, (int)layer.inChannels,
                                              (int)height, (int)width) == 0;
        ok = ok && cudnnSetFilter4dDescriptor(wd, dt, DNN_TENSOR_NCHW, (int)layer.outChannels,
                                              (int)layer.inChannels, (int)layer.kH, (int)layer.kW) == 0;
        // 16-bit tensors, fp32 accumulation, tensor cores allowed
        ok = ok && cudnnSetConvolution2dDescriptor(cd, (int)layer.padding, (int)layer.padding,
                                                   (int)layer.stride, (int)layer.stride, 1, 1,
                                                   DNN_CROSS_CORRELATION, DNN_DATA_FLOAT) == 0;
        ok = ok && cudnnSetConvolutionMathType(cd, DNN_TENSOR_OP_MATH) == 0;
        ok = ok && cudnnSetTensor4dDescriptor(yd, DNN_TENSOR_NCHW, dt, (int)batch, (int)layer.outChannels,
                                              (int)outH, (int)outW) == 0;
        int perf[DNN_PERF_INTS * DNN_PERF_COUNT];
        int returned = 0;
        int algo = -1;
        int algoMath = DNN_TENSOR_OP_MATH;
        unsigned long wsBytes = 0UL;
        ok = ok && cudnnFindConvolutionForwardAlgorithm(g_dnn, xd, wd, cd, yd, DNN_PERF_COUNT, &returned,
                                                        (void*)&perf[0]) == 0;
        int r = 0;
        while (ok && algo < 0 && r < returned) {
            int* p = (int*)&perf[r * DNN_PERF_INTS];
            unsigned long mem = *(unsigned long*)(void*)&p[4];
            if (p[1] == 0 && mem <= DNN_MAX_WORKSPACE) {
                algo = p[0];
                wsBytes = mem;
                algoMath = p[7];
            }
            r = r + 1;
        }
        ok = ok && algo >= 0 && cudnnSetConvolutionMathType(cd, algoMath) == 0;
        unsigned long long ws = 0ULL;
        if (ok && wsBytes > 0UL) ok = cuMemAlloc_v2(&ws, wsBytes) == 0;
        if (!ok) {
            if (xd != (void*)0) cudnnDestroyTensorDescriptor(xd);
            if (wd != (void*)0) cudnnDestroyFilterDescriptor(wd);
            if (cd != (void*)0) cudnnDestroyConvolutionDescriptor(cd);
            if (yd != (void*)0) cudnnDestroyTensorDescriptor(yd);
            return -1;
        }
        g_dh_x[slot] = xd;
        g_dh_w[slot] = wd;
        g_dh_conv[slot] = cd;
        g_dh_y[slot] = yd;
        g_dh_algo[slot] = algo;
        g_dh_ws[slot] = ws;
        g_dh_wsBytes[slot] = wsBytes;
        int i = 0;
        while (i < 8) { g_dh_key[slot * 8 + i] = key[i]; i = i + 1; }
        g_dh_used[slot] = 1;
    }
    return slot;
}

static int dnn_half_conv_(const &CudaHalfConv2D layer, struct CudaHalfBuf x, struct CudaHalfBuf y,
                          unsigned long batch, unsigned long height, unsigned long width, int relu) {
    if (!cudnn_ensure_() || layer.weight.dev == 0UL || layer.bias.dev == 0UL) return 0;
    if (x.dtype != layer.weight.dtype || y.dtype != layer.weight.dtype) return 0;
    unsigned long outH = cuda_conv_out_dim(height, layer.kH, layer.stride, layer.padding);
    unsigned long outW = cuda_conv_out_dim(width, layer.kW, layer.stride, layer.padding);
    if (x.dev == 0UL || y.dev == 0UL ||
        x.len < batch * layer.inChannels * height * width ||
        y.len < batch * layer.outChannels * outH * outW) return 0;
    // A 1x1, stride-1, unpadded convolution is y[b] = W x[b] for each image
    // ([outC, inC] times [inC, H*W]): one strided-batched cuBLAS GEMM with W
    // shared, then the fused bias (+ relu). cuDNN's engines for this shape
    // were about half as fast on an H200.
    if (layer.kH == 1UL && layer.kW == 1UL && layer.stride == 1UL && layer.padding == 0UL) {
        unsigned long plane = height * width;
        return cuda_half_gemm_strided(layer.weight, 0, 0UL, x, 0, layer.inChannels * plane, y,
                                      layer.outChannels * plane, layer.outChannels, plane, layer.inChannels, batch) &&
               cuda_half_bias(y, layer.bias, batch, layer.outChannels, plane, relu);
    }
    int dt = x.dtype == CUDA_BF16 ? DNN_DATA_BFLOAT16 : DNN_DATA_HALF;
    // The fused conv + bias (+ relu) graph, as for fp32, when the buffers
    // meet its 16-byte alignment.
    if ((x.dev | y.dev | layer.weight.dev | layer.bias.dev) % 16UL == 0UL) {
        struct CudaConv2D view;
        view.weight.dev = layer.weight.dev; view.weight.len = layer.weight.len;
        view.bias.dev = layer.bias.dev; view.bias.len = layer.bias.len;
        view.inChannels = layer.inChannels; view.outChannels = layer.outChannels;
        view.kH = layer.kH; view.kW = layer.kW; view.stride = layer.stride; view.padding = layer.padding;
        int g = dnn_plan_(&view, batch, height, width, relu, x.dev, y.dev, dt);
        if (g >= 0 && g_dp_used[g] == 2) return dnn_run_(g, 2, &view, x.dev, y.dev, relu);
    }
    int p = dnn_half_plan_(layer, dt, batch, height, width);
    if (p < 0) return 0;
    float one = 1.0f;
    float zero = 0.0f;
    unsafe {
        if (cudnnConvolutionForward(g_dnn, (const void*)&one, g_dh_x[p], (const void*)x.dev,
                g_dh_w[p], (const void*)layer.weight.dev, g_dh_conv[p], g_dh_algo[p],
                (void*)(unsigned long)g_dh_ws[p], g_dh_wsBytes[p],
                (const void*)&zero, g_dh_y[p], (void*)y.dev) != 0) return 0;
    }
    return cuda_half_bias(y, layer.bias, batch, layer.outChannels, outH * outW, relu);
}

int cuda_half_conv2d(const &CudaHalfConv2D layer, struct CudaHalfBuf x, struct CudaHalfBuf y,
                     unsigned long batch, unsigned long height, unsigned long width) {
    return dnn_half_conv_(layer, x, y, batch, height, width, 0);
}

int cuda_half_conv2d_relu(const &CudaHalfConv2D layer, struct CudaHalfBuf x, struct CudaHalfBuf y,
                          unsigned long batch, unsigned long height, unsigned long width) {
    return dnn_half_conv_(layer, x, y, batch, height, width, 1);
}

struct CudaHalfConv2D cuda_half_conv2d_upload(const &Conv2D layer, int dtype) {
    struct CudaHalfConv2D d;
    d.inChannels = layer.inChannels;
    d.outChannels = layer.outChannels;
    d.kH = layer.kH;
    d.kW = layer.kW;
    d.stride = layer.stride;
    d.padding = layer.padding;
    unsigned long wn = layer.outChannels * layer.inChannels * layer.kH * layer.kW;
    d.weight = cuda_half_upload(layer.weight, wn, dtype);
    d.bias = cuda_half_upload(layer.bias, layer.outChannels, dtype);
    if (d.weight.dev == 0UL || d.bias.dev == 0UL) {
        cuda_half_free(&d.weight);
        cuda_half_free(&d.bias);
    }
    return d;
}

void cuda_half_conv2d_free(&CudaHalfConv2D layer) {
    cuda_half_free(&layer.weight);
    cuda_half_free(&layer.bias);
}

// ── Layers ───────────────────────────────────────────────────────────────────

struct CudaConv2D cuda_conv2d_upload(const &Conv2D layer) {
    struct CudaConv2D d;
    d.inChannels = layer.inChannels;
    d.outChannels = layer.outChannels;
    d.kH = layer.kH;
    d.kW = layer.kW;
    d.stride = layer.stride;
    d.padding = layer.padding;
    unsigned long wn = layer.outChannels * layer.inChannels * layer.kH * layer.kW;
    d.weight = cuda_upload(layer.weight, wn);
    d.bias = cuda_upload(layer.bias, layer.outChannels);
    if (d.weight.dev == 0UL || d.bias.dev == 0UL) {
        cuda_free(&d.weight);
        cuda_free(&d.bias);
    }
    return d;
}

void cuda_conv2d_free(&CudaConv2D layer) {
    cuda_free(&layer.weight);
    cuda_free(&layer.bias);
}

// ── U-Net ────────────────────────────────────────────────────────────────────

struct CudaUNet cuda_unet_new(const &UNet net, unsigned long batch, unsigned long height, unsigned long width) {
    struct CudaUNet g;
    g.batch = batch;
    g.height = height;
    g.width = width;
    g.encConv1 = cuda_conv2d_upload(&net.encConv1);
    g.encConv2 = cuda_conv2d_upload(&net.encConv2);
    g.bottleneck = cuda_conv2d_upload(&net.bottleneck);
    g.decConv1 = cuda_conv2d_upload(&net.decConv1);
    g.decConv2 = cuda_conv2d_upload(&net.decConv2);
    unsigned long c1 = net.encConv1.outChannels;
    unsigned long c2 = net.encConv2.outChannels;
    unsigned long hw = height * width;
    unsigned long hw2 = hw / 4UL;
    unsigned long hw4 = hw / 16UL;
    unsigned long upN = c2 * hw2 > c1 * hw ? c2 * hw2 : c1 * hw;
    g.e1 = cuda_alloc(batch * c1 * hw);
    g.p1 = cuda_alloc(batch * c1 * hw2);
    g.e2 = cuda_alloc(batch * c2 * hw2);
    g.p2 = cuda_alloc(batch * c2 * hw4);
    g.mid = cuda_alloc(batch * c2 * hw4);
    g.up = cuda_alloc(batch * upN);
    g.cat1 = cuda_alloc(batch * 2UL * c2 * hw2);
    g.d1 = cuda_alloc(batch * c1 * hw2);
    g.cat2 = cuda_alloc(batch * 2UL * c1 * hw);
    g.ok = batch > 0UL && height % 4UL == 0UL && width % 4UL == 0UL && height > 0UL && width > 0UL &&
           g.encConv1.weight.dev != 0UL && g.encConv2.weight.dev != 0UL && g.bottleneck.weight.dev != 0UL &&
           g.decConv1.weight.dev != 0UL && g.decConv2.weight.dev != 0UL &&
           g.e1.dev != 0UL && g.p1.dev != 0UL && g.e2.dev != 0UL && g.p2.dev != 0UL && g.mid.dev != 0UL &&
           g.up.dev != 0UL && g.cat1.dev != 0UL && g.d1.dev != 0UL && g.cat2.dev != 0UL;
    return g;
}

int cuda_unet_forward(const &CudaUNet net, struct CudaBuf x, struct CudaBuf y) {
    if (!net.ok) return 0;
    unsigned long B = net.batch;
    unsigned long H = net.height;
    unsigned long W = net.width;
    unsigned long c1 = net.encConv1.outChannels;
    unsigned long c2 = net.encConv2.outChannels;
    // Encoder: conv+relu (one cuDNN call) -> 2x2 max pool, twice.
    return cuda_conv2d_relu(&net.encConv1, x, net.e1, B, H, W) &&
           cuda_maxpool2d(net.e1, net.p1, B, c1, H, W, 2UL, 2UL) &&
           cuda_conv2d_relu(&net.encConv2, net.p1, net.e2, B, H / 2UL, W / 2UL) &&
           cuda_maxpool2d(net.e2, net.p2, B, c2, H / 2UL, W / 2UL, 2UL, 2UL) &&
           cuda_conv2d_relu(&net.bottleneck, net.p2, net.mid, B, H / 4UL, W / 4UL) &&
           // Decoder: upsample, concat the skip, conv(+relu).
           cuda_upsample2x_nearest(net.mid, net.up, B * c2, H / 4UL, W / 4UL) &&
           cuda_concat_channels(net.up, c2, net.e2, c2, net.cat1, B, (H / 2UL) * (W / 2UL)) &&
           cuda_conv2d_relu(&net.decConv1, net.cat1, net.d1, B, H / 2UL, W / 2UL) &&
           cuda_upsample2x_nearest(net.d1, net.up, B * c1, H / 2UL, W / 2UL) &&
           cuda_concat_channels(net.up, c1, net.e1, c1, net.cat2, B, H * W) &&
           cuda_conv2d(&net.decConv2, net.cat2, y, B, H, W);
}

void cuda_unet_free(&CudaUNet net) {
    cuda_conv2d_free(&net.encConv1);
    cuda_conv2d_free(&net.encConv2);
    cuda_conv2d_free(&net.bottleneck);
    cuda_conv2d_free(&net.decConv1);
    cuda_conv2d_free(&net.decConv2);
    cuda_free(&net.e1);
    cuda_free(&net.p1);
    cuda_free(&net.e2);
    cuda_free(&net.p2);
    cuda_free(&net.mid);
    cuda_free(&net.up);
    cuda_free(&net.cat1);
    cuda_free(&net.d1);
    cuda_free(&net.cat2);
    net.ok = 0;
}

// ── Host convenience ─────────────────────────────────────────────────────────

struct FeatureMap conv2d_forward_cuda(const &Conv2D layer, const &FeatureMap input) {
    unsigned long inC = input.channels;
    unsigned long h = input.height;
    unsigned long w = input.width;
    unsigned long outH = cuda_conv_out_dim(h, layer.kH, layer.stride, layer.padding);
    unsigned long outW = cuda_conv_out_dim(w, layer.kW, layer.stride, layer.padding);
    if (inC == layer.inChannels && outH > 0UL && outW > 0UL && cudnn_ensure_()) {
        struct CudaConv2D d = cuda_conv2d_upload(layer);
        struct CudaBuf x = cuda_upload(input.data, inC * h * w);
        struct CudaBuf y = cuda_alloc(layer.outChannels * outH * outW);
        struct FeatureMap out = feature_map_new(layer.outChannels, outH, outW);
        int ok = d.weight.dev != 0UL && x.dev != 0UL && y.dev != 0UL &&
                 cuda_conv2d(&d, x, y, 1UL, h, w) && cuda_read(y, out.data, y.len);
        cuda_conv2d_free(&d);
        cuda_free(&x);
        cuda_free(&y);
        if (ok) return out;
        feature_map_free(&out);
    }
    return conv2d_forward(layer, input);
}

struct FeatureMap unet_forward_cuda(&UNet net, const &FeatureMap input) {
    unsigned long h = input.height;
    unsigned long w = input.width;
    if (input.channels == net.encConv1.inChannels && cudnn_ensure_()) {
        struct CudaUNet g = cuda_unet_new(net, 1UL, h, w);
        unsigned long outC = net.decConv2.outChannels;
        struct CudaBuf x = cuda_upload(input.data, input.channels * h * w);
        struct CudaBuf y = cuda_alloc(outC * h * w);
        struct FeatureMap out = feature_map_new(outC, h, w);
        int ok = g.ok && x.dev != 0UL && y.dev != 0UL &&
                 cuda_unet_forward(&g, x, y) && cuda_read(y, out.data, y.len);
        cuda_unet_free(&g);
        cuda_free(&x);
        cuda_free(&y);
        if (ok) return out;
        feature_map_free(&out);
    }
    return unet_forward(net, input);
}

} // namespace std
