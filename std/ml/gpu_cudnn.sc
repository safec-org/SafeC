// SafeC Standard Library — cuDNN convolution, pooling and U-Net (see gpu_cudnn.h).
//
// Uses cuDNN's descriptor API (cudnn_ops / cudnn_cnn) on gpu_cuda's stream.
// Each convolution shape gets a cached plan: tensor/filter/convolution
// descriptors, the algorithm cudnnFindConvolutionForwardAlgorithm timed
// fastest, and that algorithm's own workspace — per plan rather than shared,
// so a CudaGraph that captured a convolution keeps a valid workspace however
// many shapes come later.
#pragma once
#include <std/stddef.h>
#include <std/ml/gpu_cudnn.h>
#include <std/ml/gpu_cuda.h>
#include <std/ml/cnn.h>
#include <std/ml/unet.h>
#include <std/mem.h>

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

extern int cuMemAlloc_v2(unsigned long long* dptr, size_t bytes);
extern int cuMemFree_v2(unsigned long long dptr);
extern int cuStreamSynchronize(void* stream);

#define DNN_TENSOR_NCHW        0    // CUDNN_TENSOR_NCHW
#define DNN_DATA_FLOAT         0    // CUDNN_DATA_FLOAT
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
#define DNN_MAX_WORKSPACE      268435456UL   // 256 MiB: skip algorithms that want more

// ── State ────────────────────────────────────────────────────────────────────
static int   g_dnn_state = 0;        // 0 = not tried, 1 = ready, -1 = unavailable
static void* g_dnn = (void*)0;
static void* g_dnn_relu = (void*)0;
static void* g_dnn_pool = (void*)0;
static void* g_dnn_pool_x = (void*)0;
static void* g_dnn_pool_y = (void*)0;

// Plan cache. g_dp_key[s * 8 ..] = batch, inC, H, W, outC, kH, kW,
// stride | padding << 16 | tf32 << 32; g_dp_used[s] = 0 for a free slot.
static unsigned long g_dp_key[DNN_PLANS * 8];
static int   g_dp_used[DNN_PLANS];
static void* g_dp_x[DNN_PLANS];
static void* g_dp_w[DNN_PLANS];
static void* g_dp_conv[DNN_PLANS];
static void* g_dp_y[DNN_PLANS];
static void* g_dp_b[DNN_PLANS];
static int   g_dp_algo[DNN_PLANS];
static unsigned long long g_dp_ws[DNN_PLANS];
static unsigned long g_dp_wsBytes[DNN_PLANS];
static int   g_dp_evict = 0;

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
        while (s < DNN_PLANS) { g_dp_used[s] = 0; s = s + 1; }
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

static void dnn_plan_release_(int s) {
    unsafe {
        if (g_dp_used[s] == 0) return;
        cudnnDestroyTensorDescriptor(g_dp_x[s]);
        cudnnDestroyFilterDescriptor(g_dp_w[s]);
        cudnnDestroyConvolutionDescriptor(g_dp_conv[s]);
        cudnnDestroyTensorDescriptor(g_dp_y[s]);
        cudnnDestroyTensorDescriptor(g_dp_b[s]);
        if (g_dp_ws[s] != 0ULL) {
            cuStreamSynchronize(cuda_stream_handle());   // queued work may still use it
            cuMemFree_v2(g_dp_ws[s]);
        }
        g_dp_used[s] = 0;
    }
}

// Finds or builds the plan for this convolution shape; -1 on failure.
static int dnn_plan_(const &CudaConv2D layer, unsigned long batch, unsigned long height, unsigned long width) {
    unsigned long outH = cuda_conv_out_dim(height, layer.kH, layer.stride, layer.padding);
    unsigned long outW = cuda_conv_out_dim(width, layer.kW, layer.stride, layer.padding);
    if (!dnn_int_ok_(batch) || !dnn_int_ok_(height) || !dnn_int_ok_(width) || !dnn_int_ok_(outH) ||
        !dnn_int_ok_(outW) || !dnn_int_ok_(layer.inChannels) || !dnn_int_ok_(layer.outChannels) ||
        !dnn_int_ok_(layer.kH) || !dnn_int_ok_(layer.kW) || layer.stride > 65535UL || layer.padding > 65535UL)
        return -1;
    unsigned long key[8];
    key[0] = batch; key[1] = layer.inChannels; key[2] = height; key[3] = width;
    key[4] = layer.outChannels; key[5] = layer.kH; key[6] = layer.kW;
    key[7] = layer.stride | (layer.padding << 16) | ((unsigned long)cuda_tf32_enabled() << 32);
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
            return -1;
        }
        g_dp_x[slot] = xd;
        g_dp_w[slot] = wd;
        g_dp_conv[slot] = cd;
        g_dp_y[slot] = yd;
        g_dp_b[slot] = bd;
        g_dp_algo[slot] = algo;
        g_dp_ws[slot] = ws;
        g_dp_wsBytes[slot] = wsBytes;
        int i = 0;
        while (i < 8) { g_dp_key[slot * 8 + i] = key[i]; i = i + 1; }
        g_dp_used[slot] = 1;
    }
    return slot;
}

static int dnn_conv_(const &CudaConv2D layer, struct CudaBuf x, struct CudaBuf y,
                     unsigned long batch, unsigned long height, unsigned long width, int relu) {
    if (!cudnn_ensure_() || layer.weight.dev == 0UL || layer.bias.dev == 0UL) return 0;
    unsigned long outH = cuda_conv_out_dim(height, layer.kH, layer.stride, layer.padding);
    unsigned long outW = cuda_conv_out_dim(width, layer.kW, layer.stride, layer.padding);
    if (x.dev == 0UL || y.dev == 0UL ||
        x.len < batch * layer.inChannels * height * width ||
        y.len < batch * layer.outChannels * outH * outW) return 0;
    int p = dnn_plan_(layer, batch, height, width);
    if (p < 0) return 0;
    float one = 1.0f;
    float zero = 0.0f;
    unsafe {
        void* ws = (void*)(unsigned long)g_dp_ws[p];
        if (relu) {
            // y = relu(1 * conv(x) + 0 * z + bias), z aliased to y
            return cudnnConvolutionBiasActivationForward(g_dnn, (const void*)&one, g_dp_x[p], (const void*)x.dev,
                       g_dp_w[p], (const void*)layer.weight.dev, g_dp_conv[p], g_dp_algo[p], ws, g_dp_wsBytes[p],
                       (const void*)&zero, g_dp_y[p], (const void*)y.dev, g_dp_b[p], (const void*)layer.bias.dev,
                       g_dnn_relu, g_dp_y[p], (void*)y.dev) == 0;
        }
        // (BiasActivation with an identity activation is limited to one
        // algorithm, so the plain case is convolution + broadcast bias add.)
        if (cudnnConvolutionForward(g_dnn, (const void*)&one, g_dp_x[p], (const void*)x.dev,
                g_dp_w[p], (const void*)layer.weight.dev, g_dp_conv[p], g_dp_algo[p], ws, g_dp_wsBytes[p],
                (const void*)&zero, g_dp_y[p], (void*)y.dev) != 0) return 0;
        return cudnnAddTensor(g_dnn, (const void*)&one, g_dp_b[p], (const void*)layer.bias.dev,
                              (const void*)&one, g_dp_y[p], (void*)y.dev) == 0;
    }
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
    unsigned long b = 0UL;
    while (b < batch) {
        if (!cuda_copy(a, b * na, out, b * (na + nc), na) ||
            !cuda_copy(c, b * nc, out, b * (na + nc) + na, nc)) return 0;
        b = b + 1UL;
    }
    return 1;
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
