// SafeC Standard Library — MIOpen convolution, pooling and U-Net (see gpu_miopen.h).
//
// Uses MIOpen's descriptor API on gpu_rocm's stream. Each convolution shape
// gets a cached plan: tensor/filter/convolution descriptors, the algorithm
// miopenFindConvolutionForwardAlgorithm timed fastest, and that algorithm's
// own workspace — per plan rather than shared, so a RocmGraph that captured
// a convolution keeps a valid workspace however many shapes come later.
// MIOpen's Find runs the candidates, so it needs real buffers: the plan is
// built on the first call for a shape, with that call's x, weight and y.
#pragma once
#include <std/stddef.h>
#include <std/ml/gpu_miopen.h>
#include <std/ml/gpu_rocm.h>
#include <std/ml/cnn.h>
#include <std/ml/unet.h>
#include <std/mem.h>

namespace std {

// ── MIOpen C ABI (miopen/miopen.h; every handle/descriptor is a pointer) ─────
extern int miopenCreate(void** handle);
extern int miopenSetStream(void* handle, void* stream);
extern int miopenCreateTensorDescriptor(void** desc);
extern int miopenSet4dTensorDescriptor(void* desc, int dataType, int n, int c, int h, int w);
extern int miopenDestroyTensorDescriptor(void* desc);
extern int miopenCreateConvolutionDescriptor(void** desc);
extern int miopenInitConvolutionDescriptor(void* desc, int mode, int padH, int padW, int strideH, int strideW,
                                           int dilationH, int dilationW);
extern int miopenDestroyConvolutionDescriptor(void* desc);
extern int miopenConvolutionForwardGetWorkSpaceSize(void* handle, void* wDesc, void* xDesc, void* convDesc,
                                                    void* yDesc, size_t* bytes);
extern int miopenFindConvolutionForwardAlgorithm(void* handle, void* xDesc, const void* x, void* wDesc,
                                                 const void* w, void* convDesc, void* yDesc, void* y,
                                                 int requested, int* returned, void* perfResults,
                                                 void* workspace, size_t wsBytes, bool exhaustive);
extern int miopenConvolutionForward(void* handle, const void* alpha, void* xDesc, const void* x,
                                    void* wDesc, const void* w, void* convDesc, int algo,
                                    const void* beta, void* yDesc, void* y, void* workspace, size_t wsBytes);
extern int miopenCreatePoolingDescriptor(void** desc);
extern int miopenSet2dPoolingDescriptor(void* desc, int mode, int windowH, int windowW,
                                        int padH, int padW, int strideH, int strideW);
extern int miopenPoolingForward(void* handle, void* poolDesc, const void* alpha, void* xDesc, const void* x,
                                const void* beta, void* yDesc, void* y, bool doBackward,
                                void* workspace, size_t wsBytes);

extern int hipMalloc(void** ptr, unsigned long size);
extern int hipFree(void* ptr);
extern int hipStreamSynchronize(void* stream);

#define MIO_FLOAT          1    // miopenFloat
#define MIO_CONVOLUTION    0    // miopenConvolution (cross-correlation, like cnn.sc)
#define MIO_POOL_MAX       0    // miopenPoolingMax
#define MIO_POOL_AVG       1    // miopenPoolingAverage (padding excluded; there is none)
#define MIO_PERF_BYTES     16   // sizeof(miopenConvAlgoPerf_t): algo int, time float, memory size_t
#define MIO_PERF_COUNT     8
#define MIO_PLANS          32
#define MIO_MAX_WORKSPACE  268435456UL   // 256 MiB: skip algorithms that want more

// ── State ────────────────────────────────────────────────────────────────────
static int   g_mio_state = 0;        // 0 = not tried, 1 = ready, -1 = unavailable
static void* g_mio = (void*)0;
static void* g_mio_pool = (void*)0;
static void* g_mio_pool_x = (void*)0;
static void* g_mio_pool_y = (void*)0;

// Plan cache. g_mp_key[s * 8 ..] = batch, inC, H, W, outC, kH, kW,
// stride | padding << 16; g_mp_used[s] = 0 for a free slot.
static unsigned long g_mp_key[MIO_PLANS * 8];
static int   g_mp_used[MIO_PLANS];
static void* g_mp_x[MIO_PLANS];
static void* g_mp_w[MIO_PLANS];
static void* g_mp_conv[MIO_PLANS];
static void* g_mp_y[MIO_PLANS];
static int   g_mp_algo[MIO_PLANS];
static void* g_mp_ws[MIO_PLANS];
static unsigned long g_mp_wsBytes[MIO_PLANS];
static int   g_mp_evict = 0;

static void* mio_stream_() {
    ?&void s = rocm_stream_handle();
    unsafe { return (void*)s; }
}

static int miopen_ensure_() {
    if (g_mio_state == 1) return rocm_available();
    if (g_mio_state != 0) return 0;
    g_mio_state = -1;
    if (!rocm_available()) return 0;
    unsafe {
        if (miopenCreate(&g_mio) != 0) return 0;
        if (miopenSetStream(g_mio, mio_stream_()) != 0) return 0;
        if (miopenCreatePoolingDescriptor(&g_mio_pool) != 0 ||
            miopenCreateTensorDescriptor(&g_mio_pool_x) != 0 ||
            miopenCreateTensorDescriptor(&g_mio_pool_y) != 0) return 0;
        int s = 0;
        while (s < MIO_PLANS) { g_mp_used[s] = 0; s = s + 1; }
    }
    g_mio_state = 1;
    return 1;
}

unsigned long rocm_conv_out_dim(unsigned long dim, unsigned long kernel,
                                unsigned long stride, unsigned long padding) {
    if (stride == 0UL || dim + 2UL * padding < kernel) return 0UL;
    return (dim + 2UL * padding - kernel) / stride + 1UL;
}

static int mio_int_ok_(unsigned long v) { return v > 0UL && v <= 2147483647UL; }

static void mio_plan_release_(int s) {
    unsafe {
        if (g_mp_used[s] == 0) return;
        miopenDestroyTensorDescriptor(g_mp_x[s]);
        miopenDestroyTensorDescriptor(g_mp_w[s]);
        miopenDestroyConvolutionDescriptor(g_mp_conv[s]);
        miopenDestroyTensorDescriptor(g_mp_y[s]);
        if (g_mp_ws[s] != (void*)0) {
            hipStreamSynchronize(mio_stream_());   // queued work may still use it
            hipFree(g_mp_ws[s]);
        }
        g_mp_used[s] = 0;
    }
}

// Finds or builds the plan for this convolution shape; -1 on failure.
// x, the weights and y are the buffers MIOpen's Find times its candidates on.
static int mio_plan_(const &RocmConv2D layer, struct RocmBuf x, struct RocmBuf y,
                     unsigned long batch, unsigned long height, unsigned long width) {
    unsigned long outH = rocm_conv_out_dim(height, layer.kH, layer.stride, layer.padding);
    unsigned long outW = rocm_conv_out_dim(width, layer.kW, layer.stride, layer.padding);
    if (!mio_int_ok_(batch) || !mio_int_ok_(height) || !mio_int_ok_(width) || !mio_int_ok_(outH) ||
        !mio_int_ok_(outW) || !mio_int_ok_(layer.inChannels) || !mio_int_ok_(layer.outChannels) ||
        !mio_int_ok_(layer.kH) || !mio_int_ok_(layer.kW) || layer.stride > 65535UL || layer.padding > 65535UL)
        return -1;
    unsigned long key[8];
    key[0] = batch; key[1] = layer.inChannels; key[2] = height; key[3] = width;
    key[4] = layer.outChannels; key[5] = layer.kH; key[6] = layer.kW;
    key[7] = layer.stride | (layer.padding << 16);
    int slot = -1;
    int s = 0;
    while (s < MIO_PLANS) {
        if (g_mp_used[s] != 0) {
            int same = 1;
            int i = 0;
            while (i < 8) { if (g_mp_key[s * 8 + i] != key[i]) same = 0; i = i + 1; }
            if (same) return s;
        } else if (slot < 0) {
            slot = s;
        }
        s = s + 1;
    }
    if (slot < 0) {
        slot = g_mp_evict;
        g_mp_evict = (g_mp_evict + 1) % MIO_PLANS;
        mio_plan_release_(slot);
    }
    unsafe {
        void* xd = (void*)0;
        void* wd = (void*)0;
        void* cd = (void*)0;
        void* yd = (void*)0;
        int ok = miopenCreateTensorDescriptor(&xd) == 0 && miopenCreateTensorDescriptor(&wd) == 0 &&
                 miopenCreateConvolutionDescriptor(&cd) == 0 && miopenCreateTensorDescriptor(&yd) == 0;
        ok = ok && miopenSet4dTensorDescriptor(xd, MIO_FLOAT, (int)batch, (int)layer.inChannels,
                                               (int)height, (int)width) == 0;
        ok = ok && miopenSet4dTensorDescriptor(wd, MIO_FLOAT, (int)layer.outChannels, (int)layer.inChannels,
                                               (int)layer.kH, (int)layer.kW) == 0;
        ok = ok && miopenInitConvolutionDescriptor(cd, MIO_CONVOLUTION, (int)layer.padding, (int)layer.padding,
                                                   (int)layer.stride, (int)layer.stride, 1, 1) == 0;
        ok = ok && miopenSet4dTensorDescriptor(yd, MIO_FLOAT, (int)batch, (int)layer.outChannels,
                                               (int)outH, (int)outW) == 0;
        // Find needs a workspace big enough for every candidate it may try;
        // the winner's own (usually smaller) workspace is allocated after.
        size_t findBytes = 0UL;
        ok = ok && miopenConvolutionForwardGetWorkSpaceSize(g_mio, wd, xd, cd, yd, &findBytes) == 0;
        if (ok && (unsigned long)findBytes > MIO_MAX_WORKSPACE) findBytes = (size_t)MIO_MAX_WORKSPACE;
        void* findWs = (void*)0;
        if (ok && findBytes > 0UL) ok = hipMalloc(&findWs, (unsigned long)findBytes) == 0;
        // miopenConvAlgoPerf_t: { int fwd_algo; float time; size_t memory; },
        // fastest first.
        unsigned long perf[2 * MIO_PERF_COUNT];
        int returned = 0;
        ok = ok && miopenFindConvolutionForwardAlgorithm(g_mio, xd, (const void*)x.dev, wd,
                       (const void*)layer.weight.dev, cd, yd, (void*)y.dev, MIO_PERF_COUNT, &returned,
                       (void*)&perf[0], findWs, findBytes, false) == 0;
        if (findWs != (void*)0) hipFree(findWs);
        int algo = -1;
        unsigned long wsBytes = 0UL;
        int r = 0;
        while (ok && algo < 0 && r < returned) {
            int* p = (int*)&perf[r * 2];
            unsigned long mem = perf[r * 2 + 1];
            if (mem <= (unsigned long)findBytes) {
                algo = p[0];
                wsBytes = mem;
            }
            r = r + 1;
        }
        ok = ok && algo >= 0;
        void* ws = (void*)0;
        if (ok && wsBytes > 0UL) ok = hipMalloc(&ws, wsBytes) == 0;
        if (!ok) {
            if (xd != (void*)0) miopenDestroyTensorDescriptor(xd);
            if (wd != (void*)0) miopenDestroyTensorDescriptor(wd);
            if (cd != (void*)0) miopenDestroyConvolutionDescriptor(cd);
            if (yd != (void*)0) miopenDestroyTensorDescriptor(yd);
            return -1;
        }
        g_mp_x[slot] = xd;
        g_mp_w[slot] = wd;
        g_mp_conv[slot] = cd;
        g_mp_y[slot] = yd;
        g_mp_algo[slot] = algo;
        g_mp_ws[slot] = ws;
        g_mp_wsBytes[slot] = wsBytes;
        int i = 0;
        while (i < 8) { g_mp_key[slot * 8 + i] = key[i]; i = i + 1; }
        g_mp_used[slot] = 1;
    }
    return slot;
}

static int mio_conv_(const &RocmConv2D layer, struct RocmBuf x, struct RocmBuf y,
                     unsigned long batch, unsigned long height, unsigned long width, int relu) {
    if (!miopen_ensure_() || layer.weight.dev == 0UL || layer.bias.dev == 0UL) return 0;
    unsigned long outH = rocm_conv_out_dim(height, layer.kH, layer.stride, layer.padding);
    unsigned long outW = rocm_conv_out_dim(width, layer.kW, layer.stride, layer.padding);
    if (x.dev == 0UL || y.dev == 0UL ||
        x.len < batch * layer.inChannels * height * width ||
        y.len < batch * layer.outChannels * outH * outW) return 0;
    int p = mio_plan_(layer, x, y, batch, height, width);
    if (p < 0) return 0;
    float one = 1.0f;
    float zero = 0.0f;
    unsafe {
        if (miopenConvolutionForward(g_mio, (const void*)&one, g_mp_x[p], (const void*)x.dev,
                g_mp_w[p], (const void*)layer.weight.dev, g_mp_conv[p], g_mp_algo[p],
                (const void*)&zero, g_mp_y[p], (void*)y.dev, g_mp_ws[p], (size_t)g_mp_wsBytes[p]) != 0)
            return 0;
    }
    // Bias (and ReLU) in one pass over y.
    struct RocmBuf yv;
    yv.dev = y.dev;
    yv.len = batch * layer.outChannels * outH * outW;
    return rocm_bias_channels(yv, layer.bias, batch, layer.outChannels, outH * outW, relu);
}

int rocm_conv2d(const &RocmConv2D layer, struct RocmBuf x, struct RocmBuf y,
                unsigned long batch, unsigned long height, unsigned long width) {
    return mio_conv_(layer, x, y, batch, height, width, 0);
}

int rocm_conv2d_relu(const &RocmConv2D layer, struct RocmBuf x, struct RocmBuf y,
                     unsigned long batch, unsigned long height, unsigned long width) {
    return mio_conv_(layer, x, y, batch, height, width, 1);
}

static int mio_pool_(int mode, struct RocmBuf x, struct RocmBuf y, unsigned long batch, unsigned long channels,
                     unsigned long height, unsigned long width, unsigned long kernel, unsigned long stride) {
    if (!miopen_ensure_()) return 0;
    unsigned long outH = rocm_conv_out_dim(height, kernel, stride, 0UL);
    unsigned long outW = rocm_conv_out_dim(width, kernel, stride, 0UL);
    if (!mio_int_ok_(batch) || !mio_int_ok_(channels) || !mio_int_ok_(height) || !mio_int_ok_(width) ||
        !mio_int_ok_(kernel) || !mio_int_ok_(stride) || outH == 0UL || outW == 0UL ||
        x.dev == 0UL || y.dev == 0UL ||
        x.len < batch * channels * height * width || y.len < batch * channels * outH * outW) return 0;
    float one = 1.0f;
    float zero = 0.0f;
    unsafe {
        if (miopenSet2dPoolingDescriptor(g_mio_pool, mode, (int)kernel, (int)kernel, 0, 0,
                                         (int)stride, (int)stride) != 0 ||
            miopenSet4dTensorDescriptor(g_mio_pool_x, MIO_FLOAT, (int)batch, (int)channels,
                                        (int)height, (int)width) != 0 ||
            miopenSet4dTensorDescriptor(g_mio_pool_y, MIO_FLOAT, (int)batch, (int)channels,
                                        (int)outH, (int)outW) != 0) return 0;
        // Forward only: no index workspace.
        return miopenPoolingForward(g_mio, g_mio_pool, (const void*)&one, g_mio_pool_x, (const void*)x.dev,
                                    (const void*)&zero, g_mio_pool_y, (void*)y.dev, false,
                                    (void*)0, (size_t)0) == 0;
    }
}

int rocm_maxpool2d(struct RocmBuf x, struct RocmBuf y, unsigned long batch, unsigned long channels,
                   unsigned long height, unsigned long width, unsigned long kernel, unsigned long stride) {
    return mio_pool_(MIO_POOL_MAX, x, y, batch, channels, height, width, kernel, stride);
}

int rocm_avgpool2d(struct RocmBuf x, struct RocmBuf y, unsigned long batch, unsigned long channels,
                   unsigned long height, unsigned long width, unsigned long kernel, unsigned long stride) {
    return mio_pool_(MIO_POOL_AVG, x, y, batch, channels, height, width, kernel, stride);
}

int rocm_concat_channels(struct RocmBuf a, unsigned long aChannels, struct RocmBuf c, unsigned long cChannels,
                         struct RocmBuf out, unsigned long batch, unsigned long plane) {
    // One kernel over every image, instead of two copies per image.
    return rocm_concat(a, aChannels * plane, c, cChannels * plane, out, batch);
}

// ── Layers ───────────────────────────────────────────────────────────────────

struct RocmConv2D rocm_conv2d_upload(const &Conv2D layer) {
    struct RocmConv2D d;
    d.inChannels = layer.inChannels;
    d.outChannels = layer.outChannels;
    d.kH = layer.kH;
    d.kW = layer.kW;
    d.stride = layer.stride;
    d.padding = layer.padding;
    unsigned long wn = layer.outChannels * layer.inChannels * layer.kH * layer.kW;
    d.weight = rocm_upload(layer.weight, wn);
    d.bias = rocm_upload(layer.bias, layer.outChannels);
    if (d.weight.dev == 0UL || d.bias.dev == 0UL) {
        rocm_free(&d.weight);
        rocm_free(&d.bias);
    }
    return d;
}

void rocm_conv2d_free(&RocmConv2D layer) {
    rocm_free(&layer.weight);
    rocm_free(&layer.bias);
}

// ── U-Net ────────────────────────────────────────────────────────────────────

struct RocmUNet rocm_unet_new(const &UNet net, unsigned long batch, unsigned long height, unsigned long width) {
    struct RocmUNet g;
    g.batch = batch;
    g.height = height;
    g.width = width;
    g.encConv1 = rocm_conv2d_upload(&net.encConv1);
    g.encConv2 = rocm_conv2d_upload(&net.encConv2);
    g.bottleneck = rocm_conv2d_upload(&net.bottleneck);
    g.decConv1 = rocm_conv2d_upload(&net.decConv1);
    g.decConv2 = rocm_conv2d_upload(&net.decConv2);
    unsigned long c1 = net.encConv1.outChannels;
    unsigned long c2 = net.encConv2.outChannels;
    unsigned long hw = height * width;
    unsigned long hw2 = hw / 4UL;
    unsigned long hw4 = hw / 16UL;
    g.e1 = rocm_alloc(batch * c1 * hw);
    g.p1 = rocm_alloc(batch * c1 * hw2);
    g.e2 = rocm_alloc(batch * c2 * hw2);
    g.p2 = rocm_alloc(batch * c2 * hw4);
    g.mid = rocm_alloc(batch * c2 * hw4);
    g.cat1 = rocm_alloc(batch * 2UL * c2 * hw2);
    g.d1 = rocm_alloc(batch * c1 * hw2);
    g.cat2 = rocm_alloc(batch * 2UL * c1 * hw);
    g.ok = batch > 0UL && height % 4UL == 0UL && width % 4UL == 0UL && height > 0UL && width > 0UL &&
           g.encConv1.weight.dev != 0UL && g.encConv2.weight.dev != 0UL && g.bottleneck.weight.dev != 0UL &&
           g.decConv1.weight.dev != 0UL && g.decConv2.weight.dev != 0UL &&
           g.e1.dev != 0UL && g.p1.dev != 0UL && g.e2.dev != 0UL && g.p2.dev != 0UL && g.mid.dev != 0UL &&
           g.cat1.dev != 0UL && g.d1.dev != 0UL && g.cat2.dev != 0UL;
    return g;
}

int rocm_unet_forward(const &RocmUNet net, struct RocmBuf x, struct RocmBuf y) {
    if (!net.ok) return 0;
    unsigned long B = net.batch;
    unsigned long H = net.height;
    unsigned long W = net.width;
    unsigned long c1 = net.encConv1.outChannels;
    unsigned long c2 = net.encConv2.outChannels;
    // Encoder: conv+relu -> 2x2 max pool, twice.
    return rocm_conv2d_relu(&net.encConv1, x, net.e1, B, H, W) &&
           rocm_maxpool2d(net.e1, net.p1, B, c1, H, W, 2UL, 2UL) &&
           rocm_conv2d_relu(&net.encConv2, net.p1, net.e2, B, H / 2UL, W / 2UL) &&
           rocm_maxpool2d(net.e2, net.p2, B, c2, H / 2UL, W / 2UL, 2UL, 2UL) &&
           rocm_conv2d_relu(&net.bottleneck, net.p2, net.mid, B, H / 4UL, W / 4UL) &&
           // Decoder: upsample + concat the skip (one kernel), conv(+relu).
           rocm_upsample2x_concat(net.mid, c2, net.e2, c2, net.cat1, B, H / 4UL, W / 4UL) &&
           rocm_conv2d_relu(&net.decConv1, net.cat1, net.d1, B, H / 2UL, W / 2UL) &&
           rocm_upsample2x_concat(net.d1, c1, net.e1, c1, net.cat2, B, H / 2UL, W / 2UL) &&
           rocm_conv2d(&net.decConv2, net.cat2, y, B, H, W);
}

void rocm_unet_free(&RocmUNet net) {
    rocm_conv2d_free(&net.encConv1);
    rocm_conv2d_free(&net.encConv2);
    rocm_conv2d_free(&net.bottleneck);
    rocm_conv2d_free(&net.decConv1);
    rocm_conv2d_free(&net.decConv2);
    rocm_free(&net.e1);
    rocm_free(&net.p1);
    rocm_free(&net.e2);
    rocm_free(&net.p2);
    rocm_free(&net.mid);
    rocm_free(&net.cat1);
    rocm_free(&net.d1);
    rocm_free(&net.cat2);
    net.ok = 0;
}

// ── Host convenience ─────────────────────────────────────────────────────────

struct FeatureMap conv2d_forward_rocm(const &Conv2D layer, const &FeatureMap input) {
    unsigned long inC = input.channels;
    unsigned long h = input.height;
    unsigned long w = input.width;
    unsigned long outH = rocm_conv_out_dim(h, layer.kH, layer.stride, layer.padding);
    unsigned long outW = rocm_conv_out_dim(w, layer.kW, layer.stride, layer.padding);
    if (inC == layer.inChannels && outH > 0UL && outW > 0UL && miopen_ensure_()) {
        struct RocmConv2D d = rocm_conv2d_upload(layer);
        struct RocmBuf x = rocm_upload(input.data, inC * h * w);
        struct RocmBuf y = rocm_alloc(layer.outChannels * outH * outW);
        struct FeatureMap out = feature_map_new(layer.outChannels, outH, outW);
        int ok = d.weight.dev != 0UL && x.dev != 0UL && y.dev != 0UL &&
                 rocm_conv2d(&d, x, y, 1UL, h, w) && rocm_read(y, out.data, y.len);
        rocm_conv2d_free(&d);
        rocm_free(&x);
        rocm_free(&y);
        if (ok) return out;
        feature_map_free(&out);
    }
    return conv2d_forward(layer, input);
}

struct FeatureMap unet_forward_rocm(&UNet net, const &FeatureMap input) {
    unsigned long h = input.height;
    unsigned long w = input.width;
    if (input.channels == net.encConv1.inChannels && miopen_ensure_()) {
        struct RocmUNet g = rocm_unet_new(net, 1UL, h, w);
        unsigned long outC = net.decConv2.outChannels;
        struct RocmBuf x = rocm_upload(input.data, input.channels * h * w);
        struct RocmBuf y = rocm_alloc(outC * h * w);
        struct FeatureMap out = feature_map_new(outC, h, w);
        int ok = g.ok && x.dev != 0UL && y.dev != 0UL &&
                 rocm_unet_forward(&g, x, y) && rocm_read(y, out.data, y.len);
        rocm_unet_free(&g);
        rocm_free(&x);
        rocm_free(&y);
        if (ok) return out;
        feature_map_free(&out);
    }
    return unet_forward(net, input);
}

} // namespace std
