// SafeC Standard Library — CNN implementation (see cnn.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/cnn.h>
#include <std/simd/gemm.h>
#include <std/sync/parallel.h>

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);

struct FeatureMap feature_map_new(unsigned long channels, unsigned long height, unsigned long width) {
    struct FeatureMap fm;
    unsigned long size = channels * height * width;
    unsafe {
        float* buf = (float*)malloc(sizeof(float) * size);
        unsigned long i = 0UL;
        while (i < size) { buf[i] = (float)0.0; i = i + 1UL; }
        fm.data = (&heap float)buf;
    }
    fm.channels = channels;
    fm.height = height;
    fm.width = width;
    return fm;
}

void feature_map_free(&FeatureMap fm) {
    unsafe { free((void*)fm->data); }
}

// A map whose every element the caller is about to write: no zero pass, so
// the first touch of each page happens in the (possibly threaded) op itself.
static struct FeatureMap fm_uninit_(unsigned long channels, unsigned long height, unsigned long width) {
    struct FeatureMap fm;
    unsafe {
        float* buf = (float*)malloc(sizeof(float) * (channels * height * width));
        fm.data = (&heap float)buf;
    }
    fm.channels = channels;
    fm.height = height;
    fm.width = width;
    return fm;
}

static float __cnn_init_weight(unsigned long i, unsigned long seed) {
    unsigned long h = (i + 1UL) * 2654435761UL + seed * 40503UL;
    h = (h ^ (h >> 13UL)) * 2246822519UL;
    return (((float)(h % 2000UL) / (float)1000.0) - (float)1.0) * (float)0.3; // [-0.3, 0.3)
}

struct Conv2D conv2d_new(unsigned long inChannels, unsigned long outChannels,
                          unsigned long kH, unsigned long kW,
                          unsigned long stride, unsigned long padding) {
    struct Conv2D layer;
    unsigned long wSize = outChannels * inChannels * kH * kW;
    unsafe {
        float* w = (float*)malloc(sizeof(float) * wSize);
        unsigned long i = 0UL;
        while (i < wSize) { w[i] = __cnn_init_weight(i, 11UL); i = i + 1UL; }
        layer.weight = (&heap float)w;

        float* b = (float*)malloc(sizeof(float) * outChannels);
        i = 0UL;
        while (i < outChannels) { b[i] = (float)0.0; i = i + 1UL; }
        layer.bias = (&heap float)b;
    }
    layer.inChannels = inChannels;
    layer.outChannels = outChannels;
    layer.kH = kH;
    layer.kW = kW;
    layer.stride = stride;
    layer.padding = padding;
    return layer;
}

void conv2d_free(&Conv2D layer) {
    unsafe { free((void*)layer->weight); free((void*)layer->bias); }
}

// conv2d_forward is im2col + GEMM: for a block of output pixels, gather
// each pixel's receptive field into a column (inC*kH*kW values, zero where
// the window hangs over the padding), then out[:, block] += W . columns
// with sgemm, W being the weight viewed as an outC x (inC*kH*kW) matrix.
// Blocks of pixels run in parallel on the thread pool, each with its own
// column buffer, so the full im2col matrix is never materialized.
struct ConvJob {
    const float*  in;
    const float*  w;
    float*        out;
    unsigned long inC;
    unsigned long inH;
    unsigned long inW;
    unsigned long kH;
    unsigned long kW;
    unsigned long stride;
    unsigned long padding;
    unsigned long outC;
    unsigned long outW;
    unsigned long pixels;   // outH * outW
    unsigned long block;    // output pixels per block
    int           threaded; // let sgemm use the pool (only when there is a single block)
    const float*  bias;
    int           relu;
};

// out[oc][p0 .. p0+pc) = bias[oc], before the GEMM accumulates into it.
static void conv_bias_(const &ConvJob j, unsigned long p0, unsigned long pc) {
    unsafe {
        unsigned long oc = 0UL;
        while (oc < j.outC) {
            float b = j.bias[oc];
            float* row = &j.out[oc * j.pixels + p0];
            unsigned long q = 0UL;
            while (q < pc) { row[q] = b; q = q + 1UL; }
            oc = oc + 1UL;
        }
    }
}

static void conv_relu_(const &ConvJob j, unsigned long p0, unsigned long pc) {
    unsafe {
        unsigned long oc = 0UL;
        while (oc < j.outC) {
            float* row = &j.out[oc * j.pixels + p0];
            unsigned long q = 0UL;
            while (q < pc) { float v = row[q]; row[q] = v > (float)0.0 ? v : (float)0.0; q = q + 1UL; }
            oc = oc + 1UL;
        }
    }
}

// Column buffer, kept per thread and grown as needed (see gemm.sc's
// packing buffers for why it isn't allocated per call).
#if defined(__SAFEC_FREESTANDING__) || defined(__wasm__)
static float* g_conv_col = (float*)0;
static unsigned long g_conv_cap = 0UL;
#else
static thread_local float* g_conv_col = (float*)0;
static thread_local unsigned long g_conv_cap = 0UL;
#endif

static void conv_blocks_(&ConvJob j, unsigned long lo, unsigned long hi) {
    unsigned long K = j.inC * j.kH * j.kW;
    unsafe {
        if (g_conv_cap < K * j.block) {
            free((void*)g_conv_col);
            g_conv_col = (float*)malloc(sizeof(float) * K * j.block);
            g_conv_cap = g_conv_col == (float*)0 ? 0UL : K * j.block;
            if (g_conv_col == (float*)0) return;
        }
        float* col = g_conv_col;
        unsigned long blk = lo;
        while (blk < hi) {
            unsigned long p0 = blk * j.block;
            unsigned long pc = j.pixels - p0 < j.block ? j.pixels - p0 : j.block;
            unsigned long r = 0UL;
            unsigned long ic = 0UL;
            while (ic < j.inC) {
                const float* plane = &j.in[ic * j.inH * j.inW];
                unsigned long kh = 0UL;
                while (kh < j.kH) {
                    unsigned long kw = 0UL;
                    while (kw < j.kW) {
                        float* dst = &col[r * pc];
                        // Walk the block one output row segment at a time:
                        // within a segment ih is fixed and iw steps by
                        // 'stride', so only its two ends can hit padding.
                        unsigned long q = 0UL;
                        unsigned long oh = p0 / j.outW;
                        unsigned long ow = p0 % j.outW;
                        while (q < pc) {
                            unsigned long seg = j.outW - ow < pc - q ? j.outW - ow : pc - q;
                            long ih = (long)(oh * j.stride + kh) - (long)j.padding;
                            if (ih < 0L || (unsigned long)ih >= j.inH) {
                                unsigned long z = 0UL;
                                while (z < seg) { dst[q + z] = (float)0.0; z = z + 1UL; }
                            } else {
                                const float* src = &plane[(unsigned long)ih * j.inW];
                                long iw = (long)(ow * j.stride + kw) - (long)j.padding;
                                unsigned long z = 0UL;
                                while (z < seg && iw < 0L) {
                                    dst[q + z] = (float)0.0;
                                    z = z + 1UL;
                                    iw = iw + (long)j.stride;
                                }
                                if (j.stride == 1UL) {
                                    while (z < seg && (unsigned long)iw < j.inW) {
                                        dst[q + z] = src[(unsigned long)iw];
                                        z = z + 1UL;
                                        iw = iw + 1L;
                                    }
                                } else {
                                    while (z < seg && (unsigned long)iw < j.inW) {
                                        dst[q + z] = src[(unsigned long)iw];
                                        z = z + 1UL;
                                        iw = iw + (long)j.stride;
                                    }
                                }
                                while (z < seg) { dst[q + z] = (float)0.0; z = z + 1UL; }
                            }
                            q = q + seg;
                            ow = 0UL;
                            oh = oh + 1UL;
                        }
                        r = r + 1UL;
                        kw = kw + 1UL;
                    }
                    kh = kh + 1UL;
                }
                ic = ic + 1UL;
            }
            conv_bias_(j, p0, pc);
            if (j.threaded)
                sgemm(SGEMM_NO_TRANS, SGEMM_NO_TRANS, j.outC, pc, K, (float)1.0, (const &float)j.w, K,
                      (const &float)col, pc, (float)1.0, (&float)&j.out[p0], j.pixels);
            else
                sgemm_serial(SGEMM_NO_TRANS, SGEMM_NO_TRANS, j.outC, pc, K, (float)1.0, (const &float)j.w, K,
                             (const &float)col, pc, (float)1.0, (&float)&j.out[p0], j.pixels);
            if (j.relu) conv_relu_(j, p0, pc);
            blk = blk + 1UL;
        }
    }
}

static struct FeatureMap conv_run_(const &Conv2D layer, const &FeatureMap input, int relu) {
    unsigned long inH; unsigned long inW; unsigned long inC;
    unsigned long kH; unsigned long kW; unsigned long stride; unsigned long padding;
    unsigned long outC;
    unsafe {
        inC = input->channels; inH = input->height; inW = input->width;
        kH = layer->kH; kW = layer->kW; stride = layer->stride; padding = layer->padding;
        outC = layer->outChannels;
    }
    unsigned long outH = (inH + 2UL * padding - kH) / stride + 1UL;
    unsigned long outW = (inW + 2UL * padding - kW) / stride + 1UL;
    unsigned long pixels = outH * outW;
    unsigned long K = inC * kH * kW;
    if (pixels == 0UL || outC == 0UL || K == 0UL) {
        struct FeatureMap empty = feature_map_new(outC, outH, outW);
        unsafe {
            unsigned long oc = 0UL;
            while (oc < outC) {
                float b = layer->bias[oc];
                unsigned long q = 0UL;
                while (q < pixels) { empty.data[oc * pixels + q] = relu && b < (float)0.0 ? (float)0.0 : b; q = q + 1UL; }
                oc = oc + 1UL;
            }
        }
        return empty;
    }
    struct FeatureMap out = fm_uninit_(outC, outH, outW);

    unsafe {
        struct ConvJob j;
        j.in = (const float*)input->data;
        j.w = (const float*)layer->weight;
        j.out = (float*)out.data;
        j.bias = (const float*)layer->bias;
        j.relu = relu;
        j.inC = inC; j.inH = inH; j.inW = inW;
        j.kH = kH; j.kW = kW; j.stride = stride; j.padding = padding;
        j.outC = outC; j.outW = outW; j.pixels = pixels;
        if (kH == 1UL && kW == 1UL && stride == 1UL && padding == 0UL) {
            // 1x1 convolution: the input planes already are the columns.
            conv_bias_(&j, 0UL, pixels);
            sgemm(SGEMM_NO_TRANS, SGEMM_NO_TRANS, outC, pixels, K, (float)1.0,
                  (const &float)layer->weight, K, (const &float)input->data, pixels,
                  (float)1.0, (&float)out.data, pixels);
            if (relu) conv_relu_(&j, 0UL, pixels);
            return out;
        }
        // Blocks of 64..512 pixels, about one per thread; a column buffer
        // of K x block floats stays in L2.
        unsigned long threads = parallel_threads();
        unsigned long block = (pixels + threads - 1UL) / threads;
        block = (block + 15UL) / 16UL * 16UL;
        if (block < 64UL) block = 64UL;
        if (block > 512UL) block = 512UL;
        j.block = block;
        unsigned long blocks = (pixels + block - 1UL) / block;
        j.threaded = blocks == 1UL;
        if (blocks > 1UL && outC * pixels * K >= 262144UL) parallel_for(blocks, 1UL, conv_blocks_, &j);
        else conv_blocks_(&j, 0UL, blocks);
    }
    return out;
}

struct FeatureMap conv2d_forward(const &Conv2D layer, const &FeatureMap input) {
    return conv_run_(layer, input, 0);
}

struct FeatureMap conv2d_relu_forward(const &Conv2D layer, const &FeatureMap input) {
    return conv_run_(layer, input, 1);
}

// ── Pooling, upsampling, concatenation ───────────────────────────────────────
// Memory-bound, so they are split across the thread pool by channel once a
// map is large enough for that to beat the wake-up cost.

#define MAP_MAX     0
#define MAP_AVG     1
#define MAP_UP2     2
#define MAP_COPY    3

struct MapJob {
    const float*  in;
    float*        out;
    unsigned long h;        // input plane
    unsigned long w;
    unsigned long outH;
    unsigned long outW;
    unsigned long kernel;
    unsigned long stride;
    int           op;
};

static void map_channels_(&MapJob j, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long ch = lo;
        while (ch < hi) {
            const float* src = &j.in[ch * j.h * j.w];
            float* dst = &j.out[ch * j.outH * j.outW];
            if (j.op == MAP_COPY) {
                unsigned long n = j.h * j.w;
                unsigned long i = 0UL;
                while (i < n) { dst[i] = src[i]; i = i + 1UL; }
            } else if (j.op == MAP_UP2) {
                unsigned long y = 0UL;
                while (y < j.h) {
                    const float* s = &src[y * j.w];
                    float* d0 = &dst[(2UL * y) * j.outW];
                    float* d1 = &d0[j.outW];
                    unsigned long x = 0UL;
                    while (x < j.w) {
                        float v = s[x];
                        d0[2UL * x] = v; d0[2UL * x + 1UL] = v;
                        d1[2UL * x] = v; d1[2UL * x + 1UL] = v;
                        x = x + 1UL;
                    }
                    y = y + 1UL;
                }
            } else {
                float count = (float)(j.kernel * j.kernel);
                unsigned long oh = 0UL;
                while (oh < j.outH) {
                    unsigned long ow = 0UL;
                    while (ow < j.outW) {
                        const float* win = &src[(oh * j.stride) * j.w + ow * j.stride];
                        float acc = j.op == MAP_MAX ? win[0] : (float)0.0;
                        unsigned long kh = 0UL;
                        while (kh < j.kernel) {
                            unsigned long kw = 0UL;
                            while (kw < j.kernel) {
                                float v = win[kh * j.w + kw];
                                if (j.op == MAP_MAX) { if (v > acc) acc = v; }
                                else acc = acc + v;
                                kw = kw + 1UL;
                            }
                            kh = kh + 1UL;
                        }
                        dst[oh * j.outW + ow] = j.op == MAP_MAX ? acc : acc / count;
                        ow = ow + 1UL;
                    }
                    oh = oh + 1UL;
                }
            }
            ch = ch + 1UL;
        }
    }
}

static void map_run_(&MapJob j, unsigned long channels) {
    unsigned long per = j.outH * j.outW;
    if (per < j.h * j.w) per = j.h * j.w;
    if (per == 0UL) return;
    unsigned long grain = 16384UL / per + 1UL;
    if (channels * per >= 65536UL) parallel_for(channels, grain, map_channels_, j);
    else map_channels_(j, 0UL, channels);
}

static struct FeatureMap pool_(const &FeatureMap input, unsigned long kernel, unsigned long stride, int op) {
    unsigned long c; unsigned long h; unsigned long w;
    unsafe { c = input->channels; h = input->height; w = input->width; }
    unsigned long outH = (h - kernel) / stride + 1UL;
    unsigned long outW = (w - kernel) / stride + 1UL;
    struct FeatureMap out = fm_uninit_(c, outH, outW);
    struct MapJob j;
    unsafe {
        j.in = (const float*)input->data;
        j.out = (float*)out.data;
    }
    j.h = h; j.w = w; j.outH = outH; j.outW = outW;
    j.kernel = kernel; j.stride = stride; j.op = op;
    map_run_(&j, c);
    return out;
}

struct FeatureMap maxpool2d_forward(const &FeatureMap input, unsigned long kernel, unsigned long stride) {
    return pool_(input, kernel, stride, MAP_MAX);
}

struct FeatureMap avgpool2d_forward(const &FeatureMap input, unsigned long kernel, unsigned long stride) {
    return pool_(input, kernel, stride, MAP_AVG);
}

struct FeatureMap upsample2x_nearest(const &FeatureMap input) {
    unsigned long c; unsigned long h; unsigned long w;
    unsafe { c = input->channels; h = input->height; w = input->width; }
    struct FeatureMap out = fm_uninit_(c, h * 2UL, w * 2UL);
    struct MapJob j;
    unsafe {
        j.in = (const float*)input->data;
        j.out = (float*)out.data;
    }
    j.h = h; j.w = w; j.outH = h * 2UL; j.outW = w * 2UL;
    j.kernel = 0UL; j.stride = 0UL; j.op = MAP_UP2;
    map_run_(&j, c);
    return out;
}

struct FeatureMap concat_channels(const &FeatureMap a, const &FeatureMap b) {
    unsigned long ca; unsigned long cb; unsigned long h; unsigned long w;
    unsafe { ca = a->channels; cb = b->channels; h = a->height; w = a->width; }
    struct FeatureMap out = fm_uninit_(ca + cb, h, w);
    struct MapJob j;
    j.h = h; j.w = w; j.outH = h; j.outW = w;
    j.kernel = 0UL; j.stride = 0UL; j.op = MAP_COPY;
    unsafe {
        j.in = (const float*)a->data;
        j.out = (float*)out.data;
        map_run_(&j, ca);
        j.in = (const float*)b->data;
        j.out = (float*)&out.data[ca * h * w];
        map_run_(&j, cb);
    }
    return out;
}

} // namespace std
