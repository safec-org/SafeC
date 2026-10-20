// SafeC Standard Library — CNN implementation (see cnn.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/cnn.h>
#include <std/ml/tensor.h>
#include <std/simd/gemm.h>
#include <std/sync/parallel.h>

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);

// A map's storage: from this thread's TensorArena when one is active (see
// cnn.h), else malloc.
static struct FeatureMap fm_uninit_(unsigned long channels, unsigned long height, unsigned long width) {
    struct FeatureMap fm;
    unsigned long bytes = sizeof(float) * (channels * height * width);
    unsafe {
        float* buf = (float*)0;
        fm.fromArena = 0;
        match (tensor_arena_current_()) {
            case some(a): {
                buf = (float*)tensor_arena_alloc_raw_(a, bytes);
                fm.fromArena = 1;
            }
            case null: { buf = (float*)malloc(bytes); }
        }
        fm.data = (&heap float)buf;
    }
    fm.channels = channels;
    fm.height = height;
    fm.width = width;
    return fm;
}

struct FeatureMap feature_map_new(unsigned long channels, unsigned long height, unsigned long width) {
    struct FeatureMap fm = fm_uninit_(channels, height, width);
    unsigned long size = channels * height * width;
    unsafe {
        unsigned long i = 0UL;
        while (i < size) { fm.data[i] = (float)0.0; i = i + 1UL; }
    }
    return fm;
}

void feature_map_free(&FeatureMap fm) {
    unsafe { if (!fm->fromArena) free((void*)fm->data); }
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

// Columns for output pixels [p0, p0 + pc): row r = (ic, kh, kw) of the
// receptive field, written to col[r * ld + 0 .. pc).
static void conv_im2col_(const &ConvJob j, float* col, unsigned long ld, unsigned long p0, unsigned long pc) {
    unsafe {
        unsigned long r = 0UL;
        unsigned long ic = 0UL;
        while (ic < j.inC) {
            const float* plane = &j.in[ic * j.inH * j.inW];
            unsigned long kh = 0UL;
            while (kh < j.kH) {
                unsigned long kw = 0UL;
                while (kw < j.kW) {
                    float* dst = &col[r * ld];
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
    }
}

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
            conv_im2col_(j, col, pc, p0, pc);
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

// ── Winograd F(2x2, 3x3) ─────────────────────────────────────────────────────
// 3x3, stride-1 layers (Lavin & Gray, "Fast Algorithms for Convolutional
// Neural Networks", arXiv:1509.09308): each 2x2 block of outputs comes from
// a 4x4 input tile as Y = A^T [ (G g G^T) . (B^T d B) ] A, so a tile costs
// 16 multiplies per (in, out) channel pair instead of 36. With the input
// tiles transformed (V) and the filters transformed (U), the 16 positions
// of the 4x4 product are 16 GEMMs over channels: M_x = U_x (outC x inC)
// . V_x (inC x tiles). The input transform, the 16 GEMMs and the output
// transform each run on the thread pool.
//   B^T = [1 0 -1 0; 0 1 1 0; 0 -1 1 0; 0 1 0 -1]
//   G   = [1 0 0; 1/2 1/2 1/2; 1/2 -1/2 1/2; 0 0 1]
//   A^T = [1 1 1 0; 0 1 -1 -1]
typedef vec<float, 8> cnn_f32x8_;

struct WinoJob {
    const float*  in;
    const float*  u;        // [16][outC][inC]
    const float*  bias;
    float*        out;
    float*        v;        // [16][inC][tiles], planes pv apart
    float*        m;        // [16][outC][tiles], planes pm apart
    unsigned long pv;
    unsigned long pm;
    unsigned long inC;
    unsigned long inH;
    unsigned long inW;
    unsigned long outC;
    unsigned long outH;
    unsigned long outW;
    unsigned long pad;
    unsigned long tilesW;   // tiles per row
    unsigned long tiles;    // total
    int           relu;
};

// V and M for one call, kept by the calling thread and grown as needed.
#if defined(__SAFEC_FREESTANDING__) || defined(__wasm__)
static float* g_wino_buf = (float*)0;
static unsigned long g_wino_cap = 0UL;
#else
static thread_local float* g_wino_buf = (float*)0;
static thread_local unsigned long g_wino_cap = 0UL;
#endif

// U = G g G^T for output channels [oc0, oc1): u[(x * outC + oc) * inC + ic],
// x = 4 r + c.
struct WinoFilterJob { const float* w; float* u; unsigned long outC; unsigned long inC; };

static void wino_filters_(&WinoFilterJob fj, unsigned long oc0, unsigned long oc1) {
    const float* w = fj.w;
    float* u = fj.u;
    unsigned long outC = fj.outC;
    unsigned long inC = fj.inC;
    unsafe {
        unsigned long oc = oc0;
        while (oc < oc1) {
            unsigned long ic = 0UL;
            while (ic < inC) {
                const float* g = &w[(oc * inC + ic) * 9UL];
                float t[12];   // G g: 4 x 3
                unsigned long c = 0UL;
                while (c < 3UL) {
                    float g0 = g[c]; float g1 = g[3UL + c]; float g2 = g[6UL + c];
                    t[c] = g0;
                    t[3UL + c] = (float)0.5 * (g0 + g1 + g2);
                    t[6UL + c] = (float)0.5 * (g0 - g1 + g2);
                    t[9UL + c] = g2;
                    c = c + 1UL;
                }
                unsigned long r = 0UL;
                while (r < 4UL) {   // (G g) G^T: row r
                    float a0 = t[3UL * r]; float a1 = t[3UL * r + 1UL]; float a2 = t[3UL * r + 2UL];
                    float v[4];
                    v[0] = a0;
                    v[1] = (float)0.5 * (a0 + a1 + a2);
                    v[2] = (float)0.5 * (a0 - a1 + a2);
                    v[3] = a2;
                    unsigned long q = 0UL;
                    while (q < 4UL) { u[((4UL * r + q) * outC + oc) * inC + ic] = v[q]; q = q + 1UL; }
                    r = r + 1UL;
                }
                ic = ic + 1UL;
            }
            oc = oc + 1UL;
        }
    }
}

// Phase 1, tiles [t0, t1): V = B^T d B per channel.
static void wino_in_(&WinoJob j, unsigned long t0, unsigned long t1) {
    unsafe {
        unsigned long T = j.tiles;
        unsigned long pv = j.pv;
        unsigned long ic = 0UL;
        while (ic < j.inC) {
            const float* plane = &j.in[ic * j.inH * j.inW];
            unsigned long tile = t0;
            while (tile < t1) {
                long ih0 = (long)(2UL * (tile / j.tilesW)) - (long)j.pad;
                long iw0 = (long)(2UL * (tile % j.tilesW)) - (long)j.pad;
                // Eight tiles of one row, wholly inside the image: the same
                // transform, one tile per lane.
                if (tile % j.tilesW + 8UL <= j.tilesW && tile + 8UL <= t1 && ih0 >= 0L && iw0 >= 0L &&
                    (unsigned long)ih0 + 4UL <= j.inH && (unsigned long)iw0 + 18UL <= j.inW) {
                    cnn_f32x8_ dv[16];
                    unsigned long r8 = 0UL;
                    while (r8 < 4UL) {
                        const float* src = &plane[((unsigned long)ih0 + r8) * j.inW + (unsigned long)iw0];
                        unsigned long c8 = 0UL;
                        while (c8 < 4UL) {
                            cnn_f32x8_ v;
                            v[0] = src[c8];        v[1] = src[c8 + 2UL];  v[2] = src[c8 + 4UL];  v[3] = src[c8 + 6UL];
                            v[4] = src[c8 + 8UL];  v[5] = src[c8 + 10UL]; v[6] = src[c8 + 12UL]; v[7] = src[c8 + 14UL];
                            dv[4UL * r8 + c8] = v;
                            c8 = c8 + 1UL;
                        }
                        r8 = r8 + 1UL;
                    }
                    cnn_f32x8_ tv[16];
                    unsigned long c8 = 0UL;
                    while (c8 < 4UL) {
                        tv[c8] = dv[c8] - dv[8UL + c8];
                        tv[4UL + c8] = dv[4UL + c8] + dv[8UL + c8];
                        tv[8UL + c8] = dv[8UL + c8] - dv[4UL + c8];
                        tv[12UL + c8] = dv[4UL + c8] - dv[12UL + c8];
                        c8 = c8 + 1UL;
                    }
                    r8 = 0UL;
                    while (r8 < 4UL) {
                        float* vr = &j.v[(4UL * r8) * pv + ic * T + tile];
                        *(cnn_f32x8_*)&vr[0] = tv[4UL * r8] - tv[4UL * r8 + 2UL];
                        *(cnn_f32x8_*)&vr[pv] = tv[4UL * r8 + 1UL] + tv[4UL * r8 + 2UL];
                        *(cnn_f32x8_*)&vr[2UL * pv] = tv[4UL * r8 + 2UL] - tv[4UL * r8 + 1UL];
                        *(cnn_f32x8_*)&vr[3UL * pv] = tv[4UL * r8 + 1UL] - tv[4UL * r8 + 3UL];
                        r8 = r8 + 1UL;
                    }
                    tile = tile + 8UL;
                    continue;
                }
                float d[16];
                unsigned long r = 0UL;
                while (r < 4UL) {
                    long ih = ih0 + (long)r;
                    if (ih >= 0L && (unsigned long)ih < j.inH && iw0 >= 0L && (unsigned long)iw0 + 4UL <= j.inW) {
                        const float* src = &plane[(unsigned long)ih * j.inW + (unsigned long)iw0];
                        d[4UL * r] = src[0]; d[4UL * r + 1UL] = src[1]; d[4UL * r + 2UL] = src[2]; d[4UL * r + 3UL] = src[3];
                    } else {
                        unsigned long c = 0UL;
                        while (c < 4UL) {
                            long iw = iw0 + (long)c;
                            d[4UL * r + c] = ih >= 0L && (unsigned long)ih < j.inH && iw >= 0L && (unsigned long)iw < j.inW
                                             ? plane[(unsigned long)ih * j.inW + (unsigned long)iw] : (float)0.0;
                            c = c + 1UL;
                        }
                    }
                    r = r + 1UL;
                }
                float t[16];   // B^T d
                unsigned long c = 0UL;
                while (c < 4UL) {
                    float d0 = d[c]; float d1 = d[4UL + c]; float d2 = d[8UL + c]; float d3 = d[12UL + c];
                    t[c] = d0 - d2;
                    t[4UL + c] = d1 + d2;
                    t[8UL + c] = d2 - d1;
                    t[12UL + c] = d1 - d3;
                    c = c + 1UL;
                }
                r = 0UL;
                while (r < 4UL) {   // (B^T d) B: row r
                    float a0 = t[4UL * r]; float a1 = t[4UL * r + 1UL]; float a2 = t[4UL * r + 2UL]; float a3 = t[4UL * r + 3UL];
                    float* vr = &j.v[(4UL * r) * pv + ic * T + tile];
                    vr[0] = a0 - a2;
                    vr[pv] = a1 + a2;
                    vr[2UL * pv] = a2 - a1;
                    vr[3UL * pv] = a1 - a3;
                    r = r + 1UL;
                }
                tile = tile + 1UL;
            }
            ic = ic + 1UL;
        }
    }
}

// Phase 2, positions [x0, x1): M_x = U_x V_x over all tiles.
static void wino_gemm_(&WinoJob j, unsigned long x0, unsigned long x1) {
    unsigned long x = x0;
    unsafe {
        while (x < x1) {
            sgemm_serial(SGEMM_NO_TRANS, SGEMM_NO_TRANS, j.outC, j.tiles, j.inC, (float)1.0,
                         (const &float)&j.u[x * j.outC * j.inC], j.inC,
                         (const &float)&j.v[x * j.pv], j.tiles, (float)0.0,
                         (&float)&j.m[x * j.pm], j.tiles);
            x = x + 1UL;
        }
    }
}

// Phase 3, tiles [t0, t1): Y = A^T M A + bias (ReLU), 2 x 2 per tile,
// clipped at the edges.
static void wino_out_(&WinoJob j, unsigned long t0, unsigned long t1) {
    unsafe {
        unsigned long T = j.tiles;
        unsigned long oc = 0UL;
        while (oc < j.outC) {
            float b = j.bias[oc];
            float* oplane = &j.out[oc * j.outH * j.outW];
            const float* mb = &j.m[oc * T];
            unsigned long tile = t0;
            while (tile < t1) {
                unsigned long oh = 2UL * (tile / j.tilesW);
                unsigned long ow = 2UL * (tile % j.tilesW);
                if (tile % j.tilesW + 8UL <= j.tilesW && tile + 8UL <= t1 && oh + 2UL <= j.outH && ow + 16UL <= j.outW) {
                    cnn_f32x8_ mv[16];
                    unsigned long q8 = 0UL;
                    while (q8 < 16UL) { mv[q8] = *(const cnn_f32x8_*)&mb[q8 * j.pm + tile]; q8 = q8 + 1UL; }
                    cnn_f32x8_ bv; bv[0] = b; bv[1] = b; bv[2] = b; bv[3] = b; bv[4] = b; bv[5] = b; bv[6] = b; bv[7] = b;
                    unsigned long r8 = 0UL;
                    while (r8 < 2UL) {
                        cnn_f32x8_ s0; cnn_f32x8_ s1; cnn_f32x8_ s2; cnn_f32x8_ s3;
                        if (r8 == 0UL) {
                            s0 = mv[0] + mv[4] + mv[8];  s1 = mv[1] + mv[5] + mv[9];
                            s2 = mv[2] + mv[6] + mv[10]; s3 = mv[3] + mv[7] + mv[11];
                        } else {
                            s0 = mv[4] - mv[8] - mv[12];  s1 = mv[5] - mv[9] - mv[13];
                            s2 = mv[6] - mv[10] - mv[14]; s3 = mv[7] - mv[11] - mv[15];
                        }
                        cnn_f32x8_ y0 = s0 + s1 + s2 + bv;
                        cnn_f32x8_ y1 = s1 - s2 - s3 + bv;
                        float* orow = &oplane[(oh + r8) * j.outW + ow];
                        unsigned long k8 = 0UL;
                        while (k8 < 8UL) {
                            float a0 = y0[k8];
                            float a1 = y1[k8];
                            if (j.relu) {
                                a0 = a0 > (float)0.0 ? a0 : (float)0.0;
                                a1 = a1 > (float)0.0 ? a1 : (float)0.0;
                            }
                            orow[2UL * k8] = a0;
                            orow[2UL * k8 + 1UL] = a1;
                            k8 = k8 + 1UL;
                        }
                        r8 = r8 + 1UL;
                    }
                    tile = tile + 8UL;
                    continue;
                }
                float m[16];
                unsigned long q = 0UL;
                while (q < 16UL) { m[q] = mb[q * j.pm + tile]; q = q + 1UL; }
                float s[8];   // A^T m: 2 x 4
                unsigned long c = 0UL;
                while (c < 4UL) {
                    s[c] = m[c] + m[4UL + c] + m[8UL + c];
                    s[4UL + c] = m[4UL + c] - m[8UL + c] - m[12UL + c];
                    c = c + 1UL;
                }
                unsigned long r = 0UL;
                while (r < 2UL) {
                    if (oh + r < j.outH) {
                        float y0 = s[4UL * r] + s[4UL * r + 1UL] + s[4UL * r + 2UL] + b;
                        float y1 = s[4UL * r + 1UL] - s[4UL * r + 2UL] - s[4UL * r + 3UL] + b;
                        if (j.relu) {
                            y0 = y0 > (float)0.0 ? y0 : (float)0.0;
                            y1 = y1 > (float)0.0 ? y1 : (float)0.0;
                        }
                        float* orow = &oplane[(oh + r) * j.outW + ow];
                        orow[0] = y0;
                        if (ow + 1UL < j.outW) orow[1] = y1;
                    }
                    r = r + 1UL;
                }
                tile = tile + 1UL;
            }
            oc = oc + 1UL;
        }
    }
}

// 0 if the buffers can't be allocated (then im2col runs instead).
static int wino_run_(const &Conv2D layer, const &FeatureMap input, &FeatureMap out, int relu) {
    unsigned long inC; unsigned long outC; unsigned long pad;
    unsafe { inC = input->channels; outC = layer->outChannels; pad = layer->padding; }
    struct WinoJob j;
    unsafe {
        j.in = (const float*)input->data;
        j.bias = (const float*)layer->bias;
        j.out = (float*)out.data;
        j.inH = input->height;
        j.inW = input->width;
    }
    j.inC = inC;
    j.outC = outC;
    j.pad = pad;
    j.outH = out.height;
    j.outW = out.width;
    j.tilesW = (j.outW + 1UL) / 2UL;
    j.tiles = j.tilesW * ((j.outH + 1UL) / 2UL);
    j.relu = relu;
    // One buffer, kept by the calling thread: U, then V, then M. The 16
    // planes of V and of M are each skewed by one cache line (16 floats):
    // at power-of-two sizes a tile's 16 values would otherwise land a
    // multiple of 4 KB apart, all in one L1 set, and evict each other.
    unsigned long nu = 16UL * outC * inC;
    j.pv = inC * j.tiles + 16UL;
    j.pm = outC * j.tiles + 16UL;
    unsigned long need = nu + 16UL * (j.pv + j.pm);
    unsafe {
        if (g_wino_cap < need) {
            free((void*)g_wino_buf);
            g_wino_buf = (float*)malloc(sizeof(float) * need);
            g_wino_cap = g_wino_buf == (float*)0 ? 0UL : need;
            if (g_wino_buf == (float*)0) return 0;
        }
        j.u = (const float*)g_wino_buf;
        j.v = &g_wino_buf[nu];
        j.m = &g_wino_buf[nu + 16UL * j.pv];
        struct WinoFilterJob fj;
        fj.w = (const float*)layer->weight;
        fj.u = g_wino_buf;
        fj.outC = outC;
        fj.inC = inC;
        int par = j.tiles * inC * outC >= 65536UL;
        if (par) {
            parallel_for(outC, 4UL, wino_filters_, &fj);
            parallel_for(j.tiles, 16UL, wino_in_, &j);
            parallel_for(16UL, 1UL, wino_gemm_, &j);
            parallel_for(j.tiles, 16UL, wino_out_, &j);
        } else {
            wino_filters_(&fj, 0UL, outC);
            wino_in_(&j, 0UL, j.tiles);
            wino_gemm_(&j, 0UL, 16UL);
            wino_out_(&j, 0UL, j.tiles);
        }
    }
    return 1;
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
        if (kH == 3UL && kW == 3UL && stride == 1UL && inC >= 8UL && outC >= 8UL && pixels >= 64UL &&
            wino_run_(layer, input, &out, relu)) {
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
