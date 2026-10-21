// SafeC Standard Library — trainable convolutional networks (see convnet.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/convnet.h>
#include <std/ml/cnn.h>
#include <std/ml/tensor.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/nn.h>
#include <std/simd/gemm.h>
#include <std/math.h>
#include <std/panic.h>
#include <std/sync/parallel.h>

namespace std {

// ══ Geometry ═════════════════════════════════════════════════════════════════

unsigned long conv_out_side_ex(unsigned long dim, unsigned long kernel, unsigned long stride,
                               unsigned long padding, unsigned long dilation) {
    unsigned long span = dilation * (kernel - 1UL) + 1UL;
    if (kernel == 0UL || stride == 0UL || dim + 2UL * padding < span) return 0UL;
    return (dim + 2UL * padding - span) / stride + 1UL;
}

unsigned long conv_transpose_out_side_ex(unsigned long dim, unsigned long kernel, unsigned long stride,
                                         unsigned long padding, unsigned long dilation, unsigned long outPad) {
    if (dim == 0UL || kernel == 0UL) return 0UL;
    unsigned long full = (dim - 1UL) * stride + dilation * (kernel - 1UL) + outPad + 1UL;
    if (full < 2UL * padding) return 0UL;
    return full - 2UL * padding;
}

unsigned long conv_out_side(unsigned long dim, unsigned long kernel, unsigned long stride, unsigned long padding) {
    return conv_out_side_ex(dim, kernel, stride, padding, 1UL);
}

unsigned long conv_transpose_out_side(unsigned long dim, unsigned long kernel, unsigned long stride, unsigned long padding) {
    return conv_transpose_out_side_ex(dim, kernel, stride, padding, 1UL, 0UL);
}

unsigned long pool_out_side(unsigned long dim, unsigned long kernel, unsigned long stride,
                            unsigned long padding, int ceilMode) {
    if (kernel == 0UL || stride == 0UL || dim + 2UL * padding < kernel) return 0UL;
    unsigned long span = dim + 2UL * padding - kernel;
    unsigned long out = (ceilMode ? (span + stride - 1UL) / stride : span / stride) + 1UL;
    // the last window must start inside the input or its left padding
    if (ceilMode && (out - 1UL) * stride >= dim + padding) out = out - 1UL;
    return out;
}

struct Conv2dSpec Conv2dSpec::new(unsigned long kernel, unsigned long stride, unsigned long padding) {
    struct Conv2dSpec s;
    s.kH = kernel; s.kW = kernel;
    s.strideH = stride; s.strideW = stride;
    s.padH = padding; s.padW = padding;
    s.dilationH = 1UL; s.dilationW = 1UL;
    s.groups = 1UL;
    s.outPadH = 0UL; s.outPadW = 0UL;
    return s;
}

struct Pool2dSpec Pool2dSpec::new(unsigned long kernel, unsigned long stride, unsigned long padding) {
    struct Pool2dSpec s;
    s.kH = kernel; s.kW = kernel;
    s.strideH = stride; s.strideW = stride;
    s.padH = padding; s.padW = padding;
    s.ceilMode = 0;
    s.countIncludePad = 1;
    return s;
}

// A window over a "large" C x H x W image producing an oC x oH x oW map:
// conv2d reads it forward (x is the large side), conv_transpose2d
// backward (x is the small side), pooling with oC == C. Per axis: kernel,
// stride, padding, dilation.
struct CnGeom_ {
    unsigned long C;
    unsigned long H;
    unsigned long W;
    unsigned long kH;
    unsigned long kW;
    unsigned long sH;
    unsigned long sW;
    unsigned long pH;
    unsigned long pW;
    unsigned long dH;
    unsigned long dW;
    unsigned long groups;
    unsigned long oC;
    unsigned long oH;
    unsigned long oW;
    unsigned long mode;   // pooling: CN_POOL_*; upsample: the factor
};

#define CN_AUX_LEN 16UL

static struct CnGeom_ cn_geom_blank_() {
    struct CnGeom_ g;
    g.C = 0UL; g.H = 0UL; g.W = 0UL;
    g.kH = 1UL; g.kW = 1UL; g.sH = 1UL; g.sW = 1UL; g.pH = 0UL; g.pW = 0UL; g.dH = 1UL; g.dW = 1UL;
    g.groups = 1UL;
    g.oC = 0UL; g.oH = 0UL; g.oW = 0UL;
    g.mode = 0UL;
    return g;
}

// The geometry rides along as an aux parent (no gradient), like nn.sc's
// unary ops keep their constants.
static &FloatTensor cn_aux_(const &CnGeom_ g) {
    unsigned long shape[1];
    unsafe { shape[0] = CN_AUX_LEN; }
    &FloatTensor a = __tensor_alloc(shape, 1UL, 0);
    unsafe {
        a->data[0] = (float)g.C;   a->data[1] = (float)g.H;   a->data[2] = (float)g.W;
        a->data[3] = (float)g.kH;  a->data[4] = (float)g.kW;  a->data[5] = (float)g.sH;
        a->data[6] = (float)g.sW;  a->data[7] = (float)g.pH;  a->data[8] = (float)g.pW;
        a->data[9] = (float)g.dH;  a->data[10] = (float)g.dW; a->data[11] = (float)g.groups;
        a->data[12] = (float)g.oC; a->data[13] = (float)g.oH; a->data[14] = (float)g.oW;
        a->data[15] = (float)g.mode;
    }
    return a;
}

static struct CnGeom_ cn_geom_of_(FloatTensor* a) {
    struct CnGeom_ g;
    unsafe {
        g.C = (unsigned long)a->data[0];   g.H = (unsigned long)a->data[1];   g.W = (unsigned long)a->data[2];
        g.kH = (unsigned long)a->data[3];  g.kW = (unsigned long)a->data[4];  g.sH = (unsigned long)a->data[5];
        g.sW = (unsigned long)a->data[6];  g.pH = (unsigned long)a->data[7];  g.pW = (unsigned long)a->data[8];
        g.dH = (unsigned long)a->data[9];  g.dW = (unsigned long)a->data[10]; g.groups = (unsigned long)a->data[11];
        g.oC = (unsigned long)a->data[12]; g.oH = (unsigned long)a->data[13]; g.oW = (unsigned long)a->data[14];
        g.mode = (unsigned long)a->data[15];
    }
    return g;
}

static &FloatTensor cn_alloc2_(unsigned long rows, unsigned long cols) {
    unsigned long shape[2];
    unsafe { shape[0] = rows; shape[1] = cols; }
    return __tensor_alloc(shape, 2UL, 0);
}

static &FloatTensor cn_alloc1_(unsigned long n) {
    unsigned long shape[1];
    unsafe { shape[0] = n; }
    return __tensor_alloc(shape, 1UL, 0);
}

// Frees a tensor that is not part of any graph (heap tensors only).
static void cn_drop_(&FloatTensor t) {
    unsafe {
        match (t->owner) {
            case some(a): { }
            case null: { t->free(); free((void*)t); }
        }
    }
}

// Images in x: its rows, or 1 for a single image as a 1D tensor.
static unsigned long cn_batch_(const &FloatTensor x, unsigned long perImage, const &char op) {
    unsigned long rows;
    unsigned long size;
    unsafe {
        rows = x->ndim >= 2UL ? x->shape[0] : 1UL;
        size = x->size;
    }
    if (perImage == 0UL || rows * perImage != size) panic_at(op, "convnet.sc", __LINE__);
    return rows;
}

static float* cn_scratch_(unsigned long n) {
    unsafe {
        float* p = (float*)malloc(sizeof(float) * (n == 0UL ? 1UL : n));
        if (p == (float*)0) panic_at("convnet: out of memory", "convnet.sc", __LINE__);
        return p;
    }
}

// Splits n independent units of 'unit' elements each across the thread
// pool once there are TENSOR_EW_PAR elements in all, at least ~16K
// elements per chunk; otherwise runs body(ctx, 0, n) on the caller. Every
// pass handed to it writes disjoint outputs per unit, so the result is the
// same either way.
generic<T>
static void cn_split_(unsigned long n, unsigned long unit, fn void(&T, unsigned long, unsigned long) body, &T ctx) {
    if (n > 1UL && n * unit >= TENSOR_EW_PAR) {
        unsigned long grain = unit >= 16384UL ? 1UL : 16384UL / (unit == 0UL ? 1UL : unit);
        parallel_for(n, grain, body, ctx);
    } else {
        body(ctx, 0UL, n);
    }
}

// col[(c - c0, kh, kw)][oh*oW + ow] = img[c][oh*sH + kh*dH - pH][ow*sW + kw*dW - pW]
// for channels c0 .. c0 + nc, 0 in the padding.
static void cn_im2col_(const float* img, float* col, const &CnGeom_ g, unsigned long c0, unsigned long nc) {
    unsafe {
        unsigned long P = g.oH * g.oW;
        unsigned long r = 0UL;
        unsigned long c = c0;
        while (c < c0 + nc) {
            const float* plane = &img[c * g.H * g.W];
            unsigned long kh = 0UL;
            while (kh < g.kH) {
                unsigned long kw = 0UL;
                while (kw < g.kW) {
                    float* dst = &col[r * P];
                    unsigned long oh = 0UL;
                    while (oh < g.oH) {
                        long ih = (long)(oh * g.sH + kh * g.dH) - (long)g.pH;
                        unsigned long ow = 0UL;
                        if (ih < 0L || ih >= (long)g.H) {
                            while (ow < g.oW) { dst[oh * g.oW + ow] = (float)0.0; ow = ow + 1UL; }
                        } else {
                            const float* src = &plane[(unsigned long)ih * g.W];
                            while (ow < g.oW) {
                                long iw = (long)(ow * g.sW + kw * g.dW) - (long)g.pW;
                                dst[oh * g.oW + ow] = (iw < 0L || iw >= (long)g.W) ? (float)0.0 : src[(unsigned long)iw];
                                ow = ow + 1UL;
                            }
                        }
                        oh = oh + 1UL;
                    }
                    r = r + 1UL;
                    kw = kw + 1UL;
                }
                kh = kh + 1UL;
            }
            c = c + 1UL;
        }
    }
}

// The adjoint of cn_im2col_: img += scatter(col).
static void cn_col2im_acc_(const float* col, float* img, const &CnGeom_ g, unsigned long c0, unsigned long nc) {
    unsafe {
        unsigned long P = g.oH * g.oW;
        unsigned long r = 0UL;
        unsigned long c = c0;
        while (c < c0 + nc) {
            float* plane = &img[c * g.H * g.W];
            unsigned long kh = 0UL;
            while (kh < g.kH) {
                unsigned long kw = 0UL;
                while (kw < g.kW) {
                    const float* src = &col[r * P];
                    unsigned long oh = 0UL;
                    while (oh < g.oH) {
                        long ih = (long)(oh * g.sH + kh * g.dH) - (long)g.pH;
                        if (ih >= 0L && ih < (long)g.H) {
                            float* dst = &plane[(unsigned long)ih * g.W];
                            unsigned long ow = 0UL;
                            while (ow < g.oW) {
                                long iw = (long)(ow * g.sW + kw * g.dW) - (long)g.pW;
                                if (iw >= 0L && iw < (long)g.W) dst[(unsigned long)iw] = dst[(unsigned long)iw] + src[oh * g.oW + ow];
                                ow = ow + 1UL;
                            }
                        }
                        oh = oh + 1UL;
                    }
                    r = r + 1UL;
                    kw = kw + 1UL;
                }
                kh = kh + 1UL;
            }
            c = c + 1UL;
        }
    }
}

// out[c][0 .. plane) += bias[c]
static void cn_add_bias_planes_(float* out, const float* bias, unsigned long channels, unsigned long plane) {
    unsafe {
        unsigned long c = 0UL;
        while (c < channels) {
            float b = bias[c];
            float* row = &out[c * plane];
            unsigned long q = 0UL;
            while (q < plane) { row[q] = row[q] + b; q = q + 1UL; }
            c = c + 1UL;
        }
    }
}

// grad[c] += sum of g[c][0 .. plane)
static void cn_bias_grad_(float* grad, const float* g, unsigned long channels, unsigned long plane) {
    unsafe {
        unsigned long c = 0UL;
        while (c < channels) {
            const float* row = &g[c * plane];
            float acc = (float)0.0;
            unsigned long q = 0UL;
            while (q < plane) { acc = acc + row[q]; q = q + 1UL; }
            grad[c] = grad[c] + acc;
            c = c + 1UL;
        }
    }
}

// Links out to (x, weight, bias, aux), or drops aux when no gradient flows.
static void cn_link_param_op_(&FloatTensor out, &FloatTensor x, &FloatTensor w, &FloatTensor b, &FloatTensor aux, void* backwardFn) {
    unsafe {
        if (Autograd::get().enabled() && (x->requiresGrad || w->requiresGrad || b->requiresGrad)) {
            out->requiresGrad = 1;
            out->parents.push(x);
            out->parents.push(w);
            out->parents.push(b);
            out->parents.push(aux);
            out->gradFn = backwardFn;
        } else {
            out->requiresGrad = 0;
            cn_drop_(aux);
        }
    }
}

// ══ Convolutions ═════════════════════════════════════════════════════════════
// Per image and group, with Cg = C / groups channels of the large side,
// oCg = oC / groups of the small side, K = Cg*kH*kW rows of im2col and
// P = oH*oW output pixels (the weight is [oC, K] in both cases):
//   conv2d forward    col = im2col(x_g); out_g = W_g [oCg, K] . col + b
//   conv2d backward   dW_g += dout_g . col^T; dx_g += col2im(W_g^T . dout_g)
//   deconv forward    out_g = col2im(W_g^T [K, oCg] . x_g) + b
//   deconv backward   dcol = im2col(dout_g); dx_g += W_g . dcol; dW_g += x_g . dcol^T
// (for the transposed convolution the geometry's large side is the output,
// C = outC, and its small side the input, oC = inC).
//
// With enough work and more than one thread, the batch is split into one
// chunk of images per thread (parallel_for) running serial GEMMs on its own
// scratch. Input gradients of different images never overlap; weight and
// bias gradients go to per-chunk partial sums added up in chunk order
// afterwards, so a result does not depend on scheduling. Otherwise the
// images run in order on the caller, and sgemm itself may use the pool.

#define CN_CONV_FWD   0
#define CN_CONV_BWD   1
#define CN_DECONV_FWD 2
#define CN_DECONV_BWD 3
#define CN_PAR_WORK   262144UL   // multiply-adds per batch before splitting it

struct CnJob_ {
    struct CnGeom_ g;
    int           pass;
    int           serialGemm;
    const float*  x;         // the op's input
    const float*  w;
    const float*  bias;
    float*        out;       // forward output (pre-zeroed for the deconv)
    const float*  dout;      // backward: the output's gradient
    float*        dx;        // null: no input gradient
    float*        dw;        // weight-gradient sums, wSize per chunk (null: none)
    float*        db;        // bias-gradient sums, bSize per chunk (null: none)
    unsigned long wSize;
    unsigned long bSize;
    unsigned long batch;
    unsigned long perChunk;
};

static void cn_gemm_(int serial, int ta, int tb, unsigned long m, unsigned long n, unsigned long k,
                     const float* a, unsigned long lda, const float* b, unsigned long ldb,
                     float beta, float* c, unsigned long ldc) {
    unsafe {
        if (serial) sgemm_serial(ta, tb, m, n, k, (float)1.0, (const &float)a, lda, (const &float)b, ldb, beta, (&float)c, ldc);
        else sgemm(ta, tb, m, n, k, (float)1.0, (const &float)a, lda, (const &float)b, ldb, beta, (&float)c, ldc);
    }
}

static void cn_image_(const &CnJob_ j, unsigned long n, float* col, float* dw, float* db) {
    unsafe {
        struct CnGeom_ g = j.g;
        unsigned long Cg = g.C / g.groups;
        unsigned long oCg = g.oC / g.groups;
        unsigned long K = Cg * g.kH * g.kW;
        unsigned long P = g.oH * g.oW;
        unsigned long HW = g.H * g.W;
        unsigned long big = g.C * HW;
        unsigned long small = g.oC * P;
        int s = j.serialGemm;
        unsigned long gi = 0UL;
        while (gi < g.groups) {
            unsigned long c0 = gi * Cg;
            const float* wg = &j.w[gi * oCg * K];
            float* dwg = dw == (float*)0 ? (float*)0 : (float*)&dw[gi * oCg * K];
            if (j.pass == CN_CONV_FWD) {
                cn_im2col_(&j.x[n * big], col, &g, c0, Cg);
                cn_gemm_(s, SGEMM_NO_TRANS, SGEMM_NO_TRANS, oCg, P, K, wg, K, (const float*)col, P, (float)0.0,
                         &j.out[n * small + gi * oCg * P], P);
            } else if (j.pass == CN_CONV_BWD) {
                const float* dout = &j.dout[n * small + gi * oCg * P];
                if (dwg != (float*)0) {
                    cn_im2col_(&j.x[n * big], col, &g, c0, Cg);
                    cn_gemm_(s, SGEMM_NO_TRANS, SGEMM_TRANS, oCg, K, P, dout, P, (const float*)col, P, (float)1.0, dwg, K);
                }
                if (j.dx != (float*)0) {
                    cn_gemm_(s, SGEMM_TRANS, SGEMM_NO_TRANS, K, P, oCg, wg, K, dout, P, (float)0.0, col, P);
                    cn_col2im_acc_((const float*)col, &j.dx[n * big], &g, c0, Cg);
                }
            } else if (j.pass == CN_DECONV_FWD) {
                cn_gemm_(s, SGEMM_TRANS, SGEMM_NO_TRANS, K, P, oCg, wg, K, &j.x[n * small + gi * oCg * P], P,
                         (float)0.0, col, P);
                cn_col2im_acc_((const float*)col, &j.out[n * big], &g, c0, Cg);
            } else {
                if (j.dx != (float*)0 || dwg != (float*)0) cn_im2col_(&j.dout[n * big], col, &g, c0, Cg);
                if (j.dx != (float*)0)
                    cn_gemm_(s, SGEMM_NO_TRANS, SGEMM_NO_TRANS, oCg, P, K, wg, K, (const float*)col, P, (float)1.0,
                             &j.dx[n * small + gi * oCg * P], P);
                if (dwg != (float*)0)
                    cn_gemm_(s, SGEMM_NO_TRANS, SGEMM_TRANS, oCg, K, P, &j.x[n * small + gi * oCg * P], P,
                             (const float*)col, P, (float)1.0, dwg, K);
            }
            gi = gi + 1UL;
        }
        if (j.pass == CN_CONV_FWD) cn_add_bias_planes_(&j.out[n * small], j.bias, g.oC, P);
        else if (j.pass == CN_DECONV_FWD) cn_add_bias_planes_(&j.out[n * big], j.bias, g.C, HW);
        else if (db != (float*)0) {
            if (j.pass == CN_CONV_BWD) cn_bias_grad_(db, &j.dout[n * small], g.oC, P);
            else cn_bias_grad_(db, &j.dout[n * big], g.C, HW);
        }
    }
}

static void cn_chunks_(&CnJob_ j, unsigned long lo, unsigned long hi) {
    unsafe {
        float* col = cn_scratch_((j.g.C / j.g.groups) * j.g.kH * j.g.kW * j.g.oH * j.g.oW);
        unsigned long c = lo;
        while (c < hi) {
            float* dw = j.dw == (float*)0 ? (float*)0 : (float*)&j.dw[c * j.wSize];
            float* db = j.db == (float*)0 ? (float*)0 : (float*)&j.db[c * j.bSize];
            unsigned long n = c * j.perChunk;
            unsigned long end = n + j.perChunk < j.batch ? n + j.perChunk : j.batch;
            while (n < end) { cn_image_(j, n, col, dw, db); n = n + 1UL; }
            c = c + 1UL;
        }
        free((void*)col);
    }
}

// sums[i] += parts[c * size + i] over the chunks, in chunk order.
static void cn_reduce_(float* sums, const float* parts, unsigned long chunks, unsigned long size) {
    unsafe {
        unsigned long c = 0UL;
        while (c < chunks) {
            __tensor_ew(EW_ACC, &parts[c * size], (const float*)0, (float)0.0, sums, size);
            c = c + 1UL;
        }
    }
}

// Runs j over its batch; dwOut / dbOut (or null) receive the weight and
// bias gradients.
static void cn_run_(&CnJob_ j, float* dwOut, float* dbOut) {
    struct CnGeom_ g = j.g;
    unsigned long work = j.batch * g.oC * (g.C / g.groups) * g.kH * g.kW * g.oH * g.oW;
    unsigned long threads = parallel_threads();
    unsigned long chunks = 1UL;
    if (j.batch > 1UL && threads > 1UL && work >= CN_PAR_WORK) chunks = threads < j.batch ? threads : j.batch;
    j.perChunk = (j.batch + chunks - 1UL) / chunks;
    chunks = (j.batch + j.perChunk - 1UL) / j.perChunk;
    unsafe {
        if (chunks <= 1UL) {
            j.serialGemm = 0;
            j.dw = dwOut;
            j.db = dbOut;
            cn_chunks_(j, 0UL, 1UL);
            return;
        }
        j.serialGemm = 1;
        float* pw = (float*)0;
        float* pb = (float*)0;
        if (dwOut != (float*)0) {
            pw = cn_scratch_(chunks * j.wSize);
            __tensor_ew(EW_FILL, (const float*)0, (const float*)0, (float)0.0, pw, chunks * j.wSize);
        }
        if (dbOut != (float*)0) {
            pb = cn_scratch_(chunks * j.bSize);
            __tensor_ew(EW_FILL, (const float*)0, (const float*)0, (float)0.0, pb, chunks * j.bSize);
        }
        j.dw = pw;
        j.db = pb;
        parallel_for(chunks, 1UL, cn_chunks_, j);
        if (pw != (float*)0) { cn_reduce_(dwOut, (const float*)pw, chunks, j.wSize); free((void*)pw); }
        if (pb != (float*)0) { cn_reduce_(dbOut, (const float*)pb, chunks, j.bSize); free((void*)pb); }
    }
}

static struct CnJob_ cn_job_(const &CnGeom_ g, int pass, unsigned long batch) {
    struct CnJob_ j;
    j.g = *g;
    j.pass = pass;
    j.serialGemm = 0;
    unsafe {
        j.x = (const float*)0; j.w = (const float*)0; j.bias = (const float*)0; j.out = (float*)0;
        j.dout = (const float*)0; j.dx = (float*)0; j.dw = (float*)0; j.db = (float*)0;
    }
    j.wSize = g.oC * (g.C / g.groups) * g.kH * g.kW;
    j.bSize = pass <= CN_CONV_BWD ? g.oC : g.C;
    j.batch = batch;
    j.perChunk = batch;
    return j;
}

static void cn_conv_like_backward_(void* selfPtr, int pass) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* x = __tensor_parent(s, 0UL);
        FloatTensor* w = __tensor_parent(s, 1UL);
        FloatTensor* b = __tensor_parent(s, 2UL);
        struct CnGeom_ g = cn_geom_of_(__tensor_parent(s, 3UL));
        unsigned long outSize = pass == CN_CONV_BWD ? g.oC * g.oH * g.oW : g.C * g.H * g.W;
        struct CnJob_ j = cn_job_(&g, pass, s->size / outSize);
        j.x = (const float*)x->data;
        j.w = (const float*)w->data;
        j.dout = (const float*)s->grad;
        if (x->requiresGrad) { __tensor_ensure_grad(x); j.dx = (float*)x->grad; }
        float* dw = (float*)0;
        float* db = (float*)0;
        if (w->requiresGrad) { __tensor_ensure_grad(w); dw = (float*)w->grad; }
        if (b->requiresGrad) { __tensor_ensure_grad(b); db = (float*)b->grad; }
        cn_run_(&j, dw, db);
    }
}

static void cn_conv_backward_(void* selfPtr) { cn_conv_like_backward_(selfPtr, CN_CONV_BWD); }
static void cn_deconv_backward_(void* selfPtr) { cn_conv_like_backward_(selfPtr, CN_DECONV_BWD); }

static void cn_check_spec_(const &Conv2dSpec s, unsigned long inC, unsigned long outC, const &char op) {
    if (s.kH == 0UL || s.kW == 0UL || s.strideH == 0UL || s.strideW == 0UL || s.dilationH == 0UL
        || s.dilationW == 0UL || s.groups == 0UL || inC % s.groups != 0UL || outC % s.groups != 0UL)
        panic_at(op, "convnet.sc", __LINE__);
}

// cnn.h's forward kernels (Winograd F(2x2, 3x3) for 3x3 stride-1 layers
// with 8+ channels, a direct 1x1 path) on one image of the batch, through
// views of the tensors' memory.
static int cn_cnn_forward_ok_(const &CnGeom_ g) {
    return g.groups == 1UL && g.dH == 1UL && g.dW == 1UL && g.kH == g.kW && g.sH == g.sW && g.pH == g.pW
        && ((g.kH == 3UL && g.sH == 1UL && g.C >= 8UL && g.oC >= 8UL && g.oH * g.oW >= 64UL)
            || (g.kH == 1UL && g.sH == 1UL && g.pH == 0UL));
}

static void cn_cnn_forward_(const &CnGeom_ g, const float* x, const float* w, const float* bias, float* out,
                            unsigned long batch) {
    unsafe {
        struct Conv2D layer;
        layer.weight = (&heap float)w;
        layer.bias = (&heap float)bias;
        layer.inChannels = g.C;
        layer.outChannels = g.oC;
        layer.kH = g.kH;
        layer.kW = g.kW;
        layer.stride = g.sH;
        layer.padding = g.pH;
        unsigned long inSize = g.C * g.H * g.W;
        unsigned long outSize = g.oC * g.oH * g.oW;
        unsigned long n = 0UL;
        while (n < batch) {
            struct FeatureMap in;
            in.data = (&heap float)&x[n * inSize];
            in.channels = g.C;
            in.height = g.H;
            in.width = g.W;
            in.fromArena = 1;   // a view: FeatureMap::free must leave it
            struct FeatureMap o = layer.forward(&in);
            __tensor_ew(EW_COPY, (const float*)o.data, (const float*)0, (float)0.0, &out[n * outSize], outSize);
            o.free();
            n = n + 1UL;
        }
    }
}

&FloatTensor tensor_conv2d_ex(const &FloatTensor x, const &FloatTensor weight, const &FloatTensor bias,
                         unsigned long inC, unsigned long inH, unsigned long inW, const &Conv2dSpec spec) {
    unsigned long outC;
    unsafe { outC = bias->size; }
    cn_check_spec_(spec, inC, outC, "tensor_conv2d: kernel / stride / dilation 0, or channels not divisible by groups");
    struct CnGeom_ g = cn_geom_blank_();
    g.C = inC; g.H = inH; g.W = inW;
    g.kH = spec.kH; g.kW = spec.kW; g.sH = spec.strideH; g.sW = spec.strideW;
    g.pH = spec.padH; g.pW = spec.padW; g.dH = spec.dilationH; g.dW = spec.dilationW; g.groups = spec.groups;
    g.oC = outC;
    g.oH = conv_out_side_ex(inH, spec.kH, spec.strideH, spec.padH, spec.dilationH);
    g.oW = conv_out_side_ex(inW, spec.kW, spec.strideW, spec.padW, spec.dilationW);
    unsigned long P = g.oH * g.oW;
    unsigned long batch = cn_batch_(x, inC * inH * inW, "tensor_conv2d: x is not [batch, inC*inH*inW]");
    unsafe {
        if (weight->size != outC * (inC / spec.groups) * spec.kH * spec.kW || P == 0UL)
            panic_at("tensor_conv2d: weight is not [outC, (inC/groups)*kH*kW], or the kernel exceeds the padded input",
                     "convnet.sc", __LINE__);
    }
    &FloatTensor out = cn_alloc2_(batch, outC * P);
    unsafe {
        if (cn_cnn_forward_ok_(&g)) {
            cn_cnn_forward_(&g, (const float*)x->data, (const float*)weight->data, (const float*)bias->data,
                            (float*)out->data, batch);
        } else {
            struct CnJob_ j = cn_job_(&g, CN_CONV_FWD, batch);
            j.x = (const float*)x->data;
            j.w = (const float*)weight->data;
            j.bias = (const float*)bias->data;
            j.out = (float*)out->data;
            cn_run_(&j, (float*)0, (float*)0);
        }
    }
    cn_link_param_op_(out, x, weight, bias, cn_aux_(&g), (void*)cn_conv_backward_);
    return out;
}

&FloatTensor tensor_conv_transpose2d_ex(const &FloatTensor x, const &FloatTensor weight, const &FloatTensor bias,
                                   unsigned long inC, unsigned long inH, unsigned long inW, const &Conv2dSpec spec) {
    unsigned long outC;
    unsafe { outC = bias->size; }
    cn_check_spec_(spec, inC, outC, "tensor_conv_transpose2d: kernel / stride / dilation 0, or channels not divisible by groups");
    if (spec.outPadH >= spec.strideH && spec.outPadH >= spec.dilationH)
        panic_at("tensor_conv_transpose2d: outPadH must be below stride or dilation", "convnet.sc", __LINE__);
    if (spec.outPadW >= spec.strideW && spec.outPadW >= spec.dilationW)
        panic_at("tensor_conv_transpose2d: outPadW must be below stride or dilation", "convnet.sc", __LINE__);
    struct CnGeom_ g = cn_geom_blank_();
    g.C = outC;
    g.H = conv_transpose_out_side_ex(inH, spec.kH, spec.strideH, spec.padH, spec.dilationH, spec.outPadH);
    g.W = conv_transpose_out_side_ex(inW, spec.kW, spec.strideW, spec.padW, spec.dilationW, spec.outPadW);
    g.kH = spec.kH; g.kW = spec.kW; g.sH = spec.strideH; g.sW = spec.strideW;
    g.pH = spec.padH; g.pW = spec.padW; g.dH = spec.dilationH; g.dW = spec.dilationW; g.groups = spec.groups;
    g.oC = inC; g.oH = inH; g.oW = inW;
    unsigned long outSize = g.C * g.H * g.W;
    unsigned long batch = cn_batch_(x, inC * inH * inW, "tensor_conv_transpose2d: x is not [batch, inC*inH*inW]");
    unsafe {
        if (weight->size != inC * (outC / spec.groups) * spec.kH * spec.kW || outSize == 0UL)
            panic_at("tensor_conv_transpose2d: weight is not [inC, (outC/groups)*kH*kW]", "convnet.sc", __LINE__);
    }
    &FloatTensor out = cn_alloc2_(batch, outSize);
    struct CnJob_ j = cn_job_(&g, CN_DECONV_FWD, batch);
    unsafe {
        __tensor_ew(EW_FILL, (const float*)0, (const float*)0, (float)0.0, (float*)out->data, batch * outSize);
        j.x = (const float*)x->data;
        j.w = (const float*)weight->data;
        j.bias = (const float*)bias->data;
        j.out = (float*)out->data;
        cn_run_(&j, (float*)0, (float*)0);
    }
    cn_link_param_op_(out, x, weight, bias, cn_aux_(&g), (void*)cn_deconv_backward_);
    return out;
}

&FloatTensor tensor_conv2d(const &FloatTensor x, const &FloatTensor weight, const &FloatTensor bias,
                      unsigned long inC, unsigned long inH, unsigned long inW,
                      unsigned long kH, unsigned long kW,
                      unsigned long stride, unsigned long padding) {
    struct Conv2dSpec s = Conv2dSpec::new(kH, stride, padding);
    s.kW = kW;
    return tensor_conv2d_ex(x, weight, bias, inC, inH, inW, &s);
}

&FloatTensor tensor_conv_transpose2d(const &FloatTensor x, const &FloatTensor weight, const &FloatTensor bias,
                                unsigned long inC, unsigned long inH, unsigned long inW,
                                unsigned long kH, unsigned long kW,
                                unsigned long stride, unsigned long padding) {
    struct Conv2dSpec s = Conv2dSpec::new(kH, stride, padding);
    s.kW = kW;
    return tensor_conv_transpose2d_ex(x, weight, bias, inC, inH, inW, &s);
}

// ══ Pooling ══════════════════════════════════════════════════════════════════
// Windows start at o*stride - pad. Max pooling and avg pooling without
// countIncludePad use the part inside the plane; avg pooling with it
// divides by the window's extent inside the padded plane, as PyTorch.

#define CN_POOL_MAX       0UL
#define CN_POOL_AVG       1UL
#define CN_POOL_AVG_EXCL  2UL

struct CnWin_ {
    unsigned long h0;
    unsigned long h1;
    unsigned long w0;
    unsigned long w1;
    unsigned long padded;   // positions inside the padded plane
};

static struct CnWin_ cn_window_(const &CnGeom_ g, unsigned long oh, unsigned long ow) {
    struct CnWin_ r;
    long h0 = (long)(oh * g.sH) - (long)g.pH;
    long w0 = (long)(ow * g.sW) - (long)g.pW;
    long h1 = h0 + (long)g.kH;
    long w1 = w0 + (long)g.kW;
    long hp = h1 < (long)(g.H + g.pH) ? h1 : (long)(g.H + g.pH);
    long wp = w1 < (long)(g.W + g.pW) ? w1 : (long)(g.W + g.pW);
    r.padded = (unsigned long)((hp - h0) * (wp - w0));
    r.h0 = h0 < 0L ? 0UL : (unsigned long)h0;
    r.w0 = w0 < 0L ? 0UL : (unsigned long)w0;
    r.h1 = h1 > (long)g.H ? g.H : (unsigned long)h1;
    r.w1 = w1 > (long)g.W ? g.W : (unsigned long)w1;
    return r;
}

// Offset (within the image) of the window's (first) largest input.
static unsigned long cn_window_argmax_(const float* img, const &CnGeom_ g, unsigned long c, const &CnWin_ r) {
    unsafe {
        unsigned long plane = c * g.H * g.W;
        unsigned long best = plane + r.h0 * g.W + r.w0;
        float bv = img[best];
        unsigned long i = r.h0;
        while (i < r.h1) {
            unsigned long j = r.w0;
            while (j < r.w1) {
                unsigned long at = plane + i * g.W + j;
                if (img[at] > bv) { bv = img[at]; best = at; }
                j = j + 1UL;
            }
            i = i + 1UL;
        }
        return best;
    }
}

static float cn_window_divisor_(const &CnGeom_ g, const &CnWin_ r) {
    if (g.mode == CN_POOL_AVG) return (float)r.padded;
    return (float)((r.h1 - r.h0) * (r.w1 - r.w0));
}

// One (image, channel) plane per unit; a plane's windows touch only that
// plane, so planes never share an output or a gradient.
struct CnPoolJob_ {
    struct CnGeom_ g;
    const float* x;
    float*       out;     // forward
    const float* dout;    // backward
    float*       dx;
};

static void cn_pool_fwd_planes_(&CnPoolJob_ j, unsigned long lo, unsigned long hi) {
    unsafe {
        struct CnGeom_ g = j.g;
        unsigned long inSize = g.C * g.H * g.W;
        unsigned long outSize = g.C * g.oH * g.oW;
        unsigned long p = lo;
        while (p < hi) {
            unsigned long n = p / g.C;
            unsigned long c = p % g.C;
            const float* img = &j.x[n * inSize];
            float* o = &j.out[n * outSize];
            unsigned long oh = 0UL;
            while (oh < g.oH) {
                unsigned long ow = 0UL;
                while (ow < g.oW) {
                    struct CnWin_ r = cn_window_(&g, oh, ow);
                    float v;
                    if (g.mode == CN_POOL_MAX) {
                        v = img[cn_window_argmax_(img, &g, c, &r)];
                    } else {
                        float acc = (float)0.0;
                        unsigned long i = r.h0;
                        while (i < r.h1) {
                            unsigned long w = r.w0;
                            while (w < r.w1) { acc = acc + img[c * g.H * g.W + i * g.W + w]; w = w + 1UL; }
                            i = i + 1UL;
                        }
                        v = acc / cn_window_divisor_(&g, &r);
                    }
                    o[(c * g.oH + oh) * g.oW + ow] = v;
                    ow = ow + 1UL;
                }
                oh = oh + 1UL;
            }
            p = p + 1UL;
        }
    }
}

static void cn_pool_bwd_planes_(&CnPoolJob_ j, unsigned long lo, unsigned long hi) {
    unsafe {
        struct CnGeom_ g = j.g;
        unsigned long inSize = g.C * g.H * g.W;
        unsigned long outSize = g.C * g.oH * g.oW;
        unsigned long p = lo;
        while (p < hi) {
            unsigned long n = p / g.C;
            unsigned long c = p % g.C;
            const float* img = &j.x[n * inSize];
            float* dimg = &j.dx[n * inSize];
            const float* dout = &j.dout[n * outSize];
            unsigned long oh = 0UL;
            while (oh < g.oH) {
                unsigned long ow = 0UL;
                while (ow < g.oW) {
                    struct CnWin_ r = cn_window_(&g, oh, ow);
                    float d = dout[(c * g.oH + oh) * g.oW + ow];
                    if (g.mode == CN_POOL_MAX) {
                        unsigned long at = cn_window_argmax_(img, &g, c, &r);
                        dimg[at] = dimg[at] + d;
                    } else {
                        float share = d / cn_window_divisor_(&g, &r);
                        unsigned long i = r.h0;
                        while (i < r.h1) {
                            unsigned long w = r.w0;
                            while (w < r.w1) {
                                unsigned long at = c * g.H * g.W + i * g.W + w;
                                dimg[at] = dimg[at] + share;
                                w = w + 1UL;
                            }
                            i = i + 1UL;
                        }
                    }
                    ow = ow + 1UL;
                }
                oh = oh + 1UL;
            }
            p = p + 1UL;
        }
    }
}

static void cn_pool_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* x = __tensor_parent(s, 0UL);
        if (!x->requiresGrad) return;
        __tensor_ensure_grad(x);
        struct CnPoolJob_ j;
        j.g = cn_geom_of_(__tensor_parent(s, 1UL));
        j.x = (const float*)x->data;
        j.out = (float*)0;
        j.dout = (const float*)s->grad;
        j.dx = (float*)x->grad;
        unsigned long planes = s->size / (j.g.oH * j.g.oW);
        cn_split_(planes, j.g.H * j.g.W, cn_pool_bwd_planes_, &j);
    }
}

static &FloatTensor cn_pool_(const &FloatTensor x, unsigned long channels, unsigned long height, unsigned long width,
                        const &Pool2dSpec spec, unsigned long mode) {
    struct CnGeom_ g = cn_geom_blank_();
    g.C = channels; g.H = height; g.W = width;
    g.kH = spec.kH; g.kW = spec.kW; g.sH = spec.strideH; g.sW = spec.strideW; g.pH = spec.padH; g.pW = spec.padW;
    g.oC = channels;
    g.oH = pool_out_side(height, spec.kH, spec.strideH, spec.padH, spec.ceilMode);
    g.oW = pool_out_side(width, spec.kW, spec.strideW, spec.padW, spec.ceilMode);
    g.mode = mode;
    unsigned long inSize = channels * height * width;
    unsigned long outSize = channels * g.oH * g.oW;
    unsigned long batch = cn_batch_(x, inSize, "tensor_pool2d: x is not [batch, C*H*W]");
    if (outSize == 0UL) panic_at("tensor_pool2d: window larger than the padded image", "convnet.sc", __LINE__);
    if (2UL * spec.padH > spec.kH || 2UL * spec.padW > spec.kW)
        panic_at("tensor_pool2d: padding over half the window", "convnet.sc", __LINE__);
    &FloatTensor out = cn_alloc2_(batch, outSize);
    struct CnPoolJob_ j;
    j.g = g;
    unsafe {
        j.x = (const float*)x->data;
        j.out = (float*)out->data;
        j.dout = (const float*)0;
        j.dx = (float*)0;
    }
    cn_split_(batch * channels, height * width, cn_pool_fwd_planes_, &j);
    &FloatTensor aux = cn_aux_(&g);
    __tensor_link2(out, x, aux, (void*)cn_pool_backward_);
    unsafe { if (!out->requiresGrad) cn_drop_(aux); }
    return out;
}

&FloatTensor FloatTensor::maxpool2d_ex(unsigned long channels, unsigned long height, unsigned long width, const &Pool2dSpec spec) const {
    return cn_pool_(self, channels, height, width, spec, CN_POOL_MAX);
}

&FloatTensor FloatTensor::avgpool2d_ex(unsigned long channels, unsigned long height, unsigned long width, const &Pool2dSpec spec) const {
    return cn_pool_(self, channels, height, width, spec, spec.countIncludePad ? CN_POOL_AVG : CN_POOL_AVG_EXCL);
}

&FloatTensor FloatTensor::maxpool2d(unsigned long channels, unsigned long height, unsigned long width, unsigned long kernel,
                          unsigned long stride, unsigned long padding) const {
    struct Pool2dSpec s = Pool2dSpec::new(kernel, stride, padding);
    return self.maxpool2d_ex(channels, height, width, &s);
}

&FloatTensor FloatTensor::avgpool2d(unsigned long channels, unsigned long height, unsigned long width, unsigned long kernel,
                          unsigned long stride, unsigned long padding, int countIncludePad) const {
    struct Pool2dSpec s = Pool2dSpec::new(kernel, stride, padding);
    s.countIncludePad = countIncludePad;
    return self.avgpool2d_ex(channels, height, width, &s);
}

// ══ Global pooling ═══════════════════════════════════════════════════════════
// Each (image, channel) plane reduces to one value: [batch, C*H*W] ->
// [batch, C]. One plane per unit across the pool.

static unsigned long cn_plane_argmax_(const float* p, unsigned long n) {
    unsafe {
        unsigned long best = 0UL;
        unsigned long i = 1UL;
        while (i < n) { if (p[i] > p[best]) best = i; i = i + 1UL; }
        return best;
    }
}

static void cn_gpool_fwd_planes_(&CnPoolJob_ j, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long plane = j.g.H * j.g.W;
        unsigned long p = lo;
        while (p < hi) {
            const float* src = &j.x[p * plane];
            if (j.g.mode == CN_POOL_MAX) {
                j.out[p] = src[cn_plane_argmax_(src, plane)];
            } else {
                double acc = 0.0;
                unsigned long q = 0UL;
                while (q < plane) { acc = acc + (double)src[q]; q = q + 1UL; }
                j.out[p] = (float)(acc / (double)plane);
            }
            p = p + 1UL;
        }
    }
}

static void cn_gpool_bwd_planes_(&CnPoolJob_ j, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long plane = j.g.H * j.g.W;
        unsigned long p = lo;
        while (p < hi) {
            float d = j.dout[p];
            float* dst = &j.dx[p * plane];
            if (j.g.mode == CN_POOL_MAX) {
                unsigned long at = cn_plane_argmax_(&j.x[p * plane], plane);
                dst[at] = dst[at] + d;
            } else {
                float share = d / (float)plane;
                unsigned long q = 0UL;
                while (q < plane) { dst[q] = dst[q] + share; q = q + 1UL; }
            }
            p = p + 1UL;
        }
    }
}

static void cn_gpool_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* x = __tensor_parent(s, 0UL);
        if (!x->requiresGrad) return;
        __tensor_ensure_grad(x);
        struct CnPoolJob_ j;
        j.g = cn_geom_of_(__tensor_parent(s, 1UL));
        j.x = (const float*)x->data;
        j.out = (float*)0;
        j.dout = (const float*)s->grad;
        j.dx = (float*)x->grad;
        cn_split_(s->size, j.g.H * j.g.W, cn_gpool_bwd_planes_, &j);
    }
}

static &FloatTensor cn_gpool_(const &FloatTensor x, unsigned long channels, unsigned long height, unsigned long width,
                         unsigned long mode) {
    struct CnGeom_ g = cn_geom_blank_();
    g.C = channels; g.H = height; g.W = width; g.kH = height; g.kW = width;
    g.oC = channels; g.oH = 1UL; g.oW = 1UL;
    g.mode = mode;
    unsigned long batch = cn_batch_(x, channels * height * width, "tensor_global_pool2d: x is not [batch, C*H*W]");
    &FloatTensor out = cn_alloc2_(batch, channels);
    struct CnPoolJob_ j;
    j.g = g;
    unsafe {
        j.x = (const float*)x->data;
        j.out = (float*)out->data;
        j.dout = (const float*)0;
        j.dx = (float*)0;
    }
    cn_split_(batch * channels, height * width, cn_gpool_fwd_planes_, &j);
    &FloatTensor aux = cn_aux_(&g);
    __tensor_link2(out, x, aux, (void*)cn_gpool_backward_);
    unsafe { if (!out->requiresGrad) cn_drop_(aux); }
    return out;
}

&FloatTensor FloatTensor::global_avgpool2d(unsigned long channels, unsigned long height, unsigned long width) const {
    return cn_gpool_(self, channels, height, width, CN_POOL_AVG);
}

&FloatTensor FloatTensor::global_maxpool2d(unsigned long channels, unsigned long height, unsigned long width) const {
    return cn_gpool_(self, channels, height, width, CN_POOL_MAX);
}

// ══ Nearest upsampling by an integer factor ══════════════════════════════════
// Input row r (image, channel, h) becomes output rows r*f .. r*f + f; one
// input row per unit.

struct CnUpJob_ {
    const float* src;     // forward: x; backward: the output's gradient
    float*       dst;     // forward: out; backward: x's gradient
    unsigned long W;
    unsigned long f;
};

static void cn_up_fwd_rows_(&CnUpJob_ j, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long oW = j.W * j.f;
        unsigned long r = lo;
        while (r < hi) {
            const float* src = &j.src[r * j.W];
            unsigned long i = 0UL;
            while (i < j.f) {
                float* dst = &j.dst[(r * j.f + i) * oW];
                unsigned long w = 0UL;
                while (w < oW) { dst[w] = src[w / j.f]; w = w + 1UL; }
                i = i + 1UL;
            }
            r = r + 1UL;
        }
    }
}

static void cn_up_bwd_rows_(&CnUpJob_ j, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long oW = j.W * j.f;
        unsigned long r = lo;
        while (r < hi) {
            float* dst = &j.dst[r * j.W];
            unsigned long i = 0UL;
            while (i < j.f) {
                const float* src = &j.src[(r * j.f + i) * oW];
                unsigned long w = 0UL;
                while (w < j.W) {
                    float acc = (float)0.0;
                    unsigned long k = 0UL;
                    while (k < j.f) { acc = acc + src[w * j.f + k]; k = k + 1UL; }
                    dst[w] = dst[w] + acc;
                    w = w + 1UL;
                }
                i = i + 1UL;
            }
            r = r + 1UL;
        }
    }
}

static void cn_upsample_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* x = __tensor_parent(s, 0UL);
        if (!x->requiresGrad) return;
        __tensor_ensure_grad(x);
        struct CnGeom_ g = cn_geom_of_(__tensor_parent(s, 1UL));
        struct CnUpJob_ j;
        j.src = (const float*)s->grad;
        j.dst = (float*)x->grad;
        j.W = g.W;
        j.f = g.mode;
        cn_split_(x->size / g.W, g.W * g.mode * g.mode, cn_up_bwd_rows_, &j);
    }
}

&FloatTensor FloatTensor::upsample_nearest(unsigned long channels, unsigned long height, unsigned long width,
                                 unsigned long factor) const {
    struct CnGeom_ g = cn_geom_blank_();
    g.C = channels; g.H = height; g.W = width;
    g.oC = channels; g.oH = height * factor; g.oW = width * factor;
    g.mode = factor;
    unsigned long batch = cn_batch_(self, channels * height * width, "Tensor::upsample_nearest: x is not [batch, C*H*W]");
    if (factor == 0UL) panic_at("Tensor::upsample_nearest: factor 0", "convnet.sc", __LINE__);
    &FloatTensor out = cn_alloc2_(batch, channels * g.oH * g.oW);
    struct CnUpJob_ j;
    unsafe {
        j.src = (const float*)self->data;
        j.dst = (float*)out->data;
    }
    j.W = width;
    j.f = factor;
    cn_split_(batch * channels * height, width * factor * factor, cn_up_fwd_rows_, &j);
    &FloatTensor aux = cn_aux_(&g);
    __tensor_link2(out, self, aux, (void*)cn_upsample_backward_);
    unsafe { if (!out->requiresGrad) cn_drop_(aux); }
    return out;
}

// ══ Resizing ═════════════════════════════════════════════════════════════════
// Separable: along each axis output o reads up to 4 input positions with
// weights (1 for nearest, 2 for bilinear, 4 for bicubic), tabulated once
// per call; out = sum_i sum_j wy_i wx_j in[y_i][x_j]. One plane per unit.

#define CN_TAPS 4UL

struct CnTaps_ {
    unsigned long idx[4];
    float         w[4];
    unsigned long n;
};

static float cn_src_coord_(unsigned long o, unsigned long in, unsigned long out, int align, int clampZero) {
    if (align) {
        if (out <= 1UL) return (float)0.0;
        return (float)o * (float)(in - 1UL) / (float)(out - 1UL);
    }
    float s = ((float)o + (float)0.5) * (float)in / (float)out - (float)0.5;
    if (clampZero && s < (float)0.0) s = (float)0.0;
    return s;
}

static unsigned long cn_clamp_idx_(long i, unsigned long n) {
    if (i < 0L) return 0UL;
    if (i >= (long)n) return n - 1UL;
    return (unsigned long)i;
}

// Cubic convolution weights (Keys, a = -0.75, as PyTorch) for offsets
// -1, 0, 1, 2 from floor(src), t = src - floor(src).
static void cn_cubic_weights_(float t, float* w) {
    unsafe {
        float A = (float)-0.75;
        float x1 = t + (float)1.0;
        float x2 = (float)1.0 - t;
        float x3 = (float)2.0 - t;
        w[0] = ((A * x1 - (float)5.0 * A) * x1 + (float)8.0 * A) * x1 - (float)4.0 * A;
        w[1] = ((A + (float)2.0) * t - (A + (float)3.0)) * t * t + (float)1.0;
        w[2] = ((A + (float)2.0) * x2 - (A + (float)3.0)) * x2 * x2 + (float)1.0;
        w[3] = ((A * x3 - (float)5.0 * A) * x3 + (float)8.0 * A) * x3 - (float)4.0 * A;
    }
}

static struct CnTaps_ cn_taps_(unsigned long o, unsigned long in, unsigned long out, int mode, int align) {
    struct CnTaps_ t;
    unsafe {
        t.idx[0] = 0UL; t.idx[1] = 0UL; t.idx[2] = 0UL; t.idx[3] = 0UL;
        t.w[0] = (float)0.0; t.w[1] = (float)0.0; t.w[2] = (float)0.0; t.w[3] = (float)0.0;
        if (mode == RESIZE_NEAREST) {
            unsigned long i = (unsigned long)floor_f((float)o * ((float)in / (float)out));
            t.idx[0] = i < in ? i : in - 1UL;
            t.w[0] = (float)1.0;
            t.n = 1UL;
        } else if (mode == RESIZE_BILINEAR) {
            float s = cn_src_coord_(o, in, out, align, 1);
            unsigned long i0 = (unsigned long)s;
            if (i0 > in - 1UL) i0 = in - 1UL;
            float l = s - (float)i0;
            t.idx[0] = i0;
            t.idx[1] = i0 + 1UL < in ? i0 + 1UL : in - 1UL;
            t.w[0] = (float)1.0 - l;
            t.w[1] = l;
            t.n = 2UL;
        } else {
            float s = cn_src_coord_(o, in, out, align, 0);
            float f = floor_f(s);
            long i = (long)f;
            cn_cubic_weights_(s - f, (float*)&t.w[0]);
            unsigned long k = 0UL;
            while (k < 4UL) { t.idx[k] = cn_clamp_idx_(i - 1L + (long)k, in); k = k + 1UL; }
            t.n = 4UL;
        }
    }
    return t;
}

struct CnResizeJob_ {
    const float*  x;
    float*        out;
    const float*  dout;
    float*        dx;
    unsigned long H;
    unsigned long W;
    unsigned long oH;
    unsigned long oW;
    int mode;
    int align;
};

static void cn_resize_fwd_planes_(&CnResizeJob_ j, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long p = lo;
        while (p < hi) {
            const float* src = &j.x[p * j.H * j.W];
            float* dst = &j.out[p * j.oH * j.oW];
            unsigned long oh = 0UL;
            while (oh < j.oH) {
                struct CnTaps_ ty = cn_taps_(oh, j.H, j.oH, j.mode, j.align);
                unsigned long ow = 0UL;
                while (ow < j.oW) {
                    struct CnTaps_ tx = cn_taps_(ow, j.W, j.oW, j.mode, j.align);
                    float acc = (float)0.0;
                    unsigned long a = 0UL;
                    while (a < ty.n) {
                        const float* row = &src[ty.idx[a] * j.W];
                        float rv = (float)0.0;
                        unsigned long b = 0UL;
                        while (b < tx.n) { rv = rv + tx.w[b] * row[tx.idx[b]]; b = b + 1UL; }
                        acc = acc + ty.w[a] * rv;
                        a = a + 1UL;
                    }
                    dst[oh * j.oW + ow] = acc;
                    ow = ow + 1UL;
                }
                oh = oh + 1UL;
            }
            p = p + 1UL;
        }
    }
}

static void cn_resize_bwd_planes_(&CnResizeJob_ j, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long p = lo;
        while (p < hi) {
            const float* dout = &j.dout[p * j.oH * j.oW];
            float* dx = &j.dx[p * j.H * j.W];
            unsigned long oh = 0UL;
            while (oh < j.oH) {
                struct CnTaps_ ty = cn_taps_(oh, j.H, j.oH, j.mode, j.align);
                unsigned long ow = 0UL;
                while (ow < j.oW) {
                    struct CnTaps_ tx = cn_taps_(ow, j.W, j.oW, j.mode, j.align);
                    float d = dout[oh * j.oW + ow];
                    unsigned long a = 0UL;
                    while (a < ty.n) {
                        float* row = &dx[ty.idx[a] * j.W];
                        float da = d * ty.w[a];
                        unsigned long b = 0UL;
                        while (b < tx.n) { row[tx.idx[b]] = row[tx.idx[b]] + da * tx.w[b]; b = b + 1UL; }
                        a = a + 1UL;
                    }
                    ow = ow + 1UL;
                }
                oh = oh + 1UL;
            }
            p = p + 1UL;
        }
    }
}

// aux: [H, W, oH, oW, mode, align]
static void cn_resize_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* x = __tensor_parent(s, 0UL);
        FloatTensor* aux = __tensor_parent(s, 1UL);
        if (!x->requiresGrad) return;
        __tensor_ensure_grad(x);
        struct CnResizeJob_ j;
        j.H = (unsigned long)aux->data[0]; j.W = (unsigned long)aux->data[1];
        j.oH = (unsigned long)aux->data[2]; j.oW = (unsigned long)aux->data[3];
        j.mode = (int)aux->data[4]; j.align = (int)aux->data[5];
        j.x = (const float*)x->data;
        j.out = (float*)0;
        j.dout = (const float*)s->grad;
        j.dx = (float*)x->grad;
        cn_split_(x->size / (j.H * j.W), j.oH * j.oW, cn_resize_bwd_planes_, &j);
    }
}

&FloatTensor FloatTensor::resize2d(unsigned long channels, unsigned long height, unsigned long width, unsigned long outH,
                         unsigned long outW, int mode, int alignCorners) const {
    unsigned long batch = cn_batch_(self, channels * height * width, "Tensor::resize2d: x is not [batch, C*H*W]");
    if (outH == 0UL || outW == 0UL || mode < RESIZE_NEAREST || mode > RESIZE_BICUBIC)
        panic_at("Tensor::resize2d: empty output or unknown mode", "convnet.sc", __LINE__);
    &FloatTensor out = cn_alloc2_(batch, channels * outH * outW);
    &FloatTensor aux = cn_alloc1_(6UL);
    struct CnResizeJob_ j;
    j.H = height; j.W = width; j.oH = outH; j.oW = outW;
    j.mode = mode;
    j.align = mode == RESIZE_NEAREST ? 0 : alignCorners;
    unsafe {
        aux->data[0] = (float)height; aux->data[1] = (float)width;
        aux->data[2] = (float)outH; aux->data[3] = (float)outW;
        aux->data[4] = (float)j.mode; aux->data[5] = (float)j.align;
        j.x = (const float*)self->data;
        j.out = (float*)out->data;
        j.dout = (const float*)0;
        j.dx = (float*)0;
    }
    cn_split_(batch * channels, outH * outW * (mode == RESIZE_BICUBIC ? 16UL : 4UL), cn_resize_fwd_planes_, &j);
    __tensor_link2(out, self, aux, (void*)cn_resize_backward_);
    unsafe { if (!out->requiresGrad) cn_drop_(aux); }
    return out;
}

&FloatTensor FloatTensor::upsample_bilinear(unsigned long channels, unsigned long height, unsigned long width,
                                  unsigned long factor) const {
    if (factor == 0UL) panic_at("Tensor::upsample_bilinear: factor 0", "convnet.sc", __LINE__);
    return self.resize2d(channels, height, width, height * factor, width * factor, RESIZE_BILINEAR, 0);
}

// ══ Batch normalization ══════════════════════════════════════════════════════
// Per channel c over the batch and the plane (N = batch * plane values):
//   training  mean / var (biased) of the batch; the running statistics move
//             toward them by 'momentum' (the variance's unbiased estimate)
//   eval      the running statistics
//   y = gamma * (x - mean) * invstd + beta,  invstd = 1 / sqrt(var + eps)
// Backward, with xh the normalized input and dxh = dy * gamma:
//   training  dx = invstd / N * (N dxh - sum dxh - xh sum(dxh xh))
//   eval      dx = invstd * dxh
//   dgamma += sum dy xh,  dbeta += sum dy
// The aux parent holds channels, plane, the mode, then mean[C], invstd[C].
// Channels are independent, so both passes split across the pool by channel.

struct CnBnJob_ {
    const float* x;
    float*       out;
    const float* gamma;
    const float* beta;
    float*       runMean;
    float*       runVar;
    float*       mean;    // [C] in the aux tensor
    float*       inv;     // [C] in the aux tensor
    const float* dy;      // backward
    float*       dx;      // null: no input gradient
    float*       dgamma;  // null: none
    float*       dbeta;   // null: none
    unsigned long batch;
    unsigned long C;
    unsigned long plane;
    int   training;
    float momentum;
    float eps;
};

static void cn_bn_fwd_channels_(&CnBnJob_ j, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long N = j.batch * j.plane;
        unsigned long c = lo;
        while (c < hi) {
            float mean;
            float var;
            if (j.training) {
                double acc = 0.0;
                unsigned long n = 0UL;
                while (n < j.batch) {
                    const float* row = &j.x[(n * j.C + c) * j.plane];
                    unsigned long q = 0UL;
                    while (q < j.plane) { acc = acc + (double)row[q]; q = q + 1UL; }
                    n = n + 1UL;
                }
                double mu = acc / (double)N;
                double sq = 0.0;
                n = 0UL;
                while (n < j.batch) {
                    const float* row = &j.x[(n * j.C + c) * j.plane];
                    unsigned long q = 0UL;
                    while (q < j.plane) { double d = (double)row[q] - mu; sq = sq + d * d; q = q + 1UL; }
                    n = n + 1UL;
                }
                mean = (float)mu;
                var = (float)(sq / (double)N);
                float unbiased = N > 1UL ? (float)(sq / (double)(N - 1UL)) : var;
                j.runMean[c] = ((float)1.0 - j.momentum) * j.runMean[c] + j.momentum * mean;
                j.runVar[c] = ((float)1.0 - j.momentum) * j.runVar[c] + j.momentum * unbiased;
            } else {
                mean = j.runMean[c];
                var = j.runVar[c];
            }
            float inv = (float)1.0 / sqrt_f(var + j.eps);
            j.mean[c] = mean;
            j.inv[c] = inv;
            float gam = j.gamma[c];
            float bet = j.beta[c];
            unsigned long n = 0UL;
            while (n < j.batch) {
                unsigned long base = (n * j.C + c) * j.plane;
                unsigned long q = 0UL;
                while (q < j.plane) { j.out[base + q] = gam * (j.x[base + q] - mean) * inv + bet; q = q + 1UL; }
                n = n + 1UL;
            }
            c = c + 1UL;
        }
    }
}

static void cn_bn_bwd_channels_(&CnBnJob_ j, unsigned long lo, unsigned long hi) {
    unsafe {
        float N = (float)(j.batch * j.plane);
        unsigned long c = lo;
        while (c < hi) {
            float mean = j.mean[c];
            float inv = j.inv[c];
            double sdy = 0.0;
            double sdyxh = 0.0;
            unsigned long n = 0UL;
            while (n < j.batch) {
                unsigned long base = (n * j.C + c) * j.plane;
                unsigned long q = 0UL;
                while (q < j.plane) {
                    float dy = j.dy[base + q];
                    sdy = sdy + (double)dy;
                    sdyxh = sdyxh + (double)(dy * (j.x[base + q] - mean) * inv);
                    q = q + 1UL;
                }
                n = n + 1UL;
            }
            if (j.dgamma != (float*)0) j.dgamma[c] = j.dgamma[c] + (float)sdyxh;
            if (j.dbeta != (float*)0) j.dbeta[c] = j.dbeta[c] + (float)sdy;
            if (j.dx != (float*)0) {
                float k = j.gamma[c] * inv;
                float m1 = j.training ? (float)sdy / N : (float)0.0;
                float m2 = j.training ? (float)sdyxh / N : (float)0.0;
                n = 0UL;
                while (n < j.batch) {
                    unsigned long base = (n * j.C + c) * j.plane;
                    unsigned long q = 0UL;
                    while (q < j.plane) {
                        float xh = (j.x[base + q] - mean) * inv;
                        j.dx[base + q] = j.dx[base + q] + k * (j.dy[base + q] - m1 - xh * m2);
                        q = q + 1UL;
                    }
                    n = n + 1UL;
                }
            }
            c = c + 1UL;
        }
    }
}

static struct CnBnJob_ cn_bn_job_(FloatTensor* x, FloatTensor* gamma, FloatTensor* beta, FloatTensor* aux) {
    struct CnBnJob_ j;
    unsafe {
        j.C = (unsigned long)aux->data[0];
        j.plane = (unsigned long)aux->data[1];
        j.training = aux->data[2] != (float)0.0;
        j.batch = x->size / (j.C * j.plane);
        j.x = (const float*)x->data;
        j.gamma = (const float*)gamma->data;
        j.beta = (const float*)beta->data;
        j.mean = (float*)&aux->data[3];
        j.inv = (float*)&aux->data[3UL + j.C];
        j.out = (float*)0; j.runMean = (float*)0; j.runVar = (float*)0;
        j.dy = (const float*)0; j.dx = (float*)0; j.dgamma = (float*)0; j.dbeta = (float*)0;
    }
    j.momentum = (float)0.0;
    j.eps = (float)0.0;
    return j;
}

static void cn_bn_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* x = __tensor_parent(s, 0UL);
        FloatTensor* gm = __tensor_parent(s, 1UL);
        FloatTensor* bt = __tensor_parent(s, 2UL);
        struct CnBnJob_ j = cn_bn_job_(x, gm, bt, __tensor_parent(s, 3UL));
        j.dy = (const float*)s->grad;
        if (x->requiresGrad) { __tensor_ensure_grad(x); j.dx = (float*)x->grad; }
        if (gm->requiresGrad) { __tensor_ensure_grad(gm); j.dgamma = (float*)gm->grad; }
        if (bt->requiresGrad) { __tensor_ensure_grad(bt); j.dbeta = (float*)bt->grad; }
        cn_split_(j.C, j.batch * j.plane, cn_bn_bwd_channels_, &j);
    }
}

&FloatTensor tensor_batch_norm(const &FloatTensor x, const &FloatTensor gamma, const &FloatTensor beta,
                          &FloatTensor runMean, &FloatTensor runVar,
                          unsigned long channels, unsigned long plane,
                          int training, float momentum, float eps) {
    unsigned long batch = cn_batch_(x, channels * plane, "tensor_batch_norm: x is not [batch, channels*plane]");
    &FloatTensor out = __tensor_alloc_uninit_like(x);
    unsigned long shape[1];
    unsafe { shape[0] = 3UL + 2UL * channels; }
    &FloatTensor aux = __tensor_alloc(shape, 1UL, 0);
    unsafe {
        if (gamma->size != channels || beta->size != channels || runMean->size != channels || runVar->size != channels)
            panic_at("tensor_batch_norm: gamma, beta and running stats must have one value per channel", "convnet.sc", __LINE__);
        aux->data[0] = (float)channels;
        aux->data[1] = (float)plane;
        aux->data[2] = training ? (float)1.0 : (float)0.0;
        struct CnBnJob_ j = cn_bn_job_(x, gamma, beta, aux);
        j.out = (float*)out->data;
        j.runMean = (float*)runMean->data;
        j.runVar = (float*)runVar->data;
        j.momentum = momentum;
        j.eps = eps;
        cn_split_(channels, batch * plane, cn_bn_fwd_channels_, &j);
    }
    cn_link_param_op_(out, x, gamma, beta, aux, (void*)cn_bn_backward_);
    return out;
}

// ══ Group normalization ══════════════════════════════════════════════════════
// Per image n and group gi, over its Cg = C / groups channels and their
// planes (N = Cg * plane values, contiguous in the row):
//   y = gamma[c] * (x - mean) * invstd + beta[c]
//   dx = invstd * (dxh - mean(dxh) - xh * mean(dxh * xh)),  dxh = dy * gamma[c]
//   dgamma[c] += sum dy xh,  dbeta[c] += sum dy  (over images and the plane)
// The (image, group) passes split across the pool by (image, group); the
// parameter gradients, which sum over images, by channel. aux: C, plane,
// groups, eps, then mean[batch*groups], invstd[batch*groups].

struct CnGnJob_ {
    const float* x;
    float*       out;
    const float* gamma;
    const float* beta;
    float*       mean;
    float*       inv;
    const float* dy;
    float*       dx;
    float*       dgamma;
    float*       dbeta;
    unsigned long batch;
    unsigned long C;
    unsigned long plane;
    unsigned long groups;
    float eps;
};

static void cn_gn_fwd_units_(&CnGnJob_ j, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long Cg = j.C / j.groups;
        unsigned long N = Cg * j.plane;
        unsigned long u = lo;
        while (u < hi) {
            unsigned long n = u / j.groups;
            unsigned long gi = u % j.groups;
            unsigned long base = n * j.C * j.plane + gi * N;
            const float* xs = &j.x[base];
            double acc = 0.0;
            unsigned long i = 0UL;
            while (i < N) { acc = acc + (double)xs[i]; i = i + 1UL; }
            double mu = acc / (double)N;
            double sq = 0.0;
            i = 0UL;
            while (i < N) { double d = (double)xs[i] - mu; sq = sq + d * d; i = i + 1UL; }
            float mean = (float)mu;
            float inv = (float)1.0 / sqrt_f((float)(sq / (double)N) + j.eps);
            j.mean[u] = mean;
            j.inv[u] = inv;
            unsigned long c = 0UL;
            while (c < Cg) {
                float gam = j.gamma[gi * Cg + c];
                float bet = j.beta[gi * Cg + c];
                unsigned long q = 0UL;
                while (q < j.plane) {
                    unsigned long at = c * j.plane + q;
                    j.out[base + at] = gam * (xs[at] - mean) * inv + bet;
                    q = q + 1UL;
                }
                c = c + 1UL;
            }
            u = u + 1UL;
        }
    }
}

static void cn_gn_bwd_units_(&CnGnJob_ j, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long Cg = j.C / j.groups;
        unsigned long N = Cg * j.plane;
        unsigned long u = lo;
        while (u < hi) {
            unsigned long n = u / j.groups;
            unsigned long gi = u % j.groups;
            unsigned long base = n * j.C * j.plane + gi * N;
            float mean = j.mean[u];
            float inv = j.inv[u];
            double s1 = 0.0;
            double s2 = 0.0;
            unsigned long c = 0UL;
            while (c < Cg) {
                float gam = j.gamma[gi * Cg + c];
                unsigned long q = 0UL;
                while (q < j.plane) {
                    unsigned long at = base + c * j.plane + q;
                    float dxh = j.dy[at] * gam;
                    s1 = s1 + (double)dxh;
                    s2 = s2 + (double)(dxh * (j.x[at] - mean) * inv);
                    q = q + 1UL;
                }
                c = c + 1UL;
            }
            float m1 = (float)(s1 / (double)N);
            float m2 = (float)(s2 / (double)N);
            c = 0UL;
            while (c < Cg) {
                float gam = j.gamma[gi * Cg + c];
                unsigned long q = 0UL;
                while (q < j.plane) {
                    unsigned long at = base + c * j.plane + q;
                    float xh = (j.x[at] - mean) * inv;
                    j.dx[at] = j.dx[at] + inv * (j.dy[at] * gam - m1 - xh * m2);
                    q = q + 1UL;
                }
                c = c + 1UL;
            }
            u = u + 1UL;
        }
    }
}

static void cn_gn_param_channels_(&CnGnJob_ j, unsigned long lo, unsigned long hi) {
    unsafe {
        unsigned long Cg = j.C / j.groups;
        unsigned long c = lo;
        while (c < hi) {
            double sg = 0.0;
            double sb = 0.0;
            unsigned long n = 0UL;
            while (n < j.batch) {
                unsigned long u = n * j.groups + c / Cg;
                float mean = j.mean[u];
                float inv = j.inv[u];
                unsigned long base = (n * j.C + c) * j.plane;
                unsigned long q = 0UL;
                while (q < j.plane) {
                    float dy = j.dy[base + q];
                    sb = sb + (double)dy;
                    sg = sg + (double)(dy * (j.x[base + q] - mean) * inv);
                    q = q + 1UL;
                }
                n = n + 1UL;
            }
            if (j.dgamma != (float*)0) j.dgamma[c] = j.dgamma[c] + (float)sg;
            if (j.dbeta != (float*)0) j.dbeta[c] = j.dbeta[c] + (float)sb;
            c = c + 1UL;
        }
    }
}

static struct CnGnJob_ cn_gn_job_(FloatTensor* x, FloatTensor* gamma, FloatTensor* beta, FloatTensor* aux) {
    struct CnGnJob_ j;
    unsafe {
        j.C = (unsigned long)aux->data[0];
        j.plane = (unsigned long)aux->data[1];
        j.groups = (unsigned long)aux->data[2];
        j.eps = aux->data[3];
        j.batch = x->size / (j.C * j.plane);
        j.x = (const float*)x->data;
        j.gamma = (const float*)gamma->data;
        j.beta = (const float*)beta->data;
        j.mean = (float*)&aux->data[4];
        j.inv = (float*)&aux->data[4UL + j.batch * j.groups];
        j.out = (float*)0;
        j.dy = (const float*)0; j.dx = (float*)0; j.dgamma = (float*)0; j.dbeta = (float*)0;
    }
    return j;
}

static void cn_gn_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* x = __tensor_parent(s, 0UL);
        FloatTensor* gm = __tensor_parent(s, 1UL);
        FloatTensor* bt = __tensor_parent(s, 2UL);
        struct CnGnJob_ j = cn_gn_job_(x, gm, bt, __tensor_parent(s, 3UL));
        j.dy = (const float*)s->grad;
        unsigned long Cg = j.C / j.groups;
        if (x->requiresGrad) {
            __tensor_ensure_grad(x);
            j.dx = (float*)x->grad;
            cn_split_(j.batch * j.groups, Cg * j.plane, cn_gn_bwd_units_, &j);
        }
        if (gm->requiresGrad) { __tensor_ensure_grad(gm); j.dgamma = (float*)gm->grad; }
        if (bt->requiresGrad) { __tensor_ensure_grad(bt); j.dbeta = (float*)bt->grad; }
        if (gm->requiresGrad || bt->requiresGrad) cn_split_(j.C, j.batch * j.plane, cn_gn_param_channels_, &j);
    }
}

&FloatTensor tensor_group_norm(const &FloatTensor x, const &FloatTensor gamma, const &FloatTensor beta,
                          unsigned long channels, unsigned long plane, unsigned long groups, float eps) {
    unsigned long batch = cn_batch_(x, channels * plane, "tensor_group_norm: x is not [batch, channels*plane]");
    if (groups == 0UL || channels % groups != 0UL)
        panic_at("tensor_group_norm: channels must divide into the groups", "convnet.sc", __LINE__);
    unsafe {
        if (gamma->size != channels || beta->size != channels)
            panic_at("tensor_group_norm: gamma and beta must have one value per channel", "convnet.sc", __LINE__);
    }
    &FloatTensor out = __tensor_alloc_uninit_like(x);
    &FloatTensor aux = cn_alloc1_(4UL + 2UL * batch * groups);
    unsafe {
        aux->data[0] = (float)channels;
        aux->data[1] = (float)plane;
        aux->data[2] = (float)groups;
        aux->data[3] = eps;
        struct CnGnJob_ j = cn_gn_job_(x, gamma, beta, aux);
        j.out = (float*)out->data;
        cn_split_(batch * groups, (channels / groups) * plane, cn_gn_fwd_units_, &j);
    }
    cn_link_param_op_(out, x, gamma, beta, aux, (void*)cn_gn_backward_);
    return out;
}

// ══ Spectral normalization ═══════════════════════════════════════════════════
// W [rows, cols]; v = normalize(W^T u), u = normalize(W v) per iteration;
// sigma = u^T W v; out = W / sigma.
//   dW += G / sigma - (sum G * W) / sigma^2 * u v^T
// aux: sigma, rows, cols, u[rows], v[cols].

static void cn_normalize_(float* v, unsigned long n) {
    unsafe {
        double sq = 0.0;
        unsigned long i = 0UL;
        while (i < n) { sq = sq + (double)v[i] * (double)v[i]; i = i + 1UL; }
        double norm = sqrt_d(sq);
        float inv = (float)(1.0 / (norm > 1e-12 ? norm : 1e-12));
        i = 0UL;
        while (i < n) { v[i] = v[i] * inv; i = i + 1UL; }
    }
}

// v = normalize(W^T u)
static void cn_sn_v_(const float* w, const float* u, float* v, unsigned long rows, unsigned long cols) {
    unsafe {
        unsigned long j = 0UL;
        while (j < cols) { v[j] = (float)0.0; j = j + 1UL; }
        unsigned long i = 0UL;
        while (i < rows) {
            float ui = u[i];
            const float* row = &w[i * cols];
            j = 0UL;
            while (j < cols) { v[j] = v[j] + ui * row[j]; j = j + 1UL; }
            i = i + 1UL;
        }
        cn_normalize_(v, cols);
    }
}

// u = W v (normalized when 'norm'); returns u^T (W v) before normalizing.
static float cn_sn_u_(const float* w, float* u, const float* v, unsigned long rows, unsigned long cols, int norm) {
    unsafe {
        double sigma = 0.0;
        unsigned long i = 0UL;
        while (i < rows) {
            const float* row = &w[i * cols];
            double acc = 0.0;
            unsigned long j = 0UL;
            while (j < cols) { acc = acc + (double)row[j] * (double)v[j]; j = j + 1UL; }
            if (norm) u[i] = (float)acc;
            else sigma = sigma + (double)u[i] * acc;
            i = i + 1UL;
        }
        if (norm) cn_normalize_(u, rows);
        return (float)sigma;
    }
}

static void cn_sn_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* w = __tensor_parent(s, 0UL);
        FloatTensor* aux = __tensor_parent(s, 1UL);
        if (!w->requiresGrad) return;
        __tensor_ensure_grad(w);
        float sigma = aux->data[0];
        unsigned long rows = (unsigned long)aux->data[1];
        unsigned long cols = (unsigned long)aux->data[2];
        const float* u = (const float*)&aux->data[3];
        const float* v = (const float*)&aux->data[3UL + rows];
        double gw = 0.0;
        unsigned long i = 0UL;
        while (i < w->size) { gw = gw + (double)s->grad[i] * (double)w->data[i]; i = i + 1UL; }
        float inv = (float)1.0 / sigma;
        float k = (float)(gw / ((double)sigma * (double)sigma));
        i = 0UL;
        while (i < rows) {
            unsigned long j = 0UL;
            while (j < cols) {
                unsigned long at = i * cols + j;
                w->grad[at] = w->grad[at] + s->grad[at] * inv - k * u[i] * v[j];
                j = j + 1UL;
            }
            i = i + 1UL;
        }
    }
}

&FloatTensor tensor_spectral_norm(const &FloatTensor weight, &FloatTensor u, unsigned long iters, int training) {
    unsigned long rows;
    unsigned long cols;
    unsafe {
        rows = weight->shape[0];
        cols = weight->size / rows;
        if (u->size != rows) panic_at("tensor_spectral_norm: u must have one value per weight row", "convnet.sc", __LINE__);
    }
    &FloatTensor out = __tensor_alloc_uninit_like(weight);
    &FloatTensor aux = cn_alloc1_(3UL + rows + cols);
    unsafe {
        const float* w = (const float*)weight->data;
        float* ub = (float*)u->data;
        float* v = (float*)&aux->data[3UL + rows];
        unsigned long it = 0UL;
        while (training && it < iters) {
            cn_sn_v_(w, (const float*)ub, v, rows, cols);
            cn_sn_u_(w, ub, (const float*)v, rows, cols, 1);
            it = it + 1UL;
        }
        cn_sn_v_(w, (const float*)ub, v, rows, cols);
        float sigma = cn_sn_u_(w, ub, (const float*)v, rows, cols, 0);
        if (sigma < (float)1e-12) sigma = (float)1e-12;
        aux->data[0] = sigma;
        aux->data[1] = (float)rows;
        aux->data[2] = (float)cols;
        unsigned long i = 0UL;
        while (i < rows) { aux->data[3UL + i] = ub[i]; i = i + 1UL; }
        __tensor_ew(EW_SCALE, w, (const float*)0, (float)1.0 / sigma, (float*)out->data, weight->size);
    }
    __tensor_link2(out, weight, aux, (void*)cn_sn_backward_);
    unsafe { if (!out->requiresGrad) cn_drop_(aux); }
    return out;
}

// ══ Dropout ══════════════════════════════════════════════════════════════════
// x * mask, the mask 0 with probability p and 1 / (1 - p) otherwise (one
// draw per value, or per (image, channel) plane for dropout2d); tensor_mul
// carries the gradient.

static &FloatTensor cn_dropout_(const &FloatTensor x, unsigned long plane, float p, int training, &Rng rng) {
    if (!training || p <= (float)0.0) return x;
    &FloatTensor mask = __tensor_alloc_uninit_like(x);
    float keep = p >= (float)1.0 ? (float)0.0 : (float)1.0 / ((float)1.0 - p);
    unsafe {
        unsigned long units = x->size / plane;
        unsigned long u = 0UL;
        while (u < units) {
            float m = rng.uniform() < p ? (float)0.0 : keep;
            __tensor_ew(EW_FILL, (const float*)0, (const float*)0, m, (float*)&mask->data[u * plane], plane);
            u = u + 1UL;
        }
    }
    return tensor_mul(x, mask);
}

&FloatTensor FloatTensor::dropout(float p, int training, &Rng rng) const {
    return cn_dropout_(self, 1UL, p, training, rng);
}

&FloatTensor FloatTensor::dropout2d(unsigned long channels, unsigned long plane, float p, int training, &Rng rng) const {
    cn_batch_(self, channels * plane, "Tensor::dropout2d: x is not [batch, channels*plane]");
    return cn_dropout_(self, plane == 0UL ? 1UL : plane, p, training, rng);
}

// ══ Conditioning ═════════════════════════════════════════════════════════════

&FloatTensor FloatTensor::label_planes(unsigned long classes, unsigned long height, unsigned long width) const {
    unsigned long m;
    unsafe { m = self->size; }
    unsigned long plane = height * width;
    &FloatTensor out = FloatTensor::new_2d(m, classes * plane, 0);
    unsafe {
        unsigned long i = 0UL;
        while (i < m) {
            unsigned long c = (unsigned long)self->data[i];
            if (c < classes) {
                float* row = (float*)&out->data[(i * classes + c) * plane];
                unsigned long q = 0UL;
                while (q < plane) { row[q] = (float)1.0; q = q + 1UL; }
            }
            i = i + 1UL;
        }
    }
    return out;
}

&FloatTensor tensor_append_label_channels(const &FloatTensor x, const &FloatTensor labels, unsigned long classes,
                                     unsigned long height, unsigned long width) {
    return tensor_concat_cols(x, labels.label_planes(classes, height, width));
}

// ══ ConvNet ══════════════════════════════════════════════════════════════════

// Parameters always live on the heap, also when an arena is active.
static ?&TensorArena cn_heap_begin_() {
    ?&TensorArena cur = tensor_arena_current_();
    tensor_arena_end();
    return cur;
}

static void cn_heap_end_(?&TensorArena cur) {
    match (cur) {
        case some(a): { a.begin(); }
        case null: { }
    }
}

struct ConvNet ConvNet::new(unsigned long channels, unsigned long height, unsigned long width) {
    struct ConvNet n;
    n.layers = vec_new();
    n.params = vec_new();
    n.buffers = vec_new();
    n.rngs = vec_new();
    n.slots = 0UL;
    n.training = 1;
    n.inC = channels; n.inH = height; n.inW = width;
    n.outC = channels; n.outH = height; n.outW = width;
    return n;
}

static struct ConvLayer cn_layer_(const &ConvNet n, int kind, int act) {
    struct ConvLayer l;
    l.kind = kind;
    l.act = act;
    l.inC = n.outC; l.inH = n.outH; l.inW = n.outW;
    l.outC = n.outC; l.outH = n.outH; l.outW = n.outW;
    l.conv = Conv2dSpec::new(1UL, 1UL, 0UL);
    l.pool = Pool2dSpec::new(1UL, 1UL, 0UL);
    l.factor = 1UL;
    l.mode = 0;
    l.flag = 0;
    l.prob = (float)0.0;
    l.slot = 0UL;
    l.skipC = 0UL; l.skipH = 0UL; l.skipW = 0UL;
    l.param = 0UL;
    l.buffer = 0UL;
    l.spectral = 0;
    return l;
}

static void cn_push_(&ConvNet n, struct ConvLayer l) {
    if (l.outC * l.outH * l.outW == 0UL)
        panic_at("convnet: a layer's output would be empty (kernel larger than its input?)", "convnet.sc", __LINE__);
    n.outC = l.outC; n.outH = l.outH; n.outW = l.outW;
    n.layers.push(l);
}

// Weight and bias, U(-1/sqrt(fanIn), 1/sqrt(fanIn)), on the heap.
static unsigned long cn_add_params_(&ConvNet n, unsigned long wRows, unsigned long wCols, unsigned long nBias,
                                    unsigned long fanIn, &Rng rng) {
    ?&TensorArena cur = cn_heap_begin_();
    &FloatTensor w = FloatTensor::new_2d(wRows, wCols, 1);
    &FloatTensor b = FloatTensor::new_1d(nBias, 1);
    float bound = (float)1.0 / sqrt_f((float)fanIn);
    rng.fill_uniform(w, -bound, bound);
    rng.fill_uniform(b, -bound, bound);
    unsigned long at = n.params.length();
    n.params.push(w);
    n.params.push(b);
    cn_heap_end_(cur);
    return at;
}

// gamma 1, beta 0
static unsigned long cn_add_affine_(&ConvNet n, unsigned long channels) {
    ?&TensorArena cur = cn_heap_begin_();
    &FloatTensor gamma = FloatTensor::new_1d(channels, 1);
    &FloatTensor beta = FloatTensor::new_1d(channels, 1);
    gamma.fill((float)1.0);
    unsigned long at = n.params.length();
    n.params.push(gamma);
    n.params.push(beta);
    cn_heap_end_(cur);
    return at;
}

void ConvNet::conv_ex(unsigned long outC, const &Conv2dSpec spec, int act, &Rng rng) {
    struct ConvLayer l = cn_layer_(self, CONV_LAYER_CONV, act);
    cn_check_spec_(spec, l.inC, outC, "ConvNet::conv: kernel / stride / dilation 0, or channels not divisible by groups");
    l.conv = *spec;
    l.outC = outC;
    l.outH = conv_out_side_ex(l.inH, spec.kH, spec.strideH, spec.padH, spec.dilationH);
    l.outW = conv_out_side_ex(l.inW, spec.kW, spec.strideW, spec.padW, spec.dilationW);
    unsigned long K = (l.inC / spec.groups) * spec.kH * spec.kW;
    l.param = cn_add_params_(self, outC, K, outC, K, rng);
    cn_push_(self, l);
}

void ConvNet::deconv_ex(unsigned long outC, const &Conv2dSpec spec, int act, &Rng rng) {
    struct ConvLayer l = cn_layer_(self, CONV_LAYER_DECONV, act);
    cn_check_spec_(spec, l.inC, outC, "ConvNet::deconv: kernel / stride / dilation 0, or channels not divisible by groups");
    l.conv = *spec;
    l.outC = outC;
    l.outH = conv_transpose_out_side_ex(l.inH, spec.kH, spec.strideH, spec.padH, spec.dilationH, spec.outPadH);
    l.outW = conv_transpose_out_side_ex(l.inW, spec.kW, spec.strideW, spec.padW, spec.dilationW, spec.outPadW);
    unsigned long K = (outC / spec.groups) * spec.kH * spec.kW;
    // PyTorch's ConvTranspose2d fan-in: weight.size(1) * kernel area
    l.param = cn_add_params_(self, l.inC, K, outC, K, rng);
    cn_push_(self, l);
}

void ConvNet::conv(unsigned long outC, unsigned long kernel, unsigned long stride, unsigned long padding, int act,
                   &Rng rng) {
    struct Conv2dSpec s = Conv2dSpec::new(kernel, stride, padding);
    self.conv_ex(outC, &s, act, rng);
}

void ConvNet::deconv(unsigned long outC, unsigned long kernel, unsigned long stride, unsigned long padding, int act,
                     &Rng rng) {
    struct Conv2dSpec s = Conv2dSpec::new(kernel, stride, padding);
    self.deconv_ex(outC, &s, act, rng);
}

static void cn_pool_layer_(&ConvNet n, int kind, const &Pool2dSpec spec) {
    struct ConvLayer l = cn_layer_(n, kind, ACT_NONE);
    l.pool = *spec;
    l.outH = pool_out_side(l.inH, spec.kH, spec.strideH, spec.padH, spec.ceilMode);
    l.outW = pool_out_side(l.inW, spec.kW, spec.strideW, spec.padW, spec.ceilMode);
    cn_push_(n, l);
}

void ConvNet::maxpool_ex(const &Pool2dSpec spec) { cn_pool_layer_(self, CONV_LAYER_MAXPOOL, spec); }
void ConvNet::avgpool_ex(const &Pool2dSpec spec) { cn_pool_layer_(self, CONV_LAYER_AVGPOOL, spec); }

void ConvNet::maxpool(unsigned long kernel, unsigned long stride, unsigned long padding) {
    struct Pool2dSpec s = Pool2dSpec::new(kernel, stride, padding);
    cn_pool_layer_(self, CONV_LAYER_MAXPOOL, &s);
}

void ConvNet::avgpool(unsigned long kernel, unsigned long stride, unsigned long padding, int countIncludePad) {
    struct Pool2dSpec s = Pool2dSpec::new(kernel, stride, padding);
    s.countIncludePad = countIncludePad;
    cn_pool_layer_(self, CONV_LAYER_AVGPOOL, &s);
}

static void cn_global_layer_(&ConvNet n, int kind) {
    struct ConvLayer l = cn_layer_(n, kind, ACT_NONE);
    l.outH = 1UL;
    l.outW = 1UL;
    cn_push_(n, l);
}

void ConvNet::global_avgpool() { cn_global_layer_(self, CONV_LAYER_GLOBAL_AVGPOOL); }
void ConvNet::global_maxpool() { cn_global_layer_(self, CONV_LAYER_GLOBAL_MAXPOOL); }

void ConvNet::upsample(unsigned long factor) {
    struct ConvLayer l = cn_layer_(self, CONV_LAYER_UPSAMPLE, ACT_NONE);
    l.factor = factor;
    l.outH = l.inH * factor;
    l.outW = l.inW * factor;
    cn_push_(self, l);
}

void ConvNet::resize(unsigned long outH, unsigned long outW, int mode, int alignCorners) {
    struct ConvLayer l = cn_layer_(self, CONV_LAYER_RESIZE, ACT_NONE);
    l.mode = mode;
    l.flag = alignCorners;
    l.outH = outH;
    l.outW = outW;
    cn_push_(self, l);
}

void ConvNet::upsample_bilinear(unsigned long factor) {
    self.resize(self.outH * factor, self.outW * factor, RESIZE_BILINEAR, 0);
}

void ConvNet::dense(unsigned long outDim, int act, &Rng rng) {
    struct ConvLayer l = cn_layer_(self, CONV_LAYER_DENSE, act);
    unsigned long inDim = l.inC * l.inH * l.inW;
    l.outC = outDim; l.outH = 1UL; l.outW = 1UL;
    l.param = cn_add_params_(self, inDim, outDim, outDim, inDim, rng);
    cn_push_(self, l);
}

void ConvNet::batchnorm(int act) {
    struct ConvLayer l = cn_layer_(self, CONV_LAYER_BATCHNORM, act);
    l.param = cn_add_affine_(self, l.inC);
    ?&TensorArena cur = cn_heap_begin_();
    &FloatTensor rm = FloatTensor::new_1d(l.inC, 0);
    &FloatTensor rv = FloatTensor::new_1d(l.inC, 0);
    rv.fill((float)1.0);
    l.buffer = self.buffers.length();
    self.buffers.push(rm);
    self.buffers.push(rv);
    cn_heap_end_(cur);
    cn_push_(self, l);
}

void ConvNet::groupnorm(unsigned long groups, int act) {
    struct ConvLayer l = cn_layer_(self, CONV_LAYER_GROUPNORM, act);
    if (groups == 0UL || l.inC % groups != 0UL)
        panic_at("ConvNet::groupnorm: channels must divide into the groups", "convnet.sc", __LINE__);
    l.factor = groups;
    l.param = cn_add_affine_(self, l.inC);
    cn_push_(self, l);
}

void ConvNet::layernorm(int act) { self.groupnorm(1UL, act); }
void ConvNet::instancenorm(int act) { self.groupnorm(self.outC, act); }

void ConvNet::spectral_norm(&Rng rng) {
    unsigned long len = self.layers.length();
    if (len == 0UL) panic_at("ConvNet::spectral_norm: no layer", "convnet.sc", __LINE__);
    &ConvLayer last = self.layers.at(len - 1UL);
    if (last.kind != CONV_LAYER_CONV && last.kind != CONV_LAYER_DECONV && last.kind != CONV_LAYER_DENSE)
        panic_at("ConvNet::spectral_norm: the last layer has no weight", "convnet.sc", __LINE__);
    &FloatTensor w = *self.params.at(last.param);
    unsigned long rows;
    unsafe { rows = w->shape[0]; }
    ?&TensorArena cur = cn_heap_begin_();
    &FloatTensor u = FloatTensor::new_1d(rows, 0);
    rng.fill_normal(u, (float)0.0, (float)1.0);
    unsafe { cn_normalize_((float*)u->data, rows); }
    last.buffer = self.buffers.length();
    last.spectral = 1;
    self.buffers.push(u);
    cn_heap_end_(cur);
}

void ConvNet::dropout(float p, int byChannel, &Rng rng) {
    struct ConvLayer l = cn_layer_(self, CONV_LAYER_DROPOUT, ACT_NONE);
    l.prob = p;
    l.flag = byChannel;
    l.buffer = self.rngs.length();
    self.rngs.push(Rng::new(rng.u64()));
    cn_push_(self, l);
}

void ConvNet::reshape(unsigned long channels, unsigned long height, unsigned long width) {
    if (channels * height * width != self.outC * self.outH * self.outW)
        panic_at("ConvNet::reshape: size differs from the current output", "convnet.sc", __LINE__);
    struct ConvLayer l = cn_layer_(self, CONV_LAYER_RESHAPE, ACT_NONE);
    l.outC = channels; l.outH = height; l.outW = width;
    cn_push_(self, l);
}

unsigned long ConvNet::save() {
    struct ConvLayer l = cn_layer_(self, CONV_LAYER_SAVE, ACT_NONE);
    l.slot = self.slots;
    self.slots = self.slots + 1UL;
    cn_push_(self, l);
    return l.slot;
}

// The SAVE layer of 'slot'.
static struct ConvLayer cn_saved_layer_(const &ConvNet n, unsigned long slot) {
    unsigned long i = 0UL;
    while (i < n.layers.length()) {
        struct ConvLayer l = *n.layers.at(i);
        if (l.kind == CONV_LAYER_SAVE && l.slot == slot) return l;
        i = i + 1UL;
    }
    panic_at("convnet: no saved activation with that slot", "convnet.sc", __LINE__);
    return cn_layer_(n, CONV_LAYER_SAVE, ACT_NONE);
}

void ConvNet::add_skip(unsigned long slot, int act, &Rng rng) {
    struct ConvLayer sv = cn_saved_layer_(self, slot);
    struct ConvLayer l = cn_layer_(self, CONV_LAYER_ADD, act);
    l.slot = slot;
    l.skipC = sv.outC; l.skipH = sv.outH; l.skipW = sv.outW;
    if (sv.outC != l.inC || sv.outH != l.inH || sv.outW != l.inW) {
        // projection shortcut: a 1x1 convolution with the stride that maps
        // the saved size onto the current one
        unsigned long s = 1UL;
        while (s <= sv.outH && (conv_out_side(sv.outH, 1UL, s, 0UL) != l.inH || conv_out_side(sv.outW, 1UL, s, 0UL) != l.inW))
            s = s + 1UL;
        if (s > sv.outH) panic_at("ConvNet::add_skip: no 1x1 stride maps the saved size onto the current one", "convnet.sc", __LINE__);
        l.flag = 1;
        l.conv = Conv2dSpec::new(1UL, s, 0UL);
        l.param = cn_add_params_(self, l.inC, sv.outC, l.inC, sv.outC, rng);
    }
    cn_push_(self, l);
}

void ConvNet::concat_skip(unsigned long slot) {
    struct ConvLayer sv = cn_saved_layer_(self, slot);
    struct ConvLayer l = cn_layer_(self, CONV_LAYER_CONCAT, ACT_NONE);
    if (sv.outH != l.inH || sv.outW != l.inW)
        panic_at("ConvNet::concat_skip: the saved activation's height and width differ", "convnet.sc", __LINE__);
    l.slot = slot;
    l.skipC = sv.outC; l.skipH = sv.outH; l.skipW = sv.outW;
    l.outC = l.inC + sv.outC;
    cn_push_(self, l);
}

void ConvNet::residual_block(unsigned long outC, unsigned long stride, int batchNorm, int act, &Rng rng) {
    unsigned long slot = self.save();
    self.conv(outC, 3UL, stride, 1UL, batchNorm ? ACT_NONE : act, rng);
    if (batchNorm) self.batchnorm(act);
    self.conv(outC, 3UL, 1UL, 1UL, ACT_NONE, rng);
    if (batchNorm) self.batchnorm(ACT_NONE);
    self.add_skip(slot, act, rng);
}

void ConvNet::set_out_act(int act) {
    unsigned long len = self.layers.length();
    if (len == 0UL) return;
    &ConvLayer last = self.layers.at(len - 1UL);
    last.act = act;
}

int ConvNet::set_training(int training) {
    int was = self.training;
    self.training = training;
    return was;
}

unsigned long ConvNet::in_dim() const { return self.inC * self.inH * self.inW; }
unsigned long ConvNet::out_dim() const { return self.outC * self.outH * self.outW; }

// The layer's weight, spectrally normalized when the layer asks for it.
static &FloatTensor cn_weight_(const &ConvNet n, const &ConvLayer l) {
    &FloatTensor w = *n.params.at(l.param);
    if (!l.spectral) return w;
    return tensor_spectral_norm(w, *n.buffers.at(l.buffer), 1UL, n.training);
}

static &FloatTensor cn_layer_forward_(const &ConvNet n, const &ConvLayer l, const &FloatTensor x, &Vec<&FloatTensor> saved) {
    int k = l.kind;
    if (k == CONV_LAYER_CONV)
        return tensor_conv2d_ex(x, cn_weight_(n, l), *n.params.at(l.param + 1UL), l.inC, l.inH, l.inW, &l.conv);
    if (k == CONV_LAYER_DECONV)
        return tensor_conv_transpose2d_ex(x, cn_weight_(n, l), *n.params.at(l.param + 1UL), l.inC, l.inH, l.inW, &l.conv);
    if (k == CONV_LAYER_DENSE)
        return tensor_add_bias(tensor_matmul(x, cn_weight_(n, l)), *n.params.at(l.param + 1UL));
    if (k == CONV_LAYER_BATCHNORM)
        return tensor_batch_norm(x, *n.params.at(l.param), *n.params.at(l.param + 1UL),
                                 *n.buffers.at(l.buffer), *n.buffers.at(l.buffer + 1UL),
                                 l.inC, l.inH * l.inW, n.training, (float)0.1, (float)0.00001);
    if (k == CONV_LAYER_GROUPNORM)
        return tensor_group_norm(x, *n.params.at(l.param), *n.params.at(l.param + 1UL),
                                 l.inC, l.inH * l.inW, l.factor, (float)0.00001);
    if (k == CONV_LAYER_MAXPOOL) return x.maxpool2d_ex(l.inC, l.inH, l.inW, &l.pool);
    if (k == CONV_LAYER_AVGPOOL) return x.avgpool2d_ex(l.inC, l.inH, l.inW, &l.pool);
    if (k == CONV_LAYER_GLOBAL_AVGPOOL) return x.global_avgpool2d(l.inC, l.inH, l.inW);
    if (k == CONV_LAYER_GLOBAL_MAXPOOL) return x.global_maxpool2d(l.inC, l.inH, l.inW);
    if (k == CONV_LAYER_UPSAMPLE) return x.upsample_nearest(l.inC, l.inH, l.inW, l.factor);
    if (k == CONV_LAYER_RESIZE) return x.resize2d(l.inC, l.inH, l.inW, l.outH, l.outW, l.mode, l.flag);
    if (k == CONV_LAYER_DROPOUT) {
        &Rng r = n.rngs.at(l.buffer);
        if (l.flag) return x.dropout2d(l.inC, l.inH * l.inW, l.prob, n.training, r);
        return x.dropout(l.prob, n.training, r);
    }
    if (k == CONV_LAYER_SAVE) { saved.push(x); return x; }
    if (k == CONV_LAYER_ADD) {
        &FloatTensor sv = *saved.at(l.slot);
        if (l.flag)
            sv = tensor_conv2d_ex(sv, *n.params.at(l.param), *n.params.at(l.param + 1UL), l.skipC, l.skipH, l.skipW, &l.conv);
        return tensor_add(x, sv);
    }
    if (k == CONV_LAYER_CONCAT) return tensor_concat_cols(x, *saved.at(l.slot));
    return x;   // reshape / identity: rows already hold the data in CHW order
}

&FloatTensor ConvNet::forward(const &FloatTensor x) const {
    struct Vec<&FloatTensor> saved = vec_new();
    &FloatTensor h = x;
    unsigned long i = 0UL;
    while (i < self.layers.length()) {
        &ConvLayer l = self.layers.at(i);
        h = cn_layer_forward_(self, l, h, &saved).activate(l.act);
        i = i + 1UL;
    }
    saved.free();
    return h;
}

// ── Folding batch norms into the preceding layer ─────────────────────────────
// y = gamma (z - mean) / sqrt(var + eps) + beta with z = W x + b is
// (s W) x + (s (b - mean) + beta), s = gamma / sqrt(var + eps) per output
// channel.

// Scales the weight entries that feed output channel oc by s[oc].
static void cn_fold_weight_(const &ConvLayer l, &FloatTensor w, const float* sc) {
    unsafe {
        unsigned long rows = w->shape[0];
        unsigned long cols = w->size / rows;
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long c = 0UL;
            while (c < cols) {
                unsigned long oc;
                if (l.kind == CONV_LAYER_CONV) oc = r;                // [outC, K]
                else if (l.kind == CONV_LAYER_DENSE) oc = c;          // [in, out]
                else {                                                 // [inC, (outC/groups) kH kW]
                    unsigned long gi = r / (l.inC / l.conv.groups);
                    oc = gi * (l.outC / l.conv.groups) + c / (l.conv.kH * l.conv.kW);
                }
                w->data[r * cols + c] = w->data[r * cols + c] * sc[oc];
                c = c + 1UL;
            }
            r = r + 1UL;
        }
    }
}

unsigned long ConvNet::fold_batchnorm() {
    unsigned long folded = 0UL;
    unsigned long i = 1UL;
    while (i < self.layers.length()) {
        &ConvLayer bn = self.layers.at(i);
        &ConvLayer prev = self.layers.at(i - 1UL);
        int pk = prev.kind;
        if (bn.kind == CONV_LAYER_BATCHNORM && prev.act == ACT_NONE && !prev.spectral
            && (pk == CONV_LAYER_CONV || pk == CONV_LAYER_DECONV || pk == CONV_LAYER_DENSE)) {
            &FloatTensor gamma = *self.params.at(bn.param);
            &FloatTensor beta = *self.params.at(bn.param + 1UL);
            &FloatTensor rm = *self.buffers.at(bn.buffer);
            &FloatTensor rv = *self.buffers.at(bn.buffer + 1UL);
            &FloatTensor w = *self.params.at(prev.param);
            &FloatTensor b = *self.params.at(prev.param + 1UL);
            unsigned long C = bn.inC;
            float* sc = cn_scratch_(C);
            unsafe {
                unsigned long c = 0UL;
                while (c < C) {
                    sc[c] = gamma->data[c] / sqrt_f(rv->data[c] + (float)0.00001);
                    b->data[c] = (b->data[c] - rm->data[c]) * sc[c] + beta->data[c];
                    c = c + 1UL;
                }
                cn_fold_weight_(prev, w, (const float*)sc);
                free((void*)sc);
            }
            prev.act = bn.act;
            bn.kind = CONV_LAYER_IDENTITY;
            bn.act = ACT_NONE;
            folded = folded + 1UL;
        }
        i = i + 1UL;
    }
    return folded;
}

// ── Copies, updates, files ───────────────────────────────────────────────────

static void cn_clone_list_(&Vec<&FloatTensor> dst, const &Vec<&FloatTensor> src, int requiresGrad) {
    unsigned long i = 0UL;
    while (i < src.length()) {
        &FloatTensor p = *src.at(i);
        &FloatTensor q = p.zeros_like();
        unsafe {
            q->requiresGrad = requiresGrad;
            __tensor_ew(EW_COPY, (const float*)p->data, (const float*)0, (float)0.0, (float*)q->data, p->size);
        }
        dst.push(q);
        i = i + 1UL;
    }
}

struct ConvNet ConvNet::clone() const {
    ?&TensorArena cur = cn_heap_begin_();
    struct ConvNet c = ConvNet::new(self.inC, self.inH, self.inW);
    c.outC = self.outC; c.outH = self.outH; c.outW = self.outW;
    c.training = self.training;
    c.slots = self.slots;
    unsigned long i = 0UL;
    while (i < self.layers.length()) { c.layers.push(*self.layers.at(i)); i = i + 1UL; }
    i = 0UL;
    while (i < self.rngs.length()) { c.rngs.push(*self.rngs.at(i)); i = i + 1UL; }
    cn_clone_list_(&c.params, &self.params, 1);
    cn_clone_list_(&c.buffers, &self.buffers, 0);
    cn_heap_end_(cur);
    return c;
}

static void cn_blend_list_(&Vec<&FloatTensor> dst, const &Vec<&FloatTensor> src, float tau) {
    unsigned long i = 0UL;
    while (i < dst.length()) {
        &FloatTensor d = *dst.at(i);
        &FloatTensor s = *src.at(i);
        unsafe {
            unsigned long j = 0UL;
            while (j < d->size) { d->data[j] = tau * s->data[j] + ((float)1.0 - tau) * d->data[j]; j = j + 1UL; }
        }
        i = i + 1UL;
    }
}

void ConvNet::collect_params(&Vec<&FloatTensor> out) const {
    unsigned long i = 0UL;
    while (i < self.params.length()) { out.push(*self.params.at(i)); i = i + 1UL; }
}

void ConvNet::collect_state(&Vec<&FloatTensor> out) const {
    self.collect_params(out);
    unsigned long i = 0UL;
    while (i < self.buffers.length()) { out.push(*self.buffers.at(i)); i = i + 1UL; }
}

// params, then buffers
static void cn_free_list_(&Vec<&FloatTensor> v) {
    unsigned long i = 0UL;
    while (i < v.length()) {
        &FloatTensor p = *v.at(i);
        p->free();
        unsafe { free((void*)p); }
        i = i + 1UL;
    }
    v.free();
}

void ConvNet::free() {
    cn_free_list_(&self.params);
    cn_free_list_(&self.buffers);
    self.rngs.free();
    self.layers.free();
}

} // namespace std
