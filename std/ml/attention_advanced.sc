// SafeC Standard Library — Advanced attention implementation (see
// attention_advanced.h).
#pragma once
#include <std/ml/attention_advanced.h>
#include <std/ml/tensor.h>
#include <std/ml/attention.h>
#include <std/math.h>

namespace std {

extern void* malloc(unsigned long size);
extern void  free(void* ptr);

// ── MLA / EG-MLA ─────────────────────────────────────────────────────────────
&Tensor mla_forward(const &Tensor X, const &Tensor Wq, const &Tensor Wdkv,
                     const &Tensor Wuk, const &Tensor Wuv, unsigned long numHeads) {
    struct Tensor* Q = tensor_matmul(X, Wq);
    struct Tensor* c_kv = tensor_matmul(X, Wdkv);
    struct Tensor* K = tensor_matmul(c_kv, Wuk);
    struct Tensor* V = tensor_matmul(c_kv, Wuv);
    struct Tensor* out = mha_forward(Q, K, V, numHeads);

    Q->free(); c_kv->free(); K->free(); V->free();
    unsafe { free((void*)Q); free((void*)c_kv); free((void*)K); free((void*)V); }
    return out;
}

&Tensor eg_mla_forward(const &Tensor X, const &Tensor Wq, const &Tensor Wdkv,
                        struct Tensor** WukGroups, struct Tensor** WuvGroups,
                               unsigned long numGroups, unsigned long headsPerGroup) {
    unsigned long seqLen; unsigned long dModel;
    unsafe { seqLen = X->shape[0]; dModel = Wq->shape[1]; }
    unsigned long groupDim = dModel / numGroups;

    struct Tensor* Q = tensor_matmul(X, Wq);
    struct Tensor* c_kv = tensor_matmul(X, Wdkv);
    struct Tensor* out = tensor_new_2d(seqLen, dModel, 0);

    unsigned long g = 0UL;
    while (g < numGroups) {
        unsigned long startCol = g * groupDim;
        struct Tensor* Qg = tensor_slice_cols(Q, startCol, groupDim);
        struct Tensor* Kg;
        struct Tensor* Vg;
        unsafe { Kg = tensor_matmul(c_kv, WukGroups[g]); Vg = tensor_matmul(c_kv, WuvGroups[g]); }
        struct Tensor* groupOut = mha_forward(Qg, Kg, Vg, headsPerGroup);
        tensor_set_cols(out, startCol, groupOut);

        Qg->free(); Kg->free(); Vg->free(); groupOut->free();
        unsafe { free((void*)Qg); free((void*)Kg); free((void*)Vg); free((void*)groupOut); }
        g = g + 1UL;
    }

    Q->free(); c_kv->free();
    unsafe { free((void*)Q); free((void*)c_kv); }
    return out;
}

// ── Shifted-window attention ─────────────────────────────────────────────────
static struct Tensor* __row_slice(struct Tensor* in, unsigned long startRow, unsigned long numRows) {
    unsigned long cols;
    unsafe { cols = in->shape[1]; }
    struct Tensor* out = tensor_new_2d(numRows, cols, 0);
    unsafe {
        unsigned long r = 0UL;
        while (r < numRows) {
            unsigned long c = 0UL;
            while (c < cols) {
                out->data[r * cols + c] = in->data[(startRow + r) * cols + c];
                c = c + 1UL;
            }
            r = r + 1UL;
        }
    }
    return out;
}

static void __row_set(struct Tensor* dst, unsigned long startRow, struct Tensor* src) {
    unsigned long cols; unsigned long numRows;
    unsafe { cols = dst->shape[1]; numRows = src->shape[0]; }
    unsafe {
        unsigned long r = 0UL;
        while (r < numRows) {
            unsigned long c = 0UL;
            while (c < cols) {
                dst->data[(startRow + r) * cols + c] = src->data[r * cols + c];
                c = c + 1UL;
            }
            r = r + 1UL;
        }
    }
}

static struct Tensor* __row_cyclic_shift(struct Tensor* in, unsigned long shift) {
    unsigned long rows; unsigned long cols;
    unsafe { rows = in->shape[0]; cols = in->shape[1]; }
    struct Tensor* out = tensor_new_2d(rows, cols, 0);
    unsafe {
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long srcRow = (r + shift) % rows;
            unsigned long c = 0UL;
            while (c < cols) {
                out->data[r * cols + c] = in->data[srcRow * cols + c];
                c = c + 1UL;
            }
            r = r + 1UL;
        }
    }
    return out;
}

&Tensor windowed_attention_forward(const &Tensor Q, const &Tensor K, const &Tensor V,
                                    unsigned long windowSize, unsigned long shift) {
    unsigned long seqLen; unsigned long dim;
    unsafe { seqLen = Q->shape[0]; dim = Q->shape[1]; }

    int didShift = (shift > 0UL) ? 1 : 0;
    struct Tensor* Qs; struct Tensor* Ks; struct Tensor* Vs;
    if (didShift) {
        Qs = __row_cyclic_shift(Q, shift);
        Ks = __row_cyclic_shift(K, shift);
        Vs = __row_cyclic_shift(V, shift);
    } else {
        Qs = Q; Ks = K; Vs = V;
    }

    struct Tensor* winOut = tensor_new_2d(seqLen, dim, 0);
    unsigned long numWindows = seqLen / windowSize;
    unsigned long w = 0UL;
    while (w < numWindows) {
        unsigned long startRow = w * windowSize;
        struct Tensor* qw = __row_slice(Qs, startRow, windowSize);
        struct Tensor* kw = __row_slice(Ks, startRow, windowSize);
        struct Tensor* vw = __row_slice(Vs, startRow, windowSize);
        struct Tensor* ow = attention_forward(qw, kw, vw);
        __row_set(winOut, startRow, ow);

        qw->free(); kw->free(); vw->free(); ow->free();
        unsafe { free((void*)qw); free((void*)kw); free((void*)vw); free((void*)ow); }
        w = w + 1UL;
    }

    if (didShift) {
        Qs->free(); Ks->free(); Vs->free();
        unsafe { free((void*)Qs); free((void*)Ks); free((void*)Vs); }
        struct Tensor* out = __row_cyclic_shift(winOut, seqLen - shift);
        winOut->free();
        unsafe { free((void*)winOut); }
        return out;
    }
    return winOut;
}

// ── Gated DeltaNet ────────────────────────────────────────────────────────────
static double __gdn_init_weight(unsigned long i, unsigned long seed) {
    unsigned long h = (i + 1UL) * 2654435761UL + seed * 40503UL;
    h = (h ^ (h >> 13UL)) * 2246822519UL;
    return (((double)(h % 2000UL) / 1000.0) - 1.0) * 0.3;
}

struct GatedDeltaNet gated_deltanet_new(unsigned long dModel, unsigned long keyDim, unsigned long valueDim) {
    struct GatedDeltaNet layer;
    layer.Walpha = tensor_new_2d(dModel, 1UL, 0);
    layer.Wbeta = tensor_new_2d(dModel, 1UL, 0);
    unsafe {
        unsigned long i = 0UL;
        while (i < dModel) {
            layer.Walpha->data[i] = __gdn_init_weight(i, 21UL);
            layer.Wbeta->data[i] = __gdn_init_weight(i, 22UL);
            i = i + 1UL;
        }
    }
    layer.keyDim = keyDim;
    layer.valueDim = valueDim;
    return layer;
}

void gated_deltanet_free(&GatedDeltaNet layer) {
    unsafe {
        layer->Walpha->free(); layer->Wbeta->free();
        free((void*)layer->Walpha); free((void*)layer->Wbeta);
    }
}

&Tensor gated_deltanet_forward(const &GatedDeltaNet layer, const &Tensor X,
                                const &Tensor Q, const &Tensor K, const &Tensor V) {
    unsigned long seqLen; unsigned long dModel;
    unsigned long keyDim; unsigned long valueDim;
    unsafe {
        seqLen = X->shape[0]; dModel = X->shape[1];
        keyDim = layer->keyDim; valueDim = layer->valueDim;
    }

    struct Tensor* out = tensor_new_2d(seqLen, valueDim, 0);

    unsafe {
        double* S = (double*)malloc(sizeof(double) * keyDim * valueDim);
        unsigned long si = 0UL;
        while (si < keyDim * valueDim) { S[si] = 0.0; si = si + 1UL; }

        unsigned long t = 0UL;
        while (t < seqLen) {
            double alphaLogit = 0.0; double betaLogit = 0.0;
            unsigned long d = 0UL;
            while (d < dModel) {
                double xv = X->data[t * dModel + d];
                alphaLogit = alphaLogit + xv * layer->Walpha->data[d];
                betaLogit = betaLogit + xv * layer->Wbeta->data[d];
                d = d + 1UL;
            }
            double alpha = 1.0 / (1.0 + exp_d(-alphaLogit));
            double beta = 1.0 / (1.0 + exp_d(-betaLogit));

            double* pred = (double*)malloc(sizeof(double) * valueDim);
            unsigned long vd = 0UL;
            while (vd < valueDim) {
                double acc = 0.0;
                unsigned long kd = 0UL;
                while (kd < keyDim) {
                    acc = acc + S[kd * valueDim + vd] * K->data[t * keyDim + kd];
                    kd = kd + 1UL;
                }
                pred[vd] = acc;
                vd = vd + 1UL;
            }

            unsigned long kd2 = 0UL;
            while (kd2 < keyDim) {
                double kv = K->data[t * keyDim + kd2];
                unsigned long vd2 = 0UL;
                while (vd2 < valueDim) {
                    double deltaV = V->data[t * valueDim + vd2] - pred[vd2];
                    unsigned long idx = kd2 * valueDim + vd2;
                    S[idx] = alpha * S[idx] + beta * kv * deltaV;
                    vd2 = vd2 + 1UL;
                }
                kd2 = kd2 + 1UL;
            }
            free((void*)pred);

            unsigned long vd3 = 0UL;
            while (vd3 < valueDim) {
                double acc = 0.0;
                unsigned long kd3 = 0UL;
                while (kd3 < keyDim) {
                    acc = acc + S[kd3 * valueDim + vd3] * Q->data[t * keyDim + kd3];
                    kd3 = kd3 + 1UL;
                }
                out->data[t * valueDim + vd3] = acc;
                vd3 = vd3 + 1UL;
            }

            t = t + 1UL;
        }
        free((void*)S);
    }

    return out;
}

} // namespace std
