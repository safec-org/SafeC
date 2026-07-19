// SafeC Standard Library — Attention implementation (see attention.h).
#pragma once
#include <std/ml/attention.h>
#include <std/ml/tensor.h>
#include <std/ml/tensor.sc>
#include <std/ml/activations.h>
#include <std/ml/activations.sc>
#include <std/math.h>
#include <std/math.sc>
#include <std/mem.sc>

namespace std {

&Tensor softmax_rows(const &Tensor x) {
    unsigned long rows; unsigned long cols;
    unsafe { rows = x->shape[0]; cols = x->shape[1]; }
    struct Tensor* out = tensor_new_2d(rows, cols, 0);
    unsigned long r = 0UL;
    while (r < rows) {
        double rowMax;
        unsafe { rowMax = x->data[r * cols]; }
        unsigned long c = 1UL;
        while (c < cols) {
            double v;
            unsafe { v = x->data[r * cols + c]; }
            if (v > rowMax) rowMax = v;
            c = c + 1UL;
        }
        double sum = 0.0;
        c = 0UL;
        while (c < cols) {
            double v;
            unsafe { v = x->data[r * cols + c]; }
            double e = exp_d(v - rowMax);
            unsafe { out->data[r * cols + c] = e; }
            sum = sum + e;
            c = c + 1UL;
        }
        c = 0UL;
        while (c < cols) {
            unsafe { out->data[r * cols + c] = out->data[r * cols + c] / sum; }
            c = c + 1UL;
        }
        r = r + 1UL;
    }
    return out;
}

&Tensor tensor_transpose(const &Tensor in) {
    unsigned long rows; unsigned long cols;
    unsafe { rows = in->shape[0]; cols = in->shape[1]; }
    struct Tensor* out = tensor_new_2d(cols, rows, 0);
    unsigned long r = 0UL;
    while (r < rows) {
        unsigned long c = 0UL;
        while (c < cols) {
            unsafe { out->data[c * rows + r] = in->data[r * cols + c]; }
            c = c + 1UL;
        }
        r = r + 1UL;
    }
    return out;
}

&Tensor tensor_slice_cols(const &Tensor in, unsigned long startCol, unsigned long numCols) {
    unsigned long rows; unsigned long cols;
    unsafe { rows = in->shape[0]; cols = in->shape[1]; }
    struct Tensor* out = tensor_new_2d(rows, numCols, 0);
    unsigned long r = 0UL;
    while (r < rows) {
        unsigned long c = 0UL;
        while (c < numCols) {
            unsafe { out->data[r * numCols + c] = in->data[r * cols + startCol + c]; }
            c = c + 1UL;
        }
        r = r + 1UL;
    }
    return out;
}

void tensor_set_cols(&Tensor dst, unsigned long startCol, const &Tensor src) {
    unsigned long rows; unsigned long dstCols; unsigned long srcCols;
    unsafe { rows = dst->shape[0]; dstCols = dst->shape[1]; srcCols = src->shape[1]; }
    unsigned long r = 0UL;
    while (r < rows) {
        unsigned long c = 0UL;
        while (c < srcCols) {
            unsafe { dst->data[r * dstCols + startCol + c] = src->data[r * srcCols + c]; }
            c = c + 1UL;
        }
        r = r + 1UL;
    }
}

&Tensor attention_forward(const &Tensor Q, const &Tensor K, const &Tensor V) {
    unsigned long headDim;
    unsafe { headDim = Q->shape[1]; }
    struct Tensor* Kt = tensor_transpose(K);
    struct Tensor* scores = tensor_matmul(Q, Kt);
    double scale = 1.0 / sqrt_d((double)headDim);
    struct Tensor* scaled = tensor_scale(scores, scale);
    struct Tensor* weights = softmax_rows(scaled);
    struct Tensor* result = tensor_matmul(weights, V);

    Kt->free(); scores->free(); scaled->free(); weights->free();
    unsafe { free((void*)Kt); free((void*)scores); free((void*)scaled); free((void*)weights); }
    return result;
}

&Tensor mha_forward(const &Tensor Q, const &Tensor K, const &Tensor V, unsigned long numHeads) {
    unsigned long seqLen; unsigned long dModel;
    unsafe { seqLen = Q->shape[0]; dModel = Q->shape[1]; }
    unsigned long headDim = dModel / numHeads;

    struct Tensor* out = tensor_new_2d(seqLen, dModel, 0);
    unsigned long h = 0UL;
    while (h < numHeads) {
        unsigned long startCol = h * headDim;
        struct Tensor* Qh = tensor_slice_cols(Q, startCol, headDim);
        struct Tensor* Kh = tensor_slice_cols(K, startCol, headDim);
        struct Tensor* Vh = tensor_slice_cols(V, startCol, headDim);
        struct Tensor* headOut = attention_forward(Qh, Kh, Vh);
        tensor_set_cols(out, startCol, headOut);

        Qh->free(); Kh->free(); Vh->free(); headOut->free();
        unsafe { free((void*)Qh); free((void*)Kh); free((void*)Vh); free((void*)headOut); }
        h = h + 1UL;
    }
    return out;
}

&Tensor gated_attention_forward(const &Tensor Q, const &Tensor K, const &Tensor V,
                                 const &Tensor gateLogits) {
    struct Tensor* raw = attention_forward(Q, K, V);
    struct Tensor* gate = tensor_sigmoid(gateLogits);
    struct Tensor* out = tensor_mul(raw, gate);
    raw->free(); gate->free();
    unsafe { free((void*)raw); free((void*)gate); }
    return out;
}

&Tensor gated_mha_forward(const &Tensor Q, const &Tensor K, const &Tensor V,
                           unsigned long numHeads, const &Tensor gateLogits) {
    struct Tensor* raw = mha_forward(Q, K, V, numHeads);
    struct Tensor* gate = tensor_sigmoid(gateLogits);
    struct Tensor* out = tensor_mul(raw, gate);
    raw->free(); gate->free();
    unsafe { free((void*)raw); free((void*)gate); }
    return out;
}

// phi(x) = elu(x) + 1, elementwise (always > 0, used as linear
// attention's kernel feature map).
static struct Tensor* __phi_elu_p1(struct Tensor* x) {
    struct Tensor* out = tensor_zeros_like(x);
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) {
            double v = x->data[i];
            out->data[i] = (v > 0.0) ? (v + 1.0) : exp_d(v);
            i = i + 1UL;
        }
    }
    return out;
}

&Tensor linear_attention_forward(const &Tensor Q, const &Tensor K, const &Tensor V) {
    unsigned long seqLen; unsigned long headDim; unsigned long kvLen;
    unsafe { seqLen = Q->shape[0]; headDim = Q->shape[1]; kvLen = K->shape[0]; }

    struct Tensor* phiQ = __phi_elu_p1(Q);
    struct Tensor* phiK = __phi_elu_p1(K);
    struct Tensor* out = tensor_new_2d(seqLen, headDim, 0);

    unsafe {
        // KV[d1][d2] = sum_j phiK[j][d1] * V[j][d2]   ([headDim, headDim])
        double* KV = (double*)malloc(sizeof(double) * headDim * headDim);
        // Z[d1] = sum_j phiK[j][d1]                   ([headDim])
        double* Z = (double*)malloc(sizeof(double) * headDim);
        unsigned long d1 = 0UL;
        while (d1 < headDim) {
            double zAcc = 0.0;
            unsigned long d2 = 0UL;
            while (d2 < headDim) {
                double acc = 0.0;
                unsigned long j = 0UL;
                while (j < kvLen) {
                    acc = acc + phiK->data[j * headDim + d1] * V->data[j * headDim + d2];
                    j = j + 1UL;
                }
                KV[d1 * headDim + d2] = acc;
                d2 = d2 + 1UL;
            }
            unsigned long j2 = 0UL;
            while (j2 < kvLen) { zAcc = zAcc + phiK->data[j2 * headDim + d1]; j2 = j2 + 1UL; }
            Z[d1] = zAcc;
            d1 = d1 + 1UL;
        }

        unsigned long i = 0UL;
        while (i < seqLen) {
            double denom = 0.0;
            unsigned long dd = 0UL;
            while (dd < headDim) { denom = denom + phiQ->data[i * headDim + dd] * Z[dd]; dd = dd + 1UL; }
            unsigned long d2 = 0UL;
            while (d2 < headDim) {
                double numer = 0.0;
                unsigned long dd2 = 0UL;
                while (dd2 < headDim) {
                    numer = numer + phiQ->data[i * headDim + dd2] * KV[dd2 * headDim + d2];
                    dd2 = dd2 + 1UL;
                }
                out->data[i * headDim + d2] = numer / denom;
                d2 = d2 + 1UL;
            }
            i = i + 1UL;
        }
        free((void*)KV); free((void*)Z);
    }

    phiQ->free(); phiK->free();
    unsafe { free((void*)phiQ); free((void*)phiK); }
    return out;
}

&Tensor flash_attention_forward(const &Tensor Q, const &Tensor K, const &Tensor V,
                                 unsigned long kvBlockSize) {
    unsigned long seqLen; unsigned long headDim; unsigned long kvLen;
    unsafe { seqLen = Q->shape[0]; headDim = Q->shape[1]; kvLen = K->shape[0]; }
    double scale = 1.0 / sqrt_d((double)headDim);

    struct Tensor* out = tensor_new_2d(seqLen, headDim, 0); // doubles as the running 'acc' buffer

    unsafe {
        double* m = (double*)malloc(sizeof(double) * seqLen); // running row max
        double* l = (double*)malloc(sizeof(double) * seqLen); // running row sum
        unsigned long i0 = 0UL;
        while (i0 < seqLen) { m[i0] = -1.0e300; l[i0] = 0.0; i0 = i0 + 1UL; }

        unsigned long blockStart = 0UL;
        while (blockStart < kvLen) {
            unsigned long blockEnd = blockStart + kvBlockSize;
            if (blockEnd > kvLen) blockEnd = kvLen;
            unsigned long bLen = blockEnd - blockStart;

            unsigned long i = 0UL;
            while (i < seqLen) {
                // scores[j] = scale * Q[i,:] . K[blockStart+j,:]
                double* scores = (double*)malloc(sizeof(double) * bLen);
                double blockMax = -1.0e300;
                unsigned long j = 0UL;
                while (j < bLen) {
                    double dot = 0.0;
                    unsigned long d = 0UL;
                    while (d < headDim) {
                        dot = dot + Q->data[i * headDim + d] * K->data[(blockStart + j) * headDim + d];
                        d = d + 1UL;
                    }
                    double s = dot * scale;
                    scores[j] = s;
                    if (s > blockMax) blockMax = s;
                    j = j + 1UL;
                }

                double newMax = (m[i] > blockMax) ? m[i] : blockMax;
                double correction = exp_d(m[i] - newMax);

                double rowSumP = 0.0;
                double* p = (double*)malloc(sizeof(double) * bLen);
                j = 0UL;
                while (j < bLen) {
                    double pv = exp_d(scores[j] - newMax);
                    p[j] = pv;
                    rowSumP = rowSumP + pv;
                    j = j + 1UL;
                }

                unsigned long d2 = 0UL;
                while (d2 < headDim) {
                    double pv2 = 0.0;
                    j = 0UL;
                    while (j < bLen) { pv2 = pv2 + p[j] * V->data[(blockStart + j) * headDim + d2]; j = j + 1UL; }
                    out->data[i * headDim + d2] = out->data[i * headDim + d2] * correction + pv2;
                    d2 = d2 + 1UL;
                }

                l[i] = l[i] * correction + rowSumP;
                m[i] = newMax;

                free((void*)scores); free((void*)p);
                i = i + 1UL;
            }
            blockStart = blockEnd;
        }

        unsigned long ii = 0UL;
        while (ii < seqLen) {
            unsigned long d = 0UL;
            while (d < headDim) { out->data[ii * headDim + d] = out->data[ii * headDim + d] / l[ii]; d = d + 1UL; }
            ii = ii + 1UL;
        }
        free((void*)m); free((void*)l);
    }

    return out;
}

} // namespace std
