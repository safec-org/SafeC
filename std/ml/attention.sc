// SafeC Standard Library — Attention implementation (see attention.h).
#pragma once
#include <std/simd/vmath.h>
#include <std/stddef.h>
#include <std/ml/attention.h>
#include <std/ml/tensor.h>
#include <std/ml/activations.h>
#include <std/math.h>

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);

&FloatTensor FloatTensor::softmax_rows() const {
    unsigned long rows; unsigned long cols;
    unsafe { rows = self->shape[0]; cols = self->shape[1]; }
    &FloatTensor out = FloatTensor::new_2d(rows, cols, 0);
    unsigned long r = 0UL;
    while (r < rows) {
        float rowMax;
        unsafe { rowMax = self->data[r * cols]; }
        unsigned long c = 1UL;
        while (c < cols) {
            float v;
            unsafe { v = self->data[r * cols + c]; }
            if (v > rowMax) rowMax = v;
            c = c + 1UL;
        }
        unsafe {
            float* y = &out->data[r * cols];
            const float* xr = &self->data[r * cols];
            c = 0UL;
            while (c < cols) { y[c] = xr[c] - rowMax; c = c + 1UL; }
            vm_exp_f32((const float*)y, y, cols);
            float sum = (float)0.0;
            c = 0UL;
            while (c < cols) { sum = sum + y[c]; c = c + 1UL; }
            float inv = (float)1.0 / sum;
            c = 0UL;
            while (c < cols) { y[c] = y[c] * inv; c = c + 1UL; }
        }
        r = r + 1UL;
    }
    return out;
}

&FloatTensor FloatTensor::transpose() const {
    unsigned long rows; unsigned long cols;
    unsafe { rows = self->shape[0]; cols = self->shape[1]; }
    &FloatTensor out = FloatTensor::new_2d(cols, rows, 0);
    unsigned long r = 0UL;
    while (r < rows) {
        unsigned long c = 0UL;
        while (c < cols) {
            unsafe { out->data[c * rows + r] = self->data[r * cols + c]; }
            c = c + 1UL;
        }
        r = r + 1UL;
    }
    return out;
}

// Column slice, differentiable: the start column rides in extraScalar.
static void attn_slice_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(s, 0UL);
        if (!a->requiresGrad) return;
        __tensor_ensure_grad(a);
        unsigned long start = (unsigned long)s->extraScalar;
        unsigned long len = s->shape[1];
        unsigned long n = a->shape[1];
        unsigned long rows = s->shape[0];
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long c = 0UL;
            while (c < len) { a->grad[r * n + start + c] = a->grad[r * n + start + c] + s->grad[r * len + c]; c = c + 1UL; }
            r = r + 1UL;
        }
    }
}

&FloatTensor FloatTensor::slice_cols(unsigned long start, unsigned long len) const {
    unsigned long rows; unsigned long n;
    unsafe { rows = self->shape[0]; n = self->shape[1]; }
    &FloatTensor out = FloatTensor::new_2d(rows, len, 0);
    unsafe {
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long c = 0UL;
            while (c < len) { out->data[r * len + c] = self->data[r * n + start + c]; c = c + 1UL; }
            r = r + 1UL;
        }
        out->extraScalar = (float)start;
    }
    __tensor_link1(out, self, (void*)attn_slice_backward_);
    return out;
}

void tensor_set_cols(&FloatTensor dst, unsigned long startCol, const &FloatTensor src) {
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

&FloatTensor attention_forward(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V) {
    unsigned long headDim;
    unsafe { headDim = Q->shape[1]; }
    &FloatTensor Kt = K.transpose();
    &FloatTensor scores = tensor_matmul(Q, Kt);
    float scale = (float)1.0 / sqrt_f((float)headDim);
    &FloatTensor scaled = scores.scale(scale);
    &FloatTensor weights = scaled.softmax_rows();
    &FloatTensor result = tensor_matmul(weights, V);

    Kt->free(); scores->free(); scaled->free(); weights->free();
    unsafe { free((void*)Kt); free((void*)scores); free((void*)scaled); free((void*)weights); }
    return result;
}

&FloatTensor mha_forward(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V, unsigned long numHeads) {
    unsigned long seqLen; unsigned long dModel;
    unsafe { seqLen = Q->shape[0]; dModel = Q->shape[1]; }
    unsigned long headDim = dModel / numHeads;

    &FloatTensor out = FloatTensor::new_2d(seqLen, dModel, 0);
    unsigned long h = 0UL;
    while (h < numHeads) {
        unsigned long startCol = h * headDim;
        &FloatTensor Qh = Q.slice_cols(startCol, headDim);
        &FloatTensor Kh = K.slice_cols(startCol, headDim);
        &FloatTensor Vh = V.slice_cols(startCol, headDim);
        &FloatTensor headOut = attention_forward(Qh, Kh, Vh);
        tensor_set_cols(out, startCol, headOut);

        Qh->free(); Kh->free(); Vh->free(); headOut->free();
        unsafe { free((void*)Qh); free((void*)Kh); free((void*)Vh); free((void*)headOut); }
        h = h + 1UL;
    }
    return out;
}

&FloatTensor gated_attention_forward(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V,
                                 const &FloatTensor gateLogits) {
    &FloatTensor raw = attention_forward(Q, K, V);
    &FloatTensor gate = gateLogits.sigmoid_fwd();
    &FloatTensor out = tensor_mul(raw, gate);
    raw->free(); gate->free();
    unsafe { free((void*)raw); free((void*)gate); }
    return out;
}

&FloatTensor gated_mha_forward(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V,
                           unsigned long numHeads, const &FloatTensor gateLogits) {
    &FloatTensor raw = mha_forward(Q, K, V, numHeads);
    &FloatTensor gate = gateLogits.sigmoid_fwd();
    &FloatTensor out = tensor_mul(raw, gate);
    raw->free(); gate->free();
    unsafe { free((void*)raw); free((void*)gate); }
    return out;
}

// phi(x) = elu(x) + 1, elementwise (always > 0, used as linear
// attention's kernel feature map).
static &FloatTensor __phi_elu_p1(&FloatTensor x) {
    &FloatTensor out = x.zeros_like();
    unsafe {
        unsigned long i = 0UL;
        while (i < out->size) {
            float v = x->data[i];
            out->data[i] = (v > (float)0.0) ? (v + (float)1.0) : exp_f(v);
            i = i + 1UL;
        }
    }
    return out;
}

&FloatTensor linear_attention_forward(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V) {
    unsigned long seqLen; unsigned long headDim; unsigned long kvLen;
    unsafe { seqLen = Q->shape[0]; headDim = Q->shape[1]; kvLen = K->shape[0]; }

    &FloatTensor phiQ = __phi_elu_p1(Q);
    &FloatTensor phiK = __phi_elu_p1(K);
    &FloatTensor out = FloatTensor::new_2d(seqLen, headDim, 0);

    unsafe {
        // KV[d1][d2] = sum_j phiK[j][d1] * V[j][d2]   ([headDim, headDim])
        float* KV = (float*)malloc(sizeof(float) * headDim * headDim);
        // Z[d1] = sum_j phiK[j][d1]                   ([headDim])
        float* Z = (float*)malloc(sizeof(float) * headDim);
        unsigned long d1 = 0UL;
        while (d1 < headDim) {
            float zAcc = (float)0.0;
            unsigned long d2 = 0UL;
            while (d2 < headDim) {
                float acc = (float)0.0;
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
            float denom = (float)0.0;
            unsigned long dd = 0UL;
            while (dd < headDim) { denom = denom + phiQ->data[i * headDim + dd] * Z[dd]; dd = dd + 1UL; }
            unsigned long d2 = 0UL;
            while (d2 < headDim) {
                float numer = (float)0.0;
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

&FloatTensor flash_attention_forward(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V,
                                 unsigned long kvBlockSize) {
    unsigned long seqLen; unsigned long headDim; unsigned long kvLen;
    unsafe { seqLen = Q->shape[0]; headDim = Q->shape[1]; kvLen = K->shape[0]; }
    float scale = (float)1.0 / sqrt_f((float)headDim);

    &FloatTensor out = FloatTensor::new_2d(seqLen, headDim, 0); // doubles as the running 'acc' buffer

    unsafe {
        // -1.0e30, not double's -1.0e300 -- float32 only goes to ~3.4e38
        // in magnitude, and -1.0e300 would overflow to -inf. -1.0e30 is
        // still enormously larger than any real attention score.
        float* m = (float*)malloc(sizeof(float) * seqLen); // running row max
        float* l = (float*)malloc(sizeof(float) * seqLen); // running row sum
        unsigned long i0 = 0UL;
        while (i0 < seqLen) { m[i0] = (float)-1.0e30; l[i0] = (float)0.0; i0 = i0 + 1UL; }

        unsigned long blockStart = 0UL;
        while (blockStart < kvLen) {
            unsigned long blockEnd = blockStart + kvBlockSize;
            if (blockEnd > kvLen) blockEnd = kvLen;
            unsigned long bLen = blockEnd - blockStart;

            unsigned long i = 0UL;
            while (i < seqLen) {
                // scores[j] = scale * Q[i,:] . K[blockStart+j,:]
                float* scores = (float*)malloc(sizeof(float) * bLen);
                float blockMax = (float)-1.0e30;
                unsigned long j = 0UL;
                while (j < bLen) {
                    float dot = (float)0.0;
                    unsigned long d = 0UL;
                    while (d < headDim) {
                        dot = dot + Q->data[i * headDim + d] * K->data[(blockStart + j) * headDim + d];
                        d = d + 1UL;
                    }
                    float s = dot * scale;
                    scores[j] = s;
                    if (s > blockMax) blockMax = s;
                    j = j + 1UL;
                }

                float newMax = (m[i] > blockMax) ? m[i] : blockMax;
                float correction = exp_f(m[i] - newMax);

                float rowSumP = (float)0.0;
                float* p = (float*)malloc(sizeof(float) * bLen);
                j = 0UL;
                while (j < bLen) {
                    float pv = exp_f(scores[j] - newMax);
                    p[j] = pv;
                    rowSumP = rowSumP + pv;
                    j = j + 1UL;
                }

                unsigned long d2 = 0UL;
                while (d2 < headDim) {
                    float pv2 = (float)0.0;
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

// ── Multi-head self-attention layer ──────────────────────────────────────────

// A [rows, cols] weight, U(-1/sqrt(rows), 1/sqrt(rows)) (nn.h's Mlp init).
&FloatTensor attn_weight_(unsigned long rows, unsigned long cols, &Rng rng) {
    &FloatTensor w = FloatTensor::new_2d(rows, cols, 0);
    float bound = (float)1.0 / sqrt_f((float)rows);
    rng.fill_uniform(w, -bound, bound);
    return w;
}

static void attn_drop_(&FloatTensor t) {
    t->free();
    unsafe { free((void*)t); }
}

struct MultiHeadAttention MultiHeadAttention::new(unsigned long dModel, unsigned long numHeads, &Rng rng) {
    if (numHeads == 0UL || dModel % numHeads != 0UL) panic_at("MultiHeadAttention::new: numHeads must divide dModel", "attention.sc", __LINE__);
    struct MultiHeadAttention a;
    a.Wq = attn_weight_(dModel, dModel, rng);
    a.Wk = attn_weight_(dModel, dModel, rng);
    a.Wv = attn_weight_(dModel, dModel, rng);
    a.Wo = attn_weight_(dModel, dModel, rng);
    a.numHeads = numHeads;
    return a;
}

&FloatTensor MultiHeadAttention::forward(const &FloatTensor X) const {
    &FloatTensor Q = tensor_matmul(X, self.Wq);
    &FloatTensor K = tensor_matmul(X, self.Wk);
    &FloatTensor V = tensor_matmul(X, self.Wv);
    &FloatTensor h = mha_forward(Q, K, V, self.numHeads);
    &FloatTensor out = tensor_matmul(h, self.Wo);
    attn_drop_(Q); attn_drop_(K); attn_drop_(V); attn_drop_(h);
    return out;
}

void MultiHeadAttention::collect_params(&Vec<&FloatTensor> out) const {
    out.push(self.Wq); out.push(self.Wk); out.push(self.Wv); out.push(self.Wo);
}

void MultiHeadAttention::collect_state(&Vec<&FloatTensor> out) const { self.collect_params(out); }

void MultiHeadAttention::free() {
    attn_drop_(self.Wq); attn_drop_(self.Wk); attn_drop_(self.Wv); attn_drop_(self.Wo);
}

} // namespace std
