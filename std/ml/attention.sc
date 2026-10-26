// SafeC Standard Library — Attention implementation (see attention.h).
#pragma once
#include <std/simd/vmath.h>
#include <std/stddef.h>
#include <std/ml/attention.h>
#include <std/ml/tensor.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/nn.h>
#include <std/ml/activations.h>
#include <std/math.h>

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);

// ── Temporaries ──────────────────────────────────────────────────────────────
void attn_release_(&FloatTensor t) {
    match (t->owner) { case some(a): { return; } case null: { } }   // its TensorArena owns it
    t->free();
    unsafe { free((void*)t); }
}

void attn_release_tmp_(const &FloatTensor out, &FloatTensor t) {
    int keep = 0;
    unsafe { keep = out->requiresGrad; }
    if (!keep) attn_release_(t);
}

// Whether ops on these record a graph.
static int attn_grad_(const &FloatTensor a, const &FloatTensor b, const &FloatTensor c) {
    if (!Autograd::get().enabled()) return 0;
    unsafe { return a->requiresGrad || b->requiresGrad || c->requiresGrad; }
}

static int attn_on_cpu_(const &FloatTensor a, const &FloatTensor b, const &FloatTensor c) {
    unsafe { return a->dev == DEVICE_CPU && b->dev == DEVICE_CPU && c->dev == DEVICE_CPU; }
}

// ── Softmax and transpose ────────────────────────────────────────────────────
&FloatTensor FloatTensor::softmax_rows() const { return self.softmax(); }

&FloatTensor FloatTensor::softmax_rows_mask(int causal, unsigned long window) const {
    if (!causal && window == 0UL) return self.softmax();
    match (tdev_softmax_mask_(self, causal, window)) { case some(o): { return o; } case null: { } }
    unsafe { tensor_host_raw_((void*)self); }
    unsigned long rows; unsigned long cols;
    unsafe { cols = self->shape[self->ndim - 1UL]; rows = self->size / cols; }
    long off = (long)cols - (long)rows;
    &FloatTensor out = __tensor_alloc_uninit_like(self);
    unsigned long r = 0UL;
    while (r < rows) {
        long pos = (long)r + off;
        // visible keys: [lo, hi)
        long lo = window > 0UL ? pos - (long)window + 1L : 0L;
        long hi = causal ? pos + 1L : (long)cols;
        if (lo < 0L) lo = 0L;
        if (hi > (long)cols) hi = (long)cols;
        unsafe {
            float* y = &out->data[r * cols];
            const float* xr = &self->data[r * cols];
            unsigned long c = 0UL;
            while (c < cols) { y[c] = (float)0.0; c = c + 1UL; }
            if (lo < hi) {
                unsigned long n = (unsigned long)(hi - lo);
                float* yv = &y[(unsigned long)lo];
                const float* xv = &xr[(unsigned long)lo];
                float rowMax = xv[0];
                c = 1UL;
                while (c < n) { if (xv[c] > rowMax) rowMax = xv[c]; c = c + 1UL; }
                c = 0UL;
                while (c < n) { yv[c] = xv[c] - rowMax; c = c + 1UL; }
                vm_exp_f32((const float*)yv, yv, n);
                float sum = (float)0.0;
                c = 0UL;
                while (c < n) { sum = sum + yv[c]; c = c + 1UL; }
                float inv = (float)1.0 / sum;
                c = 0UL;
                while (c < n) { yv[c] = yv[c] * inv; c = c + 1UL; }
            }
        }
        r = r + 1UL;
    }
    // hidden entries are 0: the softmax backward gives them no gradient
    __tensor_link1(out, self, (void*)__softmax_backward);
    return out;
}

static void attn_transpose_backward_(void* selfPtr) {
    unsafe {
        FloatTensor* s = (FloatTensor*)selfPtr;
        FloatTensor* a = __tensor_parent(s, 0UL);
        if (!a->requiresGrad) return;
        __tensor_ensure_grad(a);
        unsigned long rows = a->shape[0];
        unsigned long cols = a->shape[1];
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long c = 0UL;
            while (c < cols) { a->grad[r * cols + c] = a->grad[r * cols + c] + s->grad[c * rows + r]; c = c + 1UL; }
            r = r + 1UL;
        }
    }
}

&FloatTensor FloatTensor::transpose() const {
    match (tdev_transpose_(self)) { case some(o): { return o; } case null: { } }
    unsafe { tensor_host_raw_((void*)self); }
    unsigned long rows; unsigned long cols;
    unsafe { rows = self->shape[0]; cols = self->shape[1]; }
    unsigned long shape[2];
    unsafe { shape[0] = cols; shape[1] = rows; }
    &FloatTensor out;
    unsafe { out = __tensor_alloc((const unsigned long*)&shape[0], 2UL, 0); }
    unsigned long r = 0UL;
    while (r < rows) {
        unsigned long c = 0UL;
        while (c < cols) {
            unsafe { out->data[c * rows + r] = self->data[r * cols + c]; }
            c = c + 1UL;
        }
        r = r + 1UL;
    }
    __tensor_link1(out, self, (void*)attn_transpose_backward_);
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
    match (tdev_slice_cols_(self, start, len)) { case some(o): { return o; } case null: { } }
    unsafe { tensor_host_raw_((void*)self); }
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
    unsafe { tensor_host_write_raw_((void*)dst); tensor_host_raw_((void*)src); }
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

// ── Scaled dot-product and multi-head attention ─────────────────────────────
static int g_attn_fused = 1;

int attention_set_fused(int on) {
    int was = g_attn_fused;
    g_attn_fused = on ? 1 : 0;
    return was;
}

// One head from the composed ops: softmax(Q K^T / sqrt(d), masked) V.
static &FloatTensor attn_composed_(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V, int causal,
                                   unsigned long window) {
    unsigned long headDim;
    unsafe { headDim = Q->shape[1]; }
    &FloatTensor Kt = K.transpose();
    &FloatTensor scores = tensor_matmul(Q, Kt);
    &FloatTensor scaled = scores.scale((float)1.0 / sqrt_f((float)headDim));
    &FloatTensor weights = scaled.softmax_rows_mask(causal, window);
    &FloatTensor result = tensor_matmul(weights, V);
    attn_release_tmp_(result, Kt); attn_release_tmp_(result, scores);
    attn_release_tmp_(result, scaled); attn_release_tmp_(result, weights);
    return result;
}

&FloatTensor attention_forward(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V) {
    return mha_forward_mask(Q, K, V, 1UL, 0, 0UL);
}

&FloatTensor attention_forward_mask(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V, int causal,
                                    unsigned long window) {
    return mha_forward_mask(Q, K, V, 1UL, causal, window);
}

&FloatTensor mha_forward(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V, unsigned long numHeads) {
    return mha_forward_mask(Q, K, V, numHeads, 0, 0UL);
}

&FloatTensor mha_forward_mask(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V, unsigned long numHeads,
                              int causal, unsigned long window) {
    unsigned long dModel; unsigned long dV;
    unsafe { dModel = Q->shape[1]; dV = V->shape[1]; }
    unsigned long headDim = dModel / numHeads;
    unsigned long headDimV = dV / numHeads;
    // on a GPU: every head in one launch
    if (g_attn_fused && !attn_on_cpu_(Q, K, V)) {
        match (tensor_attention_fused_(Q, K, V, numHeads, causal, window, (float)1.0 / sqrt_f((float)headDim))) {
            case some(o): { return o; }
            case null: { }
        }
    }
    if (numHeads == 1UL) return attn_composed_(Q, K, V, causal, window);
    // head by head, the outputs joined column-wise
    ?&FloatTensor acc = null;
    unsigned long h = 0UL;
    while (h < numHeads) {
        &FloatTensor Qh = Q.slice_cols(h * headDim, headDim);
        &FloatTensor Kh = K.slice_cols(h * headDim, headDim);
        &FloatTensor Vh = V.slice_cols(h * headDimV, headDimV);
        &FloatTensor headOut = attn_composed_(Qh, Kh, Vh, causal, window);
        attn_release_tmp_(headOut, Qh); attn_release_tmp_(headOut, Kh); attn_release_tmp_(headOut, Vh);
        match (acc) {
            case some(prev): {
                &FloatTensor joined = tensor_concat_cols(prev, headOut);
                attn_release_tmp_(joined, prev); attn_release_tmp_(joined, headOut);
                acc = (?&FloatTensor)joined;
            }
            case null: { acc = (?&FloatTensor)headOut; }
        }
        h = h + 1UL;
    }
    match (acc) { case some(o): { return o; } case null: { } }
    panic_at("mha_forward: numHeads must be > 0", "attention.sc", __LINE__);
    return Q.scale((float)0.0);
}

&FloatTensor gated_attention_forward(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V,
                                 const &FloatTensor gateLogits) {
    &FloatTensor raw = attention_forward(Q, K, V);
    &FloatTensor gate = gateLogits.sigmoid();
    &FloatTensor out = tensor_mul(raw, gate);
    attn_release_tmp_(out, raw); attn_release_tmp_(out, gate);
    return out;
}

&FloatTensor gated_mha_forward(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V,
                           unsigned long numHeads, const &FloatTensor gateLogits) {
    &FloatTensor raw = mha_forward(Q, K, V, numHeads);
    &FloatTensor gate = gateLogits.sigmoid();
    &FloatTensor out = tensor_mul(raw, gate);
    attn_release_tmp_(out, raw); attn_release_tmp_(out, gate);
    return out;
}

// phi(x) = elu(x) + 1, elementwise (always > 0, used as linear
// attention's kernel feature map).
static &FloatTensor __phi_elu_p1(&FloatTensor x) {
    unsafe { tensor_host_write_raw_((void*)x); }
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

// The composed form (gradients, devices): phi(Q) (phi(K)^T V) divided row by
// row by phi(Q) (phi(K)^T 1).
static &FloatTensor attn_linear_graph_(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V) {
    unsigned long dV;
    unsafe { dV = V->shape[1]; }
    &FloatTensor eq = Q.elu((float)1.0);
    &FloatTensor phiQ = eq.add_scalar((float)1.0);
    &FloatTensor ek = K.elu((float)1.0);
    &FloatTensor phiK = ek.add_scalar((float)1.0);
    &FloatTensor phiKt = phiK.transpose();               // [headDim, kvLen]
    &FloatTensor KV = tensor_matmul(phiKt, V);           // [headDim, dV]
    &FloatTensor Z = phiKt.sum_rows();                   // [headDim, 1]
    &FloatTensor numer = tensor_matmul(phiQ, KV);        // [seqLen, dV]
    &FloatTensor denom = tensor_matmul(phiQ, Z);         // [seqLen, 1], > 0
    &FloatTensor ld = denom.log();
    &FloatTensor nld = ld.scale((float)-1.0);
    &FloatTensor inv = nld.exp();                        // 1 / denom
    &FloatTensor ones = FloatTensor::new_2d(1UL, dV, 0);
    ones.fill((float)1.0);
    &FloatTensor invB = tensor_matmul(inv, ones);        // across the columns
    &FloatTensor out = tensor_mul(numer, invB);
    attn_release_tmp_(out, eq); attn_release_tmp_(out, phiQ); attn_release_tmp_(out, ek); attn_release_tmp_(out, phiK);
    attn_release_tmp_(out, phiKt); attn_release_tmp_(out, KV); attn_release_tmp_(out, Z); attn_release_tmp_(out, numer);
    attn_release_tmp_(out, denom); attn_release_tmp_(out, ld); attn_release_tmp_(out, nld); attn_release_tmp_(out, inv);
    attn_release_tmp_(out, ones); attn_release_tmp_(out, invB);
    return out;
}

&FloatTensor linear_attention_forward(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V) {
    if (attn_grad_(Q, K, V) || !attn_on_cpu_(Q, K, V)) return attn_linear_graph_(Q, K, V);
    unsafe { tensor_host_raw_((void*)Q); tensor_host_raw_((void*)K); tensor_host_raw_((void*)V); }
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
    if (attn_grad_(Q, K, V) || !attn_on_cpu_(Q, K, V)) {
        // the fused op: the same online softmax, with a backward
        match (tensor_attention_fused_(Q, K, V, 1UL, 0, 0UL, scale)) { case some(o): { return o; } case null: { } }
        return attn_composed_(Q, K, V, 0, 0UL);
    }
    unsafe { tensor_host_raw_((void*)Q); tensor_host_raw_((void*)K); tensor_host_raw_((void*)V); }

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
    &FloatTensor w = FloatTensor::new_2d(rows, cols, 1);
    float bound = (float)1.0 / sqrt_f((float)rows);
    rng.fill_uniform(w, -bound, bound);
    return w;
}

static void attn_drop_(&FloatTensor t) { attn_release_(t); }

struct MultiHeadAttention MultiHeadAttention::new(unsigned long dModel, unsigned long numHeads, &Rng rng) {
    if (numHeads == 0UL || dModel % numHeads != 0UL) panic_at("MultiHeadAttention::new: numHeads must divide dModel", "attention.sc", __LINE__);
    struct MultiHeadAttention a;
    a.Wq = attn_weight_(dModel, dModel, rng);
    a.Wk = attn_weight_(dModel, dModel, rng);
    a.Wv = attn_weight_(dModel, dModel, rng);
    a.Wo = attn_weight_(dModel, dModel, rng);
    a.numHeads = numHeads;
    a.causal = 0;
    a.window = 0UL;
    return a;
}

&FloatTensor MultiHeadAttention::forward(const &FloatTensor X) const {
    &FloatTensor Q = tensor_matmul(X, self.Wq);
    &FloatTensor K = tensor_matmul(X, self.Wk);
    &FloatTensor V = tensor_matmul(X, self.Wv);
    &FloatTensor h = mha_forward_mask(Q, K, V, self.numHeads, self.causal, self.window);
    &FloatTensor out = tensor_matmul(h, self.Wo);
    attn_release_tmp_(out, Q); attn_release_tmp_(out, K); attn_release_tmp_(out, V); attn_release_tmp_(out, h);
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
