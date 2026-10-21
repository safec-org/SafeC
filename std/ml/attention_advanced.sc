// SafeC Standard Library — Advanced attention implementation (see
// attention_advanced.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/attention_advanced.h>
#include <std/ml/tensor.h>
#include <std/ml/attention.h>
#include <std/math.h>

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);

// ── MLA / EG-MLA ─────────────────────────────────────────────────────────────
static void aa_drop_(&Tensor t) {
    t->free();
    unsafe { free((void*)t); }
}

struct Mla Mla::new(unsigned long dModel, unsigned long dLatent, unsigned long numHeads, &Rng rng) {
    if (numHeads == 0UL || dModel % numHeads != 0UL) panic_at("Mla::new: numHeads must divide dModel", "attention_advanced.sc", __LINE__);
    struct Mla m;
    m.Wq = attn_weight_(dModel, dModel, rng);
    m.Wdkv = attn_weight_(dModel, dLatent, rng);
    m.Wuk = attn_weight_(dLatent, dModel, rng);
    m.Wuv = attn_weight_(dLatent, dModel, rng);
    m.numHeads = numHeads;
    return m;
}

&Tensor Mla::forward(const &Tensor X) const {
    &Tensor Q = tensor_matmul(X, self.Wq);
    &Tensor c_kv = tensor_matmul(X, self.Wdkv);
    &Tensor K = tensor_matmul(c_kv, self.Wuk);
    &Tensor V = tensor_matmul(c_kv, self.Wuv);
    &Tensor out = mha_forward(Q, K, V, self.numHeads);
    aa_drop_(Q); aa_drop_(c_kv); aa_drop_(K); aa_drop_(V);
    return out;
}

void Mla::collect_params(&Vec<&Tensor> out) const {
    out.push(self.Wq); out.push(self.Wdkv); out.push(self.Wuk); out.push(self.Wuv);
}

void Mla::collect_state(&Vec<&Tensor> out) const { self.collect_params(out); }

void Mla::free() { aa_drop_(self.Wq); aa_drop_(self.Wdkv); aa_drop_(self.Wuk); aa_drop_(self.Wuv); }

struct EgMla EgMla::new(unsigned long dModel, unsigned long dLatent, unsigned long numGroups,
                        unsigned long headsPerGroup, &Rng rng) {
    if (numGroups == 0UL || dModel % numGroups != 0UL || headsPerGroup == 0UL ||
        (dModel / numGroups) % headsPerGroup != 0UL)
        panic_at("EgMla::new: numGroups must divide dModel and headsPerGroup the group width",
                 "attention_advanced.sc", __LINE__);
    struct EgMla m;
    m.Wq = attn_weight_(dModel, dModel, rng);
    m.Wdkv = attn_weight_(dModel, dLatent, rng);
    m.Wuk = vec_new();
    m.Wuv = vec_new();
    unsigned long g = 0UL;
    while (g < numGroups) {
        m.Wuk.push(attn_weight_(dLatent, dModel / numGroups, rng));
        m.Wuv.push(attn_weight_(dLatent, dModel / numGroups, rng));
        g = g + 1UL;
    }
    m.headsPerGroup = headsPerGroup;
    return m;
}

&Tensor EgMla::forward(const &Tensor X) const {
    unsigned long numGroups = self.Wuk.length();
    if (numGroups == 0UL || self.Wuv.length() != numGroups) {
        panic_at("EgMla::forward: Wuk / Wuv must be non-empty and the same length", "attention_advanced.sc", __LINE__);
    }
    unsigned long seqLen; unsigned long dModel;
    unsafe { seqLen = X->shape[0]; dModel = self.Wq->shape[1]; }
    unsigned long groupDim = dModel / numGroups;

    &Tensor Q = tensor_matmul(X, self.Wq);
    &Tensor c_kv = tensor_matmul(X, self.Wdkv);
    &Tensor out = Tensor::new_2d(seqLen, dModel, 0);

    unsigned long g = 0UL;
    while (g < numGroups) {
        unsigned long startCol = g * groupDim;
        &Tensor Qg = Q.slice_cols(startCol, groupDim);
        &Tensor Kg = tensor_matmul(c_kv, *self.Wuk.at(g));
        &Tensor Vg = tensor_matmul(c_kv, *self.Wuv.at(g));
        &Tensor groupOut = mha_forward(Qg, Kg, Vg, self.headsPerGroup);
        tensor_set_cols(out, startCol, groupOut);
        aa_drop_(Qg); aa_drop_(Kg); aa_drop_(Vg); aa_drop_(groupOut);
        g = g + 1UL;
    }
    aa_drop_(Q); aa_drop_(c_kv);
    return out;
}

void EgMla::collect_params(&Vec<&Tensor> out) const {
    out.push(self.Wq);
    out.push(self.Wdkv);
    unsigned long g = 0UL;
    while (g < self.Wuk.length()) { out.push(*self.Wuk.at(g)); out.push(*self.Wuv.at(g)); g = g + 1UL; }
}

void EgMla::collect_state(&Vec<&Tensor> out) const { self.collect_params(out); }

void EgMla::free() {
    aa_drop_(self.Wq);
    aa_drop_(self.Wdkv);
    unsigned long g = 0UL;
    while (g < self.Wuk.length()) { aa_drop_(*self.Wuk.at(g)); aa_drop_(*self.Wuv.at(g)); g = g + 1UL; }
    self.Wuk.free();
    self.Wuv.free();
}

// ── Shifted-window attention ─────────────────────────────────────────────────
static &Tensor __row_slice(&Tensor in, unsigned long startRow, unsigned long numRows) {
    unsigned long cols;
    unsafe { cols = in->shape[1]; }
    &Tensor out = Tensor::new_2d(numRows, cols, 0);
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

static void __row_set(&Tensor dst, unsigned long startRow, &Tensor src) {
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

static &Tensor __row_cyclic_shift(&Tensor in, unsigned long shift) {
    unsigned long rows; unsigned long cols;
    unsafe { rows = in->shape[0]; cols = in->shape[1]; }
    &Tensor out = Tensor::new_2d(rows, cols, 0);
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
    &Tensor Qs; &Tensor Ks; &Tensor Vs;
    if (didShift) {
        Qs = __row_cyclic_shift(Q, shift);
        Ks = __row_cyclic_shift(K, shift);
        Vs = __row_cyclic_shift(V, shift);
    } else {
        Qs = Q; Ks = K; Vs = V;
    }

    &Tensor winOut = Tensor::new_2d(seqLen, dim, 0);
    unsigned long numWindows = seqLen / windowSize;
    unsigned long w = 0UL;
    while (w < numWindows) {
        unsigned long startRow = w * windowSize;
        &Tensor qw = __row_slice(Qs, startRow, windowSize);
        &Tensor kw = __row_slice(Ks, startRow, windowSize);
        &Tensor vw = __row_slice(Vs, startRow, windowSize);
        &Tensor ow = attention_forward(qw, kw, vw);
        __row_set(winOut, startRow, ow);

        qw->free(); kw->free(); vw->free(); ow->free();
        unsafe { free((void*)qw); free((void*)kw); free((void*)vw); free((void*)ow); }
        w = w + 1UL;
    }

    if (didShift) {
        Qs->free(); Ks->free(); Vs->free();
        unsafe { free((void*)Qs); free((void*)Ks); free((void*)Vs); }
        &Tensor out = __row_cyclic_shift(winOut, seqLen - shift);
        winOut->free();
        unsafe { free((void*)winOut); }
        return out;
    }
    return winOut;
}

// ── Gated DeltaNet ────────────────────────────────────────────────────────────
static float __gdn_init_weight(unsigned long i, unsigned long seed) {
    unsigned long h = (i + 1UL) * 2654435761UL + seed * 40503UL;
    h = (h ^ (h >> 13UL)) * 2246822519UL;
    return (((float)(h % 2000UL) / (float)1000.0) - (float)1.0) * (float)0.3;
}

struct GatedDeltaNet GatedDeltaNet::new(unsigned long dModel, unsigned long keyDim, unsigned long valueDim) {
    struct GatedDeltaNet layer;
    layer.Walpha = Tensor::new_2d(dModel, 1UL, 0);
    layer.Wbeta = Tensor::new_2d(dModel, 1UL, 0);
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

void GatedDeltaNet::free() {
    unsafe {
        self->Walpha->free(); self->Wbeta->free();
        free((void*)self->Walpha); free((void*)self->Wbeta);
    }
}

&Tensor GatedDeltaNet::forward(const &Tensor X, const &Tensor Q, const &Tensor K, const &Tensor V) const {
    unsigned long seqLen; unsigned long dModel;
    unsigned long keyDim; unsigned long valueDim;
    unsafe {
        seqLen = X->shape[0]; dModel = X->shape[1];
        keyDim = self->keyDim; valueDim = self->valueDim;
    }

    &Tensor out = Tensor::new_2d(seqLen, valueDim, 0);

    unsafe {
        float* S = (float*)malloc(sizeof(float) * keyDim * valueDim);
        unsigned long si = 0UL;
        while (si < keyDim * valueDim) { S[si] = (float)0.0; si = si + 1UL; }

        unsigned long t = 0UL;
        while (t < seqLen) {
            float alphaLogit = (float)0.0; float betaLogit = (float)0.0;
            unsigned long d = 0UL;
            while (d < dModel) {
                float xv = X->data[t * dModel + d];
                alphaLogit = alphaLogit + xv * self->Walpha->data[d];
                betaLogit = betaLogit + xv * self->Wbeta->data[d];
                d = d + 1UL;
            }
            float alpha = (float)1.0 / ((float)1.0 + exp_f(-alphaLogit));
            float beta = (float)1.0 / ((float)1.0 + exp_f(-betaLogit));

            float* pred = (float*)malloc(sizeof(float) * valueDim);
            unsigned long vd = 0UL;
            while (vd < valueDim) {
                float acc = (float)0.0;
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
                float kv = K->data[t * keyDim + kd2];
                unsigned long vd2 = 0UL;
                while (vd2 < valueDim) {
                    float deltaV = V->data[t * valueDim + vd2] - pred[vd2];
                    unsigned long idx = kd2 * valueDim + vd2;
                    S[idx] = alpha * S[idx] + beta * kv * deltaV;
                    vd2 = vd2 + 1UL;
                }
                kd2 = kd2 + 1UL;
            }
            free((void*)pred);

            unsigned long vd3 = 0UL;
            while (vd3 < valueDim) {
                float acc = (float)0.0;
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

void GatedDeltaNet::collect_params(&Vec<&Tensor> out) const {
    out.push(self.Walpha);
    out.push(self.Wbeta);
}

void GatedDeltaNet::collect_state(&Vec<&Tensor> out) const { self.collect_params(out); }

} // namespace std
