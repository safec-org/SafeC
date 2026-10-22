// SafeC Standard Library — DiT/JiT implementation (see transformer.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/transformer.h>
#include <std/ml/tensor.h>
#include <std/ml/attention.h>
#include <std/ml/activations.h>

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);

static &FloatTensor __xf_init_weight(unsigned long rows, unsigned long cols, unsigned long seed) {
    &FloatTensor t = FloatTensor::new_2d(rows, cols, 0);
    unsafe {
        unsigned long i = 0UL;
        while (i < t->size) {
            unsigned long h = (i + 1UL) * 2654435761UL + seed * 40503UL;
            h = (h ^ (h >> 13UL)) * 2246822519UL;
            float v = ((float)(h % 2000UL) / (float)1000.0) - (float)1.0; // [-1, 1)
            t->data[i] = v * (float)0.2;
            i = i + 1UL;
        }
    }
    return t;
}

static &FloatTensor __add_broadcast_row(&FloatTensor mat, &FloatTensor row) {
    unsafe { tensor_host_write_((void*)mat); tensor_host_write_((void*)row); }
    unsigned long rows; unsigned long cols;
    unsafe { rows = mat->shape[0]; cols = mat->shape[1]; }
    &FloatTensor out = FloatTensor::new_2d(rows, cols, 0);
    unsafe {
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long c = 0UL;
            while (c < cols) { out->data[r * cols + c] = mat->data[r * cols + c] + row->data[c]; c = c + 1UL; }
            r = r + 1UL;
        }
    }
    return out;
}

static &FloatTensor __mul_broadcast_row(&FloatTensor mat, &FloatTensor row) {
    unsafe { tensor_host_write_((void*)mat); tensor_host_write_((void*)row); }
    unsigned long rows; unsigned long cols;
    unsafe { rows = mat->shape[0]; cols = mat->shape[1]; }
    &FloatTensor out = FloatTensor::new_2d(rows, cols, 0);
    unsafe {
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long c = 0UL;
            while (c < cols) { out->data[r * cols + c] = mat->data[r * cols + c] * row->data[c]; c = c + 1UL; }
            r = r + 1UL;
        }
    }
    return out;
}

// mat * (1 + row), broadcast — the "scale" half of AdaLN modulation.
static &FloatTensor __mul_broadcast_row_plus1(&FloatTensor mat, &FloatTensor row) {
    unsafe { tensor_host_write_((void*)mat); tensor_host_write_((void*)row); }
    unsigned long rows; unsigned long cols;
    unsafe { rows = mat->shape[0]; cols = mat->shape[1]; }
    &FloatTensor out = FloatTensor::new_2d(rows, cols, 0);
    unsafe {
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long c = 0UL;
            while (c < cols) {
                out->data[r * cols + c] = mat->data[r * cols + c] * ((float)1.0 + row->data[c]);
                c = c + 1UL;
            }
            r = r + 1UL;
        }
    }
    return out;
}

// ── DiT ───────────────────────────────────────────────────────────────────────
struct DiTBlock DiTBlock::new(unsigned long dModel, unsigned long dHidden, unsigned long dModelCond,
                              unsigned long numHeads) {
    struct DiTBlock blk;
    blk.Wq = __xf_init_weight(dModel, dModel, 101UL);
    blk.Wk = __xf_init_weight(dModel, dModel, 102UL);
    blk.Wv = __xf_init_weight(dModel, dModel, 103UL);
    blk.Wo = __xf_init_weight(dModel, dModel, 104UL);
    blk.W1 = __xf_init_weight(dModel, dHidden, 105UL);
    blk.b1 = FloatTensor::new_2d(1UL, dHidden, 0);
    blk.W2 = __xf_init_weight(dHidden, dModel, 106UL);
    blk.b2 = FloatTensor::new_2d(1UL, dModel, 0);
    blk.WadaLN = __xf_init_weight(dModelCond, 6UL * dModel, 107UL);
    blk.numHeads = numHeads;
    return blk;
}

void DiTBlock::free() {
    unsafe {
        self->Wq->free(); self->Wk->free(); self->Wv->free(); self->Wo->free();
        self->W1->free(); self->b1->free(); self->W2->free(); self->b2->free(); self->WadaLN->free();
        free((void*)self->Wq); free((void*)self->Wk); free((void*)self->Wv); free((void*)self->Wo);
        free((void*)self->W1); free((void*)self->b1); free((void*)self->W2); free((void*)self->b2);
        free((void*)self->WadaLN);
    }
}

&FloatTensor DiTBlock::forward(const &FloatTensor x, const &FloatTensor c) const {
    unsafe { tensor_host_((void*)x); }
    &FloatTensor Wq; &FloatTensor Wk; &FloatTensor Wv; &FloatTensor Wo;
    &FloatTensor W1; &FloatTensor b1w; &FloatTensor W2; &FloatTensor b2w; &FloatTensor WadaLN;
    unsigned long numHeads; unsigned long dModel;
    unsafe {
        Wq = self->Wq; Wk = self->Wk; Wv = self->Wv; Wo = self->Wo;
        W1 = self->W1; b1w = self->b1; W2 = self->W2; b2w = self->b2; WadaLN = self->WadaLN;
        numHeads = self->numHeads;
        dModel = x->shape[1];
    }

    &FloatTensor mod = tensor_matmul(c, WadaLN);
    &FloatTensor shift1 = mod.slice_cols(0UL, dModel);
    &FloatTensor scale1 = mod.slice_cols(dModel, dModel);
    &FloatTensor gate1 = mod.slice_cols(2UL * dModel, dModel);
    &FloatTensor shift2 = mod.slice_cols(3UL * dModel, dModel);
    &FloatTensor scale2 = mod.slice_cols(4UL * dModel, dModel);
    &FloatTensor gate2 = mod.slice_cols(5UL * dModel, dModel);

    &FloatTensor h = x.layernorm_rows((float)1e-5);
    &FloatTensor hScaled = __mul_broadcast_row_plus1(h, scale1);
    &FloatTensor hMod = __add_broadcast_row(hScaled, shift1);
    &FloatTensor Q = tensor_matmul(hMod, Wq);
    &FloatTensor K = tensor_matmul(hMod, Wk);
    &FloatTensor V = tensor_matmul(hMod, Wv);
    &FloatTensor attnRaw = mha_forward(Q, K, V, numHeads);
    &FloatTensor attnOut = tensor_matmul(attnRaw, Wo);
    &FloatTensor gatedAttn = __mul_broadcast_row(attnOut, gate1);
    &FloatTensor x1 = tensor_residual_add(x, gatedAttn);

    &FloatTensor h2 = x1.layernorm_rows((float)1e-5);
    &FloatTensor h2Scaled = __mul_broadcast_row_plus1(h2, scale2);
    &FloatTensor h2Mod = __add_broadcast_row(h2Scaled, shift2);
    &FloatTensor ffnHiddenRaw = tensor_matmul(h2Mod, W1);
    &FloatTensor ffnHiddenBias = __add_broadcast_row(ffnHiddenRaw, b1w);
    &FloatTensor ffnHidden = ffnHiddenBias.relu_fwd();
    &FloatTensor ffnOutRaw = tensor_matmul(ffnHidden, W2);
    &FloatTensor ffnOut = __add_broadcast_row(ffnOutRaw, b2w);
    &FloatTensor gatedFfn = __mul_broadcast_row(ffnOut, gate2);
    &FloatTensor out = tensor_residual_add(x1, gatedFfn);

    mod->free(); shift1->free(); scale1->free(); gate1->free(); shift2->free(); scale2->free(); gate2->free();
    h->free(); hScaled->free(); hMod->free(); Q->free(); K->free(); V->free();
    attnRaw->free(); attnOut->free(); gatedAttn->free(); x1->free();
    h2->free(); h2Scaled->free(); h2Mod->free(); ffnHiddenRaw->free(); ffnHiddenBias->free();
    ffnHidden->free(); ffnOutRaw->free(); ffnOut->free(); gatedFfn->free();
    unsafe {
        free((void*)mod); free((void*)shift1); free((void*)scale1); free((void*)gate1);
        free((void*)shift2); free((void*)scale2); free((void*)gate2);
        free((void*)h); free((void*)hScaled); free((void*)hMod); free((void*)Q); free((void*)K); free((void*)V);
        free((void*)attnRaw); free((void*)attnOut); free((void*)gatedAttn); free((void*)x1);
        free((void*)h2); free((void*)h2Scaled); free((void*)h2Mod);
        free((void*)ffnHiddenRaw); free((void*)ffnHiddenBias);
        free((void*)ffnHidden); free((void*)ffnOutRaw); free((void*)ffnOut); free((void*)gatedFfn);
    }
    return out;
}

// ── JiT ───────────────────────────────────────────────────────────────────────
struct JiTBlock JiTBlock::new(unsigned long dModel, unsigned long dHidden, unsigned long numHeads) {
    struct JiTBlock blk;
    blk.Wq = __xf_init_weight(dModel, dModel, 201UL);
    blk.Wk = __xf_init_weight(dModel, dModel, 202UL);
    blk.Wv = __xf_init_weight(dModel, dModel, 203UL);
    blk.Wo = __xf_init_weight(dModel, dModel, 204UL);
    blk.W1 = __xf_init_weight(dModel, dHidden, 205UL);
    blk.b1 = FloatTensor::new_2d(1UL, dHidden, 0);
    blk.W2 = __xf_init_weight(dHidden, dModel, 206UL);
    blk.b2 = FloatTensor::new_2d(1UL, dModel, 0);
    blk.numHeads = numHeads;
    return blk;
}

void JiTBlock::free() {
    unsafe {
        self->Wq->free(); self->Wk->free(); self->Wv->free(); self->Wo->free();
        self->W1->free(); self->b1->free(); self->W2->free(); self->b2->free();
        free((void*)self->Wq); free((void*)self->Wk); free((void*)self->Wv); free((void*)self->Wo);
        free((void*)self->W1); free((void*)self->b1); free((void*)self->W2); free((void*)self->b2);
    }
}

&FloatTensor JiTBlock::forward(const &FloatTensor x) const {
    &FloatTensor Wq; &FloatTensor Wk; &FloatTensor Wv; &FloatTensor Wo;
    &FloatTensor W1; &FloatTensor b1w; &FloatTensor W2; &FloatTensor b2w;
    unsigned long numHeads;
    unsafe {
        Wq = self->Wq; Wk = self->Wk; Wv = self->Wv; Wo = self->Wo;
        W1 = self->W1; b1w = self->b1; W2 = self->W2; b2w = self->b2;
        numHeads = self->numHeads;
    }

    &FloatTensor h = x.layernorm_rows((float)1e-5);
    &FloatTensor Q = tensor_matmul(h, Wq);
    &FloatTensor K = tensor_matmul(h, Wk);
    &FloatTensor V = tensor_matmul(h, Wv);
    &FloatTensor attnRaw = mha_forward(Q, K, V, numHeads);
    &FloatTensor attnOut = tensor_matmul(attnRaw, Wo);
    &FloatTensor x1 = tensor_residual_add(x, attnOut);

    &FloatTensor h2 = x1.layernorm_rows((float)1e-5);
    &FloatTensor ffnHiddenRaw = tensor_matmul(h2, W1);
    &FloatTensor ffnHiddenBias = __add_broadcast_row(ffnHiddenRaw, b1w);
    &FloatTensor ffnHidden = ffnHiddenBias.relu_fwd();
    &FloatTensor ffnOutRaw = tensor_matmul(ffnHidden, W2);
    &FloatTensor ffnOut = __add_broadcast_row(ffnOutRaw, b2w);
    &FloatTensor out = tensor_residual_add(x1, ffnOut);

    h->free(); Q->free(); K->free(); V->free(); attnRaw->free(); attnOut->free(); x1->free();
    h2->free(); ffnHiddenRaw->free(); ffnHiddenBias->free(); ffnHidden->free(); ffnOutRaw->free(); ffnOut->free();
    unsafe {
        free((void*)h); free((void*)Q); free((void*)K); free((void*)V);
        free((void*)attnRaw); free((void*)attnOut); free((void*)x1);
        free((void*)h2); free((void*)ffnHiddenRaw); free((void*)ffnHiddenBias);
        free((void*)ffnHidden); free((void*)ffnOutRaw); free((void*)ffnOut);
    }
    return out;
}

struct JiT JiT::new(unsigned long patchDim, unsigned long dModel, unsigned long dHidden, unsigned long numHeads,
                    unsigned long depth) {
    struct JiT m;
    m.WPatchEmbed = __xf_init_weight(patchDim, dModel, 301UL);
    m.blocks = vec_new();
    unsigned long i = 0UL;
    while (i < depth) { m.blocks.push(JiTBlock::new(dModel, dHidden, numHeads)); i = i + 1UL; }
    return m;
}

&FloatTensor JiT::forward(const &FloatTensor patches, const &FloatTensor cond) const {
    &FloatTensor embedRaw = tensor_matmul(patches, self.WPatchEmbed);
    &FloatTensor x = __add_broadcast_row(embedRaw, cond);
    embedRaw->free();
    unsafe { free((void*)embedRaw); }

    unsigned long i = 0UL;
    while (i < self.blocks.length()) {
        &FloatTensor next = self.blocks.at(i).forward(x);
        x->free();
        unsafe { free((void*)x); }
        x = next;
        i = i + 1UL;
    }
    return x;
}

void JiT::collect_params(&Vec<&FloatTensor> out) const {
    out.push(self.WPatchEmbed);
    unsigned long i = 0UL;
    while (i < self.blocks.length()) { self.blocks.at(i).collect_params(out); i = i + 1UL; }
}

void JiT::collect_state(&Vec<&FloatTensor> out) const { self.collect_params(out); }

void JiT::free() {
    self.WPatchEmbed->free();
    unsafe { free((void*)self.WPatchEmbed); }
    unsigned long i = 0UL;
    while (i < self.blocks.length()) { self.blocks.at(i).free(); i = i + 1UL; }
    self.blocks.free();
}

void DiTBlock::collect_params(&Vec<&FloatTensor> out) const {
    out.push(self.Wq);
    out.push(self.Wk);
    out.push(self.Wv);
    out.push(self.Wo);
    out.push(self.W1);
    out.push(self.b1);
    out.push(self.W2);
    out.push(self.b2);
    out.push(self.WadaLN);
}

void DiTBlock::collect_state(&Vec<&FloatTensor> out) const { self.collect_params(out); }

void JiTBlock::collect_params(&Vec<&FloatTensor> out) const {
    out.push(self.Wq);
    out.push(self.Wk);
    out.push(self.Wv);
    out.push(self.Wo);
    out.push(self.W1);
    out.push(self.b1);
    out.push(self.W2);
    out.push(self.b2);
}

void JiTBlock::collect_state(&Vec<&FloatTensor> out) const { self.collect_params(out); }

} // namespace std
