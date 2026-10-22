// SafeC Standard Library — DiT/JiT implementation (see transformer.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/transformer.h>
#include <std/ml/tensor.h>
#include <std/ml/attention.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/nn.h>
#include <std/ml/norm.h>

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);

static &FloatTensor __xf_init_weight(unsigned long rows, unsigned long cols, unsigned long seed) {
    &FloatTensor t = FloatTensor::new_2d(rows, cols, 1);
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

// A block's temporaries: freed after the forward pass unless its result is
// part of a graph (backward reads them).
static &FloatTensor xf_tmp_(&Vec<&FloatTensor> tmp, &FloatTensor t) {
    tmp.push(t);
    return t;
}

static void xf_release_(const &FloatTensor out, &Vec<&FloatTensor> tmp) {
    unsigned long i = 0UL;
    while (i < tmp.length()) { attn_release_tmp_(out, *tmp.at(i)); i = i + 1UL; }
    tmp.free();
}

// The feed-forward sublayer: relu(x W1 + b1) W2 + b2.
static &FloatTensor xf_ffn_(&Vec<&FloatTensor> tmp, const &FloatTensor x, const &FloatTensor W1, const &FloatTensor b1,
                            const &FloatTensor W2, const &FloatTensor b2) {
    &FloatTensor hid = xf_tmp_(tmp, tensor_add_bias(xf_tmp_(tmp, tensor_matmul(x, W1)), b1)).relu();
    return tensor_add_bias(xf_tmp_(tmp, tensor_matmul(xf_tmp_(tmp, hid), W2)), b2);
}

// Self-attention: mha(x Wq, x Wk, x Wv) Wo.
static &FloatTensor xf_attn_(&Vec<&FloatTensor> tmp, const &FloatTensor x, const &FloatTensor Wq, const &FloatTensor Wk,
                             const &FloatTensor Wv, const &FloatTensor Wo, unsigned long numHeads) {
    &FloatTensor Q = xf_tmp_(tmp, tensor_matmul(x, Wq));
    &FloatTensor K = xf_tmp_(tmp, tensor_matmul(x, Wk));
    &FloatTensor V = xf_tmp_(tmp, tensor_matmul(x, Wv));
    return tensor_matmul(xf_tmp_(tmp, mha_forward(Q, K, V, numHeads)), Wo);
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
    blk.b1 = FloatTensor::new_2d(1UL, dHidden, 1);
    blk.W2 = __xf_init_weight(dHidden, dModel, 106UL);
    blk.b2 = FloatTensor::new_2d(1UL, dModel, 1);
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
    unsigned long dModel;
    unsafe { dModel = x->shape[1]; }
    struct Vec<&FloatTensor> tmp = vec_new();

    &FloatTensor mod = xf_tmp_(&tmp, tensor_matmul(c, self.WadaLN));
    &FloatTensor shift1 = xf_tmp_(&tmp, mod.slice_cols(0UL, dModel));
    &FloatTensor scale1 = xf_tmp_(&tmp, mod.slice_cols(dModel, dModel));
    &FloatTensor gate1 = xf_tmp_(&tmp, mod.slice_cols(2UL * dModel, dModel));
    &FloatTensor shift2 = xf_tmp_(&tmp, mod.slice_cols(3UL * dModel, dModel));
    &FloatTensor scale2 = xf_tmp_(&tmp, mod.slice_cols(4UL * dModel, dModel));
    &FloatTensor gate2 = xf_tmp_(&tmp, mod.slice_cols(5UL * dModel, dModel));

    // x + gate1 * attn(LN(x) * (1 + scale1) + shift1)
    &FloatTensor h = xf_tmp_(&tmp, x.layer_norm((float)1e-5));
    &FloatTensor hScaled = xf_tmp_(&tmp, tensor_mul_row(h, xf_tmp_(&tmp, scale1.add_scalar((float)1.0))));
    &FloatTensor hMod = xf_tmp_(&tmp, tensor_add_bias(hScaled, shift1));
    &FloatTensor attnOut = xf_tmp_(&tmp, xf_attn_(&tmp, hMod, self.Wq, self.Wk, self.Wv, self.Wo, self.numHeads));
    &FloatTensor x1 = xf_tmp_(&tmp, tensor_add(x, xf_tmp_(&tmp, tensor_mul_row(attnOut, gate1))));

    // x1 + gate2 * ffn(LN(x1) * (1 + scale2) + shift2)
    &FloatTensor h2 = xf_tmp_(&tmp, x1.layer_norm((float)1e-5));
    &FloatTensor h2Scaled = xf_tmp_(&tmp, tensor_mul_row(h2, xf_tmp_(&tmp, scale2.add_scalar((float)1.0))));
    &FloatTensor h2Mod = xf_tmp_(&tmp, tensor_add_bias(h2Scaled, shift2));
    &FloatTensor ffnOut = xf_tmp_(&tmp, xf_ffn_(&tmp, h2Mod, self.W1, self.b1, self.W2, self.b2));
    &FloatTensor out = tensor_add(x1, xf_tmp_(&tmp, tensor_mul_row(ffnOut, gate2)));
    xf_release_(out, &tmp);
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
    blk.b1 = FloatTensor::new_2d(1UL, dHidden, 1);
    blk.W2 = __xf_init_weight(dHidden, dModel, 206UL);
    blk.b2 = FloatTensor::new_2d(1UL, dModel, 1);
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
    struct Vec<&FloatTensor> tmp = vec_new();
    &FloatTensor h = xf_tmp_(&tmp, x.layer_norm((float)1e-5));
    &FloatTensor attnOut = xf_tmp_(&tmp, xf_attn_(&tmp, h, self.Wq, self.Wk, self.Wv, self.Wo, self.numHeads));
    &FloatTensor x1 = xf_tmp_(&tmp, tensor_add(x, attnOut));
    &FloatTensor h2 = xf_tmp_(&tmp, x1.layer_norm((float)1e-5));
    &FloatTensor out = tensor_add(x1, xf_tmp_(&tmp, xf_ffn_(&tmp, h2, self.W1, self.b1, self.W2, self.b2)));
    xf_release_(out, &tmp);
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
    &FloatTensor x = tensor_add_bias(embedRaw, cond);
    attn_release_tmp_(x, embedRaw);

    unsigned long i = 0UL;
    while (i < self.blocks.length()) {
        &FloatTensor next = self.blocks.at(i).forward(x);
        attn_release_tmp_(next, x);
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
    attn_release_(self.WPatchEmbed);
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
