// SafeC Standard Library — DiT/JiT implementation (see transformer.h).
#pragma once
#include <std/ml/transformer.h>
#include <std/ml/tensor.h>
#include <std/ml/tensor.sc>
#include <std/ml/attention.h>
#include <std/ml/attention.sc>
#include <std/ml/activations.h>
#include <std/ml/activations.sc>
#include <std/mem.sc>

namespace std {

static struct Tensor* __xf_init_weight(unsigned long rows, unsigned long cols, unsigned long seed) {
    struct Tensor* t = tensor_new_2d(rows, cols, 0);
    unsafe {
        unsigned long i = 0UL;
        while (i < t->size) {
            unsigned long h = (i + 1UL) * 2654435761UL + seed * 40503UL;
            h = (h ^ (h >> 13UL)) * 2246822519UL;
            double v = ((double)(h % 2000UL) / 1000.0) - 1.0; // [-1, 1)
            t->data[i] = v * 0.2;
            i = i + 1UL;
        }
    }
    return t;
}

static struct Tensor* __add_broadcast_row(struct Tensor* mat, struct Tensor* row) {
    unsigned long rows; unsigned long cols;
    unsafe { rows = mat->shape[0]; cols = mat->shape[1]; }
    struct Tensor* out = tensor_new_2d(rows, cols, 0);
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

static struct Tensor* __mul_broadcast_row(struct Tensor* mat, struct Tensor* row) {
    unsigned long rows; unsigned long cols;
    unsafe { rows = mat->shape[0]; cols = mat->shape[1]; }
    struct Tensor* out = tensor_new_2d(rows, cols, 0);
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
static struct Tensor* __mul_broadcast_row_plus1(struct Tensor* mat, struct Tensor* row) {
    unsigned long rows; unsigned long cols;
    unsafe { rows = mat->shape[0]; cols = mat->shape[1]; }
    struct Tensor* out = tensor_new_2d(rows, cols, 0);
    unsafe {
        unsigned long r = 0UL;
        while (r < rows) {
            unsigned long c = 0UL;
            while (c < cols) {
                out->data[r * cols + c] = mat->data[r * cols + c] * (1.0 + row->data[c]);
                c = c + 1UL;
            }
            r = r + 1UL;
        }
    }
    return out;
}

// ── DiT ───────────────────────────────────────────────────────────────────────
struct DiTBlock dit_block_new(unsigned long dModel, unsigned long dHidden,
                               unsigned long dModelCond, unsigned long numHeads) {
    struct DiTBlock blk;
    blk.Wq = __xf_init_weight(dModel, dModel, 101UL);
    blk.Wk = __xf_init_weight(dModel, dModel, 102UL);
    blk.Wv = __xf_init_weight(dModel, dModel, 103UL);
    blk.Wo = __xf_init_weight(dModel, dModel, 104UL);
    blk.W1 = __xf_init_weight(dModel, dHidden, 105UL);
    blk.b1 = tensor_new_2d(1UL, dHidden, 0);
    blk.W2 = __xf_init_weight(dHidden, dModel, 106UL);
    blk.b2 = tensor_new_2d(1UL, dModel, 0);
    blk.WadaLN = __xf_init_weight(dModelCond, 6UL * dModel, 107UL);
    blk.numHeads = numHeads;
    return blk;
}

void dit_block_free(&DiTBlock block) {
    unsafe {
        block->Wq->free(); block->Wk->free(); block->Wv->free(); block->Wo->free();
        block->W1->free(); block->b1->free(); block->W2->free(); block->b2->free(); block->WadaLN->free();
        free((void*)block->Wq); free((void*)block->Wk); free((void*)block->Wv); free((void*)block->Wo);
        free((void*)block->W1); free((void*)block->b1); free((void*)block->W2); free((void*)block->b2);
        free((void*)block->WadaLN);
    }
}

&Tensor dit_block_forward(const &DiTBlock block, const &Tensor x, const &Tensor c) {
    struct Tensor* Wq; struct Tensor* Wk; struct Tensor* Wv; struct Tensor* Wo;
    struct Tensor* W1; struct Tensor* b1w; struct Tensor* W2; struct Tensor* b2w; struct Tensor* WadaLN;
    unsigned long numHeads; unsigned long dModel;
    unsafe {
        Wq = block->Wq; Wk = block->Wk; Wv = block->Wv; Wo = block->Wo;
        W1 = block->W1; b1w = block->b1; W2 = block->W2; b2w = block->b2; WadaLN = block->WadaLN;
        numHeads = block->numHeads;
        dModel = x->shape[1];
    }

    struct Tensor* mod = tensor_matmul(c, WadaLN);
    struct Tensor* shift1 = tensor_slice_cols(mod, 0UL, dModel);
    struct Tensor* scale1 = tensor_slice_cols(mod, dModel, dModel);
    struct Tensor* gate1 = tensor_slice_cols(mod, 2UL * dModel, dModel);
    struct Tensor* shift2 = tensor_slice_cols(mod, 3UL * dModel, dModel);
    struct Tensor* scale2 = tensor_slice_cols(mod, 4UL * dModel, dModel);
    struct Tensor* gate2 = tensor_slice_cols(mod, 5UL * dModel, dModel);

    struct Tensor* h = tensor_layernorm_rows(x, 1e-5);
    struct Tensor* hScaled = __mul_broadcast_row_plus1(h, scale1);
    struct Tensor* hMod = __add_broadcast_row(hScaled, shift1);
    struct Tensor* Q = tensor_matmul(hMod, Wq);
    struct Tensor* K = tensor_matmul(hMod, Wk);
    struct Tensor* V = tensor_matmul(hMod, Wv);
    struct Tensor* attnRaw = mha_forward(Q, K, V, numHeads);
    struct Tensor* attnOut = tensor_matmul(attnRaw, Wo);
    struct Tensor* gatedAttn = __mul_broadcast_row(attnOut, gate1);
    struct Tensor* x1 = tensor_residual_add(x, gatedAttn);

    struct Tensor* h2 = tensor_layernorm_rows(x1, 1e-5);
    struct Tensor* h2Scaled = __mul_broadcast_row_plus1(h2, scale2);
    struct Tensor* h2Mod = __add_broadcast_row(h2Scaled, shift2);
    struct Tensor* ffnHiddenRaw = tensor_matmul(h2Mod, W1);
    struct Tensor* ffnHiddenBias = __add_broadcast_row(ffnHiddenRaw, b1w);
    struct Tensor* ffnHidden = tensor_relu(ffnHiddenBias);
    struct Tensor* ffnOutRaw = tensor_matmul(ffnHidden, W2);
    struct Tensor* ffnOut = __add_broadcast_row(ffnOutRaw, b2w);
    struct Tensor* gatedFfn = __mul_broadcast_row(ffnOut, gate2);
    struct Tensor* out = tensor_residual_add(x1, gatedFfn);

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
struct JiTBlock jit_block_new(unsigned long dModel, unsigned long dHidden, unsigned long numHeads) {
    struct JiTBlock blk;
    blk.Wq = __xf_init_weight(dModel, dModel, 201UL);
    blk.Wk = __xf_init_weight(dModel, dModel, 202UL);
    blk.Wv = __xf_init_weight(dModel, dModel, 203UL);
    blk.Wo = __xf_init_weight(dModel, dModel, 204UL);
    blk.W1 = __xf_init_weight(dModel, dHidden, 205UL);
    blk.b1 = tensor_new_2d(1UL, dHidden, 0);
    blk.W2 = __xf_init_weight(dHidden, dModel, 206UL);
    blk.b2 = tensor_new_2d(1UL, dModel, 0);
    blk.numHeads = numHeads;
    return blk;
}

void jit_block_free(&JiTBlock block) {
    unsafe {
        block->Wq->free(); block->Wk->free(); block->Wv->free(); block->Wo->free();
        block->W1->free(); block->b1->free(); block->W2->free(); block->b2->free();
        free((void*)block->Wq); free((void*)block->Wk); free((void*)block->Wv); free((void*)block->Wo);
        free((void*)block->W1); free((void*)block->b1); free((void*)block->W2); free((void*)block->b2);
    }
}

&Tensor jit_block_forward(const &JiTBlock block, const &Tensor x) {
    struct Tensor* Wq; struct Tensor* Wk; struct Tensor* Wv; struct Tensor* Wo;
    struct Tensor* W1; struct Tensor* b1w; struct Tensor* W2; struct Tensor* b2w;
    unsigned long numHeads;
    unsafe {
        Wq = block->Wq; Wk = block->Wk; Wv = block->Wv; Wo = block->Wo;
        W1 = block->W1; b1w = block->b1; W2 = block->W2; b2w = block->b2;
        numHeads = block->numHeads;
    }

    struct Tensor* h = tensor_layernorm_rows(x, 1e-5);
    struct Tensor* Q = tensor_matmul(h, Wq);
    struct Tensor* K = tensor_matmul(h, Wk);
    struct Tensor* V = tensor_matmul(h, Wv);
    struct Tensor* attnRaw = mha_forward(Q, K, V, numHeads);
    struct Tensor* attnOut = tensor_matmul(attnRaw, Wo);
    struct Tensor* x1 = tensor_residual_add(x, attnOut);

    struct Tensor* h2 = tensor_layernorm_rows(x1, 1e-5);
    struct Tensor* ffnHiddenRaw = tensor_matmul(h2, W1);
    struct Tensor* ffnHiddenBias = __add_broadcast_row(ffnHiddenRaw, b1w);
    struct Tensor* ffnHidden = tensor_relu(ffnHiddenBias);
    struct Tensor* ffnOutRaw = tensor_matmul(ffnHidden, W2);
    struct Tensor* ffnOut = __add_broadcast_row(ffnOutRaw, b2w);
    struct Tensor* out = tensor_residual_add(x1, ffnOut);

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

&Tensor jit_forward(const &Tensor patches, const &Tensor WPatchEmbed, const &Tensor cond,
                     struct JiTBlock* blocks, unsigned long numBlocks) {
    struct Tensor* embedRaw = tensor_matmul(patches, WPatchEmbed);
    struct Tensor* x = __add_broadcast_row(embedRaw, cond);
    embedRaw->free();
    unsafe { free((void*)embedRaw); }

    unsigned long i = 0UL;
    while (i < numBlocks) {
        struct JiTBlock* blk;
        unsafe { blk = (struct JiTBlock*)&blocks[i]; }
        struct Tensor* next = jit_block_forward(blk, x);
        x->free();
        unsafe { free((void*)x); }
        x = next;
        i = i + 1UL;
    }
    return x;
}

} // namespace std
