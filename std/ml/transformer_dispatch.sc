// SafeC Standard Library — device-selectable JiTBlock forward pass
// implementation (see transformer_dispatch.h).
#pragma once
#include <std/stddef.h>
#include <std/ml/transformer_dispatch.h>

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);

// Exact duplicate of transformer.sc's own private __add_broadcast_row —
// that one is 'static' (transformer.sc-private, per this codebase's
// header/source separation convention: see tensor.h's own comment on
// why 'static' helpers can't be called from a different file), and this
// file needs the identical broadcast-add-with-no-backward-link behavior
// for the FFN bias adds. Forward-only (no __tensor_link call) in both
// copies — bias gradients aren't computed through this path in either
// version, a pre-existing property of JiTBlock's design, not something
// introduced here.
static &FloatTensor __add_broadcast_row_dispatch(&FloatTensor mat, &FloatTensor row) {
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

&FloatTensor jit_block_forward_on(const &JiTBlock block, const &FloatTensor x, enum Device device) {
    &FloatTensor Wq; &FloatTensor Wk; &FloatTensor Wv; &FloatTensor Wo;
    &FloatTensor W1; &FloatTensor b1w; &FloatTensor W2; &FloatTensor b2w;
    unsigned long numHeads;
    unsafe {
        Wq = block->Wq; Wk = block->Wk; Wv = block->Wv; Wo = block->Wo;
        W1 = block->W1; b1w = block->b1; W2 = block->W2; b2w = block->b2;
        numHeads = block->numHeads;
    }

    &FloatTensor h = x.layernorm_rows(1e-5);
    &FloatTensor Q = tensor_matmul_on(h, Wq, device);
    &FloatTensor K = tensor_matmul_on(h, Wk, device);
    &FloatTensor V = tensor_matmul_on(h, Wv, device);
    // Attention itself stays on the CPU regardless of 'device' -- see
    // this file's header comment's scope boundary.
    &FloatTensor attnRaw = mha_forward(Q, K, V, numHeads);
    &FloatTensor attnOut = tensor_matmul_on(attnRaw, Wo, device);
    &FloatTensor x1 = tensor_residual_add(x, attnOut);

    &FloatTensor h2 = x1.layernorm_rows(1e-5);
    &FloatTensor ffnHiddenRaw = tensor_matmul_on(h2, W1, device);
    &FloatTensor ffnHiddenBias = __add_broadcast_row_dispatch(ffnHiddenRaw, b1w);
    &FloatTensor ffnHidden = ffnHiddenBias.relu_on(device);
    &FloatTensor ffnOutRaw = tensor_matmul_on(ffnHidden, W2, device);
    &FloatTensor ffnOut = __add_broadcast_row_dispatch(ffnOutRaw, b2w);
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

&FloatTensor jit_forward_on(const &JiT model, const &FloatTensor patches, const &FloatTensor cond, enum Device device) {
    unsigned long numBlocks = model.blocks.length();
    &FloatTensor embedRaw = tensor_matmul_on(patches, model.WPatchEmbed, device);
    &FloatTensor x = __add_broadcast_row_dispatch(embedRaw, cond);
    embedRaw->free();
    unsafe { free((void*)embedRaw); }

    unsigned long i = 0UL;
    while (i < numBlocks) {
        &JiTBlock blk = model.blocks.at(i);
        unsafe {
            &FloatTensor next = jit_block_forward_on(blk, x, device);
            x->free();
            free((void*)x);
            x = next;
        }
        i = i + 1UL;
    }
    return x;
}

} // namespace std
