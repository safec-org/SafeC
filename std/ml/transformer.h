#pragma once
// SafeC Standard Library — Transformer image-generation backbones: DiT
// (Diffusion Transformer, Peebles & Xie 2023) and JiT ("Just-image-
// Transformer" — a plain, unconditioned-per-layer transformer backbone,
// as opposed to DiT's per-layer adaptive conditioning).
//
// Both operate on [seqLen, dModel] token sequences (already-patchified/
// embedded — no patchify/positional-encoding logic here, same scope
// choice attention.h makes) built from std::mha_forward,
// std::tensor_layernorm_rows, and plain feed-forward blocks. Forward-
// only, like the rest of std/ml.
#include <std/ml/tensor.h>
#include <std/ml/attention.h>
#include <std/ml/activations.h>

namespace std {

// ── DiT (adaLN conditioning) ─────────────────────────────────────────────────
// Each block computes 6 modulation vectors (shift1,scale1,gate1,
// shift2,scale2,gate2) from a conditioning vector c via one linear
// layer, then runs a pre-LN self-attention sublayer and a pre-LN FFN
// sublayer, each modulated by its half of the 6 and gated additively
// into the residual stream (the "adaLN-Zero" block from the DiT paper,
// minus the SiLU the paper puts before WadaLN and minus the zero-init
// convention — both orthogonal to the block's wiring, which is what
// this implements and verifies):
//   mod = c @ WadaLN                                    [1, 6*dModel]
//   h = LN(x); h = h*(1+scale1)+shift1
//   x = x + gate1 * (MHA(h@Wq,h@Wk,h@Wv,heads) @ Wo)
//   h2 = LN(x); h2 = h2*(1+scale2)+shift2
//   x = x + gate2 * (relu(h2@W1+b1)@W2+b2)
struct DiTBlock {
    struct Tensor* Wq; struct Tensor* Wk; struct Tensor* Wv; struct Tensor* Wo; // [dModel,dModel]
    struct Tensor* W1; struct Tensor* b1; // [dModel,dHidden], [1,dHidden]
    struct Tensor* W2; struct Tensor* b2; // [dHidden,dModel], [1,dModel]
    struct Tensor* WadaLN; // [dModelCond, 6*dModel]
    unsigned long numHeads;
};

struct DiTBlock dit_block_new(unsigned long dModel, unsigned long dHidden,
                               unsigned long dModelCond, unsigned long numHeads);
struct Tensor* dit_block_forward(struct DiTBlock* block, struct Tensor* x, struct Tensor* c);
void dit_block_free(struct DiTBlock* block);

// ── JiT (plain pre-LN transformer, additive conditioning) ───────────────────
// Conditioning is added directly to the token embeddings once, up front
// (jit_forward), rather than modulating every block's normalization the
// way DiT's adaLN does — the architectural distinction this
// implementation draws between the two. Each block is an ordinary
// pre-LN transformer sublayer pair:
//   h = LN(x); x = x + MHA(h@Wq,h@Wk,h@Wv,heads)@Wo
//   h2 = LN(x); x = x + relu(h2@W1+b1)@W2+b2
struct JiTBlock {
    struct Tensor* Wq; struct Tensor* Wk; struct Tensor* Wv; struct Tensor* Wo;
    struct Tensor* W1; struct Tensor* b1;
    struct Tensor* W2; struct Tensor* b2;
    unsigned long numHeads;
};

struct JiTBlock jit_block_new(unsigned long dModel, unsigned long dHidden, unsigned long numHeads);
struct Tensor* jit_block_forward(struct JiTBlock* block, struct Tensor* x);
void jit_block_free(struct JiTBlock* block);

// patches: [seqLen, patchDim], WPatchEmbed: [patchDim, dModel], cond:
// [1, dModel] (broadcast-added to every token's embedding once, up
// front). Runs 'blocks' (an array of numBlocks JiTBlock, caller-owned)
// in sequence and returns the final [seqLen, dModel] token sequence.
struct Tensor* jit_forward(struct Tensor* patches, struct Tensor* WPatchEmbed, struct Tensor* cond,
                            struct JiTBlock* blocks, unsigned long numBlocks);

} // namespace std
