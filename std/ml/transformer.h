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
#include <std/collections/slice.h>

#include <std/collections/vec.h>

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
    &FloatTensor Wq; &FloatTensor Wk; &FloatTensor Wv; &FloatTensor Wo; // [dModel,dModel]
    &FloatTensor W1; &FloatTensor b1; // [dModel,dHidden], [1,dHidden]
    &FloatTensor W2; &FloatTensor b2; // [dHidden,dModel], [1,dModel]
    &FloatTensor WadaLN; // [dModelCond, 6*dModel]
    unsigned long numHeads;

    &FloatTensor forward(const &FloatTensor x, const &FloatTensor c) const;
    // Parameterized (module.h): every weight tensor.
    void collect_params(&Vec<&FloatTensor> out) const;
    void collect_state(&Vec<&FloatTensor> out) const;
    void free();

    static struct DiTBlock new(unsigned long dModel, unsigned long dHidden, unsigned long dModelCond,
                               unsigned long numHeads);
};

// ── JiT (plain pre-LN transformer, additive conditioning) ───────────────────
// Conditioning is added directly to the token embeddings once, up front
// (JiT::forward), rather than modulating every block's normalization the
// way DiT's adaLN does — the architectural distinction this
// implementation draws between the two. Each block is an ordinary
// pre-LN transformer sublayer pair:
//   h = LN(x); x = x + MHA(h@Wq,h@Wk,h@Wv,heads)@Wo
//   h2 = LN(x); x = x + relu(h2@W1+b1)@W2+b2
struct JiTBlock {
    &FloatTensor Wq; &FloatTensor Wk; &FloatTensor Wv; &FloatTensor Wo;
    &FloatTensor W1; &FloatTensor b1;
    &FloatTensor W2; &FloatTensor b2;
    unsigned long numHeads;

    &FloatTensor forward(const &FloatTensor x) const;
    // Parameterized (module.h): every weight tensor.
    void collect_params(&Vec<&FloatTensor> out) const;
    void collect_state(&Vec<&FloatTensor> out) const;
    void free();

    static struct JiTBlock new(unsigned long dModel, unsigned long dHidden, unsigned long numHeads);
};

// A JiT model: the patch embedding and a stack of JiTBlocks.
// patches: [seqLen, patchDim]; cond: [1, dModel], broadcast-added to every
// token's embedding once, up front.
struct JiT {
    &FloatTensor WPatchEmbed;               // [patchDim, dModel]
    struct Vec<struct JiTBlock> blocks;

    &FloatTensor forward(const &FloatTensor patches, const &FloatTensor cond) const;   // -> [seqLen, dModel]
    // Parameterized (module.h): the embedding, then each block's weights.
    void collect_params(&Vec<&FloatTensor> out) const;
    void collect_state(&Vec<&FloatTensor> out) const;
    void free();

    static struct JiT new(unsigned long patchDim, unsigned long dModel, unsigned long dHidden, unsigned long numHeads,
                          unsigned long depth);
};

} // namespace std
