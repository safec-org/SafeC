#pragma once
// SafeC Standard Library — Advanced attention: Multi-head Latent
// Attention (MLA/EG-MLA), Shifted-Window (Swin) attention, and Gated
// DeltaNet.
//
// Built on std/ml/attention.h's mha_forward/attention_forward and
// std/ml/tensor.h's Tensor. MLA and EG-MLA record gradients and run on
// devices like attention.h's ops (their weights require grad: they train
// through trainer.h); the shifted-window attention and Gated DeltaNet are
// forward-only host loops. Scope notes on each section below.
#include <std/ml/tensor.h>
#include <std/ml/attention.h>
#include <std/collections/slice.h>

#include <std/collections/vec.h>

namespace std {

// ── Multi-head Latent Attention (MLA, DeepSeek-V2) ──────────────────────────
// Compresses K/V into a shared low-rank latent c_kv = X @ Wdkv before
// up-projecting back to full K/V width — the KV-cache-memory-saving idea
// MLA is built around. (Decoupled rotary keys, the other half of the
// real DeepSeek-V2 design, are out of scope — no positional encoding is
// baked into this library's attention at all, see attention.h.)
//   Q    = X @ Wq                        [seqLen, dModel]
//   c_kv = X @ Wdkv                      [seqLen, dLatent]
//   K    = c_kv @ Wuk                    [seqLen, dModel]
//   V    = c_kv @ Wuv                    [seqLen, dModel]
//   out  = mha_forward(Q, K, V, numHeads)
// (Mla::forward).
struct Mla {
    &FloatTensor Wq;     // [dModel, dModel]
    &FloatTensor Wdkv;   // [dModel, dLatent]
    &FloatTensor Wuk;    // [dLatent, dModel]
    &FloatTensor Wuv;    // [dLatent, dModel]
    unsigned long numHeads;

    &FloatTensor forward(const &FloatTensor X) const;   // [seqLen, dModel] -> [seqLen, dModel]
    // Parameterized (module.h): Wq, Wdkv, Wuk, Wuv.
    void collect_params(&Vec<&FloatTensor> out) const;
    void collect_state(&Vec<&FloatTensor> out) const;
    void drop();
    void free() consume;   // = drop()

    // Weights U(-1/sqrt(in), 1/sqrt(in)) from rng; numHeads divides dModel.
    static struct Mla new(unsigned long dModel, unsigned long dLatent, unsigned long numHeads, &Rng rng);
};

// EG-MLA (Extended-Group MLA): shares one down-projected latent c_kv
// across 'numGroups' independent up-projection groups (grouped-query
// attention's KV-sharing idea layered onto MLA's low-rank compression).
// Group g owns WukGroups[g]/WuvGroups[g] ([dLatent, groupDim], groupDim
// = dModel/numGroups) and runs 'headsPerGroup' ordinary attention heads
// over its own up-projected K/V slice; results are concatenated back
// into one [seqLen, dModel] tensor. WukGroups/WuvGroups hold one tensor
// per group; their length is the group count.
struct EgMla {
    &FloatTensor Wq;                   // [dModel, dModel]
    &FloatTensor Wdkv;                 // [dModel, dLatent]
    struct Vec<&FloatTensor> Wuk;      // per group [dLatent, groupDim]
    struct Vec<&FloatTensor> Wuv;
    unsigned long headsPerGroup;

    &FloatTensor forward(const &FloatTensor X) const;   // [seqLen, dModel] -> [seqLen, dModel]
    // Parameterized (module.h): Wq, Wdkv, then each group's Wuk, Wuv.
    void collect_params(&Vec<&FloatTensor> out) const;
    void collect_state(&Vec<&FloatTensor> out) const;
    void drop();
    void free() consume;   // = drop()

    // numGroups groups of headsPerGroup heads; numGroups divides dModel.
    static struct EgMla new(unsigned long dModel, unsigned long dLatent, unsigned long numGroups,
                            unsigned long headsPerGroup, &Rng rng);
};

// ── Shifted-Window attention (Swin, adapted to 1D sequences) ────────────────
// Swin's two core ideas — partition into fixed-size local windows and
// attend only within a window, then cyclically shift the partition so
// the *next* layer's windows straddle this layer's boundaries — carried
// over to this library's 1D [seqLen, dim] sequence shape rather than
// Swin's original 2D image-patch grid (this library has no 2D spatial
// tensor type; see cnn.h's FeatureMap for the closest thing, which this
// doesn't use). 'seqLen' must be divisible by 'windowSize'. 'shift' (0
// for the unshifted variant, typically windowSize/2 for the shifted one)
// cyclically rotates rows before windowing and rotates the result back
// after, so windows at the boundary contain rows from both original
// neighbors on the next layer.
&FloatTensor windowed_attention_forward(const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V,
                                    unsigned long windowSize, unsigned long shift);

// ── Gated DeltaNet ────────────────────────────────────────────────────────────
// Linear-recurrent attention replacement (Yang et al. 2024): a
// [keyDim, valueDim] associative-memory state S is updated one timestep
// at a time with a *delta rule* (write the residual between the new
// value and what the current state already predicts for this key,
// rather than just accumulating) plus a scalar gate controlling how much
// of the old state decays away each step:
//   alpha_t = sigmoid(x_t . w_alpha)                 decay gate, scalar
//   beta_t  = sigmoid(x_t . w_beta)                   write gate, scalar
//   pred_t  = S_{t-1}^T @ k_t                         [valueDim]
//   delta_t = v_t - pred_t                            [valueDim]
//   S_t     = alpha_t * S_{t-1} + beta_t * outer(k_t, delta_t)   [keyDim, valueDim]
//   out_t   = S_t^T @ q_t                             [valueDim]
// Scope note: alpha_t/beta_t are scalar-per-timestep here (a single gate
// value applied uniformly across the state), not per-channel as in the
// full paper — documented simplification, same spirit as this file's
// other scope cuts.
struct GatedDeltaNet {
    &FloatTensor Walpha; // [dModel, 1]
    &FloatTensor Wbeta;  // [dModel, 1]
    unsigned long  keyDim;
    unsigned long  valueDim;

    // X: [seqLen, dModel] (gate input only), Q/K: [seqLen, keyDim], V: [seqLen, valueDim]
    // -> result: [seqLen, valueDim], computed via a sequential scan over time.
    &FloatTensor forward(const &FloatTensor X, const &FloatTensor Q, const &FloatTensor K, const &FloatTensor V) const;
    // Parameterized (module.h): every weight tensor.
    void collect_params(&Vec<&FloatTensor> out) const;
    void collect_state(&Vec<&FloatTensor> out) const;
    void drop();
    void free() consume;   // = drop()

    static struct GatedDeltaNet new(unsigned long dModel, unsigned long keyDim, unsigned long valueDim);
};

} // namespace std
