#pragma once
// SafeC Standard Library — state space models (Mamba's selective SSM).
//
// A state space model mixes a sequence through a per-channel linear
// recurrence instead of attention: each of D channels carries an N-value
// hidden state h that every token updates and reads,
//   h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * x_t
//   y_t = C_t . h_t + Dskip * x_t
// so a sequence costs O(T * D * N) time and decoding a token costs O(D * N)
// with a fixed-size state, however long the context — no KV cache.
// Mamba (Gu & Dao 2023, "S6") makes the step size delta and the B / C
// projections functions of the input ("selective"), which lets the model
// choose per token what to remember and what to forget.
//
// Forward-only (inference), like the other layers in attention_advanced.h.
// The CPU scan runs the D channels in parallel on std/sync/parallel.h's pool.
#include <std/ml/tensor.h>

#include <std/collections/vec.h>

namespace std {

// The selective scan on its own. x, delta: [T, D] (delta already positive,
// e.g. after softplus); A: [D, N] (negative for a decaying state); B, C:
// [T, N]; Dskip: [D] (the skip connection, D in the paper). Returns y: [T, D].
// The state starts at zero.
&Tensor ssm_selective_scan(const &Tensor x, const &Tensor delta, const &Tensor A,
                           const &Tensor B, const &Tensor C, const &Tensor Dskip);

// One Mamba block (the mixer of Mamba / Mamba-1 language models, without
// the surrounding RMSNorm and residual add):
//   (x, z)  = split(X @ inProj)                     dInner each
//   x       = silu(causal_depthwise_conv1d(x))      kernel dConv
//   (dt, B, C) = split(x @ xProj)                   dtRank, dState, dState
//   delta   = softplus(dt @ dtProj + dtBias)
//   y       = selective_scan(x, delta, -exp(Alog), B, C, Dskip) * silu(z)
//   out     = y @ outProj
// Weights are stored [in, out] like the rest of std::ml (transpose a
// PyTorch nn.Linear's weight to load it); convW is [dInner, dConv], the
// PyTorch Conv1d weight [dInner, 1, dConv] with the middle axis dropped.
struct Mamba {
    &Tensor inProj;   // [dModel, 2 * dInner]
    &Tensor convW;    // [dInner, dConv]
    &Tensor convB;    // [dInner]
    &Tensor xProj;    // [dInner, dtRank + 2 * dState]
    &Tensor dtProj;   // [dtRank, dInner]
    &Tensor dtBias;   // [dInner]
    &Tensor Alog;     // [dInner, dState]; A = -exp(Alog)
    &Tensor Dskip;    // [dInner]
    &Tensor outProj;  // [dInner, dModel]
    unsigned long dModel;
    unsigned long dInner;   // expand * dModel
    unsigned long dState;   // N
    unsigned long dConv;
    unsigned long dtRank;   // ceil(dModel / 16)

    // X: [T, dModel] -> [T, dModel], the whole sequence at once (prefill).
    &Tensor forward(const &Tensor X) const;
    // Parameterized (module.h): every weight tensor.
    void collect_params(&Vec<&Tensor> out) const;
    void collect_state(&Vec<&Tensor> out) const;
    void free();
    // x: [1, dModel] (or [dModel]) -> [1, dModel]; advances 'state'.
    &Tensor step(&MambaState state, const &Tensor x) const;
};

// Deterministic initialization following the paper (and HF transformers'
// MambaMixer): Alog[d][n] = log(n + 1), Dskip = 1, dtBias chosen so
// softplus(dtBias) spans [0.001, 0.1] log-uniformly, other weights small
// and pseudo-random. Typical: dState 16, dConv 4, expand 2.
struct Mamba mamba_new(unsigned long dModel, unsigned long dState, unsigned long dConv, unsigned long expand);

// Recurrent decoding, one token at a time with a fixed-size state.
// Feeding tokens 0..T-1 through Mamba::step gives the same rows as
// Mamba::forward on the whole sequence.
struct MambaState {
    &Tensor conv;  // [dInner, dConv - 1], the last inputs to the convolution, oldest first
    &Tensor h;     // [dInner, dState], the SSM state

    void reset();
    void free();
};
struct MambaState mamba_state_new(const &Mamba layer);  // zero state

} // namespace std
