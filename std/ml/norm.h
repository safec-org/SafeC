#pragma once
// SafeC Standard Library — trainable normalization: LayerNorm and AdaLN.
//
// Both normalize each row of a [batch, dim] tensor to mean 0 and variance
// 1 and carry reverse-mode autograd (activations.h's
// Tensor::layernorm_rows is the forward-only version).
//
//   LayerNorm (Ba et al. 2016):  y = LN(x) * gamma + beta, learned
//     per-feature gamma (init 1) and beta (init 0).
//   AdaLN (adaptive LayerNorm, as in DiT, Peebles & Xie 2023):
//     (shift, scale[, gate]) = cond @ W + b, per sample
//     y = LN(x) * (1 + scale) + shift
//     A conditioning vector (timestep / class embedding) steers every
//     sample's normalization. W and b start at zero ("adaLN-Zero"), so a
//     fresh AdaLN is a plain LayerNorm, and with 'gated' the gate starts
//     at 0, so a residual branch x + gate * f(y) starts as the identity.
#include <std/ml/tensor.h>
#include <std/ml/nn.h>
#include <std/ml/attention.h>   // Tensor::slice_cols

namespace std {

// LN(x) without the affine part, row by row, with Autograd::get.
&FloatTensor FloatTensor::layer_norm(float eps) const;
// x [m,n] * g [n] on every row, with Autograd::get.
&FloatTensor tensor_mul_row(const &FloatTensor x, const &FloatTensor g);

struct LayerNorm {
    &FloatTensor gamma;   // [dim], init 1
    &FloatTensor beta;    // [dim], init 0
    unsigned long dim;
    float eps;       // 1e-5

    &FloatTensor forward(const &FloatTensor x) const;
    // Parameterized (module.h): the trained tensors, and what a copy or a file carries.
    void collect_params(&Vec<&FloatTensor> out) const;
    void collect_state(&Vec<&FloatTensor> out) const;
    void free();

    static struct LayerNorm new(unsigned long dim);
};

struct AdaLN {
    &FloatTensor W;       // [condDim, chunks * dim], init 0
    &FloatTensor b;       // [chunks * dim], init 0
    unsigned long dim;
    unsigned long condDim;
    int   gated;     // 1: a third chunk, the residual gate
    float eps;

    // x [m, dim], cond [m, condDim] (one conditioning row per sample).
    struct AdaLNOut forward(const &FloatTensor x, const &FloatTensor cond) const;
    // Parameterized (module.h): the trained tensors, and what a copy or a file carries.
    void collect_params(&Vec<&FloatTensor> out) const;
    void collect_state(&Vec<&FloatTensor> out) const;
    void free();

    static struct AdaLN new(unsigned long dim, unsigned long condDim, int gated);
};
struct AdaLNOut {
    &FloatTensor out;     // LN(x) * (1 + scale) + shift, [m, dim]
    &FloatTensor gate;    // [m, dim]; all ones when not gated
};

} // namespace std
