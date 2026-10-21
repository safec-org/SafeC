#pragma once
// SafeC Standard Library — trainable normalization: LayerNorm and AdaLN.
//
// Both normalize each row of a [batch, dim] tensor to mean 0 and variance
// 1 and carry reverse-mode autograd (activations.h's
// tensor_layernorm_rows is the forward-only version).
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
#include <std/ml/attention.h>   // tensor_slice_cols

namespace std {

// LN(x) without the affine part, row by row, with autograd.
&Tensor tensor_layer_norm(const &Tensor x, float eps);
// x [m,n] * g [n] on every row, with autograd.
&Tensor tensor_mul_row(const &Tensor x, const &Tensor g);

struct LayerNorm {
    &Tensor gamma;   // [dim], init 1
    &Tensor beta;    // [dim], init 0
    unsigned long dim;
    float eps;       // 1e-5
};
struct LayerNorm layer_norm_new(unsigned long dim);
&Tensor          layer_norm_forward(const &LayerNorm ln, const &Tensor x);
void             layer_norm_add_params(&Optimizer opt, const &LayerNorm ln);
void             layer_norm_free(&LayerNorm ln);

struct AdaLN {
    &Tensor W;       // [condDim, chunks * dim], init 0
    &Tensor b;       // [chunks * dim], init 0
    unsigned long dim;
    unsigned long condDim;
    int   gated;     // 1: a third chunk, the residual gate
    float eps;
};
struct AdaLNOut {
    &Tensor out;     // LN(x) * (1 + scale) + shift, [m, dim]
    &Tensor gate;    // [m, dim]; all ones when not gated
};
struct AdaLN    adaln_new(unsigned long dim, unsigned long condDim, int gated);
// x [m, dim], cond [m, condDim] (one conditioning row per sample).
struct AdaLNOut adaln_forward(const &AdaLN a, const &Tensor x, const &Tensor cond);
void            adaln_add_params(&Optimizer opt, const &AdaLN a);
void            adaln_free(&AdaLN a);

} // namespace std
