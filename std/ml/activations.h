#pragma once
// SafeC Standard Library — Activation functions on Tensor.
//
// Forward-only (see attention.h's header comment — this library's scope
// is inference, not training end-to-end; tensor.h's core ops already
// carry real autograd for the pieces that need it).
#include <std/ml/tensor.h>

namespace std {

&Tensor tensor_sigmoid(const &Tensor a);  // 1 / (1 + exp(-x)), elementwise
&Tensor tensor_tanh(const &Tensor a);     // tanh(x), elementwise
&Tensor tensor_silu(const &Tensor a);     // x * sigmoid(x) ("swish"), elementwise
&Tensor tensor_gelu(const &Tensor a);     // tanh-approximation GELU, elementwise

// out[r][c] = (x[r][c] - rowMean) / sqrt(rowVar + eps), per row (mean 0,
// var 1 across each row). No affine gain/bias — apply your own
// elementwise scale/shift after calling this if you need one (e.g.
// AdaLN-style conditioning, see transformer.h).
&Tensor tensor_layernorm_rows(const &Tensor x, double eps);

// x + sublayerOut. A plain alias for tensor_add, named for residual/
// skip-connection call sites (ResNet blocks, transformer sublayers).
&Tensor tensor_residual_add(const &Tensor x, const &Tensor sublayerOut);

} // namespace std
