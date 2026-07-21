#pragma once
// SafeC Standard Library — Activation functions on Tensor.
//
// Forward-only (see attention.h's header comment — this library's scope
// is inference, not training end-to-end; tensor.h's core ops already
// carry real autograd for the pieces that need it).
//
// Every function here has a '_fwd' suffix specifically to avoid colliding
// with tensor_nn.h's same-shaped-but-autograd-carrying versions
// (tensor_relu, tensor_sigmoid, etc.) — this file and tensor_nn.h/.sc
// used to share those exact names, which meant a program could never
// link both (e.g. transformer.sc, which needs this file's
// tensor_layernorm_rows/tensor_residual_add, previously couldn't also
// link any tensor_<backend>.sc GPU file, since every one of those
// #includes tensor_nn.h/.sc internally for its own backward-linking
// needs). Renamed once this stopped being hypothetical: tensor_dispatch.h
// combining GPU-backed ops with transformer.h's layers hit exactly this
// collision.
#include <std/ml/tensor.h>

namespace std {

&Tensor tensor_relu_fwd(const &Tensor a);     // max(x, 0), elementwise -- forward-only, like every other op in this file
&Tensor tensor_sigmoid_fwd(const &Tensor a);  // 1 / (1 + exp(-x)), elementwise
&Tensor tensor_tanh_fwd(const &Tensor a);     // tanh(x), elementwise
&Tensor tensor_silu_fwd(const &Tensor a);     // x * sigmoid(x) ("swish"), elementwise
&Tensor tensor_gelu_fwd(const &Tensor a);     // tanh-approximation GELU, elementwise

// out[r][c] = (x[r][c] - rowMean) / sqrt(rowVar + eps), per row (mean 0,
// var 1 across each row). No affine gain/bias — apply your own
// elementwise scale/shift after calling this if you need one (e.g.
// AdaLN-style conditioning, see transformer.h).
&Tensor tensor_layernorm_rows(const &Tensor x, float eps);

// x + sublayerOut. A plain alias for tensor_add, named for residual/
// skip-connection call sites (ResNet blocks, transformer sublayers).
&Tensor tensor_residual_add(const &Tensor x, const &Tensor sublayerOut);

} // namespace std
