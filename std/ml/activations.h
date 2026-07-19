#pragma once
// SafeC Standard Library — Activation functions on Tensor.
//
// Forward-only (see attention.h's header comment — this library's scope
// is inference, not training end-to-end; tensor.h's core ops already
// carry real autograd for the pieces that need it).
#include <std/ml/tensor.h>

namespace std {

struct Tensor* tensor_sigmoid(struct Tensor* a);  // 1 / (1 + exp(-x)), elementwise
struct Tensor* tensor_tanh(struct Tensor* a);     // tanh(x), elementwise
struct Tensor* tensor_silu(struct Tensor* a);     // x * sigmoid(x) ("swish"), elementwise
struct Tensor* tensor_gelu(struct Tensor* a);     // tanh-approximation GELU, elementwise

// out[r][c] = (x[r][c] - rowMean) / sqrt(rowVar + eps), per row (mean 0,
// var 1 across each row). No affine gain/bias — apply your own
// elementwise scale/shift after calling this if you need one (e.g.
// AdaLN-style conditioning, see transformer.h).
struct Tensor* tensor_layernorm_rows(struct Tensor* x, double eps);

// x + sublayerOut. A plain alias for tensor_add, named for residual/
// skip-connection call sites (ResNet blocks, transformer sublayers).
struct Tensor* tensor_residual_add(struct Tensor* x, struct Tensor* sublayerOut);

} // namespace std
