#pragma once
// SafeC Standard Library — vDSP-accelerated elementwise Tensor ops (see
// tensor_vdsp.sc). Same Autograd::get math/backward wiring as tensor.h's plain
// tensor_add/sub/mul/sum (reuses __add_backward/__sub_backward/
// __mul_backward/__sum_backward directly — only the forward computation
// changes), just vectorized via Accelerate's vDSP instead of a scalar loop.
// Link with '-framework Accelerate', same as tensor_blas.sc.
#include <std/ml/tensor.h>

namespace std {

&FloatTensor tensor_add_vdsp(const &FloatTensor a, const &FloatTensor b);
&FloatTensor tensor_sub_vdsp(const &FloatTensor a, const &FloatTensor b);
&FloatTensor tensor_mul_vdsp(const &FloatTensor a, const &FloatTensor b);
&FloatTensor FloatTensor::sum_vdsp() const;

} // namespace std
