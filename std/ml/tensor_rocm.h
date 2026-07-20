#pragma once
// SafeC Standard Library — GPU-backed Tensor ops (ROCm/HIP (AMD)).
//
// Same shape/rationale as tensor_gpu.h (MPS) — see that file's header
// comment for the general design (why this is separate from tensor.sc,
// what backward-reuse means, the float32-conversion and dispatch-overhead
// costs every op here pays). UNVERIFIED end to end (see gpu_rocm.h): no AMD GPU/ROCm toolkit in this sandbox, and every rocm_* op currently always returns 0 (missing HSACO binary), so every op here always falls back to CPU.
#include <std/ml/tensor.h>

namespace std {

&Tensor tensor_add_rocm(const &Tensor a, const &Tensor b);
&Tensor tensor_sub_rocm(const &Tensor a, const &Tensor b);
&Tensor tensor_mul_rocm(const &Tensor a, const &Tensor b);
&Tensor tensor_scale_rocm(const &Tensor a, double k);
&Tensor tensor_relu_rocm(const &Tensor a);
&Tensor tensor_sum_rocm(const &Tensor a);
&Tensor tensor_matmul_rocm(const &Tensor a, const &Tensor b);

} // namespace std
