#pragma once
// SafeC Standard Library — GPU-backed Tensor ops (Vulkan/SPIR-V (cross-vendor)).
//
// Same shape/rationale as tensor_gpu.h (MPS) — see that file's header
// comment for the general design (why this is separate from tensor.sc,
// what backward-reuse means, the float32-conversion and dispatch-overhead
// costs every op here pays). UNVERIFIED end to end (see gpu_spirv.h): no Vulkan SDK/driver/SPIR-V toolchain in this sandbox, and every spirv_* op currently always returns 0 (missing SPIR-V bytecode), so every op here always falls back to CPU.
#include <std/ml/tensor.h>

namespace std {

&Tensor tensor_add_spirv(const &Tensor a, const &Tensor b);
&Tensor tensor_sub_spirv(const &Tensor a, const &Tensor b);
&Tensor tensor_mul_spirv(const &Tensor a, const &Tensor b);
&Tensor tensor_scale_spirv(const &Tensor a, double k);
&Tensor tensor_relu_spirv(const &Tensor a);
&Tensor tensor_sum_spirv(const &Tensor a);
&Tensor tensor_matmul_spirv(const &Tensor a, const &Tensor b);

} // namespace std
