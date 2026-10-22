#pragma once
// SafeC Standard Library — GPU-backed Tensor ops (Vulkan/SPIR-V (cross-vendor)).
//
// Per-op GPU tensor functions for a backend without a device.h backend yet
// comment for the general design (why this is separate from tensor.sc,
// what backward-reuse means, the dispatch-overhead cost every op here
// pays). Tensor is float32 (see tensor.h), the same precision every
// spirv_*_f32 buffer already uses, so these ops read/write a Tensor's
// 'data'/'grad' directly — no per-call CPU-side conversion pass.
// UNVERIFIED end to end (see gpu_spirv.h): no Vulkan SDK/driver/SPIR-V toolchain in this sandbox, and every spirv_* op currently always returns 0 (missing SPIR-V bytecode), so every op here always falls back to CPU.
#include <std/ml/tensor.h>

namespace std {

&FloatTensor tensor_add_spirv(const &FloatTensor a, const &FloatTensor b);
&FloatTensor tensor_sub_spirv(const &FloatTensor a, const &FloatTensor b);
&FloatTensor tensor_mul_spirv(const &FloatTensor a, const &FloatTensor b);
&FloatTensor FloatTensor::scale_spirv(float k) const;
&FloatTensor FloatTensor::relu_spirv() const;
&FloatTensor FloatTensor::sum_spirv() const;
&FloatTensor tensor_matmul_spirv(const &FloatTensor a, const &FloatTensor b);

} // namespace std
