#pragma once
// SafeC Standard Library — GPU-backed Tensor ops (Vulkan/SPIR-V (cross-vendor)).
//
// Per-op GPU tensor functions over gpu_spirv.h's host-array ops: each
// uploads its operands, runs on the Vulkan GPU and reads the result back
// into a CPU tensor, which joins the graph as the CPU op's result would
// (backward runs on the CPU), and computes on the CPU without a Vulkan GPU.
// Tensor is float32 (see tensor.h), the precision every spirv_*_f32 op
// takes, so these read and write a Tensor's 'data' directly. Each op pays
// an upload and a wait: for a model, place its tensors on DEVICE_VULKAN
// (device.h) instead, which keeps them on the GPU across ops and runs the
// backward pass there too.
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
