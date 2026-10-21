#pragma once
// SafeC Standard Library — GPU-backed Tensor ops (ROCm/HIP (AMD)).
//
// Same shape/rationale as tensor_gpu.h (MPS) — see that file's header
// comment for the general design (why this is separate from tensor.sc,
// what backward-reuse means, the dispatch-overhead cost every op here
// pays). Tensor is float32 (see tensor.h), the same precision every
// rocm_*_f32 buffer already uses, so these ops read/write a Tensor's
// 'data'/'grad' directly — no per-call CPU-side conversion pass.
// Each op uploads its operands, computes on the GPU and reads the result
// back, so it pays two PCIe transfers per call: fine for occasional large
// ops, slow inside a training loop. For that, keep data on the GPU with
// gpu_rocm.h's RocmBuf API (see the ROCm MLP benchmark).
#include <std/ml/tensor.h>

namespace std {

&FloatTensor tensor_add_rocm(const &FloatTensor a, const &FloatTensor b);
&FloatTensor tensor_sub_rocm(const &FloatTensor a, const &FloatTensor b);
&FloatTensor tensor_mul_rocm(const &FloatTensor a, const &FloatTensor b);
&FloatTensor FloatTensor::scale_rocm(float k) const;
&FloatTensor FloatTensor::relu_rocm() const;
&FloatTensor FloatTensor::sum_rocm() const;
&FloatTensor tensor_matmul_rocm(const &FloatTensor a, const &FloatTensor b);

// tensor_matmul_rocm uses this library's own tiled HIP kernel;
// tensor_matmul_rocm_blas uses rocBLAS, which is several times faster.
&FloatTensor tensor_matmul_rocm_blas(const &FloatTensor a, const &FloatTensor b);

} // namespace std
