#pragma once
// SafeC Standard Library — GPU-backed Tensor ops (CUDA (NVIDIA)).
//
// Same shape/rationale as tensor_gpu.h (MPS) — see that file's header
// comment for the general design (why this is separate from tensor.sc,
// what backward-reuse means, the dispatch-overhead cost every op here
// pays). Tensor is float32 (see tensor.h), the same precision every
// cuda_*_f32 buffer already uses, so these ops read/write a Tensor's
// 'data'/'grad' directly — no per-call CPU-side conversion pass.
// UNVERIFIED end to end (see gpu_cuda.h): no NVIDIA GPU/CUDA toolkit in this sandbox.
#include <std/ml/tensor.h>

namespace std {

&FloatTensor tensor_add_cuda(const &FloatTensor a, const &FloatTensor b);
&FloatTensor tensor_sub_cuda(const &FloatTensor a, const &FloatTensor b);
&FloatTensor tensor_mul_cuda(const &FloatTensor a, const &FloatTensor b);
&FloatTensor FloatTensor::scale_cuda(float k) const;
&FloatTensor FloatTensor::relu_cuda() const;
&FloatTensor FloatTensor::sum_cuda() const;
&FloatTensor tensor_matmul_cuda(const &FloatTensor a, const &FloatTensor b);

// Same op, dispatched to cuBLAS (gpu_cuda.h's cuda_matmul_f32_blas)
// instead of the hand-written PTX kernel tensor_matmul_cuda uses — the
// CUDA-vendor equivalent of tensor_blas.h's Accelerate-backed
// tensor_matmul_blas.
&FloatTensor tensor_matmul_cuda_blas(const &FloatTensor a, const &FloatTensor b);

} // namespace std
