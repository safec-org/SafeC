#pragma once
// SafeC Standard Library — GPU-backed Tensor ops (CUDA (NVIDIA)).
//
// Same shape/rationale as tensor_gpu.h (MPS) — see that file's header
// comment for the general design (why this is separate from tensor.sc,
// what backward-reuse means, the float32-conversion and dispatch-overhead
// costs every op here pays). UNVERIFIED end to end (see gpu_cuda.h): no NVIDIA GPU/CUDA toolkit in this sandbox.
#include <std/ml/tensor.h>

namespace std {

&Tensor tensor_add_cuda(const &Tensor a, const &Tensor b);
&Tensor tensor_sub_cuda(const &Tensor a, const &Tensor b);
&Tensor tensor_mul_cuda(const &Tensor a, const &Tensor b);
&Tensor tensor_scale_cuda(const &Tensor a, double k);
&Tensor tensor_relu_cuda(const &Tensor a);
&Tensor tensor_sum_cuda(const &Tensor a);
&Tensor tensor_matmul_cuda(const &Tensor a, const &Tensor b);

// Same op, dispatched to cuBLAS (gpu_cuda.h's cuda_matmul_f32_blas)
// instead of the hand-written PTX kernel tensor_matmul_cuda uses — the
// CUDA-vendor equivalent of tensor_blas.h's Accelerate-backed
// tensor_matmul_blas.
&Tensor tensor_matmul_cuda_blas(const &Tensor a, const &Tensor b);

} // namespace std
