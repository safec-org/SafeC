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

&Tensor tensor_add_rocm(const &Tensor a, const &Tensor b);
&Tensor tensor_sub_rocm(const &Tensor a, const &Tensor b);
&Tensor tensor_mul_rocm(const &Tensor a, const &Tensor b);
&Tensor tensor_scale_rocm(const &Tensor a, float k);
&Tensor tensor_relu_rocm(const &Tensor a);
&Tensor tensor_sum_rocm(const &Tensor a);
&Tensor tensor_matmul_rocm(const &Tensor a, const &Tensor b);

// tensor_matmul_rocm uses this library's own tiled HIP kernel;
// tensor_matmul_rocm_blas uses rocBLAS, which is several times faster.
&Tensor tensor_matmul_rocm_blas(const &Tensor a, const &Tensor b);

} // namespace std
