#pragma once
// SafeC Standard Library — GPU-backed Tensor ops (WebGPU (wgpu-native/Dawn)).
//
// Same shape/rationale as tensor_gpu.h (MPS) — see that file's header
// comment for the general design (why this is separate from tensor.sc,
// what backward-reuse means, the float32-conversion and dispatch-overhead
// costs every op here pays). UNVERIFIED end to end (see gpu_webgpu.h): no WebGPU native library in this sandbox.
#include <std/ml/tensor.h>

namespace std {

&Tensor tensor_add_webgpu(const &Tensor a, const &Tensor b);
&Tensor tensor_sub_webgpu(const &Tensor a, const &Tensor b);
&Tensor tensor_mul_webgpu(const &Tensor a, const &Tensor b);
&Tensor tensor_scale_webgpu(const &Tensor a, double k);
&Tensor tensor_relu_webgpu(const &Tensor a);
&Tensor tensor_sum_webgpu(const &Tensor a);
&Tensor tensor_matmul_webgpu(const &Tensor a, const &Tensor b);

} // namespace std
