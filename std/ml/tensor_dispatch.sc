// SafeC Standard Library — unified device dispatch implementation (see
// tensor_dispatch.h).
#pragma once
#include <std/ml/tensor_dispatch.h>

namespace std {

&Tensor tensor_matmul_on(const &Tensor a, const &Tensor b, enum Device device) {
    if (device == DEVICE_CPU_BLAS) { return tensor_matmul_blas(a, b); }
    if (device == DEVICE_MPS) { return tensor_matmul_gpu(a, b); }
    if (device == DEVICE_CUDA) { return tensor_matmul_cuda(a, b); }
    if (device == DEVICE_CUDA_BLAS) { return tensor_matmul_cuda_blas(a, b); }
    if (device == DEVICE_ROCM) { return tensor_matmul_rocm(a, b); }
    if (device == DEVICE_ROCM_BLAS) { return tensor_matmul_rocm_blas(a, b); }
    if (device == DEVICE_SPIRV) { return tensor_matmul_spirv(a, b); }
    if (device == DEVICE_WEBGPU) { return tensor_matmul_webgpu(a, b); }
    return tensor_matmul(a, b); // DEVICE_CPU, and the default for any future enum value
}

&Tensor tensor_relu_on(const &Tensor a, enum Device device) {
    if (device == DEVICE_MPS) { return tensor_relu_gpu(a); }
    if (device == DEVICE_CUDA || device == DEVICE_CUDA_BLAS) { return tensor_relu_cuda(a); }
    if (device == DEVICE_ROCM || device == DEVICE_ROCM_BLAS) { return tensor_relu_rocm(a); }
    // DEVICE_SPIRV/DEVICE_WEBGPU/DEVICE_CPU_BLAS: relu has no SPIR-V/
    // WebGPU/vendor-BLAS-specific implementation (BLAS covers matmul
    // only -- relu isn't a BLAS operation at all), so these fall back to
    // the portable CPU path, same as DEVICE_CPU itself.
    return tensor_relu(a);
}

} // namespace std
