#pragma once
// SafeC Standard Library — unified device dispatch for Tensor ops.
//
// Every backend up to this file names its accelerated ops explicitly
// (tensor_matmul vs tensor_matmul_blas vs tensor_matmul_gpu vs
// tensor_matmul_cuda vs tensor_matmul_cuda_blas vs tensor_matmul_rocm vs
// tensor_matmul_rocm_blas vs tensor_matmul_spirv vs tensor_matmul_webgpu)
// — the right design for a library that wants every backend individually
// inspectable and independently linkable, but awkward for CALLERS that
// want to pick a device once (e.g. per neural-net layer, or from a
// config/CLI flag) rather than hard-coding one function name per call
// site. tensor_matmul_on(a, b, device)/tensor_relu_on(a, device) below
// are that single entry point: an ordinary runtime switch over an enum,
// dispatching to whichever backend function the caller asked for.
//
// This file — unlike tensor.h — is NOT portable/dependency-free by
// design: it #includes every backend header that exists (tensor_blas.h,
// tensor_gpu.h, tensor_cuda.h, tensor_rocm.h, tensor_spirv.h,
// tensor_webgpu.h), so linking it pulls in whichever of those your build
// actually links against. Only include this file (and link the backends
// you actually use) where you genuinely want runtime device selection;
// a program that only ever wants one specific backend should keep
// calling that backend's own tensor_<op>_<backend> function directly,
// the way the rest of std/ml does.
#include <std/ml/tensor.h>
#include <std/ml/tensor_nn.h>
#include <std/ml/tensor_blas.h>
#include <std/ml/tensor_gpu.h>
#include <std/ml/tensor_cuda.h>
#include <std/ml/tensor_rocm.h>
#include <std/ml/tensor_spirv.h>
#include <std/ml/tensor_webgpu.h>

namespace std {

// NOTE: reference these unqualified (DEVICE_CPU), not std::DEVICE_CPU —
// SafeC enum constants aren't namespace-scoped the way the enum TYPE
// itself is (confirmed: 'enum Color { RED };' declared inside
// 'namespace std' makes 'RED' usable bare from anywhere, but
// 'std::RED' fails to resolve) — a real language quirk, not a mistake
// in the examples below.
//
// Not every value is available on every machine/build — e.g. DEVICE_CUDA
// needs an NVIDIA GPU; DEVICE_MPS only ever exists on macOS. Passing an
// unavailable device isn't an error: each underlying tensor_<op>_<backend>
// function already falls back to the portable CPU path on its own
// (mps_available()/cuda_available()/etc-gated), and the _on() dispatchers
// below inherit that same fallback behavior for free, simply by calling
// into those functions rather than reimplementing their availability
// checks.
enum Device {
    DEVICE_CPU = 0,        // portable, dependency-free triple loop (tensor.sc)
    DEVICE_CPU_BLAS = 1,   // vendor BLAS on the CPU (Accelerate today; tensor_blas.sc)
    DEVICE_MPS = 2,        // Apple Metal (tensor_gpu.sc)
    DEVICE_CUDA = 3,       // NVIDIA, hand-written PTX kernel (tensor_cuda.sc)
    DEVICE_CUDA_BLAS = 4,  // NVIDIA, cuBLAS (tensor_cuda.sc)
    DEVICE_ROCM = 5,       // AMD, HIP kernel (tensor_rocm.sc) -- see that file's HSACO-gap caveat
    DEVICE_ROCM_BLAS = 6,  // AMD, rocBLAS (tensor_rocm.sc) -- not affected by the HSACO gap
    DEVICE_SPIRV = 7,      // Vulkan/SPIR-V (tensor_spirv.sc)
    DEVICE_WEBGPU = 8,     // WebGPU (tensor_webgpu.sc)
};

&Tensor tensor_matmul_on(const &Tensor a, const &Tensor b, enum Device device);
&Tensor tensor_relu_on(const &Tensor a, enum Device device);

} // namespace std
