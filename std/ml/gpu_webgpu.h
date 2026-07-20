#pragma once
// SafeC Standard Library — GPU tensor ops via WebGPU (native, wgpu-native/
// Dawn's webgpu.h C API — the same API a browser's WebGPU implementation
// exposes to JS, just called directly from native code).
//
// UNVERIFIED: no WebGPU native library (wgpu-native or Dawn), no
// webgpu.h, and no GPU-capable WebGPU backend (Vulkan/Metal/D3D12
// instance) available to link against in this sandbox. Unlike
// gpu_spirv.h's SPIR-V binary gap, WGSL (WebGPU's shader language) *is*
// plain text, so — like gpu_mps.sc's runtime-compiled MSL and
// gpu_cuda.sc's embedded PTX — every kernel below embeds real WGSL
// source and compiles it at runtime via wgpuDeviceCreateShaderModule, no
// missing-binary gap the way gpu_rocm.sc/gpu_spirv.sc have. What's
// unverified here is the C API call sequence and struct layouts
// (hand-written from memory against webgpu.h, not checked against a real
// copy of it) and, fundamentally, that WebGPU's adapter/device
// acquisition is asynchronous even in the native C API — see
// gpu_webgpu.sc's header comment for how this handles that without a
// real event loop.
//
// Gated behind the 'webgpu' feature (see Package.toml's [features]).
namespace std {

int webgpu_add_f32(const float* a, const float* b, float* out, unsigned long n);
int webgpu_sub_f32(const float* a, const float* b, float* out, unsigned long n);
int webgpu_mul_f32(const float* a, const float* b, float* out, unsigned long n);
int webgpu_div_f32(const float* a, const float* b, float* out, unsigned long n);
int webgpu_pow_f32(const float* a, const float* b, float* out, unsigned long n);
int webgpu_relu_f32(const float* a, float* out, unsigned long n);
int webgpu_log_f32(const float* a, float* out, unsigned long n);
int webgpu_exp_f32(const float* a, float* out, unsigned long n);
int webgpu_sqrt_f32(const float* a, float* out, unsigned long n);
int webgpu_scale_f32(const float* a, float k, float* out, unsigned long n);
int webgpu_sum_f32(const float* a, float* out, unsigned long n);
int webgpu_matmul_f32(const float* a, const float* b, float* out,
                       unsigned long M, unsigned long K, unsigned long N);

// True if wgpuCreateInstance succeeds and an adapter request resolves
// (see gpu_webgpu.sc's polling-loop caveat) to a non-null adapter.
int webgpu_available();

} // namespace std
