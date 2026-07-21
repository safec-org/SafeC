#pragma once
// SafeC Standard Library — GPU tensor ops via Vulkan compute + SPIR-V
// (cross-vendor: NVIDIA/AMD/Intel/Apple-via-MoltenVK, wherever a Vulkan
// 1.0+ driver is present).
//
// UNVERIFIED, more so than gpu_cuda.h/gpu_rocm.h: this sandbox has no
// Vulkan SDK, no Vulkan-capable driver/ICD, and no SPIR-V toolchain
// (glslc/glslangValidator/spirv-as — none installed, confirmed). The
// Vulkan C API calls/struct declarations below were hand-written from
// memory against the Vulkan 1.0 core spec, not checked against a real
// vulkan.h or run through a real Vulkan validation layer — treat the
// struct field layouts especially as "should be right" rather than
// "confirmed right" until sanity-checked against a real Vulkan SDK.
//
// Same shape as gpu_rocm.h's HSACO gap, for the same underlying reason:
// SPIR-V is a *binary* IR (unlike CUDA's textual PTX), so there's no
// hand-writable string literal to embed the way gpu_mps.sc/gpu_cuda.sc do.
// A real deployment compiles a GLSL/HLSL/WGSL compute shader to SPIR-V
// offline (e.g. 'glslc kernel.comp -o kernel.spv') and embeds the
// resulting bytes; every op below has that byte array as an explicit
// null placeholder and returns 0 before reaching any real Vulkan call
// that would need it, so all of these are safe to call (as a no-op
// failure) even with no Vulkan runtime present at all — same
// fail-gracefully behavior as gpu_rocm.h's ops.
//
// Gated behind the 'spirv' feature (see Package.toml's [features]).
namespace std {

int spirv_add_f32(const float* a, const float* b, float* out, unsigned long n);
int spirv_sub_f32(const float* a, const float* b, float* out, unsigned long n);
int spirv_mul_f32(const float* a, const float* b, float* out, unsigned long n);
int spirv_div_f32(const float* a, const float* b, float* out, unsigned long n);
int spirv_pow_f32(const float* a, const float* b, float* out, unsigned long n);
int spirv_relu_f32(const float* a, float* out, unsigned long n);
int spirv_log_f32(const float* a, float* out, unsigned long n);
int spirv_exp_f32(const float* a, float* out, unsigned long n);
int spirv_sqrt_f32(const float* a, float* out, unsigned long n);
int spirv_scale_f32(const float* a, float k, float* out, unsigned long n);
int spirv_sum_f32(const float* a, float* out, unsigned long n);
int spirv_matmul_f32(const float* a, const float* b, float* out,
                      unsigned long M, unsigned long K, unsigned long N);

// True if vkCreateInstance succeeds and at least one physical device
// enumerates. Real, callable, and the one function in this file with a
// chance of doing something on a machine with a Vulkan driver installed
// (even without any SPIR-V kernel bytes, instance/device creation alone
// don't need one) — everything else always returns 0 here regardless,
// per the SPIR-V-bytecode gap above.
int spirv_available();

// ── Persistent-buffer tier ───────────────────────────────────────────────
// Real Vulkan device-memory management (vkCreateBuffer/vkAllocateMemory/
// vkMapMemory/vkUnmapMemory) plus a persistent VkInstance/VkDevice
// singleton -- unlike compute dispatch, buffer upload and device creation
// need no SPIR-V module at all, so neither hits this file's SPIR-V-
// bytecode gap (see this file's header comment). Mirrors gpu_mps.h/
// gpu_cuda.h/gpu_rocm.h's persistent-buffer tier: allocate + upload once,
// reuse the handle across many dispatches instead of paying
// vkCreateInstance/vkCreateDevice/vkCreateBuffer/vkAllocateMemory/
// vkMapMemory/copy/vkUnmapMemory on every call the way spirv_*_f32 above
// does internally (once real SPIR-V bytecode exists to dispatch with).
//
// What's deliberately NOT here: a persistent-buffer compute-dispatch
// variant (the Vulkan analogue of mps_matmul_f32_persistent/
// cuda_matmul_f32_persistent) still needs real SPIR-V bytecode this
// sandbox has no glslc/glslangValidator/spirv-as toolchain to produce --
// see this file's header comment. This tier is scoped to what's real
// without one: a persistent instance/device (spirv_persistent_available)
// and persistent buffers, ready for a future dispatch layer to bind
// directly once the SPIR-V-bytecode gap closes.

// Uploads 'bytes' from host memory to a new persistent, host-visible
// device buffer. Returns (void*)0 on failure (no device, or Vulkan
// buffer/memory allocation failure). The returned handle is opaque (an
// internal {VkBuffer, VkDeviceMemory} pair) — pass it only to
// spirv_release_persistent, not to spirv_*_f32 above (which take raw
// host float* pointers, not device buffer handles).
void* spirv_upload_persistent(const float* data, unsigned long bytes);

// Frees a buffer returned by spirv_upload_persistent.
void spirv_release_persistent(void* buf);

// True if the shared persistent Vulkan instance/device (used by
// spirv_upload_persistent/spirv_release_persistent) is available.
int spirv_persistent_available();

} // namespace std
