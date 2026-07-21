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

// ── Persistent-device / GPU-resident tier ───────────────────────────────────
// Applies the same fix as gpu_cuda.h's persistent-context tier and
// gpu_mps.h's GPU-resident training rewrite to WebGPU's version of the
// same problem: every webgpu_*_f32 function above re-runs the full
// instance/adapter/device acquisition sequence (__wgpu_init, including
// its async-callback poll loops) AND recompiles the shader module +
// pipeline, on every single call. The functions below fix both: a
// process-lifetime instance/device/queue (see gpu_webgpu.sc's
// __wgpu_ensure_device) created once and reused by every call in this
// tier, persistent device buffers uploaded once and reused across many
// dispatches, and cached compute pipelines (compiled once per kernel, not
// once per call). webgpu_sgd_update_f32 mirrors gpu_mps.h's
// mps_sgd_update_f32_chained / gpu_cuda.h's cuda_sgd_update_f32: weights
// updated in place on the GPU, never copied back to host memory, so a
// training loop can stay fully GPU-resident across steps.
//
// UNVERIFIED like the rest of this file (no wgpu-native/Dawn library in
// this sandbox). One gap this tier closes that the file's original
// __wgpu_run_kernel deliberately left open (see its comment): a STORAGE
// buffer generally can't also be MAP_READ, so webgpu_matmul_f32_persistent
// copies its output into a dedicated MAP_READ|COPY_DST staging buffer via
// wgpuCommandEncoderCopyBufferToBuffer before mapping it, the structurally
// correct WebGPU readback path.
//
// This tier has no command-batching layer of its own (the WebGPU analogue
// of gpu_mps.sc's mps_batch_begin/mps_batch_end, which lets many chained
// dispatches share one command-buffer submission and one wait) -- each
// call below still does its own encoder/submit. A caller chaining several
// of these in a training loop and reading results back only at the end
// (e.g. via a later persistent buffer's own map) still benefits from the
// device/pipeline caching this tier adds; folding dispatches into a
// single submission per step is a further step this pass doesn't take.

// Uploads 'bytes' from host memory to a new persistent GPU buffer usable
// as both storage input/output and copy source/destination. Returns
// (void*)0 on failure.
void* webgpu_upload_persistent(const float* data, unsigned long bytes);

// Releases a buffer returned by webgpu_upload_persistent (wgpuBufferRelease).
void webgpu_release_persistent(void* buf);

// Same computation as webgpu_matmul_f32, but 'bufA'/'bufB' are already
// device buffers (e.g. from webgpu_upload_persistent) -- no upload for
// either operand -- and both the compute pipeline and the shared device
// are cached/reused instead of being rebuilt on every call. Still copies
// the MxN result back to host 'out' (see this section's header comment on
// the batching gap). Returns 1 on success, 0 on failure.
int webgpu_matmul_f32_persistent(void* bufA, void* bufB, float* out,
                                  unsigned long M, unsigned long K, unsigned long N);

// In-place SGD step against a persistent device buffer:
// bufW[i] -= lr * bufGrad[i] for i in [0, n), computed entirely on-GPU --
// the result is never copied back to host memory. Returns 1 on success,
// 0 on failure.
int webgpu_sgd_update_f32(void* bufW, void* bufGrad, float lr, unsigned long n);

// True if the shared persistent device (used by every function in this
// tier) is available.
int webgpu_persistent_available();

} // namespace std
