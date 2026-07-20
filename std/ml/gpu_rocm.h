#pragma once
// SafeC Standard Library — GPU tensor ops via ROCm/HIP (AMD).
//
// Written against the real HIP Runtime API's C ABI (hand-matched to
// hip_runtime_api.h) and type-checked by safec, but — like gpu_cuda.h,
// gui_win32.sc, and gui_x11.sc — genuinely UNLINKABLE/UNRUNNABLE in this
// sandbox: no AMD GPU, no ROCm toolkit installed here. Sanity-check
// against a real ROCm-capable host before depending on it.
//
// Narrower than gpu_cuda.h's PTX approach: CUDA's PTX is a stable,
// human-writable *text* IR, so gpu_cuda.sc can embed a kernel as a string
// literal and load it with cuModuleLoadData at runtime with no build-time
// toolchain dependency. HIP's equivalent module format (HSACO) is a
// compiled ELF-like *binary* blob — there's no practical hand-written-text
// equivalent — so hipModuleLoadData here is real and correctly declared,
// but actually calling it requires a real hipcc/offline-compiled HSACO
// byte array this header does not (and, without a ROCm toolchain
// available to produce one, cannot) embed. See gpu_rocm.sc's
// rocm_add_f32 for exactly where that gap is and what it would take to
// close it on a real ROCm host.
//
// Gated behind the 'rocm' feature (see Package.toml's [features]).
namespace std {

// Elementwise 'out[i] = a[i] + b[i]' for i in [0, n) on an AMD GPU.
// Currently always returns 0 (see gpu_rocm.sc's header comment on the
// HSACO gap) — the HIP Runtime API plumbing around it is real and
// type-checked, but there's no embeddable kernel body yet.
int rocm_add_f32(const float* a, const float* b, float* out, unsigned long n);

// Same shape and same HSACO-gap caveat as rocm_add_f32 — all real HIP
// Runtime API plumbing, all currently always returning 0 for lack of a
// compiled kernel image. See gpu_rocm.sc's header comment.
int rocm_sub_f32(const float* a, const float* b, float* out, unsigned long n);
int rocm_mul_f32(const float* a, const float* b, float* out, unsigned long n);
int rocm_div_f32(const float* a, const float* b, float* out, unsigned long n);
int rocm_pow_f32(const float* a, const float* b, float* out, unsigned long n);
int rocm_relu_f32(const float* a, float* out, unsigned long n);
int rocm_log_f32(const float* a, float* out, unsigned long n);
int rocm_exp_f32(const float* a, float* out, unsigned long n);
int rocm_sqrt_f32(const float* a, float* out, unsigned long n);
int rocm_scale_f32(const float* a, float k, float* out, unsigned long n);
int rocm_sum_f32(const float* a, float* out, unsigned long n);
int rocm_matmul_f32(const float* a, const float* b, float* out,
                     unsigned long M, unsigned long K, unsigned long N);

// Same computation as rocm_matmul_f32, dispatched to rocBLAS's
// rocblas_sgemm instead — the ROCm-vendor equivalent of tensor_blas.h's
// Accelerate-backed tensor_matmul_blas / gpu_cuda.h's cuBLAS-backed
// cuda_matmul_f32_blas. Unlike rocm_matmul_f32, this does NOT hit the
// HSACO-embedding gap described above: rocblas_sgemm is an ordinary
// exported symbol in librocblas.so, not a kernel image that needs
// offline compilation, so this is actually usable on real ROCm hardware
// today (still UNVERIFIED in this sandbox — no AMD GPU/ROCm toolkit here
// either).
int rocm_matmul_f32_blas(const float* a, const float* b, float* out,
                          unsigned long M, unsigned long K, unsigned long N);

// True if a ROCm-capable device is present (hipInit(0) == hipSuccess &&
// hipGetDeviceCount() > 0).
int rocm_available();

} // namespace std
