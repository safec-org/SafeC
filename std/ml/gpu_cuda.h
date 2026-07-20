#pragma once
// SafeC Standard Library — GPU tensor ops via CUDA (NVIDIA).
//
// Written carefully against the real CUDA Driver API's C ABI (function
// names/signatures hand-matched to cuda.h) and type-checked by safec, but
// — like std/gui/gui_win32.sc and gui_x11.sc earlier in this library —
// genuinely UNLINKABLE/UNRUNNABLE in this sandbox: no NVIDIA GPU, no CUDA
// toolkit installed here (confirmed: no nvcc, no nvidia-smi, no
// /usr/local/cuda). Sanity-check against a real CUDA-capable host before
// depending on it, same caveat as those two files.
//
// Uses the Driver API (cu*, not the higher-level cudart cuda*) plus an
// inline PTX string compiled at runtime via cuModuleLoadData — the CUDA
// analogue of gpu_mps.sc's runtime-compiled Metal Shading Language kernel
// (PTX is NVIDIA's stable, textual, forward-compatible IR; embedding it
// directly avoids needing nvcc/nvrtc as a build-time dependency, the same
// reasoning that made "compile MSL from a string at runtime" the right
// choice for Metal). Discrete memory model, unlike MPS's unified memory
// (see gpu_mps.h) — every buffer needs explicit cuMemcpyHtoD/DtoH.
//
// Gated behind the 'cuda' feature (see Package.toml's [features]).
namespace std {

// Elementwise 'out[i] = a[i] + b[i]' for i in [0, n), computed on an
// NVIDIA GPU. Returns 1 on success, 0 on any failure (no CUDA-capable
// device, driver/runtime mismatch, PTX load failure, etc.).
int cuda_add_f32(const float* a, const float* b, float* out, unsigned long n);

// Same shape/caveats as cuda_add_f32 -- all hand-written PTX, all
// UNVERIFIED (no ptxas, no NVIDIA GPU in this sandbox). See gpu_cuda.sc
// for each kernel's source and the pow/log/exp approximation notes.
int cuda_sub_f32(const float* a, const float* b, float* out, unsigned long n);
int cuda_mul_f32(const float* a, const float* b, float* out, unsigned long n);
int cuda_div_f32(const float* a, const float* b, float* out, unsigned long n);
int cuda_pow_f32(const float* a, const float* b, float* out, unsigned long n);
int cuda_relu_f32(const float* a, float* out, unsigned long n);
int cuda_log_f32(const float* a, float* out, unsigned long n);
int cuda_exp_f32(const float* a, float* out, unsigned long n);
int cuda_sqrt_f32(const float* a, float* out, unsigned long n);
int cuda_scale_f32(const float* a, float k, float* out, unsigned long n);
int cuda_sum_f32(const float* a, float* out, unsigned long n);
int cuda_matmul_f32(const float* a, const float* b, float* out,
                     unsigned long M, unsigned long K, unsigned long N);

// Same computation as cuda_matmul_f32, dispatched to cuBLAS's cublasSgemm
// instead of the hand-written PTX kernel above — the CUDA-vendor
// equivalent of tensor_blas.h's Accelerate-backed tensor_matmul_blas.
// Real vendor GEMMs (cuBLAS included) are tuned per-GPU-architecture in
// ways a single hand-written PTX kernel isn't; see gpu_cuda.sc's
// implementation comment for the row-major/column-major handling (cuBLAS
// is Fortran-convention column-major; SafeC's Tensor is row-major).
// Same UNVERIFIED caveat as every other function in this file (no NVIDIA
// GPU, no CUDA toolkit, no cuBLAS library in this sandbox).
int cuda_matmul_f32_blas(const float* a, const float* b, float* out,
                          unsigned long M, unsigned long K, unsigned long N);

// True if a CUDA-capable device is present and the driver initializes
// successfully (cuInit(0) == CUDA_SUCCESS && cuDeviceGetCount() > 0).
int cuda_available();

} // namespace std
