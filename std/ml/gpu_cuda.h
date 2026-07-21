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

// ── Persistent-context / GPU-resident tier ──────────────────────────────────
// Applies the same fix the MPS backend's GPU-resident training rewrite
// used, to a worse version of the same problem: every cuda_*_f32 function
// above pays a full cuInit/cuCtxCreate_v2/cuCtxDestroy_v2 cycle AND a
// fresh cuMemAlloc_v2/cuMemcpyHtoD_v2/cuMemFree_v2 round trip on EVERY
// call -- there is no context or buffer reuse at all above (MPS's
// pre-session code at least kept one shared MTLDevice/command queue
// alive). The functions below fix both: a process-lifetime CUcontext
// (see gpu_cuda.sc's __cuda_ensure_context) created once and reused by
// every call in this tier, and persistent device buffers uploaded once
// and reused across many dispatches -- e.g. a training loop's weights,
// updated in place by cuda_sgd_update_f32 without ever coming back to the
// host, the same shape as gpu_mps.h's mps_upload_persistent/
// mps_matmul_f32_persistent/mps_sgd_update_f32_chained family. UNVERIFIED
// like the rest of this file (no NVIDIA GPU/CUDA toolkit in this
// sandbox), but the reason for this tier is a real, structural fix, not
// speculative tuning: the existing per-call functions above are left
// untouched (still correct, still usable standalone) so this tier is
// additive, not a rewrite.
//
// CUdeviceptr (an unsigned long long) is boxed/unboxed through a plain
// void* here, the same round-trip cuda_matmul_f32_blas already uses for
// cuBLAS's pointer-typed arguments -- safe on any platform where
// sizeof(void*) >= 8, true of every target this compiler supports.

// Uploads 'bytes' from host memory to a new persistent device buffer.
// Returns (void*)0 on failure (no device, context creation failure, or
// allocation failure).
void* cuda_upload_persistent(const float* data, unsigned long bytes);

// Frees a buffer returned by cuda_upload_persistent.
void cuda_release_persistent(void* devPtr);

// Same computation as cuda_matmul_f32, but 'devA'/'devB' are already
// device buffers (e.g. from cuda_upload_persistent) -- no upload for
// either operand -- and dispatch reuses the shared persistent context and
// a cached module/kernel instead of reloading PTX and creating/destroying
// a context on every call. Still copies the MxN result back to host
// 'out': unlike MPS, this file has no command-buffer batching layer (see
// gpu_mps.sc's mps_batch_begin/mps_batch_end) to defer that across
// multiple chained ops, so each call here still does one cuCtxSynchronize
// + cuMemcpyDtoH_v2. Returns 1 on success, 0 on failure.
int cuda_matmul_f32_persistent(void* devA, void* devB, float* out,
                                unsigned long M, unsigned long K, unsigned long N);

// In-place SGD step against a persistent device buffer:
// devW[i] -= lr * devGrad[i] for i in [0, n), computed entirely on-GPU --
// the result is never copied back to host memory, mirroring
// gpu_mps.h's mps_sgd_update_f32_chained. This is the primitive that lets
// a training loop keep weights GPU-resident across steps: upload once via
// cuda_upload_persistent, forward/backward via cuda_matmul_f32_persistent
// (and this file's other _persistent ops), update in place via this
// function, repeat -- with no per-step re-upload the way a CPU-resident
// weight array read/written by a CPU-side SGD loop would need. Returns 1
// on success, 0 on failure.
int cuda_sgd_update_f32(void* devW, void* devGrad, float lr, unsigned long n);

// True if the shared persistent context (used by every function in this
// tier) is available -- like cuda_available(), but also confirms context
// creation succeeded, not just device enumeration.
int cuda_persistent_available();

// ── fp16 / bf16 tier ─────────────────────────────────────────────────────
// Real half/bfloat16 arithmetic in PTX, not just storage: NVIDIA GPUs
// have native f16 datapaths since Pascal (sm_60, 2016) and native bf16
// since Ampere (sm_80, 2020) -- this is a real hardware-generation split,
// not an arbitrary choice, so cuda_matmul_f16_persistent's embedded PTX
// targets sm_60 and cuda_matmul_bf16_persistent's targets sm_80. Storage
// is '.f16'/'.bf16' (half the bytes/element of the f32 tier); each PTX
// kernel converts to f32 at the point of arithmetic ('cvt.f32.f16'/
// 'cvt.f32.bf16') and back at the point of store ('cvt.rn.f16.f32'/
// 'cvt.rn.bf16.f32') -- the same storage-half/compute-float mixed-
// precision approach gpu_mps_kernels.metal's matmul_kernel_f16/bf16 use,
// just expressed in PTX's cvt instructions instead of MSL's implicit
// half<->float conversions.
//
// Deliberately smaller than gpu_mps.h's fp16/bf16 persistent tier: only
// upload/release, matmul, and in-place SGD are here, not the full
// sub/scale/matmul-transposed/relu-backward set needed for a complete
// backward pass. The MPS tier could be (and was) compiled and run on
// real Apple Silicon hardware to catch mistakes; this file has no ptxas
// in this sandbox to catch a register-count or instruction-encoding
// error before a real CUDA host would -- see gpu_cuda.sc's/gpu_cuda.h's
// top-of-file UNVERIFIED warning, which applies here at least as much as
// to the rest of this file.

void* cuda_upload_f16_persistent(const float* data, unsigned long n);
void* cuda_upload_bf16_persistent(const float* data, unsigned long n);
// Both release through cuda_release_persistent(devPtr) above -- storage
// is 2 bytes/element regardless of fp16 vs bf16.

// out = devA[M,K] . devB[K,N], both operands already device buffers (see
// cuda_upload_f16_persistent/cuda_upload_bf16_persistent above). 'out' is
// a host array of raw fp16/bf16 bits (see float16.h's fp16_to_f32_bulk/
// bf16_to_f32_bulk to convert it back to float32).
int cuda_matmul_f16_persistent(void* devA, void* devB, unsigned short* out,
                                unsigned long M, unsigned long K, unsigned long N);
int cuda_matmul_bf16_persistent(void* devA, void* devB, unsigned short* out,
                                 unsigned long M, unsigned long K, unsigned long N);

// In-place SGD step against a persistent fp16/bf16 device buffer:
// devW[i] -= lr*devGrad[i], computed via cvt-to-f32/cvt-back the same
// way the matmul kernels above do -- mirrors cuda_sgd_update_f32.
int cuda_sgd_update_f16(void* devW, void* devGrad, float lr, unsigned long n);
int cuda_sgd_update_bf16(void* devW, void* devGrad, float lr, unsigned long n);

} // namespace std
