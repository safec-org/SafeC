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

// True if a CUDA-capable device is present and the driver initializes
// successfully (cuInit(0) == CUDA_SUCCESS && cuDeviceGetCount() > 0).
int cuda_available();

} // namespace std
