#pragma once
// SafeC Standard Library — GPU tensor ops via Metal (Apple Silicon/macOS).
//
// The "unified memory" piece of the ML story (MLX's actual selling point):
// a Metal buffer created with the default storage mode is CPU+GPU shared
// memory on Apple Silicon — no explicit host<->device copy step the way
// CUDA/ROCm need (see gpu_cuda.h/gpu_rocm.h's discrete-memory model) —
// 'contents' on an MTLBuffer just IS a pointer the CPU can read straight
// out of after the GPU finishes writing it.
//
// Same "no real Objective-C support in safec" situation as
// std/gui/gui_cocoa.sc — every call here goes through the plain-C
// Objective-C runtime (objc_msgSend et al.) and the Metal framework's own
// C entry point (MTLCreateSystemDefaultDevice). See that file's header
// comment for the full technique (typedef fn-cast objc_msgSend per
// message shape). Only linkable/runnable on macOS with a Metal-capable
// GPU (every Apple Silicon Mac, and most Intel Macs from the last
// decade) — link with '-framework Metal -framework Foundation'.
//
// Gated behind the 'mps' feature (see Package.toml's [features] — same
// convention as every other backend-specific piece of std/ this session
// added): only compile/link this on a build that opted in, since a
// non-Apple target obviously can't provide these frameworks.
namespace std {

// Elementwise 'out[i] = a[i] + b[i]' for i in [0, n), computed on the GPU
// via a small inline Metal Shading Language compute kernel compiled at
// runtime. Returns 1 on success, 0 if no Metal device is available or any
// step of the pipeline setup fails (compile error in the embedded MSL
// source, no GPU present, etc.) — 'out' is left untouched on failure.
//
// Verified working end-to-end on real Apple Silicon hardware (correct
// output for a real GPU dispatch). Previously segfaulted at the
// dispatchThreadgroups:threadsPerThreadgroup: call specifically — root
// cause was a genuine SafeC compiler codegen bug (fixed in CodeGen.cpp's
// genCall), not anything specific to this file: any indirect call passing
// two or more consecutive struct-by-value arguments over 16 bytes (and not
// a Homogeneous Floating-point Aggregate) through a function-pointer cast
// hit it, because this compiler represented such arguments as raw LLVM
// aggregate values instead of the AAPCS64-correct "caller copies to a
// stack temporary, passes a plain pointer" form — self-consistent for
// SafeC-to-SafeC calls (which is why that never showed up before), but not
// what a real, externally-compiled function like objc_msgSend's target IMP
// expects to receive in its argument registers.
int mps_add_f32(const float* a, const float* b, float* out, unsigned long n);

// Same shape/verification status as mps_add_f32 (elementwise, GPU-executed,
// unified-memory readback) — sub/mul/div/pow are two-input, log/exp/sqrt
// are one-input. All share mps_add_f32's setup/dispatch/readback path
// (see __mps_run_binary_kernel/__mps_run_unary_kernel in gpu_mps.sc), so
// the fix that made mps_add_f32's GPU dispatch work applies to all of them
// identically — verified directly for mps_sub_f32/mps_mul_f32 as
// representative binary/no-special-case cases; div/pow/log/exp/sqrt follow
// the exact same code path with only the one-line kernel body differing.
int mps_sub_f32(const float* a, const float* b, float* out, unsigned long n);
int mps_mul_f32(const float* a, const float* b, float* out, unsigned long n);
int mps_div_f32(const float* a, const float* b, float* out, unsigned long n);
int mps_pow_f32(const float* a, const float* b, float* out, unsigned long n);
int mps_log_f32(const float* a, float* out, unsigned long n);
int mps_exp_f32(const float* a, float* out, unsigned long n);
int mps_sqrt_f32(const float* a, float* out, unsigned long n);
int mps_relu_f32(const float* a, float* out, unsigned long n);
int mps_scale_f32(const float* a, float k, float* out, unsigned long n);

// out[0] = sum(a[0..n)) -- serial single-thread reduction, not a real
// parallel tree reduction. See gpu_mps.sc's comment for why.
int mps_sum_f32(const float* a, float* out, unsigned long n);

// out[M,N] = a[M,K] . b[K,N], computed on the GPU. Naive (no threadgroup-
// memory tiling), float32-only (Metal has no 'double' type at all) — see
// gpu_mps.sc's comment on the implementation for both caveats in detail.
int mps_matmul_f32(const float* a, const float* b, float* out,
                    unsigned long M, unsigned long K, unsigned long N);

// True if a Metal device is available on this machine at all (checks
// MTLCreateSystemDefaultDevice() != nil) — lets callers fall back to the
// CPU Tensor ops (tensor.h) when there's no GPU, without needing to
// attempt an mps_* call first just to find out.
int mps_available();

} // namespace std
