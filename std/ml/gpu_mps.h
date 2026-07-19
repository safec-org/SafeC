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
// KNOWN ISSUE (see gpu_mps.sc): segfaults on real hardware at the
// dispatchThreadgroups:threadsPerThreadgroup: call specifically — every
// earlier pipeline-setup step (device/queue, a real runtime MSL compile,
// pipeline state, buffers, encoder) was individually confirmed working.
// Apparent SafeC codegen gap with two consecutive struct-by-value
// objc_msgSend arguments; not yet fixed. Do not call this outside of
// further debugging — use mps_available() (fully verified) to at least
// confirm GPU presence.
int mps_add_f32(const float* a, const float* b, float* out, unsigned long n);

// True if a Metal device is available on this machine at all (checks
// MTLCreateSystemDefaultDevice() != nil) — lets callers fall back to the
// CPU Tensor ops (tensor.h) when there's no GPU, without needing to
// attempt an mps_* call first just to find out.
int mps_available();

} // namespace std
