// SafeC Standard Library — std::simd, CUDA (NVPTX) convenience layer
//
// Pure naming/documentation layer over std/simd/simd.h's portable
// vec<T,N>-based types and functions — LLVM's NVPTX backend lowers every
// operation in simd.sc to real PTX instructions from the portable source;
// verified directly against this compiler's output (--target
// nvptx64-nvidia-cuda), not merely assumed from the other backends.
//
// Important difference from the CPU backends (x86_64/AArch64/RISC-V):
// PTX has no packed/SIMD arithmetic instructions the way SSE or NEON do —
// there is no single "add 4 floats in one instruction" on this ISA. A GPU
// gets its parallelism from launching many threads (a warp of 32 lockstep
// threads), each executing scalar code on its own data, not from wide
// vector registers within one thread. Accordingly, LLVM's NVPTX backend
// lowers a vec<float,4> add into four independent scalar 'add.rn.f32'
// instructions (confirmed by inspecting real llc output), not one wide
// op — vec<T,N> here is a convenient, portable way to write "4 related
// values," not a promise of single-instruction throughput the way it is
// on a CPU target. Real GPU-scale parallelism comes from how many threads
// the kernel launch itself spans, which vec<T,N> has no part in.
//
// Also unlike the CPU targets: NVPTX code only runs as a CUDA kernel
// launched by a host program through the CUDA driver/runtime API — there
// is no free-standing "main" or host libc to call into from device code.
// Everything in std/simd/simd.h that touches a raw host pointer (the
// simd_load_*/simd_store_* functions) is meaningful *inside* a kernel
// body operating on device memory the host already bound, not as
// standalone host-callable functions. This header does not attempt to
// define kernel *entry points* (which need __global__/ptx_kernel-style
// calling-convention plus launch-configuration support, a separate
// feature) — it only provides the portable arithmetic types usable
// *within* a kernel body, the same scope std::simd's SPIR-V layer takes.
#pragma once
#include <std/simd/simd.h>

namespace std {

typedef f32x4 cuda_f32x4;
typedef f32x8 cuda_f32x8;
typedef f64x2 cuda_f64x2;
typedef i32x4 cuda_i32x4;
typedef i32x8 cuda_i32x8;

} // namespace std
