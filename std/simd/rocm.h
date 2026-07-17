// SafeC Standard Library — std::simd, ROCm (AMDGPU/GCN) convenience layer
//
// Pure naming/documentation layer over std/simd/simd.h's portable
// vec<T,N>-based types and functions — LLVM's AMDGPU backend lowers every
// operation in simd.sc to real GCN/RDNA vector-ALU instructions from the
// portable source; verified directly against this compiler's output
// (--target amdgcn-amd-amdhsa -mcpu=gfx900), not merely assumed from the
// other backends.
//
// Same SIMT execution model as std/simd/cuda.h (see its longer note): GCN
// has no single instruction that adds four packed 32-bit lanes the way
// SSE/NEON do — a vec<float,4> add lowers to four independent
// 'v_add_f32_e32' instructions (confirmed by inspecting real llc output),
// one per lane, each a "vector" ALU op in AMD's per-lane/per-wavefront
// sense rather than a CPU-style packed op. Real throughput comes from how
// many work-items a wavefront/kernel launch spans, not from vec<T,N>
// itself. Like the CUDA layer, this only provides portable arithmetic
// types usable *within* a kernel body (device memory, no host libc) — it
// does not define kernel entry points (amdgpu_kernel calling convention
// plus launch-configuration support is a separate feature).
#pragma once
#include <std/simd/simd.h>

namespace std {

typedef f32x4 rocm_f32x4;
typedef f32x8 rocm_f32x8;
typedef f64x2 rocm_f64x2;
typedef i32x4 rocm_i32x4;
typedef i32x8 rocm_i32x8;

} // namespace std
