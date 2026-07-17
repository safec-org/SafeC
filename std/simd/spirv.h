// SafeC Standard Library — std::simd, SPIR-V convenience layer
//
// Pure naming/documentation layer over std/simd/simd.h's portable
// vec<T,N>-based types and functions — LLVM's (experimental) SPIR-V
// backend lowers vec<T,N> arithmetic directly to real SPIR-V vector
// instructions (OpTypeVector + OpFAdd/OpFMul/...) from the portable
// source; verified directly against this compiler's output (--target
// spirv64-unknown-unknown), not merely assumed from the other backends.
//
// Important caveat, unlike the other four std::simd targets: SPIR-V is a
// GPU/compute IR (Vulkan, OpenCL), not a CPU ISA — a SPIR-V module is a
// *kernel* executed by a GPU driver, with no host libc, so functions like
// simd_load_f32x4/simd_store_f32x4 in the portable core (which read/write
// an arbitrary host pointer) and anything using variadic host calls (e.g.
// printf, used in this library's own doc examples) are meaningful on the
// CPU targets but not in an actual SPIR-V kernel — a real SPIR-V compute
// shader gets its data through bound buffer arguments and writes results
// the same way, per whatever host API (Vulkan/OpenCL) launched it. The
// arithmetic types and operators below are exactly what a numeric SPIR-V
// kernel body would use.
#pragma once
#include <std/simd/simd.h>

namespace std {

typedef f32x4 spv_f32x4;
typedef f32x8 spv_f32x8;
typedef f64x2 spv_f64x2;
typedef f64x4 spv_f64x4;
typedef i32x4 spv_i32x4;
typedef i32x8 spv_i32x8;

} // namespace std
