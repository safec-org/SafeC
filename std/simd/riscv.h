// SafeC Standard Library — std::simd, RISC-V convenience layer
//
// Pure naming/documentation layer over std/simd/simd.h's portable
// vec<T,N>-based types and functions — LLVM's RISC-V backend lowers every
// operation in simd.sc to real V-extension (RVV) instructions (vsetivli,
// vfadd.vv, vle32.v/vse32.v, ...) when the target has '+v' enabled, from
// the portable source; there is no separate hand-written implementation.
//
// Important difference from x86_64/AArch64/WASM: RVV registers are
// *scalable* (VLEN is implementation-defined, not fixed by the ISA), so
// there is no single "native width" the way SSE=128-bit or NEON=128-bit
// is. SafeC's vec<T,N> is a *fixed*-width type (N is a compile-time
// constant lane count, matching LLVM's fixed-vector IR), so these aliases
// pick a commonly-supported baseline (VLEN=128, the minimum required by
// the RVV 1.0 'V' extension's zvl128b requirement) rather than modeling
// true scalable vectors — LLVM still generates correct RVV code for a
// fixed-width vec<T,N> (as the vsetivli-prefixed instructions above show),
// it just fixes the vector length at compile time instead of querying it
// at runtime via vsetvli the way idiomatic scalable-RVV C code would.
#pragma once
#include <std/simd/simd.h>

namespace std {

typedef f32x4 rvv_f32x4;   // 128-bit lane group (zve32f/zve64f baseline)
typedef i32x4 rvv_i32x4;
typedef i8x16 rvv_i8x16;
typedef u8x16 rvv_u8x16;
typedef f64x2 rvv_f64x2;   // requires zve64d (double-precision RVV)

typedef f32x8 rvv_f32x8;   // 256-bit lane group (needs VLEN>=256 for one
typedef i32x8 rvv_i32x8;   // instruction; VLEN=128 targets legalize this
                            // into two 128-bit-group vector ops instead)

} // namespace std
