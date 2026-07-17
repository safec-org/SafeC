// SafeC Standard Library — std::simd, WebAssembly convenience layer
//
// Pure naming/documentation layer over std/simd/simd.h's portable
// vec<T,N>-based types and functions — LLVM's WebAssembly backend lowers
// every operation in simd.sc to real SIMD128 instructions (f32x4.add,
// i32x4.mul, v128.load/store, ...) when targeting wasm32/wasm64 with the
// simd128 feature enabled, from the portable source; there is no separate
// hand-written implementation here.
//
// WASM's SIMD128 proposal defines exactly one 128-bit vector register
// shape (v128), reinterpreted as different lane layouts — there is no
// wider native register the way x86_64 has AVX. The *x8 float32 / *x4
// float64 aliases below are provided for API symmetry with the other
// per-ISA headers, but LLVM lowers them as a pair of v128 operations.
#pragma once
#include <std/simd/simd.h>

namespace std {

typedef f32x4 wasm_f32x4;   // v128, f32x4 lane interpretation (native)
typedef i32x4 wasm_i32x4;   // v128, i32x4 lane interpretation (native)
typedef i16x8 wasm_i16x8;   // v128, i16x8 lane interpretation (native)
typedef i8x16 wasm_i8x16;   // v128, i8x16 lane interpretation (native)
typedef f64x2 wasm_f64x2;   // v128, f64x2 lane interpretation (native)

typedef f32x8 wasm_f32x8;   // two v128 ops (not a single native register)
typedef f64x4 wasm_f64x4;   // two v128 ops (not a single native register)

} // namespace std
