// SafeC Standard Library — std::simd, AArch64 convenience layer
//
// Pure naming/documentation layer over std/simd/simd.h's portable
// vec<T,N>-based types and functions — LLVM's AArch64 backend lowers every
// operation in simd.sc directly to NEON (Advanced SIMD) instructions on the
// 32 128-bit v0-v31 registers (fadd/fmul .4s/.2d forms, etc.) from the
// portable source; there is no separate hand-written implementation here.
//
// NEON registers are natively 128-bit — unlike x86_64's AVX, there is no
// 256-bit hardware vector register, so the *x8 float32 / *x4 float64 types
// below are provided for API symmetry with the other per-ISA headers, but
// LLVM lowers them as a pair of 128-bit NEON operations (still correct,
// just two instructions instead of one). Prefer the 128-bit-native aliases
// (v4f32, v16i8, v2f64, ...) when writing AArch64-focused code.
#pragma once
#include <std/simd/simd.h>

namespace std {

typedef f32x4 v4f32;   // NEON float32x4_t  (128-bit, native)
typedef i32x4 v4i32;   // NEON int32x4_t    (128-bit, native)
typedef i16x8 v8i16;   // NEON int16x8_t    (128-bit, native)
typedef i8x16 v16i8;   // NEON int8x16_t    (128-bit, native)
typedef u8x16 v16u8;   // NEON uint8x16_t   (128-bit, native)
typedef u32x4 v4u32;   // NEON uint32x4_t   (128-bit, native)
typedef f64x2 v2f64;   // NEON float64x2_t  (128-bit, native)

typedef f32x8 v8f32;   // two NEON float32x4_t ops (not a single native register)
typedef f64x4 v4f64;   // two NEON float64x2_t ops (not a single native register)

} // namespace std
