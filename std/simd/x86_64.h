// SafeC Standard Library — std::simd, x86_64 convenience layer
//
// Pure naming/documentation layer over std/simd/simd.h's portable
// vec<T,N>-based types and functions — there is no separate x86_64
// implementation here; LLVM's X86 backend already lowers every operation in
// simd.sc to real SSE/SSE2 (128-bit) or AVX/AVX2 (256-bit) instructions
// (vaddps, vmulps, xmm/ymm registers, ...) directly from the portable
// source. This header just documents the native-width mapping and
// re-exports the types under names that read naturally for x86_64 code:
//   - 128-bit (SSE/SSE2, always available in x86-64's baseline ISA):
//       f32x4, i32x4, i8x16, u8x16, f64x2  ≈ __m128/__m128i/__m128d
//   - 256-bit (AVX/AVX2 — requires the target CPU to actually support it,
//     e.g. compile with '-mattr=+avx2' when lowering the emitted IR):
//       f32x8, i32x8, i8x32, f64x4  ≈ __m256/__m256i/__m256d
// If AVX2 isn't available on the target CPU, LLVM legalizes the 256-bit
// vector types by splitting them into a pair of 128-bit SSE operations —
// still correct, just not a single instruction.
#pragma once
#include <std/simd/simd.h>

namespace std {

typedef f32x4 m128;   // SSE   (4 x float,  128-bit)
typedef i32x4 m128i4;  // SSE2  (4 x int32,  128-bit)
typedef i8x16 m128i;  // SSE2  (16 x int8,  128-bit)
typedef f64x2 m128d;  // SSE2  (2 x double, 128-bit)

typedef f32x8 m256;   // AVX   (8 x float,  256-bit)
typedef i32x8 m256i4;  // AVX2  (8 x int32,  256-bit)
typedef i8x32 m256i;  // AVX2  (32 x int8,  256-bit)
typedef f64x4 m256d;  // AVX   (4 x double, 256-bit)

} // namespace std
