// SafeC Standard Library — std::simd (portable core)
//
// Built on SafeC's native 'vec<T, N>' language type, which lowers directly
// to LLVM's <N x T> vector IR. Elementwise arithmetic (+, -, *, /) and lane
// access (v[i], including assignment) already work on any vec<T,N> through
// ordinary operator/subscript syntax — nothing in this file is "the SIMD
// implementation" in the sense of hand-written per-instruction code. What
// this file adds is:
//   - convenient, ISA-agnostic type names (f32x4, i32x8, ...)
//   - load/store between a vector and a raw pointer
//   - broadcast (splat), horizontal reductions, and fused multiply-add
//
// Every operation here compiles correctly on every target this compiler
// supports (x86_64, AArch64, RISC-V, WebAssembly, SPIR-V, ...): LLVM's
// backend selects the actual per-ISA instructions (SSE/AVX, NEON, the V
// extension, SIMD128, ...) from the vector IR alone, so this same source
// does not need to be rewritten per architecture. See std/simd/x86_64.h,
// aarch64.h, riscv.h, wasm.h and spirv.h for thin per-target convenience
// layers (native-preferred-width naming, doc notes on real hardware
// mapping) built on top of this file — they contain no separate logic.
#pragma once

namespace std {

// ── Vector type aliases ───────────────────────────────────────────────────────
// Name convention: <elem><bits>x<lanes>. Chosen widths cover every target's
// baseline SIMD register (128-bit: SSE, NEON, WASM SIMD128, RVV with
// VLEN>=128) plus the common 256-bit case (AVX2; on other targets LLVM
// legalizes a 256-bit vec by splitting it into two 128-bit register ops,
// still correct, just not a single native instruction).
typedef vec<float, 4>          f32x4;
typedef vec<float, 8>          f32x8;
typedef vec<double, 2>         f64x2;
typedef vec<double, 4>         f64x4;
typedef vec<int, 4>             i32x4;
typedef vec<int, 8>             i32x8;
typedef vec<long long, 2>       i64x2;
typedef vec<long long, 4>       i64x4;
typedef vec<short, 8>           i16x8;
typedef vec<short, 16>          i16x16;
typedef vec<signed char, 16>    i8x16;
typedef vec<signed char, 32>    i8x32;
typedef vec<unsigned int, 4>    u32x4;
typedef vec<unsigned int, 8>    u32x8;
typedef vec<unsigned char, 16>  u8x16;
typedef vec<unsigned char, 32>  u8x32;

// ── Load / store ──────────────────────────────────────────────────────────────
// Unaligned by construction (element-by-element); safe to use on any
// pointer. Reads/writes require 'unsafe' at the call site like any other
// raw-pointer access in SafeC.
f32x4  simd_load_f32x4(const float* p);
f32x8  simd_load_f32x8(const float* p);
f64x2  simd_load_f64x2(const double* p);
f64x4  simd_load_f64x4(const double* p);
i32x4  simd_load_i32x4(const int* p);
i32x8  simd_load_i32x8(const int* p);
u32x4  simd_load_u32x4(const unsigned int* p);
u32x8  simd_load_u32x8(const unsigned int* p);
i8x16  simd_load_i8x16(const signed char* p);
u8x16  simd_load_u8x16(const unsigned char* p);

void simd_store_f32x4(f32x4 v, float* p);
void simd_store_f32x8(f32x8 v, float* p);
void simd_store_f64x2(f64x2 v, double* p);
void simd_store_f64x4(f64x4 v, double* p);
void simd_store_i32x4(i32x4 v, int* p);
void simd_store_i32x8(i32x8 v, int* p);
void simd_store_u32x4(u32x4 v, unsigned int* p);
void simd_store_u32x8(u32x8 v, unsigned int* p);
void simd_store_i8x16(i8x16 v, signed char* p);
void simd_store_u8x16(u8x16 v, unsigned char* p);

// ── Broadcast (splat) ─────────────────────────────────────────────────────────
f32x4 simd_splat_f32x4(float x);
f32x8 simd_splat_f32x8(float x);
f64x2 simd_splat_f64x2(double x);
f64x4 simd_splat_f64x4(double x);
i32x4 simd_splat_i32x4(int x);
i32x8 simd_splat_i32x8(int x);
u32x4 simd_splat_u32x4(unsigned int x);

// ── Fused multiply-add: a*b + c ────────────────────────────────────────────────
f32x4 simd_fma_f32x4(f32x4 a, f32x4 b, f32x4 c);
f32x8 simd_fma_f32x8(f32x8 a, f32x8 b, f32x8 c);
f64x2 simd_fma_f64x2(f64x2 a, f64x2 b, f64x2 c);
f64x4 simd_fma_f64x4(f64x4 a, f64x4 b, f64x4 c);

// ── Elementwise min/max ────────────────────────────────────────────────────────
f32x4 simd_min_f32x4(f32x4 a, f32x4 b);
f32x4 simd_max_f32x4(f32x4 a, f32x4 b);
f32x8 simd_min_f32x8(f32x8 a, f32x8 b);
f32x8 simd_max_f32x8(f32x8 a, f32x8 b);
i32x4 simd_min_i32x4(i32x4 a, i32x4 b);
i32x4 simd_max_i32x4(i32x4 a, i32x4 b);

// ── Horizontal reductions ─────────────────────────────────────────────────────
float  simd_hsum_f32x4(f32x4 v);
float  simd_hmin_f32x4(f32x4 v);
float  simd_hmax_f32x4(f32x4 v);
float  simd_hsum_f32x8(f32x8 v);
double simd_hsum_f64x2(f64x2 v);
double simd_hsum_f64x4(f64x4 v);
int    simd_hsum_i32x4(i32x4 v);
int    simd_hsum_i32x8(i32x8 v);
unsigned int simd_hsum_u32x4(u32x4 v);

} // namespace std
