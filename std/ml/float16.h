#pragma once
// SafeC Standard Library — fp16 (IEEE 754 binary16) and bf16 (bfloat16)
// storage-format conversions.
//
// SafeC's type system has no native 16-bit float scalar (see Type.h:
// TypeKind only has Float32/Float64) — adding one would mean a real
// compiler feature (lexer/parser/sema/codegen changes for a new
// arithmetic type), well beyond what this file does. What's here instead
// is the same approach most systems use before/without a compiler-native
// half type: fp16/bf16 values are carried as raw bits in 'unsigned short'
// for storage, with explicit conversion to/from 'float' for any CPU-side
// arithmetic. This is a real, useful, and correctly-rounded (round-to-
// nearest-even) implementation, not a stub -- see gpu_mps.h's
// MpsBuf::upload_f16/upload_bf16 family for where this actually pays off:
// Apple GPUs have native 'half'/'bfloat' hardware types (confirmed by
// compiling both through this machine's real Metal toolchain), so a
// tensor converted to fp16/bf16 here and uploaded there gets real
// half/bfloat-precision compute, not just halved storage.
//
// Precision notes:
//  - fp16 (5 exponent bits, 10 mantissa bits, bias 15): much narrower
//    dynamic range than float32 (max ~65504, min normal ~6.1e-5) but the
//    same relative precision class GPUs have used for inference/training
//    for years. Values outside fp16's range round to +/-infinity.
//  - bf16 (8 exponent bits, 10 mantissa bits, bias 127): identical
//    exponent range to float32 (no overflow/underflow surprises going
//    float32 -> bf16), just fewer mantissa bits. Converting bf16 -> float32
//    is exact and lossless (a zero-extend, no rounding); float32 -> bf16
//    is a truncation with round-to-nearest-even.
namespace std {

// Converts a float32 value to its IEEE 754 binary16 (fp16) bit pattern,
// round-to-nearest-even. Overflow rounds to +/-infinity; NaN inputs
// produce a valid (payload-losing but still-NaN) fp16 NaN pattern.
unsigned short f32_to_fp16(float v);

// Converts an fp16 bit pattern back to float32. Exact -- every fp16
// value (including subnormals, infinities, and NaNs) is exactly
// representable in float32.
float fp16_to_f32(unsigned short h);

// Converts a float32 value to its bfloat16 bit pattern, round-to-
// nearest-even (a truncation of float32's top 16 bits, since bf16 shares
// float32's exponent width exactly). NaN inputs are kept as NaN (a
// mantissa bit is forced nonzero if truncation would otherwise zero it
// out, which would silently turn a NaN into an infinity).
unsigned short f32_to_bf16(float v);

// Converts a bf16 bit pattern back to float32. Exact and lossless (a
// zero-extension into the low 16 mantissa bits) -- every bf16 value is
// exactly representable in float32 by construction.
float bf16_to_f32(unsigned short b);

// ── What the CPU does natively ───────────────────────────────────────────────
// fp16 conversion is in every x86-64 CPU since 2012 (F16C) and in every
// AArch64 core; bf16 is newer and optional -- x86 AVX512-BF16 (Cooper Lake,
// Sapphire Rapids, Zen 4 and later) and Arm FEAT_BF16 (Armv8.6: Neoverse
// V1 / N2, Apple M2 and later) -- and so is fp16 arithmetic (AVX512-FP16;
// Arm FEAT_FP16). Detected once at run time (cpuid + xgetbv on x86-64,
// getauxval on Linux AArch64, sysctl on Apple). SAFEC_HALF=soft reports
// none and makes the bulk conversions below use the portable code.
#define HALF_CPU_F16_CONVERT   1   // fp16 <-> float32 instructions (F16C / AArch64 FCVTL, FCVTN)
#define HALF_CPU_F16_ARITH     2   // fp16 arithmetic (AVX512-FP16 / FEAT_FP16)
#define HALF_CPU_BF16_CONVERT  4   // float32 -> bf16 instructions (AVX512-BF16 / FEAT_BF16)
#define HALF_CPU_BF16_DOT      8   // bf16 dot products (VDPBF16PS / BFDOT)
int half_cpu_features();
// "F16C, AVX512-BF16" and the like; "none" for none.
const char* half_cpu_features_name();

// ── Bulk conversions (src and dst must not overlap) ─────────────────────────
// Bit-identical to the scalar functions above for every input except NaN
// payloads (the hardware keeps more of a NaN's payload; a NaN stays a NaN).
// x86-64 uses F16C for fp16 and, where present, AVX512-BF16 for
// float32 -> bf16 (16 values per instruction; a block holding a float32
// subnormal, which that instruction flushes to zero, is converted in
// software instead). AArch64 uses FCVTL / FCVTN for fp16. bf16 -> float32
// is a shift and float32 -> bf16 an integer rounding everywhere else, both
// in loops the compiler vectorizes. Arrays of 256K values or more are
// split across std/sync/parallel.h's thread pool.
void f32_to_fp16_bulk(const &float src, &unsigned short dst, unsigned long n);
void fp16_to_f32_bulk(const &unsigned short src, &float dst, unsigned long n);
void f32_to_bf16_bulk(const &float src, &unsigned short dst, unsigned long n);
void bf16_to_f32_bulk(const &unsigned short src, &float dst, unsigned long n);
// Between the two 16-bit formats (through float32, rounded to nearest even):
// fp16 -> bf16 loses 3 mantissa bits, bf16 -> fp16 rounds to fp16's range
// (overflow to infinity, small values to subnormals or zero).
void fp16_to_bf16_bulk(const &unsigned short src, &unsigned short dst, unsigned long n);
void bf16_to_fp16_bulk(const &unsigned short src, &unsigned short dst, unsigned long n);

} // namespace std
