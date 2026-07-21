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
// mps_upload_f16/mps_upload_bf16 family for where this actually pays off:
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

// Converts 'n' float32 values to fp16 in bulk -- src/dst may not overlap.
void f32_to_fp16_bulk(const float* src, unsigned short* dst, unsigned long n);
// Converts 'n' fp16 values back to float32 in bulk.
void fp16_to_f32_bulk(const unsigned short* src, float* dst, unsigned long n);
// Converts 'n' float32 values to bf16 in bulk.
void f32_to_bf16_bulk(const float* src, unsigned short* dst, unsigned long n);
// Converts 'n' bf16 values back to float32 in bulk.
void bf16_to_f32_bulk(const unsigned short* src, float* dst, unsigned long n);

} // namespace std
