// SafeC Standard Library — std::simd, ARM Cortex-M convenience layer
//
// Unlike the other five per-ISA headers in std/simd/, Cortex-M support is
// *partial* and this header documents exactly where the line is, verified
// directly against this compiler's output (not assumed):
//
//   * Cortex-M55 / Cortex-M85 (Armv8.1-M, MVE/"Helium", target attribute
//     '+mve'): vec<T,N> arithmetic lowers to real MVE vector instructions
//     (q-register vadd.i32/vldrw.u32/vstrw.32, ...) — verified via
//     `--target thumbv8.1m.main-none-eabi` + `llc -mattr=+mve`. The
//     f32x4/i32x4/i16x8/i8x16 aliases below are meaningful and native
//     (one 128-bit MVE q-register) on these cores.
//
//   * Cortex-M4 / Cortex-M7 (Armv7E-M, DSP extension, target attribute
//     '+dsp'): the DSP extension's packed-SIMD instructions (SADD16,
//     SMLAD, ...) operate on 2x16-bit or 4x8-bit lanes packed into a
//     single 32-bit *scalar* register, not a real vector register file.
//     LLVM does not auto-vectorize generic vec<T,N> IR into these
//     instructions the way it does for MVE/NEON/SSE/RVV/SIMD128 — a
//     `vec<short,2>` add on `thumbv7em` lowers to two independent 32-bit
//     `add`s, not a single `sadd16` (confirmed by inspecting compiler
//     output), so there is no vec<T,N>-based type alias for DSP below.
//     Instead, the dsp_* functions declared below wrap compiler builtins
//     (__arm_dsp_*, see CodeGen.cpp) that map 1:1 onto the real
//     instructions — verified against real llc output on
//     '-mtriple=thumbv7em-none-eabi -mcpu=cortex-m4' (qadd/sadd16/smlad/
//     usad8/ssat all emit the single named instruction, not a libcall).
//
//   * Cortex-M0 / M0+ / M3 / M23 (no DSP, no MVE): vec<T,N> still compiles
//     (verified: whole stdlib compiles clean for `thumbv6m-none-eabi
//     --freestanding`) and is fully correct, but every lane operation is
//     scalarized — there is no SIMD hardware to target at all on these
//     cores, so vec<T,N> here is purely a portable-code convenience, not
//     a performance feature.
#pragma once
#include <std/simd/simd.h>

namespace std {

// Cortex-M55/M85 (MVE, '+mve'): native 128-bit q-register operations.
typedef f32x4 mve_f32x4;
typedef i32x4 mve_i32x4;
typedef i16x8 mve_i16x8;
typedef i8x16 mve_i8x16;

// ── DSP extension: packed-SIMD / saturating arithmetic (Cortex-M4/M7) ────────
// Each 'int'/'unsigned int' operand packs 2x16-bit or 4x8-bit lanes into one
// 32-bit word (lane 0 in the low bits); packing/unpacking individual lanes
// is the caller's job (shifts, or reading two adjacent short/char values as
// one int through a union or pointer cast). Requires an ARM target — using
// any of these while compiling for a non-ARM target is a compile error at
// the call site (enforced by the underlying __arm_dsp_* builtin).
int          dsp_qadd(int a, int b);       // saturating 32-bit add
int          dsp_qsub(int a, int b);       // saturating 32-bit sub
int          dsp_qadd16(int a, int b);     // saturating 2x16-bit add
int          dsp_qadd8(int a, int b);      // saturating 4x8-bit add
int          dsp_qsub16(int a, int b);     // saturating 2x16-bit sub
int          dsp_qsub8(int a, int b);      // saturating 4x8-bit sub
int          dsp_sadd16(int a, int b);     // wrapping 2x16-bit add
int          dsp_sadd8(int a, int b);      // wrapping 4x8-bit add
int          dsp_ssub16(int a, int b);     // wrapping 2x16-bit sub
int          dsp_ssub8(int a, int b);      // wrapping 4x8-bit sub
unsigned int dsp_uqadd16(unsigned int a, unsigned int b);
unsigned int dsp_uqadd8(unsigned int a, unsigned int b);
unsigned int dsp_uqsub16(unsigned int a, unsigned int b);
unsigned int dsp_uqsub8(unsigned int a, unsigned int b);
int          dsp_smlad(int a, int b, int acc);   // dual 16x16 multiply-accumulate
int          dsp_smladx(int a, int b, int acc);  // exchanged (cross) lanes
int          dsp_smlsd(int a, int b, int acc);   // dual 16x16 multiply-subtract
int          dsp_smlsdx(int a, int b, int acc);
int          dsp_smuad(int a, int b);            // dual 16x16 multiply, summed
int          dsp_smuadx(int a, int b);
int          dsp_smusd(int a, int b);            // dual 16x16 multiply, difference
int          dsp_smusdx(int a, int b);
unsigned int dsp_usad8(unsigned int a, unsigned int b);  // sum of |byte diffs|, 4 lanes
unsigned int dsp_usada8(unsigned int a, unsigned int b, unsigned int acc);
int          dsp_sxtab16(int a, int b);          // sign-extend b's two bytes, add to a's halfwords
unsigned int dsp_uxtab16(unsigned int a, unsigned int b);

// dsp_ssat/dsp_usat/dsp_ssat16/dsp_usat16 are NOT wrapped here: the
// underlying SSAT/USAT instructions take their bit-width as an immediate
// encoded directly into the instruction, and SafeC's Sema enforces that at
// the __arm_dsp_ssat(val, bits) call site — 'bits' must literally be an
// integer-literal expression there. A forwarding function parameter can
// never satisfy that (it's a runtime value from the wrapper's own callers,
// not a literal), so call the builtin directly with a literal bit width:
//   int y = __arm_dsp_ssat(x, 8);   // saturate to signed 8-bit range

} // namespace std
