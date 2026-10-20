#pragma once
// SafeC Standard Library — which SIMD instruction sets the CPU runs.
//
// std's hot kernels (sgemm, vmath, the FFT butterflies, dsp_dot_f64) are
// built in several versions and use the best one this returns, in order:
//   x86-64      AVX-512 (F+BW+DQ+VL) > AVX2 + FMA > AVX > SSE4.1 > SSE2
//   AArch64     SVE > Neon
//   Cortex-M    Helium (MVE) > DSP extension   -- compile time only: pick
//               the target (arm-81m for Helium, arm-7em / arm-8m for DSP)
//   WebAssembly SIMD128 when built with it
// On x86-64 and AArch64 Linux the level is detected once at run time
// (cpuid + xgetbv / getauxval), so one binary uses AVX-512 where it exists
// and still runs on an SSE2-only CPU. Whatever the build already
// guarantees (e.g. --target-cpu=native defines __AVX2__) is the floor.
// SAFEC_SIMD=sse2|sse4.1|avx|avx2|avx512|neon|sve caps the level, to
// test or compare the slower paths.

namespace std {

#define SIMD_SCALAR   0
#define SIMD_SSE2     1
#define SIMD_SSE41    2
#define SIMD_AVX      3
#define SIMD_AVX2     4   // AVX2 + FMA
#define SIMD_AVX512   5   // AVX-512 F, BW, DQ and VL
#define SIMD_NEON     10
#define SIMD_SVE      11
#define SIMD_DSP      20  // Cortex-M DSP extension
#define SIMD_HELIUM   21  // Cortex-M MVE
#define SIMD_WASM128  30

// The best level available (detected once, then cached).
int simd_level();
// "AVX-512", "AVX2", ..., "scalar".
const char* simd_level_name(int level);

} // namespace std
