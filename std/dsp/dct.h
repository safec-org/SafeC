// SafeC Standard Library — Discrete Cosine Transform (DCT-II / DCT-III)
//
// dct2 is the "DCT-II" (what's usually just called "the DCT" — JPEG/MPEG
// use it for the same reason: real-valued input, real-valued output, and
// its energy-compaction property concentrates most of a typical signal's
// energy into the first few coefficients). idct3 is its exact inverse
// (the "DCT-III", sometimes called "the IDCT") — dct2 then idct3
// reconstructs the original signal exactly (to floating-point/fixed-point
// rounding), the same round-trip relationship dft.h/idft and fft/ifft have.
//
// Direct O(n^2) summation, like dft.h — see fft.h if n is a power of two
// and O(n log n) matters more than simplicity (there's no fast-DCT
// equivalent here yet; add one via an FFT-based algorithm if that
// becomes the bottleneck).
#pragma once
#include <std/dsp/fixed.h>

namespace std {

// X[k] = sum_{n=0}^{N-1} x[n] * cos(pi/N * (n+0.5) * k), k = 0..N-1.
int dct2([]const double in, []double out);

// Exact inverse of dct2: x[n] = (1/N) * (X[0] + 2*sum_{k=1}^{N-1} X[k] *
// cos(pi/N * (n+0.5) * k)).
int idct3([]const double in, []double out);

// Float in and out (computed in double, rounded once). Same contracts.
int dct2_f32([]const float in, []float out);
int idct3_f32([]const float in, []float out);

int fdct2([]const Fixed in, []Fixed out);
int fidct3([]const Fixed in, []Fixed out);

// Internal: the (reference, length) forms the slice API above wraps
// (std/dsp's own code calls these; not API).
void dct2_n_(const &double in, &double out, unsigned long n);
void idct3_n_(const &double in, &double out, unsigned long n);
void dct2_f32_n_(const &float in, &float out, unsigned long n);
void idct3_f32_n_(const &float in, &float out, unsigned long n);
void fdct2_n_(const &Fixed in, &Fixed out, unsigned long n);
void fidct3_n_(const &Fixed in, &Fixed out, unsigned long n);

} // namespace std
