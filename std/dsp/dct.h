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
void dct2(const double* in, double* out, unsigned long n);

// Exact inverse of dct2: x[n] = (1/N) * (X[0] + 2*sum_{k=1}^{N-1} X[k] *
// cos(pi/N * (n+0.5) * k)).
void idct3(const double* in, double* out, unsigned long n);

void fdct2(const Fixed* in, Fixed* out, unsigned long n);
void fidct3(const Fixed* in, Fixed* out, unsigned long n);

} // namespace std
