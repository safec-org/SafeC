// SafeC Standard Library — Discrete Fourier Transform (direct, O(n^2))
//
// The direct-summation DFT: X[k] = sum_{n=0}^{N-1} x[n] * e^(-i*2*pi*k*n/N).
// Works for any N (not just powers of two, unlike fft.h's radix-2
// Cooley-Tukey) — use this for arbitrary-length transforms, and as the
// independent reference implementation fft.h's own tests verify against.
// For N large enough that O(n^2) matters, use fft.h instead when N is a
// power of two.
#pragma once
#include <std/dsp/complex_dsp.h>

namespace std {

// Forward transform, complex input -> complex output. 'out' must have
// space for 'n' elements; 'out' may not alias 'in'.
void dft(const struct Complex* in, struct Complex* out, unsigned long n);

// Forward transform, real input -> complex output (convenience: treats
// each sample as Complex(x, 0) internally without requiring the caller
// to build that array themselves — the common case for analyzing a real
// audio/sensor signal).
void dft_real(const double* in, struct Complex* out, unsigned long n);

// Inverse transform (includes the 1/N normalization), complex -> complex.
void idft(const struct Complex* in, struct Complex* out, unsigned long n);

// Q8.24 fixed-point forward/inverse transforms. Twiddle factors are
// computed in double precision and quantized to Fixed once per use (see
// complex_dsp.h's design-time-float/runtime-fixed note) — the summation
// itself runs entirely in Q8.24 arithmetic.
void fdft(const struct FComplex* in, struct FComplex* out, unsigned long n);
void fidft(const struct FComplex* in, struct FComplex* out, unsigned long n);

} // namespace std
