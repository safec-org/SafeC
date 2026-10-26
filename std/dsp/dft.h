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
int dft([]const Complex in, []Complex out);

// Forward transform, real input -> complex output (convenience: treats
// each sample as Complex(x, 0) internally without requiring the caller
// to build that array themselves — the common case for analyzing a real
// audio/sensor signal).
int dft_real([]const double in, []Complex out);

// Inverse transform (includes the 1/N normalization), complex -> complex.
int idft([]const Complex in, []Complex out);

// Float in and out (computed in double, rounded once). Same contracts.
int dft_f32([]const Complex32 in, []Complex32 out);
int dft_real_f32([]const float in, []Complex32 out);
int idft_f32([]const Complex32 in, []Complex32 out);

// Q8.24 fixed-point forward/inverse transforms. Twiddle factors are
// computed in double precision and quantized to Fixed once per use (see
// complex_dsp.h's design-time-float/runtime-fixed note) — the summation
// itself runs entirely in Q8.24 arithmetic.
int fdft([]const FComplex in, []FComplex out);
int fidft([]const FComplex in, []FComplex out);

// Internal: the (reference, length) forms the slice API above wraps
// (std/dsp's own code calls these; not API).
void dft_n_(const &Complex in, &Complex out, unsigned long n);
void dft_real_n_(const &double in, &Complex out, unsigned long n);
void idft_n_(const &Complex in, &Complex out, unsigned long n);
void dft_f32_n_(const &Complex32 in, &Complex32 out, unsigned long n);
void dft_real_f32_n_(const &float in, &Complex32 out, unsigned long n);
void idft_f32_n_(const &Complex32 in, &Complex32 out, unsigned long n);
void fdft_n_(const &FComplex in, &FComplex out, unsigned long n);
void fidft_n_(const &FComplex in, &FComplex out, unsigned long n);

} // namespace std
