// SafeC Standard Library — Fast Fourier Transform (radix-2 Cooley-Tukey)
//
// In-place, iterative decimation-in-time FFT: O(n log n) instead of
// dft.h's O(n^2) direct summation, at the cost of requiring n to be a
// power of two (0 is returned without modifying 'data' if it isn't —
// use dft.h for arbitrary lengths). Produces the same result as dft.h's
// dft()/idft() to within floating-point/fixed-point rounding — fft.sc's
// own tests verify this directly against the independent DFT
// implementation, not just against hand-computed reference values.
#pragma once
#include <std/dsp/complex_dsp.h>

namespace std {

// True if n is a power of two (n >= 1).
int dsp_is_pow2(unsigned long n);

// In-place forward FFT. Returns 1 on success, 0 if n isn't a power of two
// (data is left unmodified in that case).
int fft(struct Complex* data, unsigned long n);

// In-place inverse FFT (includes the 1/N normalization). Returns 1 on
// success, 0 if n isn't a power of two.
int ifft(struct Complex* data, unsigned long n);

// Real input -> complex output convenience wrapper: packs 'in' into 'out'
// as Complex(x, 0) and FFTs 'out' in place. 'out' must have space for 'n'
// elements. Returns 1 on success, 0 if n isn't a power of two.
int fft_real(const double* in, struct Complex* out, unsigned long n);

// Q8.24 fixed-point forward/inverse FFT, same in-place/power-of-two
// contract as the float versions above.
int ffft(struct FComplex* data, unsigned long n);
int fifft(struct FComplex* data, unsigned long n);

} // namespace std
