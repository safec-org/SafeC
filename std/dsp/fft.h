// SafeC Standard Library — Fast Fourier Transform (radix-2 Cooley-Tukey)
//
// In-place, iterative decimation-in-time FFT: O(n log n) instead of
// dft.h's O(n^2) direct summation, at the cost of requiring n to be a
// power of two (0 is returned without modifying 'data' if it isn't —
// use dft.h for arbitrary lengths). Produces the same result as dft.h's
// dft()/idft() to within floating-point/fixed-point rounding — fft.sc's
// own tests verify this directly against the independent DFT
// implementation, not just against hand-computed reference values.
//
// Twiddle factors come from one table shared by all threads, built the
// first time a length needs it and kept. Stages run cache-blocked, and
// transforms of 32768 points or more are split across std/sync/parallel.h's
// thread pool; fft_batch / fft_real_batch spread many transforms over it.
//
// Real-time code (audio callbacks): call fft_prepare(maxN) once beforehand,
// then use fft_serial / ifft_serial, which never allocate, lock or touch the
// thread pool. fft / ifft below 32768 points don't either, once prepared.
#pragma once
#include <std/dsp/complex_dsp.h>

namespace std {

// True if n is a power of two (n >= 1).
int dsp_is_pow2(unsigned long n);

// In-place forward FFT. Returns 1 on success, 0 if n isn't a power of two
// (data is left unmodified in that case).
int fft(&Complex data, unsigned long n);

// In-place inverse FFT (includes the 1/N normalization). Returns 1 on
// success, 0 if n isn't a power of two.
int ifft(&Complex data, unsigned long n);

// fft / ifft on the calling thread only, whatever n.
int fft_serial(&Complex data, unsigned long n);
int ifft_serial(&Complex data, unsigned long n);

// Builds the twiddle table for every length up to n (a power of two), so
// later transforms of those lengths allocate nothing. 1 on success, 0 if n
// isn't a power of two or memory ran out.
int fft_prepare(unsigned long n);

// Real input -> its full n-point complex spectrum in 'out' (n elements;
// out[n-k] = conj(out[k])). Computed with one n/2-point complex transform.
// Returns 1 on success, 0 if n isn't a power of two.
int fft_real(const &double in, &Complex out, unsigned long n);

// 'batch' independent length-n transforms stored back to back (row r at
// data[r*n]), spread over the thread pool. inverse != 0 runs ifft.
int fft_batch(&Complex data, unsigned long n, unsigned long batch, int inverse);
// fft_real of 'batch' rows: in[r*n ..] -> out[r*n ..].
int fft_real_batch(const &double in, &Complex out, unsigned long n, unsigned long batch);

// Q8.24 fixed-point forward/inverse FFT, same in-place/power-of-two
// contract as the float versions above.
int ffft(&FComplex data, unsigned long n);
int fifft(&FComplex data, unsigned long n);

} // namespace std
