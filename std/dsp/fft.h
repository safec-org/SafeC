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
// With oneMKL linked (on hosted targets other than Windows), its DFTI
// transforms run instead below 65536 points, for every batch row and for
// fft2d; see fft.sc. fft_vendor() is 1 when that is the case. No
// FFTW-based library is used (they are GPL).
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
int fft([]Complex data);

// In-place inverse FFT (includes the 1/N normalization). Returns 1 on
// success, 0 if n isn't a power of two.
int ifft([]Complex data);

// fft / ifft on the calling thread only, whatever n.
int fft_serial([]Complex data);
int ifft_serial([]Complex data);

// 1 if fft/ifft/fft_batch/fft_real use a linked vendor library (oneMKL).
int fft_vendor();

// Builds the twiddle table for every length up to n (a power of two), so
// later transforms of those lengths allocate nothing. 1 on success, 0 if n
// isn't a power of two or memory ran out.
int fft_prepare(unsigned long n);

// Real input -> its full n-point complex spectrum in 'out' (n elements;
// out[n-k] = conj(out[k])). Computed with one n/2-point complex transform.
// Returns 1 on success, 0 if n isn't a power of two.
int fft_real([]const double in, []Complex out);

// 'batch' independent length-n transforms stored back to back (row r at
// data[r*n]), spread over the thread pool. inverse != 0 runs ifft.
int fft_batch([]Complex data, unsigned long n, int inverse);
// fft_real of 'batch' rows: in[r*n ..] -> out[r*n ..].
int fft_real_batch([]const double in, []Complex out, unsigned long n);
// fft_real_batch in place: row r (data[r*n], n complex slots) holds its n
// real inputs packed in its first n/2 slots (as doubles), and gets its full
// spectrum. No second buffer; the vendor library isn't used for it.
int fft_real_batch_inplace([]Complex data, unsigned long n);
// Only the n/2 + 1 non-redundant bins (0 .. n/2) of fft_real, the rest being
// their conjugates: half the output to write. out holds n/2 + 1 values; in
// the batch, row r's bins start at out[r * (n/2 + 1)].
int fft_real_half([]const double in, []Complex out);
int fft_real_half_batch([]const double in, []Complex out, unsigned long n);

// Single precision: the functions above on Complex32 / float, with the same
// contracts (fft_f32.sc). Same algorithm and threading; about 1e-6 relative
// error instead of 1e-15, half the memory traffic. fft_prepare_f32 fills
// the float twiddle table (fft_prepare doesn't), up to 4096 points it is
// compile-time like the double one.
int fft_f32([]Complex32 data);
int ifft_f32([]Complex32 data);
int fft_serial_f32([]Complex32 data);
int ifft_serial_f32([]Complex32 data);
int fft_prepare_f32(unsigned long n);
int fft_real_f32([]const float in, []Complex32 out);
int fft_batch_f32([]Complex32 data, unsigned long n, int inverse);
int fft_real_batch_f32([]const float in, []Complex32 out, unsigned long n);
int fft_real_batch_inplace_f32([]Complex32 data, unsigned long n);
int fft_real_half_f32([]const float in, []Complex32 out);
int fft_real_half_batch_f32([]const float in, []Complex32 out, unsigned long n);

// Q8.24 fixed-point forward/inverse FFT, same in-place/power-of-two
// contract as the float versions above.
int ffft([]FComplex data);
int fifft([]FComplex data);

// Internal: the (reference, length) forms the slice API above wraps
// (std/dsp's own code calls these; not API).
int fft_n_(&Complex data, unsigned long n);
int ifft_n_(&Complex data, unsigned long n);
int fft_serial_n_(&Complex data, unsigned long n);
int ifft_serial_n_(&Complex data, unsigned long n);
int fft_f32_n_(&Complex32 data, unsigned long n);
int ifft_f32_n_(&Complex32 data, unsigned long n);
int fft_serial_f32_n_(&Complex32 data, unsigned long n);
int ifft_serial_f32_n_(&Complex32 data, unsigned long n);
int ffft_n_(&FComplex data, unsigned long n);
int fifft_n_(&FComplex data, unsigned long n);
int fft_real_n_(const &double in, &Complex out, unsigned long n);
int fft_batch_n_(&Complex data, unsigned long n, unsigned long batch, int inverse);
int fft_real_batch_n_(const &double in, &Complex out, unsigned long n, unsigned long batch);
int fft_real_batch_inplace_n_(&Complex data, unsigned long n, unsigned long batch);
int fft_real_half_n_(const &double in, &Complex out, unsigned long n);
int fft_real_half_batch_n_(const &double in, &Complex out, unsigned long n, unsigned long batch);
int fft_real_f32_n_(const &float in, &Complex32 out, unsigned long n);
int fft_batch_f32_n_(&Complex32 data, unsigned long n, unsigned long batch, int inverse);
int fft_real_batch_f32_n_(const &float in, &Complex32 out, unsigned long n, unsigned long batch);
int fft_real_batch_inplace_f32_n_(&Complex32 data, unsigned long n, unsigned long batch);
int fft_real_half_f32_n_(const &float in, &Complex32 out, unsigned long n);
int fft_real_half_batch_f32_n_(const &float in, &Complex32 out, unsigned long n, unsigned long batch);

} // namespace std
