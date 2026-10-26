// SafeC Standard Library — Convolution
//
// Linear (non-circular) convolution: y[n] = sum_k x[k] * h[n-k], output
// length len_x + len_h - 1. Two implementations with the same output for
// the same input, trading off differently:
//   conv_direct / fconv_direct — O(len_x * len_h), no allocation, exact
//     (fconv_direct: exact up to Q8.24 rounding). Best for short filters
//     (e.g. convolving with a small FIR kernel).
//   conv_fft — O(n log n) via fft.h, allocates a scratch buffer sized to
//     the next power of two >= output length. Best for long inputs/long
//     kernels, where direct convolution's O(len_x*len_h) cost dominates.
#pragma once
#include <std/dsp/fixed.h>

namespace std {

// out must have space for (len_x + len_h - 1) samples.
int conv_direct([]const double x, []const double h, []double out);

int fconv_direct([]const Fixed x, []const Fixed h, []Fixed out);

// FFT-accelerated linear convolution — same output as conv_direct (up to
// floating-point rounding), computed via zero-padded forward FFTs,
// pointwise complex multiply, and one inverse FFT. out must have space
// for (len_x + len_h - 1) samples.
int conv_fft([]const double x, []const double h, []double out);

// Single precision: conv_direct / conv_fft on floats, same contracts.
int conv_direct_f32([]const float x, []const float h, []float out);
int conv_fft_f32([]const float x, []const float h, []float out);

// Internal: the (reference, length) forms the slice API above wraps
// (std/dsp's own code calls these; not API).
void conv_direct_n_(const &double x, unsigned long len_x,
                  const &double h, unsigned long len_h, &double out);
void conv_fft_n_(const &double x, unsigned long len_x,
              const &double h, unsigned long len_h, &double out);
void conv_direct_f32_n_(const &float x, unsigned long len_x,
                     const &float h, unsigned long len_h, &float out);
void conv_fft_f32_n_(const &float x, unsigned long len_x,
                  const &float h, unsigned long len_h, &float out);
void fconv_direct_n_(const &Fixed x, unsigned long len_x,
                   const &Fixed h, unsigned long len_h, &Fixed out);

} // namespace std
