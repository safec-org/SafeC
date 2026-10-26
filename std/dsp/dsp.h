// SafeC Standard Library — DSP array primitives (Q8.24 fixed-point)
//
// Basic array-level building blocks (dot product, scaling, clipping,
// level metering) shared by the rest of std/dsp/'s real signal-processing
// modules — filter.h (FIR/IIR/biquad/bilinear transform), fft.h/dft.h
// (spectral transforms), convolution.h, window.h. Those live in separate
// files/headers (see dsp_all.h to pull in everything at once) since each
// is substantial enough to stand on its own.
#pragma once
#include <std/dsp/fixed.h>

// Dot product of two Fixed arrays of length n.
namespace std {

Fixed dsp_dot([]const Fixed a, []const Fixed b);

// Scale array in-place: a[i] *= scale.
void  dsp_scale([]Fixed a, Fixed scale);

// Add arrays element-wise: out[i] = a[i] + b[i].
unsigned long dsp_add([]const Fixed a, []const Fixed b, []Fixed out);

// Clip all values to [lo, hi].
void  dsp_clip([]Fixed a, Fixed lo, Fixed hi);

// Peak magnitude (maximum absolute value) in array.
Fixed dsp_peak([]const Fixed a);

// Root mean square value of array.
Fixed dsp_rms([]const Fixed a);

// SIMD FMA-accelerated dot product of two raw double arrays (see
// std/simd/simd.h) — processes 4 elements/iteration via simd_fma_f64x4,
// with a scalar remainder loop for n % 4 != 0. This is the float64
// counterpart to dsp_dot above (which stays scalar: Q8.24 fixed-point
// multiply needs a 64-bit widening shift per element that std::simd's
// integer FMA doesn't provide, so there's no safe SIMD win there). Used
// by convolution.sc's conv_direct as the core accumulation primitive
// once the kernel is reversed into a contiguous access pattern, and
// available directly for any other double-array dot product.
double dsp_dot_f64([]const double a, []const double b);

// The same in single precision, eight floats per vector (f32x8).
float dsp_dot_f32([]const float a, []const float b);

// out[q] = sum_t a[t] * x[q + t] for q < nOut (x holds nOut + n - 1 values):
// the dot product of a with every n-long window of x, as in an FIR filter
// (a = reversed taps) or a correlation. Many outputs per vector instead of
// one horizontal sum each, so for long runs it is several times faster than
// calling dsp_dot per output.
int dsp_sliding_dot_f64([]const double a, []const double x, []double out);
int dsp_sliding_dot_f32([]const float a, []const float x, []float out);

// Internal: the (reference, length) forms the slice API above wraps
// (std/dsp's own code calls these; not API).
Fixed dsp_dot_n_(const &Fixed a, const &Fixed b, unsigned long n);
void  dsp_scale_n_(&Fixed a, Fixed scale, unsigned long n);
void  dsp_add_n_(const &Fixed a, const &Fixed b, &Fixed out, unsigned long n);
void  dsp_clip_n_(&Fixed a, Fixed lo, Fixed hi, unsigned long n);
Fixed dsp_peak_n_(const &Fixed a, unsigned long n);
Fixed dsp_rms_n_(&Fixed a, unsigned long n);
double dsp_dot_f64_n_(const &double a, const &double b, unsigned long n);
float dsp_dot_f32_n_(const &float a, const &float b, unsigned long n);
void dsp_sliding_dot_f64_n_(const &double a, const &double x, &double out, unsigned long nOut, unsigned long n);
void dsp_sliding_dot_f32_n_(const &float a, const &float x, &float out, unsigned long nOut, unsigned long n);

} // namespace std
