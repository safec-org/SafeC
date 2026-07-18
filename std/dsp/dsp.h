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

Fixed dsp_dot(&stack Fixed a, &stack Fixed b, unsigned long n);

// Scale array in-place: a[i] *= scale.
void  dsp_scale(&stack Fixed a, Fixed scale, unsigned long n);

// Add arrays element-wise: out[i] = a[i] + b[i].
void  dsp_add(&stack Fixed a, &stack Fixed b, &stack Fixed out, unsigned long n);

// Clip all values to [lo, hi].
void  dsp_clip(&stack Fixed a, Fixed lo, Fixed hi, unsigned long n);

// Peak magnitude (maximum absolute value) in array.
Fixed dsp_peak(&stack Fixed a, unsigned long n);

// Root mean square value of array.
Fixed dsp_rms(&stack Fixed a, unsigned long n);

// SIMD FMA-accelerated dot product of two raw double arrays (see
// std/simd/simd.h) — processes 4 elements/iteration via simd_fma_f64x4,
// with a scalar remainder loop for n % 4 != 0. This is the float64
// counterpart to dsp_dot above (which stays scalar: Q8.24 fixed-point
// multiply needs a 64-bit widening shift per element that std::simd's
// integer FMA doesn't provide, so there's no safe SIMD win there). Used
// by convolution.sc's conv_direct as the core accumulation primitive
// once the kernel is reversed into a contiguous access pattern, and
// available directly for any other double-array dot product.
double dsp_dot_f64(const double* a, const double* b, unsigned long n);

} // namespace std
