// SafeC Standard Library — DSP Primitives (Q8.24 fixed-point)
#pragma once
#include <std/dsp/fixed.h>

// Dot product of two Fixed arrays of length n.
namespace std {

Fixed dsp_dot(&stack Fixed a, &stack Fixed b, unsigned long n);

// Scale array in-place: a[i] *= scale.
void  dsp_scale(&stack Fixed a, Fixed scale, unsigned long n);

// Add arrays element-wise: out[i] = a[i] + b[i].
void  dsp_add(&stack Fixed a, &stack Fixed b, &stack Fixed out, unsigned long n);

// Moving average (causal, order `order`).
// state must point to `order` Fixed elements initialised to zero by the caller.
void  dsp_moving_avg(&stack Fixed in, &stack Fixed out, unsigned long n,
                     &stack Fixed state, unsigned long order);

// First-order IIR low-pass filter.
// y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
// alpha is Q8.24.  prev_y is updated in-place on each call.
void  dsp_iir_lp(&stack Fixed in, &stack Fixed out, unsigned long n,
                 Fixed alpha, &stack Fixed prev_y);

// Clip all values to [lo, hi].
void  dsp_clip(&stack Fixed a, Fixed lo, Fixed hi, unsigned long n);

// Peak magnitude (maximum absolute value) in array.
Fixed dsp_peak(&stack Fixed a, unsigned long n);

// Root mean square value of array.
Fixed dsp_rms(&stack Fixed a, unsigned long n);

} // namespace std
