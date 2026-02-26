// SafeC Standard Library — DSP Primitives (Q16.16 fixed-point)
#pragma once
#include "fixed.h"

// Dot product of two Fixed arrays of length n.
Fixed dsp_dot(const Fixed* a, const Fixed* b, unsigned long n);

// Scale array in-place: a[i] *= scale.
void  dsp_scale(Fixed* a, Fixed scale, unsigned long n);

// Add arrays element-wise: out[i] = a[i] + b[i].
void  dsp_add(const Fixed* a, const Fixed* b, Fixed* out, unsigned long n);

// Moving average (causal, order `order`).
// state must point to `order` Fixed elements initialised to zero by the caller.
void  dsp_moving_avg(const Fixed* in, Fixed* out, unsigned long n,
                     Fixed* state, unsigned long order);

// First-order IIR low-pass filter.
// y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
// alpha is Q16.16.  prev_y is updated in-place on each call.
void  dsp_iir_lp(const Fixed* in, Fixed* out, unsigned long n,
                 Fixed alpha, &stack Fixed prev_y);

// Clip all values to [lo, hi].
void  dsp_clip(Fixed* a, Fixed lo, Fixed hi, unsigned long n);

// Peak magnitude (maximum absolute value) in array.
Fixed dsp_peak(const Fixed* a, unsigned long n);

// Root mean square value of array.
Fixed dsp_rms(const Fixed* a, unsigned long n);
