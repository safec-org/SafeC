// SafeC Standard Library — Resampling (upsampling / downsampling)
//
// All three functions map an input buffer of 'inLen' samples onto an
// output buffer the caller has already sized to 'outLen' samples —
// outLen > inLen upsamples, outLen < inLen downsamples, the same
// function handles both directions (there's no separate "upsample" vs
// "downsample" entry point since the only real difference is the ratio).
//
//   resample_nearest — nearest-neighbor: cheapest, introduces the most
//     aliasing/imaging distortion. Fine for non-critical use (e.g. UI
//     waveform thumbnails).
//   resample_linear  — linear interpolation: cheap, smoother than
//     nearest-neighbor, still not band-limited (some aliasing/imaging
//     remains, but the leading source of low-order high-frequency error
//     that nearest-neighbor has is gone).
//   resample_sinc    — windowed-sinc (band-limited) interpolation: the
//     highest-quality option. When downsampling, the sinc kernel's cutoff
//     is automatically scaled down to the new (lower) Nyquist rate so it
//     also acts as the anti-aliasing filter, exactly like a proper
//     sample-rate converter needs. 'kernelHalfWidth' trades quality for
//     cost (a good default is 8-16 — the number of input samples on each
//     side of the interpolation point that contribute).
#pragma once
#include <std/dsp/fixed.h>

namespace std {

void resample_nearest(const double* in, unsigned long inLen, double* out, unsigned long outLen);
void resample_linear(const double* in, unsigned long inLen, double* out, unsigned long outLen);
void resample_sinc(const double* in, unsigned long inLen, double* out, unsigned long outLen,
                    unsigned long kernelHalfWidth);

// Fixed-point nearest/linear (straightforward, no transcendentals in the
// per-sample path). resample_sinc has no Q8.24 counterpart: a windowed
// sinc kernel needs per-tap sin/cos evaluation, and std/dsp's established
// design-time-float/runtime-fixed split doesn't apply cleanly here since
// the kernel itself depends on the (runtime) fractional source position,
// not just fixed design-time coefficients — so there's no fixed sinc
// variant, matching the same "if available" scoping used for
// dsp_dot_f64/SIMD in dsp.h.
void fresample_nearest(const Fixed* in, unsigned long inLen, Fixed* out, unsigned long outLen);
void fresample_linear(const Fixed* in, unsigned long inLen, Fixed* out, unsigned long outLen);

} // namespace std
