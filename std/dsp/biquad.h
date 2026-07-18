// SafeC Standard Library — Biquad (2nd-order IIR section) + design tools
//
// struct Biquad is a single Direct-Form-II-Transposed 2nd-order section:
//   y[n] = b0*x[n] + z1
//   z1'  = b1*x[n] - a1*y[n] + z2
//   z2'  = b2*x[n] - a2*y[n]
// (a0 is always normalized to 1 — every coefficient below already has it
// divided out.) DF2T is the conventional choice for a single biquad: only
// two state variables, and better numerical behavior under coefficient
// quantization than Direct Form I. Cascade several for a higher-order
// filter (e.g. two biquads for a 4th-order response) — that composition
// isn't provided as a separate type here since it's just "call process()
// on each in sequence."
//
// biquad_lowpass/highpass/bandpass/notch/allpass/peaking/lowshelf/
// highshelf compute coefficients using Robert Bristow-Johnson's "Audio EQ
// Cookbook" formulas — the standard, widely-implemented reference for
// these particular filter shapes. bilinear_2nd_order is the more general
// tool underneath: given *any* analog (s-domain) 2nd-order prototype, it
// derives the matching digital biquad via the substitution
// s = 2*fs*(z-1)/(z+1) (the bilinear transform) — use it directly for an
// analog prototype the cookbook doesn't cover (e.g. a Butterworth/
// Chebyshev/Bessel design), the same way the cookbook formulas were
// themselves derived from an analog prototype internally.
#pragma once
#include <std/dsp/complex_dsp.h>

namespace std {

struct Biquad {
    double b0;
    double b1;
    double b2;
    double a1;
    double a2;
    double z1;
    double z2;

    double process(double x);
    void   reset();
};

struct Biquad biquad_new(double b0, double b1, double b2, double a1, double a2);

// fs = sample rate (Hz), f0 = cutoff/center frequency (Hz), q = Q factor
// (> 0; higher = narrower/more resonant).
struct Biquad biquad_lowpass(double fs, double f0, double q);
struct Biquad biquad_highpass(double fs, double f0, double q);
struct Biquad biquad_bandpass(double fs, double f0, double q);  // constant 0 dB peak gain
struct Biquad biquad_notch(double fs, double f0, double q);
struct Biquad biquad_allpass(double fs, double f0, double q);

// dbGain: boost (positive) or cut (negative) at f0, in decibels.
struct Biquad biquad_peaking(double fs, double f0, double q, double dbGain);

// s: shelf slope (1.0 = maximally steep without overshoot — the RBJ
// cookbook's own recommended default when no other constraint applies).
struct Biquad biquad_lowshelf(double fs, double f0, double s, double dbGain);
struct Biquad biquad_highshelf(double fs, double f0, double s, double dbGain);

// General analog-to-digital 2nd-order section via the bilinear transform.
// H(s) = (B2*s^2 + B1*s + B0) / (A2*s^2 + A1*s + A0) is the analog
// prototype; fs is the target digital sample rate. Frequencies in H(s)
// should already be pre-warped (w_analog = 2*fs*tan(w_digital/(2*fs)))
// by the caller if matching a specific digital cutoff matters — this
// function performs the algebraic substitution only, the same division
// of responsibility real bilinear-transform tools use.
struct Biquad bilinear_2nd_order(double B0, double B1, double B2,
                                  double A0, double A1, double A2, double fs);

// Z-domain frequency response of 'bq' at 'freqHz' (evaluates H(e^{j*omega}),
// omega = 2*pi*freqHz/fs) — magnitude (linear, not dB) and phase (radians).
double biquad_response_mag(const struct Biquad* bq, double fs, double freqHz);
double biquad_response_phase(const struct Biquad* bq, double fs, double freqHz);

// Q8.24 fixed-point biquad: coefficients are designed in double precision
// via the functions above, then quantized once with biquad_to_fixed —
// the same design-time-float/runtime-fixed split as the rest of std/dsp.
struct FBiquad {
    Fixed b0;
    Fixed b1;
    Fixed b2;
    Fixed a1;
    Fixed a2;
    Fixed z1;
    Fixed z2;

    Fixed process(Fixed x);
    void  reset();
};

struct FBiquad biquad_to_fixed(struct Biquad bq);

} // namespace std
