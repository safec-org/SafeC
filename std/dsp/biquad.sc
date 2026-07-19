// SafeC Standard Library — Biquad implementation (see biquad.h)
//
// biquad_lowpass/highpass/bandpass/notch/allpass/peaking/lowshelf/
// highshelf implement Robert Bristow-Johnson's "Audio EQ Cookbook"
// formulas verbatim. Correctness is checked in this module's own test
// suite via the well-known invariants each filter type must satisfy
// (unity DC gain for lowpass, unity Nyquist gain for highpass, exactly
// flat |H|=1 at every frequency for allpass, a peaking/shelf filter with
// 0 dB gain reducing to the identity, and the general bilinear
// transform's DC-gain-preservation property H_digital(z=1) = H_analog(s=0))
// rather than only spot-checking numeric coefficients.
#pragma once
#include <std/dsp/biquad.h>
#include <std/dsp/complex_dsp.h>
#include <std/math.h>

#define SAFEC_DSP_TWO_PI_B 6.28318530717958647692

namespace std {

inline struct Biquad biquad_new(double b0, double b1, double b2, double a1, double a2) {
    struct Biquad bq;
    bq.b0 = b0; bq.b1 = b1; bq.b2 = b2; bq.a1 = a1; bq.a2 = a2;
    bq.z1 = 0.0; bq.z2 = 0.0;
    return bq;
}

inline double Biquad::process(double x) {
    double y = self.b0 * x + self.z1;
    self.z1 = self.b1 * x - self.a1 * y + self.z2;
    self.z2 = self.b2 * x - self.a2 * y;
    return y;
}

inline void Biquad::reset() {
    self.z1 = 0.0;
    self.z2 = 0.0;
}

// Shared RBJ setup: w0, cos(w0), sin(w0), alpha (bandwidth form).
struct RbjCtx_ {
    double cosW0;
    double sinW0;
    double alpha;
};

static struct RbjCtx_ rbj_setup_(double fs, double f0, double q) {
    double w0 = SAFEC_DSP_TWO_PI_B * f0 / fs;
    struct RbjCtx_ ctx;
    ctx.cosW0 = cos_d(w0);
    ctx.sinW0 = sin_d(w0);
    ctx.alpha = ctx.sinW0 / (2.0 * q);
    return ctx;
}

struct Biquad biquad_lowpass(double fs, double f0, double q) {
    struct RbjCtx_ c = rbj_setup_(fs, f0, q);
    double a0 = 1.0 + c.alpha;
    double b0 = (1.0 - c.cosW0) / 2.0;
    double b1 = 1.0 - c.cosW0;
    double b2 = (1.0 - c.cosW0) / 2.0;
    double a1 = -2.0 * c.cosW0;
    double a2 = 1.0 - c.alpha;
    return biquad_new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
}

struct Biquad biquad_highpass(double fs, double f0, double q) {
    struct RbjCtx_ c = rbj_setup_(fs, f0, q);
    double a0 = 1.0 + c.alpha;
    double b0 = (1.0 + c.cosW0) / 2.0;
    double b1 = -(1.0 + c.cosW0);
    double b2 = (1.0 + c.cosW0) / 2.0;
    double a1 = -2.0 * c.cosW0;
    double a2 = 1.0 - c.alpha;
    return biquad_new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
}

struct Biquad biquad_bandpass(double fs, double f0, double q) {
    struct RbjCtx_ c = rbj_setup_(fs, f0, q);
    double a0 = 1.0 + c.alpha;
    double b0 = c.alpha;
    double b1 = 0.0;
    double b2 = -c.alpha;
    double a1 = -2.0 * c.cosW0;
    double a2 = 1.0 - c.alpha;
    return biquad_new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
}

struct Biquad biquad_notch(double fs, double f0, double q) {
    struct RbjCtx_ c = rbj_setup_(fs, f0, q);
    double a0 = 1.0 + c.alpha;
    double b0 = 1.0;
    double b1 = -2.0 * c.cosW0;
    double b2 = 1.0;
    double a1 = -2.0 * c.cosW0;
    double a2 = 1.0 - c.alpha;
    return biquad_new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
}

struct Biquad biquad_allpass(double fs, double f0, double q) {
    struct RbjCtx_ c = rbj_setup_(fs, f0, q);
    double a0 = 1.0 + c.alpha;
    double b0 = 1.0 - c.alpha;
    double b1 = -2.0 * c.cosW0;
    double b2 = 1.0 + c.alpha;
    double a1 = -2.0 * c.cosW0;
    double a2 = 1.0 - c.alpha;
    return biquad_new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
}

struct Biquad biquad_peaking(double fs, double f0, double q, double dbGain) {
    double A = pow_d(10.0, dbGain / 40.0);
    struct RbjCtx_ c = rbj_setup_(fs, f0, q);
    double a0 = 1.0 + c.alpha / A;
    double b0 = 1.0 + c.alpha * A;
    double b1 = -2.0 * c.cosW0;
    double b2 = 1.0 - c.alpha * A;
    double a1 = -2.0 * c.cosW0;
    double a2 = 1.0 - c.alpha / A;
    return biquad_new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
}

struct Biquad biquad_lowshelf(double fs, double f0, double s, double dbGain) {
    double A = pow_d(10.0, dbGain / 40.0);
    double w0 = SAFEC_DSP_TWO_PI_B * f0 / fs;
    double cosW0 = cos_d(w0);
    double sinW0 = sin_d(w0);
    double alpha = sinW0 / 2.0 * sqrt_d((A + 1.0 / A) * (1.0 / s - 1.0) + 2.0);
    double sqrtA = sqrt_d(A);
    double a0 =        (A + 1.0) + (A - 1.0) * cosW0 + 2.0 * sqrtA * alpha;
    double b0 =   A * ((A + 1.0) - (A - 1.0) * cosW0 + 2.0 * sqrtA * alpha);
    double b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cosW0);
    double b2 =   A * ((A + 1.0) - (A - 1.0) * cosW0 - 2.0 * sqrtA * alpha);
    double a1 =  -2.0 * ((A - 1.0) + (A + 1.0) * cosW0);
    double a2 =        (A + 1.0) + (A - 1.0) * cosW0 - 2.0 * sqrtA * alpha;
    return biquad_new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
}

struct Biquad biquad_highshelf(double fs, double f0, double s, double dbGain) {
    double A = pow_d(10.0, dbGain / 40.0);
    double w0 = SAFEC_DSP_TWO_PI_B * f0 / fs;
    double cosW0 = cos_d(w0);
    double sinW0 = sin_d(w0);
    double alpha = sinW0 / 2.0 * sqrt_d((A + 1.0 / A) * (1.0 / s - 1.0) + 2.0);
    double sqrtA = sqrt_d(A);
    double a0 =        (A + 1.0) - (A - 1.0) * cosW0 + 2.0 * sqrtA * alpha;
    double b0 =   A * ((A + 1.0) + (A - 1.0) * cosW0 + 2.0 * sqrtA * alpha);
    double b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cosW0);
    double b2 =   A * ((A + 1.0) + (A - 1.0) * cosW0 - 2.0 * sqrtA * alpha);
    double a1 =   2.0 * ((A - 1.0) - (A + 1.0) * cosW0);
    double a2 =        (A + 1.0) - (A - 1.0) * cosW0 - 2.0 * sqrtA * alpha;
    return biquad_new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
}

struct Biquad bilinear_2nd_order(double B0, double B1, double B2,
                                         double A0, double A1, double A2, double fs) {
    double k = 2.0 * fs;
    double k2 = k * k;

    double nz2 = B2 * k2 + B1 * k + B0;
    double nz1 = 2.0 * (B0 - B2 * k2);
    double nz0 = B2 * k2 - B1 * k + B0;

    double dz2 = A2 * k2 + A1 * k + A0;
    double dz1 = 2.0 * (A0 - A2 * k2);
    double dz0 = A2 * k2 - A1 * k + A0;

    return biquad_new(nz2 / dz2, nz1 / dz2, nz0 / dz2, dz1 / dz2, dz0 / dz2);
}

// H(e^{j*omega}) itself — shared by biquad_response_mag/phase below so the
// num/den evaluation (and the addressable-temporary decomposition it needs;
// SafeC operator calls require an lvalue receiver, so each intermediate
// product/sum is bound to a named local rather than chained inline) exists
// in exactly one place.
static struct Complex biquad_freq_response_(const &stack Biquad bq, double fs, double freqHz) {
    double omega = SAFEC_DSP_TWO_PI_B * freqHz / fs;
    struct Complex zInv = complex_new(cos_d(-omega), sin_d(-omega));
    struct Complex zInv2 = zInv * zInv;
    double b0 = bq.b0; double b1 = bq.b1; double b2 = bq.b2; double a1 = bq.a1; double a2 = bq.a2;

    struct Complex cb0 = complex_new(b0, 0.0);
    struct Complex cb1 = complex_new(b1, 0.0);
    struct Complex cb2 = complex_new(b2, 0.0);
    struct Complex numT1 = cb1 * zInv;
    struct Complex numT2 = cb2 * zInv2;
    struct Complex numSum1 = cb0 + numT1;
    struct Complex num = numSum1 + numT2;

    struct Complex ca0 = complex_new(1.0, 0.0);
    struct Complex ca1 = complex_new(a1, 0.0);
    struct Complex ca2 = complex_new(a2, 0.0);
    struct Complex denT1 = ca1 * zInv;
    struct Complex denT2 = ca2 * zInv2;
    struct Complex denSum1 = ca0 + denT1;
    struct Complex den = denSum1 + denT2;

    return num / den;
}

double biquad_response_mag(const &stack Biquad bq, double fs, double freqHz) {
    struct Complex h = biquad_freq_response_(bq, fs, freqHz);
    return h.abs();
}

double biquad_response_phase(const &stack Biquad bq, double fs, double freqHz) {
    struct Complex h = biquad_freq_response_(bq, fs, freqHz);
    return h.arg();
}

inline Fixed FBiquad::process(Fixed x) {
    Fixed y = fixed_add(fixed_mul(self.b0, x), self.z1);
    self.z1 = fixed_add(fixed_sub(fixed_mul(self.b1, x), fixed_mul(self.a1, y)), self.z2);
    self.z2 = fixed_sub(fixed_mul(self.b2, x), fixed_mul(self.a2, y));
    return y;
}

inline void FBiquad::reset() {
    self.z1 = (Fixed)0;
    self.z2 = (Fixed)0;
}

struct FBiquad biquad_to_fixed(struct Biquad bq) {
    struct FBiquad fbq;
    fbq.b0 = fixed_from_float(bq.b0);
    fbq.b1 = fixed_from_float(bq.b1);
    fbq.b2 = fixed_from_float(bq.b2);
    fbq.a1 = fixed_from_float(bq.a1);
    fbq.a2 = fixed_from_float(bq.a2);
    fbq.z1 = (Fixed)0;
    fbq.z2 = (Fixed)0;
    return fbq;
}

} // namespace std
