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
#include <std/simd/cpu.h>
#include <std/simd/simd.h>
#include <std/dsp/biquad.h>
#include <std/dsp/complex_dsp.h>
#include <std/math.h>

#define SAFEC_DSP_TWO_PI_B 6.28318530717958647692

namespace std {

typedef vec<float, 2> bq_f32x2_;

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

// The whole block with the state in registers. The recurrence is
// latency-bound, so the block runs direct form I (one FMA per sample on the
// dependency chain, see the body) and, from the AVX2 level (std/simd/cpu.h),
// fused multiply-adds; results can differ from process() in the last bits.
// Allocation- and lock-free.
static void biquad_block_body_(double* st, const double* c, const double* x, double* y, unsigned long n)
    __attribute__((always_inline)) {
    unsafe {
        double b0 = c[0]; double b1 = c[1]; double b2 = c[2];
        double a1 = c[3]; double a2 = c[4];
        double z1 = st[0]; double z2 = st[1];
        // Direct form I from the fifth sample on:
        //   y[i] = -a1 y[i-1] + (b0 x[i] + b1 x[i-1] + b2 x[i-2] - a2 y[i-2])
        // rewritten with look-ahead below. The first four samples run
        // transposed form II from the state; the state is rebuilt from the
        // last two at the end.
        // Inputs and outputs live in registers, so in == out is fine.
        double x1 = 0.0; double x2 = 0.0; double x3 = 0.0; double x4 = 0.0;   // x[i-1] .. x[i-4]
        double y1 = 0.0; double y2 = 0.0; double y3 = 0.0; double y4 = 0.0;   // y[i-1] .. y[i-4]
        unsigned long i = 0UL;
        while (i < n && i < 4UL) {
            double xi = x[i];
            double yi = __builtin_fmuladd(b0, xi, z1);
            z1 = __builtin_fmuladd(-a1, yi, __builtin_fmuladd(b1, xi, z2));
            z2 = __builtin_fmuladd(-a2, yi, b2 * xi);
            y[i] = yi;
            x4 = x3; x3 = x2; x2 = x1; x1 = xi;
            y4 = y3; y3 = y2; y2 = y1; y1 = yi;
            i = i + 1UL;
        }
        if (n > 4UL) {
            // Scattered look-ahead (Parhi & Messerschmitt): multiplying the
            // recursion's denominator D(z) = 1 + a1 z^-1 + a2 z^-2 by D(-z)
            // leaves only even powers, so with p[i] = b0 x[i] + b1 x[i-1] +
            // b2 x[i-2] and q[i] = p[i] - a1 p[i-1] + a2 p[i-2]:
            //   y[i] = q[i] + (a1^2 - 2 a2) y[i-2] - a2^2 y[i-4]
            // The even and odd outputs are two independent chains (two
            // cycles a sample instead of four), and the added poles mirror
            // the filter's own (-p, the same radius), so it stays stable.
            double la = a1 * a1 - (double)2.0 * a2;
            double lb = -a2 * a2;
            double p1 = __builtin_fmuladd(b0, x1, __builtin_fmuladd(b1, x2, b2 * x3));   // p[i-1]
            double p2 = __builtin_fmuladd(b0, x2, __builtin_fmuladd(b1, x3, b2 * x4));   // p[i-2]
            // The two chains share each instruction: lanes (even, odd), two
            // outputs per step of one FMA latency.
            f64x2 vla; vla[0] = la; vla[1] = la;
            f64x2 vlb; vlb[0] = lb; vlb[1] = lb;
            f64x2 vb0; vb0[0] = b0; vb0[1] = b0;
            f64x2 vb1; vb1[0] = b1; vb1[1] = b1;
            f64x2 vb2; vb2[0] = b2; vb2[1] = b2;
            f64x2 va1; va1[0] = -a1; va1[1] = -a1;
            f64x2 va2; va2[0] = a2; va2[1] = a2;
            f64x2 Y1; Y1[0] = y2; Y1[1] = y1;       // y[i-2], y[i-1]
            f64x2 Y2; Y2[0] = y4; Y2[1] = y3;       // y[i-4], y[i-3]
            f64x2 P1; P1[0] = p2; P1[1] = p1;       // p[i-2], p[i-1]
            f64x2 Xp; Xp[0] = x2; Xp[1] = x1;       // x[i-2], x[i-1]
            while (i + 2UL <= n) {
                f64x2 X0; X0[0] = x[i]; X0[1] = x[i + 1UL];
                f64x2 Xm1; Xm1[0] = Xp[1]; Xm1[1] = X0[0];
                f64x2 P = __builtin_fmuladd(vb0, X0, __builtin_fmuladd(vb1, Xm1, vb2 * Xp));
                f64x2 Pm1; Pm1[0] = P1[1]; Pm1[1] = P[0];
                f64x2 Q = __builtin_fmuladd(va2, P1, __builtin_fmuladd(va1, Pm1, P));
                f64x2 Y = __builtin_fmuladd(vla, Y1, __builtin_fmuladd(vlb, Y2, Q));
                y[i] = Y[0];
                y[i + 1UL] = Y[1];
                Y2 = Y1; Y1 = Y; P1 = P; Xp = X0;
                i = i + 2UL;
            }
            x2 = Xp[0]; x1 = Xp[1];
            y2 = Y1[0]; y1 = Y1[1];
            if (i < n) {   // one left: direct form I
                double xi = x[i];
                double yi = __builtin_fmuladd(-a1, y1, __builtin_fmuladd(-a2, y2, __builtin_fmuladd(b0, xi, __builtin_fmuladd(b1, x1, b2 * x2))));
                y[i] = yi;
                x2 = x1; x1 = xi; y2 = y1; y1 = yi;
                i = i + 1UL;
            }
            z1 = __builtin_fmuladd(b1, x1, __builtin_fmuladd(b2, x2, __builtin_fmuladd(-a1, y1, -a2 * y2)));
            z2 = __builtin_fmuladd(b2, x1, -a2 * y1);
        }
        st[0] = z1;
        st[1] = z2;
    }
}

static void biquad_block_base_(double* st, const double* c, const double* x, double* y, unsigned long n) {
    biquad_block_body_(st, c, x, y, n);
}
#if defined(__x86_64__)
static void biquad_block_fma_(double* st, const double* c, const double* x, double* y, unsigned long n)
    __attribute__((target("avx2,fma"))) {
    biquad_block_body_(st, c, x, y, n);
}
#endif

void biquad_process_block(&Biquad bq, const &double in, &double out, unsigned long n) {
    double c[5];
    c[0] = bq.b0; c[1] = bq.b1; c[2] = bq.b2; c[3] = bq.a1; c[4] = bq.a2;
    double st[2];
    st[0] = bq.z1; st[1] = bq.z2;
    unsafe {
#if defined(__x86_64__)
        if (simd_level() >= SIMD_AVX2) biquad_block_fma_(&st[0], (const double*)&c[0], (const double*)in, (double*)out, n);
        else
#endif
        biquad_block_base_(&st[0], (const double*)&c[0], (const double*)in, (double*)out, n);
    }
    bq.z1 = st[0];
    bq.z2 = st[1];
}

inline void Biquad::reset() {
    self.z1 = 0.0;
    self.z2 = 0.0;
}

// ── Single precision ────────────────────────────────────────────────────────
// Biquad32: the same filter on floats; design in double, then biquad_to_f32.

inline struct Biquad32 biquad_new_f32(float b0, float b1, float b2, float a1, float a2) {
    struct Biquad32 bq;
    bq.b0 = b0; bq.b1 = b1; bq.b2 = b2; bq.a1 = a1; bq.a2 = a2;
    bq.z1 = 0.0; bq.z2 = 0.0;
    return bq;
}

inline float Biquad32::process(float x) {
    float y = self.b0 * x + self.z1;
    self.z1 = self.b1 * x - self.a1 * y + self.z2;
    self.z2 = self.b2 * x - self.a2 * y;
    return y;
}

static void biquad32_block_body_(float* st, const float* c, const float* x, float* y, unsigned long n)
    __attribute__((always_inline)) {
    unsafe {
        float b0 = c[0]; float b1 = c[1]; float b2 = c[2];
        float a1 = c[3]; float a2 = c[4];
        float z1 = st[0]; float z2 = st[1];
        // Direct form I from the fifth sample on:
        //   y[i] = -a1 y[i-1] + (b0 x[i] + b1 x[i-1] + b2 x[i-2] - a2 y[i-2])
        // rewritten with look-ahead below. The first four samples run
        // transposed form II from the state; the state is rebuilt from the
        // last two at the end.
        // Inputs and outputs live in registers, so in == out is fine.
        float x1 = (float)0.0; float x2 = (float)0.0; float x3 = (float)0.0; float x4 = (float)0.0;   // x[i-1] .. x[i-4]
        float y1 = (float)0.0; float y2 = (float)0.0; float y3 = (float)0.0; float y4 = (float)0.0;   // y[i-1] .. y[i-4]
        unsigned long i = 0UL;
        while (i < n && i < 4UL) {
            float xi = x[i];
            float yi = __builtin_fmuladd(b0, xi, z1);
            z1 = __builtin_fmuladd(-a1, yi, __builtin_fmuladd(b1, xi, z2));
            z2 = __builtin_fmuladd(-a2, yi, b2 * xi);
            y[i] = yi;
            x4 = x3; x3 = x2; x2 = x1; x1 = xi;
            y4 = y3; y3 = y2; y2 = y1; y1 = yi;
            i = i + 1UL;
        }
        if (n > 4UL) {
            // Scattered look-ahead (Parhi & Messerschmitt): multiplying the
            // recursion's denominator D(z) = 1 + a1 z^-1 + a2 z^-2 by D(-z)
            // leaves only even powers, so with p[i] = b0 x[i] + b1 x[i-1] +
            // b2 x[i-2] and q[i] = p[i] - a1 p[i-1] + a2 p[i-2]:
            //   y[i] = q[i] + (a1^2 - 2 a2) y[i-2] - a2^2 y[i-4]
            // The even and odd outputs are two independent chains (two
            // cycles a sample instead of four), and the added poles mirror
            // the filter's own (-p, the same radius), so it stays stable.
            float la = a1 * a1 - (float)2.0 * a2;
            float lb = -a2 * a2;
            float p1 = __builtin_fmuladd(b0, x1, __builtin_fmuladd(b1, x2, b2 * x3));   // p[i-1]
            float p2 = __builtin_fmuladd(b0, x2, __builtin_fmuladd(b1, x3, b2 * x4));   // p[i-2]
            // The two chains share each instruction: lanes (even, odd), two
            // outputs per step of one FMA latency.
            bq_f32x2_ vla; vla[0] = la; vla[1] = la;
            bq_f32x2_ vlb; vlb[0] = lb; vlb[1] = lb;
            bq_f32x2_ vb0; vb0[0] = b0; vb0[1] = b0;
            bq_f32x2_ vb1; vb1[0] = b1; vb1[1] = b1;
            bq_f32x2_ vb2; vb2[0] = b2; vb2[1] = b2;
            bq_f32x2_ va1; va1[0] = -a1; va1[1] = -a1;
            bq_f32x2_ va2; va2[0] = a2; va2[1] = a2;
            bq_f32x2_ Y1; Y1[0] = y2; Y1[1] = y1;       // y[i-2], y[i-1]
            bq_f32x2_ Y2; Y2[0] = y4; Y2[1] = y3;       // y[i-4], y[i-3]
            bq_f32x2_ P1; P1[0] = p2; P1[1] = p1;       // p[i-2], p[i-1]
            bq_f32x2_ Xp; Xp[0] = x2; Xp[1] = x1;       // x[i-2], x[i-1]
            while (i + 2UL <= n) {
                bq_f32x2_ X0; X0[0] = x[i]; X0[1] = x[i + 1UL];
                bq_f32x2_ Xm1; Xm1[0] = Xp[1]; Xm1[1] = X0[0];
                bq_f32x2_ P = __builtin_fmuladd(vb0, X0, __builtin_fmuladd(vb1, Xm1, vb2 * Xp));
                bq_f32x2_ Pm1; Pm1[0] = P1[1]; Pm1[1] = P[0];
                bq_f32x2_ Q = __builtin_fmuladd(va2, P1, __builtin_fmuladd(va1, Pm1, P));
                bq_f32x2_ Y = __builtin_fmuladd(vla, Y1, __builtin_fmuladd(vlb, Y2, Q));
                y[i] = Y[0];
                y[i + 1UL] = Y[1];
                Y2 = Y1; Y1 = Y; P1 = P; Xp = X0;
                i = i + 2UL;
            }
            x2 = Xp[0]; x1 = Xp[1];
            y2 = Y1[0]; y1 = Y1[1];
            if (i < n) {   // one left: direct form I
                float xi = x[i];
                float yi = __builtin_fmuladd(-a1, y1, __builtin_fmuladd(-a2, y2, __builtin_fmuladd(b0, xi, __builtin_fmuladd(b1, x1, b2 * x2))));
                y[i] = yi;
                x2 = x1; x1 = xi; y2 = y1; y1 = yi;
                i = i + 1UL;
            }
            z1 = __builtin_fmuladd(b1, x1, __builtin_fmuladd(b2, x2, __builtin_fmuladd(-a1, y1, -a2 * y2)));
            z2 = __builtin_fmuladd(b2, x1, -a2 * y1);
        }
        st[0] = z1;
        st[1] = z2;
    }
}

static void biquad32_block_base_(float* st, const float* c, const float* x, float* y, unsigned long n) {
    biquad32_block_body_(st, c, x, y, n);
}
#if defined(__x86_64__)
static void biquad32_block_fma_(float* st, const float* c, const float* x, float* y, unsigned long n)
    __attribute__((target("avx2,fma"))) {
    biquad32_block_body_(st, c, x, y, n);
}
#endif

void biquad_process_block_f32(&Biquad32 bq, const &float in, &float out, unsigned long n) {
    float c[5];
    c[0] = bq.b0; c[1] = bq.b1; c[2] = bq.b2; c[3] = bq.a1; c[4] = bq.a2;
    float st[2];
    st[0] = bq.z1; st[1] = bq.z2;
    unsafe {
#if defined(__x86_64__)
        if (simd_level() >= SIMD_AVX2) biquad32_block_fma_(&st[0], (const float*)&c[0], (const float*)in, (float*)out, n);
        else
#endif
        biquad32_block_base_(&st[0], (const float*)&c[0], (const float*)in, (float*)out, n);
    }
    bq.z1 = st[0];
    bq.z2 = st[1];
}

inline void Biquad32::reset() {
    self.z1 = 0.0;
    self.z2 = 0.0;
}

inline struct Biquad32 biquad_to_f32(struct Biquad bq) {
    struct Biquad32 r = biquad_new_f32((float)bq.b0, (float)bq.b1, (float)bq.b2, (float)bq.a1, (float)bq.a2);
    r.z1 = (float)bq.z1;
    r.z2 = (float)bq.z2;
    return r;
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
