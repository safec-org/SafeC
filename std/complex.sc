// SafeC Standard Library — complex number implementation
// Represented as float[2] or double[2]: [0]=real, [1]=imag.
// Uses math.h functions for transcendentals.
#pragma once
#include <std/complex.h>
#include <std/math.h>

// ── Construction / access ─────────────────────────────────────────────────────
namespace std {

void  cmplx_f(&float out, float re, float im) { unsafe { out[0] = re; out[1] = im; } }
void  cmplx_d(&double out, double re, double im) { unsafe { out[0] = re; out[1] = im; } }
float  creal_f(const &float z)  { unsafe { return z[0]; } }
double creal_d(const &double z) { unsafe { return z[0]; } }
float  cimag_f(const &float z)  { unsafe { return z[1]; } }
double cimag_d(const &double z) { unsafe { return z[1]; } }

// ── Magnitude / angle ─────────────────────────────────────────────────────────
float  cabs_f(const &float z)  { unsafe { return hypot_f(z[0], z[1]); } }
double cabs_d(const &double z) { unsafe { return hypot_d(z[0], z[1]); } }
float  carg_f(const &float z)  { unsafe { return atan2_f(z[1], z[0]); } }
double carg_d(const &double z) { unsafe { return atan2_d(z[1], z[0]); } }

// ── Arithmetic ────────────────────────────────────────────────────────────────
void cadd_f(&float out, const &float a, const &float b) { unsafe {
    out[0] = a[0] + b[0]; out[1] = a[1] + b[1];
} }
void cadd_d(&double out, const &double a, const &double b) { unsafe {
    out[0] = a[0] + b[0]; out[1] = a[1] + b[1];
} }
void csub_f(&float out, const &float a, const &float b) { unsafe {
    out[0] = a[0] - b[0]; out[1] = a[1] - b[1];
} }
void csub_d(&double out, const &double a, const &double b) { unsafe {
    out[0] = a[0] - b[0]; out[1] = a[1] - b[1];
} }
void cmul_f(&float out, const &float a, const &float b) { unsafe {
    float ar = a[0]; float ai = a[1]; float br = b[0]; float bi = b[1];
    out[0] = ar*br - ai*bi;
    out[1] = ar*bi + ai*br;
} }
void cmul_d(&double out, const &double a, const &double b) { unsafe {
    double ar = a[0]; double ai = a[1]; double br = b[0]; double bi = b[1];
    out[0] = ar*br - ai*bi;
    out[1] = ar*bi + ai*br;
} }
void cdiv_f(&float out, const &float a, const &float b) { unsafe {
    float ar = a[0]; float ai = a[1]; float br = b[0]; float bi = b[1];
    float denom = br*br + bi*bi;
    out[0] = (ar*br + ai*bi) / denom;
    out[1] = (ai*br - ar*bi) / denom;
} }
void cdiv_d(&double out, const &double a, const &double b) { unsafe {
    double ar = a[0]; double ai = a[1]; double br = b[0]; double bi = b[1];
    double denom = br*br + bi*bi;
    out[0] = (ar*br + ai*bi) / denom;
    out[1] = (ai*br - ar*bi) / denom;
} }
void cneg_f(&float out, const &float z)   { unsafe { out[0] = -z[0]; out[1] = -z[1]; } }
void cneg_d(&double out, const &double z) { unsafe { out[0] = -z[0]; out[1] = -z[1]; } }
void cconj_f(&float out, const &float z)  { unsafe { out[0] = z[0]; out[1] = -z[1]; } }
void cconj_d(&double out, const &double z){ unsafe { out[0] = z[0]; out[1] = -z[1]; } }

// ── Transcendental ────────────────────────────────────────────────────────────
// sqrt(z) = sqrt(|z|) * (cos(arg/2) + i*sin(arg/2))
void csqrt_f(&float out, const &float z) { unsafe {
    float r = cabs_f(z);
    float half_arg = carg_f(z) / (float)2.0;
    float sr = sqrt_f(r);
    out[0] = sr * cos_f(half_arg);
    out[1] = sr * sin_f(half_arg);
} }
void csqrt_d(&double out, const &double z) { unsafe {
    double r = cabs_d(z);
    double half_arg = carg_d(z) / 2.0;
    double sr = sqrt_d(r);
    out[0] = sr * cos_d(half_arg);
    out[1] = sr * sin_d(half_arg);
} }
// exp(z) = e^re * (cos(im) + i*sin(im))
void cexp_f(&float out, const &float z) { unsafe {
    float er = exp_f(z[0]);
    out[0] = er * cos_f(z[1]);
    out[1] = er * sin_f(z[1]);
} }
void cexp_d(&double out, const &double z) { unsafe {
    double er = exp_d(z[0]);
    out[0] = er * cos_d(z[1]);
    out[1] = er * sin_d(z[1]);
} }
// log(z) = log(|z|) + i*arg(z)
void clog_f(&float out, const &float z) { unsafe {
    float mag = cabs_f(z);
    float ang = carg_f(z);
    out[0] = log_f(mag);
    out[1] = ang;
} }
void clog_d(&double out, const &double z) { unsafe {
    double mag = cabs_d(z);
    double ang = carg_d(z);
    out[0] = log_d(mag);
    out[1] = ang;
} }
// pow(base, exp) = exp(exp * log(base))
void cpow_f(&float out, const &float base, const &float ex) { unsafe {
    float lb[2]; clog_f(lb, base);
    float tmp[2]; cmul_f(tmp, ex, lb);
    cexp_f(out, tmp);
} }
void cpow_d(&double out, const &double base, const &double ex) { unsafe {
    double lb[2]; clog_d(lb, base);
    double tmp[2]; cmul_d(tmp, ex, lb);
    cexp_d(out, tmp);
} }
// sin(z) = sin(re)*cosh(im) + i*cos(re)*sinh(im)
void csin_f(&float out, const &float z) { unsafe {
    float re = z[0]; float im = z[1];
    out[0] = sin_f(re) * cosh_f(im);
    out[1] = cos_f(re) * sinh_f(im);
} }
void csin_d(&double out, const &double z) { unsafe {
    double re = z[0]; double im = z[1];
    out[0] = sin_d(re) * cosh_d(im);
    out[1] = cos_d(re) * sinh_d(im);
} }
// cos(z) = cos(re)*cosh(im) - i*sin(re)*sinh(im)
void ccos_f(&float out, const &float z) { unsafe {
    float re = z[0]; float im = z[1];
    out[0] =  cos_f(re) * cosh_f(im);
    out[1] = -sin_f(re) * sinh_f(im);
} }
void ccos_d(&double out, const &double z) { unsafe {
    double re = z[0]; double im = z[1];
    out[0] =  cos_d(re) * cosh_d(im);
    out[1] = -sin_d(re) * sinh_d(im);
} }

} // namespace std
