// SafeC Standard Library — complex number implementation
// Represented as float[2] or double[2]: [0]=real, [1]=imag.
// Uses math.h functions for transcendentals.
#include "complex.h"
#include "math.h"

// ── Construction / access ─────────────────────────────────────────────────────
void  cmplx_f(float* out, float re, float im) { out[0] = re; out[1] = im; }
void  cmplx_d(double* out, double re, double im) { out[0] = re; out[1] = im; }
float  creal_f(const float* z)  { return z[0]; }
double creal_d(const double* z) { return z[0]; }
float  cimag_f(const float* z)  { return z[1]; }
double cimag_d(const double* z) { return z[1]; }

// ── Magnitude / angle ─────────────────────────────────────────────────────────
float  cabs_f(const float* z)  { return hypot_f(z[0], z[1]); }
double cabs_d(const double* z) { return hypot_d(z[0], z[1]); }
float  carg_f(const float* z)  { return atan2_f(z[1], z[0]); }
double carg_d(const double* z) { return atan2_d(z[1], z[0]); }

// ── Arithmetic ────────────────────────────────────────────────────────────────
void cadd_f(float* out, const float* a, const float* b) {
    out[0] = a[0] + b[0]; out[1] = a[1] + b[1];
}
void cadd_d(double* out, const double* a, const double* b) {
    out[0] = a[0] + b[0]; out[1] = a[1] + b[1];
}
void csub_f(float* out, const float* a, const float* b) {
    out[0] = a[0] - b[0]; out[1] = a[1] - b[1];
}
void csub_d(double* out, const double* a, const double* b) {
    out[0] = a[0] - b[0]; out[1] = a[1] - b[1];
}
void cmul_f(float* out, const float* a, const float* b) {
    out[0] = a[0]*b[0] - a[1]*b[1];
    out[1] = a[0]*b[1] + a[1]*b[0];
}
void cmul_d(double* out, const double* a, const double* b) {
    out[0] = a[0]*b[0] - a[1]*b[1];
    out[1] = a[0]*b[1] + a[1]*b[0];
}
void cdiv_f(float* out, const float* a, const float* b) {
    float denom = b[0]*b[0] + b[1]*b[1];
    out[0] = (a[0]*b[0] + a[1]*b[1]) / denom;
    out[1] = (a[1]*b[0] - a[0]*b[1]) / denom;
}
void cdiv_d(double* out, const double* a, const double* b) {
    double denom = b[0]*b[0] + b[1]*b[1];
    out[0] = (a[0]*b[0] + a[1]*b[1]) / denom;
    out[1] = (a[1]*b[0] - a[0]*b[1]) / denom;
}
void cneg_f(float* out, const float* z)   { out[0] = -z[0]; out[1] = -z[1]; }
void cneg_d(double* out, const double* z) { out[0] = -z[0]; out[1] = -z[1]; }
void cconj_f(float* out, const float* z)  { out[0] = z[0]; out[1] = -z[1]; }
void cconj_d(double* out, const double* z){ out[0] = z[0]; out[1] = -z[1]; }

// ── Transcendental ────────────────────────────────────────────────────────────
// sqrt(z) = sqrt(|z|) * (cos(arg/2) + i*sin(arg/2))
void csqrt_f(float* out, const float* z) {
    float r = cabs_f(z);
    float half_arg = carg_f(z) / (float)2.0;
    float sr = sqrt_f(r);
    out[0] = sr * cos_f(half_arg);
    out[1] = sr * sin_f(half_arg);
}
void csqrt_d(double* out, const double* z) {
    double r = cabs_d(z);
    double half_arg = carg_d(z) / 2.0;
    double sr = sqrt_d(r);
    out[0] = sr * cos_d(half_arg);
    out[1] = sr * sin_d(half_arg);
}
// exp(z) = e^re * (cos(im) + i*sin(im))
void cexp_f(float* out, const float* z) {
    float er = exp_f(z[0]);
    out[0] = er * cos_f(z[1]);
    out[1] = er * sin_f(z[1]);
}
void cexp_d(double* out, const double* z) {
    double er = exp_d(z[0]);
    out[0] = er * cos_d(z[1]);
    out[1] = er * sin_d(z[1]);
}
// log(z) = log(|z|) + i*arg(z)
void clog_f(float* out, const float* z) {
    out[0] = log_f(cabs_f(z));
    out[1] = carg_f(z);
}
void clog_d(double* out, const double* z) {
    out[0] = log_d(cabs_d(z));
    out[1] = carg_d(z);
}
// pow(base, exp) = exp(exp * log(base))
void cpow_f(float* out, const float* base, const float* ex) {
    float lb[2]; clog_f(lb, base);
    float tmp[2]; cmul_f(tmp, ex, lb);
    cexp_f(out, tmp);
}
void cpow_d(double* out, const double* base, const double* ex) {
    double lb[2]; clog_d(lb, base);
    double tmp[2]; cmul_d(tmp, ex, lb);
    cexp_d(out, tmp);
}
// sin(z) = sin(re)*cosh(im) + i*cos(re)*sinh(im)
void csin_f(float* out, const float* z) {
    out[0] = sin_f(z[0]) * cosh_f(z[1]);
    out[1] = cos_f(z[0]) * sinh_f(z[1]);
}
void csin_d(double* out, const double* z) {
    out[0] = sin_d(z[0]) * cosh_d(z[1]);
    out[1] = cos_d(z[0]) * sinh_d(z[1]);
}
// cos(z) = cos(re)*cosh(im) - i*sin(re)*sinh(im)
void ccos_f(float* out, const float* z) {
    out[0] =  cos_f(z[0]) * cosh_f(z[1]);
    out[1] = -sin_f(z[0]) * sinh_f(z[1]);
}
void ccos_d(double* out, const double* z) {
    out[0] =  cos_d(z[0]) * cosh_d(z[1]);
    out[1] = -sin_d(z[0]) * sinh_d(z[1]);
}
