#pragma once
// SafeC Standard Library — Complex Numbers (C99 <complex.h>)
// Represented as two-element float/double arrays: [real, imag].
// All functions use the naming convention: cXXX_f (float), cXXX_d (double).

// ── Construction ─────────────────────────────────────────────────────────────
// Create complex from real+imag parts (pack into a 2-element array).
// NOTE: Pass a float[2] or double[2] out-array; result is written there.
void  cmplx_f(float* out, float re, float im);
void  cmplx_d(double* out, double re, double im);

// ── Access ───────────────────────────────────────────────────────────────────
float  creal_f(const float* z);
double creal_d(const double* z);
float  cimag_f(const float* z);
double cimag_d(const double* z);

// ── Magnitude / angle ────────────────────────────────────────────────────────
float  cabs_f(const float* z);   // |z| = sqrt(re²+im²)
double cabs_d(const double* z);
float  carg_f(const float* z);   // phase angle = atan2(im, re)
double carg_d(const double* z);

// ── Arithmetic ───────────────────────────────────────────────────────────────
void cadd_f(float* out, const float* a, const float* b);
void cadd_d(double* out, const double* a, const double* b);
void csub_f(float* out, const float* a, const float* b);
void csub_d(double* out, const double* a, const double* b);
void cmul_f(float* out, const float* a, const float* b);
void cmul_d(double* out, const double* a, const double* b);
void cdiv_f(float* out, const float* a, const float* b);
void cdiv_d(double* out, const double* a, const double* b);
void cneg_f(float* out, const float* z);
void cneg_d(double* out, const double* z);
void cconj_f(float* out, const float* z);  // complex conjugate
void cconj_d(double* out, const double* z);

// ── Transcendental ───────────────────────────────────────────────────────────
void csqrt_f(float* out, const float* z);
void csqrt_d(double* out, const double* z);
void cexp_f(float* out, const float* z);
void cexp_d(double* out, const double* z);
void clog_f(float* out, const float* z);
void clog_d(double* out, const double* z);
void cpow_f(float* out, const float* base, const float* exp);
void cpow_d(double* out, const double* base, const double* exp);
void csin_f(float* out, const float* z);
void csin_d(double* out, const double* z);
void ccos_f(float* out, const float* z);
void ccos_d(double* out, const double* z);
