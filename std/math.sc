// SafeC Standard Library — Math
// Integer utilities are pure const functions eligible for compile-time evaluation.
// Float/double utilities wrap the corresponding <math.h> functions.
#pragma once
#include "math.h"
#include <math.h>

// ── Constants ─────────────────────────────────────────────────────────────────

const double PI_D()    { return 3.141592653589793238462643383; }
const double E_D()     { return 2.718281828459045235360287471; }
const double LN2_D()   { return 0.693147180559945309417232121; }
const double LN10_D()  { return 2.302585092994045684017991455; }
const double SQRT2_D() { return 1.414213562373095048801688724; }
const double INF_D()   { unsafe { return (double)(1.0 / 0.0); } }

const float  PI_F()    { return (float)3.14159265358979323846; }
const float  E_F()     { return (float)2.71828182845904523536; }
const float  SQRT2_F() { return (float)1.41421356237309504880; }

// ── Integer ───────────────────────────────────────────────────────────────────

const int abs_int(int x) {
    return x < 0 ? -x : x;
}

const long long abs_ll(long long x) {
    return x < 0 ? -x : x;
}

const int min_int(int a, int b) {
    return a < b ? a : b;
}

const int max_int(int a, int b) {
    return a > b ? a : b;
}

const long long min_ll(long long a, long long b) {
    return a < b ? a : b;
}

const long long max_ll(long long a, long long b) {
    return a > b ? a : b;
}

const int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

const long long clamp_ll(long long v, long long lo, long long hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ── Single-precision float ────────────────────────────────────────────────────

float abs_f(float x)                      { unsafe { return fabsf(x); } }
float sqrt_f(float x)                     { unsafe { return sqrtf(x); } }
float cbrt_f(float x)                     { unsafe { return cbrtf(x); } }
float floor_f(float x)                    { unsafe { return floorf(x); } }
float ceil_f(float x)                     { unsafe { return ceilf(x); } }
float round_f(float x)                    { unsafe { return roundf(x); } }
float trunc_f(float x)                    { unsafe { return truncf(x); } }
float pow_f(float base, float exp)        { unsafe { return powf(base, exp); } }
float exp_f(float x)                      { unsafe { return expf(x); } }
float exp2_f(float x)                     { unsafe { return exp2f(x); } }
float log_f(float x)                      { unsafe { return logf(x); } }
float log2_f(float x)                     { unsafe { return log2f(x); } }
float log10_f(float x)                    { unsafe { return log10f(x); } }
float sin_f(float x)                      { unsafe { return sinf(x); } }
float cos_f(float x)                      { unsafe { return cosf(x); } }
float tan_f(float x)                      { unsafe { return tanf(x); } }
float asin_f(float x)                     { unsafe { return asinf(x); } }
float acos_f(float x)                     { unsafe { return acosf(x); } }
float atan_f(float x)                     { unsafe { return atanf(x); } }
float atan2_f(float y, float x)           { unsafe { return atan2f(y, x); } }
float sinh_f(float x)                     { unsafe { return sinhf(x); } }
float cosh_f(float x)                     { unsafe { return coshf(x); } }
float tanh_f(float x)                     { unsafe { return tanhf(x); } }
float hypot_f(float x, float y)           { unsafe { return hypotf(x, y); } }
float fmod_f(float x, float y)            { unsafe { return fmodf(x, y); } }
float copysign_f(float mag, float sgn)    { unsafe { return copysignf(mag, sgn); } }
float fma_f(float a, float b, float c)    { unsafe { return fmaf(a, b, c); } }

float min_f(float a, float b)             { return a < b ? a : b; }
float max_f(float a, float b)             { return a > b ? a : b; }
float clamp_f(float v, float lo, float hi){ return v < lo ? lo : (v > hi ? hi : v); }

// IEEE 754: NaN is the only value not equal to itself.
int isnan_f(float x)    { return x != x; }
// IEEE 754: inf * 2 == inf (and != 0), finite * 2 != itself.
int isinf_f(float x)    { return x != (float)0 && x + x == x; }
// Finite iff neither NaN nor inf.
int isfinite_f(float x) { return !isnan_f(x) && !isinf_f(x); }

// ── Double-precision double ───────────────────────────────────────────────────

double abs_d(double x)                        { unsafe { return fabs(x); } }
double sqrt_d(double x)                       { unsafe { return sqrt(x); } }
double cbrt_d(double x)                       { unsafe { return cbrt(x); } }
double floor_d(double x)                      { unsafe { return floor(x); } }
double ceil_d(double x)                       { unsafe { return ceil(x); } }
double round_d(double x)                      { unsafe { return round(x); } }
double trunc_d(double x)                      { unsafe { return trunc(x); } }
double pow_d(double base, double exp)         { unsafe { return pow(base, exp); } }
double exp_d(double x)                        { unsafe { return exp(x); } }
double exp2_d(double x)                       { unsafe { return exp2(x); } }
double log_d(double x)                        { unsafe { return log(x); } }
double log2_d(double x)                       { unsafe { return log2(x); } }
double log10_d(double x)                      { unsafe { return log10(x); } }
double sin_d(double x)                        { unsafe { return sin(x); } }
double cos_d(double x)                        { unsafe { return cos(x); } }
double tan_d(double x)                        { unsafe { return tan(x); } }
double asin_d(double x)                       { unsafe { return asin(x); } }
double acos_d(double x)                       { unsafe { return acos(x); } }
double atan_d(double x)                       { unsafe { return atan(x); } }
double atan2_d(double y, double x)            { unsafe { return atan2(y, x); } }
double sinh_d(double x)                       { unsafe { return sinh(x); } }
double cosh_d(double x)                       { unsafe { return cosh(x); } }
double tanh_d(double x)                       { unsafe { return tanh(x); } }
double hypot_d(double x, double y)            { unsafe { return hypot(x, y); } }
double fmod_d(double x, double y)             { unsafe { return fmod(x, y); } }
double copysign_d(double mag, double sgn)     { unsafe { return copysign(mag, sgn); } }
double fma_d(double a, double b, double c)    { unsafe { return fma(a, b, c); } }

double min_d(double a, double b)              { return a < b ? a : b; }
double max_d(double a, double b)              { return a > b ? a : b; }
double clamp_d(double v, double lo, double hi){ return v < lo ? lo : (v > hi ? hi : v); }

int isnan_d(double x)    { return x != x; }
int isinf_d(double x)    { return x != 0.0 && x + x == x; }
int isfinite_d(double x) { return !isnan_d(x) && !isinf_d(x); }
