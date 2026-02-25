// SafeC Standard Library — Math declarations
// Naming convention:
//   _int / _ll  → integer (const-evaluable at compile time)
//   _f          → single-precision float  (wraps <math.h> *f variants)
//   _d          → double-precision double (wraps <math.h> base variants)
#pragma once

// ── Constants (double and float) ─────────────────────────────────────────────

const double PI_D();
const double E_D();
const double LN2_D();
const double LN10_D();
const double SQRT2_D();
const double INF_D();

const float  PI_F();
const float  E_F();
const float  SQRT2_F();

// ── Integer ───────────────────────────────────────────────────────────────────

const int           abs_int(int x);
const long long     abs_ll(long long x);
const int           min_int(int a, int b);
const int           max_int(int a, int b);
const long long     min_ll(long long a, long long b);
const long long     max_ll(long long a, long long b);
const int           clamp_int(int v, int lo, int hi);
const long long     clamp_ll(long long v, long long lo, long long hi);

// ── Single-precision float ────────────────────────────────────────────────────

float abs_f(float x);
float sqrt_f(float x);
float cbrt_f(float x);
float floor_f(float x);
float ceil_f(float x);
float round_f(float x);
float trunc_f(float x);
float pow_f(float base, float exp);
float exp_f(float x);
float exp2_f(float x);
float log_f(float x);
float log2_f(float x);
float log10_f(float x);
float sin_f(float x);
float cos_f(float x);
float tan_f(float x);
float asin_f(float x);
float acos_f(float x);
float atan_f(float x);
float atan2_f(float y, float x);
float sinh_f(float x);
float cosh_f(float x);
float tanh_f(float x);
float hypot_f(float x, float y);
float fmod_f(float x, float y);
float copysign_f(float mag, float sgn);
float fma_f(float a, float b, float c);
float min_f(float a, float b);
float max_f(float a, float b);
float clamp_f(float v, float lo, float hi);
int   isnan_f(float x);
int   isinf_f(float x);
int   isfinite_f(float x);

// ── Double-precision double ───────────────────────────────────────────────────

double abs_d(double x);
double sqrt_d(double x);
double cbrt_d(double x);
double floor_d(double x);
double ceil_d(double x);
double round_d(double x);
double trunc_d(double x);
double pow_d(double base, double exp);
double exp_d(double x);
double exp2_d(double x);
double log_d(double x);
double log2_d(double x);
double log10_d(double x);
double sin_d(double x);
double cos_d(double x);
double tan_d(double x);
double asin_d(double x);
double acos_d(double x);
double atan_d(double x);
double atan2_d(double y, double x);
double sinh_d(double x);
double cosh_d(double x);
double tanh_d(double x);
double hypot_d(double x, double y);
double fmod_d(double x, double y);
double copysign_d(double mag, double sgn);
double fma_d(double a, double b, double c);
double min_d(double a, double b);
double max_d(double a, double b);
double clamp_d(double v, double lo, double hi);
int    isnan_d(double x);
int    isinf_d(double x);
int    isfinite_d(double x);
