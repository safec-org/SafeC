// SafeC Standard Library — Math
// Integer utilities are pure const functions eligible for compile-time evaluation.
// Float/double utilities wrap the corresponding <math.h> functions.
#pragma once
#include <std/math.h>

// ── Explicit extern declarations for libm functions ───────────────────────────
// '#include <math.h>' isn't usable here: the system header's macro-heavy,
// attribute-laden declarations aren't something this compiler's preprocessor
// re-parses, so nothing was ever actually declared — every call below was
// silently unresolved. Declare the exact symbols this file calls instead,
// same as mem.sc/str.sc do for libc.
namespace std {

extern float  fabsf(float x);
extern float  sqrtf(float x);
extern float  cbrtf(float x);
extern float  floorf(float x);
extern float  ceilf(float x);
extern float  roundf(float x);
extern float  truncf(float x);
extern float  powf(float base, float exp);
extern float  expf(float x);
extern float  exp2f(float x);
extern float  logf(float x);
extern float  log2f(float x);
extern float  log10f(float x);
extern float  sinf(float x);
extern float  cosf(float x);
extern float  tanf(float x);
extern float  asinf(float x);
extern float  acosf(float x);
extern float  atanf(float x);
extern float  atan2f(float y, float x);
extern float  sinhf(float x);
extern float  coshf(float x);
extern float  tanhf(float x);
extern float  hypotf(float x, float y);
extern float  fmodf(float x, float y);
extern float  copysignf(float mag, float sgn);
extern float  fmaf(float a, float b, float c);

extern double fabs(double x);
extern double sqrt(double x);
extern double cbrt(double x);
extern double floor(double x);
extern double ceil(double x);
extern double round(double x);
extern double trunc(double x);
extern double pow(double base, double exp);
extern double exp(double x);
extern double exp2(double x);
extern double log(double x);
extern double log2(double x);
extern double log10(double x);
extern double sin(double x);
extern double cos(double x);
extern double tan(double x);
extern double asin(double x);
extern double acos(double x);
extern double atan(double x);
extern double atan2(double y, double x);
extern double sinh(double x);
extern double cosh(double x);
extern double tanh(double x);
extern double hypot(double x, double y);
extern double fmod(double x, double y);
extern double copysign(double mag, double sgn);
extern double fma(double a, double b, double c);

// ── Constants ─────────────────────────────────────────────────────────────────

inline const double PI_D()    { return 3.141592653589793238462643383; }
inline const double E_D()     { return 2.718281828459045235360287471; }
inline const double LN2_D()   { return 0.693147180559945309417232121; }
inline const double LN10_D()  { return 2.302585092994045684017991455; }
inline const double SQRT2_D() { return 1.414213562373095048801688724; }
inline const double INF_D()   { unsafe { return (double)(1.0 / 0.0); } }

const float  PI_F()    { return (float)3.14159265358979323846; }
const float  E_F()     { return (float)2.71828182845904523536; }
const float  SQRT2_F() { return (float)1.41421356237309504880; }

// ── Integer ───────────────────────────────────────────────────────────────────

inline const int abs_int(int x) {
    return x < 0 ? -x : x;
}

inline const long long abs_ll(long long x) {
    return x < 0 ? -x : x;
}

inline const int min_int(int a, int b) {
    return a < b ? a : b;
}

inline const int max_int(int a, int b) {
    return a > b ? a : b;
}

inline const long long min_ll(long long a, long long b) {
    return a < b ? a : b;
}

inline const long long max_ll(long long a, long long b) {
    return a > b ? a : b;
}

inline const int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline const long long clamp_ll(long long v, long long lo, long long hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ── Single-precision float ────────────────────────────────────────────────────

inline float abs_f(float x)                      { unsafe { return fabsf(x); } }
inline float sqrt_f(float x)                     { unsafe { return sqrtf(x); } }
inline float cbrt_f(float x)                     { unsafe { return cbrtf(x); } }
inline float floor_f(float x)                    { unsafe { return floorf(x); } }
inline float ceil_f(float x)                     { unsafe { return ceilf(x); } }
inline float round_f(float x)                    { unsafe { return roundf(x); } }
inline float trunc_f(float x)                    { unsafe { return truncf(x); } }
inline float pow_f(float base, float exp)        { unsafe { return powf(base, exp); } }
inline float exp_f(float x)                      { unsafe { return expf(x); } }
inline float exp2_f(float x)                     { unsafe { return exp2f(x); } }
inline float log_f(float x)                      { unsafe { return logf(x); } }
inline float log2_f(float x)                     { unsafe { return log2f(x); } }
inline float log10_f(float x)                    { unsafe { return log10f(x); } }
inline float sin_f(float x)                      { unsafe { return sinf(x); } }
inline float cos_f(float x)                      { unsafe { return cosf(x); } }
inline float tan_f(float x)                      { unsafe { return tanf(x); } }
inline float asin_f(float x)                     { unsafe { return asinf(x); } }
inline float acos_f(float x)                     { unsafe { return acosf(x); } }
inline float atan_f(float x)                     { unsafe { return atanf(x); } }
inline float atan2_f(float y, float x)           { unsafe { return atan2f(y, x); } }
inline float sinh_f(float x)                     { unsafe { return sinhf(x); } }
inline float cosh_f(float x)                     { unsafe { return coshf(x); } }
inline float tanh_f(float x)                     { unsafe { return tanhf(x); } }
inline float hypot_f(float x, float y)           { unsafe { return hypotf(x, y); } }
inline float fmod_f(float x, float y)            { unsafe { return fmodf(x, y); } }
inline float copysign_f(float mag, float sgn)    { unsafe { return copysignf(mag, sgn); } }
inline float fma_f(float a, float b, float c)    { unsafe { return fmaf(a, b, c); } }

inline const float min_f(float a, float b)             { return a < b ? a : b; }
inline const float max_f(float a, float b)             { return a > b ? a : b; }
inline const float clamp_f(float v, float lo, float hi){ return v < lo ? lo : (v > hi ? hi : v); }

// IEEE 754: NaN is the only value not equal to itself.
inline const int isnan_f(float x)    { return x != x; }
// IEEE 754: inf * 2 == inf (and != 0), finite * 2 != itself.
inline const int isinf_f(float x)    { return x != (float)0 && x + x == x; }
// Finite iff neither NaN nor inf.
inline const int isfinite_f(float x) { return !isnan_f(x) && !isinf_f(x); }

// ── Double-precision double ───────────────────────────────────────────────────

inline double abs_d(double x)                        { unsafe { return fabs(x); } }
inline double sqrt_d(double x)                       { unsafe { return sqrt(x); } }
inline double cbrt_d(double x)                       { unsafe { return cbrt(x); } }
inline double floor_d(double x)                      { unsafe { return floor(x); } }
inline double ceil_d(double x)                       { unsafe { return ceil(x); } }
inline double round_d(double x)                      { unsafe { return round(x); } }
inline double trunc_d(double x)                      { unsafe { return trunc(x); } }
inline double pow_d(double base, double exp)         { unsafe { return pow(base, exp); } }
inline double exp_d(double x)                        { unsafe { return exp(x); } }
inline double exp2_d(double x)                       { unsafe { return exp2(x); } }
inline double log_d(double x)                        { unsafe { return log(x); } }
inline double log2_d(double x)                       { unsafe { return log2(x); } }
inline double log10_d(double x)                      { unsafe { return log10(x); } }
inline double sin_d(double x)                        { unsafe { return sin(x); } }
inline double cos_d(double x)                        { unsafe { return cos(x); } }
inline double tan_d(double x)                        { unsafe { return tan(x); } }
inline double asin_d(double x)                       { unsafe { return asin(x); } }
inline double acos_d(double x)                       { unsafe { return acos(x); } }
inline double atan_d(double x)                       { unsafe { return atan(x); } }
inline double atan2_d(double y, double x)            { unsafe { return atan2(y, x); } }
inline double sinh_d(double x)                       { unsafe { return sinh(x); } }
inline double cosh_d(double x)                       { unsafe { return cosh(x); } }
inline double tanh_d(double x)                       { unsafe { return tanh(x); } }
inline double hypot_d(double x, double y)            { unsafe { return hypot(x, y); } }
inline double fmod_d(double x, double y)             { unsafe { return fmod(x, y); } }
inline double copysign_d(double mag, double sgn)     { unsafe { return copysign(mag, sgn); } }
inline double fma_d(double a, double b, double c)    { unsafe { return fma(a, b, c); } }

inline const double min_d(double a, double b)              { return a < b ? a : b; }
inline const double max_d(double a, double b)              { return a > b ? a : b; }
inline const double clamp_d(double v, double lo, double hi){ return v < lo ? lo : (v > hi ? hi : v); }

inline const int isnan_d(double x)    { return x != x; }
inline const int isinf_d(double x)    { return x != 0.0 && x + x == x; }
inline const int isfinite_d(double x) { return !isnan_d(x) && !isinf_d(x); }

} // namespace std
