// math.h — playground subset (implemented by the scc runtime)
#pragma once
#define M_PI 3.14159265358979323846
#define M_E 2.7182818284590452354
#define M_SQRT2 1.41421356237309504880
#define M_LN2 0.69314718055994530942
#define INFINITY (1.0 / 0.0)
#define HUGE_VAL (1.0 / 0.0)
#define NAN (0.0 / 0.0)
extern double sqrt(double x);
extern double fabs(double x);
extern double floor(double x);
extern double ceil(double x);
extern double trunc(double x);
extern double round(double x);
extern double fmod(double x, double y);
extern double fmin(double a, double b);
extern double fmax(double a, double b);
extern double ldexp(double x, int n);
extern double exp(double x);
extern double log(double x);
extern double log2(double x);
extern double log10(double x);
extern double pow(double x, double y);
extern double cbrt(double x);
extern double hypot(double a, double b);
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
extern float sqrtf(float x);
extern float fabsf(float x);
extern float floorf(float x);
extern float ceilf(float x);
extern float sinf(float x);
extern float cosf(float x);
extern float tanf(float x);
extern float expf(float x);
extern float logf(float x);
extern float powf(float x, float y);
extern float atan2f(float y, float x);
extern float fmodf(float x, float y);
extern float roundf(float x);
