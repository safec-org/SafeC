// SafeC Standard Library — DFT implementation (see dft.h)
#pragma once
#include <std/dsp/dft.h>
#include <std/dsp/complex_dsp.h>

#define SAFEC_DSP_TWO_PI 6.28318530717958647692

namespace std {

void dft(const struct Complex* in, struct Complex* out, unsigned long n) {
    unsigned long k = 0UL;
    while (k < n) {
        struct Complex sum = complex_new(0.0, 0.0);
        unsigned long i = 0UL;
        while (i < n) {
            // e^(-i*2*pi*k*i/n)
            double theta = -SAFEC_DSP_TWO_PI * (double)k * (double)i / (double)n;
            struct Complex twiddle = complex_from_polar(1.0, theta);
            struct Complex xi;
            unsafe { xi = in[i]; }
            sum = sum + xi * twiddle;
            i = i + 1UL;
        }
        unsafe { out[k] = sum; }
        k = k + 1UL;
    }
}

void dft_real(const double* in, struct Complex* out, unsigned long n) {
    unsigned long k = 0UL;
    while (k < n) {
        struct Complex sum = complex_new(0.0, 0.0);
        unsigned long i = 0UL;
        while (i < n) {
            double theta = -SAFEC_DSP_TWO_PI * (double)k * (double)i / (double)n;
            struct Complex twiddle = complex_from_polar(1.0, theta);
            double xi;
            unsafe { xi = in[i]; }
            sum = sum + complex_new(xi * twiddle.re, xi * twiddle.im);
            i = i + 1UL;
        }
        unsafe { out[k] = sum; }
        k = k + 1UL;
    }
}

void idft(const struct Complex* in, struct Complex* out, unsigned long n) {
    if (n == 0UL) { return; }
    double invN = 1.0 / (double)n;
    unsigned long k = 0UL;
    while (k < n) {
        struct Complex sum = complex_new(0.0, 0.0);
        unsigned long i = 0UL;
        while (i < n) {
            // e^(+i*2*pi*k*i/n) — sign flipped vs. the forward transform
            double theta = SAFEC_DSP_TWO_PI * (double)k * (double)i / (double)n;
            struct Complex twiddle = complex_from_polar(1.0, theta);
            struct Complex xi;
            unsafe { xi = in[i]; }
            sum = sum + xi * twiddle;
            i = i + 1UL;
        }
        unsafe { out[k] = complex_new(sum.re * invN, sum.im * invN); }
        k = k + 1UL;
    }
}

void fdft(const struct FComplex* in, struct FComplex* out, unsigned long n) {
    unsigned long k = 0UL;
    while (k < n) {
        struct FComplex sum = fcomplex_new((Fixed)0, (Fixed)0);
        unsigned long i = 0UL;
        while (i < n) {
            double theta = -SAFEC_DSP_TWO_PI * (double)k * (double)i / (double)n;
            struct FComplex twiddle = complex_to_fixed(complex_from_polar(1.0, theta));
            struct FComplex xi;
            unsafe { xi = in[i]; }
            sum = sum + xi * twiddle;
            i = i + 1UL;
        }
        unsafe { out[k] = sum; }
        k = k + 1UL;
    }
}

void fidft(const struct FComplex* in, struct FComplex* out, unsigned long n) {
    if (n == 0UL) { return; }
    Fixed invN = fixed_div((Fixed)FIXED_ONE, fixed_from_int((int)n));
    unsigned long k = 0UL;
    while (k < n) {
        struct FComplex sum = fcomplex_new((Fixed)0, (Fixed)0);
        unsigned long i = 0UL;
        while (i < n) {
            double theta = SAFEC_DSP_TWO_PI * (double)k * (double)i / (double)n;
            struct FComplex twiddle = complex_to_fixed(complex_from_polar(1.0, theta));
            struct FComplex xi;
            unsafe { xi = in[i]; }
            sum = sum + xi * twiddle;
            i = i + 1UL;
        }
        unsafe {
            out[k] = fcomplex_new(fixed_mul(sum.re, invN), fixed_mul(sum.im, invN));
        }
        k = k + 1UL;
    }
}

} // namespace std
