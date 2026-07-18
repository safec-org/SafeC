// SafeC Standard Library — DCT implementation (see dct.h)
#pragma once
#include <std/dsp/dct.h>
#include <std/dsp/fixed.h>
#include <std/math.h>

#define SAFEC_DSP_PI_DCT 3.14159265358979323846

namespace std {

void dct2(const double* in, double* out, unsigned long n) {
    if (n == 0UL) { return; }
    double nD = (double)n;
    unsigned long k = 0UL;
    while (k < n) {
        double sum = 0.0;
        unsigned long i = 0UL;
        while (i < n) {
            double xi;
            unsafe { xi = in[i]; }
            double angle = SAFEC_DSP_PI_DCT / nD * ((double)i + 0.5) * (double)k;
            sum = sum + xi * cos_d(angle);
            i = i + 1UL;
        }
        unsafe { out[k] = sum; }
        k = k + 1UL;
    }
}

void idct3(const double* in, double* out, unsigned long n) {
    if (n == 0UL) { return; }
    double nD = (double)n;
    double x0;
    unsafe { x0 = in[0]; }
    unsigned long i = 0UL;
    while (i < n) {
        double sum = x0;
        unsigned long k = 1UL;
        while (k < n) {
            double xk;
            unsafe { xk = in[k]; }
            double angle = SAFEC_DSP_PI_DCT / nD * ((double)i + 0.5) * (double)k;
            sum = sum + 2.0 * xk * cos_d(angle);
            k = k + 1UL;
        }
        unsafe { out[i] = sum / nD; }
        i = i + 1UL;
    }
}

void fdct2(const Fixed* in, Fixed* out, unsigned long n) {
    if (n == 0UL) { return; }
    double nD = (double)n;
    unsigned long k = 0UL;
    while (k < n) {
        double sum = 0.0;
        unsigned long i = 0UL;
        while (i < n) {
            Fixed xiF;
            unsafe { xiF = in[i]; }
            double xi = fixed_to_float(xiF);
            double angle = SAFEC_DSP_PI_DCT / nD * ((double)i + 0.5) * (double)k;
            sum = sum + xi * cos_d(angle);
            i = i + 1UL;
        }
        unsafe { out[k] = fixed_from_float(sum); }
        k = k + 1UL;
    }
}

void fidct3(const Fixed* in, Fixed* out, unsigned long n) {
    if (n == 0UL) { return; }
    double nD = (double)n;
    Fixed x0F;
    unsafe { x0F = in[0]; }
    double x0 = fixed_to_float(x0F);
    unsigned long i = 0UL;
    while (i < n) {
        double sum = x0;
        unsigned long k = 1UL;
        while (k < n) {
            Fixed xkF;
            unsafe { xkF = in[k]; }
            double xk = fixed_to_float(xkF);
            double angle = SAFEC_DSP_PI_DCT / nD * ((double)i + 0.5) * (double)k;
            sum = sum + 2.0 * xk * cos_d(angle);
            k = k + 1UL;
        }
        unsafe { out[i] = fixed_from_float(sum / nD); }
        i = i + 1UL;
    }
}

} // namespace std
