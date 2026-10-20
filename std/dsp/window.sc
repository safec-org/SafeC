// SafeC Standard Library — Window Functions implementation (see window.h)
#pragma once
#include <std/dsp/window.h>
#include <std/math.h>

#define SAFEC_DSP_TWO_PI_W 6.28318530717958647692
#define SAFEC_DSP_FOUR_PI_W 12.56637061435917295384

namespace std {

void window_rectangular(&double w, unsigned long n) {
    unsigned long i = 0UL;
    while (i < n) {
        unsafe { w[i] = 1.0; }
        i = i + 1UL;
    }
}

void window_hann(&double w, unsigned long n) {
    if (n <= 1UL) {
        unsigned long i = 0UL;
        while (i < n) { unsafe { w[i] = 1.0; } i = i + 1UL; }
        return;
    }
    double denom = (double)(n - 1UL);
    unsigned long i = 0UL;
    while (i < n) {
        double theta = SAFEC_DSP_TWO_PI_W * (double)i / denom;
        unsafe { w[i] = 0.5 * (1.0 - cos_d(theta)); }
        i = i + 1UL;
    }
}

void window_hamming(&double w, unsigned long n) {
    if (n <= 1UL) {
        unsigned long i = 0UL;
        while (i < n) { unsafe { w[i] = 1.0; } i = i + 1UL; }
        return;
    }
    double denom = (double)(n - 1UL);
    unsigned long i = 0UL;
    while (i < n) {
        double theta = SAFEC_DSP_TWO_PI_W * (double)i / denom;
        unsafe { w[i] = 0.54 - 0.46 * cos_d(theta); }
        i = i + 1UL;
    }
}

void window_blackman(&double w, unsigned long n) {
    if (n <= 1UL) {
        unsigned long i = 0UL;
        while (i < n) { unsafe { w[i] = 1.0; } i = i + 1UL; }
        return;
    }
    double denom = (double)(n - 1UL);
    unsigned long i = 0UL;
    while (i < n) {
        double theta1 = SAFEC_DSP_TWO_PI_W * (double)i / denom;
        double theta2 = SAFEC_DSP_FOUR_PI_W * (double)i / denom;
        unsafe { w[i] = 0.42 - 0.5 * cos_d(theta1) + 0.08 * cos_d(theta2); }
        i = i + 1UL;
    }
}

void window_apply(&double x, const &double w, unsigned long n) {
    unsigned long i = 0UL;
    while (i < n) {
        unsafe { x[i] = x[i] * w[i]; }
        i = i + 1UL;
    }
}

// Single precision: the same windows, computed in double and rounded once.
// kind: 0 rectangular, 1 hann, 2 hamming, 3 blackman.
static void window_f32_(&float w, unsigned long n, int kind) {
    if (n <= 1UL) kind = 0;
    double denom = (double)(n - 1UL);
    unsigned long i = 0UL;
    while (i < n) {
        double v = 1.0;
        double t1 = SAFEC_DSP_TWO_PI_W * (double)i / denom;
        if (kind == 1) v = 0.5 * (1.0 - cos_d(t1));
        else if (kind == 2) v = 0.54 - 0.46 * cos_d(t1);
        else if (kind == 3) v = 0.42 - 0.5 * cos_d(t1) + 0.08 * cos_d(SAFEC_DSP_FOUR_PI_W * (double)i / denom);
        unsafe { w[i] = (float)v; }
        i = i + 1UL;
    }
}

void window_rectangular_f32(&float w, unsigned long n) { window_f32_(w, n, 0); }
void window_hann_f32(&float w, unsigned long n)        { window_f32_(w, n, 1); }
void window_hamming_f32(&float w, unsigned long n)     { window_f32_(w, n, 2); }
void window_blackman_f32(&float w, unsigned long n)    { window_f32_(w, n, 3); }

void window_apply_f32(&float x, const &float w, unsigned long n) {
    unsigned long i = 0UL;
    while (i < n) {
        unsafe { x[i] = x[i] * w[i]; }
        i = i + 1UL;
    }
}

} // namespace std
