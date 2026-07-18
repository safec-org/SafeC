// SafeC Standard Library — Convolution implementation (see convolution.h)
#pragma once
#include <std/dsp/convolution.h>
#include <std/dsp/fft.h>
#include <std/dsp/complex_dsp.h>
#include <std/dsp/dsp.h>
#include <std/mem.h>

namespace std {

// Direct convolution's inner sum is sum_k x[k] * h[n-k]: x is walked
// forward, h is walked in reverse, so it isn't literally a dot product of
// two contiguous ranges as written. Reversing h once up front (hrev[i] =
// h[len_h-1-i]) turns h[n-k] into hrev[(len_h-1-n)+k] — now increasing
// with k, same direction as x[k] — so the whole inner loop becomes one
// contiguous-range call into dsp_dot_f64's SIMD FMA accumulation instead
// of a scalar multiply-add per sample.
void conv_direct(const double* x, unsigned long len_x,
                  const double* h, unsigned long len_h, double* out) {
    if (len_x == 0UL || len_h == 0UL) {
        return;
    }

    double* hrev;
    unsafe { hrev = (double*)alloc(len_h * (unsigned long)sizeof(double)); }
    unsigned long r = 0UL;
    while (r < len_h) {
        unsafe { hrev[r] = h[len_h - 1UL - r]; }
        r = r + 1UL;
    }

    unsigned long outLen = len_x + len_h - 1UL;
    unsigned long n = 0UL;
    while (n < outLen) {
        unsigned long kStart = 0UL;
        if (n >= len_h) {
            kStart = n - len_h + 1UL;
        }
        unsigned long kEnd = n;
        if (kEnd >= len_x) {
            kEnd = len_x - 1UL;
        }
        unsigned long segLen = kEnd - kStart + 1UL;
        // base = (len_h - 1 + kStart) - n, reordered so the subtraction of
        // n happens last (both operands are unsigned): the convolution
        // range invariant guarantees len_h - 1 + kStart >= n here.
        unsigned long base = len_h - 1UL + kStart - n;
        double sum;
        unsafe { sum = dsp_dot_f64(x + kStart, hrev + base, segLen); }
        unsafe { out[n] = sum; }
        n = n + 1UL;
    }

    unsafe { dealloc((void*)hrev); }
}

void fconv_direct(const Fixed* x, unsigned long len_x,
                   const Fixed* h, unsigned long len_h, Fixed* out) {
    if (len_x == 0UL || len_h == 0UL) {
        return;
    }
    unsigned long outLen = len_x + len_h - 1UL;
    unsigned long n = 0UL;
    while (n < outLen) {
        unsigned long kStart = 0UL;
        if (n >= len_h) {
            kStart = n - len_h + 1UL;
        }
        unsigned long kEnd = n;
        if (kEnd >= len_x) {
            kEnd = len_x - 1UL;
        }
        Fixed sum = (Fixed)0;
        unsigned long k = kStart;
        while (k <= kEnd) {
            unsafe { sum = fixed_add(sum, fixed_mul(x[k], h[n - k])); }
            k = k + 1UL;
        }
        unsafe { out[n] = sum; }
        n = n + 1UL;
    }
}

void conv_fft(const double* x, unsigned long len_x,
              const double* h, unsigned long len_h, double* out) {
    if (len_x == 0UL || len_h == 0UL) {
        return;
    }
    unsigned long outLen = len_x + len_h - 1UL;
    unsigned long fftLen = 1UL;
    while (fftLen < outLen) {
        fftLen = fftLen * 2UL;
    }

    struct Complex* X;
    struct Complex* H;
    unsafe {
        X = (struct Complex*)alloc(fftLen * (unsigned long)sizeof(struct Complex));
        H = (struct Complex*)alloc(fftLen * (unsigned long)sizeof(struct Complex));
    }

    unsigned long i = 0UL;
    while (i < fftLen) {
        double xv = 0.0;
        double hv = 0.0;
        if (i < len_x) {
            unsafe { xv = x[i]; }
        }
        if (i < len_h) {
            unsafe { hv = h[i]; }
        }
        unsafe {
            X[i] = complex_new(xv, 0.0);
            H[i] = complex_new(hv, 0.0);
        }
        i = i + 1UL;
    }

    fft(X, fftLen);
    fft(H, fftLen);

    i = 0UL;
    while (i < fftLen) {
        unsafe { X[i] = X[i] * H[i]; }
        i = i + 1UL;
    }

    ifft(X, fftLen);

    i = 0UL;
    while (i < outLen) {
        unsafe { out[i] = X[i].re; }
        i = i + 1UL;
    }

    unsafe {
        dealloc((void*)X);
        dealloc((void*)H);
    }
}

} // namespace std
