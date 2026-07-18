// SafeC Standard Library — FFT implementation (see fft.h)
#pragma once
#include <std/dsp/fft.h>
#include <std/dsp/complex_dsp.h>

#define SAFEC_DSP_TWO_PI 6.28318530717958647692

namespace std {

inline int dsp_is_pow2(unsigned long n) {
    if (n == 0UL) { return 0; }
    return (n & (n - 1UL)) == 0UL ? 1 : 0;
}

static int dsp_log2_(unsigned long n) {
    int bits = 0;
    unsigned long v = n;
    while (v > 1UL) {
        v = v >> 1;
        bits = bits + 1;
    }
    return bits;
}

static unsigned long dsp_bit_reverse_(unsigned long x, int numBits) {
    unsigned long result = 0UL;
    int k = 0;
    unsigned long v = x;
    while (k < numBits) {
        result = (result << 1) | (v & 1UL);
        v = v >> 1;
        k = k + 1;
    }
    return result;
}

int fft(struct Complex* data, unsigned long n) {
    if (dsp_is_pow2(n) == 0) {
        return 0;
    }
    if (n <= 1UL) {
        return 1;
    }
    int numBits = dsp_log2_(n);

    // Bit-reversal permutation.
    unsigned long i = 0UL;
    while (i < n) {
        unsigned long j = dsp_bit_reverse_(i, numBits);
        if (i < j) {
            unsafe {
                struct Complex t = data[i];
                data[i] = data[j];
                data[j] = t;
            }
        }
        i = i + 1UL;
    }

    // Iterative Cooley-Tukey butterfly stages.
    unsigned long len = 2UL;
    while (len <= n) {
        unsigned long half = len / 2UL;
        unsigned long block = 0UL;
        while (block < n) {
            unsigned long k = 0UL;
            while (k < half) {
                double theta = -SAFEC_DSP_TWO_PI * (double)k / (double)len;
                struct Complex w = complex_from_polar(1.0, theta);
                unsafe {
                    struct Complex u = data[block + k];
                    struct Complex v = data[block + k + half] * w;
                    data[block + k]        = u + v;
                    data[block + k + half] = u - v;
                }
                k = k + 1UL;
            }
            block = block + len;
        }
        len = len * 2UL;
    }
    return 1;
}

int ifft(struct Complex* data, unsigned long n) {
    if (dsp_is_pow2(n) == 0) {
        return 0;
    }
    // Standard conjugate trick: conjugate, forward FFT, conjugate + scale.
    unsigned long i = 0UL;
    while (i < n) {
        unsafe { data[i] = data[i].conj(); }
        i = i + 1UL;
    }
    fft(data, n);
    if (n == 0UL) {
        return 1;
    }
    double invN = 1.0 / (double)n;
    i = 0UL;
    while (i < n) {
        unsafe {
            struct Complex c = data[i].conj();
            data[i] = complex_new(c.re * invN, c.im * invN);
        }
        i = i + 1UL;
    }
    return 1;
}

int fft_real(const double* in, struct Complex* out, unsigned long n) {
    if (dsp_is_pow2(n) == 0) {
        return 0;
    }
    unsigned long i = 0UL;
    while (i < n) {
        unsafe { out[i] = complex_new(in[i], 0.0); }
        i = i + 1UL;
    }
    return fft(out, n);
}

int ffft(struct FComplex* data, unsigned long n) {
    if (dsp_is_pow2(n) == 0) {
        return 0;
    }
    if (n <= 1UL) {
        return 1;
    }
    int numBits = dsp_log2_(n);

    unsigned long i = 0UL;
    while (i < n) {
        unsigned long j = dsp_bit_reverse_(i, numBits);
        if (i < j) {
            unsafe {
                struct FComplex t = data[i];
                data[i] = data[j];
                data[j] = t;
            }
        }
        i = i + 1UL;
    }

    unsigned long len = 2UL;
    while (len <= n) {
        unsigned long half = len / 2UL;
        unsigned long block = 0UL;
        while (block < n) {
            unsigned long k = 0UL;
            while (k < half) {
                double theta = -SAFEC_DSP_TWO_PI * (double)k / (double)len;
                struct FComplex w = complex_to_fixed(complex_from_polar(1.0, theta));
                unsafe {
                    struct FComplex u = data[block + k];
                    struct FComplex v = data[block + k + half] * w;
                    data[block + k]        = u + v;
                    data[block + k + half] = u - v;
                }
                k = k + 1UL;
            }
            block = block + len;
        }
        len = len * 2UL;
    }
    return 1;
}

int fifft(struct FComplex* data, unsigned long n) {
    if (dsp_is_pow2(n) == 0) {
        return 0;
    }
    unsigned long i = 0UL;
    while (i < n) {
        unsafe { data[i] = data[i].conj(); }
        i = i + 1UL;
    }
    ffft(data, n);
    if (n == 0UL) {
        return 1;
    }
    Fixed invN = fixed_div((Fixed)FIXED_ONE, fixed_from_int((int)n));
    i = 0UL;
    while (i < n) {
        unsafe {
            struct FComplex c = data[i].conj();
            data[i] = fcomplex_new(fixed_mul(c.re, invN), fixed_mul(c.im, invN));
        }
        i = i + 1UL;
    }
    return 1;
}

} // namespace std
