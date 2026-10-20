// SafeC Standard Library — FFT implementation (see fft.h)
#pragma once
#include <std/dsp/fft.h>
#include <std/dsp/complex_dsp.h>
#include <std/math.h>
#include <std/mem.h>
#include <std/atomic.h>
#include <std/thread.h>
#include <std/sync/parallel.h>
#include <std/simd/simd.h>
#include <std/simd/cpu.h>

#define SAFEC_DSP_TWO_PI 6.28318530717958647692

namespace std {

inline int dsp_is_pow2(unsigned long n) {
    if (n == 0UL) { return 0; }
    return (n & (n - 1UL)) == 0UL ? 1 : 0;
}

// The FFT core (fft_core.h) in double; fft_f32.sc builds it in float.
typedef vec<double, 8> fft_f64x8_;
#define FFT_R            double
#define FFT_V2           f64x2
#define FFT_V4           f64x4
#define FFT_V8           fft_f64x8_
#define FFT_SINGLE       0
#define FFT_DFTI_CREATE  DftiCreateDescriptor_d_1d
#include <std/dsp/fft_core.h>

int fft_prepare(unsigned long n) {
    if (dsp_is_pow2(n) == 0) return 0;
    return fft_twiddles_(n) != (const double*)0 ? 1 : 0;
}

int fft_vendor() { return fft_vendor_linked_(); }

static int fft_dispatch_(&Complex data, unsigned long n, int inv) {
    if (dsp_is_pow2(n) == 0) return 0;
    if (n <= 1UL) return 1;
    unsafe { return fft_run_((double*)data, n, inv, FFT_MODE_TOP); }
}

int fft(&Complex data, unsigned long n) { return fft_dispatch_(data, n, 0); }
int ifft(&Complex data, unsigned long n) { return fft_dispatch_(data, n, 1); }

int fft_serial(&Complex data, unsigned long n) {
    if (dsp_is_pow2(n) == 0) return 0;
    if (n <= 1UL) return 1;
    unsafe { return fft_run_((double*)data, n, 0, FFT_MODE_RT); }
}

int ifft_serial(&Complex data, unsigned long n) {
    if (dsp_is_pow2(n) == 0) return 0;
    if (n <= 1UL) return 1;
    unsafe { return fft_run_((double*)data, n, 1, FFT_MODE_RT); }
}

int fft_real(const &double in, &Complex out, unsigned long n) {
    if (dsp_is_pow2(n) == 0) return 0;
    unsafe { return fft_real_run_((const double*)in, (double*)out, n, FFT_MODE_TOP); }
}

int fft_batch(&Complex data, unsigned long n, unsigned long batch, int inverse) {
    if (dsp_is_pow2(n) == 0) return 0;
    if (n <= 1UL || batch == 0UL) return 1;
    if (fft_twiddles_(n) == (const double*)0) return 0;
    struct FftBatchJob j;
    unsafe {
        j.in = (const double*)0;
        j.d = (double*)data;
    }
    j.n = n;
    j.inv = inverse ? 1 : 0;
    j.ok = 1;
    if (batch == 1UL) return fft_run_(j.d, n, j.inv, FFT_MODE_TOP);
    // Rows go to threads whole; ~16K complex values per chunk.
    parallel_for(batch, 16384UL / n + 1UL, fft_batch_job_, &j);
    return j.ok;
}

int fft_real_batch(const &double in, &Complex out, unsigned long n, unsigned long batch) {
    if (dsp_is_pow2(n) == 0) return 0;
    if (batch == 0UL) return 1;
    if (fft_twiddles_(n) == (const double*)0) return 0;
    struct FftBatchJob j;
    unsafe {
        j.in = (const double*)in;
        j.d = (double*)out;
    }
    j.n = n;
    j.inv = 0;
    j.ok = 1;
    if (batch == 1UL) { unsafe { return fft_real_run_(j.in, j.d, n, FFT_MODE_TOP); } }
    parallel_for(batch, 16384UL / n + 1UL, fft_real_batch_job_, &j);
    return j.ok;
}

int fft_real_batch_inplace(&Complex data, unsigned long n, unsigned long batch) {
    unsafe { return fft_real_inplace_((double*)data, n, batch); }
}

int ffft(&FComplex data, unsigned long n) {
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

int fifft(&FComplex data, unsigned long n) {
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
