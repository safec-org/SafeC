// SafeC Standard Library — Convolution implementation (see convolution.h)
#pragma once
#include <std/dsp/convolution.h>
#include <std/dsp/fft.h>
#include <std/dsp/complex_dsp.h>
#include <std/dsp/dsp.h>
#include <std/mem.h>
#include <std/sync/parallel.h>
#include <std/simd/cpu.h>

namespace std {

// The shared core (conv_core.h) in double; convolution_f32.sc builds it in float.
#define CV_R          double
#define CV_C          struct Complex
#define CV_CNEW       complex_new
#define CV_DOT        dsp_dot_f64
#define CV_SDOT       dsp_sliding_dot_f64
#define CV_FFT        fft
#define CV_IFFT       ifft
#define CV_FFT_BATCH  fft_batch
#define CV_DIRECT     conv_direct
#define CV_FFTCONV    conv_fft
#include <std/dsp/conv_core.h>

void fconv_direct(const &Fixed x, unsigned long len_x,
                   const &Fixed h, unsigned long len_h, &Fixed out) {
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

} // namespace std
