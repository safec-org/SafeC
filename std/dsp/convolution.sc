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
#define CV_CNEW       Complex::new
#define CV_DOT        dsp_dot_f64_n_
#define CV_SDOT       dsp_sliding_dot_f64_n_
#define CV_FFT        fft_n_
#define CV_IFFT       ifft_n_
#define CV_FFT_BATCH  fft_batch_n_
#define CV_DIRECT     conv_direct_n_
#define CV_FFTCONV    conv_fft_n_
#include <std/dsp/conv_core.h>

void fconv_direct_n_(const &Fixed x, unsigned long len_x,
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

// ── Slice API (the public signatures; the (reference, length) cores above
// are the *_n_ functions) ─────────────────────────────────────────────────────

int conv_direct([]const double x, []const double h, []double out) {
    if (x.len == 0UL || h.len == 0UL || out.len < x.len + h.len - 1UL) return 0;
    unsafe { conv_direct_n_((const &double)x.ptr, x.len, (const &double)h.ptr, h.len, (&double)out.ptr); }
    return 1;
}

int conv_fft([]const double x, []const double h, []double out) {
    if (x.len == 0UL || h.len == 0UL || out.len < x.len + h.len - 1UL) return 0;
    unsafe { conv_fft_n_((const &double)x.ptr, x.len, (const &double)h.ptr, h.len, (&double)out.ptr); }
    return 1;
}

int fconv_direct([]const Fixed x, []const Fixed h, []Fixed out) {
    if (x.len == 0UL || h.len == 0UL || out.len < x.len + h.len - 1UL) return 0;
    unsafe { fconv_direct_n_((const &Fixed)x.ptr, x.len, (const &Fixed)h.ptr, h.len, (&Fixed)out.ptr); }
    return 1;
}

} // namespace std
