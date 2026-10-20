// SafeC Standard Library — single-precision FFT (see fft.h)
//
// The same transform as fft.sc, built from fft_core.h in float: half the
// memory traffic, twice the values per vector register. Twiddles are
// computed in double and rounded once.
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

typedef vec<float, 2> fft_f32x2_;

#define FFT_R            float
#define FFT_V2           fft_f32x2_
#define FFT_V4           f32x4
#define FFT_V8           f32x8
#define FFT_SINGLE       1
#define FFT_DFTI_CREATE  DftiCreateDescriptor_s_1d
#include <std/dsp/fft_core.h>

int fft_prepare_f32(unsigned long n) {
    if (dsp_is_pow2(n) == 0) return 0;
    return fft_twiddles_(n) != (const float*)0 ? 1 : 0;
}

static int fft_dispatch32_(&Complex32 data, unsigned long n, int inv) {
    if (dsp_is_pow2(n) == 0) return 0;
    if (n <= 1UL) return 1;
    unsafe { return fft_run_((float*)data, n, inv, FFT_MODE_TOP); }
}

int fft_f32(&Complex32 data, unsigned long n) { return fft_dispatch32_(data, n, 0); }
int ifft_f32(&Complex32 data, unsigned long n) { return fft_dispatch32_(data, n, 1); }

int fft_serial_f32(&Complex32 data, unsigned long n) {
    if (dsp_is_pow2(n) == 0) return 0;
    if (n <= 1UL) return 1;
    unsafe { return fft_run_((float*)data, n, 0, FFT_MODE_RT); }
}

int ifft_serial_f32(&Complex32 data, unsigned long n) {
    if (dsp_is_pow2(n) == 0) return 0;
    if (n <= 1UL) return 1;
    unsafe { return fft_run_((float*)data, n, 1, FFT_MODE_RT); }
}

int fft_real_f32(const &float in, &Complex32 out, unsigned long n) {
    if (dsp_is_pow2(n) == 0) return 0;
    unsafe { return fft_real_run_((const float*)in, (float*)out, n, FFT_MODE_TOP, 1); }
}

int fft_batch_f32(&Complex32 data, unsigned long n, unsigned long batch, int inverse) {
    if (dsp_is_pow2(n) == 0) return 0;
    if (n <= 1UL || batch == 0UL) return 1;
    if (fft_twiddles_(n) == (const float*)0) return 0;
    struct FftBatchJob j;
    unsafe {
        j.in = (const float*)0;
        j.d = (float*)data;
    }
    j.n = n;
    j.inv = inverse ? 1 : 0;
    j.ok = 1;
    if (batch == 1UL) return fft_run_(j.d, n, j.inv, FFT_MODE_TOP);
    // Rows go to threads whole; ~16K complex values per chunk.
    parallel_for(batch, 16384UL / n + 1UL, fft_batch_job_, &j);
    return j.ok;
}

int fft_real_batch_f32(const &float in, &Complex32 out, unsigned long n, unsigned long batch) {
    if (dsp_is_pow2(n) == 0) return 0;
    if (batch == 0UL) return 1;
    if (fft_twiddles_(n) == (const float*)0) return 0;
    struct FftBatchJob j;
    unsafe {
        j.in = (const float*)in;
        j.d = (float*)out;
    }
    j.n = n;
    j.inv = 0;
    j.ok = 1;
    if (batch == 1UL) { unsafe { return fft_real_run_(j.in, j.d, n, FFT_MODE_TOP, 1); } }
    parallel_for(batch, 16384UL / n + 1UL, fft_real_batch_job_, &j);
    return j.ok;
}

int fft_real_batch_inplace_f32(&Complex32 data, unsigned long n, unsigned long batch) {
    unsafe { return fft_real_inplace_((float*)data, n, batch); }
}

int fft_real_half_f32(const &float in, &Complex32 out, unsigned long n) {
    unsafe { return fft_real_half_batch_((const float*)in, (float*)out, n, 1UL); }
}

int fft_real_half_batch_f32(const &float in, &Complex32 out, unsigned long n, unsigned long batch) {
    unsafe { return fft_real_half_batch_((const float*)in, (float*)out, n, batch); }
}

} // namespace std
