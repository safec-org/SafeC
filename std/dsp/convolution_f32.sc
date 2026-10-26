// SafeC Standard Library — single-precision convolution (see convolution.h)
#pragma once
#include <std/dsp/convolution.h>
#include <std/dsp/fft.h>
#include <std/dsp/complex_dsp.h>
#include <std/dsp/dsp.h>
#include <std/mem.h>
#include <std/sync/parallel.h>
#include <std/simd/cpu.h>

namespace std {

#define CV_R          float
#define CV_C          struct Complex32
#define CV_CNEW       Complex32::new
#define CV_DOT        dsp_dot_f32_n_
#define CV_SDOT       dsp_sliding_dot_f32_n_
#define CV_FFT        fft_f32_n_
#define CV_IFFT       ifft_f32_n_
#define CV_FFT_BATCH  fft_batch_f32_n_
#define CV_DIRECT     conv_direct_f32_n_
#define CV_FFTCONV    conv_fft_f32_n_
#include <std/dsp/conv_core.h>

// ── Slice API (the public signatures; the (reference, length) cores above
// are the *_n_ functions) ─────────────────────────────────────────────────────

int conv_direct_f32([]const float x, []const float h, []float out) {
    if (x.len == 0UL || h.len == 0UL || out.len < x.len + h.len - 1UL) return 0;
    unsafe { conv_direct_f32_n_((const &float)x.ptr, x.len, (const &float)h.ptr, h.len, (&float)out.ptr); }
    return 1;
}

int conv_fft_f32([]const float x, []const float h, []float out) {
    if (x.len == 0UL || h.len == 0UL || out.len < x.len + h.len - 1UL) return 0;
    unsafe { conv_fft_f32_n_((const &float)x.ptr, x.len, (const &float)h.ptr, h.len, (&float)out.ptr); }
    return 1;
}

} // namespace std
