// SafeC Standard Library — single-precision imaging (see imaging.h)
#pragma once
#include <std/simd/cpu.h>
#include <std/dsp/imaging.h>
#include <std/dsp/complex_dsp.h>
#include <std/dsp/fft.h>
#include <std/math.h>
#include <std/mem.h>
#include <std/sync/parallel.h>

namespace std {

#define IM_R          float
#define IM_C          struct Complex32
#define IM_CREF       Complex32
#define IM_FFT_BATCH  fft_batch_f32_n_
#define IM_CONV2D     conv2d_f32_n_
#define IM_FFT2D      fft2d_f32_n_
#define IM_IFFT2D     ifft2d_f32_n_
#include <std/dsp/imaging_core.h>

// ── Slice API (the public signatures; the (reference, length) cores above
// are the *_n_ functions) ─────────────────────────────────────────────────────

int conv2d_f32([]const float img, unsigned long h, unsigned long w,
            []const float kernel, unsigned long kh, unsigned long kw, []float out) {
    if (h == 0UL || w == 0UL || kh == 0UL || kw == 0UL || img.len / w < h || kernel.len / kw < kh || out.len / w < h) return 0;
    unsafe { conv2d_f32_n_((const &float)img.ptr, h, w, (const &float)kernel.ptr, kh, kw, (&float)out.ptr); }
    return 1;
}

int fft2d_f32([]Complex32 img, unsigned long h, unsigned long w) {
    if (h == 0UL || w == 0UL || img.len / w < h) return 0;
    unsafe { fft2d_f32_n_((&Complex32)img.ptr, h, w); }
    return 1;
}

int ifft2d_f32([]Complex32 img, unsigned long h, unsigned long w) {
    if (h == 0UL || w == 0UL || img.len / w < h) return 0;
    unsafe { ifft2d_f32_n_((&Complex32)img.ptr, h, w); }
    return 1;
}

} // namespace std
