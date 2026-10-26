// SafeC Standard Library — Imaging implementation (see imaging.h)
#pragma once
#include <std/simd/cpu.h>
#include <std/dsp/imaging.h>
#include <std/dsp/complex_dsp.h>
#include <std/dsp/fft.h>
#include <std/math.h>
#include <std/mem.h>
#include <std/sync/parallel.h>

namespace std {

// conv2d and fft2d come from imaging_core.h, here in double;
// imaging_f32.sc builds them in float.
#define IM_R          double
#define IM_C          struct Complex
#define IM_CREF       Complex
#define IM_FFT_BATCH  fft_batch_n_
#define IM_CONV2D     conv2d_n_
#define IM_FFT2D      fft2d_n_
#define IM_IFFT2D     ifft2d_n_
#include <std/dsp/imaging_core.h>

void gaussian_kernel_n_(&double kernel, unsigned long size, double sigma) {
    unsigned long center = size / 2UL;
    double sum = 0.0;
    unsigned long y = 0UL;
    while (y < size) {
        unsigned long x = 0UL;
        while (x < size) {
            double dy = (double)y - (double)center;
            double dx = (double)x - (double)center;
            double v = exp_d(0.0 - (dx * dx + dy * dy) / (2.0 * sigma * sigma));
            unsafe { kernel[y * size + x] = v; }
            sum = sum + v;
            x = x + 1UL;
        }
        y = y + 1UL;
    }
    y = 0UL;
    while (y < size) {
        unsigned long x = 0UL;
        while (x < size) {
            double v;
            unsafe { v = kernel[y * size + x]; }
            unsafe { kernel[y * size + x] = v / sum; }
            x = x + 1UL;
        }
        y = y + 1UL;
    }
}

inline void sobel_x_kernel_n_(&double kernel) {
    unsafe {
        kernel[0] = 0.0 - 1.0; kernel[1] = 0.0; kernel[2] = 1.0;
        kernel[3] = 0.0 - 2.0; kernel[4] = 0.0; kernel[5] = 2.0;
        kernel[6] = 0.0 - 1.0; kernel[7] = 0.0; kernel[8] = 1.0;
    }
}

inline void sobel_y_kernel_n_(&double kernel) {
    unsafe {
        kernel[0] = 0.0 - 1.0; kernel[1] = 0.0 - 2.0; kernel[2] = 0.0 - 1.0;
        kernel[3] = 0.0;       kernel[4] = 0.0;       kernel[5] = 0.0;
        kernel[6] = 1.0;       kernel[7] = 2.0;       kernel[8] = 1.0;
    }
}

// ── Slice API (the public signatures; the (reference, length) cores above
// are the *_n_ functions) ─────────────────────────────────────────────────────

int conv2d([]const double img, unsigned long h, unsigned long w,
            []const double kernel, unsigned long kh, unsigned long kw, []double out) {
    if (h == 0UL || w == 0UL || kh == 0UL || kw == 0UL || img.len / w < h || kernel.len / kw < kh || out.len / w < h) return 0;
    unsafe { conv2d_n_((const &double)img.ptr, h, w, (const &double)kernel.ptr, kh, kw, (&double)out.ptr); }
    return 1;
}

int fft2d([]Complex img, unsigned long h, unsigned long w) {
    if (h == 0UL || w == 0UL || img.len / w < h) return 0;
    unsafe { fft2d_n_((&Complex)img.ptr, h, w); }
    return 1;
}

int ifft2d([]Complex img, unsigned long h, unsigned long w) {
    if (h == 0UL || w == 0UL || img.len / w < h) return 0;
    unsafe { ifft2d_n_((&Complex)img.ptr, h, w); }
    return 1;
}

int gaussian_kernel([]double kernel, unsigned long size, double sigma) {
    if (size == 0UL || kernel.len / size < size) return 0;
    unsafe { gaussian_kernel_n_((&double)kernel.ptr, size, sigma); }
    return 1;
}

int sobel_x_kernel([]double kernel) {
    if (kernel.len < 9UL) return 0;
    unsafe { sobel_x_kernel_n_((&double)kernel.ptr); }
    return 1;
}

int sobel_y_kernel([]double kernel) {
    if (kernel.len < 9UL) return 0;
    unsafe { sobel_y_kernel_n_((&double)kernel.ptr); }
    return 1;
}

} // namespace std
