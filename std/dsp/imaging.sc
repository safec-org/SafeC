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
#define IM_FFT_BATCH  fft_batch
#define IM_CONV2D     conv2d
#define IM_FFT2D      fft2d
#define IM_IFFT2D     ifft2d
#include <std/dsp/imaging_core.h>

void gaussian_kernel(&double kernel, unsigned long size, double sigma) {
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

inline void sobel_x_kernel(&double kernel) {
    unsafe {
        kernel[0] = 0.0 - 1.0; kernel[1] = 0.0; kernel[2] = 1.0;
        kernel[3] = 0.0 - 2.0; kernel[4] = 0.0; kernel[5] = 2.0;
        kernel[6] = 0.0 - 1.0; kernel[7] = 0.0; kernel[8] = 1.0;
    }
}

inline void sobel_y_kernel(&double kernel) {
    unsafe {
        kernel[0] = 0.0 - 1.0; kernel[1] = 0.0 - 2.0; kernel[2] = 0.0 - 1.0;
        kernel[3] = 0.0;       kernel[4] = 0.0;       kernel[5] = 0.0;
        kernel[6] = 1.0;       kernel[7] = 2.0;       kernel[8] = 1.0;
    }
}

} // namespace std
