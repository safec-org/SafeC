// SafeC Standard Library — Imaging implementation (see imaging.h)
#pragma once
#include <std/dsp/imaging.h>
#include <std/dsp/complex_dsp.h>
#include <std/dsp/fft.h>
#include <std/math.h>
#include <std/mem.h>

namespace std {

void conv2d(const double* img, unsigned long h, unsigned long w,
            const double* kernel, unsigned long kh, unsigned long kw, double* out) {
    unsigned long kcy = kh / 2UL;
    unsigned long kcx = kw / 2UL;

    unsigned long y = 0UL;
    while (y < h) {
        unsigned long x = 0UL;
        while (x < w) {
            double sum = 0.0;
            unsigned long ky = 0UL;
            while (ky < kh) {
                unsigned long srcYoff = y + ky;
                if (srcYoff >= kcy) {
                    unsigned long srcY = srcYoff - kcy;
                    if (srcY < h) {
                        unsigned long kx = 0UL;
                        while (kx < kw) {
                            unsigned long srcXoff = x + kx;
                            if (srcXoff >= kcx) {
                                unsigned long srcX = srcXoff - kcx;
                                if (srcX < w) {
                                    double pv;
                                    double kv;
                                    unsafe {
                                        pv = img[srcY * w + srcX];
                                        kv = kernel[ky * kw + kx];
                                    }
                                    sum = sum + pv * kv;
                                }
                            }
                            kx = kx + 1UL;
                        }
                    }
                }
                ky = ky + 1UL;
            }
            unsafe { out[y * w + x] = sum; }
            x = x + 1UL;
        }
        y = y + 1UL;
    }
}

void fft2d(struct Complex* img, unsigned long h, unsigned long w) {
    unsigned long y = 0UL;
    while (y < h) {
        unsafe { fft(img + y * w, w); }
        y = y + 1UL;
    }

    struct Complex* col;
    unsafe { col = (struct Complex*)alloc(h * (unsigned long)sizeof(struct Complex)); }
    unsigned long x = 0UL;
    while (x < w) {
        unsigned long yy = 0UL;
        while (yy < h) {
            unsafe { col[yy] = img[yy * w + x]; }
            yy = yy + 1UL;
        }
        fft(col, h);
        yy = 0UL;
        while (yy < h) {
            unsafe { img[yy * w + x] = col[yy]; }
            yy = yy + 1UL;
        }
        x = x + 1UL;
    }
    unsafe { dealloc((void*)col); }
}

void ifft2d(struct Complex* img, unsigned long h, unsigned long w) {
    unsigned long y = 0UL;
    while (y < h) {
        unsafe { ifft(img + y * w, w); }
        y = y + 1UL;
    }

    struct Complex* col;
    unsafe { col = (struct Complex*)alloc(h * (unsigned long)sizeof(struct Complex)); }
    unsigned long x = 0UL;
    while (x < w) {
        unsigned long yy = 0UL;
        while (yy < h) {
            unsafe { col[yy] = img[yy * w + x]; }
            yy = yy + 1UL;
        }
        ifft(col, h);
        yy = 0UL;
        while (yy < h) {
            unsafe { img[yy * w + x] = col[yy]; }
            yy = yy + 1UL;
        }
        x = x + 1UL;
    }
    unsafe { dealloc((void*)col); }
}

void gaussian_kernel(double* kernel, unsigned long size, double sigma) {
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

inline void sobel_x_kernel(double* kernel) {
    unsafe {
        kernel[0] = 0.0 - 1.0; kernel[1] = 0.0; kernel[2] = 1.0;
        kernel[3] = 0.0 - 2.0; kernel[4] = 0.0; kernel[5] = 2.0;
        kernel[6] = 0.0 - 1.0; kernel[7] = 0.0; kernel[8] = 1.0;
    }
}

inline void sobel_y_kernel(double* kernel) {
    unsafe {
        kernel[0] = 0.0 - 1.0; kernel[1] = 0.0 - 2.0; kernel[2] = 0.0 - 1.0;
        kernel[3] = 0.0;       kernel[4] = 0.0;       kernel[5] = 0.0;
        kernel[6] = 1.0;       kernel[7] = 2.0;       kernel[8] = 1.0;
    }
}

} // namespace std
