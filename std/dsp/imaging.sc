// SafeC Standard Library — Imaging implementation (see imaging.h)
#pragma once
#include <std/dsp/imaging.h>
#include <std/dsp/complex_dsp.h>
#include <std/dsp/fft.h>
#include <std/math.h>
#include <std/mem.h>
#include <std/sync/parallel.h>

namespace std {

void conv2d(const &double img, unsigned long h, unsigned long w,
            const &double kernel, unsigned long kh, unsigned long kw, &double out) {
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

// 2D transform = row transforms (fft_batch), then column transforms. The
// columns are gathered 8 at a time into a contiguous buffer, transformed and
// scattered back, so each pass over the image reads whole cache lines; groups
// of columns run on the thread pool.
#define FFT2D_COLS 8UL

struct Fft2dJob {
    struct Complex* img;
    unsigned long   h;
    unsigned long   w;
    int             inv;
};

static void fft2d_cols_(&Fft2dJob j, unsigned long lo, unsigned long hi) {
    unsafe {
        struct Complex* buf = (struct Complex*)alloc_raw_(FFT2D_COLS * j.h * (unsigned long)sizeof(struct Complex));
        if (buf == (struct Complex*)0) return;
        unsigned long g = lo;
        while (g < hi) {
            unsigned long x0 = g * FFT2D_COLS;
            unsigned long nc = j.w - x0 < FFT2D_COLS ? j.w - x0 : FFT2D_COLS;
            unsigned long y = 0UL;
            while (y < j.h) {
                const struct Complex* row = &j.img[y * j.w + x0];
                unsigned long c = 0UL;
                while (c < nc) { buf[c * j.h + y] = row[c]; c = c + 1UL; }
                y = y + 1UL;
            }
            unsigned long c = 0UL;
            while (c < nc) {
                if (j.inv) ifft_serial(&buf[c * j.h], j.h);
                else fft_serial(&buf[c * j.h], j.h);
                c = c + 1UL;
            }
            y = 0UL;
            while (y < j.h) {
                struct Complex* row = &j.img[y * j.w + x0];
                c = 0UL;
                while (c < nc) { row[c] = buf[c * j.h + y]; c = c + 1UL; }
                y = y + 1UL;
            }
            g = g + 1UL;
        }
        dealloc_raw_((void*)buf);
    }
}

static void fft2d_run_(&Complex img, unsigned long h, unsigned long w, int inv) {
    if (h == 0UL || w == 0UL) return;
    fft_batch(img, w, h, inv);
    if (!dsp_is_pow2(h)) return;
    struct Fft2dJob j;
    unsafe { j.img = (struct Complex*)img; }
    j.h = h;
    j.w = w;
    j.inv = inv;
    unsigned long groups = (w + FFT2D_COLS - 1UL) / FFT2D_COLS;
    if (h * w >= 32768UL) parallel_for(groups, 1UL, fft2d_cols_, &j);
    else fft2d_cols_(&j, 0UL, groups);
}

void fft2d(&Complex img, unsigned long h, unsigned long w) { fft2d_run_(img, h, w, 0); }

void ifft2d(&Complex img, unsigned long h, unsigned long w) { fft2d_run_(img, h, w, 1); }

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
