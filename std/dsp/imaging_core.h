// SafeC Standard Library — imaging core, written once for both precisions.
//
// Internal: included by imaging.sc (double) and imaging_f32.sc (float),
// each its own translation unit. The includer defines IM_R (real type),
// IM_C / IM_CREF (its complex type as "struct X" / "X"), IM_FFT_BATCH, and
// the public names IM_CONV2D, IM_FFT2D, IM_IFFT2D.
#pragma once

// Each output row is a sum of shifted input rows times one kernel tap:
// row-wide multiply-adds over the x range where the tap stays inside the
// image, which vectorize. Rows run on the thread pool for larger images.
struct Conv2dJob {
    const IM_R* img;
    const IM_R* kernel;
    IM_R*       out;
    unsigned long h;
    unsigned long w;
    unsigned long kh;
    unsigned long kw;
};

static void conv2d_rows_body_(const &Conv2dJob j, unsigned long lo, unsigned long hi) __attribute__((always_inline)) {
    unsafe {
        unsigned long kcy = j.kh / 2UL;
        unsigned long kcx = j.kw / 2UL;
        unsigned long y = lo;
        while (y < hi) {
            IM_R* o = &j.out[y * j.w];
            unsigned long x = 0UL;
            while (x < j.w) { o[x] = (IM_R)0.0; x = x + 1UL; }
            unsigned long ky = 0UL;
            while (ky < j.kh) {
                unsigned long sy = y + ky;
                if (sy >= kcy && sy - kcy < j.h) {
                    const IM_R* src = &j.img[(sy - kcy) * j.w];
                    unsigned long kx = 0UL;
                    while (kx < j.kw) {
                        IM_R kv = j.kernel[ky * j.kw + kx];
                        // srcX = x + kx - kcx must be in [0, w)
                        unsigned long x0 = kx < kcx ? kcx - kx : 0UL;
                        unsigned long x1 = kx > kcx ? (j.w > kx - kcx ? j.w - (kx - kcx) : 0UL) : j.w;
                        if (x0 < x1) {
                            const IM_R* sp = &src[x0 + kx - kcx];
                            IM_R* op = &o[x0];
                            unsigned long cnt = x1 - x0;
                            unsigned long q = 0UL;
                            while (q < cnt) { op[q] = __builtin_fmuladd(kv, sp[q], op[q]); q = q + 1UL; }
                        }
                        kx = kx + 1UL;
                    }
                }
                ky = ky + 1UL;
            }
            y = y + 1UL;
        }
    }
}

static void conv2d_rows_base_(const &Conv2dJob j, unsigned long lo, unsigned long hi) { conv2d_rows_body_(j, lo, hi); }
#if defined(__x86_64__)
static void conv2d_rows_fma_(const &Conv2dJob j, unsigned long lo, unsigned long hi)
    __attribute__((target("avx2,fma"))) { conv2d_rows_body_(j, lo, hi); }
#endif

static void conv2d_rows_(&Conv2dJob j, unsigned long lo, unsigned long hi) {
#if defined(__x86_64__)
    if (simd_level() >= SIMD_AVX2) { conv2d_rows_fma_(j, lo, hi); return; }
#endif
    conv2d_rows_base_(j, lo, hi);
}

void IM_CONV2D(const &IM_R img, unsigned long h, unsigned long w,
            const &IM_R kernel, unsigned long kh, unsigned long kw, &IM_R out) {
    struct Conv2dJob j;
    unsafe {
        j.img = (const IM_R*)img;
        j.kernel = (const IM_R*)kernel;
        j.out = (IM_R*)out;
    }
    j.h = h; j.w = w; j.kh = kh; j.kw = kw;
    if (h > 1UL && h * w * kh * kw >= 262144UL) parallel_for(h, 1UL, conv2d_rows_, &j);
    else conv2d_rows_(&j, 0UL, h);
}

// 2D transform = row transforms (fft_batch), then column transforms. The
// columns are gathered 8 at a time into a contiguous buffer, transformed and
// scattered back, so each pass over the image reads whole cache lines; groups
// of columns run on the thread pool.
#define FFT2D_COLS 8UL

struct Fft2dJob {
    IM_C* img;
    unsigned long   h;
    unsigned long   w;
    int             inv;
};

static void fft2d_cols_(&Fft2dJob j, unsigned long lo, unsigned long hi) {
    unsafe {
        IM_C* buf = (IM_C*)alloc_raw_(FFT2D_COLS * j.h * (unsigned long)sizeof(IM_C));
        if (buf == (IM_C*)0) return;
        unsigned long g = lo;
        while (g < hi) {
            unsigned long x0 = g * FFT2D_COLS;
            unsigned long nc = j.w - x0 < FFT2D_COLS ? j.w - x0 : FFT2D_COLS;
            unsigned long y = 0UL;
            while (y < j.h) {
                const IM_C* row = &j.img[y * j.w + x0];
                unsigned long c = 0UL;
                while (c < nc) { buf[c * j.h + y] = row[c]; c = c + 1UL; }
                y = y + 1UL;
            }
            // Already on the pool, so this runs the group's columns inline.
            IM_FFT_BATCH(buf, j.h, nc, j.inv);
            y = 0UL;
            while (y < j.h) {
                IM_C* row = &j.img[y * j.w + x0];
                unsigned long c = 0UL;
                while (c < nc) { row[c] = buf[c * j.h + y]; c = c + 1UL; }
                y = y + 1UL;
            }
            g = g + 1UL;
        }
        dealloc_raw_((void*)buf);
    }
}

static void fft2d_run_(&IM_CREF img, unsigned long h, unsigned long w, int inv) {
    if (h == 0UL || w == 0UL) return;
    IM_FFT_BATCH(img, w, h, inv);
    if (!dsp_is_pow2(h)) return;
    struct Fft2dJob j;
    unsafe { j.img = (IM_C*)img; }
    j.h = h;
    j.w = w;
    j.inv = inv;
    unsigned long groups = (w + FFT2D_COLS - 1UL) / FFT2D_COLS;
    if (h * w >= 32768UL) parallel_for(groups, 1UL, fft2d_cols_, &j);
    else fft2d_cols_(&j, 0UL, groups);
}

void IM_FFT2D(&IM_CREF img, unsigned long h, unsigned long w) { fft2d_run_(img, h, w, 0); }

void IM_IFFT2D(&IM_CREF img, unsigned long h, unsigned long w) { fft2d_run_(img, h, w, 1); }
