// SafeC Standard Library — STFT / MR-STFT implementation (see stft.h)
#pragma once
#include <std/dsp/stft.h>
#include <std/dsp/complex_dsp.h>
#include <std/dsp/fft.h>
#include <std/dsp/window.h>
#include <std/math.h>
#include <std/mem.h>

namespace std {

static void stft_fill_window_(double* w, unsigned long n, int windowType) {
    if (windowType == STFT_WIN_HANN) {
        unsafe { window_hann(w, n); }
    } else if (windowType == STFT_WIN_HAMMING) {
        unsafe { window_hamming(w, n); }
    } else if (windowType == STFT_WIN_BLACKMAN) {
        unsafe { window_blackman(w, n); }
    } else {
        unsafe { window_rectangular(w, n); }
    }
}

unsigned long stft_num_frames(unsigned long signalLen, unsigned long frameSize, unsigned long hopSize) {
    if (signalLen < frameSize || hopSize == 0UL) { return 0UL; }
    return (signalLen - frameSize) / hopSize + 1UL;
}

void stft_forward(const &double signal, unsigned long signalLen,
                   unsigned long frameSize, unsigned long hopSize,
                   int windowType, &Complex out_ref) { struct Complex* out = out_ref;
    double* win;
    unsafe { win = (double*)alloc_raw_(frameSize * (unsigned long)sizeof(double)); }
    stft_fill_window_(win, frameSize, windowType);

    unsigned long numFrames = stft_num_frames(signalLen, frameSize, hopSize);
    unsigned long f = 0UL;
    while (f < numFrames) {
        unsigned long base = f * hopSize;
        double* row;
        unsafe { row = (double*)&out[f * frameSize]; }   // the real frame, packed in the row's first half
        unsigned long i = 0UL;
        while (i < frameSize) {
            double s;
            double w;
            unsafe {
                s = signal[base + i];
                w = win[i];
                row[i] = s * w;
            }
            i = i + 1UL;
        }
        f = f + 1UL;
    }
    // All frames at once, as real transforms (half the work of complex
    // ones), spread over the thread pool (stft_forward is an
    // offline analysis; streaming code should window and fft_serial a frame
    // at a time instead).
    unsafe { fft_real_batch_inplace(out, frameSize, numFrames); }

    unsafe { dealloc_raw_((void*)win); }
}

void stft_inverse(const &Complex frames, unsigned long numFrames,
                   unsigned long frameSize, unsigned long hopSize,
                   int windowType, &double outSignal) {
    if (numFrames == 0UL) { return; }
    double* win;
    unsafe { win = (double*)alloc_raw_(frameSize * (unsigned long)sizeof(double)); }
    stft_fill_window_(win, frameSize, windowType);

    unsigned long outLen = (numFrames - 1UL) * hopSize + frameSize;
    double* winSumSq;
    unsafe { winSumSq = (double*)alloc_raw_(outLen * (unsigned long)sizeof(double)); }
    unsigned long z = 0UL;
    while (z < outLen) {
        unsafe { winSumSq[z] = 0.0; }
        z = z + 1UL;
    }

    struct Complex* frameBuf;
    unsafe { frameBuf = (struct Complex*)alloc_raw_(frameSize * (unsigned long)sizeof(struct Complex)); }

    unsigned long f = 0UL;
    while (f < numFrames) {
        unsigned long base = f * hopSize;
        unsigned long i = 0UL;
        while (i < frameSize) {
            unsafe { frameBuf[i] = frames[f * frameSize + i]; }
            i = i + 1UL;
        }
        unsafe { ifft(frameBuf, frameSize); }
        i = 0UL;
        while (i < frameSize) {
            double w;
            struct Complex c;
            unsafe { w = win[i]; c = frameBuf[i]; }
            double sample = c.re * w;
            unsafe {
                outSignal[base + i] = outSignal[base + i] + sample;
                winSumSq[base + i] = winSumSq[base + i] + w * w;
            }
            i = i + 1UL;
        }
        f = f + 1UL;
    }

    unsigned long n = 0UL;
    while (n < outLen) {
        double denom;
        unsafe { denom = winSumSq[n]; }
        if (denom > 1e-8) {
            unsafe { outSignal[n] = outSignal[n] / denom; }
        }
        n = n + 1UL;
    }

    unsafe {
        dealloc_raw_((void*)win);
        dealloc_raw_((void*)winSumSq);
        dealloc_raw_((void*)frameBuf);
    }
}

// ── Single precision ─────────────────────────────────────────────────────────

static void stft_fill_window32_(float* w, unsigned long n, int windowType) {
    if (windowType == STFT_WIN_HANN) {
        unsafe { window_hann_f32(w, n); }
    } else if (windowType == STFT_WIN_HAMMING) {
        unsafe { window_hamming_f32(w, n); }
    } else if (windowType == STFT_WIN_BLACKMAN) {
        unsafe { window_blackman_f32(w, n); }
    } else {
        unsafe { window_rectangular_f32(w, n); }
    }
}

void stft_forward_f32(const &float signal, unsigned long signalLen,
                   unsigned long frameSize, unsigned long hopSize,
                   int windowType, &Complex32 out_ref) { struct Complex32* out = out_ref;
    float* win;
    unsafe { win = (float*)alloc_raw_(frameSize * (unsigned long)sizeof(float)); }
    stft_fill_window32_(win, frameSize, windowType);

    unsigned long numFrames = stft_num_frames(signalLen, frameSize, hopSize);
    unsigned long f = 0UL;
    while (f < numFrames) {
        unsigned long base = f * hopSize;
        float* row;
        unsafe { row = (float*)&out[f * frameSize]; }
        unsigned long i = 0UL;
        while (i < frameSize) {
            float s;
            float w;
            unsafe {
                s = signal[base + i];
                w = win[i];
                row[i] = s * w;
            }
            i = i + 1UL;
        }
        f = f + 1UL;
    }
    // All frames at once, as real transforms (half the work of complex
    // ones), spread over the thread pool (stft_forward is an
    // offline analysis; streaming code should window and fft_serial a frame
    // at a time instead).
    unsafe { fft_real_batch_inplace_f32(out, frameSize, numFrames); }

    unsafe { dealloc_raw_((void*)win); }
}

void stft_inverse_f32(const &Complex32 frames, unsigned long numFrames,
                   unsigned long frameSize, unsigned long hopSize,
                   int windowType, &float outSignal) {
    if (numFrames == 0UL) { return; }
    float* win;
    unsafe { win = (float*)alloc_raw_(frameSize * (unsigned long)sizeof(float)); }
    stft_fill_window32_(win, frameSize, windowType);

    unsigned long outLen = (numFrames - 1UL) * hopSize + frameSize;
    float* winSumSq;
    unsafe { winSumSq = (float*)alloc_raw_(outLen * (unsigned long)sizeof(float)); }
    unsigned long z = 0UL;
    while (z < outLen) {
        unsafe { winSumSq[z] = 0.0; }
        z = z + 1UL;
    }

    struct Complex32* frameBuf;
    unsafe { frameBuf = (struct Complex32*)alloc_raw_(frameSize * (unsigned long)sizeof(struct Complex32)); }

    unsigned long f = 0UL;
    while (f < numFrames) {
        unsigned long base = f * hopSize;
        unsigned long i = 0UL;
        while (i < frameSize) {
            unsafe { frameBuf[i] = frames[f * frameSize + i]; }
            i = i + 1UL;
        }
        unsafe { ifft_f32(frameBuf, frameSize); }
        i = 0UL;
        while (i < frameSize) {
            float w;
            struct Complex32 c;
            unsafe { w = win[i]; c = frameBuf[i]; }
            float sample = c.re * w;
            unsafe {
                outSignal[base + i] = outSignal[base + i] + sample;
                winSumSq[base + i] = winSumSq[base + i] + w * w;
            }
            i = i + 1UL;
        }
        f = f + 1UL;
    }

    unsigned long n = 0UL;
    while (n < outLen) {
        float denom;
        unsafe { denom = winSumSq[n]; }
        if (denom > (float)1e-8) {
            unsafe { outSignal[n] = outSignal[n] / denom; }
        }
        n = n + 1UL;
    }

    unsafe {
        dealloc_raw_((void*)win);
        dealloc_raw_((void*)winSumSq);
        dealloc_raw_((void*)frameBuf);
    }
}

double mr_stft_loss(const &double x, const &double y, unsigned long len,
                     const &unsigned long frameSizes, unsigned long numResolutions) {
    if (numResolutions == 0UL) { return 0.0; }
    double totalLoss = 0.0;
    unsigned long validCount = 0UL;
    unsigned long r = 0UL;
    while (r < numResolutions) {
        unsigned long frameSize;
        unsafe { frameSize = frameSizes[r]; }
        unsigned long hop = frameSize / 4UL;
        if (hop == 0UL) { hop = 1UL; }
        unsigned long numFrames = stft_num_frames(len, frameSize, hop);
        if (numFrames > 0UL) {
            unsigned long total = numFrames * frameSize;
            struct Complex* sx;
            struct Complex* sy;
            unsafe {
                sx = (struct Complex*)alloc_raw_(total * (unsigned long)sizeof(struct Complex));
                sy = (struct Complex*)alloc_raw_(total * (unsigned long)sizeof(struct Complex));
            }
            unsafe { stft_forward(x, len, frameSize, hop, STFT_WIN_HANN, sx); }
            unsafe { stft_forward(y, len, frameSize, hop, STFT_WIN_HANN, sy); }

            double diffSq = 0.0;
            double xSq = 0.0;
            double logDiffSum = 0.0;
            unsigned long i = 0UL;
            while (i < total) {
                struct Complex cx;
                struct Complex cy;
                unsafe { cx = sx[i]; cy = sy[i]; }
                double mx = cx.abs();
                double my = cy.abs();
                double d = mx - my;
                diffSq = diffSq + d * d;
                xSq = xSq + mx * mx;
                double lx = log_d(mx + 1e-7);
                double ly = log_d(my + 1e-7);
                double ld = lx - ly;
                if (ld < 0.0) { ld = 0.0 - ld; }
                logDiffSum = logDiffSum + ld;
                i = i + 1UL;
            }

            double specConv = sqrt_d(diffSq) / (sqrt_d(xSq) + 1e-7);
            double logMagLoss = logDiffSum / (double)total;
            totalLoss = totalLoss + specConv + logMagLoss;
            validCount = validCount + 1UL;

            unsafe {
                dealloc_raw_((void*)sx);
                dealloc_raw_((void*)sy);
            }
        }
        r = r + 1UL;
    }

    if (validCount == 0UL) { return 0.0; }
    return totalLoss / (double)validCount;
}

} // namespace std
