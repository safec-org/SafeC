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
        window_hann(w, n);
    } else if (windowType == STFT_WIN_HAMMING) {
        window_hamming(w, n);
    } else if (windowType == STFT_WIN_BLACKMAN) {
        window_blackman(w, n);
    } else {
        window_rectangular(w, n);
    }
}

unsigned long stft_num_frames(unsigned long signalLen, unsigned long frameSize, unsigned long hopSize) {
    if (signalLen < frameSize || hopSize == 0UL) { return 0UL; }
    return (signalLen - frameSize) / hopSize + 1UL;
}

void stft_forward(const double* signal, unsigned long signalLen,
                   unsigned long frameSize, unsigned long hopSize,
                   int windowType, struct Complex* out) {
    double* win;
    unsafe { win = (double*)alloc(frameSize * (unsigned long)sizeof(double)); }
    stft_fill_window_(win, frameSize, windowType);

    unsigned long numFrames = stft_num_frames(signalLen, frameSize, hopSize);
    unsigned long f = 0UL;
    while (f < numFrames) {
        unsigned long base = f * hopSize;
        unsigned long i = 0UL;
        while (i < frameSize) {
            double s;
            double w;
            unsafe {
                s = signal[base + i];
                w = win[i];
                out[f * frameSize + i] = complex_new(s * w, 0.0);
            }
            i = i + 1UL;
        }
        unsafe { fft(out + f * frameSize, frameSize); }
        f = f + 1UL;
    }

    unsafe { dealloc((void*)win); }
}

void stft_inverse(const struct Complex* frames, unsigned long numFrames,
                   unsigned long frameSize, unsigned long hopSize,
                   int windowType, double* outSignal) {
    if (numFrames == 0UL) { return; }
    double* win;
    unsafe { win = (double*)alloc(frameSize * (unsigned long)sizeof(double)); }
    stft_fill_window_(win, frameSize, windowType);

    unsigned long outLen = (numFrames - 1UL) * hopSize + frameSize;
    double* winSumSq;
    unsafe { winSumSq = (double*)alloc(outLen * (unsigned long)sizeof(double)); }
    unsigned long z = 0UL;
    while (z < outLen) {
        unsafe { winSumSq[z] = 0.0; }
        z = z + 1UL;
    }

    struct Complex* frameBuf;
    unsafe { frameBuf = (struct Complex*)alloc(frameSize * (unsigned long)sizeof(struct Complex)); }

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
        dealloc((void*)win);
        dealloc((void*)winSumSq);
        dealloc((void*)frameBuf);
    }
}

double mr_stft_loss(const double* x, const double* y, unsigned long len,
                     const unsigned long* frameSizes, unsigned long numResolutions) {
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
                sx = (struct Complex*)alloc(total * (unsigned long)sizeof(struct Complex));
                sy = (struct Complex*)alloc(total * (unsigned long)sizeof(struct Complex));
            }
            stft_forward(x, len, frameSize, hop, STFT_WIN_HANN, sx);
            stft_forward(y, len, frameSize, hop, STFT_WIN_HANN, sy);

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
                dealloc((void*)sx);
                dealloc((void*)sy);
            }
        }
        r = r + 1UL;
    }

    if (validCount == 0UL) { return 0.0; }
    return totalLoss / (double)validCount;
}

} // namespace std
