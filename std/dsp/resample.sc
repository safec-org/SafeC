// SafeC Standard Library — Resampling implementation (see resample.h)
#pragma once
#include <std/dsp/resample.h>
#include <std/dsp/fixed.h>
#include <std/math.h>

#define SAFEC_DSP_PI_RESAMPLE 3.14159265358979323846

namespace std {

void resample_nearest(const double* in, unsigned long inLen, double* out, unsigned long outLen) {
    if (inLen == 0UL || outLen == 0UL) { return; }
    double scale = (double)inLen / (double)outLen;
    unsigned long i = 0UL;
    while (i < outLen) {
        double srcPos = (double)i * scale;
        unsigned long idx = (unsigned long)(srcPos + 0.5);
        if (idx >= inLen) { idx = inLen - 1UL; }
        double v;
        unsafe {
            v = in[idx];
            out[i] = v;
        }
        i = i + 1UL;
    }
}

void resample_linear(const double* in, unsigned long inLen, double* out, unsigned long outLen) {
    if (inLen == 0UL || outLen == 0UL) { return; }
    if (inLen == 1UL) {
        double v;
        unsafe { v = in[0]; }
        unsigned long i = 0UL;
        while (i < outLen) {
            unsafe { out[i] = v; }
            i = i + 1UL;
        }
        return;
    }
    double scale = 0.0;
    if (outLen > 1UL) {
        scale = (double)(inLen - 1UL) / (double)(outLen - 1UL);
    }
    unsigned long i = 0UL;
    while (i < outLen) {
        double srcPos = (double)i * scale;
        unsigned long idx0 = (unsigned long)srcPos;
        unsigned long idx1 = idx0 + 1UL;
        if (idx1 >= inLen) { idx1 = inLen - 1UL; }
        double frac = srcPos - (double)idx0;
        double v0;
        double v1;
        unsafe {
            v0 = in[idx0];
            v1 = in[idx1];
            out[i] = v0 + frac * (v1 - v0);
        }
        i = i + 1UL;
    }
}

static double resample_sinc_(double x) {
    if (x == 0.0) { return 1.0; }
    double px = SAFEC_DSP_PI_RESAMPLE * x;
    return sin_d(px) / px;
}

void resample_sinc(const double* in, unsigned long inLen, double* out, unsigned long outLen,
                    unsigned long kernelHalfWidth) {
    if (inLen == 0UL || outLen == 0UL || kernelHalfWidth == 0UL) { return; }
    double ratio = (double)outLen / (double)inLen;
    double cutoff = ratio;
    if (cutoff > 1.0) { cutoff = 1.0; }
    double scale = (double)inLen / (double)outLen;

    unsigned long i = 0UL;
    while (i < outLen) {
        double srcPos = (double)i * scale;
        unsigned long center = (unsigned long)srcPos;

        unsigned long kStart = 0UL;
        if (center >= kernelHalfWidth) { kStart = center - kernelHalfWidth; }
        unsigned long kEnd = center + kernelHalfWidth;
        if (kEnd >= inLen) { kEnd = inLen - 1UL; }

        double sum = 0.0;
        unsigned long k = kStart;
        while (k <= kEnd) {
            double x = srcPos - (double)k;
            double sincVal = resample_sinc_(x * cutoff) * cutoff;
            double winArg = SAFEC_DSP_PI_RESAMPLE * x / (double)kernelHalfWidth;
            double w = 0.5 * (1.0 + cos_d(winArg));
            double sampleVal;
            unsafe { sampleVal = in[k]; }
            sum = sum + sampleVal * sincVal * w;
            k = k + 1UL;
        }
        unsafe { out[i] = sum; }
        i = i + 1UL;
    }
}

void fresample_nearest(const Fixed* in, unsigned long inLen, Fixed* out, unsigned long outLen) {
    if (inLen == 0UL || outLen == 0UL) { return; }
    double scale = (double)inLen / (double)outLen;
    unsigned long i = 0UL;
    while (i < outLen) {
        double srcPos = (double)i * scale;
        unsigned long idx = (unsigned long)(srcPos + 0.5);
        if (idx >= inLen) { idx = inLen - 1UL; }
        Fixed v;
        unsafe {
            v = in[idx];
            out[i] = v;
        }
        i = i + 1UL;
    }
}

void fresample_linear(const Fixed* in, unsigned long inLen, Fixed* out, unsigned long outLen) {
    if (inLen == 0UL || outLen == 0UL) { return; }
    if (inLen == 1UL) {
        Fixed v;
        unsafe { v = in[0]; }
        unsigned long i = 0UL;
        while (i < outLen) {
            unsafe { out[i] = v; }
            i = i + 1UL;
        }
        return;
    }
    double scale = 0.0;
    if (outLen > 1UL) {
        scale = (double)(inLen - 1UL) / (double)(outLen - 1UL);
    }
    unsigned long i = 0UL;
    while (i < outLen) {
        double srcPos = (double)i * scale;
        unsigned long idx0 = (unsigned long)srcPos;
        unsigned long idx1 = idx0 + 1UL;
        if (idx1 >= inLen) { idx1 = inLen - 1UL; }
        double frac = srcPos - (double)idx0;
        Fixed v0;
        Fixed v1;
        unsafe {
            v0 = in[idx0];
            v1 = in[idx1];
        }
        Fixed fracFixed = fixed_from_float(frac);
        Fixed diff = fixed_sub(v1, v0);
        Fixed interp = fixed_add(v0, fixed_mul(fracFixed, diff));
        unsafe { out[i] = interp; }
        i = i + 1UL;
    }
}

} // namespace std
