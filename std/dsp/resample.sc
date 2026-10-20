// SafeC Standard Library — Resampling implementation (see resample.h)
#pragma once
#include <std/simd/cpu.h>
#include <std/simd/simd.h>
#include <std/sync/parallel.h>
#include <std/dsp/resample.h>
#include <std/dsp/fixed.h>
#include <std/math.h>

#define SAFEC_DSP_PI_RESAMPLE 3.14159265358979323846

namespace std {

// The shared core (resample_core.h) on double samples; resample_f32.sc
// builds it on float samples.
#define RS_T        double
#define RS_NEAREST  resample_nearest
#define RS_LINEAR   resample_linear
#define RS_SINC     resample_sinc
#include <std/dsp/resample_core.h>

void fresample_nearest(const &Fixed in, unsigned long inLen, &Fixed out, unsigned long outLen) {
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

void fresample_linear(const &Fixed in, unsigned long inLen, &Fixed out, unsigned long outLen) {
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
