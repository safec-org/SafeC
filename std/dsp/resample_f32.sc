// SafeC Standard Library — resampling of float samples (see resample.h)
#pragma once
#include <std/simd/cpu.h>
#include <std/simd/simd.h>
#include <std/sync/parallel.h>
#include <std/dsp/resample.h>
#include <std/dsp/fixed.h>
#include <std/math.h>

#define SAFEC_DSP_PI_RESAMPLE 3.14159265358979323846

namespace std {

#define RS_T        float
#define RS_NEAREST  resample_nearest_f32_n_
#define RS_LINEAR   resample_linear_f32_n_
#define RS_SINC     resample_sinc_f32_n_
#include <std/dsp/resample_core.h>

// ── Slice API (the public signatures; the (reference, length) cores above
// are the *_n_ functions) ─────────────────────────────────────────────────────

void resample_nearest_f32([]const float in, []float out) {
    if (in.len == 0UL || out.len == 0UL) return;
    unsafe { resample_nearest_f32_n_((const &float)in.ptr, in.len, (&float)out.ptr, out.len); }
}

void resample_linear_f32([]const float in, []float out) {
    if (in.len == 0UL || out.len == 0UL) return;
    unsafe { resample_linear_f32_n_((const &float)in.ptr, in.len, (&float)out.ptr, out.len); }
}

void resample_sinc_f32([]const float in, []float out, unsigned long kernelHalfWidth) {
    if (in.len == 0UL || out.len == 0UL) return;
    unsafe { resample_sinc_f32_n_((const &float)in.ptr, in.len, (&float)out.ptr, out.len, kernelHalfWidth); }
}

} // namespace std
