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
#define RS_NEAREST  resample_nearest_f32
#define RS_LINEAR   resample_linear_f32
#define RS_SINC     resample_sinc_f32
#include <std/dsp/resample_core.h>

} // namespace std
