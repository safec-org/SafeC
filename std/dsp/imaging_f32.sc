// SafeC Standard Library — single-precision imaging (see imaging.h)
#pragma once
#include <std/simd/cpu.h>
#include <std/dsp/imaging.h>
#include <std/dsp/complex_dsp.h>
#include <std/dsp/fft.h>
#include <std/math.h>
#include <std/mem.h>
#include <std/sync/parallel.h>

namespace std {

#define IM_R          float
#define IM_C          struct Complex32
#define IM_CREF       Complex32
#define IM_FFT_BATCH  fft_batch_f32
#define IM_CONV2D     conv2d_f32
#define IM_FFT2D      fft2d_f32
#define IM_IFFT2D     ifft2d_f32
#include <std/dsp/imaging_core.h>

} // namespace std
