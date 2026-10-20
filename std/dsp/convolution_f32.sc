// SafeC Standard Library — single-precision convolution (see convolution.h)
#pragma once
#include <std/dsp/convolution.h>
#include <std/dsp/fft.h>
#include <std/dsp/complex_dsp.h>
#include <std/dsp/dsp.h>
#include <std/mem.h>
#include <std/sync/parallel.h>
#include <std/simd/cpu.h>

namespace std {

#define CV_R          float
#define CV_C          struct Complex32
#define CV_CNEW       complex32_new
#define CV_DOT        dsp_dot_f32
#define CV_SDOT       dsp_sliding_dot_f32
#define CV_FFT        fft_f32
#define CV_IFFT       ifft_f32
#define CV_FFT_BATCH  fft_batch_f32
#define CV_DIRECT     conv_direct_f32
#define CV_FFTCONV    conv_fft_f32
#include <std/dsp/conv_core.h>

} // namespace std
