#pragma once
// SafeC Standard Library — FFT on AMD GPUs (hipFFT / rocFFT).
//
// GPU counterparts of fft.h and imaging.h's fft2d, plus device-resident
// fp32 transforms on gpu_rocm.h's RocmBufs. All work is queued on
// gpu_rocm's stream, so it orders with the RocmBuf ops there. Same API as
// dsp/fft_cuda.h, with rocm_ in place of cuda_.
//
// Unlike fft.h these accept any length n >= 1, not only powers of two
// (rocFFT handles mixed radices and, via Bluestein, primes). Each
// (kind, size, batch) gets a hipFFT plan the first time it is used, kept in
// a small cache; make that first call outside graph capture.
// Link with: -lhipfft plus gpu_rocm's -lamdhip64 -lhiprtc -lrocblas.
#include <std/dsp/complex_dsp.h>
#include <std/ml/gpu_rocm.h>

namespace std {

// ── Host arrays, double precision (upload → transform → read back) ──────────
// 1 on success, 0 on failure (no GPU, n == 0, hipFFT error). Same
// conventions as fft.h: forward is unnormalized, inverse includes 1/N.

int rocm_fft(&Complex data, unsigned long n);
int rocm_ifft(&Complex data, unsigned long n);
// 'batch' independent length-n transforms stored back to back, in one call.
int rocm_fft_batch(&Complex data, unsigned long n, unsigned long batch, int inverse);
// 2D transform of a row-major h x w image (imaging.h's fft2d / ifft2d).
int rocm_fft2d(&Complex img, unsigned long h, unsigned long w);
int rocm_ifft2d(&Complex img, unsigned long h, unsigned long w);

// ── Queued device ops, fp32 (1 = enqueued) ───────────────────────────────────
// Complex RocmBufs hold interleaved (re, im) float pairs, so a buffer of
// k complex values has len >= 2k. Inverse transforms include 1/n.

// batch x n complex -> batch x n complex. in == out is allowed.
int rocm_fft_c2c(struct RocmBuf in, struct RocmBuf out, unsigned long n, unsigned long batch, int inverse);
// batch x n real -> batch x (n/2 + 1) complex (the non-redundant half).
int rocm_fft_r2c(struct RocmBuf in, struct RocmBuf out, unsigned long n, unsigned long batch);
// batch x (n/2 + 1) complex -> batch x n real. May overwrite 'in' (rocFFT's
// complex-to-real transforms use their input as scratch).
int rocm_fft_c2r(struct RocmBuf in, struct RocmBuf out, unsigned long n, unsigned long batch);

} // namespace std
