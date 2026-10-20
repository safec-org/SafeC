#pragma once
// SafeC Standard Library — FFT on NVIDIA GPUs (cuFFT).
//
// GPU counterparts of fft.h and imaging.h's fft2d, plus device-resident
// fp32 transforms on gpu_cuda.h's CudaBufs. All work is queued on
// gpu_cuda's stream, so it orders with the CudaBuf ops there.
//
// Unlike fft.h these accept any length n >= 1, not only powers of two
// (cuFFT handles mixed radices and, via Bluestein, primes). Each
// (kind, size, batch) gets a cuFFT plan the first time it is used, kept in
// a small cache; make that first call outside graph capture.
// Link with: -lcufft plus gpu_cuda's -lcuda -lcublas -lcublasLt -lnvrtc.
#include <std/dsp/complex_dsp.h>
#include <std/ml/gpu_cuda.h>

namespace std {

// ── Host arrays, double precision (upload → transform → read back) ──────────
// 1 on success, 0 on failure (no GPU, n == 0, cuFFT error). Same
// conventions as fft.h: forward is unnormalized, inverse includes 1/N.

int cuda_fft(&Complex data, unsigned long n);
int cuda_ifft(&Complex data, unsigned long n);
// 'batch' independent length-n transforms stored back to back, in one call.
int cuda_fft_batch(&Complex data, unsigned long n, unsigned long batch, int inverse);
// 2D transform of a row-major h x w image (imaging.h's fft2d / ifft2d).
int cuda_fft2d(&Complex img, unsigned long h, unsigned long w);
int cuda_ifft2d(&Complex img, unsigned long h, unsigned long w);

// ── Queued device ops, fp32 (1 = enqueued) ───────────────────────────────────
// Complex CudaBufs hold interleaved (re, im) float pairs, so a buffer of
// k complex values has len >= 2k. Inverse transforms include 1/n.

// batch x n complex -> batch x n complex. in == out is allowed.
int cuda_fft_c2c(struct CudaBuf in, struct CudaBuf out, unsigned long n, unsigned long batch, int inverse);
// batch x n real -> batch x (n/2 + 1) complex (the non-redundant half).
int cuda_fft_r2c(struct CudaBuf in, struct CudaBuf out, unsigned long n, unsigned long batch);
// batch x (n/2 + 1) complex -> batch x n real. Overwrites 'in' (cuFFT's
// complex-to-real transforms use their input as scratch).
int cuda_fft_c2r(struct CudaBuf in, struct CudaBuf out, unsigned long n, unsigned long batch);

} // namespace std
