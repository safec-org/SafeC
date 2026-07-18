// SafeC Standard Library — 2D DSP (imaging)
//
// The 2D analogues of std/dsp's 1D building blocks: conv2d is 2D
// convolution with zero-padded ("same" size) boundaries; fft2d/ifft2d
// are the row-column-decomposition 2D FFT (apply the existing 1D fft.h
// transform to every row, then to every column — the standard way to get
// a 2D FFT from a 1D one, exact because the 2D DFT is separable); the
// kernel generators are the handful of convolution kernels almost every
// image pipeline needs (Gaussian blur, Sobel edge gradients).
#pragma once
#include <std/dsp/complex_dsp.h>

namespace std {

// 'img' is h*w row-major; 'kernel' is kh*kw row-major (kh, kw should be
// odd so the kernel has a well-defined center tap); 'out' must have room
// for h*w doubles (same size as 'img' — out-of-bounds kernel taps read as
// zero, i.e. zero-padded boundary handling).
void conv2d(const double* img, unsigned long h, unsigned long w,
            const double* kernel, unsigned long kh, unsigned long kw, double* out);

// In-place 2D FFT/IFFT via row-column decomposition. Both h and w must
// be powers of two (see fft.h).
void fft2d(struct Complex* img, unsigned long h, unsigned long w);
void ifft2d(struct Complex* img, unsigned long h, unsigned long w);

// Fills a size*size buffer (size should be odd) with a normalized
// (sums to 1) 2D Gaussian kernel of standard deviation 'sigma'.
void gaussian_kernel(double* kernel, unsigned long size, double sigma);

// Fixed 3x3 Sobel gradient kernels (horizontal/vertical edge response);
// 'kernel' must have room for 9 doubles, row-major.
void sobel_x_kernel(double* kernel);
void sobel_y_kernel(double* kernel);

} // namespace std
