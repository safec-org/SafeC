// SafeC Standard Library — Short-Time Fourier Transform + Multi-Resolution
// STFT loss
//
// stft_forward slides a windowed frame across the signal and FFTs each one
// (see fft.h — frameSize must be a power of two), producing a spectrogram:
// stft_num_frames(...) frames of frameSize Complex bins each. stft_inverse
// reconstructs the time-domain signal via windowed overlap-add, normalized
// by the accumulated squared analysis window at each output sample — the
// standard robust OLA method, exact whenever the window/hop combination
// satisfies the constant-overlap-add condition (e.g. Hann with 50% or 75%
// overlap) and well-behaved otherwise.
//
// mr_stft_loss computes the "multi-resolution STFT loss" from the neural
// vocoder literature (Yamamoto et al., Parallel WaveGAN, and others): for
// each requested frame size, take the spectral convergence (relative
// Frobenius-norm magnitude error) plus the mean L1 log-magnitude error
// between two signals' spectrograms, then average across resolutions —
// a similarity metric that's sensitive to spectral (not just sample-wise
// time-domain) differences at multiple time/frequency trade-offs at once.
#pragma once
#include <std/dsp/complex_dsp.h>

namespace std {

#define STFT_WIN_RECTANGULAR 0
#define STFT_WIN_HANN 1
#define STFT_WIN_HAMMING 2
#define STFT_WIN_BLACKMAN 3

// Number of full frames stft_forward extracts from a signal of length
// 'signalLen' (a trailing partial frame, if any, is dropped).
unsigned long stft_num_frames(unsigned long signalLen, unsigned long frameSize, unsigned long hopSize);

// 'out' must have room for stft_num_frames(signalLen, frameSize, hopSize)
// * frameSize Complex values (frames stored consecutively).
void stft_forward(const double* signal, unsigned long signalLen,
                   unsigned long frameSize, unsigned long hopSize,
                   int windowType, struct Complex* out);

// 'outSignal' must have room for (numFrames-1)*hopSize + frameSize samples
// and should be zero-initialized by the caller (stft_inverse accumulates
// into it).
void stft_inverse(const struct Complex* frames, unsigned long numFrames,
                   unsigned long frameSize, unsigned long hopSize,
                   int windowType, double* outSignal);

// Multi-resolution STFT loss between signals x and y (same length 'len').
// 'frameSizes' is an array of 'numResolutions' power-of-two frame sizes;
// each resolution uses a Hann window with hop = frameSize/4 (the standard
// convention in the MR-STFT loss literature). Lower is more similar; 0 for
// identical signals.
double mr_stft_loss(const double* x, const double* y, unsigned long len,
                     const unsigned long* frameSizes, unsigned long numResolutions);

} // namespace std
