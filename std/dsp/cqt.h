// SafeC Standard Library — Constant-Q Transform
//
// Unlike the DFT/FFT (uniform frequency spacing, uniform time resolution
// across all bins), the CQT uses logarithmically-spaced center
// frequencies fk = fmin * 2^(k/binsPerOctave) with a constant ratio
// fk/bandwidth ("Q") at every bin — matching how musical pitch and human
// hearing are both roughly logarithmic in frequency, which is why the
// CQT (not the plain DFT) is the standard choice for chromagrams, pitch
// detection, and other music-analysis tasks. The trade-off is that lower
// bins need longer analysis windows than higher bins (constant Q means
// constant *relative* bandwidth, so absolute bandwidth — and therefore
// the window length needed to resolve it — scales with 1/fk).
//
// This is the direct-correlation form (Brown 1991): each bin's kernel
// (a Hann-windowed complex exponential, own length per bin) is
// correlated directly against the start of 'signal'. It doesn't need an
// FFT and needs no fast-kernel-matrix precomputation, at the cost of
// O(numBins * average-kernel-length) per call — fine for analysis/offline
// use; a production real-time CQT would typically precompute a sparse
// kernel matrix once and reuse it, which isn't done here.
#pragma once
#include <std/dsp/complex_dsp.h>

namespace std {

// 'signal' must have at least enough samples for the lowest bin's
// (longest) kernel — roughly Q*fs/fmin samples, Q = 1/(2^(1/binsPerOctave)-1)
// — or that bin's kernel is silently clipped to signalLen (a degraded,
// not incorrect, result: still valid, just with worse frequency
// resolution than requested for that bin). 'out' must have room for
// 'numBins' Complex values.
void cqt_forward(const double* signal, unsigned long signalLen, double fs,
                  double fmin, unsigned long numBins, unsigned long binsPerOctave,
                  struct Complex* out);

} // namespace std
