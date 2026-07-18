// SafeC Standard Library — Continuous Wavelet Transform (Morlet wavelet)
//
// The CQT (cqt.h) picks logarithmically-spaced *frequency* bins; the CWT
// instead picks a set of *scales* directly (the caller chooses them,
// typically log-spaced for the same musically/perceptually-relevant
// reason the CQT uses log-spaced bins) and correlates a scaled/shifted
// copy of a mother wavelet against the signal at every sample position,
// producing a full time-scale "scalogram" rather than one value per
// frame. The Morlet wavelet — a complex sinusoid windowed by a Gaussian,
// psi(t) = pi^-0.25 * exp(-t^2/2) * exp(i*w0*t) — is the standard choice
// for time-frequency analysis (its Gaussian envelope gives it the best
// achievable time-frequency localization for a sinusoid-based wavelet).
//
// A scale s corresponds to a center frequency of approximately
// f = w0*fs / (2*pi*s) Hz (fs = the signal's sample rate) — smaller scales
// resolve higher frequencies with finer time resolution, larger scales
// resolve lower frequencies with coarser time resolution (the same
// time/frequency trade-off the CQT's per-bin window lengths express,
// just parameterized directly by scale here instead of by bin index).
//
// Direct (not FFT-accelerated) correlation, kernel support truncated to
// +/-4 standard deviations of the Gaussian envelope (negligible energy
// beyond that) — O(numScales * signalLen * scale) in the worst case;
// fine for offline/analysis use.
#pragma once
#include <std/dsp/complex_dsp.h>

namespace std {

// 'scales' has 'numScales' entries (each > 0); 'w0' is the Morlet
// wavelet's center frequency parameter (5-6 is the standard choice — high
// enough to satisfy the admissibility condition without truncating the
// wavelet's own oscillation). 'out' must have room for
// numScales*signalLen Complex values, row-major by scale (row si covers
// out[si*signalLen .. si*signalLen+signalLen-1]).
void cwt_morlet(const double* signal, unsigned long signalLen,
                 const double* scales, unsigned long numScales,
                 double w0, struct Complex* out);

} // namespace std
