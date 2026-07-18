// SafeC Standard Library — Minimum-phase reconstruction (real cepstrum
// method)
//
// Given an impulse response x[n] (e.g. a linear-phase FIR designed purely
// from a target magnitude response), minimum_phase produces a different
// impulse response with the *same magnitude spectrum* but the minimum
// possible group delay/phase — the standard "minimum-phase equivalent"
// used to halve a linear-phase filter's latency for the same magnitude
// response. Classic homomorphic (cepstral) algorithm: take the log-
// magnitude spectrum, transform it back to a "real cepstrum", fold the
// anti-causal half onto the causal half (this is what forces the result
// causal and minimum-phase — a signal is minimum-phase iff its complex
// cepstrum is zero for n < 0), then exponentiate back through the
// spectral domain.
#pragma once
#include <std/dsp/complex_dsp.h>

namespace std {

// 'x' has 'n' samples (n <= fftLen); 'fftLen' must be a power of two (see
// fft.h) and should be comfortably larger than n for accurate cepstral
// liftering (a common rule of thumb: at least 4-8x n). 'outMinPhase' must
// have room for 'fftLen' samples — the full-length minimum-phase impulse
// response (its energy concentrates near the start; callers typically
// keep only the first n-ish samples).
void minimum_phase(const double* x, unsigned long n, unsigned long fftLen, double* outMinPhase);

} // namespace std
