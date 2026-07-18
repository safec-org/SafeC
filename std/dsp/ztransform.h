// SafeC Standard Library — General N-th order Laplace-to-Z (bilinear)
// transform + Z-domain frequency response evaluator
//
// biquad.h's bilinear_2nd_order and biquad_response_mag/phase cover the
// 2nd-order case (the common one — cascaded biquads build any even order
// in practice). This file generalizes both to arbitrary order M, for the
// less common case of designing/analyzing a single higher-order IIR
// section directly instead of as a cascade of biquads: an analog
// prototype H(s) = (sum_{k=0}^{M} B[k] s^k) / (sum_{k=0}^{M} A[k] s^k),
// via the same substitution s = 2*fs*(z-1)/(z+1) bilinear_2nd_order uses,
// producing bOut/aOut directly in the negative-power-of-z form
// filter.h's IirFilter/FIirFilter already expect (H(z) = (sum b[i] z^-i)
// / (sum a[i] z^-i)) — bilinear_nth_order's output can be handed straight
// to iir_init.
//
// No polynomial root-finder is included (finding a filter's poles/zeros
// individually needs an iterative eigenvalue or Durand-Kerner-type
// algorithm — a separate, substantial piece of numerical code that isn't
// needed for either designing a filter via the bilinear transform or
// evaluating its frequency response, both of which work directly on the
// coefficient arrays).
#pragma once
#include <std/dsp/fixed.h>

namespace std {

// B, A: order+1 analog (s-domain) coefficients each, ascending powers of
// s (B[0]/A[0] is the constant term, B[order]/A[order] is the s^order
// term). bOut, aOut: order+1 digital (z-domain) coefficients each,
// written in the negative-power-of-z convention (bOut[0]/aOut[0] is the
// z^0 term, bOut[order]/aOut[order] is the z^-order term), normalized so
// aOut[0] = 1 — the same normalization bilinear_2nd_order's a0-divide-out
// already uses, and what IirFilter expects.
void bilinear_nth_order(const double* B, const double* A, unsigned long order,
                         double fs, double* bOut, double* aOut);

// Z-domain frequency response of an order-'order' digital filter
// (b[]/a[] in the negative-power-of-z convention above) at 'freqHz' —
// the general-order counterpart of biquad_response_mag/phase.
double ztransform_response_mag(const double* b, const double* a, unsigned long order,
                                double fs, double freqHz);
double ztransform_response_phase(const double* b, const double* a, unsigned long order,
                                  double fs, double freqHz);

} // namespace std
