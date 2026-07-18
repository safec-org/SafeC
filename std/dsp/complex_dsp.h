// SafeC Standard Library — DSP Complex Numbers
//
// Value-type complex numbers built for the rest of std/dsp/'s spectral
// code (fft.h/dft.h's butterflies and twiddle factors, convolution.h's
// frequency-domain multiply) — distinct from std/complex.h's C99-style
// API, which represents a complex number as a float[2]/double[2] out-
// array and writes results through a pointer on every operation. That
// shape matches C99 <complex.h> closely, but every FFT butterfly doing
// several arithmetic ops back-to-back would need a temporary out-array
// per intermediate result; a plain value type with operators (SafeC
// supports operator overloading — see reference/functions.md) reads the
// same way the math does: 'w * a + b', not a chain of cmul_d/cadd_d
// calls each writing through a pointer.
//
// Two variants, matching std/dsp's two numeric domains:
//   struct Complex  — double re/im, for the float-precision DSP path.
//   struct FComplex — Fixed (Q8.24) re/im, for the fixed-point path.
#pragma once
#include <std/dsp/fixed.h>

namespace std {

struct Complex {
    double re;
    double im;

    Complex operator+(Complex other) const;
    Complex operator-(Complex other) const;
    Complex operator*(Complex other) const;
    Complex operator/(Complex other) const;
    Complex neg() const; // negate — SafeC operator overloads aren't distinguished by arity, so unary '-' can't share the 'operator-' name with binary subtraction above

    double  abs() const;   // magnitude: sqrt(re^2 + im^2)
    double  arg() const;   // phase angle: atan2(im, re)
    Complex conj() const;  // complex conjugate: (re, -im)
};

Complex complex_new(double re, double im);

// e^(i*theta) = (cos(theta), sin(theta)) — the form FFT twiddle factors
// and oscillator/DFT kernels are naturally expressed in.
Complex complex_from_polar(double magnitude, double theta);

struct FComplex {
    Fixed re;
    Fixed im;

    FComplex operator+(FComplex other) const;
    FComplex operator-(FComplex other) const;
    FComplex operator*(FComplex other) const;
    FComplex neg() const;

    Fixed     abs() const;
    FComplex  conj() const;
};

FComplex fcomplex_new(Fixed re, Fixed im);

// Converts a double-precision Complex to Q8.24 FComplex and back. Twiddle
// factors and other coefficients computed once (at filter/FFT-plan design
// time) in double precision are quantized to Fixed via these — the same
// design-time-float / runtime-fixed split std/dsp/filter.h's biquad
// cookbook formulas use, and completely standard practice (even hardware
// FFT accelerators precompute twiddle tables offline in floating point).
FComplex complex_to_fixed(Complex c);
Complex  fcomplex_to_float(FComplex c);

} // namespace std
