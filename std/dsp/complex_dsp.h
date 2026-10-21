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

    static Complex new(double re, double im);

    // magnitude * e^(i*theta) = magnitude * (cos(theta), sin(theta)) — the
    // form FFT twiddle factors and oscillator/DFT kernels are naturally
    // expressed in.
    static Complex polar(double magnitude, double theta);
};

// Single-precision complex: the float counterpart of Complex, with the same
// layout (re then im). The _f32 DSP functions (fft_f32, stft_forward_f32,
// ...) take it. Not to be confused with FComplex below, which is fixed point.
struct Complex32 {
    float re;
    float im;

    Complex32 operator+(Complex32 other) const;
    Complex32 operator-(Complex32 other) const;
    Complex32 operator*(Complex32 other) const;
    Complex32 operator/(Complex32 other) const;
    Complex32 neg() const;

    float     abs() const;
    float     arg() const;
    Complex32 conj() const;

    static Complex32 new(float re, float im);
    static Complex32 polar(float magnitude, float theta);
    static Complex32 from(Complex c);    // rounds each part to float
    Complex          to_complex() const; // back to double (exact)
};

struct FComplex {
    Fixed re;
    Fixed im;

    FComplex operator+(FComplex other) const;
    FComplex operator-(FComplex other) const;
    FComplex operator*(FComplex other) const;
    FComplex neg() const;

    Fixed     abs() const;
    FComplex  conj() const;

    static FComplex new(Fixed re, Fixed im);

    // Quantizes a double-precision Complex to Q8.24 (to_complex converts
    // back). Twiddle factors and other coefficients computed once
    // (at filter/FFT-plan design time) in double precision are quantized to
    // Fixed via these — the same design-time-float / runtime-fixed split
    // std/dsp/biquad.h's cookbook designers use (FBiquad::from), and
    // completely standard practice (even hardware FFT accelerators
    // precompute twiddle tables offline in floating point).
    static FComplex from(Complex c);
    Complex         to_complex() const;
};

} // namespace std
