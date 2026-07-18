// SafeC Standard Library — DSP Complex Numbers implementation (see complex_dsp.h)
#pragma once
#include <std/dsp/complex_dsp.h>
#include <std/dsp/fixed.h>
#include <std/math.h>

namespace std {

inline Complex complex_new(double re, double im) {
    Complex c;
    c.re = re;
    c.im = im;
    return c;
}

inline Complex Complex::operator+(Complex other) const {
    return complex_new(self.re + other.re, self.im + other.im);
}

inline Complex Complex::operator-(Complex other) const {
    return complex_new(self.re - other.re, self.im - other.im);
}

inline Complex Complex::operator*(Complex other) const {
    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    return complex_new(self.re * other.re - self.im * other.im,
                        self.re * other.im + self.im * other.re);
}

inline Complex Complex::operator/(Complex other) const {
    // (a+bi)/(c+di) = (a+bi)(c-di) / (c^2+d^2)
    double denom = other.re * other.re + other.im * other.im;
    double nre = self.re * other.re + self.im * other.im;
    double nim = self.im * other.re - self.re * other.im;
    return complex_new(nre / denom, nim / denom);
}

inline Complex Complex::neg() const {
    return complex_new(-self.re, -self.im);
}

inline double Complex::abs() const {
    return sqrt_d(self.re * self.re + self.im * self.im);
}

inline double Complex::arg() const {
    return atan2_d(self.im, self.re);
}

inline Complex Complex::conj() const {
    return complex_new(self.re, -self.im);
}

inline Complex complex_from_polar(double magnitude, double theta) {
    return complex_new(magnitude * cos_d(theta), magnitude * sin_d(theta));
}

inline FComplex fcomplex_new(Fixed re, Fixed im) {
    FComplex c;
    c.re = re;
    c.im = im;
    return c;
}

inline FComplex FComplex::operator+(FComplex other) const {
    return fcomplex_new(fixed_add(self.re, other.re), fixed_add(self.im, other.im));
}

inline FComplex FComplex::operator-(FComplex other) const {
    return fcomplex_new(fixed_sub(self.re, other.re), fixed_sub(self.im, other.im));
}

inline FComplex FComplex::operator*(FComplex other) const {
    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i, each product a Q8.24 fixed_mul.
    Fixed ac = fixed_mul(self.re, other.re);
    Fixed bd = fixed_mul(self.im, other.im);
    Fixed ad = fixed_mul(self.re, other.im);
    Fixed bc = fixed_mul(self.im, other.re);
    return fcomplex_new(fixed_sub(ac, bd), fixed_add(ad, bc));
}

inline FComplex FComplex::neg() const {
    return fcomplex_new(fixed_neg(self.re), fixed_neg(self.im));
}

inline Fixed FComplex::abs() const {
    Fixed sumSq = fixed_add(fixed_mul(self.re, self.re), fixed_mul(self.im, self.im));
    return fixed_sqrt(sumSq);
}

inline FComplex FComplex::conj() const {
    return fcomplex_new(self.re, fixed_neg(self.im));
}

inline FComplex complex_to_fixed(Complex c) {
    return fcomplex_new(fixed_from_float(c.re), fixed_from_float(c.im));
}

inline Complex fcomplex_to_float(FComplex c) {
    return complex_new(fixed_to_float(c.re), fixed_to_float(c.im));
}

} // namespace std
