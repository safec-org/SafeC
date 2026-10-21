// SafeC Standard Library — DSP Complex Numbers implementation (see complex_dsp.h)
#pragma once
#include <std/dsp/complex_dsp.h>
#include <std/dsp/fixed.h>
#include <std/math.h>

namespace std {

inline Complex Complex::new(double re, double im) {
    Complex c;
    c.re = re;
    c.im = im;
    return c;
}

inline Complex Complex::operator+(Complex other) const {
    return Complex::new(self.re + other.re, self.im + other.im);
}

inline Complex Complex::operator-(Complex other) const {
    return Complex::new(self.re - other.re, self.im - other.im);
}

inline Complex Complex::operator*(Complex other) const {
    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    return Complex::new(self.re * other.re - self.im * other.im,
                        self.re * other.im + self.im * other.re);
}

inline Complex Complex::operator/(Complex other) const {
    // (a+bi)/(c+di) = (a+bi)(c-di) / (c^2+d^2)
    double denom = other.re * other.re + other.im * other.im;
    double nre = self.re * other.re + self.im * other.im;
    double nim = self.im * other.re - self.re * other.im;
    return Complex::new(nre / denom, nim / denom);
}

inline Complex Complex::neg() const {
    return Complex::new(-self.re, -self.im);
}

inline double Complex::abs() const {
    return sqrt_d(self.re * self.re + self.im * self.im);
}

inline double Complex::arg() const {
    return atan2_d(self.im, self.re);
}

inline Complex Complex::conj() const {
    return Complex::new(self.re, -self.im);
}

inline Complex Complex::polar(double magnitude, double theta) {
    return Complex::new(magnitude * cos_d(theta), magnitude * sin_d(theta));
}

inline Complex32 Complex32::new(float re, float im) {
    Complex32 c;
    c.re = re;
    c.im = im;
    return c;
}

inline Complex32 Complex32::operator+(Complex32 other) const {
    return Complex32::new(self.re + other.re, self.im + other.im);
}

inline Complex32 Complex32::operator-(Complex32 other) const {
    return Complex32::new(self.re - other.re, self.im - other.im);
}

inline Complex32 Complex32::operator*(Complex32 other) const {
    return Complex32::new(self.re * other.re - self.im * other.im,
                         self.re * other.im + self.im * other.re);
}

inline Complex32 Complex32::operator/(Complex32 other) const {
    float denom = other.re * other.re + other.im * other.im;
    float nre = self.re * other.re + self.im * other.im;
    float nim = self.im * other.re - self.re * other.im;
    return Complex32::new(nre / denom, nim / denom);
}

inline Complex32 Complex32::neg() const {
    return Complex32::new(-self.re, -self.im);
}

inline float Complex32::abs() const {
    return sqrt_f(self.re * self.re + self.im * self.im);
}

inline float Complex32::arg() const {
    return atan2_f(self.im, self.re);
}

inline Complex32 Complex32::conj() const {
    return Complex32::new(self.re, -self.im);
}

inline Complex32 Complex32::polar(float magnitude, float theta) {
    return Complex32::new(magnitude * cos_f(theta), magnitude * sin_f(theta));
}

inline Complex32 Complex32::from(Complex c) {
    return Complex32::new((float)c.re, (float)c.im);
}

inline Complex Complex32::to_complex() const {
    return Complex::new((double)self.re, (double)self.im);
}

inline FComplex FComplex::new(Fixed re, Fixed im) {
    FComplex c;
    c.re = re;
    c.im = im;
    return c;
}

inline FComplex FComplex::operator+(FComplex other) const {
    return FComplex::new(fixed_add(self.re, other.re), fixed_add(self.im, other.im));
}

inline FComplex FComplex::operator-(FComplex other) const {
    return FComplex::new(fixed_sub(self.re, other.re), fixed_sub(self.im, other.im));
}

inline FComplex FComplex::operator*(FComplex other) const {
    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i, each product a Q8.24 fixed_mul.
    Fixed ac = fixed_mul(self.re, other.re);
    Fixed bd = fixed_mul(self.im, other.im);
    Fixed ad = fixed_mul(self.re, other.im);
    Fixed bc = fixed_mul(self.im, other.re);
    return FComplex::new(fixed_sub(ac, bd), fixed_add(ad, bc));
}

inline FComplex FComplex::neg() const {
    return FComplex::new(fixed_neg(self.re), fixed_neg(self.im));
}

inline Fixed FComplex::abs() const {
    Fixed sumSq = fixed_add(fixed_mul(self.re, self.re), fixed_mul(self.im, self.im));
    return fixed_sqrt(sumSq);
}

inline FComplex FComplex::conj() const {
    return FComplex::new(self.re, fixed_neg(self.im));
}

inline FComplex FComplex::from(Complex c) {
    return FComplex::new(fixed_from_float(c.re), fixed_from_float(c.im));
}

inline Complex FComplex::to_complex() const {
    return Complex::new(fixed_to_float(self.re), fixed_to_float(self.im));
}

} // namespace std
