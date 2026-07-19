// SafeC Standard Library — General FIR/IIR filter implementation (see filter.h)
#pragma once
#include <std/dsp/filter.h>
#include <std/dsp/fixed.h>

namespace std {

// ── FIR ──────────────────────────────────────────────────────────────────────

inline struct FirFilter fir_init(const double* coeffs, unsigned long numTaps, double* history) {
    struct FirFilter f;
    f.coeffs = coeffs;
    f.numTaps = numTaps;
    f.history = history;
    f.pos = 0UL;
    return f;
}

double FirFilter::process(double x) {
    unsafe { self.history[self.pos] = x; }
    double y = 0.0;
    unsigned long k = 0UL;
    while (k < self.numTaps) {
        unsigned long idx = (self.pos + self.numTaps - k) % self.numTaps;
        unsafe { y = y + self.coeffs[k] * self.history[idx]; }
        k = k + 1UL;
    }
    self.pos = (self.pos + 1UL) % self.numTaps;
    return y;
}

void FirFilter::reset() {
    unsigned long i = 0UL;
    while (i < self.numTaps) {
        unsafe { self.history[i] = 0.0; }
        i = i + 1UL;
    }
    self.pos = 0UL;
}

void fir_process_block(&FirFilter f, const double* in, double* out, unsigned long n) {
    unsigned long i = 0UL;
    while (i < n) {
        double xi;
        unsafe { xi = in[i]; }
        double yi;
        unsafe { yi = f->process(xi); }
        unsafe { out[i] = yi; }
        i = i + 1UL;
    }
}

inline struct FFirFilter ffir_init(const Fixed* coeffs, unsigned long numTaps, Fixed* history) {
    struct FFirFilter f;
    f.coeffs = coeffs;
    f.numTaps = numTaps;
    f.history = history;
    f.pos = 0UL;
    return f;
}

Fixed FFirFilter::process(Fixed x) {
    unsafe { self.history[self.pos] = x; }
    Fixed y = (Fixed)0;
    unsigned long k = 0UL;
    while (k < self.numTaps) {
        unsigned long idx = (self.pos + self.numTaps - k) % self.numTaps;
        unsafe { y = fixed_add(y, fixed_mul(self.coeffs[k], self.history[idx])); }
        k = k + 1UL;
    }
    self.pos = (self.pos + 1UL) % self.numTaps;
    return y;
}

void FFirFilter::reset() {
    unsigned long i = 0UL;
    while (i < self.numTaps) {
        unsafe { self.history[i] = (Fixed)0; }
        i = i + 1UL;
    }
    self.pos = 0UL;
}

void ffir_process_block(&FFirFilter f, const Fixed* in, Fixed* out, unsigned long n) {
    unsigned long i = 0UL;
    while (i < n) {
        Fixed xi;
        unsafe { xi = in[i]; }
        Fixed yi;
        unsafe { yi = f->process(xi); }
        unsafe { out[i] = yi; }
        i = i + 1UL;
    }
}

// ── IIR (general order, direct form I) ────────────────────────────────────────

inline struct IirFilter iir_init(const double* b, unsigned long numB,
                                  const double* a, unsigned long numA,
                                  double* xHistory, double* yHistory) {
    struct IirFilter f;
    f.b = b;
    f.numB = numB;
    f.a = a;
    f.numA = numA;
    f.xHistory = xHistory;
    f.yHistory = yHistory;
    f.xPos = 0UL;
    f.yPos = 0UL;
    return f;
}

double IirFilter::process(double x) {
    unsafe { self.xHistory[self.xPos] = x; }
    double ffwd = 0.0;
    unsigned long k = 0UL;
    while (k < self.numB) {
        unsigned long idx = (self.xPos + self.numB - k) % self.numB;
        unsafe { ffwd = ffwd + self.b[k] * self.xHistory[idx]; }
        k = k + 1UL;
    }
    double fback = 0.0;
    k = 1UL;
    while (k < self.numA) {
        unsigned long idx = (self.yPos + self.numA - k) % self.numA;
        unsafe { fback = fback + self.a[k] * self.yHistory[idx]; }
        k = k + 1UL;
    }
    double y = ffwd - fback;
    unsafe { self.yHistory[self.yPos] = y; }
    self.xPos = (self.xPos + 1UL) % self.numB;
    self.yPos = (self.yPos + 1UL) % self.numA;
    return y;
}

void IirFilter::reset() {
    unsigned long i = 0UL;
    while (i < self.numB) { unsafe { self.xHistory[i] = 0.0; } i = i + 1UL; }
    i = 0UL;
    while (i < self.numA) { unsafe { self.yHistory[i] = 0.0; } i = i + 1UL; }
    self.xPos = 0UL;
    self.yPos = 0UL;
}

void iir_process_block(&IirFilter f, const double* in, double* out, unsigned long n) {
    unsigned long i = 0UL;
    while (i < n) {
        double xi;
        unsafe { xi = in[i]; }
        double yi;
        unsafe { yi = f->process(xi); }
        unsafe { out[i] = yi; }
        i = i + 1UL;
    }
}

inline struct FIirFilter fiir_init(const Fixed* b, unsigned long numB,
                                    const Fixed* a, unsigned long numA,
                                    Fixed* xHistory, Fixed* yHistory) {
    struct FIirFilter f;
    f.b = b;
    f.numB = numB;
    f.a = a;
    f.numA = numA;
    f.xHistory = xHistory;
    f.yHistory = yHistory;
    f.xPos = 0UL;
    f.yPos = 0UL;
    return f;
}

Fixed FIirFilter::process(Fixed x) {
    unsafe { self.xHistory[self.xPos] = x; }
    Fixed ffwd = (Fixed)0;
    unsigned long k = 0UL;
    while (k < self.numB) {
        unsigned long idx = (self.xPos + self.numB - k) % self.numB;
        unsafe { ffwd = fixed_add(ffwd, fixed_mul(self.b[k], self.xHistory[idx])); }
        k = k + 1UL;
    }
    Fixed fback = (Fixed)0;
    k = 1UL;
    while (k < self.numA) {
        unsigned long idx = (self.yPos + self.numA - k) % self.numA;
        unsafe { fback = fixed_add(fback, fixed_mul(self.a[k], self.yHistory[idx])); }
        k = k + 1UL;
    }
    Fixed y = fixed_sub(ffwd, fback);
    unsafe { self.yHistory[self.yPos] = y; }
    self.xPos = (self.xPos + 1UL) % self.numB;
    self.yPos = (self.yPos + 1UL) % self.numA;
    return y;
}

void FIirFilter::reset() {
    unsigned long i = 0UL;
    while (i < self.numB) { unsafe { self.xHistory[i] = (Fixed)0; } i = i + 1UL; }
    i = 0UL;
    while (i < self.numA) { unsafe { self.yHistory[i] = (Fixed)0; } i = i + 1UL; }
    self.xPos = 0UL;
    self.yPos = 0UL;
}

void fiir_process_block(&FIirFilter f, const Fixed* in, Fixed* out, unsigned long n) {
    unsigned long i = 0UL;
    while (i < n) {
        Fixed xi;
        unsafe { xi = in[i]; }
        Fixed yi;
        unsafe { yi = f->process(xi); }
        unsafe { out[i] = yi; }
        i = i + 1UL;
    }
}

} // namespace std
