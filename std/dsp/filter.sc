// SafeC Standard Library — General FIR/IIR filter implementation (see filter.h)
#pragma once
#include <std/dsp/dsp.h>
#include <std/dsp/filter.h>
#include <std/dsp/fixed.h>

namespace std {

// ── FIR ──────────────────────────────────────────────────────────────────────

inline struct FirFilter fir_init(const &double coeffs, unsigned long numTaps, &double history) {
    struct FirFilter f;
    f.coeffs = coeffs;
    f.numTaps = numTaps;
    f.history = history;
    f.pos = 0UL;
    return f;
}

double FirFilter::process(double x) {
    unsafe { self.history[self.pos] = x; }
    // y = sum_k coeffs[k] * history[(pos - k) mod numTaps], as two runs
    // without the modulo: k = 0..pos walks history down from pos, then
    // k = pos+1..numTaps-1 walks it down from the end. Four partial sums
    // keep the multiply-adds independent.
    double s0 = 0.0; double s1 = 0.0; double s2 = 0.0; double s3 = 0.0;
    unsafe {
        const double* c = (const double*)self.coeffs;
        const double* h = (const double*)self.history;
        unsigned long k = 0UL;
        unsigned long last = self.pos;              // first run: k <= pos
        while (k + 4UL <= last + 1UL) {
            s0 = __builtin_fmuladd(c[k], h[self.pos - k], s0);
            s1 = __builtin_fmuladd(c[k + 1UL], h[self.pos - k - 1UL], s1);
            s2 = __builtin_fmuladd(c[k + 2UL], h[self.pos - k - 2UL], s2);
            s3 = __builtin_fmuladd(c[k + 3UL], h[self.pos - k - 3UL], s3);
            k = k + 4UL;
        }
        while (k <= last) { s0 = __builtin_fmuladd(c[k], h[self.pos - k], s0); k = k + 1UL; }
        unsigned long top = self.pos + self.numTaps;  // second run: index top - k
        while (k + 4UL <= self.numTaps) {
            s0 = __builtin_fmuladd(c[k], h[top - k], s0);
            s1 = __builtin_fmuladd(c[k + 1UL], h[top - k - 1UL], s1);
            s2 = __builtin_fmuladd(c[k + 2UL], h[top - k - 2UL], s2);
            s3 = __builtin_fmuladd(c[k + 3UL], h[top - k - 3UL], s3);
            k = k + 4UL;
        }
        while (k < self.numTaps) { s0 = __builtin_fmuladd(c[k], h[top - k], s0); k = k + 1UL; }
    }
    self.pos = self.pos + 1UL == self.numTaps ? 0UL : self.pos + 1UL;
    return (s0 + s1) + (s2 + s3);
}

void FirFilter::reset() {
    unsigned long i = 0UL;
    while (i < self.numTaps) {
        unsafe { self.history[i] = 0.0; }
        i = i + 1UL;
    }
    self.pos = 0UL;
}

// Blocks go through a small window on the stack -- the last numTaps - 1
// inputs in time order, then up to FIR_CHUNK new ones -- so each output is
// a dot product with the reversed taps, many outputs per vector
// (dsp_sliding_dot_f64)
// instead of a walk around the ring. No allocation, so still real-time
// safe. Filters longer than FIR_STACK_TAPS run sample by sample.
#ifdef __SAFEC_FREESTANDING__
#define FIR_STACK_TAPS 64UL
#define FIR_CHUNK      64UL
#else
#define FIR_STACK_TAPS 256UL
#define FIR_CHUNK      512UL
#endif

void fir_process_block(&FirFilter f, const &double in, &double out, unsigned long n) {
    unsigned long T = f.numTaps;
    if (T == 0UL || T > FIR_STACK_TAPS) {
        unsigned long i = 0UL;
        while (i < n) {
            double xi;
            unsafe { xi = in[i]; }
            double yi = f.process(xi);
            unsafe { out[i] = yi; }
            i = i + 1UL;
        }
        return;
    }
    double crev[FIR_STACK_TAPS];
    double win[FIR_STACK_TAPS + FIR_CHUNK];
    unsafe {
        unsigned long t = 0UL;
        while (t < T) { crev[t] = f.coeffs[T - 1UL - t]; t = t + 1UL; }
        // win[0 .. T-1) = x[n-(T-1)] .. x[n-1], from the ring (the sample at
        // ring index pos - j is x[n-j]).
        unsigned long j = 0UL;
        while (j + 1UL < T) {
            unsigned long back = T - 1UL - j;                 // x[n - back]
            win[j] = f.history[(f.pos + T - back) % T];
            j = j + 1UL;
        }
        unsigned long done = 0UL;
        while (done < n) {
            unsigned long len = n - done < FIR_CHUNK ? n - done : FIR_CHUNK;
            unsigned long q = 0UL;
            while (q < len) { win[T - 1UL + q] = in[done + q]; q = q + 1UL; }
            dsp_sliding_dot_f64((const double*)&crev[0], (const double*)&win[0], &out[done], len, T);
            // keep the newest T - 1 samples at the front
            q = 0UL;
            while (q + 1UL < T) { win[q] = win[len + q]; q = q + 1UL; }
            done = done + len;
        }
        // Back into the ring with pos = 0: x[n-j] at index T - j.
        j = 1UL;
        while (j < T) { f.history[T - j] = win[T - 1UL - j]; j = j + 1UL; }
    }
    f.pos = 0UL;
}

// The float FIR: the same code on floats (dsp_dot_f32 for blocks).

inline struct FirFilter32 fir_init_f32(const &float coeffs, unsigned long numTaps, &float history) {
    struct FirFilter32 f;
    f.coeffs = coeffs;
    f.numTaps = numTaps;
    f.history = history;
    f.pos = 0UL;
    return f;
}

float FirFilter32::process(float x) {
    unsafe { self.history[self.pos] = x; }
    // y = sum_k coeffs[k] * history[(pos - k) mod numTaps], as two runs
    // without the modulo: k = 0..pos walks history down from pos, then
    // k = pos+1..numTaps-1 walks it down from the end. Four partial sums
    // keep the multiply-adds independent.
    float s0 = 0.0; float s1 = 0.0; float s2 = 0.0; float s3 = 0.0;
    unsafe {
        const float* c = (const float*)self.coeffs;
        const float* h = (const float*)self.history;
        unsigned long k = 0UL;
        unsigned long last = self.pos;              // first run: k <= pos
        while (k + 4UL <= last + 1UL) {
            s0 = __builtin_fmuladd(c[k], h[self.pos - k], s0);
            s1 = __builtin_fmuladd(c[k + 1UL], h[self.pos - k - 1UL], s1);
            s2 = __builtin_fmuladd(c[k + 2UL], h[self.pos - k - 2UL], s2);
            s3 = __builtin_fmuladd(c[k + 3UL], h[self.pos - k - 3UL], s3);
            k = k + 4UL;
        }
        while (k <= last) { s0 = __builtin_fmuladd(c[k], h[self.pos - k], s0); k = k + 1UL; }
        unsigned long top = self.pos + self.numTaps;  // second run: index top - k
        while (k + 4UL <= self.numTaps) {
            s0 = __builtin_fmuladd(c[k], h[top - k], s0);
            s1 = __builtin_fmuladd(c[k + 1UL], h[top - k - 1UL], s1);
            s2 = __builtin_fmuladd(c[k + 2UL], h[top - k - 2UL], s2);
            s3 = __builtin_fmuladd(c[k + 3UL], h[top - k - 3UL], s3);
            k = k + 4UL;
        }
        while (k < self.numTaps) { s0 = __builtin_fmuladd(c[k], h[top - k], s0); k = k + 1UL; }
    }
    self.pos = self.pos + 1UL == self.numTaps ? 0UL : self.pos + 1UL;
    return (s0 + s1) + (s2 + s3);
}

void FirFilter32::reset() {
    unsigned long i = 0UL;
    while (i < self.numTaps) {
        unsafe { self.history[i] = 0.0; }
        i = i + 1UL;
    }
    self.pos = 0UL;
}

void fir_process_block_f32(&FirFilter32 f, const &float in, &float out, unsigned long n) {
    unsigned long T = f.numTaps;
    if (T == 0UL || T > FIR_STACK_TAPS) {
        unsigned long i = 0UL;
        while (i < n) {
            float xi;
            unsafe { xi = in[i]; }
            float yi = f.process(xi);
            unsafe { out[i] = yi; }
            i = i + 1UL;
        }
        return;
    }
    float crev[FIR_STACK_TAPS];
    float win[FIR_STACK_TAPS + FIR_CHUNK];
    unsafe {
        unsigned long t = 0UL;
        while (t < T) { crev[t] = f.coeffs[T - 1UL - t]; t = t + 1UL; }
        // win[0 .. T-1) = x[n-(T-1)] .. x[n-1], from the ring (the sample at
        // ring index pos - j is x[n-j]).
        unsigned long j = 0UL;
        while (j + 1UL < T) {
            unsigned long back = T - 1UL - j;                 // x[n - back]
            win[j] = f.history[(f.pos + T - back) % T];
            j = j + 1UL;
        }
        unsigned long done = 0UL;
        while (done < n) {
            unsigned long len = n - done < FIR_CHUNK ? n - done : FIR_CHUNK;
            unsigned long q = 0UL;
            while (q < len) { win[T - 1UL + q] = in[done + q]; q = q + 1UL; }
            dsp_sliding_dot_f32((const float*)&crev[0], (const float*)&win[0], &out[done], len, T);
            // keep the newest T - 1 samples at the front
            q = 0UL;
            while (q + 1UL < T) { win[q] = win[len + q]; q = q + 1UL; }
            done = done + len;
        }
        // Back into the ring with pos = 0: x[n-j] at index T - j.
        j = 1UL;
        while (j < T) { f.history[T - j] = win[T - 1UL - j]; j = j + 1UL; }
    }
    f.pos = 0UL;
}

inline struct FFirFilter ffir_init(const &Fixed coeffs, unsigned long numTaps, &Fixed history) {
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

void ffir_process_block(&FFirFilter f, const &Fixed in, &Fixed out, unsigned long n) {
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

inline struct IirFilter iir_init(const &double b, unsigned long numB,
                                  const &double a, unsigned long numA,
                                  &double xHistory, &double yHistory) {
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

void iir_process_block(&IirFilter f, const &double in, &double out, unsigned long n) {
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

// The float IIR (high orders lose precision in float: prefer Biquad32
// sections for those).

inline struct IirFilter32 iir_init_f32(const &float b, unsigned long numB,
                                  const &float a, unsigned long numA,
                                  &float xHistory, &float yHistory) {
    struct IirFilter32 f;
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

float IirFilter32::process(float x) {
    unsafe { self.xHistory[self.xPos] = x; }
    float ffwd = (float)0.0;
    unsigned long k = 0UL;
    while (k < self.numB) {
        unsigned long idx = (self.xPos + self.numB - k) % self.numB;
        unsafe { ffwd = ffwd + self.b[k] * self.xHistory[idx]; }
        k = k + 1UL;
    }
    float fback = (float)0.0;
    k = 1UL;
    while (k < self.numA) {
        unsigned long idx = (self.yPos + self.numA - k) % self.numA;
        unsafe { fback = fback + self.a[k] * self.yHistory[idx]; }
        k = k + 1UL;
    }
    float y = ffwd - fback;
    unsafe { self.yHistory[self.yPos] = y; }
    self.xPos = (self.xPos + 1UL) % self.numB;
    self.yPos = (self.yPos + 1UL) % self.numA;
    return y;
}

void IirFilter32::reset() {
    unsigned long i = 0UL;
    while (i < self.numB) { unsafe { self.xHistory[i] = (float)0.0; } i = i + 1UL; }
    i = 0UL;
    while (i < self.numA) { unsafe { self.yHistory[i] = (float)0.0; } i = i + 1UL; }
    self.xPos = 0UL;
    self.yPos = 0UL;
}

void iir_process_block_f32(&IirFilter32 f, const &float in, &float out, unsigned long n) {
    unsigned long i = 0UL;
    while (i < n) {
        float xi;
        unsafe { xi = in[i]; }
        float yi;
        unsafe { yi = f->process(xi); }
        unsafe { out[i] = yi; }
        i = i + 1UL;
    }
}

inline struct FIirFilter fiir_init(const &Fixed b, unsigned long numB,
                                    const &Fixed a, unsigned long numA,
                                    &Fixed xHistory, &Fixed yHistory) {
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

void fiir_process_block(&FIirFilter f, const &Fixed in, &Fixed out, unsigned long n) {
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
