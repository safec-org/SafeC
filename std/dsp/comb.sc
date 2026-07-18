// SafeC Standard Library — Comb filter / Karplus-Strong implementation
// (see comb.h)
#pragma once
#include <std/dsp/comb.h>
#include <std/dsp/fixed.h>

namespace std {

inline struct CombFF comb_ff_init(double* history, unsigned long delay, double gain) {
    struct CombFF c;
    c.history = history;
    c.delay = delay;
    c.pos = 0UL;
    c.gain = gain;
    return c;
}

inline double CombFF::process(double x) {
    double tapped;
    unsafe { tapped = self.history[self.pos]; }
    double out = x + self.gain * tapped;
    unsafe { self.history[self.pos] = x; }
    self.pos = (self.pos + 1UL) % self.delay;
    return out;
}

void CombFF::reset() {
    unsigned long i = 0UL;
    while (i < self.delay) {
        unsafe { self.history[i] = 0.0; }
        i = i + 1UL;
    }
    self.pos = 0UL;
}

inline struct CombFB comb_fb_init(double* history, unsigned long delay, double gain) {
    struct CombFB c;
    c.history = history;
    c.delay = delay;
    c.pos = 0UL;
    c.gain = gain;
    return c;
}

inline double CombFB::process(double x) {
    double tapped;
    unsafe { tapped = self.history[self.pos]; }
    double y = x + self.gain * tapped;
    unsafe { self.history[self.pos] = y; }
    self.pos = (self.pos + 1UL) % self.delay;
    return y;
}

void CombFB::reset() {
    unsigned long i = 0UL;
    while (i < self.delay) {
        unsafe { self.history[i] = 0.0; }
        i = i + 1UL;
    }
    self.pos = 0UL;
}

inline struct KarplusStrong ks_init(double* buffer, unsigned long length, double damping) {
    struct KarplusStrong ks;
    ks.buffer = buffer;
    ks.length = length;
    ks.pos = 0UL;
    ks.damping = damping;
    return ks;
}

inline double KarplusStrong::process() {
    unsigned long readPos = self.pos;
    unsigned long nextPos = (self.pos + 1UL) % self.length;
    double v0;
    double v1;
    unsafe { v0 = self.buffer[readPos]; v1 = self.buffer[nextPos]; }
    double newVal = self.damping * 0.5 * (v0 + v1);
    unsafe { self.buffer[readPos] = newVal; }
    self.pos = nextPos;
    return v0;
}

void KarplusStrong::reset(const double* excitation, unsigned long n) {
    unsigned long copyLen = n;
    if (copyLen > self.length) { copyLen = self.length; }
    unsigned long i = 0UL;
    while (i < copyLen) {
        double v;
        unsafe { v = excitation[i]; }
        unsafe { self.buffer[i] = v; }
        i = i + 1UL;
    }
    self.pos = 0UL;
}

inline struct FCombFF comb_ff_init_fixed(Fixed* history, unsigned long delay, Fixed gain) {
    struct FCombFF c;
    c.history = history;
    c.delay = delay;
    c.pos = 0UL;
    c.gain = gain;
    return c;
}

inline Fixed FCombFF::process(Fixed x) {
    Fixed tapped;
    unsafe { tapped = self.history[self.pos]; }
    Fixed out = fixed_add(x, fixed_mul(self.gain, tapped));
    unsafe { self.history[self.pos] = x; }
    self.pos = (self.pos + 1UL) % self.delay;
    return out;
}

void FCombFF::reset() {
    unsigned long i = 0UL;
    while (i < self.delay) {
        unsafe { self.history[i] = (Fixed)0; }
        i = i + 1UL;
    }
    self.pos = 0UL;
}

inline struct FCombFB comb_fb_init_fixed(Fixed* history, unsigned long delay, Fixed gain) {
    struct FCombFB c;
    c.history = history;
    c.delay = delay;
    c.pos = 0UL;
    c.gain = gain;
    return c;
}

inline Fixed FCombFB::process(Fixed x) {
    Fixed tapped;
    unsafe { tapped = self.history[self.pos]; }
    Fixed y = fixed_add(x, fixed_mul(self.gain, tapped));
    unsafe { self.history[self.pos] = y; }
    self.pos = (self.pos + 1UL) % self.delay;
    return y;
}

void FCombFB::reset() {
    unsigned long i = 0UL;
    while (i < self.delay) {
        unsafe { self.history[i] = (Fixed)0; }
        i = i + 1UL;
    }
    self.pos = 0UL;
}

inline struct FKarplusStrong ks_init_fixed(Fixed* buffer, unsigned long length, Fixed damping) {
    struct FKarplusStrong ks;
    ks.buffer = buffer;
    ks.length = length;
    ks.pos = 0UL;
    ks.damping = damping;
    return ks;
}

inline Fixed FKarplusStrong::process() {
    unsigned long readPos = self.pos;
    unsigned long nextPos = (self.pos + 1UL) % self.length;
    Fixed v0;
    Fixed v1;
    unsafe { v0 = self.buffer[readPos]; v1 = self.buffer[nextPos]; }
    Fixed avg = fixed_mul(fixed_add(v0, v1), (Fixed)FIXED_HALF);
    Fixed newVal = fixed_mul(self.damping, avg);
    unsafe { self.buffer[readPos] = newVal; }
    self.pos = nextPos;
    return v0;
}

void FKarplusStrong::reset(const Fixed* excitation, unsigned long n) {
    unsigned long copyLen = n;
    if (copyLen > self.length) { copyLen = self.length; }
    unsigned long i = 0UL;
    while (i < copyLen) {
        Fixed v;
        unsafe { v = excitation[i]; }
        unsafe { self.buffer[i] = v; }
        i = i + 1UL;
    }
    self.pos = 0UL;
}

} // namespace std
