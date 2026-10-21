// SafeC Standard Library — Comb filter / Karplus-Strong implementation
// (see comb.h)
#pragma once
#include <std/dsp/comb.h>
#include <std/dsp/fixed.h>

namespace std {

inline struct CombFF CombFF::new(&double history, unsigned long delay, double gain) {
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

void CombFF::process_block(const &double in, &double out, unsigned long n) {
    filter_process_each(self, in, out, n);
}

inline struct CombFB CombFB::new(&double history, unsigned long delay, double gain) {
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

void CombFB::process_block(const &double in, &double out, unsigned long n) {
    filter_process_each(self, in, out, n);
}

inline struct KarplusStrong KarplusStrong::new(&double buffer, unsigned long length, double damping) {
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

void KarplusStrong::reset(const &double excitation, unsigned long n) {
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

// ── Single precision ─────────────────────────────────────────────────────────

inline struct CombFF32 CombFF32::new(&float history, unsigned long delay, float gain) {
    struct CombFF32 c;
    c.history = history;
    c.delay = delay;
    c.pos = 0UL;
    c.gain = gain;
    return c;
}

inline float CombFF32::process(float x) {
    float tapped;
    unsafe { tapped = self.history[self.pos]; }
    float out = x + self.gain * tapped;
    unsafe { self.history[self.pos] = x; }
    self.pos = (self.pos + 1UL) % self.delay;
    return out;
}

void CombFF32::reset() {
    unsigned long i = 0UL;
    while (i < self.delay) {
        unsafe { self.history[i] = 0.0; }
        i = i + 1UL;
    }
    self.pos = 0UL;
}

void CombFF32::process_block(const &float in, &float out, unsigned long n) {
    filter_process_each_f32(self, in, out, n);
}

inline struct CombFB32 CombFB32::new(&float history, unsigned long delay, float gain) {
    struct CombFB32 c;
    c.history = history;
    c.delay = delay;
    c.pos = 0UL;
    c.gain = gain;
    return c;
}

inline float CombFB32::process(float x) {
    float tapped;
    unsafe { tapped = self.history[self.pos]; }
    float y = x + self.gain * tapped;
    unsafe { self.history[self.pos] = y; }
    self.pos = (self.pos + 1UL) % self.delay;
    return y;
}

void CombFB32::reset() {
    unsigned long i = 0UL;
    while (i < self.delay) {
        unsafe { self.history[i] = 0.0; }
        i = i + 1UL;
    }
    self.pos = 0UL;
}

void CombFB32::process_block(const &float in, &float out, unsigned long n) {
    filter_process_each_f32(self, in, out, n);
}

inline struct KarplusStrong32 KarplusStrong32::new(&float buffer, unsigned long length, float damping) {
    struct KarplusStrong32 ks;
    ks.buffer = buffer;
    ks.length = length;
    ks.pos = 0UL;
    ks.damping = damping;
    return ks;
}

inline float KarplusStrong32::process() {
    unsigned long readPos = self.pos;
    unsigned long nextPos = (self.pos + 1UL) % self.length;
    float v0;
    float v1;
    unsafe { v0 = self.buffer[readPos]; v1 = self.buffer[nextPos]; }
    float newVal = self.damping * (float)0.5 * (v0 + v1);
    unsafe { self.buffer[readPos] = newVal; }
    self.pos = nextPos;
    return v0;
}

void KarplusStrong32::reset(const &float excitation, unsigned long n) {
    unsigned long copyLen = n;
    if (copyLen > self.length) { copyLen = self.length; }
    unsigned long i = 0UL;
    while (i < copyLen) {
        float v;
        unsafe { v = excitation[i]; }
        unsafe { self.buffer[i] = v; }
        i = i + 1UL;
    }
    self.pos = 0UL;
}

inline struct FCombFF FCombFF::new(&Fixed history, unsigned long delay, Fixed gain) {
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

void FCombFF::process_block(const &Fixed in, &Fixed out, unsigned long n) {
    filter_process_each_fixed(self, in, out, n);
}

inline struct FCombFB FCombFB::new(&Fixed history, unsigned long delay, Fixed gain) {
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

void FCombFB::process_block(const &Fixed in, &Fixed out, unsigned long n) {
    filter_process_each_fixed(self, in, out, n);
}

inline struct FKarplusStrong FKarplusStrong::new(&Fixed buffer, unsigned long length, Fixed damping) {
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

void FKarplusStrong::reset(const &Fixed excitation, unsigned long n) {
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
