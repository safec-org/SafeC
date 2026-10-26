// SafeC Standard Library — Comb filter / Karplus-Strong implementation
// (see comb.h)
#pragma once
#include <std/dsp/comb.h>
#include <std/dsp/fixed.h>

namespace std {

inline struct CombFF CombFF::new([]double history, double gain) {
    struct CombFF c;
    unsafe { c.history = (&double)history.ptr; }
    c.delay = history.len;
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

void CombFF::process_block([]const double in_s, []double out_s) {
    unsigned long n = in_s.len < out_s.len ? in_s.len : out_s.len;
    if (n == 0UL) return;
    const &double in;
    &double out;
    unsafe { in = (const &double)in_s.ptr; out = (&double)out_s.ptr; }
    filter_process_each_n_(self, in, out, n);
}

inline struct CombFB CombFB::new([]double history, double gain) {
    struct CombFB c;
    unsafe { c.history = (&double)history.ptr; }
    c.delay = history.len;
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

void CombFB::process_block([]const double in_s, []double out_s) {
    unsigned long n = in_s.len < out_s.len ? in_s.len : out_s.len;
    if (n == 0UL) return;
    const &double in;
    &double out;
    unsafe { in = (const &double)in_s.ptr; out = (&double)out_s.ptr; }
    filter_process_each_n_(self, in, out, n);
}

inline struct KarplusStrong KarplusStrong::new([]double buffer, double damping) {
    struct KarplusStrong ks;
    unsafe { ks.buffer = (&double)buffer.ptr; }
    ks.length = buffer.len;
    ks.pos = 0UL;
    ks.damping = damping;
    return ks;
}

inline double KarplusStrong::next() {
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

void KarplusStrong::pluck([]const double excitation) {
    unsigned long copyLen = excitation.len;
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

inline struct CombFF32 CombFF32::new([]float history, float gain) {
    struct CombFF32 c;
    unsafe { c.history = (&float)history.ptr; }
    c.delay = history.len;
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

void CombFF32::process_block([]const float in_s, []float out_s) {
    unsigned long n = in_s.len < out_s.len ? in_s.len : out_s.len;
    if (n == 0UL) return;
    const &float in;
    &float out;
    unsafe { in = (const &float)in_s.ptr; out = (&float)out_s.ptr; }
    filter_process_each_n_(self, in, out, n);
}

inline struct CombFB32 CombFB32::new([]float history, float gain) {
    struct CombFB32 c;
    unsafe { c.history = (&float)history.ptr; }
    c.delay = history.len;
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

void CombFB32::process_block([]const float in_s, []float out_s) {
    unsigned long n = in_s.len < out_s.len ? in_s.len : out_s.len;
    if (n == 0UL) return;
    const &float in;
    &float out;
    unsafe { in = (const &float)in_s.ptr; out = (&float)out_s.ptr; }
    filter_process_each_n_(self, in, out, n);
}

inline struct KarplusStrong32 KarplusStrong32::new([]float buffer, float damping) {
    struct KarplusStrong32 ks;
    unsafe { ks.buffer = (&float)buffer.ptr; }
    ks.length = buffer.len;
    ks.pos = 0UL;
    ks.damping = damping;
    return ks;
}

inline float KarplusStrong32::next() {
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

void KarplusStrong32::pluck([]const float excitation) {
    unsigned long copyLen = excitation.len;
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

inline struct FCombFF FCombFF::new([]Fixed history, Fixed gain) {
    struct FCombFF c;
    unsafe { c.history = (&Fixed)history.ptr; }
    c.delay = history.len;
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

void FCombFF::process_block([]const Fixed in_s, []Fixed out_s) {
    unsigned long n = in_s.len < out_s.len ? in_s.len : out_s.len;
    if (n == 0UL) return;
    const &Fixed in;
    &Fixed out;
    unsafe { in = (const &Fixed)in_s.ptr; out = (&Fixed)out_s.ptr; }
    filter_process_each_n_(self, in, out, n);
}

inline struct FCombFB FCombFB::new([]Fixed history, Fixed gain) {
    struct FCombFB c;
    unsafe { c.history = (&Fixed)history.ptr; }
    c.delay = history.len;
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

void FCombFB::process_block([]const Fixed in_s, []Fixed out_s) {
    unsigned long n = in_s.len < out_s.len ? in_s.len : out_s.len;
    if (n == 0UL) return;
    const &Fixed in;
    &Fixed out;
    unsafe { in = (const &Fixed)in_s.ptr; out = (&Fixed)out_s.ptr; }
    filter_process_each_n_(self, in, out, n);
}

inline struct FKarplusStrong FKarplusStrong::new([]Fixed buffer, Fixed damping) {
    struct FKarplusStrong ks;
    unsafe { ks.buffer = (&Fixed)buffer.ptr; }
    ks.length = buffer.len;
    ks.pos = 0UL;
    ks.damping = damping;
    return ks;
}

inline Fixed FKarplusStrong::next() {
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

void FKarplusStrong::pluck([]const Fixed excitation) {
    unsigned long copyLen = excitation.len;
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
