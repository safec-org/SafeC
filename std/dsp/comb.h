// SafeC Standard Library — Comb filters + Karplus-Strong string synthesis
//
// CombFF (feedforward comb): y[n] = x[n] + gain*x[n-delay] — an FIR delay-
// and-add; produces evenly-spaced notches in the frequency response.
// CombFB (feedback comb): y[n] = x[n] + gain*y[n-delay] — an IIR delay-
// and-add; produces evenly-spaced resonant peaks instead (the basis of
// most artificial reverb algorithms — a bank of feedback combs at
// different delays/gains, often followed by allpass filters).
//
// KarplusStrong is the classic plucked-string/percussion synthesis
// algorithm (Karplus & Strong, 1983): a feedback comb whose feedback path
// is itself a 2-tap averaging lowpass (delayLine[n] and delayLine[n-1]
// averaged, then scaled by 'damping') instead of a plain scalar gain —
// the averaging progressively removes high-frequency energy each time
// around the loop, which is what makes a plucked string's timbre brighten-
// then-decay the way it audibly does. Seed the delay line with an
// excitation buffer (e.g. white noise, or a single impulse) via
// ks_init/ks_reset, then call process() repeatedly with no input — the
// string is self-sustaining except for the built-in decay.
#pragma once
#include <std/dsp/fixed.h>

namespace std {

struct CombFF {
    double* history; // ring buffer, length 'delay'
    unsigned long delay;
    unsigned long pos;
    double gain;

    double process(double x);
    void   reset();
};

struct CombFB {
    double* history;
    unsigned long delay;
    unsigned long pos;
    double gain;

    double process(double x);
    void   reset();
};

struct KarplusStrong {
    double* buffer; // ring buffer, length 'length' — doubles as the delay line and the pitch period
    unsigned long length;
    unsigned long pos;
    double damping; // 0..1; closer to 1 = slower decay (typical: 0.99-0.999)

    double process();
    // Refill the delay line with a new excitation (n samples copied in;
    // if n < length the remainder is left at its previous contents —
    // callers doing a fresh pluck should size the excitation to 'length').
    void   reset(const double* excitation, unsigned long n);
};

struct CombFF comb_ff_init(double* history, unsigned long delay, double gain);
struct CombFB comb_fb_init(double* history, unsigned long delay, double gain);
struct KarplusStrong ks_init(double* buffer, unsigned long length, double damping);

struct FCombFF {
    Fixed* history;
    unsigned long delay;
    unsigned long pos;
    Fixed gain;

    Fixed process(Fixed x);
    void  reset();
};

struct FCombFB {
    Fixed* history;
    unsigned long delay;
    unsigned long pos;
    Fixed gain;

    Fixed process(Fixed x);
    void  reset();
};

struct FKarplusStrong {
    Fixed* buffer;
    unsigned long length;
    unsigned long pos;
    Fixed damping;

    Fixed process();
    void  reset(const Fixed* excitation, unsigned long n);
};

struct FCombFF comb_ff_init_fixed(Fixed* history, unsigned long delay, Fixed gain);
struct FCombFB comb_fb_init_fixed(Fixed* history, unsigned long delay, Fixed gain);
struct FKarplusStrong ks_init_fixed(Fixed* buffer, unsigned long length, Fixed damping);

} // namespace std
