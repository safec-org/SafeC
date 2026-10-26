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
// KarplusStrong::new/pluck, then call next() repeatedly — the string is
// self-sustaining except for the built-in decay.
//
// The comb filters are SampleFilters and the Karplus-Strong strings are
// SampleGenerators (see sample_filter.h): a string takes no input, so its
// methods are next() and pluck(), not process() and reset().
#pragma once
#include <std/dsp/fixed.h>
#include <std/dsp/sample_filter.h>

namespace std {

struct CombFF {
    &double history; // ring buffer, length 'delay'
    unsigned long delay;
    unsigned long pos;
    double gain;

    double process(double x);
    void   reset();
    void   process_block([]const double in, []double out);   // out may alias in

    // 'history' is the caller's ring buffer of 'delay' samples, zeroed.
    static struct CombFF new([]double history, double gain);   // delay = history.len
};

struct CombFB {
    &double history;
    unsigned long delay;
    unsigned long pos;
    double gain;

    double process(double x);
    void   reset();
    void   process_block([]const double in, []double out);   // out may alias in

    static struct CombFB new([]double history, double gain);   // delay = history.len
};

struct KarplusStrong {
    &double buffer; // ring buffer, length 'length' — doubles as the delay line and the pitch period
    unsigned long length;
    unsigned long pos;
    double damping; // 0..1; closer to 1 = slower decay (typical: 0.99-0.999)

    // The next output sample.
    double next();
    // Refill the delay line with a new excitation (n samples copied in;
    // if n < length the remainder is left at its previous contents —
    // callers doing a fresh pluck should size the excitation to 'length').
    void   pluck([]const double excitation);

    // 'buffer' is the caller's delay line of 'length' samples; seed it
    // (or call pluck) with an excitation before next().
    static struct KarplusStrong new([]double buffer, double damping);   // length = buffer.len
};

// Single precision: the three filters above on float buffers.
struct CombFF32 {
    &float history;
    unsigned long delay;
    unsigned long pos;
    float gain;

    float process(float x);
    void  reset();
    void  process_block([]const float in, []float out);   // out may alias in

    static struct CombFF32 new([]float history, float gain);   // delay = history.len
};

struct CombFB32 {
    &float history;
    unsigned long delay;
    unsigned long pos;
    float gain;

    float process(float x);
    void  reset();
    void  process_block([]const float in, []float out);   // out may alias in

    static struct CombFB32 new([]float history, float gain);   // delay = history.len
};

struct KarplusStrong32 {
    &float buffer;
    unsigned long length;
    unsigned long pos;
    float damping;

    float next();
    void  pluck([]const float excitation);

    static struct KarplusStrong32 new([]float buffer, float damping);   // length = buffer.len
};

struct FCombFF {
    &Fixed history;
    unsigned long delay;
    unsigned long pos;
    Fixed gain;

    Fixed process(Fixed x);
    void  reset();
    void  process_block([]const Fixed in, []Fixed out);   // out may alias in

    static struct FCombFF new([]Fixed history, Fixed gain);   // delay = history.len
};

struct FCombFB {
    &Fixed history;
    unsigned long delay;
    unsigned long pos;
    Fixed gain;

    Fixed process(Fixed x);
    void  reset();
    void  process_block([]const Fixed in, []Fixed out);   // out may alias in

    static struct FCombFB new([]Fixed history, Fixed gain);   // delay = history.len
};

struct FKarplusStrong {
    &Fixed buffer;
    unsigned long length;
    unsigned long pos;
    Fixed damping;

    Fixed next();
    void  pluck([]const Fixed excitation);

    static struct FKarplusStrong new([]Fixed buffer, Fixed damping);   // length = buffer.len
};

} // namespace std
