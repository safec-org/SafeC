#pragma once
// SafeC Standard Library — one streaming-filter interface per sample type.
//
// std/dsp's streaming filters share one shape: process() one sample,
// reset() the state to silence, process_block() a buffer (out may alias
// in). The traits below name that shape, one per sample type, and the
// generic helpers are written once against them. Traits are structural: a
// type conforms by having the methods, so your own filter works with the
// helpers (and with any generic<F: SampleFilter> code) as soon as it
// defines them with these signatures. Each call is resolved at compile
// time (no dispatch at run time).
//
//   SampleFilter       double: FirFilter, IirFilter, Biquad, CombFF, CombFB
//   SampleFilter32     float:  FirFilter32, IirFilter32, Biquad32,
//                              CombFF32, CombFB32
//   SampleFilterFixed  Q8.24:  FFirFilter, FIirFilter, FBiquad,
//                              FCombFF, FCombFB
//
// KarplusStrong (and its float/fixed versions) is a generator, not a
// filter: process() takes no input and reset() takes an excitation, so it
// doesn't conform.
//
//   struct Biquad hp = Biquad::highpass(48000.0, 80.0, 0.7071);
//   struct Biquad lp = Biquad::lowpass(48000.0, 8000.0, 0.7071);
//   filter_cascade_block(&hp, &lp, in, out, n);      // band-limit a block
//
//   double h[64];
//   filter_impulse_response(&fir, &h[0], 64UL);      // h[i] = taps, then 0
//
// The per-type process_block bodies with a fast path of their own (FIR's
// sliding dot products, the biquad's look-ahead recurrence) stay hand-
// written; the others are filter_process_each over process().
#include <std/dsp/fixed.h>

namespace std {

trait SampleFilter {
    double process(double x);
    void   reset();
    // out[i] = process(in[i]) for i < n; out may alias in.
    void   process_block(const &double in, &double out, unsigned long n);
}

trait SampleFilter32 {
    float process(float x);
    void  reset();
    void  process_block(const &float in, &float out, unsigned long n);
}

trait SampleFilterFixed {
    Fixed process(Fixed x);
    void  reset();
    void  process_block(const &Fixed in, &Fixed out, unsigned long n);
}

// ── Per-sample block loop ─────────────────────────────────────────────────────
// out[i] = f.process(in[i]), one sample at a time (in == out is fine): the
// process_block of a filter with no faster block form.

generic<F: SampleFilter>
void filter_process_each(&F f, const &double in, &double out, unsigned long n) {
    unsigned long i = 0UL;
    while (i < n) {
        double xi;
        unsafe { xi = in[i]; }
        double yi = f.process(xi);
        unsafe { out[i] = yi; }
        i = i + 1UL;
    }
}

generic<F: SampleFilter32>
void filter_process_each_f32(&F f, const &float in, &float out, unsigned long n) {
    unsigned long i = 0UL;
    while (i < n) {
        float xi;
        unsafe { xi = in[i]; }
        float yi = f.process(xi);
        unsafe { out[i] = yi; }
        i = i + 1UL;
    }
}

generic<F: SampleFilterFixed>
void filter_process_each_fixed(&F f, const &Fixed in, &Fixed out, unsigned long n) {
    unsigned long i = 0UL;
    while (i < n) {
        Fixed xi;
        unsafe { xi = in[i]; }
        Fixed yi = f.process(xi);
        unsafe { out[i] = yi; }
        i = i + 1UL;
    }
}

// ── Two-stage cascade ─────────────────────────────────────────────────────────
// 'in' through a, then b: out = b(a(in)), each stage over the whole block
// with its own process_block (so each keeps its fast path). out is the
// intermediate buffer, so no scratch is needed; out may alias in. Chain
// calls (or nest a cascade in your own type) for more stages.

generic<A: SampleFilter, B: SampleFilter>
void filter_cascade_block(&A a, &B b, const &double in, &double out, unsigned long n) {
    a.process_block(in, out, n);
    b.process_block(out, out, n);
}

generic<A: SampleFilter32, B: SampleFilter32>
void filter_cascade_block_f32(&A a, &B b, const &float in, &float out, unsigned long n) {
    a.process_block(in, out, n);
    b.process_block(out, out, n);
}

generic<A: SampleFilterFixed, B: SampleFilterFixed>
void filter_cascade_block_fixed(&A a, &B b, const &Fixed in, &Fixed out, unsigned long n) {
    a.process_block(in, out, n);
    b.process_block(out, out, n);
}

// ── Impulse and step response ─────────────────────────────────────────────────
// The first n samples of f's response to a unit impulse (1, 0, 0, ...) or
// a unit step (1, 1, 1, ...), from a reset state. f is left reset, ready
// for real input. For a FIR the impulse response is its taps.

generic<F: SampleFilter>
void filter_impulse_response(&F f, &double out, unsigned long n) {
    f.reset();
    unsigned long i = 0UL;
    while (i < n) {
        double yi = f.process(i == 0UL ? 1.0 : 0.0);
        unsafe { out[i] = yi; }
        i = i + 1UL;
    }
    f.reset();
}

generic<F: SampleFilter>
void filter_step_response(&F f, &double out, unsigned long n) {
    f.reset();
    unsigned long i = 0UL;
    while (i < n) {
        double yi = f.process(1.0);
        unsafe { out[i] = yi; }
        i = i + 1UL;
    }
    f.reset();
}

generic<F: SampleFilter32>
void filter_impulse_response_f32(&F f, &float out, unsigned long n) {
    f.reset();
    unsigned long i = 0UL;
    while (i < n) {
        float yi = f.process(i == 0UL ? (float)1.0 : (float)0.0);
        unsafe { out[i] = yi; }
        i = i + 1UL;
    }
    f.reset();
}

generic<F: SampleFilter32>
void filter_step_response_f32(&F f, &float out, unsigned long n) {
    f.reset();
    unsigned long i = 0UL;
    while (i < n) {
        float yi = f.process((float)1.0);
        unsafe { out[i] = yi; }
        i = i + 1UL;
    }
    f.reset();
}

generic<F: SampleFilterFixed>
void filter_impulse_response_fixed(&F f, &Fixed out, unsigned long n) {
    f.reset();
    unsigned long i = 0UL;
    while (i < n) {
        Fixed yi = f.process(i == 0UL ? (Fixed)FIXED_ONE : (Fixed)0);
        unsafe { out[i] = yi; }
        i = i + 1UL;
    }
    f.reset();
}

generic<F: SampleFilterFixed>
void filter_step_response_fixed(&F f, &Fixed out, unsigned long n) {
    f.reset();
    unsigned long i = 0UL;
    while (i < n) {
        Fixed yi = f.process((Fixed)FIXED_ONE);
        unsafe { out[i] = yi; }
        i = i + 1UL;
    }
    f.reset();
}

} // namespace std
