#pragma once
// SafeC Standard Library — streaming DSP as composable traits.
//
// std/dsp's streaming filters share one shape: process() one sample,
// process_block() a buffer (out may alias in), reset() the state to
// silence. The traits below name each capability on its own and compose
// into the roles generic code asks for (trait composition: 'trait
// SampleFilter : SampleProcessor + BlockProcessor + Resettable {}' requires
// all three). The generic helpers after them are written once and ask only
// for what they use; each call is resolved at compile time (no dispatch at
// run time, so a generic wrapper around a hand-tuned method costs nothing).
//
//   SampleProcessor  process(x): one sample in, one out
//   BlockProcessor   process_block(in, out, n)
//   Resettable       reset(): back to silence
//   SampleFilter     SampleProcessor + BlockProcessor + Resettable:
//                      double  FirFilter, IirFilter, Biquad, CombFF, CombFB
//                      float   FirFilter32, IirFilter32, Biquad32,
//                              CombFF32, CombFB32
//                      Q8.24   FFirFilter, FIirFilter, FBiquad,
//                              FCombFF, FCombFB
//                      and FilterChain<S, A, B> of any two of one type
//   SampleGenerator  next(): one sample out, no input: KarplusStrong,
//                    KarplusStrong32, FKarplusStrong
//
// One set of traits covers all three sample types. A trait is satisfied by
// method name, so the signatures below are written with double, and a
// float or Q8.24 (Fixed) filter conforms the same way; the helpers take
// the sample type as a parameter of their own (S), inferred from the
// buffers. Your own filter works with them (and with any
// generic<F: SampleFilter> code) as soon as it has the methods.
//
//   struct Biquad hp = Biquad::highpass(48000.0, 80.0, 0.7071);
//   struct Biquad lp = Biquad::lowpass(48000.0, 8000.0, 0.7071);
//   struct FilterChain<double, struct Biquad, struct Biquad> band =
//       FilterChain::of(hp, lp);
//   band.process_block(in, out, n);                  // band-limit a block
//
//   double h[64];
//   filter_impulse_response(&fir, &h[0], 64UL);      // h[i] = taps, then 0
//
// The per-type process_block bodies with a fast path of their own (FIR's
// sliding dot products, the biquad's look-ahead recurrence) stay hand-
// written; the others are filter_process_each over process().
#include <std/dsp/fixed.h>
#include <std/traits.h>   // Resettable

namespace std {

trait SampleProcessor {
    double process(double x);
}

trait BlockProcessor {
    // out[i] = process(in[i]) for i < n; out may alias in.
    void process_block(const &double in, &double out, unsigned long n);
}

trait SampleFilter : SampleProcessor + BlockProcessor + Resettable {}

trait SampleGenerator {
    // The next output sample (no input: the generator runs on its state).
    double next();
}

// ── Sample constants ──────────────────────────────────────────────────────────
// 1.0 as a sample of type S: FIXED_ONE for Q8.24, 1.0 for double/float.
// The test tells Fixed (an int newtype, where 0.5 truncates to 0) from the
// floating types; the optimizer folds it. (It is a plain 'if': 'if const'
// doesn't fold a cast to a newtype yet, so it can't make that test.)

generic<S>
inline S sample_one() {
    if ((S)0.5 == (S)0) { return (S)FIXED_ONE; }
    return (S)1.0;
}

// ── Per-sample block loop ─────────────────────────────────────────────────────
// out[i] = f.process(in[i]), one sample at a time (in == out is fine): the
// process_block of a filter with no faster block form.

generic<F: SampleProcessor, S>
void filter_process_each(&F f, const &S in, &S out, unsigned long n) {
    unsigned long i = 0UL;
    while (i < n) {
        S xi;
        unsafe { xi = in[i]; }
        S yi = f.process(xi);
        unsafe { out[i] = yi; }
        i = i + 1UL;
    }
}

// ── Impulse and step response ─────────────────────────────────────────────────
// The first n samples of f's response to a unit impulse (1, 0, 0, ...) or
// a unit step (1, 1, 1, ...), from a reset state. f is left reset, ready
// for real input. For a FIR the impulse response is its taps.

generic<F: SampleProcessor + Resettable, S>
void filter_impulse_response(&F f, &S out, unsigned long n) {
    S one = sample_one();
    f.reset();
    unsigned long i = 0UL;
    while (i < n) {
        S yi = f.process(i == 0UL ? one : (S)0);
        unsafe { out[i] = yi; }
        i = i + 1UL;
    }
    f.reset();
}

generic<F: SampleProcessor + Resettable, S>
void filter_step_response(&F f, &S out, unsigned long n) {
    S one = sample_one();
    f.reset();
    unsigned long i = 0UL;
    while (i < n) {
        S yi = f.process(one);
        unsafe { out[i] = yi; }
        i = i + 1UL;
    }
    f.reset();
}

// ── Generators ────────────────────────────────────────────────────────────────
// out[i] = g.next() for i < n.

generic<G: SampleGenerator, S>
void generate_block(&G g, &S out, unsigned long n) {
    unsigned long i = 0UL;
    while (i < n) {
        S yi = g.next();
        unsafe { out[i] = yi; }
        i = i + 1UL;
    }
}

// ── Cascade ───────────────────────────────────────────────────────────────────
// Two filters in series, itself a SampleFilter: process(x) is
// second(first(x)), and process_block runs each stage over the whole block
// with its own process_block (so each keeps its fast path), using 'out' as
// the intermediate buffer (no scratch; out may alias in). S is the sample
// type (double, float or Fixed). Chains nest, so any number of stages is a
// chain of chains:
//
//   struct FilterChain<double, struct Biquad, struct Biquad> band =
//       FilterChain::of(hp, lp);
//   struct FilterChain<double, struct FilterChain<double, struct Biquad,
//       struct Biquad>, struct FirFilter> eq = FilterChain::of(band, fir);
//
// of() copies the stages into the chain; reach them as .first / .second.
// A stage that keeps its history in your buffer (FIR, IIR, comb) still
// uses that buffer, so don't run the original alongside the chain's copy.

generic<S, A: SampleFilter, B: SampleFilter>
struct FilterChain {
    A first;
    B second;

    S    process(S x);
    void reset();
    void process_block(const &S in, &S out, unsigned long n);

    static struct FilterChain<S, A, B> of(A first, B second);
};

generic<S, A: SampleFilter, B: SampleFilter>
inline S FilterChain::process(S x) {
    return self.second.process(self.first.process(x));
}

generic<S, A: SampleFilter, B: SampleFilter>
void FilterChain::reset() {
    self.first.reset();
    self.second.reset();
}

generic<S, A: SampleFilter, B: SampleFilter>
void FilterChain::process_block(const &S in, &S out, unsigned long n) {
    self.first.process_block(in, out, n);
    self.second.process_block(out, out, n);
}

generic<S, A: SampleFilter, B: SampleFilter>
struct FilterChain<S, A, B> FilterChain::of(A first, B second) {
    struct FilterChain<S, A, B> c;
    c.first = first;
    c.second = second;
    return c;
}

} // namespace std
