// SafeC Standard Library — Fuzz Harness
// Lightweight structure-aware in-process fuzzer.
// Mutates a seed corpus and feeds inputs to a target function.
// Not coverage-guided; use with libFuzzer for production fuzzing.
#pragma once

namespace std {

struct FuzzTarget {
    fn void([]const unsigned char) func;   // the target, called with each input
    unsigned long seed;      // PRNG seed (set before run())
    unsigned long iters;     // number of mutations to try

    // A fuzz target over `func` with a default seed and `iters` mutations.
    static struct FuzzTarget new(fn void([]const unsigned char) func, unsigned long iters);

    // Run the fuzzer: start from `corpus` and apply `iters` random mutations.
    // Crashes/panics in func are not caught; they surface as normal crashes.
    void run([]const unsigned char corpus);
};

} // namespace std
