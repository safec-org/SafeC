// SafeC Standard Library — Fuzz Harness
// Lightweight structure-aware in-process fuzzer.
// Mutates a seed corpus and feeds inputs to a target function.
// Not coverage-guided; use with libFuzzer for production fuzzing.
#pragma once

// Fuzz target signature: void func(const unsigned char* data, unsigned long size)
namespace std {

struct FuzzTarget {
    void*         func;        // fuzz target function
    unsigned long seed;      // PRNG seed (set before run())
    unsigned long iters;     // number of mutations to try

    // Run the fuzzer: start from `corpus` and apply `iters` random mutations.
    // Crashes/panics in func are not caught; they surface as normal crashes.
    void run(const unsigned char* corpus, unsigned long corpus_size);
};

// Initialise a fuzz target with a default seed and iteration count.
struct FuzzTarget fuzz_target_init(void* func, unsigned long iters);

} // namespace std
