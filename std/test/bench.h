// SafeC Standard Library — Benchmark Framework
// Measures wall-clock time (via clock()) for repeated function calls.
// Freestanding note: clock() requires hosted libc; in freestanding mode
// timing is skipped and only iteration counts are reported.
#pragma once

#define BENCH_MAX      64
#define BENCH_NAME_MAX 64

struct BenchCase {
    char          name[BENCH_NAME_MAX];
    void*         fn;           // void (*fn)(void* arg)
    void*         arg;          // user argument passed to fn
    unsigned long iters;        // number of iterations to run
    double        elapsed_s;    // measured elapsed seconds (0 if unavailable)
    double        ops_per_s;    // iters / elapsed_s (0 if unavailable)
};

struct BenchSuite {
    struct BenchCase cases[BENCH_MAX];
    int count;

    // Register a benchmark: fn is called `iters` times per run.
    void add(const char* name, void* fn, void* arg, unsigned long iters);

    // Run all benchmarks and record results.
    void run();

    // Print results table to stdout.
    void print_results() const;
};

struct BenchSuite bench_suite_init();
