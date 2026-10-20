// SafeC Standard Library — Benchmark Framework
// Measures wall-clock time (via clock()) for repeated function calls.
// Freestanding note: clock() requires hosted libc; in freestanding mode
// timing is skipped and only iteration counts are reported.
#pragma once

#define BENCH_MAX      64
#define BENCH_NAME_MAX 64

namespace std {

struct BenchCase {
    char          name[BENCH_NAME_MAX];
    void*         func;         // type-erased fn void(?&T), set by bench_add
    void*         arg;          // type-erased ?&T passed to func
    unsigned long iters;        // number of iterations to run
    double        elapsed_s;    // measured elapsed seconds (0 if unavailable)
    double        ops_per_s;    // iters / elapsed_s (0 if unavailable)
};

struct BenchSuite {
    struct BenchCase cases[BENCH_MAX];
    int count;

    // Type-erased registration behind bench_add.
    void add_raw_(const char* name, void* func, void* arg, unsigned long iters);

    // Run all benchmarks and record results.
    void run();

    // Print results table to stdout.
    void print_results() const;
};

struct BenchSuite bench_suite_init();

// Register a benchmark: func(arg) is called `iters` times per run.
generic<T>
void bench_add(&BenchSuite b, const &char name, fn void(?&T) func, ?&T arg, unsigned long iters) {
    unsafe { b.add_raw_(name, (void*)func, (void*)arg, iters); }
}

} // namespace std
