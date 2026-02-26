// SafeC Standard Library — Benchmark Framework
#pragma once
#include "bench.h"

extern int    printf(const char* fmt, ...);
extern long   clock();           // returns clock_t; CLOCKS_PER_SEC = 1000000
extern void*  memset(void* p, int v, unsigned long n);

#define CLOCKS_PER_SEC_VAL 1000000

static void bench_copy_(char* dst, const char* src, unsigned long n) {
    unsigned long i = (unsigned long)0;
    while (i < n - (unsigned long)1 && src[i] != '\0') {
        dst[i] = src[i];
        i = i + (unsigned long)1;
    }
    dst[i] = '\0';
}

struct BenchSuite bench_suite_init() {
    struct BenchSuite s;
    s.count = 0;
    return s;
}

void BenchSuite::add(const char* name, void* fn, void* arg, unsigned long iters) {
    if (self.count >= BENCH_MAX) { return; }
    int idx = self.count;
    bench_copy_(self.cases[idx].name, name, (unsigned long)BENCH_NAME_MAX);
    self.cases[idx].fn        = fn;
    self.cases[idx].arg       = arg;
    self.cases[idx].iters     = iters;
    self.cases[idx].elapsed_s = 0.0;
    self.cases[idx].ops_per_s = 0.0;
    self.count = self.count + 1;
}

void BenchSuite::run() {
    int i = 0;
    while (i < self.count) {
        unsigned long n = self.cases[i].iters;

        unsafe {
            long t0 = clock();
            unsigned long j = (unsigned long)0;
            while (j < n) {
                ((void(*)(void*))self.cases[i].fn)(self.cases[i].arg);
                j = j + (unsigned long)1;
            }
            long t1 = clock();

            double elapsed = (double)(t1 - t0) / (double)CLOCKS_PER_SEC_VAL;
            self.cases[i].elapsed_s = elapsed;
            if (elapsed > 0.0) {
                self.cases[i].ops_per_s = (double)n / elapsed;
            }
        }
        i = i + 1;
    }
}

void BenchSuite::print_results() const {
    unsafe {
        printf("%-40s %10s %14s %14s\n",
               "benchmark", "iters", "elapsed (s)", "ops/s");
        printf("%-40s %10s %14s %14s\n",
               "─────────────────────────────────────────",
               "──────────", "─────────────", "─────────────");
        int i = 0;
        while (i < self.count) {
            printf("%-40s %10llu %14.6f %14.0f\n",
                   self.cases[i].name,
                   (unsigned long long)self.cases[i].iters,
                   self.cases[i].elapsed_s,
                   self.cases[i].ops_per_s);
            i = i + 1;
        }
    }
}
