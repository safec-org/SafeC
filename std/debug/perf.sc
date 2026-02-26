// SafeC Standard Library — Performance Counter Implementation
#pragma once
#include "perf.h"

// ── Hosted-mode extern declarations ──────────────────────────────────────────
// clock_gettime and clock are used only in hosted mode.
// CLOCK_MONOTONIC = 1 on Linux/macOS.
extern int   clock_gettime(int clk, void* ts);
extern long  clock();    // POSIX clock() returns long; CLOCKS_PER_SEC = 1000000

// ── perf_read_cycle ───────────────────────────────────────────────────────────
// Returns raw hardware cycle / virtual counter ticks.
// Architecture selected at compile time via predefined macros.

unsigned long long perf_read_cycle() {
#if defined(__x86_64__) || defined(_M_X64)
    // RDTSC: returns TSC in EDX:EAX.
    unsigned int lo = (unsigned int)0;
    unsigned int hi = (unsigned int)0;
    unsafe {
        asm volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    }
    return ((unsigned long long)hi << 32) | (unsigned long long)lo;

#elif defined(__aarch64__) || defined(__arm64__)
    // Virtual counter — available at EL0 when CNTKCTL_EL1.EL0VCTEN is set.
    unsigned long long v = (unsigned long long)0;
    unsafe {
        asm volatile ("mrs %0, cntvct_el0" : "=r"(v));
    }
    return v;

#elif defined(__riscv)
    // Unprivileged cycle CSR.
    unsigned long long v = (unsigned long long)0;
    unsafe {
        asm volatile ("csrr %0, cycle" : "=r"(v));
    }
    return v;

#else
    // Fallback: unknown architecture — return 0.
    return (unsigned long long)0;
#endif
}

// ── perf_read_ns ──────────────────────────────────────────────────────────────
// Hosted: use CLOCK_MONOTONIC for nanosecond accuracy.
// Freestanding: return 0 (caller should use ticks + known freq).

unsigned long long perf_read_ns() {
#ifndef __SAFEC_FREESTANDING__
    unsafe {
        // struct timespec { time_t tv_sec; long tv_nsec; } — 16 bytes on 64-bit.
        long long ts[2];
        if (clock_gettime(1, (void*)ts) != 0) { return (unsigned long long)0; }
        return (unsigned long long)ts[0] * (unsigned long long)1000000000
             + (unsigned long long)ts[1];
    }
#endif
    return (unsigned long long)0;
}

// ── PerfCounter methods ───────────────────────────────────────────────────────

void PerfCounter::start() {
    self.start_val = perf_read_cycle();
    self.end_val   = self.start_val;
}

void PerfCounter::stop() {
    self.end_val = perf_read_cycle();
}

unsigned long long PerfCounter::ticks() const {
    return self.end_val - self.start_val;
}

unsigned long long PerfCounter::ns() const {
    if (self.freq_hz == (unsigned long long)0) { return (unsigned long long)0; }
    unsigned long long t = self.end_val - self.start_val;
    // ns = ticks * 1e9 / freq_hz.  Avoid overflow: compute (ticks / freq_hz) * 1e9
    // then add the remainder contribution.
    unsigned long long q = t / self.freq_hz;
    unsigned long long r = t % self.freq_hz;
    return q * (unsigned long long)1000000000
         + r * (unsigned long long)1000000000 / self.freq_hz;
}

// ── perf_counter_init ─────────────────────────────────────────────────────────
// Attempts to calibrate the cycle counter frequency using clock() over a short
// spin.  In freestanding mode, or if calibration fails, sets freq_hz = 0.

struct PerfCounter perf_counter_init() {
    struct PerfCounter ctr;
    ctr.start_val = (unsigned long long)0;
    ctr.end_val   = (unsigned long long)0;
    ctr.freq_hz   = (unsigned long long)0;

#ifndef __SAFEC_FREESTANDING__
    // Measure cycles elapsed during a known clock() interval.
    // clock() resolution is 1/CLOCKS_PER_SEC (= 1 µs on most POSIX systems).
    // We spin until clock() advances by at least ~10 ms worth of ticks.
    // CLOCKS_PER_SEC is typically 1,000,000 on POSIX.
    unsigned long long target_clocks = (unsigned long long)10000; // 10 ms @ 1e6 clks/s

    unsafe {
        long c0 = clock();
        unsigned long long cy0 = perf_read_cycle();
        long c1 = clock();
        // Spin until at least target_clocks have elapsed.
        while ((unsigned long long)(c1 - c0) < target_clocks) {
            c1 = clock();
        }
        unsigned long long cy1 = perf_read_cycle();

        unsigned long long elapsed_clocks = (unsigned long long)(c1 - c0);
        if (elapsed_clocks > (unsigned long long)0 && cy1 > cy0) {
            // freq = (cy1-cy0) * CLOCKS_PER_SEC / elapsed_clocks
            // Use 1000000 as CLOCKS_PER_SEC (standard POSIX value).
            unsigned long long clks_per_sec = (unsigned long long)1000000;
            ctr.freq_hz = (cy1 - cy0) * clks_per_sec / elapsed_clocks;
        }
    }
#endif

    return ctr;
}
