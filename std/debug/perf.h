// SafeC Standard Library — Performance Counters
// Hardware performance counter access (x86-64 RDTSC, AArch64 PMCCNTR, RISC-V cycle).
// Freestanding-safe.
#pragma once

struct PerfCounter {
    unsigned long long start_val;
    unsigned long long end_val;
    unsigned long long freq_hz;    // counter frequency (0 = unknown)

    // Start measurement.
    void  start();

    // Stop measurement.
    void  stop();

    // Return elapsed counter ticks.
    unsigned long long ticks() const;

    // Return elapsed nanoseconds (0 if freq unknown).
    unsigned long long ns() const;
};

// Initialise a PerfCounter (detect frequency from arch-specific sources).
struct PerfCounter perf_counter_init();

// Read the raw cycle counter for the current arch.
unsigned long long perf_read_cycle();

// Read timestamp in nanoseconds (best-effort; returns 0 in freestanding without freq).
unsigned long long perf_read_ns();

#define PERF_BEGIN(ctr)  do { (ctr).start(); } while(0)
#define PERF_END(ctr)    do { (ctr).stop();  } while(0)
