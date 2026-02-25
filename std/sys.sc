// SafeC Standard Library — System (C11/C17/C23 stdlib coverage)
// Process control, PRNG, conversions, sorting, and clock wrappers.
#pragma once
#include "sys.h"

// ── Explicit extern declarations (avoid complex C header parsing) ─────────────
extern void  exit(int code);
extern void  abort();
extern char* getenv(const char* name);
extern int   system(const char* cmd);
extern int   rand();
extern void  srand(unsigned int seed);
extern int   abs(int x);
extern long long llabs(long long x);
extern int   atoi(const char* s);
extern long long atoll(const char* s);
extern double atof(const char* s);
extern long long strtoll(const char* s, char** end, int base);
extern unsigned long long strtoull(const char* s, char** end, int base);
extern double strtod(const char* s, char** end);
extern float  strtof(const char* s, char** end);
// atexit: declared with void* for SafeC compatibility (no function pointer type syntax yet)
extern int   atexit(void* fn);
// qsort/bsearch: declared with void* comparator for SafeC compatibility
extern void  qsort(void* base, unsigned long n, unsigned long size, void* cmp);
extern void* bsearch(const void* key, const void* base, unsigned long n,
                     unsigned long size, void* cmp);
// clock_gettime / POSIX time
extern int   clock_gettime(int clk_id, void* ts);

// ── POSIX clock IDs ──────────────────────────────────────────────────────────
// CLOCK_REALTIME=0, CLOCK_MONOTONIC=6, CLOCK_PROCESS_CPUTIME_ID=12 on macOS
#define SC_CLOCK_REALTIME         0
#define SC_CLOCK_MONOTONIC        6
#define SC_CLOCK_PROCESS_CPUTIME  12

// ── Process control ───────────────────────────────────────────────────────────

// Terminate the process with the given exit code.
void sys_exit(int code) {
    unsafe { exit(code); }
}

// Terminate the process abnormally (raises SIGABRT / triggers abort handler).
void sys_abort() {
    unsafe { abort(); }
}

// Return the value of the environment variable named `name`,
// or NULL if not set.  The returned pointer is owned by the environment.
const char* sys_getenv(const char* name) {
    unsafe { return getenv(name); }
}

// Return processor time consumed by the program in nanoseconds.
// Uses CLOCK_PROCESS_CPUTIME_ID on POSIX.  Returns -1 on error.
long long sys_cpu_time_ns() {
    unsafe {
        // timespec layout: { long tv_sec; long tv_nsec; } — 16 bytes on 64-bit
        long long ts[2];
        if (clock_gettime(SC_CLOCK_PROCESS_CPUTIME, (void*)ts) != 0) {
            return -1LL;
        }
        return ts[0] * 1000000000LL + ts[1];
    }
}

// Return wall-clock time in nanoseconds since the Unix epoch.
// Uses CLOCK_REALTIME.  Returns -1 on error.
long long sys_time_now_ns() {
    unsafe {
        long long ts[2];
        if (clock_gettime(SC_CLOCK_REALTIME, (void*)ts) != 0) {
            return -1LL;
        }
        return ts[0] * 1000000000LL + ts[1];
    }
}

// Return a monotonic timestamp in nanoseconds (suitable for elapsed-time
// measurement; epoch is unspecified).  Returns -1 on error.
long long sys_monotonic_ns() {
    unsafe {
        long long ts[2];
        if (clock_gettime(SC_CLOCK_MONOTONIC, (void*)ts) != 0) {
            return -1LL;
        }
        return ts[0] * 1000000000LL + ts[1];
    }
}

// ── Process control ───────────────────────────────────────────────────────────

// Run a shell command and return its exit status.  NULL → check if shell available.
int sys_system(const char* cmd) {
    unsafe { return system(cmd); }
}

// Register a function to be called at normal process exit.
// `fn` must point to a void(void) function.
// SafeC does not yet support function pointer cast syntax; fn is passed as void*.
// Returns 0 on success, non-zero if the registration limit is exceeded.
int sys_atexit(void* fn) {
    unsafe { return atexit(fn); }
}

// ── PRNG ──────────────────────────────────────────────────────────────────────

// Return a pseudo-random integer in [0, RAND_MAX].
int sys_rand() {
    unsafe { return rand(); }
}

// Seed the PRNG.  Use sys_time_now_ns() for a time-based seed.
void sys_srand(unsigned int seed) {
    unsafe { srand(seed); }
}

// ── Integer arithmetic ────────────────────────────────────────────────────────

// Absolute value of a 32-bit signed integer.
int sys_abs(int x) {
    if (x < 0) { return -x; }
    return x;
}

// Absolute value of a 64-bit signed integer.
long long sys_llabs(long long x) {
    if (x < 0LL) { return -x; }
    return x;
}

// ── String → number conversions ───────────────────────────────────────────────

// Convert a decimal string to int.  Stops at the first non-digit.
int sys_atoi(const char* s) {
    unsafe { return atoi(s); }
}

// Convert a decimal string to long long.
long long sys_atoll(const char* s) {
    unsafe { return atoll(s); }
}

// Convert a decimal/scientific string to double.
double sys_atof(const char* s) {
    unsafe { return atof(s); }
}

// Convert string to long long with explicit base (0 = auto-detect 0x/0 prefix).
long long sys_strtoll(const char* s, char** end, int base) {
    unsafe { return strtoll(s, end, base); }
}

// Convert string to unsigned long long.
unsigned long long sys_strtoull(const char* s, char** end, int base) {
    unsafe { return strtoull(s, end, base); }
}

// Convert string to double.
double sys_strtod(const char* s, char** end) {
    unsafe { return strtod(s, end); }
}

// Convert string to float.
float sys_strtof(const char* s, char** end) {
    unsafe { return strtof(s, end); }
}

// ── Sorting and searching ─────────────────────────────────────────────────────

// Sort n elements of elem_size bytes in place.
// cmp is a pointer to a function: int cmp(const void* a, const void* b)
void sys_qsort(void* base, unsigned long n, unsigned long elem_size, void* cmp) {
    unsafe { qsort(base, n, elem_size, cmp); }
}

// Binary search in sorted array.  Returns pointer to match or NULL.
void* sys_bsearch(const void* key, const void* base,
                  unsigned long n, unsigned long elem_size, void* cmp) {
    unsafe { return bsearch(key, base, n, elem_size, cmp); }
}
