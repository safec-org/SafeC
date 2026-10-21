// SafeC Standard Library — System (C11/C17/C23 stdlib coverage)
// Process control, PRNG, conversions, sorting, and clock wrappers.
#pragma once
#include <std/stddef.h>
#include <std/sys.h>
#include <std/time.h>

// ── Explicit extern declarations (avoid complex C header parsing) ─────────────
namespace std {

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
extern int   atexit(void* func);
// qsort/bsearch: declared with void* comparator for SafeC compatibility
extern void  qsort(void* base, size_t n, size_t size, void* cmp);
extern void* bsearch(const void* key, const void* base, size_t n,
                     size_t size, void* cmp);

// ── Process control ───────────────────────────────────────────────────────────

// Terminate the process with the given exit code.
inline void sys_exit(int code) {
    unsafe { exit(code); }
}

// Terminate the process abnormally (raises SIGABRT / triggers abort handler).
inline void sys_abort() {
    unsafe { abort(); }
}

// Return the value of the environment variable named `name`,
// or NULL if not set.  The returned pointer is owned by the environment.
inline const ?&char sys_getenv(const &char name) {
    unsafe { return getenv(name); }
}

// Return processor time consumed by the program in nanoseconds.
// Same clock as time_cpu_ns() (time.h).  Returns -1 on error.
inline long long sys_cpu_time_ns() {
    return time_cpu_ns();
}

// Return wall-clock time in nanoseconds since the Unix epoch.
// Same clock as time_wall_ns() (time.h).  Returns -1 on error.
inline long long sys_time_now_ns() {
    return time_wall_ns();
}

// Return a monotonic timestamp in nanoseconds (suitable for elapsed-time
// measurement; epoch is unspecified).  Returns -1 on error.
inline long long sys_monotonic_ns() {
    return time_mono_ns();
}

// ── Process control ───────────────────────────────────────────────────────────

// Run a shell command and return its exit status.  NULL → check if shell available.
inline int sys_system(const &char cmd) {
    unsafe { return system(cmd); }
}

// Register a function to be called at normal process exit.
// Returns 0 on success, non-zero if the registration limit is exceeded.
inline int sys_atexit(fn void() func) {
    unsafe { return atexit((void*)func); }
}

// ── PRNG ──────────────────────────────────────────────────────────────────────

// Return a pseudo-random integer in [0, RAND_MAX].
inline int sys_rand() {
    unsafe { return rand(); }
}

// Seed the PRNG.  Use sys_time_now_ns() for a time-based seed.
inline void sys_srand(unsigned int seed) {
    unsafe { srand(seed); }
}

// ── Integer arithmetic ────────────────────────────────────────────────────────

// Absolute value of a 32-bit signed integer.
inline int sys_abs(int x) {
    if (x < 0) { return -x; }
    return x;
}

// Absolute value of a 64-bit signed integer.
inline long long sys_llabs(long long x) {
    if (x < 0LL) { return -x; }
    return x;
}

// ── String → number conversions ───────────────────────────────────────────────

// Convert a decimal string to int.  Stops at the first non-digit.
inline int sys_atoi(const &char s) {
    unsafe { return atoi(s); }
}

// Convert a decimal string to long long.
inline long long sys_atoll(const &char s) {
    unsafe { return atoll(s); }
}

// Convert a decimal/scientific string to double.
inline double sys_atof(const &char s) {
    unsafe { return atof(s); }
}

// Convert string to long long with explicit base (0 = auto-detect 0x/0 prefix).
inline long long sys_strtoll(const &char s, ?&&const char end, int base) {
    unsafe {
        char* e = (char*)0;
        long long v = strtoll(s, &e, base);
        match (end) { case some(out): { *out = e; } case null: { } }
        return v;
    }
}

// Convert string to unsigned long long.
inline unsigned long long sys_strtoull(const &char s, ?&&const char end, int base) {
    unsafe {
        char* e = (char*)0;
        unsigned long long v = strtoull(s, &e, base);
        match (end) { case some(out): { *out = e; } case null: { } }
        return v;
    }
}

// Convert string to double.
inline double sys_strtod(const &char s, ?&&const char end) {
    unsafe {
        char* e = (char*)0;
        double v = strtod(s, &e);
        match (end) { case some(out): { *out = e; } case null: { } }
        return v;
    }
}

// Convert string to float.
inline float sys_strtof(const &char s, ?&&const char end) {
    unsafe {
        char* e = (char*)0;
        float v = strtof(s, &e);
        match (end) { case some(out): { *out = e; } case null: { } }
        return v;
    }
}

// ── Sorting and searching ─────────────────────────────────────────────────────

// Sort n elements of elem_size bytes in place.
// cmp is a pointer to a function: int cmp(const void* a, const void* b)
inline void sys_qsort_raw_(void* base, unsigned long n, unsigned long elem_size, void* cmp) {
    unsafe { qsort(base, n, elem_size, cmp); }
}

// Binary search in sorted array.  Returns pointer to match or NULL.
void* sys_bsearch_raw_(const void* key, const void* base,
                  unsigned long n, unsigned long elem_size, void* cmp) {
    unsafe { return bsearch(key, base, n, elem_size, cmp); }
}

} // namespace std
