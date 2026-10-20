// SafeC Standard Library — System declarations (C11/C17/C23 stdlib coverage)
#pragma once

// ── Process control constants ─────────────────────────────────────────────────
#define EXIT_SUCCESS  0
#define EXIT_FAILURE  1

// ── PRNG constants ────────────────────────────────────────────────────────────
#define RAND_MAX  2147483647

// ── Process control ───────────────────────────────────────────────────────────
namespace std {

void          sys_exit(int code);
void          sys_abort();
int           sys_system(const &char cmd);
int           sys_atexit(fn void() func);

// ── Environment ───────────────────────────────────────────────────────────────
const ?&char  sys_getenv(const &char name);

// ── PRNG ──────────────────────────────────────────────────────────────────────
int           sys_rand();
void          sys_srand(unsigned int seed);

// ── Integer arithmetic ────────────────────────────────────────────────────────
int           sys_abs(int x);
long long     sys_llabs(long long x);

// ── String → number conversions ───────────────────────────────────────────────
int           sys_atoi(const &char s);
long long     sys_atoll(const &char s);
double        sys_atof(const &char s);

// Strtoll with explicit base (0 = auto-detect 0x/0 prefix).
// If `end` is non-null it is set to the first character not consumed.
long long     sys_strtoll(const &char s, ?&&const char end, int base);
unsigned long long sys_strtoull(const &char s, ?&&const char end, int base);
double        sys_strtod(const &char s, ?&&const char end);
float         sys_strtof(const &char s, ?&&const char end);

// ── Sorting and searching ─────────────────────────────────────────────────────
// libc-boundary sort/search over untyped elements; use sys_qsort/sys_bsearch.
void  sys_qsort_raw_(void* base, unsigned long n, unsigned long elem_size, void* cmp);
void* sys_bsearch_raw_(const void* key, const void* base,
                       unsigned long n, unsigned long elem_size, void* cmp);

// Sort the n elements starting at `first`; cmp(a, b) returns <0 / 0 / >0.
generic<T>
void sys_qsort(&T first, unsigned long n, fn int(const &T, const &T) cmp) {
    unsafe { sys_qsort_raw_((void*)first, n, sizeof(T), (void*)cmp); }
}

// Binary search of a sorted array; the matching element, or null.
generic<T>
?&T sys_bsearch(const &T key, const &T first, unsigned long n, fn int(const &T, const &T) cmp) {
    unsafe { return (?&T)sys_bsearch_raw_((const void*)key, (const void*)first, n, sizeof(T), (void*)cmp); }
}

// ── Clocks ────────────────────────────────────────────────────────────────────
long long     sys_cpu_time_ns();
long long     sys_time_now_ns();
long long     sys_monotonic_ns();

} // namespace std
