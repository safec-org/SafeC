// SafeC Standard Library — System declarations (C11/C17/C23 stdlib coverage)
#pragma once

// ── Process control constants ─────────────────────────────────────────────────
#define EXIT_SUCCESS  0
#define EXIT_FAILURE  1

// ── PRNG constants ────────────────────────────────────────────────────────────
#define RAND_MAX  2147483647

// ── Process control ───────────────────────────────────────────────────────────
void          sys_exit(int code);
void          sys_abort();
int           sys_system(const char* cmd);
int           sys_atexit(void* fn);

// ── Environment ───────────────────────────────────────────────────────────────
const char*   sys_getenv(const char* name);

// ── PRNG ──────────────────────────────────────────────────────────────────────
int           sys_rand();
void          sys_srand(unsigned int seed);

// ── Integer arithmetic ────────────────────────────────────────────────────────
int           sys_abs(int x);
long long     sys_llabs(long long x);

// ── String → number conversions ───────────────────────────────────────────────
int           sys_atoi(const char* s);
long long     sys_atoll(const char* s);
double        sys_atof(const char* s);

// Strtoll with explicit base (0 = auto-detect 0x/0 prefix).
// If `end` is non-NULL it is set to the first character not consumed.
long long     sys_strtoll(const char* s, char** end, int base);
unsigned long long sys_strtoull(const char* s, char** end, int base);
double        sys_strtod(const char* s, char** end);
float         sys_strtof(const char* s, char** end);

// ── Sorting and searching ─────────────────────────────────────────────────────
// Sort `n` elements of size `elem_size` starting at `base` using comparator `cmp`.
// cmp(a, b) must return <0 / 0 / >0  (same as strcmp convention).
void  sys_qsort(void* base, unsigned long n, unsigned long elem_size, void* cmp);

// Binary search in a sorted array.  Returns pointer to matching element or NULL.
void* sys_bsearch(const void* key, const void* base,
                  unsigned long n, unsigned long elem_size, void* cmp);

// ── Clocks ────────────────────────────────────────────────────────────────────
long long     sys_cpu_time_ns();
long long     sys_time_now_ns();
long long     sys_monotonic_ns();
