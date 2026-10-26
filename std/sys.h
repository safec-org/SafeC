// SafeC Standard Library — System declarations (C11/C17/C23 stdlib coverage)
#pragma once
#include <std/erased.h>

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
// libc-boundary sort/search over type-erased elements (see std/erased.h):
// 'base' is the elements' bytes, cmp a typed comparator erased by
// sys_qsort/sys_bsearch, which are what to call.
typedef fn int(const &Erased, const &Erased) SysCmpRaw_;
void  sys_qsort_raw_(unsigned char* base, unsigned long n, unsigned long elem_size, SysCmpRaw_ cmp);
const unsigned char* sys_bsearch_raw_(const &Erased key, const unsigned char* base,
                                      unsigned long n, unsigned long elem_size, SysCmpRaw_ cmp);

// Sorts items in place; cmp(a, b) returns <0 / 0 / >0.
generic<T>
void sys_qsort([]T items, fn int(const &T, const &T) cmp) {
    if (items.len < 2UL) { return; }
    unsafe { sys_qsort_raw_((unsigned char*)items.ptr, items.len, sizeof(T), (SysCmpRaw_)cmp); }
}

// Binary search of sorted items; the index of an element equal to key, or
// null.
generic<T>
?unsigned long sys_bsearch(const &T key, []const T items, fn int(const &T, const &T) cmp) {
    if (items.len == 0UL) { return null; }
    const unsigned char* hit;
    unsafe {
        hit = sys_bsearch_raw_((const &Erased)key, (const unsigned char*)items.ptr, items.len, sizeof(T), (SysCmpRaw_)cmp);
    }
    if (hit == (const unsigned char*)0) { return null; }
    unsigned long off;
    unsafe { off = ((unsigned long)hit - (unsigned long)items.ptr) / sizeof(T); }
    return off;
}

// ── Clocks ────────────────────────────────────────────────────────────────────
long long     sys_cpu_time_ns();
long long     sys_time_now_ns();
long long     sys_monotonic_ns();

} // namespace std
