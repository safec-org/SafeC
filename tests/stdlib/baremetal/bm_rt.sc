#include <std/stddef.h>

// Minimal bare-metal runtime for the emulator test image: the handful of C
// library symbols freestanding std code links against (a real board gets
// these from newlib/picolibc or its own BSP). Bump allocator over a static
// arena; free() only reclaims the most recent block.

#ifndef BM_HEAP_SIZE
#define BM_HEAP_SIZE (1024 * 1024)
#endif

static unsigned char g_heap[BM_HEAP_SIZE];
static unsigned long g_heap_top = 0UL;
static unsigned long g_last_block = 0UL;

// Every block is preceded by an 8-byte header holding its size, so realloc
// knows how much to copy; blocks are 16-byte aligned.
?&unsigned char malloc(size_t size) {
    unsigned long start = (g_heap_top + 15UL) & ~15UL;
    unsigned long total = 16UL + (((unsigned long)size + 15UL) & ~15UL);
    if (start + total > (unsigned long)BM_HEAP_SIZE) { return null; }
    unsafe {
        &unsigned long hdr = (&unsigned long)&g_heap[start + 8UL];
        *hdr = size;
        g_last_block = start;
        g_heap_top = start + total;
        return (?&unsigned char)&g_heap[start + 16UL];
    }
}

void free(?&unsigned char p) {
    if (p.is_null()) { return; }
    unsafe {
        unsigned long off = (unsigned long)p - (unsigned long)&g_heap[0];
        if (off - 16UL == g_last_block) { g_heap_top = g_last_block; }
    }
}

&unsigned char memcpy(&unsigned char dst, const &unsigned char src, size_t n) {
    unsafe {
        &unsigned char d = (&unsigned char)dst;
        const &unsigned char s = (const &unsigned char)src;
        unsigned long i = 0UL;
        while (i < n) { d[i] = s[i]; i = i + 1UL; }
    }
    return dst;
}

&unsigned char memmove(&unsigned char dst, const &unsigned char src, size_t n) {
    unsafe {
        &unsigned char d = (&unsigned char)dst;
        const &unsigned char s = (const &unsigned char)src;
        if ((unsigned long)dst < (unsigned long)src) {
            unsigned long i = 0UL;
            while (i < n) { d[i] = s[i]; i = i + 1UL; }
        } else {
            unsigned long i = n;
            while (i > 0UL) { i = i - 1UL; d[i] = s[i]; }
        }
    }
    return dst;
}

&unsigned char memset(&unsigned char p, int v, size_t n) {
    unsafe {
        &unsigned char d = (&unsigned char)p;
        unsigned long i = 0UL;
        while (i < n) { d[i] = (unsigned char)v; i = i + 1UL; }
    }
    return p;
}

int memcmp(const &unsigned char a, const &unsigned char b, size_t n) {
    unsafe {
        const &unsigned char x = (const &unsigned char)a;
        const &unsigned char y = (const &unsigned char)b;
        unsigned long i = 0UL;
        while (i < n) {
            if (x[i] != y[i]) { return (int)x[i] - (int)y[i]; }
            i = i + 1UL;
        }
    }
    return 0;
}

?&unsigned char calloc(size_t count, size_t size) {
    ?&unsigned char p = malloc(count * size);
    match (p) {
        case null: { }
        case some(b): { memset(b, 0, count * size); }
    }
    return p;
}

?&unsigned char realloc(?&unsigned char p, size_t size) {
    match (p) {
        case null: { return malloc(size); }
        case some(src): {
            ?&unsigned char q = malloc(size);
            match (q) {
                case null: { return null; }
                case some(dst): {
                    unsafe {
                        unsigned long old = *(&unsigned long)((unsigned long)src - 8UL);
                        memcpy(dst, src, old < (unsigned long)size ? (size_t)old : size);
                    }
                }
            }
            return q;
        }
    }
    return null;
}

size_t strlen(const &char s) {
    unsigned long n = 0UL;
    unsafe { while (s[n] != (char)0) { n = n + 1UL; } }
    return (size_t)n;
}

// No console: diagnostics are dropped; abort() parks the CPU where the
// emulator's harness detects it.
struct BmFile_ { char opaque_; };   // C's FILE: never a real stream here
int fprintf(?&BmFile_ stream, const &char fmt, ...) { return 0; }
int printf(const &char fmt, ...) { return 0; }
?&BmFile_ stderr = null;
?&BmFile_ stdout = null;

volatile int g_aborted = 0;
void abort() {
    g_aborted = 1;
    while (1) { }
}

// ── Atomics for cores without atomic instructions (Thumb-1, RV32I/E and
// RV64I/E without 'A'): the compiler lowers atomic ops to these libcalls.
// Single-core, no preemption in this test image, so plain loads/stores are
// atomic; a real BSP would mask interrupts around each body.

unsigned char __atomic_load_1(const &unsigned char p, int order) { return *p; }
void __atomic_store_1(&unsigned char p, unsigned char v, int order) { *p = v; }
unsigned char __atomic_exchange_1(&unsigned char p, unsigned char v, int order) { unsigned char old = *p; *p = v; return old; }
bool __atomic_compare_exchange_1(&unsigned char p, &unsigned char expected, unsigned char desired, int so, int fo) {
    if (*p == *expected) { *p = desired; return true; }
    *expected = *p;
    return false;
}
unsigned char __atomic_fetch_add_1(&unsigned char p, unsigned char v, int order) { unsigned char old = *p; *p = (unsigned char)(old + v); return old; }
unsigned char __atomic_fetch_sub_1(&unsigned char p, unsigned char v, int order) { unsigned char old = *p; *p = (unsigned char)(old - v); return old; }
unsigned char __atomic_fetch_and_1(&unsigned char p, unsigned char v, int order) { unsigned char old = *p; *p = (unsigned char)(old & v); return old; }
unsigned char __atomic_fetch_or_1(&unsigned char p, unsigned char v, int order) { unsigned char old = *p; *p = (unsigned char)(old | v); return old; }
unsigned char __atomic_fetch_xor_1(&unsigned char p, unsigned char v, int order) { unsigned char old = *p; *p = (unsigned char)(old ^ v); return old; }

unsigned short __atomic_load_2(const &unsigned short p, int order) { return *p; }
void __atomic_store_2(&unsigned short p, unsigned short v, int order) { *p = v; }
unsigned short __atomic_exchange_2(&unsigned short p, unsigned short v, int order) { unsigned short old = *p; *p = v; return old; }
bool __atomic_compare_exchange_2(&unsigned short p, &unsigned short expected, unsigned short desired, int so, int fo) {
    if (*p == *expected) { *p = desired; return true; }
    *expected = *p;
    return false;
}
unsigned short __atomic_fetch_add_2(&unsigned short p, unsigned short v, int order) { unsigned short old = *p; *p = (unsigned short)(old + v); return old; }
unsigned short __atomic_fetch_sub_2(&unsigned short p, unsigned short v, int order) { unsigned short old = *p; *p = (unsigned short)(old - v); return old; }
unsigned short __atomic_fetch_and_2(&unsigned short p, unsigned short v, int order) { unsigned short old = *p; *p = (unsigned short)(old & v); return old; }
unsigned short __atomic_fetch_or_2(&unsigned short p, unsigned short v, int order) { unsigned short old = *p; *p = (unsigned short)(old | v); return old; }
unsigned short __atomic_fetch_xor_2(&unsigned short p, unsigned short v, int order) { unsigned short old = *p; *p = (unsigned short)(old ^ v); return old; }

unsigned int __atomic_load_4(const &unsigned int p, int order) { return *p; }
void __atomic_store_4(&unsigned int p, unsigned int v, int order) { *p = v; }
unsigned int __atomic_exchange_4(&unsigned int p, unsigned int v, int order) { unsigned int old = *p; *p = v; return old; }
bool __atomic_compare_exchange_4(&unsigned int p, &unsigned int expected, unsigned int desired, int so, int fo) {
    if (*p == *expected) { *p = desired; return true; }
    *expected = *p;
    return false;
}
unsigned int __atomic_fetch_add_4(&unsigned int p, unsigned int v, int order) { unsigned int old = *p; *p = (unsigned int)(old + v); return old; }
unsigned int __atomic_fetch_sub_4(&unsigned int p, unsigned int v, int order) { unsigned int old = *p; *p = (unsigned int)(old - v); return old; }
unsigned int __atomic_fetch_and_4(&unsigned int p, unsigned int v, int order) { unsigned int old = *p; *p = (unsigned int)(old & v); return old; }
unsigned int __atomic_fetch_or_4(&unsigned int p, unsigned int v, int order) { unsigned int old = *p; *p = (unsigned int)(old | v); return old; }
unsigned int __atomic_fetch_xor_4(&unsigned int p, unsigned int v, int order) { unsigned int old = *p; *p = (unsigned int)(old ^ v); return old; }

unsigned long long __atomic_load_8(const &unsigned long long p, int order) { return *p; }
void __atomic_store_8(&unsigned long long p, unsigned long long v, int order) { *p = v; }
unsigned long long __atomic_exchange_8(&unsigned long long p, unsigned long long v, int order) { unsigned long long old = *p; *p = v; return old; }
bool __atomic_compare_exchange_8(&unsigned long long p, &unsigned long long expected, unsigned long long desired, int so, int fo) {
    if (*p == *expected) { *p = desired; return true; }
    *expected = *p;
    return false;
}
unsigned long long __atomic_fetch_add_8(&unsigned long long p, unsigned long long v, int order) { unsigned long long old = *p; *p = (unsigned long long)(old + v); return old; }
unsigned long long __atomic_fetch_sub_8(&unsigned long long p, unsigned long long v, int order) { unsigned long long old = *p; *p = (unsigned long long)(old - v); return old; }
unsigned long long __atomic_fetch_and_8(&unsigned long long p, unsigned long long v, int order) { unsigned long long old = *p; *p = (unsigned long long)(old & v); return old; }
unsigned long long __atomic_fetch_or_8(&unsigned long long p, unsigned long long v, int order) { unsigned long long old = *p; *p = (unsigned long long)(old | v); return old; }
unsigned long long __atomic_fetch_xor_8(&unsigned long long p, unsigned long long v, int order) { unsigned long long old = *p; *p = (unsigned long long)(old ^ v); return old; }
