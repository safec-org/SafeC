// SafeC Standard Library — freestanding wasm32 runtime (see wasm_rt.h).
// Top-level (non-namespaced) definitions on purpose: these must link
// against the *bare*, unmangled 'malloc'/'free'/'strlen'/etc. symbols
// that std's 'extern' declarations reference (SafeC only mangles function
// *definitions* inside a namespace — 'extern' declarations and top-level
// definitions both keep the bare C-ABI name, which is how this file's
// malloc() satisfies std::malloc's extern declaration at link time
// without either side needing to know about the other).
// safeguard: runtime wasm                (malloc & co. for a WebAssembly module)
#pragma once
#include <std/wasm/wasm_rt.h>
//
// NOT part of the general stdlib archive (libsafec_std.a): this file
// defines bare libc-shaped symbols (malloc, free, memcpy, strlen, ...) —
// linking those same names into a *hosted* build's archive collides with
// the real libc there (confirmed in CI: Windows' libucrt raised LNK2005
// "already defined" once this file made it into libsafec_std.a).
// safeguard's Builder.cpp excludes 'std/wasm/' from the stdlib archive
// build specifically because of this file — see the 'kExcludedStdSubdirs'
// comment there. A wasm32 build never links against that archive anyway
// (it's a host-arch .a); it reaches this file through the 'runtime wasm'
// directive above.

// ── Heap: size classes over linear memory ────────────────────────────────────
//
// Every block is a power of two from 16 bytes up (header included), so a
// freed block goes on its class's free list and is handed out again by
// the next allocation of that class — O(1) malloc/free, at most 2x
// internal fragmentation, and no coalescing (a page's allocations are
// dominated by short strings and small vectors). Fresh blocks are carved
// from the end of the heap, which starts at the linker's __heap_base and
// grows the linear memory (memory.grow) on demand; memory is never handed
// back to the host.
//
// Header (8 bytes, keeps payloads 8-aligned): the class (log2 of the block
// size) and a magic word, checked by free()/realloc() so a pointer this
// heap didn't hand out is ignored rather than corrupting a free list.

extern unsigned char __heap_base;   // defined by wasm-ld: first byte past static data

#define WASM_RT_MIN_CLASS 4U          // 16-byte blocks
#define WASM_RT_MAX_CLASS 31U         // 2 GiB blocks
#define WASM_RT_HEADER    8U
#define WASM_RT_MAGIC     0x5AFEC0DEU
#define WASM_RT_PAGE      65536U

static uintptr_t gWasmRtBrk = 0U;       // next fresh byte (0 = not set up yet)
static uintptr_t gWasmRtEnd = 0U;       // end of the linear memory
static uintptr_t gWasmRtFree[32];       // free-list heads, by class (0 = empty)

static int wasm_rt_memory_pages_() {
    int pages;
    unsafe { asm volatile ("memory.size 0\n\tlocal.set %0" : "=r"(pages)); }
    return pages;
}

// Grows the memory by 'pages' 64 KiB pages; the old page count, or -1.
static int wasm_rt_memory_grow_(int pages) {
    int old;
    unsafe { asm volatile ("local.get %1\n\tmemory.grow 0\n\tlocal.set %0" : "=r"(old) : "r"(pages)); }
    return old;
}

static void wasm_rt_init_() {
    unsafe {
        uintptr_t base = (uintptr_t)&__heap_base;
        gWasmRtBrk = (base + 15U) & ~(uintptr_t)15U;
        gWasmRtEnd = (uintptr_t)wasm_rt_memory_pages_() * WASM_RT_PAGE;
        for (unsigned int i = 0U; i < 32U; i = i + 1U) { gWasmRtFree[i] = 0U; }
    }
}

// Smallest class whose block holds 'size' payload bytes, or 0 if too big.
static unsigned int wasm_rt_class_(size_t size) {
    uintptr_t need = (uintptr_t)size + WASM_RT_HEADER;
    if (need < (uintptr_t)size) { return 0U; }   // overflow
    unsigned int c = WASM_RT_MIN_CLASS;
    while (c <= WASM_RT_MAX_CLASS && ((uintptr_t)1U << c) < need) { c = c + 1U; }
    return c > WASM_RT_MAX_CLASS ? 0U : c;
}

// A fresh block of 2^c bytes from the end of the heap; 0 when memory can't grow.
static uintptr_t wasm_rt_carve_(unsigned int c) {
    uintptr_t bytes = (uintptr_t)1U << c;
    // Align big blocks a little (to 16) — the heap start already is.
    if (gWasmRtBrk + bytes < gWasmRtBrk) { return 0U; }
    if (gWasmRtBrk + bytes > gWasmRtEnd) {
        uintptr_t short_by = gWasmRtBrk + bytes - gWasmRtEnd;
        uintptr_t pages = (short_by + WASM_RT_PAGE - 1U) / WASM_RT_PAGE;
        if (pages < 16U) { pages = 16U; }   // grow 1 MiB at a time at least
        if (wasm_rt_memory_grow_((int)pages) < 0) {
            pages = (short_by + WASM_RT_PAGE - 1U) / WASM_RT_PAGE;   // retry with just enough
            if (wasm_rt_memory_grow_((int)pages) < 0) { return 0U; }
        }
        gWasmRtEnd = (uintptr_t)wasm_rt_memory_pages_() * WASM_RT_PAGE;
    }
    uintptr_t b = gWasmRtBrk;
    gWasmRtBrk = gWasmRtBrk + bytes;
    return b;
}

// The block header of payload 'p' if this heap handed it out, else 0.
static uintptr_t wasm_rt_block_(void* p) {
    if (p == (void*)0) { return 0U; }
    unsafe {
        uintptr_t b = (uintptr_t)p - WASM_RT_HEADER;
        if (gWasmRtBrk == 0U || b < (uintptr_t)&__heap_base || b >= gWasmRtBrk) { return 0U; }
        unsigned int* h = (unsigned int*)b;
        if (h[1] != WASM_RT_MAGIC || h[0] < WASM_RT_MIN_CLASS || h[0] > WASM_RT_MAX_CLASS) { return 0U; }
        return b;
    }
}

void* malloc(size_t size) {
    if (gWasmRtBrk == 0U) { wasm_rt_init_(); }
    unsigned int c = wasm_rt_class_(size);
    if (c == 0U) { return (void*)0; }
    uintptr_t b = gWasmRtFree[c];
    unsafe {
        if (b != 0U) {
            gWasmRtFree[c] = *(uintptr_t*)(b + WASM_RT_HEADER);   // next link lives in the payload
        } else {
            b = wasm_rt_carve_(c);
            if (b == 0U) { return (void*)0; }
        }
        unsigned int* h = (unsigned int*)b;
        h[0] = c;
        h[1] = WASM_RT_MAGIC;
        return (void*)(b + WASM_RT_HEADER);
    }
}

void free(void* ptr) {
    uintptr_t b = wasm_rt_block_(ptr);
    if (b == 0U) { return; }
    unsafe {
        unsigned int* h = (unsigned int*)b;
        unsigned int c = h[0];
        h[1] = 0U;   // a second free() of the same block is now ignored
        *(uintptr_t*)(b + WASM_RT_HEADER) = gWasmRtFree[c];
        gWasmRtFree[c] = b;
    }
}

void* calloc(size_t count, size_t size) {
    size_t total = count * size;
    if (size != 0U && total / size != count) { return (void*)0; }
    void* p = malloc(total);
    if (p != (void*)0) {
        unsafe { memset(p, 0, total); }
    }
    return p;
}

// Usable payload bytes of a block from malloc (0 for anything else).
size_t malloc_usable_size(void* ptr) {
    uintptr_t b = wasm_rt_block_(ptr);
    if (b == 0U) { return 0U; }
    unsafe { return (size_t)(((uintptr_t)1U << ((unsigned int*)b)[0]) - WASM_RT_HEADER); }
}

void* realloc(void* ptr, size_t new_size) {
    if (ptr == (void*)0) { return malloc(new_size); }
    size_t have = malloc_usable_size(ptr);
    if (have == 0U) { return (void*)0; }
    if (new_size <= have && new_size > have / 4U) { return ptr; }   // still fits, not wastefully
    void* q = malloc(new_size);
    if (q == (void*)0) { return (void*)0; }
    unsafe { memcpy(q, ptr, have < new_size ? have : new_size); }
    free(ptr);
    return q;
}

// Alignment up to 16 comes for free (payloads are 8-aligned; blocks of
// 32+ bytes start 16-aligned); larger alignments over-allocate.
int posix_memalign(void** memptr, size_t alignment, size_t size) {
    if (alignment <= 8U) {
        void* p = malloc(size);
        unsafe { *memptr = p; }
        return p == (void*)0 ? 12 : 0;   // ENOMEM
    }
    // Over-allocate and leak the slack before the aligned address: the
    // aligned pointer can't be free()d through this heap's header check,
    // so free() ignores it (std uses this for long-lived slabs only).
    void* raw = malloc(size + alignment);
    if (raw == (void*)0) { return 12; }
    unsafe {
        uintptr_t a = ((uintptr_t)raw + alignment - 1U) & ~((uintptr_t)alignment - 1U);
        *memptr = (void*)a;
    }
    return 0;
}

// ── memory ───────────────────────────────────────────────────────────────────

void* memcpy(void* dst, const void* src, size_t n) {
    unsafe {
        unsigned char* d = (unsigned char*)dst;
        const unsigned char* s = (const unsigned char*)src;
        for (size_t i = 0U; i < n; i = i + 1U) { d[i] = s[i]; }
    }
    return dst;
}

void* memmove(void* dst, const void* src, size_t n) {
    unsafe {
        unsigned char* d = (unsigned char*)dst;
        const unsigned char* s = (const unsigned char*)src;
        if (d < s) {
            for (size_t i = 0U; i < n; i = i + 1U) { d[i] = s[i]; }
        } else {
            size_t i = n;
            while (i > 0U) { i = i - 1U; d[i] = s[i]; }
        }
    }
    return dst;
}

void* memset(void* ptr, int val, size_t n) {
    unsafe {
        unsigned char* p = (unsigned char*)ptr;
        unsigned char b = (unsigned char)val;
        for (size_t i = 0U; i < n; i = i + 1U) { p[i] = b; }
    }
    return ptr;
}

int memcmp(const void* a, const void* b, size_t n) {
    unsafe {
        const unsigned char* pa = (const unsigned char*)a;
        const unsigned char* pb = (const unsigned char*)b;
        for (size_t i = 0U; i < n; i = i + 1U) {
            if (pa[i] != pb[i]) { return (int)pa[i] - (int)pb[i]; }
        }
    }
    return 0;
}

void* memchr(const void* s, int c, size_t n) {
    unsafe {
        const unsigned char* p = (const unsigned char*)s;
        for (size_t i = 0U; i < n; i = i + 1U) {
            if (p[i] == (unsigned char)c) { return (void*)(p + i); }
        }
    }
    return (void*)0;
}

// ── strings ──────────────────────────────────────────────────────────────────

size_t strlen(const char* s) {
    size_t n = 0U;
    unsafe { while (s[n] != '\0') { n = n + 1U; } }
    return n;
}

int strcmp(const char* a, const char* b) {
    unsafe {
        size_t i = 0U;
        while (a[i] != '\0' && a[i] == b[i]) { i = i + 1U; }
        return (int)(unsigned char)a[i] - (int)(unsigned char)b[i];
    }
}

int strncmp(const char* a, const char* b, size_t n) {
    unsafe {
        for (size_t i = 0U; i < n; i = i + 1U) {
            if (a[i] != b[i] || a[i] == '\0') { return (int)(unsigned char)a[i] - (int)(unsigned char)b[i]; }
        }
    }
    return 0;
}

char* strchr(const char* s, int c) {
    unsafe {
        size_t i = 0U;
        for (;;) {
            if (s[i] == (char)c) { return (char*)(s + i); }
            if (s[i] == '\0') { return (char*)0; }
            i = i + 1U;
        }
    }
    return (char*)0;
}

char* strrchr(const char* s, int c) {
    char* last = (char*)0;
    unsafe {
        size_t i = 0U;
        for (;;) {
            if (s[i] == (char)c) { last = (char*)(s + i); }
            if (s[i] == '\0') { break; }
            i = i + 1U;
        }
    }
    return last;
}

char* strstr(const char* h, const char* n) {
    unsafe {
        if (n[0] == '\0') { return (char*)h; }
        for (size_t i = 0U; h[i] != '\0'; i = i + 1U) {
            size_t k = 0U;
            while (n[k] != '\0' && h[i + k] == n[k]) { k = k + 1U; }
            if (n[k] == '\0') { return (char*)(h + i); }
        }
    }
    return (char*)0;
}

static int wasm_rt_in_set_(char c, const char* set) {
    unsafe {
        for (size_t i = 0U; set[i] != '\0'; i = i + 1U) { if (set[i] == c) { return 1; } }
    }
    return 0;
}

size_t strspn(const char* s, const char* accept) {
    size_t n = 0U;
    unsafe { while (s[n] != '\0' && wasm_rt_in_set_(s[n], accept)) { n = n + 1U; } }
    return n;
}

size_t strcspn(const char* s, const char* reject) {
    size_t n = 0U;
    unsafe { while (s[n] != '\0' && !wasm_rt_in_set_(s[n], reject)) { n = n + 1U; } }
    return n;
}

char* strpbrk(const char* s, const char* accept) {
    unsafe {
        size_t n = strcspn(s, accept);
        return s[n] == '\0' ? (char*)0 : (char*)(s + n);
    }
}

char* strncpy(char* dst, const char* src, size_t n) {
    unsafe {
        size_t i = 0U;
        while (i < n && src[i] != '\0') { dst[i] = src[i]; i = i + 1U; }
        while (i < n) { dst[i] = '\0'; i = i + 1U; }
    }
    return dst;
}

char* strcat(char* dst, const char* src) {
    unsafe {
        size_t d = strlen(dst);
        size_t i = 0U;
        while (src[i] != '\0') { dst[d + i] = src[i]; i = i + 1U; }
        dst[d + i] = '\0';
    }
    return dst;
}

char* strncat(char* dst, const char* src, size_t n) {
    unsafe {
        size_t d = strlen(dst);
        size_t i = 0U;
        while (i < n && src[i] != '\0') { dst[d + i] = src[i]; i = i + 1U; }
        dst[d + i] = '\0';
    }
    return dst;
}

char* strtok_r(char* s, const char* delim, char** saveptr) {
    unsafe {
        char* p = s != (char*)0 ? s : *saveptr;
        if (p == (char*)0) { return (char*)0; }
        p = p + strspn(p, delim);
        if (*p == '\0') { *saveptr = p; return (char*)0; }
        char* end = p + strcspn(p, delim);
        if (*end != '\0') { *end = '\0'; *saveptr = end + 1; }
        else { *saveptr = end; }
        return p;
    }
}

// ── number parsing ───────────────────────────────────────────────────────────

static int wasm_rt_digit_(char c, int base) {
    int d = 99;
    if (c >= '0' && c <= '9') { d = (int)(c - '0'); }
    else if (c >= 'a' && c <= 'z') { d = (int)(c - 'a') + 10; }
    else if (c >= 'A' && c <= 'Z') { d = (int)(c - 'A') + 10; }
    return d < base ? d : -1;
}

unsigned long long strtoull(const char* s, char** end, int base) {
    unsafe {
        size_t i = 0U;
        while (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r') { i = i + 1U; }
        int neg = 0;
        if (s[i] == '+' || s[i] == '-') { neg = s[i] == '-'; i = i + 1U; }
        if ((base == 0 || base == 16) && s[i] == '0' && (s[i + 1U] == 'x' || s[i + 1U] == 'X') &&
            wasm_rt_digit_(s[i + 2U], 16) >= 0) { base = 16; i = i + 2U; }
        else if (base == 0 && s[i] == '0') { base = 8; }
        else if (base == 0) { base = 10; }
        size_t start = i;
        unsigned long long v = 0ULL;
        for (;;) {
            int d = wasm_rt_digit_(s[i], base);
            if (d < 0) { break; }
            v = v * (unsigned long long)base + (unsigned long long)d;
            i = i + 1U;
        }
        if (end != (char**)0) { *end = (char*)(i == start ? s : s + i); }
        return neg ? (unsigned long long)0 - v : v;
    }
}

long long strtoll(const char* s, char** end, int base) {
    return (long long)strtoull(s, end, base);
}

int atoi(const char* s) { return (int)strtoll(s, (char**)0, 10); }
long long atoll(const char* s) { return strtoll(s, (char**)0, 10); }

// Decimal floating point ('-1.5e3', 'inf', 'nan' not supported): exact for
// the integer part, then scaled; good to ~15 significant digits.
double strtod(const char* s, char** end) {
    unsafe {
        size_t i = 0U;
        while (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r') { i = i + 1U; }
        double sign = 1.0;
        if (s[i] == '+' || s[i] == '-') { if (s[i] == '-') { sign = -1.0; } i = i + 1U; }
        size_t start = i;
        double v = 0.0;
        while (s[i] >= '0' && s[i] <= '9') { v = v * 10.0 + (double)(s[i] - '0'); i = i + 1U; }
        if (s[i] == '.') {
            i = i + 1U;
            double scale = 0.1;
            while (s[i] >= '0' && s[i] <= '9') { v = v + (double)(s[i] - '0') * scale; scale = scale / 10.0; i = i + 1U; }
        }
        if (i == start || (i == start + 1U && s[start] == '.')) {
            if (end != (char**)0) { *end = (char*)s; }
            return 0.0;
        }
        if (s[i] == 'e' || s[i] == 'E') {
            size_t e = i + 1U;
            int eneg = 0;
            if (s[e] == '+' || s[e] == '-') { eneg = s[e] == '-'; e = e + 1U; }
            if (s[e] >= '0' && s[e] <= '9') {
                int ex = 0;
                while (s[e] >= '0' && s[e] <= '9') { if (ex < 400) { ex = ex * 10 + (int)(s[e] - '0'); } e = e + 1U; }
                double f = 1.0;
                for (int k = 0; k < ex; k = k + 1) { f = f * 10.0; }
                v = eneg ? v / f : v * f;
                i = e;
            }
        }
        if (end != (char**)0) { *end = (char*)(s + i); }
        return sign * v;
    }
}

// ── sorting ──────────────────────────────────────────────────────────────────

static void wasm_rt_swap_(unsigned char* a, unsigned char* b, size_t n) {
    unsafe {
        for (size_t i = 0U; i < n; i = i + 1U) { unsigned char t = a[i]; a[i] = b[i]; b[i] = t; }
    }
}

typedef fn int(const void*, const void*) WasmRtCmp_;

// Heapsort: in place, no recursion, O(n log n).
static void wasm_rt_sift_(unsigned char* base, size_t root, size_t n, size_t sz, WasmRtCmp_ cmp) {
    unsafe {
        for (;;) {
            size_t child = 2U * root + 1U;
            if (child >= n) { return; }
            if (child + 1U < n && cmp((const void*)(base + child * sz), (const void*)(base + (child + 1U) * sz)) < 0) {
                child = child + 1U;
            }
            if (cmp((const void*)(base + root * sz), (const void*)(base + child * sz)) >= 0) { return; }
            wasm_rt_swap_(base + root * sz, base + child * sz, sz);
            root = child;
        }
    }
}

void qsort(void* base, size_t n, size_t sz, void* cmp_fn) {
    if (n < 2U) { return; }
    unsafe {
        unsigned char* b = (unsigned char*)base;
        WasmRtCmp_ cmp = (WasmRtCmp_)cmp_fn;
        size_t i = n / 2U;
        while (i > 0U) { i = i - 1U; wasm_rt_sift_(b, i, n, sz, cmp); }
        size_t end = n;
        while (end > 1U) {
            end = end - 1U;
            wasm_rt_swap_(b, b + end * sz, sz);
            wasm_rt_sift_(b, 0U, end, sz, cmp);
        }
    }
}

// ── compiler-rt builtins ─────────────────────────────────────────────────────
// 128-bit helpers LLVM emits calls to on wasm32 (no native i128): __multi3
// for i64 multiplies with overflow checks (std's checked_mul_size), and the
// 128-bit shifts that expansions use. A 128-bit value travels as (lo, hi)
// i64 halves; the result is written through the hidden sret pointer, which
// is exactly how these signatures lower on wasm32.

void __multi3(unsigned long long* ret, unsigned long long alo, unsigned long long ahi,
              unsigned long long blo, unsigned long long bhi) {
    unsigned long long mask = 0xFFFFFFFFULL;
    unsigned long long a0 = alo & mask;
    unsigned long long a1 = alo >> 32;
    unsigned long long b0 = blo & mask;
    unsigned long long b1 = blo >> 32;
    unsigned long long p00 = a0 * b0;
    unsigned long long p01 = a0 * b1;
    unsigned long long p10 = a1 * b0;
    unsigned long long p11 = a1 * b1;
    unsigned long long mid = (p00 >> 32) + (p01 & mask) + (p10 & mask);
    unsigned long long lo = (p00 & mask) | (mid << 32);
    unsigned long long hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32) + alo * bhi + ahi * blo;
    unsafe { ret[0] = lo; ret[1] = hi; }
}

void __ashlti3(unsigned long long* ret, unsigned long long lo, unsigned long long hi, int b) {
    unsigned long long rl = lo;
    unsigned long long rh = hi;
    if (b >= 128) { rl = 0ULL; rh = 0ULL; }
    else if (b >= 64) { rh = lo << (unsigned long long)(b - 64); rl = 0ULL; }
    else if (b > 0) { rh = (hi << (unsigned long long)b) | (lo >> (unsigned long long)(64 - b)); rl = lo << (unsigned long long)b; }
    unsafe { ret[0] = rl; ret[1] = rh; }
}

void __lshrti3(unsigned long long* ret, unsigned long long lo, unsigned long long hi, int b) {
    unsigned long long rl = lo;
    unsigned long long rh = hi;
    if (b >= 128) { rl = 0ULL; rh = 0ULL; }
    else if (b >= 64) { rl = hi >> (unsigned long long)(b - 64); rh = 0ULL; }
    else if (b > 0) { rl = (lo >> (unsigned long long)b) | (hi << (unsigned long long)(64 - b)); rh = hi >> (unsigned long long)b; }
    unsafe { ret[0] = rl; ret[1] = rh; }
}

void __ashrti3(unsigned long long* ret, unsigned long long lo, unsigned long long hi, int b) {
    long long shi = (long long)hi;
    unsigned long long fill = shi < 0LL ? 0xFFFFFFFFFFFFFFFFULL : 0ULL;
    unsigned long long rl = lo;
    unsigned long long rh = hi;
    if (b >= 128) { rl = fill; rh = fill; }
    else if (b >= 64) { rl = (unsigned long long)(shi >> (long long)(b - 64)); rh = fill; }
    else if (b > 0) { rl = (lo >> (unsigned long long)b) | (hi << (unsigned long long)(64 - b)); rh = (unsigned long long)(shi >> (long long)b); }
    unsafe { ret[0] = rl; ret[1] = rh; }
}

// ── the rest ─────────────────────────────────────────────────────────────────

void abort() {
    unsafe { asm volatile ("unreachable"); }
}

int fprintf(void* stream, const char* fmt, ...) { return 0; }
int printf(const char* fmt, ...) { return 0; }
