// The scc runtime, compiled into every program (as the pseudo-file
// "<runtime>"). Host services come from three wasm imports; everything else
// (allocator, string and character functions, printf-style formatting with
// exact float conversion, number parsing, math) is SafeC, compiled by scc
// itself. Internal helpers are named __rt_*.
extern int __host_write(int fd, const char* buf, int len);
extern int __host_read(int fd, char* buf, int len);
extern void __host_exit(int code);

// a failed runtime check (kind: 1 index out of bounds, 2 arena capacity
// exceeded, 3 stack overflow): reported on stderr, then the exit code a
// native safec program gets from the same failure (SIGABRT 134, SIGSEGV 139)
void __rt_trap(int kind) {
    if (kind == 1) { __rt_errs("runtime error: index out of bounds\n"); __host_exit(134); }
    if (kind == 2) { __rt_errs("runtime error: arena capacity exceeded\n"); __host_exit(134); }
    if (kind == 3) { __rt_errs("runtime error: stack overflow\n"); __host_exit(139); }
    __host_exit(134);
}

void __rt_assert_fail(const char* expr, const char* file, int line) {
    __rt_errs("Assertion failed: ");
    __rt_errs(expr);
    __rt_errs("\n");
    __host_exit(134);
}

// atexit handlers, run by exit() (a return from main exits through the host)
void* __rt_atexit_fn[32];
int __rt_natexit = 0;
int atexit(void* fnp) {
    if (__rt_natexit >= 32) { return -1; }
    __rt_atexit_fn[__rt_natexit] = fnp;
    __rt_natexit = __rt_natexit + 1;
    return 0;
}
void exit(int code) {
    while (__rt_natexit > 0) {
        __rt_natexit = __rt_natexit - 1;
        unsafe {
            fn void() h = (fn void())__rt_atexit_fn[__rt_natexit];
            h();
        }
    }
    __rt_oflush();
    __host_exit(code);
}

// command-line arguments: the host writes argv (pointer table + strings) here
char __rt_argmem[16384];
__attribute__((export_name("__rt_argv")))
int __rt_argv() {
    unsafe { return (int)&__rt_argmem[0]; }
}
void abort() { __host_exit(134); }
void _Exit(int code) { __host_exit(code); }

// ── memory allocator ─────────────────────────────────────────────────────────
// First-fit free list over a heap that grows with memory.grow. Every block
// has a 16-byte header: size (8 bytes) and a live/free tag (8 bytes).
unsigned long __rt_heap_top = 0UL;
unsigned long __rt_heap_end = 0UL;
unsigned long __rt_free_list = 0UL;

unsigned long __rt_live_tag = 6102007314592371761UL;
unsigned long __rt_dead_tag = 4848522148045017445UL;

void* malloc(unsigned long size) {
    if (__rt_heap_top == 0UL) {
        __rt_heap_top = (__builtin_heap_base() + 15UL) & ~15UL;
        __rt_heap_end = (unsigned long)__builtin_memory_size() * 65536UL;
    }
    unsigned long need = (size + 15UL) & ~15UL;
    if (need == 0UL) { need = 16UL; }
    unsafe {
        // free list: first fit
        unsigned long prev = 0UL;
        unsigned long cur = __rt_free_list;
        while (cur != 0UL) {
            unsigned long* h = (unsigned long*)(int)cur;
            unsigned long bsize = h[0];
            unsigned long next = h[1];
            if (bsize >= need) {
                if (prev == 0UL) { __rt_free_list = next; }
                else { unsigned long* ph = (unsigned long*)(int)prev; ph[1] = next; }
                if (bsize >= need + 64UL) {
                    // split: the tail goes back on the free list
                    unsigned long rest = cur + 16UL + need;
                    unsigned long* rh = (unsigned long*)(int)rest;
                    rh[0] = bsize - need - 16UL;
                    rh[1] = __rt_free_list;
                    __rt_free_list = rest;
                    h[0] = need;
                }
                h[1] = __rt_live_tag;
                return (void*)(int)(cur + 16UL);
            }
            prev = cur;
            cur = next;
        }
        unsigned long block = __rt_heap_top;
        unsigned long end = block + 16UL + need;
        if (end > __rt_heap_end) {
            unsigned long pages = (end - __rt_heap_end + 65535UL) / 65536UL;
            if (pages < 16UL) { pages = 16UL; }
            if (__builtin_memory_grow((int)pages) < 0) { return null; }
            __rt_heap_end = __rt_heap_end + pages * 65536UL;
        }
        __rt_heap_top = end;
        unsigned long* h = (unsigned long*)(int)block;
        h[0] = need;
        h[1] = __rt_live_tag;
        return (void*)(int)(block + 16UL);
    }
}

void free(void* p) {
    if (p == null) { return; }
    unsafe {
        unsigned long a = (unsigned long)(int)p;
        unsigned long* h = (unsigned long*)(int)(a - 16UL);
        if (h[1] != __rt_live_tag) {
            __rt_errs("free(): invalid pointer or double free\n");
            __host_exit(134);
        }
        h[1] = __rt_free_list;
        __rt_free_list = a - 16UL;
    }
}

void* calloc(unsigned long n, unsigned long size) {
    void* p = malloc(n * size);
    if (p != null) { memset(p, 0, n * size); }
    return p;
}

void* realloc(void* p, unsigned long size) {
    if (p == null) { return malloc(size); }
    void* q = malloc(size);
    if (q == null) { return null; }
    unsafe {
        unsigned long a = (unsigned long)(int)p;
        unsigned long* h = (unsigned long*)(int)(a - 16UL);
        unsigned long old = h[0];
        if (old > size) { old = size; }
        memcpy(q, (const void*)p, old);
    }
    free(p);
    return q;
}

// 'new T' on the heap
void* __rt_new(unsigned long size) {
    void* p = malloc(size);
    if (p == null) {
        __rt_errs("out of memory\n");
        __host_exit(134);
    }
    memset(p, 0, size);
    return p;
}

// ── memory and strings ───────────────────────────────────────────────────────
void* memcpy(void* dst, const void* src, unsigned long n) {
    unsafe {
        unsigned char* d = (unsigned char*)dst;
        const unsigned char* s = (const unsigned char*)src;
        unsigned long i = 0UL;
        while (i < n) { d[i] = s[i]; i = i + 1UL; }
    }
    return dst;
}

void* memmove(void* dst, const void* src, unsigned long n) {
    unsafe {
        unsigned char* d = (unsigned char*)dst;
        const unsigned char* s = (const unsigned char*)src;
        if ((unsigned long)(int)dst < (unsigned long)(int)(void*)src) {
            unsigned long i = 0UL;
            while (i < n) { d[i] = s[i]; i = i + 1UL; }
        } else {
            unsigned long i = n;
            while (i > 0UL) { i = i - 1UL; d[i] = s[i]; }
        }
    }
    return dst;
}

void* memset(void* p, int v, unsigned long n) {
    unsafe {
        unsigned char* d = (unsigned char*)p;
        unsigned long i = 0UL;
        while (i < n) { d[i] = (unsigned char)v; i = i + 1UL; }
    }
    return p;
}

int memcmp(const void* a, const void* b, unsigned long n) {
    unsafe {
        const unsigned char* x = (const unsigned char*)a;
        const unsigned char* y = (const unsigned char*)b;
        unsigned long i = 0UL;
        while (i < n) {
            if (x[i] != y[i]) { return (int)x[i] - (int)y[i]; }
            i = i + 1UL;
        }
    }
    return 0;
}

unsigned long strlen(const char* s) {
    unsigned long n = 0UL;
    unsafe { while (s[n] != (char)0) { n = n + 1UL; } }
    return n;
}

int strcmp(const char* a, const char* b) {
    unsafe {
        int i = 0;
        while (a[i] != (char)0 && a[i] == b[i]) { i = i + 1; }
        return (int)(unsigned char)a[i] - (int)(unsigned char)b[i];
    }
}

int strncmp(const char* a, const char* b, unsigned long n) {
    unsafe {
        unsigned long i = 0UL;
        while (i < n) {
            if (a[i] != b[i] || a[i] == (char)0) { return (int)(unsigned char)a[i] - (int)(unsigned char)b[i]; }
            i = i + 1UL;
        }
    }
    return 0;
}

char* strcpy(char* d, const char* s) {
    unsafe {
        int i = 0;
        while (s[i] != (char)0) { d[i] = s[i]; i = i + 1; }
        d[i] = (char)0;
    }
    return d;
}

char* strncpy(char* d, const char* s, unsigned long n) {
    unsafe {
        unsigned long i = 0UL;
        while (i < n && s[i] != (char)0) { d[i] = s[i]; i = i + 1UL; }
        while (i < n) { d[i] = (char)0; i = i + 1UL; }
    }
    return d;
}

char* strcat(char* d, const char* s) {
    unsafe { strcpy(d + strlen(d), s); }
    return d;
}

char* strncat(char* d, const char* s, unsigned long n) {
    unsafe {
        unsigned long k = strlen(d);
        unsigned long i = 0UL;
        while (i < n && s[i] != (char)0) { d[k + i] = s[i]; i = i + 1UL; }
        d[k + i] = (char)0;
    }
    return d;
}

char* strchr(const char* s, int c) {
    unsafe {
        int i = 0;
        while (1) {
            if ((int)(unsigned char)s[i] == (c & 255)) { return (char*)(s + i); }
            if (s[i] == (char)0) { return null; }
            i = i + 1;
        }
    }
    return null;
}

char* strrchr(const char* s, int c) {
    char* r = null;
    unsafe {
        int i = 0;
        while (1) {
            if ((int)(unsigned char)s[i] == (c & 255)) { r = (char*)(s + i); }
            if (s[i] == (char)0) { return r; }
            i = i + 1;
        }
    }
    return r;
}

char* strstr(const char* h, const char* n) {
    unsafe {
        unsigned long ln = strlen(n);
        if (ln == 0UL) { return (char*)h; }
        int i = 0;
        while (h[i] != (char)0) {
            if (strncmp(h + i, n, ln) == 0) { return (char*)(h + i); }
            i = i + 1;
        }
    }
    return null;
}

char* strdup(const char* s) {
    unsigned long n = strlen(s);
    unsafe {
        char* d = (char*)malloc(n + 1UL);
        if (d != null) { memcpy((void*)d, (const void*)s, n + 1UL); }
        return d;
    }
}

// ── ctype ────────────────────────────────────────────────────────────────────
void* memchr(const void* s, int c, unsigned long n) {
    unsafe {
        const unsigned char* p = (const unsigned char*)s;
        unsigned long i = 0UL;
        while (i < n) { if ((int)p[i] == (c & 255)) { return (void*)(p + i); } i = i + 1UL; }
    }
    return null;
}
int __rt_inset(int c, const char* set) {
    unsafe {
        int i = 0;
        while (set[i] != (char)0) { if ((int)set[i] == c) { return 1; } i = i + 1; }
    }
    return 0;
}
unsigned long strspn(const char* s, const char* accept) {
    unsigned long n = 0UL;
    unsafe { while (s[n] != (char)0 && __rt_inset((int)s[n], accept) == 1) { n = n + 1UL; } }
    return n;
}
unsigned long strcspn(const char* s, const char* reject) {
    unsigned long n = 0UL;
    unsafe { while (s[n] != (char)0 && __rt_inset((int)s[n], reject) == 0) { n = n + 1UL; } }
    return n;
}
char* strpbrk(const char* s, const char* accept) {
    unsafe {
        int i = 0;
        while (s[i] != (char)0) { if (__rt_inset((int)s[i], accept) == 1) { return (char*)(s + i); } i = i + 1; }
    }
    return null;
}
char* strtok_r(char* s, const char* delim, char** save) {
    unsafe {
        char* p = s;
        if (p == null) { p = *save; }
        if (p == null) { return null; }
        while (*p != (char)0 && __rt_inset((int)*p, delim) == 1) { p = p + 1; }
        if (*p == (char)0) { *save = p; return null; }
        char* start = p;
        while (*p != (char)0 && __rt_inset((int)*p, delim) == 0) { p = p + 1; }
        if (*p != (char)0) { *p = (char)0; p = p + 1; }
        *save = p;
        return start;
    }
}
char* __rt_strtok_save = null;
char* strtok(char* s, const char* delim) {
    unsafe { return strtok_r(s, delim, &__rt_strtok_save); }
}

int isdigit(int c) { return (int)(c >= 48 && c <= 57); }
int isalpha(int c) { return (int)((c >= 65 && c <= 90) || (c >= 97 && c <= 122)); }
int isalnum(int c) { return (int)(isdigit(c) != 0 || isalpha(c) != 0); }
int isspace(int c) { return (int)(c == 32 || (c >= 9 && c <= 13)); }
int isupper(int c) { return (int)(c >= 65 && c <= 90); }
int islower(int c) { return (int)(c >= 97 && c <= 122); }
int isxdigit(int c) { return (int)(isdigit(c) != 0 || (c >= 65 && c <= 70) || (c >= 97 && c <= 102)); }
int ispunct(int c) { return (int)(c > 32 && c < 127 && isalnum(c) == 0); }
int isprint(int c) { return (int)(c >= 32 && c < 127); }
int iscntrl(int c) { return (int)(c < 32 || c == 127); }
int toupper(int c) { if (c >= 97 && c <= 122) { return c - 32; } return c; }
int tolower(int c) { if (c >= 65 && c <= 90) { return c + 32; } return c; }

// ── integers ─────────────────────────────────────────────────────────────────
int abs(int x) { if (x < 0) { return 0 - x; } return x; }
long labs(long x) { if (x < 0L) { return 0L - x; } return x; }
long llabs(long x) { if (x < 0L) { return 0L - x; } return x; }

long strtol(const char* s, char** end, int base) {
    long v = 0L;
    int neg = 0;
    unsafe {
        int i = 0;
        while (isspace((int)s[i]) != 0) { i = i + 1; }
        if (s[i] == '-') { neg = 1; i = i + 1; } else if (s[i] == '+') { i = i + 1; }
        if ((base == 0 || base == 16) && s[i] == '0' && (s[i + 1] == 'x' || s[i + 1] == 'X')) { base = 16; i = i + 2; }
        else if (base == 0 && s[i] == '0') { base = 8; }
        else if (base == 0) { base = 10; }
        while (1) {
            int c = (int)s[i];
            int d = 99;
            if (c >= 48 && c <= 57) { d = c - 48; }
            else if (c >= 97 && c <= 122) { d = c - 87; }
            else if (c >= 65 && c <= 90) { d = c - 55; }
            if (d >= base) { break; }
            v = v * (long)base + (long)d;
            i = i + 1;
        }
        if (end != null) { *end = (char*)(s + i); }
    }
    if (neg == 1) { return 0L - v; }
    return v;
}

unsigned long strtoul(const char* s, char** end, int base) { return (unsigned long)strtol(s, end, base); }
unsigned long long strtoull(const char* s, char** end, int base) { return (unsigned long long)strtoul(s, end, base); }
long long strtoll(const char* s, char** end, int base) { return strtol(s, end, base); }
int atoi(const char* s) { return (int)strtol(s, null, 10); }
long atol(const char* s) { return strtol(s, null, 10); }
long long atoll(const char* s) { return strtol(s, null, 10); }

// rand(): glibc's generator (TYPE_3, additive feedback r[i] = r[i-3] +
// r[i-31]), so a seeded sequence matches what a native Linux build prints
int __rt_rtab[34];
int __rt_rpos = 0;
int __rt_rready = 0;
void srand(unsigned int seed) {
    long w = (long)seed;
    if (w == 0L) { w = 1L; }
    long r[34];
    r[0] = w;
    int i = 1;
    while (i < 31) {
        long hi = r[i - 1] / 127773L;
        long lo = r[i - 1] % 127773L;
        long v = 16807L * lo - 2836L * hi;
        if (v < 0L) { v = v + 2147483647L; }
        r[i] = v;
        i = i + 1;
    }
    while (i < 34) { r[i] = r[i - 31]; i = i + 1; }
    i = 0;
    while (i < 34) { __rt_rtab[i] = (int)(r[i] & 4294967295L); i = i + 1; }
    __rt_rpos = 0;
    __rt_rready = 1;
    // the first 310 outputs are discarded
    i = 34;
    while (i < 344) { __rt_rnext(); i = i + 1; }
}
unsigned int __rt_rnext() {
    // r[k] = r[k-31] + r[k-3] over a ring of 34
    int k = __rt_rpos;
    unsigned int a = (unsigned int)__rt_rtab[(k + 34 - 31) % 34];
    unsigned int b = (unsigned int)__rt_rtab[(k + 34 - 3) % 34];
    unsigned int v = a + b;
    __rt_rtab[k] = (int)v;
    __rt_rpos = (k + 1) % 34;
    return v;
}
int rand() {
    if (__rt_rready == 0) { srand(1U); }
    return (int)(__rt_rnext() >> 1U);
}

// saturating 64-bit arithmetic (the +% -% *% operators on long)
long __rt_satadd_s(long a, long b) {
    long r = a +| b;
    if (a > 0L && b > 0L && r < 0L) { return 9223372036854775807L; }
    if (a < 0L && b < 0L && r >= 0L) { return -9223372036854775807L - 1L; }
    return r;
}
long __rt_satsub_s(long a, long b) {
    long r = a -| b;
    if (a >= 0L && b < 0L && r < 0L) { return 9223372036854775807L; }
    if (a < 0L && b > 0L && r >= 0L) { return -9223372036854775807L - 1L; }
    return r;
}
long __rt_satmul_s(long a, long b) {
    if (a == 0L || b == 0L) { return 0L; }
    long r = a *| b;
    if (r / b != a || (a == -1L && b == -9223372036854775807L - 1L) || (b == -1L && a == -9223372036854775807L - 1L)) {
        if ((a < 0L) == (b < 0L)) { return 9223372036854775807L; }
        return -9223372036854775807L - 1L;
    }
    return r;
}
unsigned long __rt_satadd_u(unsigned long a, unsigned long b) {
    unsigned long r = a + b;
    if (r < a) { return 18446744073709551615UL; }
    return r;
}
unsigned long __rt_satsub_u(unsigned long a, unsigned long b) {
    if (b > a) { return 0UL; }
    return a - b;
}
unsigned long __rt_satmul_u(unsigned long a, unsigned long b) {
    if (a == 0UL || b == 0UL) { return 0UL; }
    unsigned long r = a * b;
    if (r / b != a) { return 18446744073709551615UL; }
    return r;
}

// ── output ───────────────────────────────────────────────────────────────────
// A format sink: stdout (fd 1), stderr (fd 2), or a caller's buffer.
char __rt_obuf[1024];
int __rt_olen = 0;
int __rt_ofd = 1;
char* __rt_sbuf = null;
unsigned long __rt_scap = 0UL;
unsigned long __rt_spos = 0UL;
int __rt_count = 0;

void __rt_oflush() {
    if (__rt_olen > 0) {
        unsafe { __host_write(__rt_ofd, (const char*)&__rt_obuf[0], __rt_olen); }
        __rt_olen = 0;
    }
}

void __rt_put(int c) {
    __rt_count = __rt_count + 1;
    if (__rt_sbuf != null) {
        unsafe {
            if (__rt_spos + 1UL < __rt_scap) { __rt_sbuf[__rt_spos] = (char)c; }
        }
        __rt_spos = __rt_spos + 1UL;
        return;
    }
    if (__rt_olen >= 1024) { __rt_oflush(); }
    __rt_obuf[__rt_olen] = (char)c;
    __rt_olen = __rt_olen + 1;
}

void __rt_pad(int c, int n) {
    while (n > 0) { __rt_put(c); n = n - 1; }
}

void __rt_errs(const char* s) {
    unsafe { __host_write(2, s, (int)strlen(s)); }
}

int putchar(int c) {
    char b[1];
    b[0] = (char)c;
    unsafe { __host_write(1, (const char*)&b[0], 1); }
    return c;
}

int puts(const char* s) {
    unsafe { __host_write(1, s, (int)strlen(s)); }
    putchar(10);
    return 0;
}

// ── exact decimal conversion of doubles ──────────────────────────────────────
// digits of |x|: __rt_dg[0..__rt_dn), the decimal point after __rt_dpt digits
char __rt_dg[1300];
int __rt_dn = 0;
int __rt_dpt = 0;
int __rt_dmore = 0;        // nonzero remainder beyond the generated digits
unsigned int __rt_bf[40];  // fraction numerator, base 2^32, little endian
int __rt_bfn = 0;
int __rt_bk = 0;           // fraction denominator is 2^__rt_bk
unsigned int __rt_bi[40];  // integer part
int __rt_bin = 0;

// next fraction digit: F *= 10; d = F >> k; F &= 2^k - 1
int __rt_fdigit() {
    unsigned long carry = 0UL;
    int i = 0;
    while (i < __rt_bfn) {
        unsigned long v = (unsigned long)__rt_bf[i] * 10UL + carry;
        __rt_bf[i] = (unsigned int)(v & 4294967295UL);
        carry = v >> 32UL;
        i = i + 1;
    }
    // the digit lives in bits k.. (k is within or just past the limbs)
    int limb = __rt_bk / 32;
    int sh = __rt_bk % 32;
    unsigned long hi = carry;
    unsigned long d = 0UL;
    if (limb < __rt_bfn) {
        d = (unsigned long)__rt_bf[limb] >> (unsigned long)sh;
        if (limb + 1 < __rt_bfn) { d = d | ((unsigned long)__rt_bf[limb + 1] << (unsigned long)(32 - sh)); }
        else { d = d | (hi << (unsigned long)(32 - sh)); }
        // clear bits >= k
        if (sh == 0) { __rt_bf[limb] = 0U; }
        else { __rt_bf[limb] = __rt_bf[limb] & ((1U << (unsigned int)sh) - 1U); }
        int j = limb + 1;
        while (j < __rt_bfn) { __rt_bf[j] = 0U; j = j + 1; }
    } else {
        d = hi;
    }
    return (int)(d & 15UL);
}

int __rt_fzero() {
    int i = 0;
    while (i < __rt_bfn) { if (__rt_bf[i] != 0U) { return 0; } i = i + 1; }
    return 1;
}

// x = m * 2^e (x ≥ 0, finite); generates the integer digits and then
// fraction digits until 'nfrac' of them, or 'nsig' significant digits
// (whichever rule is selected by nsig > 0), plus one more for rounding.
void __rt_dtoa(double x, int nfrac, int nsig) {
    long bits = __builtin_f64_bits(x);
    long ex = (bits >> 52L) & 2047L;
    long m = bits & 4503599627370495L;
    long e = 0L;
    if (ex == 0L) { e = -1074L; } else { m = m | 4503599627370496L; e = ex - 1075L; }
    __rt_bin = 0;
    __rt_bfn = 0;
    __rt_bk = 0;
    // integer part
    if (e >= 0L) {
        // m << e as limbs
        int sh = (int)e;
        int nl = sh / 32 + 3;
        int i = 0;
        while (i < nl) { __rt_bi[i] = 0U; i = i + 1; }
        int lb = sh / 32;
        int bs = sh % 32;
        unsigned long mm = (unsigned long)m;
        __rt_bi[lb] = (unsigned int)((mm << (unsigned long)bs) & 4294967295UL);
        __rt_bi[lb + 1] = (unsigned int)(((mm << (unsigned long)bs) >> 32UL) & 4294967295UL);
        if (bs > 0) { __rt_bi[lb + 2] = (unsigned int)(mm >> (unsigned long)(64 - bs)); }
        __rt_bin = nl;
    } else {
        int k = (int)(0L - e);
        unsigned long ip = 0UL;
        if (k < 64) { ip = (unsigned long)m >> (unsigned long)k; }
        __rt_bi[0] = (unsigned int)(ip & 4294967295UL);
        __rt_bi[1] = (unsigned int)(ip >> 32UL);
        __rt_bin = 2;
        // fraction m & (2^k - 1) over 2^k
        __rt_bk = k;
        __rt_bfn = k / 32 + 2;
        int i = 0;
        while (i < __rt_bfn) { __rt_bf[i] = 0U; i = i + 1; }
        unsigned long fr = (unsigned long)m;
        if (k < 64) { fr = fr & ((1UL << (unsigned long)k) - 1UL); }
        __rt_bf[0] = (unsigned int)(fr & 4294967295UL);
        __rt_bf[1] = (unsigned int)(fr >> 32UL);
    }
    // integer digits: repeated division by 10^9
    char tmp[320];
    int tn = 0;
    while (1) {
        int nz = 0;
        unsigned long rem = 0UL;
        int i = __rt_bin - 1;
        while (i >= 0) {
            unsigned long cur = (rem << 32UL) | (unsigned long)__rt_bi[i];
            __rt_bi[i] = (unsigned int)(cur / 1000000000UL);
            rem = cur % 1000000000UL;
            if (__rt_bi[i] != 0U) { nz = 1; }
            i = i - 1;
        }
        int j = 0;
        while (j < 9 && tn < 320) {
            tmp[tn] = (char)(48UL + rem % 10UL);
            rem = rem / 10UL;
            tn = tn + 1;
            j = j + 1;
        }
        if (nz == 0) { break; }
    }
    while (tn > 1 && tmp[tn - 1] == '0') { tn = tn - 1; }
    __rt_dn = 0;
    if (!(tn == 1 && tmp[0] == '0')) {
        while (tn > 0) { tn = tn - 1; __rt_dg[__rt_dn] = tmp[tn]; __rt_dn = __rt_dn + 1; }
    }
    __rt_dpt = __rt_dn;
    // fraction digits
    int sig = __rt_dn;
    int nf = 0;
    while (__rt_dn < 1290) {
        if (nsig > 0) { if (sig > nsig) { break; } }
        else if (nf > nfrac) { break; }
        if (__rt_bfn == 0 || __rt_fzero() == 1) {
            if (nsig > 0 && sig == 0) { break; }   // the value is zero
            __rt_dg[__rt_dn] = '0';
        } else {
            __rt_dg[__rt_dn] = (char)(48 + __rt_fdigit());
        }
        if (sig > 0 || __rt_dg[__rt_dn] != '0') { sig = sig + 1; }
        __rt_dn = __rt_dn + 1;
        nf = nf + 1;
    }
    __rt_dmore = 0;
    if (__rt_bfn > 0 && __rt_fzero() == 0) { __rt_dmore = 1; }
}

// rounds the digit string to its first n digits (round half to even);
// returns 1 if a carry added a new leading digit
int __rt_round(int n) {
    if (n < 0) { n = 0; }
    if (n >= __rt_dn) { return 0; }
    int d = (int)__rt_dg[n] - 48;
    int more = __rt_dmore;
    int i = n + 1;
    while (i < __rt_dn && more == 0) { if (__rt_dg[i] != '0') { more = 1; } i = i + 1; }
    int up = 0;
    if (d > 5) { up = 1; }
    else if (d == 5) {
        if (more == 1) { up = 1; }
        else if (n > 0 && ((int)__rt_dg[n - 1] - 48) % 2 == 1) { up = 1; }
    }
    __rt_dn = n;
    if (up == 0) { return 0; }
    i = n - 1;
    while (i >= 0) {
        if (__rt_dg[i] == '9') { __rt_dg[i] = '0'; i = i - 1; continue; }
        __rt_dg[i] = (char)((int)__rt_dg[i] + 1);
        return 0;
    }
    // all nines: shift right, leading 1
    i = __rt_dn;
    while (i > 0) { __rt_dg[i] = __rt_dg[i - 1]; i = i - 1; }
    __rt_dg[0] = '1';
    __rt_dn = __rt_dn + 1;
    __rt_dpt = __rt_dpt + 1;
    return 1;
}

// formatted float into fbuf; returns its length (no sign)
char __rt_fb[1400];
int __rt_fl = 0;

void __rt_fput(int c) {
    if (__rt_fl < 1400) { __rt_fb[__rt_fl] = (char)c; __rt_fl = __rt_fl + 1; }
}

void __rt_ffixed(double x, int prec, int alt) {
    __rt_dtoa(x, prec, 0);
    // digits after the point that we keep: prec
    __rt_round(__rt_dpt + prec);
    __rt_fl = 0;
    int i = 0;
    if (__rt_dpt <= 0) { __rt_fput(48); }
    while (i < __rt_dpt) {
        if (i < __rt_dn) { __rt_fput((int)__rt_dg[i]); } else { __rt_fput(48); }
        i = i + 1;
    }
    if (prec > 0 || alt == 1) { __rt_fput(46); }
    int k = 0;
    while (k < prec) {
        int idx = __rt_dpt + k;
        if (idx >= 0 && idx < __rt_dn) { __rt_fput((int)__rt_dg[idx]); } else { __rt_fput(48); }
        k = k + 1;
    }
}

// scientific: returns the decimal exponent
int __rt_fexp(double x, int prec, int alt, int upper) {
    int exp10 = 0;
    if (x == 0.0) {
        __rt_dn = prec + 1;
        int i = 0;
        while (i < __rt_dn) { __rt_dg[i] = '0'; i = i + 1; }
        exp10 = 0;
    } else {
        __rt_dtoa(x, 0, prec + 1);
        // skip leading zeros (numbers below 1)
        int lead = 0;
        while (lead < __rt_dn && __rt_dg[lead] == '0') { lead = lead + 1; }
        exp10 = __rt_dpt - lead - 1;
        int i = 0;
        while (lead + i < __rt_dn) { __rt_dg[i] = __rt_dg[lead + i]; i = i + 1; }
        __rt_dn = __rt_dn - lead;
        __rt_dpt = 1;
        if (__rt_round(prec + 1) == 1) { exp10 = exp10 + 1; __rt_dn = prec + 1; }
    }
    __rt_fl = 0;
    __rt_fput((int)__rt_dg[0]);
    if (prec > 0 || alt == 1) { __rt_fput(46); }
    int k = 1;
    while (k <= prec) {
        if (k < __rt_dn) { __rt_fput((int)__rt_dg[k]); } else { __rt_fput(48); }
        k = k + 1;
    }
    if (upper == 1) { __rt_fput(69); } else { __rt_fput(101); }
    int ae = exp10;
    if (ae < 0) { __rt_fput(45); ae = 0 - ae; } else { __rt_fput(43); }
    if (ae >= 100) { __rt_fput(48 + ae / 100); }
    __rt_fput(48 + (ae / 10) % 10);
    __rt_fput(48 + ae % 10);
    return exp10;
}

void __rt_fgen(double x, int prec, int alt, int upper) {
    if (prec == 0) { prec = 1; }
    int ex = 0;
    if (x != 0.0) {
        __rt_fexp(x, prec - 1, 0, upper);
        // the exponent after rounding
        int i = 0;
        while (i < __rt_fl && __rt_fb[i] != 'e' && __rt_fb[i] != 'E') { i = i + 1; }
        int sign = 1;
        if (__rt_fb[i + 1] == '-') { sign = -1; }
        int v = 0;
        int j = i + 2;
        while (j < __rt_fl) { v = v * 10 + ((int)__rt_fb[j] - 48); j = j + 1; }
        ex = sign * v;
    }
    if (ex < -4 || ex >= prec) {
        __rt_fexp(x, prec - 1, alt, upper);
        if (alt == 0) {
            // strip trailing zeros of the mantissa
            int epos = 0;
            while (epos < __rt_fl && __rt_fb[epos] != 'e' && __rt_fb[epos] != 'E') { epos = epos + 1; }
            int end = epos;
            int hasdot = 0;
            int q = 0;
            while (q < epos) { if (__rt_fb[q] == '.') { hasdot = 1; } q = q + 1; }
            if (hasdot == 1) {
                while (end > 0 && __rt_fb[end - 1] == '0') { end = end - 1; }
                if (end > 0 && __rt_fb[end - 1] == '.') { end = end - 1; }
            }
            int d = epos - end;
            q = epos;
            while (q < __rt_fl) { __rt_fb[q - d] = __rt_fb[q]; q = q + 1; }
            __rt_fl = __rt_fl - d;
        }
        return;
    }
    __rt_ffixed(x, prec - 1 - ex, alt);
    if (alt == 0) {
        int hasdot = 0;
        int q = 0;
        while (q < __rt_fl) { if (__rt_fb[q] == '.') { hasdot = 1; } q = q + 1; }
        if (hasdot == 1) {
            while (__rt_fl > 0 && __rt_fb[__rt_fl - 1] == '0') { __rt_fl = __rt_fl - 1; }
            if (__rt_fl > 0 && __rt_fb[__rt_fl - 1] == '.') { __rt_fl = __rt_fl - 1; }
        }
    }
}

// ── printf ───────────────────────────────────────────────────────────────────
char __rt_nb[80];

int __rt_vfmt(const char* fmt, unsigned char* ap) {
    __rt_count = 0;
    unsafe {
        int ai = 0;
        int i = 0;
        while (fmt[i] != (char)0) {
            int c = (int)fmt[i];
            if (c != 37) { __rt_put(c); i = i + 1; continue; }
            i = i + 1;
            int left = 0;
            int plus = 0;
            int space = 0;
            int alt = 0;
            int zero = 0;
            while (1) {
                c = (int)fmt[i];
                if (c == 45) { left = 1; } else if (c == 43) { plus = 1; } else if (c == 32) { space = 1; }
                else if (c == 35) { alt = 1; } else if (c == 48) { zero = 1; } else { break; }
                i = i + 1;
            }
            int width = 0;
            if (fmt[i] == '*') {
                width = (int)*(long*)(ap + 8 * ai);
                ai = ai + 1;
                if (width < 0) { left = 1; width = 0 - width; }
                i = i + 1;
            } else {
                while (fmt[i] >= '0' && fmt[i] <= '9') { width = width * 10 + ((int)fmt[i] - 48); i = i + 1; }
            }
            int prec = -1;
            if (fmt[i] == '.') {
                i = i + 1;
                prec = 0;
                if (fmt[i] == '*') {
                    prec = (int)*(long*)(ap + 8 * ai);
                    ai = ai + 1;
                    i = i + 1;
                } else {
                    while (fmt[i] >= '0' && fmt[i] <= '9') { prec = prec * 10 + ((int)fmt[i] - 48); i = i + 1; }
                }
            }
            // length modifiers: hh h l ll z j t L q
            int len = 0;
            while (fmt[i] == 'h' || fmt[i] == 'l' || fmt[i] == 'z' || fmt[i] == 'j' || fmt[i] == 't' || fmt[i] == 'L' || fmt[i] == 'q') {
                if (fmt[i] == 'h') { len = len - 1; } else { len = 2; }
                i = i + 1;
            }
            c = (int)fmt[i];
            if (c == 0) { break; }
            i = i + 1;
            long* slot = (long*)(ap + 8 * ai);
            if (c == 37) { __rt_put(37); continue; }
            if (c == 'c') {
                ai = ai + 1;
                if (left == 0) { __rt_pad(32, width - 1); }
                __rt_put((int)(*slot & 255L));
                if (left == 1) { __rt_pad(32, width - 1); }
                continue;
            }
            if (c == 's') {
                ai = ai + 1;
                const char* s = (const char*)(int)*slot;
                if (s == null) { s = "(null)"; }
                int n = (int)strlen(s);
                if (prec >= 0 && prec < n) { n = prec; }
                if (left == 0) { __rt_pad(32, width - n); }
                int k = 0;
                while (k < n) { __rt_put((int)s[k]); k = k + 1; }
                if (left == 1) { __rt_pad(32, width - n); }
                continue;
            }
            if (c == 'd' || c == 'i' || c == 'u' || c == 'x' || c == 'X' || c == 'o' || c == 'p') {
                ai = ai + 1;
                long sv = *slot;
                int neg = 0;
                unsigned long uv = (unsigned long)sv;
                int is_signed = (int)(c == 'd' || c == 'i');
                if (c == 'p') { len = 2; uv = uv & 4294967295UL; alt = 1; }
                if (len == 0) {
                    if (is_signed == 1) { sv = (long)(int)sv; } else { uv = uv & 4294967295UL; }
                } else if (len == -1) {
                    if (is_signed == 1) { sv = (long)(short)sv; } else { uv = uv & 65535UL; }
                } else if (len < -1) {
                    if (is_signed == 1) { sv = (long)(signed char)sv; } else { uv = uv & 255UL; }
                }
                if (is_signed == 1) {
                    if (sv < 0L) { neg = 1; uv = (unsigned long)(0L - sv); } else { uv = (unsigned long)sv; }
                }
                unsigned long base = 10UL;
                if (c == 'x' || c == 'X' || c == 'p') { base = 16UL; }
                if (c == 'o') { base = 8UL; }
                int nd = 0;
                while (uv > 0UL) {
                    int d = (int)(uv % base);
                    if (d < 10) { __rt_nb[nd] = (char)(48 + d); }
                    else if (c == 'X') { __rt_nb[nd] = (char)(55 + d); }
                    else { __rt_nb[nd] = (char)(87 + d); }
                    uv = uv / base;
                    nd = nd + 1;
                }
                int digits = nd;
                if (prec < 0 && digits == 0) { digits = 1; __rt_nb[0] = '0'; nd = 1; }
                if (prec >= 0) { zero = 0; }
                int zeros = 0;
                if (prec > digits) { zeros = prec - digits; }
                int signch = 0;
                if (neg == 1) { signch = 45; } else if (is_signed == 1 && plus == 1) { signch = 43; } else if (is_signed == 1 && space == 1) { signch = 32; }
                int prefix = 0;
                if (alt == 1 && (c == 'x' || c == 'X' || c == 'p') && (nd > 0 && !(nd == 1 && __rt_nb[0] == '0' && c != 'p'))) { prefix = 2; }
                if (alt == 1 && c == 'o' && zeros == 0 && !(nd == 1 && __rt_nb[0] == '0')) { zeros = 1; }
                int total = nd + zeros + prefix;
                if (signch != 0) { total = total + 1; }
                if (left == 0 && zero == 0) { __rt_pad(32, width - total); }
                if (signch != 0) { __rt_put(signch); }
                if (prefix == 2) { __rt_put(48); if (c == 'X') { __rt_put(88); } else { __rt_put(120); } }
                if (left == 0 && zero == 1) { __rt_pad(48, width - total); }
                __rt_pad(48, zeros);
                while (nd > 0) { nd = nd - 1; __rt_put((int)__rt_nb[nd]); }
                if (left == 1) { __rt_pad(32, width - total); }
                continue;
            }
            if (c == 'f' || c == 'F' || c == 'e' || c == 'E' || c == 'g' || c == 'G') {
                ai = ai + 1;
                double x = __builtin_f64_from_bits(*slot);
                int neg = 0;
                long xb = __builtin_f64_bits(x);
                if (xb < 0L) { neg = 1; x = 0.0 - x; }
                if (prec < 0) { prec = 6; }
                int upper = (int)(c == 'F' || c == 'E' || c == 'G');
                if (x != x || x > 1.7976931348623157e308) {
                    __rt_fl = 0;
                    if (x != x) {
                        if (upper == 1) { __rt_fput(78); __rt_fput(65); __rt_fput(78); } else { __rt_fput(110); __rt_fput(97); __rt_fput(110); }
                    } else {
                        if (upper == 1) { __rt_fput(73); __rt_fput(78); __rt_fput(70); } else { __rt_fput(105); __rt_fput(110); __rt_fput(102); }
                    }
                    zero = 0;
                } else if (c == 'f' || c == 'F') { __rt_ffixed(x, prec, alt); }
                else if (c == 'e' || c == 'E') { __rt_fexp(x, prec, alt, upper); }
                else { __rt_fgen(x, prec, alt, upper); }
                int signch = 0;
                if (neg == 1) { signch = 45; } else if (plus == 1) { signch = 43; } else if (space == 1) { signch = 32; }
                int total = __rt_fl;
                if (signch != 0) { total = total + 1; }
                if (left == 0 && zero == 0) { __rt_pad(32, width - total); }
                if (signch != 0) { __rt_put(signch); }
                if (left == 0 && zero == 1) { __rt_pad(48, width - total); }
                int k = 0;
                while (k < __rt_fl) { __rt_put((int)__rt_fb[k]); k = k + 1; }
                if (left == 1) { __rt_pad(32, width - total); }
                continue;
            }
            // unknown conversion: print it
            __rt_put(37);
            __rt_put(c);
        }
    }
    return __rt_count;
}

int printf(const char* fmt, ...) {
    __rt_ofd = 1;
    __rt_sbuf = null;
    int n = __rt_vfmt(fmt, __va_ptr());
    __rt_oflush();
    return n;
}

// ── atomics, volatile access, byte swaps (the builtins of those names map
// here; one thread, so plain loads and stores) ──────────────────────────────
generic<T> T __rt_atomic_load(T* p, ...) { unsafe { return *p; } }
generic<T, U> void __rt_atomic_store(T* p, U v, ...) { unsafe { *p = (T)v; } }
generic<T, U> T __rt_atomic_fetch_add(T* p, U v, ...) { unsafe { T old = *p; *p = old + (T)v; return old; } }
generic<T, U> T __rt_atomic_fetch_sub(T* p, U v, ...) { unsafe { T old = *p; *p = old - (T)v; return old; } }
generic<T, U> T __rt_atomic_fetch_and(T* p, U v, ...) { unsafe { T old = *p; *p = old & (T)v; return old; } }
generic<T, U> T __rt_atomic_fetch_or(T* p, U v, ...) { unsafe { T old = *p; *p = old | (T)v; return old; } }
generic<T, U> T __rt_atomic_fetch_xor(T* p, U v, ...) { unsafe { T old = *p; *p = old ^ (T)v; return old; } }
generic<T, U> T __rt_atomic_exchange(T* p, U v, ...) { unsafe { T old = *p; *p = (T)v; return old; } }
generic<T, U, V> bool __rt_atomic_cas(T* p, U expected, V desired, ...) {
    unsafe { if (*p == (T)expected) { *p = (T)desired; return true; } }
    return false;
}
generic<T, E, V> bool __rt_atomic_compare_exchange_strong(T* p, E expected, V desired, ...) {
    unsafe {
        if (*p == *expected) { *p = (T)desired; return true; }
        *expected = *p;
    }
    return false;
}
void __rt_atomic_fence(...) { }
generic<T> T __rt_volatile_load(T* p, ...) { unsafe { return *p; } }
generic<T, U> void __rt_volatile_store(T* p, U v, ...) { unsafe { *p = (T)v; } }
generic<T> void __rt___builtin_prefetch(T p) { }
unsigned int __rt___builtin_bswap32(unsigned int x) {
    return ((x >> 24U) & 255U) | ((x >> 8U) & 65280U) | ((x << 8U) & 16711680U) | (x << 24U);
}
unsigned long __rt___builtin_bswap64(unsigned long x) {
    unsigned long lo = (unsigned long)__rt___builtin_bswap32((unsigned int)(x & 4294967295UL));
    unsigned long hi = (unsigned long)__rt___builtin_bswap32((unsigned int)(x >> 32UL));
    return (lo << 32UL) | hi;
}

// FILE: just a file descriptor
struct __FILE { int fd; };
struct __FILE __rt_files[3];
struct __FILE* stdin = &__rt_files[0];
struct __FILE* stdout = &__rt_files[1];
struct __FILE* stderr = &__rt_files[2];

// stdin/stdout/stderr are &__rt_files[0..2]: the fd is the index
int __rt_fd(struct __FILE* f) {
    int fd = 1;
    unsafe { if (f != null) { fd = ((int)f - (int)&__rt_files[0]) / 4; } }
    return fd;
}

int fprintf(struct __FILE* f, const char* fmt, ...) {
    __rt_ofd = __rt_fd(f);
    __rt_sbuf = null;
    int n = __rt_vfmt(fmt, __va_ptr());
    __rt_oflush();
    __rt_ofd = 1;
    return n;
}

int fputs(const char* s, struct __FILE* f) {
    int fd = __rt_fd(f);
    unsafe { __host_write(fd, s, (int)strlen(s)); }
    return 0;
}

int fputc(int c, struct __FILE* f) {
    char b[1];
    b[0] = (char)c;
    int fd = __rt_fd(f);
    unsafe { __host_write(fd, (const char*)&b[0], 1); }
    return c;
}

int putc(int c, struct __FILE* f) { return fputc(c, f); }
int fflush(struct __FILE* f) { return 0; }

int sprintf(char* buf, const char* fmt, ...) {
    __rt_sbuf = buf;
    __rt_scap = 2147483647UL;
    __rt_spos = 0UL;
    int n = __rt_vfmt(fmt, __va_ptr());
    unsafe { buf[__rt_spos] = (char)0; }
    __rt_sbuf = null;
    return n;
}

int snprintf(char* buf, unsigned long cap, const char* fmt, ...) {
    __rt_sbuf = buf;
    __rt_scap = cap;
    __rt_spos = 0UL;
    int n = __rt_vfmt(fmt, __va_ptr());
    unsafe {
        if (cap > 0UL) {
            if (__rt_spos < cap) { buf[__rt_spos] = (char)0; } else { buf[cap - 1UL] = (char)0; }
        }
    }
    __rt_sbuf = null;
    return n;
}

// ── input ────────────────────────────────────────────────────────────────────
char __rt_ibuf[4096];
int __rt_ipos = 0;
int __rt_ilen = 0;
int __rt_ieof = 0;

int __rt_unget = -1;   // one character pushed back by ungetc

int getchar() {
    if (__rt_unget >= 0) { int u = __rt_unget; __rt_unget = -1; return u; }
    if (__rt_ipos >= __rt_ilen) {
        if (__rt_ieof == 1) { return -1; }
        unsafe { __rt_ilen = __host_read(0, (char*)&__rt_ibuf[0], 4096); }
        __rt_ipos = 0;
        if (__rt_ilen <= 0) { __rt_ilen = 0; __rt_ieof = 1; return -1; }
    }
    int c = (int)(unsigned char)__rt_ibuf[__rt_ipos];
    __rt_ipos = __rt_ipos + 1;
    return c;
}

int __rt_peekc() {
    if (__rt_unget >= 0) { return __rt_unget; }
    int c = getchar();
    if (c >= 0) { __rt_ipos = __rt_ipos - 1; }
    return c;
}

// the scanner's input: a string (sscanf) or stdin
const char* __rt_scan_str = null;
int __rt_scan_pos = 0;
int __rt_speek() {
    if (__rt_scan_str != null) {
        int c = 0;
        unsafe { c = (int)(unsigned char)__rt_scan_str[__rt_scan_pos]; }
        if (c == 0) { return -1; }
        return c;
    }
    return __rt_peekc();
}
int __rt_sget() {
    if (__rt_scan_str != null) {
        int c = __rt_speek();
        if (c >= 0) { __rt_scan_pos = __rt_scan_pos + 1; }
        return c;
    }
    if (__rt_unget >= 0) { int c = __rt_unget; __rt_unget = -1; return c; }
    return getchar();
}

// files: there is no file system, so fopen fails; stdin/stdout/stderr work
struct __FILE* fopen(const char* path, const char* mode) { return null; }
struct __FILE* tmpfile() { return null; }
int fclose(struct __FILE* f) { return 0; }
int remove(const char* path) { return -1; }
int rename(const char* from, const char* to) { return -1; }
int fseek(struct __FILE* f, long off, int whence) { return -1; }
long ftell(struct __FILE* f) { return -1L; }
void rewind(struct __FILE* f) { }
int feof(struct __FILE* f) { if (__rt_fd(f) == 0) { return __rt_ieof; } return 0; }
int ferror(struct __FILE* f) { return 0; }
int fgetc(struct __FILE* f) { if (__rt_fd(f) != 0) { return -1; } return getchar(); }
int getc(struct __FILE* f) { return fgetc(f); }
int ungetc(int c, struct __FILE* f) {
    if (c < 0 || __rt_fd(f) != 0) { return -1; }
    __rt_unget = c & 255;
    return c & 255;
}
unsigned long fread(void* buf, unsigned long size, unsigned long count, struct __FILE* f) {
    if (__rt_fd(f) != 0 || size == 0UL) { return 0UL; }
    unsigned long total = size * count;
    unsigned long i = 0UL;
    unsafe {
        unsigned char* b = (unsigned char*)buf;
        while (i < total) {
            int c = getchar();
            if (c < 0) { break; }
            b[i] = (unsigned char)c;
            i = i + 1UL;
        }
    }
    return i / size;
}
unsigned long fwrite(const void* buf, unsigned long size, unsigned long count, struct __FILE* f) {
    int fd = __rt_fd(f);
    if (fd != 1 && fd != 2) { return 0UL; }
    __rt_oflush();
    unsafe { __host_write(fd, (const char*)buf, (int)(size * count)); }
    return count;
}

char* fgets(char* buf, int n, struct __FILE* f) {
    int i = 0;
    unsafe {
        while (i < n - 1) {
            int c = getchar();
            if (c < 0) { break; }
            buf[i] = (char)c;
            i = i + 1;
            if (c == 10) { break; }
        }
        buf[i] = (char)0;
        if (i == 0) { return null; }
    }
    return buf;
}

double strtod(const char* s, char** end) {
    double v = 0.0;
    int neg = 0;
    unsafe {
        int i = 0;
        while (isspace((int)s[i]) != 0) { i = i + 1; }
        if (s[i] == '-') { neg = 1; i = i + 1; } else if (s[i] == '+') { i = i + 1; }
        long mant = 0L;
        int nd = 0;
        int ex = 0;
        while (s[i] >= '0' && s[i] <= '9') {
            if (nd < 18) { mant = mant * 10L + (long)((int)s[i] - 48); if (mant != 0L) { nd = nd + 1; } } else { ex = ex + 1; }
            i = i + 1;
        }
        if (s[i] == '.') {
            i = i + 1;
            while (s[i] >= '0' && s[i] <= '9') {
                if (nd < 18) { mant = mant * 10L + (long)((int)s[i] - 48); if (mant != 0L) { nd = nd + 1; } ex = ex - 1; }
                i = i + 1;
            }
        }
        if (s[i] == 'e' || s[i] == 'E') {
            int j = i + 1;
            int es = 1;
            if (s[j] == '-') { es = -1; j = j + 1; } else if (s[j] == '+') { j = j + 1; }
            if (s[j] >= '0' && s[j] <= '9') {
                int ev = 0;
                while (s[j] >= '0' && s[j] <= '9') { if (ev < 10000) { ev = ev * 10 + ((int)s[j] - 48); } j = j + 1; }
                ex = ex + es * ev;
                i = j;
            }
        }
        v = (double)mant;
        double p = 1.0;
        double b = 10.0;
        int ae = ex;
        if (ae < 0) { ae = 0 - ae; }
        while (ae > 0) {
            if ((ae & 1) != 0) { p = p * b; }
            b = b * b;
            ae = ae >> 1;
        }
        if (ex < 0) { v = v / p; } else { v = v * p; }
        if (end != null) { *end = (char*)(s + i); }
    }
    if (neg == 1) { return 0.0 - v; }
    return v;
}

double atof(const char* s) { return strtod(s, null); }
float strtof(const char* s, char** end) { return (float)strtod(s, end); }

char* getenv(const char* name) { return null; }   // no environment
int system(const char* cmd) { return -1; }

// qsort/bsearch: insertion sort through the comparison function
void qsort(void* base, unsigned long n, unsigned long size, void* cmp) {
    unsafe {
        fn int(const void*, const void*) c = (fn int(const void*, const void*))cmp;
        unsigned char* b = (unsigned char*)base;
        unsigned char* tmp = (unsigned char*)malloc(size);
        if (tmp == null) { return; }
        unsigned long i = 1UL;
        while (i < n) {
            memcpy((void*)tmp, (const void*)(b + i * size), size);
            unsigned long j = i;
            while (j > 0UL && c((const void*)(b + (j - 1UL) * size), (const void*)tmp) > 0) {
                memcpy((void*)(b + j * size), (const void*)(b + (j - 1UL) * size), size);
                j = j - 1UL;
            }
            memcpy((void*)(b + j * size), (const void*)tmp, size);
            i = i + 1UL;
        }
        free((void*)tmp);
    }
}
void* bsearch(const void* key, const void* base, unsigned long n, unsigned long size, void* cmp) {
    unsafe {
        fn int(const void*, const void*) c = (fn int(const void*, const void*))cmp;
        const unsigned char* b = (const unsigned char*)base;
        unsigned long lo = 0UL;
        unsigned long hi = n;
        while (lo < hi) {
            unsigned long mid = lo + (hi - lo) / 2UL;
            int r = c(key, (const void*)(b + mid * size));
            if (r == 0) { return (void*)(b + mid * size); }
            if (r < 0) { hi = mid; } else { lo = mid + 1UL; }
        }
    }
    return null;
}

// scanf: %d %i %u %ld %lld %f %lf %s %c %%, whitespace in the format
// skips input whitespace
int __rt_vscan(const char* fmt, unsigned char* ap);
int scanf(const char* fmt, ...) {
    __rt_scan_str = null;
    unsafe { return __rt_vscan(fmt, __va_ptr()); }
}
int sscanf(const char* s, const char* fmt, ...) {
    __rt_scan_str = s;
    __rt_scan_pos = 0;
    int n = 0;
    unsafe { n = __rt_vscan(fmt, __va_ptr()); }
    __rt_scan_str = null;
    return n;
}
int fscanf(struct __FILE* f, const char* fmt, ...) {
    if (__rt_fd(f) != 0) { return -1; }
    __rt_scan_str = null;
    unsafe { return __rt_vscan(fmt, __va_ptr()); }
}
int __rt_vscan(const char* fmt, unsigned char* ap) {
    int assigned = 0;
    unsafe {
        int ai = 0;
        int i = 0;
        while (fmt[i] != (char)0) {
            int c = (int)fmt[i];
            if (isspace(c) != 0) {
                while (isspace(__rt_speek()) != 0) { __rt_sget(); }
                i = i + 1;
                continue;
            }
            if (c != 37) {
                if (__rt_speek() != c) { return assigned; }
                __rt_sget();
                i = i + 1;
                continue;
            }
            i = i + 1;
            int len = 0;
            while (fmt[i] == 'l' || fmt[i] == 'h') { if (fmt[i] == 'l') { len = len + 1; } i = i + 1; }
            c = (int)fmt[i];
            i = i + 1;
            if (c == 37) { if (__rt_speek() != 37) { return assigned; } __rt_sget(); continue; }
            int target = (int)*(long*)(ap + 8 * ai);
            ai = ai + 1;
            if (c != 'c') {
                while (isspace(__rt_speek()) != 0) { __rt_sget(); }
            }
            if (__rt_speek() < 0) {
                if (assigned == 0) { return -1; }
                return assigned;
            }
            if (c == 'c') { *(char*)target = (char)__rt_sget(); assigned = assigned + 1; continue; }
            char tok[128];
            int n = 0;
            if (c == 's') {
                while (n < 127 && __rt_speek() >= 0 && isspace(__rt_speek()) == 0) { tok[n] = (char)__rt_sget(); n = n + 1; }
                tok[n] = (char)0;
                strcpy((char*)target, (const char*)&tok[0]);
                assigned = assigned + 1;
                continue;
            }
            int pc = __rt_speek();
            if (pc == 45 || pc == 43) { tok[n] = (char)__rt_sget(); n = n + 1; }
            while (n < 127) {
                pc = __rt_speek();
                int ok = isdigit(pc);
                if ((c == 'f' || c == 'e' || c == 'g') && (pc == 46 || pc == 101 || pc == 69)) { ok = 1; }
                if (ok == 0) { break; }
                tok[n] = (char)__rt_sget();
                n = n + 1;
            }
            tok[n] = (char)0;
            if (n == 0) { return assigned; }
            if (c == 'f' || c == 'e' || c == 'g') {
                double d = atof((const char*)&tok[0]);
                if (len > 0) { *(double*)target = d; } else { *(float*)target = (float)d; }
            } else {
                long v = strtol((const char*)&tok[0], null, 10);
                if (len >= 1) { *(long*)target = v; } else { *(int*)target = (int)v; }
            }
            assigned = assigned + 1;
        }
    }
    return assigned;
}

// ── math ─────────────────────────────────────────────────────────────────────
double sqrt(double x) { return __builtin_sqrt(x); }
double fabs(double x) { return __builtin_fabs(x); }
double floor(double x) { return __builtin_floor(x); }
double ceil(double x) { return __builtin_ceil(x); }
double trunc(double x) { return __builtin_trunc(x); }
double round(double x) {
    double t = __builtin_trunc(x);
    if (x - t >= 0.5) { return t + 1.0; }
    if (t - x >= 0.5) { return t - 1.0; }
    return t;
}
float sqrtf(float x) { return __builtin_sqrt(x); }
float fabsf(float x) { return __builtin_fabs(x); }
float floorf(float x) { return __builtin_floor(x); }
float ceilf(float x) { return __builtin_ceil(x); }

double fmod(double x, double y) {
    if (y == 0.0 || x != x || y != y) { return 0.0 / 0.0; }
    double q = __builtin_trunc(x / y);
    double r = x - q * y;
    // correct the rare rounding of x/y
    if (y > 0.0) {
        if (x >= 0.0 && r < 0.0) { r = r + y; }
        if (x < 0.0 && r > 0.0) { r = r - y; }
    }
    return r;
}

double fmin(double a, double b) { if (a < b) { return a; } return b; }
double fmax(double a, double b) { if (a > b) { return a; } return b; }

// scale by 2^n exactly
double __rt_ldexp(double x, int n) {
    while (n > 1000) { x = x * 1.0715086071862673e301; n = n - 1000; }
    while (n < -1000) { x = x * 9.332636185032189e-302; n = n + 1000; }
    long b = (long)(n + 1023) << 52L;
    if (n >= -1022) { return x * __builtin_f64_from_bits(b); }
    return x * 2.2250738585072014e-308 * __builtin_f64_from_bits((long)(n + 1022 + 1023) << 52L);
}

double ldexp(double x, int n) { return __rt_ldexp(x, n); }

// exp: x = k ln2 + r, |r| <= ln2/2 (fdlibm's e_exp)
double exp(double x) {
    if (x != x) { return x; }
    if (x > 709.782712893384) { return 1.0 / 0.0; }
    if (x < -745.1332191019411) { return 0.0; }
    double ln2hi = 6.93147180369123816490e-01;
    double ln2lo = 1.90821492927058770002e-10;
    double invln2 = 1.44269504088896338700e+00;
    int k = 0;
    double hi = x;
    double lo = 0.0;
    double ax = __builtin_fabs(x);
    if (ax > 0.34657359027997264) {
        double kf = __builtin_nearest(x * invln2);
        k = (int)kf;
        hi = x - kf * ln2hi;
        lo = kf * ln2lo;
    } else if (ax < 3.725290298461914e-09) {
        return 1.0 + x;
    }
    double r = hi - lo;
    double t = r * r;
    double c = r - t * (1.66666666666666019037e-01 + t * (-2.77777777770155933842e-03 + t * (6.61375632143793436117e-05 +
               t * (-1.65339022054652515390e-06 + t * 4.13813679705723846039e-08))));
    double y = 1.0 - ((lo - (r * c) / (2.0 - c)) - hi);
    if (k == 0) { return y; }
    return __rt_ldexp(y, k);
}

// log: fdlibm's e_log
double log(double x) {
    if (x != x || x < 0.0) { return 0.0 / 0.0; }
    if (x == 0.0) { return -1.0 / 0.0; }
    if (x > 1.7976931348623157e308) { return x; }
    long bits = __builtin_f64_bits(x);
    int k = 0;
    if (bits < 4503599627370496L) {
        // subnormal: scale up
        x = x * 18014398509481984.0;
        bits = __builtin_f64_bits(x);
        k = -54;
    }
    long hx = bits >> 32L;
    k = k + (int)((hx >> 20L) - 1023L);
    hx = hx & 1048575L;
    long i = (hx + 614244L) & 1048576L;
    // normalize x or x/2 to [sqrt(2)/2, sqrt(2)]
    long nb = ((hx | (i ^ 1072693248L)) << 32L) | (bits & 4294967295L);
    x = __builtin_f64_from_bits(nb);
    k = k + (int)(i >> 20L);
    double f = x - 1.0;
    double dk = (double)k;
    double s = f / (2.0 + f);
    double z = s * s;
    double w = z * z;
    double t1 = w * (3.999999999940941908e-01 + w * (2.222219843214978396e-01 + w * 1.531383769920937332e-01));
    double t2 = z * (6.666666666666735130e-01 + w * (2.857142874366239149e-01 + w * (1.818357216161805012e-01 + w * 1.479819860511658591e-01)));
    double R = t2 + t1;
    double hfsq = 0.5 * f * f;
    double ln2hi = 6.93147180369123816490e-01;
    double ln2lo = 1.90821492927058770002e-10;
    return dk * ln2hi - ((hfsq - (s * (hfsq + R) + dk * ln2lo)) - f);
}

double log2(double x) { return log(x) / 0.6931471805599453; }
double log10(double x) { return log(x) / 2.302585092994046; }

double pow(double x, double y) {
    if (y == 0.0) { return 1.0; }
    if (x != x || y != y) { return 0.0 / 0.0; }
    // integral exponents: exact repeated squaring when |y| is moderate
    double yi = __builtin_trunc(y);
    if (yi == y && __builtin_fabs(y) <= 1048576.0) {
        long n = (long)y;
        int neg = 0;
        if (n < 0L) { neg = 1; n = 0L - n; }
        double r = 1.0;
        double b = x;
        while (n > 0L) {
            if ((n & 1L) != 0L) { r = r * b; }
            b = b * b;
            n = n >> 1L;
        }
        if (neg == 1) { return 1.0 / r; }
        return r;
    }
    if (x < 0.0) { return 0.0 / 0.0; }
    if (x == 0.0) { if (y > 0.0) { return 0.0; } return 1.0 / 0.0; }
    return exp(y * log(x));
}

double cbrt(double x) {
    if (x == 0.0 || x != x) { return x; }
    double a = __builtin_fabs(x);
    double r = exp(log(a) / 3.0);
    // two Newton steps
    r = r - (r * r * r - a) / (3.0 * r * r);
    r = r - (r * r * r - a) / (3.0 * r * r);
    if (x < 0.0) { return 0.0 - r; }
    return r;
}

double hypot(double a, double b) { return __builtin_sqrt(a * a + b * b); }

// sin/cos: reduction by pi/2 (Cody–Waite, 3 parts) and fdlibm's kernels
double __rt_ksin(double x, double y, int iy) {
    double z = x * x;
    double v = z * x;
    double r = 8.33333333332248946124e-03 + z * (-1.98412698298579493134e-04 + z * (2.75573137070700676789e-06 +
               z * (-2.50507602534068634195e-08 + z * 1.58969099521155010221e-10)));
    if (iy == 0) { return x + v * (-1.66666666666666324348e-01 + z * r); }
    return x - ((z * (0.5 * y - v * r) - y) - v * -1.66666666666666324348e-01);
}

double __rt_kcos(double x, double y) {
    double z = x * x;
    double r = z * (4.16666666666666019037e-02 + z * (-1.38888888888741095749e-03 + z * (2.48015872894767294178e-05 +
               z * (-2.75573143513906633035e-07 + z * (2.08757232129817482790e-09 + z * -1.13596475577881948265e-11)))));
    double hz = 0.5 * z;
    double w = 1.0 - hz;
    return w + (((1.0 - w) - hz) + (z * r - x * y));
}

double __rt_rem_y = 0.0;
int __rt_rem_pio2(double x, double* yout) {
    double invpio2 = 6.36619772367581382433e-01;
    double p1 = 1.57079632673412561417e+00;
    double p1t = 6.07710050650619224932e-11;
    double p2 = 6.07710050630396597660e-11;
    double p2t = 2.02226624879595063154e-21;
    double p3 = 2.02226624871116645580e-21;
    double p3t = 8.47842766036889956997e-32;
    double fnv = __builtin_nearest(x * invpio2);
    double r = x - fnv * p1;
    double w = fnv * p1t;
    double y0 = r - w;
    // second and third iterations for accuracy
    double t = r;
    w = fnv * p2;
    r = t - w;
    w = fnv * p2t - ((t - r) - w);
    y0 = r - w;
    t = r;
    w = fnv * p3;
    r = t - w;
    w = fnv * p3t - ((t - r) - w);
    y0 = r - w;
    double y1 = (r - y0) - w;
    unsafe { yout[0] = y0; yout[1] = y1; }
    long n = (long)fnv;
    return (int)(n & 3L);
}

double sin(double x) {
    if (x != x || __builtin_fabs(x) > 1.7976931348623157e308) { return 0.0 / 0.0; }
    if (__builtin_fabs(x) < 0.7853981633974483) { return __rt_ksin(x, 0.0, 0); }
    double y[2];
    int n = 0;
    unsafe { n = __rt_rem_pio2(x, &y[0]); }
    if (n == 0) { return __rt_ksin(y[0], y[1], 1); }
    if (n == 1) { return __rt_kcos(y[0], y[1]); }
    if (n == 2) { return 0.0 - __rt_ksin(y[0], y[1], 1); }
    return 0.0 - __rt_kcos(y[0], y[1]);
}

double cos(double x) {
    if (x != x || __builtin_fabs(x) > 1.7976931348623157e308) { return 0.0 / 0.0; }
    if (__builtin_fabs(x) < 0.7853981633974483) { return __rt_kcos(x, 0.0); }
    double y[2];
    int n = 0;
    unsafe { n = __rt_rem_pio2(x, &y[0]); }
    if (n == 0) { return __rt_kcos(y[0], y[1]); }
    if (n == 1) { return 0.0 - __rt_ksin(y[0], y[1], 1); }
    if (n == 2) { return 0.0 - __rt_kcos(y[0], y[1]); }
    return __rt_ksin(y[0], y[1], 1);
}

double tan(double x) { return sin(x) / cos(x); }

double atan(double x) {
    // reduce to [0, 0.4375] with fdlibm's breakpoints
    if (x != x) { return x; }
    int neg = 0;
    if (x < 0.0) { neg = 1; x = 0.0 - x; }
    double hi = 0.0;
    double lo = 0.0;
    int id = -1;
    if (x >= 0.4375) {
        if (x < 1.1875) {
            if (x < 0.6875) { id = 0; x = (2.0 * x - 1.0) / (2.0 + x); hi = 4.63647609000806093515e-01; lo = 2.26987774529616870924e-17; }
            else { id = 1; x = (x - 1.0) / (x + 1.0); hi = 7.85398163397448278999e-01; lo = 3.06161699786838301793e-17; }
        } else {
            if (x < 2.4375) { id = 2; x = (x - 1.5) / (1.0 + 1.5 * x); hi = 9.82793723247329054082e-01; lo = 1.39033110312309984516e-17; }
            else { id = 3; x = -1.0 / x; hi = 1.57079632679489655800e+00; lo = 6.12323399573676603587e-17; }
        }
    }
    double z = x * x;
    double w = z * z;
    double s1 = z * (3.33333333333329318027e-01 + w * (1.42857142725034663711e-01 + w * (9.09088713343650656196e-02 +
                w * (6.66107313738753120669e-02 + w * (4.97687799461593236017e-02 + w * 1.62858201153657823623e-02)))));
    double s2 = w * (-1.99999999998764832476e-01 + w * (-1.11111104054623557880e-01 + w * (-7.69187620504482999495e-02 +
                w * (-5.83357013379057348645e-02 + w * -3.65315727442169155270e-02))));
    double r = 0.0;
    if (id < 0) { r = x - x * (s1 + s2); }
    else { r = hi - ((x * (s1 + s2) - lo) - x); }
    if (neg == 1) { return 0.0 - r; }
    return r;
}

double atan2(double y, double x) {
    double pi = 3.14159265358979311600e+00;
    if (x != x || y != y) { return x + y; }
    if (x == 0.0) {
        if (y > 0.0) { return pi / 2.0; }
        if (y < 0.0) { return 0.0 - pi / 2.0; }
        return 0.0;
    }
    double a = atan(y / x);
    if (x > 0.0) { return a; }
    if (y >= 0.0) { return a + pi; }
    return a - pi;
}

double asin(double x) {
    if (x > 1.0 || x < -1.0) { return 0.0 / 0.0; }
    return atan2(x, __builtin_sqrt(1.0 - x * x));
}

double acos(double x) {
    if (x > 1.0 || x < -1.0) { return 0.0 / 0.0; }
    return atan2(__builtin_sqrt(1.0 - x * x), x);
}

double sinh(double x) { double e = exp(x); return (e - 1.0 / e) / 2.0; }
double cosh(double x) { double e = exp(x); return (e + 1.0 / e) / 2.0; }
double tanh(double x) {
    if (x > 20.0) { return 1.0; }
    if (x < -20.0) { return -1.0; }
    double e = exp(2.0 * x);
    return (e - 1.0) / (e + 1.0);
}

float sinf(float x) { return (float)sin((double)x); }
float cosf(float x) { return (float)cos((double)x); }
float tanf(float x) { return (float)tan((double)x); }
float expf(float x) { return (float)exp((double)x); }
float logf(float x) { return (float)log((double)x); }
float powf(float x, float y) { return (float)pow((double)x, (double)y); }
float atan2f(float y, float x) { return (float)atan2((double)y, (double)x); }
double copysign(double mag, double sgn) {
    double m = fabs(mag);
    if (__builtin_f64_bits(sgn) < 0L) { return 0.0 - m; }
    return m;
}
double exp2(double x) { return pow(2.0, x); }
double fma(double a, double b, double c) { return a * b + c; }
float acosf(float x) { return (float)acos((double)x); }
float asinf(float x) { return (float)asin((double)x); }
float atanf(float x) { return (float)atan((double)x); }
float cbrtf(float x) { return (float)cbrt((double)x); }
float coshf(float x) { return (float)cosh((double)x); }
float sinhf(float x) { return (float)sinh((double)x); }
float tanhf(float x) { return (float)tanh((double)x); }
float exp2f(float x) { return (float)exp2((double)x); }
float log10f(float x) { return (float)log10((double)x); }
float log2f(float x) { return (float)log2((double)x); }
float hypotf(float x, float y) { return (float)hypot((double)x, (double)y); }
float truncf(float x) { return (float)trunc((double)x); }
float copysignf(float m, float s) { return (float)copysign((double)m, (double)s); }
float fmaf(float a, float b, float c) { return (float)((double)a * (double)b + (double)c); }
float fmodf(float x, float y) { return (float)fmod((double)x, (double)y); }
float roundf(float x) { return (float)round((double)x); }

// ── time ─────────────────────────────────────────────────────────────────────
// The playground has no clock: time() and clock_gettime() report 0 (the
// epoch) and nanosleep() returns at once. Calendar conversion is real and
// exact (proleptic Gregorian, any year), and there is no time zone: local
// time is UTC. A struct tm is handled as its 9 leading int fields, as
// std/time.sc passes it: sec, min, hour, mday, mon (0-11), year (since
// 1900), wday (0 = Sunday), yday (0-365), isdst.
long time(long* t) {
    unsafe { if (t != null) { *t = 0L; } }
    return 0L;
}
long clock() { return 0L; }

// struct timespec on wasm32: two 4-byte fields
struct SafecTimespec_ { int tv_sec; int tv_nsec; };
int clock_gettime(int clk, SafecTimespec_* ts) {
    unsafe { if (ts != null) { ts->tv_sec = 0; ts->tv_nsec = 0; } }
    return 0;
}
int nanosleep(const SafecTimespec_* req, SafecTimespec_* rem) {
    unsafe { if (rem != null) { rem->tv_sec = 0; rem->tv_nsec = 0; } }
    return 0;
}

// floor division and the matching non-negative remainder
long long __rt_fdiv(long long a, long long b) {
    long long q = a / b;
    if ((a % b != 0LL) && ((a < 0LL) != (b < 0LL))) { q = q - 1LL; }
    return q;
}
long long __rt_fmod(long long a, long long b) { return a - __rt_fdiv(a, b) * b; }

// days since 1970-01-01 of y-m-d (m 1-12), and back (H. Hinnant's algorithm)
long long __rt_days_from_civil(long long y, long long m, long long d) {
    if (m <= 2LL) { y = y - 1LL; }
    long long era = __rt_fdiv(y, 400LL);
    long long yoe = y - era * 400LL;
    long long mp = m > 2LL ? m - 3LL : m + 9LL;
    long long doy = (153LL * mp + 2LL) / 5LL + d - 1LL;
    long long doe = yoe * 365LL + yoe / 4LL - yoe / 100LL + doy;
    return era * 146097LL + doe - 719468LL;
}
void __rt_civil_from_days(long long z, long long* yo, long long* mo, long long* dd) {
    z = z + 719468LL;
    long long era = __rt_fdiv(z, 146097LL);
    long long doe = z - era * 146097LL;
    long long yoe = (doe - doe / 1460LL + doe / 36524LL - doe / 146096LL) / 365LL;
    long long doy = doe - (365LL * yoe + yoe / 4LL - yoe / 100LL);
    long long mp = (5LL * doy + 2LL) / 153LL;
    long long d = doy - (153LL * mp + 2LL) / 5LL + 1LL;
    long long m = mp < 10LL ? mp + 3LL : mp - 9LL;
    unsafe {
        *yo = yoe + era * 400LL + (m <= 2LL ? 1LL : 0LL);
        *mo = m;
        *dd = d;
    }
}

int* gmtime_r(const long long* t, int* tm) {
    unsafe {
        long long s = *t;
        long long days = __rt_fdiv(s, 86400LL);
        long long rem = s - days * 86400LL;
        long long y = 0LL;
        long long m = 0LL;
        long long d = 0LL;
        __rt_civil_from_days(days, &y, &m, &d);
        tm[0] = (int)(rem % 60LL);
        tm[1] = (int)((rem / 60LL) % 60LL);
        tm[2] = (int)(rem / 3600LL);
        tm[3] = (int)d;
        tm[4] = (int)(m - 1LL);
        tm[5] = (int)(y - 1900LL);
        tm[6] = (int)__rt_fmod(days + 4LL, 7LL);     // 1970-01-01 was a Thursday
        tm[7] = (int)(days - __rt_days_from_civil(y, 1LL, 1LL));
        tm[8] = 0;
    }
    return tm;
}
int* localtime_r(const long long* t, int* tm) { return gmtime_r(t, tm); }

// the seconds tm's fields stand for (out of range fields carry over), and tm
// normalized to them, wday and yday included
long long __rt_timegm(int* tm) {
    long long secs = 0LL;
    unsafe {
        long long mon = (long long)tm[4];
        long long y = (long long)tm[5] + 1900LL + __rt_fdiv(mon, 12LL);
        mon = __rt_fmod(mon, 12LL);
        long long days = __rt_days_from_civil(y, mon + 1LL, 1LL) + (long long)tm[3] - 1LL;
        secs = days * 86400LL + (long long)tm[2] * 3600LL + (long long)tm[1] * 60LL + (long long)tm[0];
        gmtime_r(&secs, tm);
    }
    return secs;
}
long long mktime(int* tm) { return __rt_timegm(tm); }

const char* __rt_wday_name(int i, int full) {
    if (i < 0 || i > 6) { return "?"; }
    const char* longn[7] = { "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday" };
    const char* shortn[7] = { "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat" };
    unsafe { if (full != 0) { return longn[i]; } return shortn[i]; }
}
const char* __rt_mon_name(int i, int full) {
    if (i < 0 || i > 11) { return "?"; }
    const char* longn[12] = { "January", "February", "March", "April", "May", "June", "July",
                              "August", "September", "October", "November", "December" };
    const char* shortn[12] = { "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };
    unsafe { if (full != 0) { return longn[i]; } return shortn[i]; }
}

// strftime's output cursor: n bytes so far into s (of max); past max only counts
void __rt_tf_ch(char* s, unsigned long max, unsigned long* n, char c) {
    unsafe {
        if (*n < max) { s[*n] = c; }
        *n = *n + 1UL;
    }
}
void __rt_tf_str(char* s, unsigned long max, unsigned long* n, const char* t) {
    unsafe {
        int i = 0;
        while (t[i] != (char)0) { __rt_tf_ch(s, max, n, t[i]); i = i + 1; }
    }
}
// v in decimal, at least w digits, padded with 'pad' (0: no padding)
void __rt_tf_num(char* s, unsigned long max, unsigned long* n, long long v, int w, char pad) {
    char d[24];
    int k = 0;
    int neg = 0;
    if (v < 0LL) { neg = 1; v = 0LL - v; }
    unsafe {
        while (v > 0LL || k == 0) { d[k] = (char)(48LL + v % 10LL); v = v / 10LL; k = k + 1; }
        if (neg == 1) { __rt_tf_ch(s, max, n, '-'); w = w - 1; }
        int i = k;
        while (i < w && pad != (char)0) { __rt_tf_ch(s, max, n, pad); i = i + 1; }
        while (k > 0) { k = k - 1; __rt_tf_ch(s, max, n, d[k]); }
    }
}
void __rt_tf(char* s, unsigned long max, unsigned long* n, const char* fmt, const int* tm) {
    unsafe {
        int i = 0;
        while (fmt[i] != (char)0) {
            char c = fmt[i];
            if (c != '%' || fmt[i + 1] == (char)0) { __rt_tf_ch(s, max, n, c); i = i + 1; continue; }
            char f = fmt[i + 1];
            i = i + 2;
            int hour12 = tm[2] % 12;
            if (hour12 == 0) { hour12 = 12; }
            long long year = (long long)tm[5] + 1900LL;
            if (f == 'Y') { __rt_tf_num(s, max, n, year, 1, (char)0); }
            else if (f == 'C') { __rt_tf_num(s, max, n, __rt_fdiv(year, 100LL), 2, '0'); }
            else if (f == 'y') { __rt_tf_num(s, max, n, __rt_fmod(year, 100LL), 2, '0'); }
            else if (f == 'm') { __rt_tf_num(s, max, n, (long long)tm[4] + 1LL, 2, '0'); }
            else if (f == 'd') { __rt_tf_num(s, max, n, (long long)tm[3], 2, '0'); }
            else if (f == 'e') { __rt_tf_num(s, max, n, (long long)tm[3], 2, ' '); }
            else if (f == 'H') { __rt_tf_num(s, max, n, (long long)tm[2], 2, '0'); }
            else if (f == 'k') { __rt_tf_num(s, max, n, (long long)tm[2], 2, ' '); }
            else if (f == 'I') { __rt_tf_num(s, max, n, (long long)hour12, 2, '0'); }
            else if (f == 'l') { __rt_tf_num(s, max, n, (long long)hour12, 2, ' '); }
            else if (f == 'M') { __rt_tf_num(s, max, n, (long long)tm[1], 2, '0'); }
            else if (f == 'S') { __rt_tf_num(s, max, n, (long long)tm[0], 2, '0'); }
            else if (f == 'j') { __rt_tf_num(s, max, n, (long long)tm[7] + 1LL, 3, '0'); }
            else if (f == 'w') { __rt_tf_num(s, max, n, (long long)tm[6], 1, (char)0); }
            else if (f == 'u') { __rt_tf_num(s, max, n, tm[6] == 0 ? 7LL : (long long)tm[6], 1, (char)0); }
            else if (f == 'a') { __rt_tf_str(s, max, n, __rt_wday_name(tm[6], 0)); }
            else if (f == 'A') { __rt_tf_str(s, max, n, __rt_wday_name(tm[6], 1)); }
            else if (f == 'b' || f == 'h') { __rt_tf_str(s, max, n, __rt_mon_name(tm[4], 0)); }
            else if (f == 'B') { __rt_tf_str(s, max, n, __rt_mon_name(tm[4], 1)); }
            else if (f == 'p') { __rt_tf_str(s, max, n, tm[2] < 12 ? "AM" : "PM"); }
            else if (f == 'P') { __rt_tf_str(s, max, n, tm[2] < 12 ? "am" : "pm"); }
            else if (f == 'Z') { __rt_tf_str(s, max, n, "UTC"); }
            else if (f == 'z') { __rt_tf_str(s, max, n, "+0000"); }
            else if (f == 's') {
                int c2[11];
                int k = 0;
                while (k < 9) { c2[k] = tm[k]; k = k + 1; }
                __rt_tf_num(s, max, n, __rt_timegm(&c2[0]), 1, (char)0);
            }
            else if (f == 'n') { __rt_tf_ch(s, max, n, '\n'); }
            else if (f == 't') { __rt_tf_ch(s, max, n, '\t'); }
            else if (f == '%') { __rt_tf_ch(s, max, n, '%'); }
            else if (f == 'F') { __rt_tf(s, max, n, "%Y-%m-%d", tm); }
            else if (f == 'T') { __rt_tf(s, max, n, "%H:%M:%S", tm); }
            else if (f == 'R') { __rt_tf(s, max, n, "%H:%M", tm); }
            else if (f == 'D' || f == 'x') { __rt_tf(s, max, n, "%m/%d/%y", tm); }
            else if (f == 'X') { __rt_tf(s, max, n, "%H:%M:%S", tm); }
            else if (f == 'r') { __rt_tf(s, max, n, "%I:%M:%S %p", tm); }
            else if (f == 'c') { __rt_tf(s, max, n, "%a %b %e %H:%M:%S %Y", tm); }
            else { __rt_tf_ch(s, max, n, '%'); __rt_tf_ch(s, max, n, f); }
        }
    }
}
// the bytes written, not counting the NUL; 0 when they don't fit in max
unsigned long strftime(char* s, unsigned long max, const char* fmt, const int* tm) {
    unsigned long n = 0UL;
    __rt_tf(s, max, &n, fmt, tm);
    unsafe {
        if (n + 1UL > max) { if (max > 0UL) { s[0] = (char)0; } return 0UL; }
        s[n] = (char)0;
    }
    return n;
}

// strptime's number: up to 'w' digits (after optional blanks, and a sign
// when 'sign' is set) in [lo, hi]; the index past it, or -1
int __rt_tp_num(const char* s, int i, int w, int sign, long long lo, long long hi, long long* out) {
    unsafe {
        while (isspace((int)s[i]) != 0) { i = i + 1; }
        int neg = 0;
        if (sign != 0 && (s[i] == '-' || s[i] == '+')) { neg = s[i] == '-' ? 1 : 0; i = i + 1; }
        long long v = 0LL;
        int k = 0;
        while (k < w && s[i] >= '0' && s[i] <= '9') { v = v * 10LL + (long long)(s[i] - '0'); i = i + 1; k = k + 1; }
        if (k == 0) { return -1; }
        if (neg == 1) { v = 0LL - v; }
        if (v < lo || v > hi) { return -1; }
        *out = v;
    }
    return i;
}
// a month or weekday name (full or abbreviated, any case) at s+i: its
// index, with *next past it, or -1
int __rt_tp_name(const char* s, int i, int months, int* next) {
    int count = months != 0 ? 12 : 7;
    int best = -1;
    int bestLen = 0;
    int k = 0;
    while (k < count) {
        int full = 1;
        while (full >= 0) {
            const char* nm = months != 0 ? __rt_mon_name(k, full) : __rt_wday_name(k, full);
            int len = (int)strlen(nm);
            int j = 0;
            unsafe {
                while (j < len && s[i + j] != (char)0 && tolower((int)s[i + j]) == tolower((int)nm[j])) { j = j + 1; }
            }
            if (j == len && len > bestLen) { best = k; bestLen = len; }
            full = full - 1;
        }
        k = k + 1;
    }
    unsafe { *next = i + bestLen; }
    return best;
}
int __rt_tp(const char* s, int i, const char* fmt, int* tm, int* have) {
    unsafe {
        int f = 0;
        while (fmt[f] != (char)0 && i >= 0) {
            char c = fmt[f];
            if (isspace((int)c) != 0) {
                while (isspace((int)s[i]) != 0) { i = i + 1; }
                f = f + 1;
                continue;
            }
            if (c != '%' || fmt[f + 1] == (char)0) {
                if (s[i] != c) { return -1; }
                i = i + 1;
                f = f + 1;
                continue;
            }
            char d = fmt[f + 1];
            f = f + 2;
            long long v = 0LL;
            if (d == 'Y') { i = __rt_tp_num(s, i, 4, 1, -9999LL, 9999LL, &v); if (i >= 0) { tm[5] = (int)(v - 1900LL); have[0] = 1; } }
            else if (d == 'y') {
                i = __rt_tp_num(s, i, 2, 0, 0LL, 99LL, &v);
                if (i >= 0) { tm[5] = (int)(v < 69LL ? v + 100LL : v); have[0] = 1; }
            }
            else if (d == 'm') { i = __rt_tp_num(s, i, 2, 0, 1LL, 12LL, &v); if (i >= 0) { tm[4] = (int)(v - 1LL); have[1] = 1; } }
            else if (d == 'd' || d == 'e') { i = __rt_tp_num(s, i, 2, 0, 1LL, 31LL, &v); if (i >= 0) { tm[3] = (int)v; have[2] = 1; } }
            else if (d == 'H' || d == 'k') { i = __rt_tp_num(s, i, 2, 0, 0LL, 23LL, &v); if (i >= 0) { tm[2] = (int)v; have[3] = 0; } }
            else if (d == 'I' || d == 'l') { i = __rt_tp_num(s, i, 2, 0, 1LL, 12LL, &v); if (i >= 0) { tm[2] = (int)(v % 12LL); have[3] = 1; } }
            else if (d == 'M') { i = __rt_tp_num(s, i, 2, 0, 0LL, 59LL, &v); if (i >= 0) { tm[1] = (int)v; } }
            else if (d == 'S') { i = __rt_tp_num(s, i, 2, 0, 0LL, 61LL, &v); if (i >= 0) { tm[0] = (int)v; } }
            else if (d == 'j') { i = __rt_tp_num(s, i, 3, 0, 1LL, 366LL, &v); if (i >= 0) { tm[7] = (int)(v - 1LL); have[5] = 1; } }
            else if (d == 'b' || d == 'B' || d == 'h') {
                while (isspace((int)s[i]) != 0) { i = i + 1; }
                int nx = i;
                int k = __rt_tp_name(s, i, 1, &nx);
                if (k < 0) { return -1; }
                tm[4] = k; have[1] = 1; i = nx;
            }
            else if (d == 'a' || d == 'A') {
                while (isspace((int)s[i]) != 0) { i = i + 1; }
                int nx = i;
                int k = __rt_tp_name(s, i, 0, &nx);
                if (k < 0) { return -1; }
                tm[6] = k; have[6] = 1; i = nx;
            }
            else if (d == 'p') {
                while (isspace((int)s[i]) != 0) { i = i + 1; }
                int a = tolower((int)s[i]);
                if ((a != 'a' && a != 'p') || tolower((int)s[i + 1]) != 'm') { return -1; }
                have[4] = a == 'p' ? 1 : 0;
                i = i + 2;
            }
            else if (d == 'n' || d == 't') { while (isspace((int)s[i]) != 0) { i = i + 1; } }
            else if (d == '%') { if (s[i] != '%') { return -1; } i = i + 1; }
            else if (d == 'T') { i = __rt_tp(s, i, "%H:%M:%S", tm, have); }
            else if (d == 'R') { i = __rt_tp(s, i, "%H:%M", tm, have); }
            else if (d == 'F') { i = __rt_tp(s, i, "%Y-%m-%d", tm, have); }
            else if (d == 'D') { i = __rt_tp(s, i, "%m/%d/%y", tm, have); }
            else { return -1; }
        }
    }
    return i;
}
// parses s by fmt into tm's fields; past the parsed text, or null. Like
// glibc, wday and yday are computed when the date fields allow it.
char* strptime(const char* s, const char* fmt, int* tm) {
    int have[7];   // year, mon, mday, 12-hour, pm, yday, wday
    int k = 0;
    while (k < 7) { have[k] = 0; k = k + 1; }
    int i = __rt_tp(s, 0, fmt, tm, &have[0]);
    if (i < 0) { return null; }
    unsafe {
        if (have[3] == 1 && have[4] == 1) { tm[2] = tm[2] + 12; }
        if (have[0] == 1 && have[1] == 1 && have[2] == 1) {
            long long y = (long long)tm[5] + 1900LL;
            long long days = __rt_days_from_civil(y, (long long)tm[4] + 1LL, (long long)tm[3]);
            if (have[6] == 0) { tm[6] = (int)__rt_fmod(days + 4LL, 7LL); }
            if (have[5] == 0) { tm[7] = (int)(days - __rt_days_from_civil(y, 1LL, 1LL)); }
        }
        return (char*)(s + i);
    }
}

// ── spawn/join (the playground has no threads: spawn runs the function
// to completion right away; join is then a no-op) ─────────────────────────────
