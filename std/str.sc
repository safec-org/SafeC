// SafeC Standard Library — Strings
// Safe wrappers around libc string functions.
#pragma once
#include <std/stddef.h>
#include <std/str.h>
#include <std/mem.h>

// ── Explicit extern declarations for libc string/memory functions ─────────────
namespace std {

#ifndef __SAFEC_FREESTANDING__
extern size_t strlen(const char* s);
extern int   strcmp(const char* a, const char* b);
extern int   strncmp(const char* a, const char* b, size_t n);
extern char* strncpy(char* dst, const char* src, size_t n);
extern char* strstr(const char* haystack, const char* needle);
extern char* strchr(const char* s, int c);
extern char* strrchr(const char* s, int c);
extern char* strcat(char* dst, const char* src);
extern char* strncat(char* dst, const char* src, size_t n);
extern char* strpbrk(const char* s, const char* accept);
extern size_t strspn(const char* s, const char* accept);
extern size_t strcspn(const char* s, const char* reject);
// MSVC's CRT has no 'strtok_r' — 'strtok_s' is the same reentrant
// tokenizer under a different name, same (str, delim, saveptr) shape.
#ifdef _WIN32
extern char* strtok_s(char* s, const char* delim, char** saveptr);
#else
extern char* strtok_r(char* s, const char* delim, char** saveptr);
#endif
extern void* memchr(const void* s, int c, size_t n);
#else
// Freestanding: no libc, so the same entry points are implemented natively
// (file-private; the str_* wrappers below are the public API either way).
static unsigned long strlen(const &char s) {
    unsigned long n = 0UL;
    unsafe { while (s[n] != (char)0) { n = n + 1UL; } }
    return n;
}
static int strncmp(const &char a, const &char b, unsigned long n) {
    unsafe {
        unsigned long i = 0UL;
        while (i < n) {
            unsigned char x = (unsigned char)a[i];
            unsigned char y = (unsigned char)b[i];
            if (x != y) { return (int)x - (int)y; }
            if (x == (unsigned char)0) { return 0; }
            i = i + 1UL;
        }
    }
    return 0;
}
static int strcmp(const &char a, const &char b) { return strncmp(a, b, ~0UL); }
static ?&char strncpy(&char dst, const &char src, unsigned long n) {
    unsafe {
        unsigned long i = 0UL;
        while (i < n && src[i] != (char)0) { dst[i] = src[i]; i = i + 1UL; }
        while (i < n) { dst[i] = (char)0; i = i + 1UL; }
    }
    return dst;
}
static ?&char strchr(const &char s, int c) {
    unsafe {
        unsigned long i = 0UL;
        while (1) {
            if (s[i] == (char)c) { return (?&char)&s[i]; }
            if (s[i] == (char)0) { return null; }
            i = i + 1UL;
        }
    }
    return null;
}
static ?&char strrchr(const &char s, int c) {
    ?&char last = null;
    unsafe {
        unsigned long i = 0UL;
        while (1) {
            if (s[i] == (char)c) { last = (?&char)&s[i]; }
            if (s[i] == (char)0) { return last; }
            i = i + 1UL;
        }
    }
    return last;
}
static ?&char strstr(const &char h, const &char nd) {
    unsigned long m = strlen(nd);
    unsafe {
        unsigned long i = 0UL;
        while (h[i] != (char)0 || m == 0UL) {
            if (strncmp(&h[i], nd, m) == 0) { return (?&char)&h[i]; }
            if (h[i] == (char)0) { return null; }
            i = i + 1UL;
        }
    }
    return null;
}
static unsigned long str_span_impl_(const &char s, const &char set, int inSet) {
    unsigned long i = 0UL;
    unsafe {
        while (s[i] != (char)0) {
            int found = strchr(set, (int)s[i]).is_null() ? 0 : 1;
            if (found != inSet) { return i; }
            i = i + 1UL;
        }
    }
    return i;
}
static unsigned long strspn(const &char s, const &char accept) { return str_span_impl_(s, accept, 1); }
static unsigned long strcspn(const &char s, const &char reject) { return str_span_impl_(s, reject, 0); }
static ?&char strpbrk(const &char s, const &char accept) {
    unsigned long i = strcspn(s, accept);
    unsafe {
        if (s[i] == (char)0) { return null; }
        return (?&char)&s[i];
    }
}
static ?&char strncat(&char dst, const &char src, unsigned long n) {
    unsigned long d = strlen(dst);
    unsafe {
        unsigned long i = 0UL;
        while (i < n && src[i] != (char)0) { dst[d + i] = src[i]; i = i + 1UL; }
        dst[d + i] = (char)0;
    }
    return dst;
}
static ?&char strcat(&char dst, const &char src) { return strncat(dst, src, ~0UL); }
static ?&char strtok_r(?&char s, const &char delim, &?&char saveptr) {
    ?&char p = s.is_null() ? *saveptr : s;
    match (p) {
        case null: return null;
        case some(q): { return str_tok_from_(q, delim, saveptr); }
    }
    return null;
}
static const ?&unsigned char memchr(const &unsigned char b, int c, unsigned long n) {
    unsafe {
        unsigned long i = 0UL;
        while (i < n) {
            if (b[i] == (unsigned char)c) { return (const ?&unsigned char)&b[i]; }
            i = i + 1UL;
        }
    }
    return null;
}
#endif
extern void* memcpy(void* dst, const void* src, size_t n);

// Return the length (in bytes) of a null-terminated string.
inline unsigned long str_len(const &char s) {
    unsafe { return strlen(s); }
}

// Lexicographic comparison.  Returns <0, 0, or >0.
inline int str_cmp(const &char a, const &char b) {
    unsafe { return strcmp(a, b); }
}

// Lexicographic comparison of at most `n` bytes.
inline int str_ncmp(const &char a, const &char b, unsigned long n) {
    unsafe { return strncmp(a, b, n); }
}

// Return 1 if `a` and `b` are equal, 0 otherwise.
inline int str_eq(const &char a, const &char b) {
    unsafe { return strcmp(a, b) == 0; }
}

// Copy src into dst (at most `n` bytes including the NUL terminator).
// dst must have room for at least `n` bytes.
inline void str_copy(&char dst, const &char src, unsigned long n) {
    unsafe { strncpy(dst, src, n); }
}

// Return a pointer to the first occurrence of `needle` in `haystack`,
// or NULL if not found.
inline const ?&char str_find(const &char haystack, const &char needle) {
    unsafe { return strstr(haystack, needle); }
}

// Return a pointer to the first occurrence of byte `c` in `s`,
// or NULL if not found.
inline const ?&char str_find_char(const &char s, int c) {
    unsafe { return strchr(s, c); }
}

// Return a heap-allocated copy of `s`.  Caller must call dealloc() on it.
inline ?&heap char str_dup(const &char s) {
    unsafe {
        unsigned long len = strlen(s) + (unsigned long)1;
        char* buf = (char*)alloc_raw_(len);
        if (buf != (char*)0) {
            memcpy((void*)buf, (const void*)s, len);
        }
        return buf;
    }
}

// ── Concatenation ─────────────────────────────────────────────────────────────

// Append src to the NUL-terminated string at dst.  dst must have room.
inline void str_cat(&char dst, const &char src) {
    unsafe { strcat(dst, src); }
}

// Append at most `n` bytes of src to dst and always NUL-terminate.
inline void str_ncat(&char dst, const &char src, unsigned long n) {
    unsafe { strncat(dst, src, n); }
}

// ── Extended search ───────────────────────────────────────────────────────────

// Find last occurrence of character `c` in `s`.  Returns pointer or NULL.
inline const ?&char str_rfind_char(const &char s, int c) {
    unsafe { return strrchr(s, c); }
}

// Find first char in `s` that is also in `accept`.  Returns pointer or NULL.
inline const ?&char str_find_any(const &char s, const &char accept) {
    unsafe { return strpbrk(s, accept); }
}

// ── Span / classification ─────────────────────────────────────────────────────

// Length of initial segment of `s` with only chars in `accept`.
inline unsigned long str_span(const &char s, const &char accept) {
    unsafe { return strspn(s, accept); }
}

// Length of initial segment of `s` with no chars from `reject`.
inline unsigned long str_cspan(const &char s, const &char reject) {
    unsafe { return strcspn(s, reject); }
}

// ── Tokenisation ──────────────────────────────────────────────────────────────

// Reentrant tokeniser.  Pass s=NULL on subsequent calls.
// Modifies `s` in place by writing NUL terminators.
static ?&char str_tok_from_(&char q, const &char delim, &?&char saveptr) {
    unsafe {
        unsigned long start = strspn(q, delim);
        if (q[start] == (char)0) { *saveptr = null; return null; }
        unsigned long len = strcspn(&q[start], delim);
        if (q[start + len] == (char)0) {
            *saveptr = null;
        } else {
            q[start + len] = (char)0;
            *saveptr = (?&char)&q[start + len + 1UL];
        }
        return (?&char)&q[start];
    }
}

inline ?&char str_tok(?&char s, const &char delim, &?&char saveptr) {
    ?&char p = s.is_null() ? *saveptr : s;
    match (p) {
        case null: return null;
        case some(q): { return str_tok_from_(q, delim, saveptr); }
    }
    return null;
}

// ── Memory search ─────────────────────────────────────────────────────────────

// Return pointer to first byte == `c` in the `n`-byte block at `s`.
inline const ?&unsigned char mem_chr(const &unsigned char s, int c, unsigned long n) {
    unsafe { return memchr(s, c, n); }
}

// ── Extended allocation ───────────────────────────────────────────────────────

// Return heap copy of first `n` bytes of `s`, NUL-terminated.
inline ?&heap char str_ndup(const &char s, unsigned long n) {
    unsafe {
        // Stop at the terminator: s may be shorter than n.
        unsigned long len = 0UL;
        while (len < n && s[len] != (char)0) { len = len + 1UL; }
        n = len;
        char* buf = (char*)alloc_raw_(n + (unsigned long)1);
        if (buf != (char*)0) {
            memcpy((void*)buf, (const void*)s, n);
            buf[n] = '\0';
        }
        return buf;
    }
}

} // namespace std
