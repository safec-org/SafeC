// SafeC Standard Library — Strings
// Safe wrappers around libc string functions.
#pragma once
#include "str.h"

// ── Explicit extern declarations for libc string/memory functions ─────────────
extern unsigned long strlen(const char* s);
extern int   strcmp(const char* a, const char* b);
extern int   strncmp(const char* a, const char* b, unsigned long n);
extern char* strncpy(char* dst, const char* src, unsigned long n);
extern char* strstr(const char* haystack, const char* needle);
extern char* strchr(const char* s, int c);
extern char* strrchr(const char* s, int c);
extern char* strcat(char* dst, const char* src);
extern char* strncat(char* dst, const char* src, unsigned long n);
extern char* strpbrk(const char* s, const char* accept);
extern unsigned long strspn(const char* s, const char* accept);
extern unsigned long strcspn(const char* s, const char* reject);
extern char* strtok_r(char* s, const char* delim, char** saveptr);
extern void* memchr(const void* s, int c, unsigned long n);
extern void* malloc(unsigned long size);
extern void* memcpy(void* dst, const void* src, unsigned long n);

// Return the length (in bytes) of a null-terminated string.
unsigned long str_len(const char* s) {
    unsafe { return strlen(s); }
}

// Lexicographic comparison.  Returns <0, 0, or >0.
int str_cmp(const char* a, const char* b) {
    unsafe { return strcmp(a, b); }
}

// Lexicographic comparison of at most `n` bytes.
int str_ncmp(const char* a, const char* b, unsigned long n) {
    unsafe { return strncmp(a, b, n); }
}

// Return 1 if `a` and `b` are equal, 0 otherwise.
int str_eq(const char* a, const char* b) {
    unsafe { return strcmp(a, b) == 0; }
}

// Copy src into dst (at most `n` bytes including the NUL terminator).
// dst must have room for at least `n` bytes.
void str_copy(char* dst, const char* src, unsigned long n) {
    unsafe { strncpy(dst, src, n); }
}

// Return a pointer to the first occurrence of `needle` in `haystack`,
// or NULL if not found.
const char* str_find(const char* haystack, const char* needle) {
    unsafe { return strstr(haystack, needle); }
}

// Return a pointer to the first occurrence of byte `c` in `s`,
// or NULL if not found.
const char* str_find_char(const char* s, int c) {
    unsafe { return strchr(s, c); }
}

// Return a heap-allocated copy of `s`.  Caller must call dealloc() on it.
char* str_dup(const char* s) {
    unsafe {
        unsigned long len = strlen(s) + (unsigned long)1;
        char* buf = (char*)malloc(len);
        if (buf != (char*)0) {
            memcpy((void*)buf, (const void*)s, len);
        }
        return buf;
    }
}

// ── Concatenation ─────────────────────────────────────────────────────────────

// Append src to the NUL-terminated string at dst.  dst must have room.
void str_cat(char* dst, const char* src) {
    unsafe { strcat(dst, src); }
}

// Append at most `n` bytes of src to dst and always NUL-terminate.
void str_ncat(char* dst, const char* src, unsigned long n) {
    unsafe { strncat(dst, src, n); }
}

// ── Extended search ───────────────────────────────────────────────────────────

// Find last occurrence of character `c` in `s`.  Returns pointer or NULL.
const char* str_rfind_char(const char* s, int c) {
    unsafe { return strrchr(s, c); }
}

// Find first char in `s` that is also in `accept`.  Returns pointer or NULL.
const char* str_find_any(const char* s, const char* accept) {
    unsafe { return strpbrk(s, accept); }
}

// ── Span / classification ─────────────────────────────────────────────────────

// Length of initial segment of `s` with only chars in `accept`.
unsigned long str_span(const char* s, const char* accept) {
    unsafe { return strspn(s, accept); }
}

// Length of initial segment of `s` with no chars from `reject`.
unsigned long str_cspan(const char* s, const char* reject) {
    unsafe { return strcspn(s, reject); }
}

// ── Tokenisation ──────────────────────────────────────────────────────────────

// Reentrant tokeniser.  Pass s=NULL on subsequent calls.
// Modifies `s` in place by writing NUL terminators.
char* str_tok(char* s, const char* delim, &stack char* saveptr) {
    unsafe { return strtok_r(s, delim, (char**)saveptr); }
}

// ── Memory search ─────────────────────────────────────────────────────────────

// Return pointer to first byte == `c` in the `n`-byte block at `s`.
void* mem_chr(const void* s, int c, unsigned long n) {
    unsafe { return memchr(s, c, n); }
}

// ── Extended allocation ───────────────────────────────────────────────────────

// Return heap copy of first `n` bytes of `s`, NUL-terminated.
char* str_ndup(const char* s, unsigned long n) {
    unsafe {
        char* buf = (char*)malloc(n + (unsigned long)1);
        if (buf != (char*)0) {
            memcpy((void*)buf, (const void*)s, n);
            buf[n] = '\0';
        }
        return buf;
    }
}
