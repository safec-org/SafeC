// SafeC Standard Library — String and memory declarations (C11/C17/C23 coverage)
#pragma once

// ── Length / comparison ───────────────────────────────────────────────────────
unsigned long str_len(const char* s);
int           str_cmp(const char* a, const char* b);
int           str_ncmp(const char* a, const char* b, unsigned long n);
int           str_eq(const char* a, const char* b);

// ── Copy ──────────────────────────────────────────────────────────────────────
// Copy src into dst — at most `n` bytes including NUL.  dst must hold ≥ n bytes.
void          str_copy(char* dst, const char* src, unsigned long n);

// ── Concatenation ─────────────────────────────────────────────────────────────
// Append src to the NUL-terminated string at dst.  dst must have room.
void          str_cat(char* dst, const char* src);

// Append at most `n` bytes of src to dst and always NUL-terminate.
void          str_ncat(char* dst, const char* src, unsigned long n);

// ── Search ────────────────────────────────────────────────────────────────────
// Find first occurrence of needle in haystack.  Returns pointer or NULL.
const char*   str_find(const char* haystack, const char* needle);

// Find first occurrence of character `c` in `s`.  Returns pointer or NULL.
const char*   str_find_char(const char* s, int c);

// Find last occurrence of character `c` in `s`.  Returns pointer or NULL.
const char*   str_rfind_char(const char* s, int c);

// Find first character in `s` that is also in `accept`.  Returns pointer or NULL.
const char*   str_find_any(const char* s, const char* accept);

// ── Span / classification ─────────────────────────────────────────────────────
// Return length of initial segment of `s` consisting only of chars in `accept`.
unsigned long str_span(const char* s, const char* accept);

// Return length of initial segment of `s` with no chars from `reject`.
unsigned long str_cspan(const char* s, const char* reject);

// ── Tokenisation ─────────────────────────────────────────────────────────────
// Reentrant tokeniser (wraps strtok_r / strtok_s).
// On first call: pass the string to tokenise as `s`, any non-NULL char** for `saveptr`.
// Subsequent calls: pass NULL as `s`.
// Returns pointer to next token, or NULL when exhausted.
// The input string IS modified (NUL-terminators are written).
char*         str_tok(char* s, const char* delim, char** saveptr);

// ── Memory search ─────────────────────────────────────────────────────────────
// Return pointer to first occurrence of byte `c` in the `n`-byte block at `s`.
void*         mem_chr(const void* s, int c, unsigned long n);

// ── Allocation / duplication ──────────────────────────────────────────────────
// Return a heap-allocated NUL-terminated copy of `s`.  Caller must call dealloc().
char*         str_dup(const char* s);

// Return a heap-allocated copy of the first `n` bytes of `s`, NUL-terminated.
char*         str_ndup(const char* s, unsigned long n);
