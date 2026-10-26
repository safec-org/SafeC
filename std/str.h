// SafeC Standard Library — String and memory declarations (C11/C17/C23 coverage)
#pragma once

// ── Length / comparison ───────────────────────────────────────────────────────
namespace std {

unsigned long str_len(const &char s);
// The bytes of a NUL-terminated string (without the NUL) as a slice — the
// way to hand a literal or C string to an API taking []const char /
// []const unsigned char:  base64_encode(str_bytes("foobar"))
[]const char  str_bytes(const &char s);
int           str_cmp(const &char a, const &char b);
int           str_ncmp(const &char a, const &char b, unsigned long n);
int           str_eq(const &char a, const &char b);

// ── Copy ──────────────────────────────────────────────────────────────────────
// Copy src into dst — at most dst.len-1 bytes, then a NUL (when dst is not
// empty). Returns 1 when all of src fit, 0 when it was truncated.
int           str_copy([]char dst, const &char src);

// ── Concatenation ─────────────────────────────────────────────────────────────
// Append src to the NUL-terminated string in dst, within dst.len (always
// NUL-terminated). Returns 1 when all of src fit, 0 when it was truncated
// (or dst held no NUL).
int           str_cat([]char dst, const &char src);

// Append at most `n` bytes of src to the string in dst, within dst.len, and
// NUL-terminate. Returns 1 when those bytes fit, 0 when truncated.
int           str_ncat([]char dst, const &char src, unsigned long n);

// ── Search ────────────────────────────────────────────────────────────────────
// Find first occurrence of needle in haystack.  Returns pointer or NULL.
const ?&char  str_find(const &char haystack, const &char needle);

// Find first occurrence of character `c` in `s`.  Returns pointer or NULL.
const ?&char  str_find_char(const &char s, int c);

// Find last occurrence of character `c` in `s`.  Returns pointer or NULL.
const ?&char  str_rfind_char(const &char s, int c);

// Find first character in `s` that is also in `accept`.  Returns pointer or NULL.
const ?&char  str_find_any(const &char s, const &char accept);

// ── Span / classification ─────────────────────────────────────────────────────
// Return length of initial segment of `s` consisting only of chars in `accept`.
unsigned long str_span(const &char s, const &char accept);

// Return length of initial segment of `s` with no chars from `reject`.
unsigned long str_cspan(const &char s, const &char reject);

// ── Tokenisation ─────────────────────────────────────────────────────────────
// Reentrant tokeniser (wraps strtok_r / strtok_s).
// On first call: pass the string to tokenise as `s` and a '?&char' cursor
// for `saveptr`. Subsequent calls: pass null as `s`.
// Returns the next token, or null when exhausted.
// The input string IS modified (NUL-terminators are written).
?&char        str_tok(?&char s, const &char delim, &?&char saveptr);

// ── Memory search ─────────────────────────────────────────────────────────────
// The first byte == `c` in `s`, or null.
const ?&unsigned char mem_chr([]const unsigned char s, int c);

// ── Allocation / duplication ──────────────────────────────────────────────────
// Return a heap-allocated NUL-terminated copy of `s`.  Caller must call dealloc().
?&heap char   str_dup(const &char s);

// Return a heap-allocated copy of at most the first `n` bytes of `s` (stopping
// at its terminator), NUL-terminated.
?&heap char   str_ndup(const &char s, unsigned long n);

} // namespace std
