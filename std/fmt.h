// SafeC Standard Library — String formatting
// Safe snprintf-based builders that write into caller-supplied buffers.
#pragma once

// Format into buf (at most buf.len-1 characters); always NUL-terminates
// when buf is not empty.
// Returns number of characters written (not counting NUL), or -1 on error.
namespace std {

int  fmt_int([]char buf, long long v);
int  fmt_uint([]char buf, unsigned long long v);
int  fmt_float([]char buf, double v, int decimals);
int  fmt_hex([]char buf, unsigned long long v);
int  fmt_hex_upper([]char buf, unsigned long long v);
int  fmt_bool([]char buf, int v);

// Copy src into dst (at most dst.len-1 bytes), NUL-terminate.
// Returns number of characters copied.
int  fmt_str([]char dst, const &char src);

// Concatenate src onto dst (treats dst as a NUL-terminated string).
// Returns total length after append, or -1 if dst.len was exceeded.
int  fmt_append([]char dst, const &char src);

} // namespace std
