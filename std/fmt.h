// SafeC Standard Library — String formatting
// Safe snprintf-based builders that write into caller-supplied buffers.
#pragma once

// Format into buf[0..cap-1]; always NUL-terminates.
// Returns number of characters written (not counting NUL), or -1 on error.
int  fmt_int(char* buf, unsigned long cap, long long v);
int  fmt_uint(char* buf, unsigned long cap, unsigned long long v);
int  fmt_float(char* buf, unsigned long cap, double v, int decimals);
int  fmt_hex(char* buf, unsigned long cap, unsigned long long v);
int  fmt_hex_upper(char* buf, unsigned long cap, unsigned long long v);
int  fmt_bool(char* buf, unsigned long cap, int v);

// Copy src into dst (at most cap-1 bytes), NUL-terminate.
// Returns number of characters copied.
int  fmt_str(char* dst, unsigned long cap, const char* src);

// Concatenate src onto dst (treats dst as a NUL-terminated string).
// Returns total length after append, or -1 if cap was exceeded.
int  fmt_append(char* dst, unsigned long cap, const char* src);
