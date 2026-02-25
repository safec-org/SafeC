// SafeC Standard Library — Type conversions (C11/C17 string ↔ number)
// All parse functions set *ok = 1 on success, 0 on failure (NULL ok is safe).
#pragma once

// ── String → number ───────────────────────────────────────────────────────────

// Parse a decimal integer.  On failure *ok = 0 and returns 0.
long long    str_to_int(const char* s, int* ok);

// Parse an unsigned decimal integer.
unsigned long long str_to_uint(const char* s, int* ok);

// Parse a hex integer (with or without "0x" prefix).
unsigned long long str_to_hex(const char* s, int* ok);

// Parse a floating-point number.
double       str_to_float(const char* s, int* ok);

// ── Number → heap string ──────────────────────────────────────────────────────
// Returned pointer is heap-allocated; caller must call dealloc() on it.

char* int_to_str(long long v);
char* uint_to_str(unsigned long long v);
char* float_to_str(double v, int decimals);

// ── Numeric conversions ───────────────────────────────────────────────────────

// Safe narrowing: return 1 and set *out if v fits in an int, else 0.
int  ll_to_int(long long v, int* out);

// Clamp a double to [min, max], then round to nearest long long.
long long float_clamp_to_ll(double v, long long min, long long max);

// Return 1 if the string is a valid decimal integer representation.
int  str_is_int(const char* s);

// Return 1 if the string is a valid floating-point representation.
int  str_is_float(const char* s);
