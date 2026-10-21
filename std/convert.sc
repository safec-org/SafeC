// SafeC Standard Library — Type conversions implementation
#pragma once
#include <std/stddef.h>
#include <std/convert.h>
#include <std/mem.h>
#include <std/fmt.h>

// ── Explicit extern declarations for libc functions ───────────────────────────
// '#include <stdlib.h>' etc. aren't usable here: this compiler's preprocessor
// doesn't parse the system headers' macro-heavy declarations, so nothing was
// ever actually declared. Declare the exact symbols used below instead, same
// as mem.sc/str.sc/math.sc do for their libc dependencies.
namespace std {

#ifndef __SAFEC_FREESTANDING__
extern double          strtod(const char* s, char** endptr);
extern int             snprintf(char* buf, size_t n, const char* fmt, ...);
#endif

#define INT_MIN (-2147483647 - 1)
#define INT_MAX 2147483647

// Integer parsing is native (no strtoll/errno): the whole string must be an
// optionally-signed number in 'base', with no overflow; anything else fails.
static int conv_digit_(char c, unsigned int base) {
    unsigned int d = 99u;
    if (c >= '0' && c <= '9') { d = (unsigned int)(c - '0'); }
    else if (c >= 'a' && c <= 'f') { d = (unsigned int)(c - 'a') + 10u; }
    else if (c >= 'A' && c <= 'F') { d = (unsigned int)(c - 'A') + 10u; }
    return d < base ? (int)d : -1;
}

// Parses s[i..] as an unsigned magnitude; sets *ok, returns the value.
static unsigned long long conv_parse_u_(const &char s, unsigned long i, unsigned int base,
                                        unsigned long long limit, &int ok) {
    unsigned long long v = 0ULL;
    int any = 0;
    *ok = 0;
    unsafe {
        while (s[i] != (char)0) {
            int d = conv_digit_(s[i], base);
            if (d < 0) { return 0ULL; }
            if (v > (limit - (unsigned long long)d) / (unsigned long long)base) { return 0ULL; }
            v = v * (unsigned long long)base + (unsigned long long)d;
            any = 1;
            i = i + 1UL;
        }
    }
    *ok = any;
    return v;
}

inline long long str_to_int(const &char s, ?&int ok) {
    int good = 0;
    long long out = 0LL;
    unsafe {
        int neg = s[0] == '-';
        unsigned long start = (s[0] == '-' || s[0] == '+') ? 1UL : 0UL;
        unsigned long long limit = neg ? 9223372036854775808ULL : 9223372036854775807ULL;
        unsigned long long mag = conv_parse_u_(s, start, 10u, limit, (&int)&good);
        if (good) { out = neg ? (long long)(0ULL - mag) : (long long)mag; }
        if (ok != (int*)0) *ok = good;
    }
    return out;
}

inline unsigned long long str_to_uint(const &char s, ?&int ok) {
    int good = 0;
    unsigned long long v = 0ULL;
    unsafe {
        unsigned long start = s[0] == '+' ? 1UL : 0UL;
        v = conv_parse_u_(s, start, 10u, 18446744073709551615ULL, (&int)&good);
        if (ok != (int*)0) *ok = good;
    }
    return good ? v : 0ULL;
}

inline unsigned long long str_to_hex(const &char s, ?&int ok) {
    int good = 0;
    unsigned long long v = 0ULL;
    unsafe {
        unsigned long start = (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) ? 2UL : 0UL;
        v = conv_parse_u_(s, start, 16u, 18446744073709551615ULL, (&int)&good);
        if (ok != (int*)0) *ok = good;
    }
    return good ? v : 0ULL;
}

inline double str_to_float(const &char s, ?&int ok) {
#ifndef __SAFEC_FREESTANDING__
    unsafe {
        char* end = (char*)0;
        double v = strtod(s, (char**)&end);
        int success = (end != s && *end == '\0');
        if (ok != (int*)0) *ok = success;
        return success ? v : 0.0;
    }
#else
    // [+-]digits[.digits][(e|E)[+-]digits], accumulated in double (not
    // correctly rounded in every case like strtod; fine for config/JSON).
    double v = 0.0;
    int good = 0;
    unsafe {
        unsigned long i = 0UL;
        int neg = s[0] == '-';
        if (s[0] == '-' || s[0] == '+') { i = 1UL; }
        while (s[i] >= '0' && s[i] <= '9') { v = v * 10.0 + (double)(s[i] - '0'); i = i + 1UL; good = 1; }
        if (s[i] == '.') {
            i = i + 1UL;
            double place = 0.1;
            while (s[i] >= '0' && s[i] <= '9') { v = v + place * (double)(s[i] - '0'); place = place / 10.0; i = i + 1UL; good = 1; }
        }
        if (good && (s[i] == 'e' || s[i] == 'E')) {
            i = i + 1UL;
            int eneg = s[i] == '-';
            if (s[i] == '-' || s[i] == '+') { i = i + 1UL; }
            int e = 0;
            int anyE = 0;
            while (s[i] >= '0' && s[i] <= '9' && e < 400) { e = e * 10 + (int)(s[i] - '0'); i = i + 1UL; anyE = 1; }
            if (!anyE) { good = 0; }
            while (e > 0) { v = eneg ? v / 10.0 : v * 10.0; e = e - 1; }
        }
        if (s[i] != (char)0) { good = 0; }
        if (neg) { v = -v; }
        if (ok != (int*)0) *ok = good;
    }
    return good ? v : 0.0;
#endif
}

inline ?&heap char int_to_str(long long v) {
    unsafe {
        // -9223372036854775808 is 20 digits + sign + NUL = 22 bytes
        char* buf = (char*)alloc_raw_((unsigned long)22);
        if (buf != (char*)0) fmt_int(buf, (unsigned long)22, v);
        return buf;
    }
}

inline ?&heap char uint_to_str(unsigned long long v) {
    unsafe {
        char* buf = (char*)alloc_raw_((unsigned long)21);
        if (buf != (char*)0) fmt_uint(buf, (unsigned long)21, v);
        return buf;
    }
}

inline ?&heap char float_to_str(double v, int decimals) {
    unsafe {
        char* buf = (char*)alloc_raw_((unsigned long)64);
        if (buf != (char*)0) fmt_float(buf, (unsigned long)64, v, decimals);
        return buf;
    }
}

inline int ll_to_int(long long v, &int out) {
    if (v < (long long)INT_MIN || v > (long long)INT_MAX) return 0;
    unsafe { *out = (int)v; }
    return 1;
}

inline const long long float_clamp_to_ll(double v, long long min, long long max) {
    if (v != v) return min;                 // NaN
    if (v < (double)min) return min;
    // '>=': (double)max can round up past max (2^63 for LLONG_MAX), and
    // converting that back to long long is undefined.
    if (v >= (double)max) return max;
    // Round half away from zero (the cast alone truncates).
    return (long long)(v < 0.0 ? v - 0.5 : v + 0.5);
}

inline int str_is_int(const &char s) {
    int ok = 0;
    unsafe { str_to_int(s, (int*)&ok); }
    return ok;
}

inline int str_is_float(const &char s) {
    int ok = 0;
    unsafe { str_to_float(s, (int*)&ok); }
    return ok;
}

} // namespace std
