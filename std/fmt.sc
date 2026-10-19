// SafeC Standard Library — String formatting implementation
#pragma once
#include <std/stddef.h>
#include <std/fmt.h>

namespace std {

#ifndef __SAFEC_FREESTANDING__
extern int            snprintf(char* buf, size_t n, const char* fmt, ...);
#endif
extern size_t  strlen(const char* s);
extern void*          memcpy(void* dst, const void* src, size_t n);

// Integer formatting is done here directly rather than through snprintf:
// it's on the hot path of String.push_int/JSON/HTTP, where parsing a format
// string per call dominated, and it must work with no libc at all. Same
// contract as snprintf: writes at most cap-1 chars + NUL, returns the full
// length the value needs.
static int fmt_emit_(&char buf, unsigned long cap, const &char digits, int n) {
    if (cap > 0UL) {
        unsigned long w = (unsigned long)n < cap ? (unsigned long)n : cap - 1UL;
        unsafe {
            unsigned long i = 0UL;
            while (i < w) { buf[i] = digits[i]; i = i + 1UL; }
            buf[w] = (char)0;
        }
    }
    return n;
}

static int fmt_u64_(&char buf, unsigned long cap, unsigned long long v,
                    unsigned int base, int upper, int negative) {
    char tmp[24];
    int n = 0;
    const &char dig = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    unsafe {
        do {
            tmp[n] = dig[v % (unsigned long long)base];
            v = v / (unsigned long long)base;
            n = n + 1;
        } while (v != 0ULL);
        char out[24];
        int k = 0;
        if (negative) { out[0] = '-'; k = 1; }
        while (n > 0) { n = n - 1; out[k] = tmp[n]; k = k + 1; }
        return fmt_emit_(buf, cap, out, k);
    }
}

inline int fmt_int(char* buf, unsigned long cap, long long v) {
    // Negate in unsigned arithmetic so LLONG_MIN doesn't overflow.
    unsigned long long mag = v < 0LL ? 0ULL - (unsigned long long)v : (unsigned long long)v;
    return fmt_u64_(buf, cap, mag, 10u, 0, v < 0LL);
}

inline int fmt_uint(char* buf, unsigned long cap, unsigned long long v) {
    return fmt_u64_(buf, cap, v, 10u, 0, 0);
}

inline int fmt_hex(char* buf, unsigned long cap, unsigned long long v) {
    return fmt_u64_(buf, cap, v, 16u, 0, 0);
}

inline int fmt_hex_upper(char* buf, unsigned long cap, unsigned long long v) {
    return fmt_u64_(buf, cap, v, 16u, 1, 0);
}

inline int fmt_bool(char* buf, unsigned long cap, int v) {
    return v ? fmt_emit_(buf, cap, "true", 4) : fmt_emit_(buf, cap, "false", 5);
}

inline int fmt_float(char* buf, unsigned long cap, double v, int decimals) {
#ifndef __SAFEC_FREESTANDING__
    unsafe { return snprintf(buf, cap, "%.*f", decimals, v); }
#else
    // Fixed-point rendering without libc: round to 'decimals' places, then
    // print integer and fraction parts. Exact for magnitudes below 2^63 /
    // 10^decimals; not correctly rounded in every last-digit case like a
    // full printf, but deterministic.
    if (v != v) { return fmt_emit_(buf, cap, "nan", 3); }
    if (v > 1.7976931348623157e308) { return fmt_emit_(buf, cap, "inf", 3); }
    if (v < -1.7976931348623157e308) { return fmt_emit_(buf, cap, "-inf", 4); }
    if (decimals < 0) { decimals = 0; }
    if (decimals > 17) { decimals = 17; }
    int neg = v < 0.0;
    double a = neg ? -v : v;
    double scale = 1.0;
    int i = 0;
    while (i < decimals) { scale = scale * 10.0; i = i + 1; }
    unsigned long long whole = (unsigned long long)a;
    unsigned long long frac = (unsigned long long)((a - (double)whole) * scale + 0.5);
    if ((double)frac >= scale) { whole = whole + 1ULL; frac = 0ULL; }
    char out[48];
    int n = fmt_u64_(out, 48UL, whole, 10u, 0, neg);
    if (decimals > 0) {
        unsafe {
            out[n] = '.';
            int d = decimals;
            while (d > 0) {
                d = d - 1;
                out[n + 1 + d] = (char)('0' + (int)(frac % 10ULL));
                frac = frac / 10ULL;
            }
        }
        n = n + 1 + decimals;
    }
    return fmt_emit_(buf, cap, out, n);
#endif
}

inline int fmt_str(char* dst, unsigned long cap, const char* src) {
    if (cap == (unsigned long)0) return 0;
    unsafe {
        unsigned long n = strlen(src);
        if (n >= cap) n = cap - (unsigned long)1;
        memcpy(dst, src, n);
        dst[n] = '\0';
        return (int)n;
    }
}

inline int fmt_append(char* dst, unsigned long cap, const char* src) {
    unsafe {
        unsigned long dlen = strlen(dst);
        unsigned long slen = strlen(src);
        if (dlen + slen + (unsigned long)1 > cap) return -1;
        memcpy(dst + dlen, src, slen + (unsigned long)1);
        return (int)(dlen + slen);
    }
}

} // namespace std
