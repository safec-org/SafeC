// SafeC Standard Library — Type conversions implementation
#pragma once
#include <std/convert.h>

// ── Explicit extern declarations for libc functions ───────────────────────────
// '#include <stdlib.h>' etc. aren't usable here: this compiler's preprocessor
// doesn't parse the system headers' macro-heavy declarations, so nothing was
// ever actually declared. Declare the exact symbols used below instead, same
// as mem.sc/str.sc/math.sc do for their libc dependencies.
namespace std {

extern long long      strtoll(const char* s, char** endptr, int base);
extern unsigned long long strtoull(const char* s, char** endptr, int base);
extern double          strtod(const char* s, char** endptr);
extern void*           malloc(unsigned long size);
extern int             snprintf(char* buf, unsigned long n, const char* fmt, ...);

// errno: macOS's libc exposes it via a thread-local accessor, not a plain
// extern global (unlike glibc's 'extern int errno'). '__error()' is the
// documented macOS entry point that the real <errno.h> macro expands to.
extern int* __error(void);
#define errno (*__error())

#define INT_MIN (-2147483647 - 1)
#define INT_MAX 2147483647

inline long long str_to_int(const char* s, int* ok) {
    unsafe {
        char* end = (char*)0;
        errno = 0;
        long long v = strtoll(s, (char**)&end, 10);
        int success = (end != s && *end == '\0' && errno == 0);
        if (ok != (int*)0) *ok = success;
        return success ? v : (long long)0;
    }
}

inline unsigned long long str_to_uint(const char* s, int* ok) {
    unsafe {
        char* end = (char*)0;
        errno = 0;
        unsigned long long v = strtoull(s, (char**)&end, 10);
        int success = (end != s && *end == '\0' && errno == 0);
        if (ok != (int*)0) *ok = success;
        return success ? v : (unsigned long long)0;
    }
}

inline unsigned long long str_to_hex(const char* s, int* ok) {
    unsafe {
        char* end = (char*)0;
        errno = 0;
        unsigned long long v = strtoull(s, (char**)&end, 16);
        int success = (end != s && *end == '\0' && errno == 0);
        if (ok != (int*)0) *ok = success;
        return success ? v : (unsigned long long)0;
    }
}

inline double str_to_float(const char* s, int* ok) {
    unsafe {
        char* end = (char*)0;
        errno = 0;
        double v = strtod(s, (char**)&end);
        int success = (end != s && *end == '\0' && errno == 0);
        if (ok != (int*)0) *ok = success;
        return success ? v : 0.0;
    }
}

inline char* int_to_str(long long v) {
    unsafe {
        // -9223372036854775808 is 20 digits + sign + NUL = 22 bytes
        char* buf = (char*)malloc((unsigned long)22);
        if (buf != (char*)0) snprintf(buf, (unsigned long)22, "%lld", v);
        return buf;
    }
}

inline char* uint_to_str(unsigned long long v) {
    unsafe {
        char* buf = (char*)malloc((unsigned long)21);
        if (buf != (char*)0) snprintf(buf, (unsigned long)21, "%llu", v);
        return buf;
    }
}

inline char* float_to_str(double v, int decimals) {
    unsafe {
        char* buf = (char*)malloc((unsigned long)64);
        if (buf != (char*)0) snprintf(buf, (unsigned long)64, "%.*f", decimals, v);
        return buf;
    }
}

inline int ll_to_int(long long v, int* out) {
    if (v < (long long)INT_MIN || v > (long long)INT_MAX) return 0;
    unsafe { *out = (int)v; }
    return 1;
}

inline const long long float_clamp_to_ll(double v, long long min, long long max) {
    if (v < (double)min) return min;
    if (v > (double)max) return max;
    return (long long)v;
}

inline int str_is_int(const char* s) {
    int ok = 0;
    unsafe { str_to_int(s, (int*)&ok); }
    return ok;
}

inline int str_is_float(const char* s) {
    int ok = 0;
    unsafe { str_to_float(s, (int*)&ok); }
    return ok;
}

} // namespace std
