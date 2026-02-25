// SafeC Standard Library — Type conversions implementation
#pragma once
#include "convert.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <limits.h>

long long str_to_int(const char* s, int* ok) {
    unsafe {
        char* end = (char*)0;
        errno = 0;
        long long v = strtoll(s, &end, 10);
        int success = (end != s && *end == '\0' && errno == 0);
        if (ok != (int*)0) *ok = success;
        return success ? v : 0;
    }
}

unsigned long long str_to_uint(const char* s, int* ok) {
    unsafe {
        char* end = (char*)0;
        errno = 0;
        unsigned long long v = strtoull(s, &end, 10);
        int success = (end != s && *end == '\0' && errno == 0);
        if (ok != (int*)0) *ok = success;
        return success ? v : (unsigned long long)0;
    }
}

unsigned long long str_to_hex(const char* s, int* ok) {
    unsafe {
        char* end = (char*)0;
        errno = 0;
        unsigned long long v = strtoull(s, &end, 16);
        int success = (end != s && *end == '\0' && errno == 0);
        if (ok != (int*)0) *ok = success;
        return success ? v : (unsigned long long)0;
    }
}

double str_to_float(const char* s, int* ok) {
    unsafe {
        char* end = (char*)0;
        errno = 0;
        double v = strtod(s, &end);
        int success = (end != s && *end == '\0' && errno == 0);
        if (ok != (int*)0) *ok = success;
        return success ? v : 0.0;
    }
}

char* int_to_str(long long v) {
    unsafe {
        // -9223372036854775808 is 20 digits + sign + NUL = 22 bytes
        char* buf = (char*)malloc((unsigned long)22);
        if (buf != (char*)0) snprintf(buf, (unsigned long)22, "%lld", v);
        return buf;
    }
}

char* uint_to_str(unsigned long long v) {
    unsafe {
        char* buf = (char*)malloc((unsigned long)21);
        if (buf != (char*)0) snprintf(buf, (unsigned long)21, "%llu", v);
        return buf;
    }
}

char* float_to_str(double v, int decimals) {
    unsafe {
        char* buf = (char*)malloc((unsigned long)64);
        if (buf != (char*)0) snprintf(buf, (unsigned long)64, "%.*f", decimals, v);
        return buf;
    }
}

int ll_to_int(long long v, int* out) {
    if (v < (long long)INT_MIN || v > (long long)INT_MAX) return 0;
    unsafe { *out = (int)v; }
    return 1;
}

long long float_clamp_to_ll(double v, long long min, long long max) {
    if (v < (double)min) return min;
    if (v > (double)max) return max;
    return (long long)v;
}

int str_is_int(const char* s) {
    int ok = 0;
    str_to_int(s, &ok);
    return ok;
}

int str_is_float(const char* s) {
    int ok = 0;
    str_to_float(s, &ok);
    return ok;
}
