// SafeC Standard Library — String formatting implementation
#pragma once
#include "fmt.h"
#include <stdio.h>
#include <string.h>

int fmt_int(char* buf, unsigned long cap, long long v) {
    unsafe { return snprintf(buf, cap, "%lld", v); }
}

int fmt_uint(char* buf, unsigned long cap, unsigned long long v) {
    unsafe { return snprintf(buf, cap, "%llu", v); }
}

int fmt_float(char* buf, unsigned long cap, double v, int decimals) {
    unsafe { return snprintf(buf, cap, "%.*f", decimals, v); }
}

int fmt_hex(char* buf, unsigned long cap, unsigned long long v) {
    unsafe { return snprintf(buf, cap, "%llx", v); }
}

int fmt_hex_upper(char* buf, unsigned long cap, unsigned long long v) {
    unsafe { return snprintf(buf, cap, "%llX", v); }
}

int fmt_bool(char* buf, unsigned long cap, int v) {
    unsafe { return snprintf(buf, cap, "%s", v ? "true" : "false"); }
}

int fmt_str(char* dst, unsigned long cap, const char* src) {
    if (cap == (unsigned long)0) return 0;
    unsafe {
        unsigned long n = strlen(src);
        if (n >= cap) n = cap - (unsigned long)1;
        memcpy(dst, src, n);
        dst[n] = '\0';
        return (int)n;
    }
}

int fmt_append(char* dst, unsigned long cap, const char* src) {
    unsafe {
        unsigned long dlen = strlen(dst);
        unsigned long slen = strlen(src);
        if (dlen + slen + (unsigned long)1 > cap) return -1;
        memcpy(dst + dlen, src, slen + (unsigned long)1);
        return (int)(dlen + slen);
    }
}
