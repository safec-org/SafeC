// SafeC Standard Library — URL encoding implementation (see url.h)
#pragma once
#include <std/encoding/url.h>
#include <std/collections/string.sc>

namespace std {

static int url_is_unreserved_(char c) {
    if (c >= 'A' && c <= 'Z') { return 1; }
    if (c >= 'a' && c <= 'z') { return 1; }
    if (c >= '0' && c <= '9') { return 1; }
    return c == '-' || c == '_' || c == '.' || c == '~';
}

static char url_hex_digit_(int v) {
    if (v < 10) { return (char)((int)'0' + v); }
    return (char)((int)'A' + (v - 10));
}

struct String url_encode(const char* s) {
    struct String out = string_new();
    unsigned long i = 0UL;
    while (1) {
        char c;
        unsafe { c = s[i]; }
        if (c == (char)0) { break; }
        if (url_is_unreserved_(c)) {
            unsafe { out.push_char(c); }
        } else {
            unsigned char b;
            unsafe { b = (unsigned char)c; }
            unsafe {
                out.push_char('%');
                out.push_char(url_hex_digit_(((int)b >> 4) & 0xF));
                out.push_char(url_hex_digit_((int)b & 0xF));
            }
        }
        i = i + 1UL;
    }
    return out;
}

static int url_hex_val_(char c, int* ok) {
    if (c >= '0' && c <= '9') { return (int)c - (int)'0'; }
    if (c >= 'a' && c <= 'f') { return (int)c - (int)'a' + 10; }
    if (c >= 'A' && c <= 'F') { return (int)c - (int)'A' + 10; }
    unsafe { *ok = 0; }
    return 0;
}

struct String url_decode(const char* s, int* ok) {
    struct String out = string_new();
    int good = 1;
    unsigned long i = 0UL;
    while (1) {
        char c;
        unsafe { c = s[i]; }
        if (c == (char)0) { break; }
        if (c == '%') {
            char h1;
            char h2;
            unsafe { h1 = s[i + 1UL]; }
            if (h1 == (char)0) { good = 0; break; }
            unsafe { h2 = s[i + 2UL]; }
            if (h2 == (char)0) { good = 0; break; }
            int hv1 = url_hex_val_(h1, &good);
            int hv2 = url_hex_val_(h2, &good);
            if (good == 0) { break; }
            unsafe { out.push_char((char)((hv1 << 4) | hv2)); }
            i = i + 3UL;
        } else {
            unsafe { out.push_char(c); }
            i = i + 1UL;
        }
    }
    if (ok != (int*)0) { unsafe { *ok = good; } }
    return out;
}

} // namespace std
