// SafeC Standard Library — Base64 implementation (see base64.h)
#pragma once
#include <std/encoding/base64.h>
#include <std/collections/string.sc>
#include <std/collections/vec.sc>

namespace std {

static char base64_char_(int sextet) {
    // A 'static const char*' initializer isn't accepted as a global (must be
    // a local, per reference/memory.md's "static const char* initializers
    // must be local, not global" note), so the alphabet lives inside this
    // function instead of as a file-scope constant.
    static const char* alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char c;
    unsafe { c = alphabet[sextet]; }
    return c;
}

struct String base64_encode(const unsigned char* data, unsigned long len) {
    struct String out = string_new();
    unsigned long i = 0UL;
    while (i + 3UL <= len) {
        unsigned char b0;
        unsigned char b1;
        unsigned char b2;
        unsafe {
            b0 = data[i];
            b1 = data[i + 1UL];
            b2 = data[i + 2UL];
        }
        int n0 = ((int)b0 >> 2) & 0x3F;
        int n1 = (((int)b0 & 0x3) << 4) | (((int)b1 >> 4) & 0xF);
        int n2 = (((int)b1 & 0xF) << 2) | (((int)b2 >> 6) & 0x3);
        int n3 = (int)b2 & 0x3F;
        unsafe {
            out.push_char(base64_char_(n0));
            out.push_char(base64_char_(n1));
            out.push_char(base64_char_(n2));
            out.push_char(base64_char_(n3));
        }
        i = i + 3UL;
    }
    unsigned long rem = len - i;
    if (rem == 1UL) {
        unsigned char b0;
        unsafe { b0 = data[i]; }
        int n0 = ((int)b0 >> 2) & 0x3F;
        int n1 = ((int)b0 & 0x3) << 4;
        unsafe {
            out.push_char(base64_char_(n0));
            out.push_char(base64_char_(n1));
            out.push_char('=');
            out.push_char('=');
        }
    } else if (rem == 2UL) {
        unsigned char b0;
        unsigned char b1;
        unsafe {
            b0 = data[i];
            b1 = data[i + 1UL];
        }
        int n0 = ((int)b0 >> 2) & 0x3F;
        int n1 = (((int)b0 & 0x3) << 4) | (((int)b1 >> 4) & 0xF);
        int n2 = ((int)b1 & 0xF) << 2;
        unsafe {
            out.push_char(base64_char_(n0));
            out.push_char(base64_char_(n1));
            out.push_char(base64_char_(n2));
            out.push_char('=');
        }
    }
    return out;
}

// -1 for whitespace (skip), -2 for invalid, 0-63 for a valid alphabet char.
static int base64_val_(char c) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { return -1; }
    if (c >= 'A' && c <= 'Z') { return (int)c - (int)'A'; }
    if (c >= 'a' && c <= 'z') { return (int)c - (int)'a' + 26; }
    if (c >= '0' && c <= '9') { return (int)c - (int)'0' + 52; }
    if (c == '+') { return 62; }
    if (c == '/') { return 63; }
    return -2;
}

struct Vec base64_decode(const char* s, int* ok) {
    struct Vec out = vec_new(1UL);
    int good = 1;
    int group[4];
    int groupLen = 0;
    int paddingSeen = 0;
    unsigned long i = 0UL;
    while (1) {
        char c;
        unsafe { c = s[i]; }
        if (c == (char)0) { break; }
        if (c == '=') {
            unsafe { paddingSeen = paddingSeen + 1; }
            i = i + 1UL;
            continue;
        }
        int v = base64_val_(c);
        if (v == -1) { i = i + 1UL; continue; } // whitespace
        if (v == -2 || paddingSeen > 0) {
            // A non-alphabet byte, or an alphabet char appearing after '='
            // padding already started within this group — both malformed.
            good = 0;
            break;
        }
        unsafe { group[groupLen] = v; }
        groupLen = groupLen + 1;
        if (groupLen == 4) {
            unsigned char b0 = (unsigned char)(((group[0] << 2) | (group[1] >> 4)) & 0xFF);
            unsigned char b1 = (unsigned char)((((group[1] & 0xF) << 4) | (group[2] >> 2)) & 0xFF);
            unsigned char b2 = (unsigned char)((((group[2] & 0x3) << 6) | group[3]) & 0xFF);
            unsafe {
                out.push((void*)&b0);
                out.push((void*)&b1);
                out.push((void*)&b2);
            }
            groupLen = 0;
        }
        i = i + 1UL;
    }
    if (good != 0) {
        if (groupLen == 1) {
            good = 0; // a single leftover sextet can never be valid Base64
        } else if (groupLen == 2) {
            unsigned char b0 = (unsigned char)(((group[0] << 2) | (group[1] >> 4)) & 0xFF);
            unsafe { out.push((void*)&b0); }
        } else if (groupLen == 3) {
            unsigned char b0 = (unsigned char)(((group[0] << 2) | (group[1] >> 4)) & 0xFF);
            unsigned char b1 = (unsigned char)((((group[1] & 0xF) << 4) | (group[2] >> 2)) & 0xFF);
            unsafe {
                out.push((void*)&b0);
                out.push((void*)&b1);
            }
        }
    }
    if (ok != (int*)0) { unsafe { *ok = good; } }
    return out;
}

} // namespace std
