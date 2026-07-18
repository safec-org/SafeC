// SafeC Standard Library — UTF-8 implementation (see utf8.h)
#pragma once
#include <std/encoding/utf8.h>

namespace std {

int utf8_encoded_len(unsigned int cp) {
    if (cp > 0x10FFFFU) { return 0; }
    if (cp >= 0xD800U && cp <= 0xDFFFU) { return 0; } // surrogate range
    if (cp <= 0x7FU) { return 1; }
    if (cp <= 0x7FFU) { return 2; }
    if (cp <= 0xFFFFU) { return 3; }
    return 4;
}

int utf8_encode(unsigned int cp, char* out) {
    int n = utf8_encoded_len(cp);
    if (n == 0) { return 0; }
    if (n == 1) {
        unsafe { out[0] = (char)cp; }
        return 1;
    }
    if (n == 2) {
        unsafe {
            out[0] = (char)(0xC0U | (cp >> 6));
            out[1] = (char)(0x80U | (cp & 0x3FU));
        }
        return 2;
    }
    if (n == 3) {
        unsafe {
            out[0] = (char)(0xE0U | (cp >> 12));
            out[1] = (char)(0x80U | ((cp >> 6) & 0x3FU));
            out[2] = (char)(0x80U | (cp & 0x3FU));
        }
        return 3;
    }
    unsafe {
        out[0] = (char)(0xF0U | (cp >> 18));
        out[1] = (char)(0x80U | ((cp >> 12) & 0x3FU));
        out[2] = (char)(0x80U | ((cp >> 6) & 0x3FU));
        out[3] = (char)(0x80U | (cp & 0x3FU));
    }
    return 4;
}

// Reads one continuation byte (10xxxxxx) at s[pos]. Returns its 6 payload
// bits shifted into place, or -1 if it's not a valid continuation byte
// (including running past a NUL terminator).
static int utf8_cont_(const char* s, unsigned long pos) {
    unsigned char b;
    unsafe { b = (unsigned char)s[pos]; }
    if (b == (unsigned char)0) { return -1; }
    if (((unsigned int)b & 0xC0U) != 0x80U) { return -1; }
    return (int)((unsigned int)b & 0x3FU);
}

int utf8_decode(const char* s, unsigned long pos, unsigned int* cp_out) {
    unsigned char b0;
    unsafe { b0 = (unsigned char)s[pos]; }
    if (b0 == (unsigned char)0) { return 0; }

    if (((unsigned int)b0 & 0x80U) == 0U) {
        unsafe { *cp_out = (unsigned int)b0; }
        return 1;
    }
    if (((unsigned int)b0 & 0xE0U) == 0xC0U) {
        int c1 = utf8_cont_(s, pos + 1UL);
        if (c1 < 0) { return 0; }
        unsigned int cp = (((unsigned int)b0 & 0x1FU) << 6) | (unsigned int)c1;
        if (cp < 0x80U) { return 0; } // overlong encoding
        unsafe { *cp_out = cp; }
        return 2;
    }
    if (((unsigned int)b0 & 0xF0U) == 0xE0U) {
        int c1 = utf8_cont_(s, pos + 1UL);
        if (c1 < 0) { return 0; }
        int c2 = utf8_cont_(s, pos + 2UL);
        if (c2 < 0) { return 0; }
        unsigned int cp = (((unsigned int)b0 & 0x0FU) << 12) |
                           ((unsigned int)c1 << 6) | (unsigned int)c2;
        if (cp < 0x800U) { return 0; } // overlong
        if (cp >= 0xD800U && cp <= 0xDFFFU) { return 0; } // encoded surrogate
        unsafe { *cp_out = cp; }
        return 3;
    }
    if (((unsigned int)b0 & 0xF8U) == 0xF0U) {
        int c1 = utf8_cont_(s, pos + 1UL);
        if (c1 < 0) { return 0; }
        int c2 = utf8_cont_(s, pos + 2UL);
        if (c2 < 0) { return 0; }
        int c3 = utf8_cont_(s, pos + 3UL);
        if (c3 < 0) { return 0; }
        unsigned int cp = (((unsigned int)b0 & 0x07U) << 18) |
                           ((unsigned int)c1 << 12) |
                           ((unsigned int)c2 << 6) | (unsigned int)c3;
        if (cp < 0x10000U) { return 0; } // overlong
        if (cp > 0x10FFFFU) { return 0; }
        unsafe { *cp_out = cp; }
        return 4;
    }
    return 0; // stray continuation byte or 0xF8..0xFF (never valid lead bytes)
}

int utf8_is_valid(const char* s) {
    unsigned long pos = 0UL;
    while (1) {
        char c;
        unsafe { c = s[pos]; }
        if (c == (char)0) { return 1; }
        unsigned int cp;
        int n = utf8_decode(s, pos, &cp);
        if (n == 0) { return 0; }
        pos = pos + (unsigned long)n;
    }
}

unsigned long utf8_len(const char* s) {
    unsigned long pos = 0UL;
    unsigned long count = 0UL;
    while (1) {
        char c;
        unsafe { c = s[pos]; }
        if (c == (char)0) { return count; }
        unsigned int cp;
        int n = utf8_decode(s, pos, &cp);
        if (n == 0) { return 0UL; }
        pos = pos + (unsigned long)n;
        count = count + 1UL;
    }
}

} // namespace std
