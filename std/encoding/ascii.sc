// SafeC Standard Library — ASCII implementation (see ascii.h)
#pragma once
#include <std/encoding/ascii.h>

namespace std {

int ascii_is_valid_n(const char* s, unsigned long len) {
    unsigned long i = 0UL;
    while (i < len) {
        char c;
        unsafe { c = s[i]; }
        if ((int)c < 0 || (int)c > 127) { return 0; }
        i = i + 1UL;
    }
    return 1;
}

int ascii_is_valid(const char* s) {
    unsigned long i = 0UL;
    while (1) {
        char c;
        unsafe { c = s[i]; }
        if (c == (char)0) { return 1; }
        if ((int)c < 0 || (int)c > 127) { return 0; }
        i = i + 1UL;
    }
}

int ascii_is_alpha(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

int ascii_is_digit(char c) {
    return c >= '0' && c <= '9';
}

int ascii_is_alnum(char c) {
    return ascii_is_alpha(c) || ascii_is_digit(c);
}

int ascii_is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

int ascii_is_upper(char c) {
    return c >= 'A' && c <= 'Z';
}

int ascii_is_lower(char c) {
    return c >= 'a' && c <= 'z';
}

int ascii_is_print(char c) {
    return (int)c >= 0x20 && (int)c <= 0x7E;
}

char ascii_to_upper(char c) {
    if (ascii_is_lower(c)) { return (char)((int)c - 32); }
    return c;
}

char ascii_to_lower(char c) {
    if (ascii_is_upper(c)) { return (char)((int)c + 32); }
    return c;
}

} // namespace std
