// SafeC Standard Library — constant-time helpers (see ct.h).
#pragma once
#include <std/crypto/ct.h>

namespace std {

int ct_eq_n_(const &unsigned char a, const &unsigned char b, unsigned long n) {
    unsigned int diff = 0U;
    unsafe {
        const unsigned char* pa = (const unsigned char*)a;
        const unsigned char* pb = (const unsigned char*)b;
        unsigned long i = 0UL;
        while (i < n) { diff = diff | (unsigned int)(pa[i] ^ pb[i]); i = i + 1UL; }
    }
    // diff in 0..255: (diff - 1) >> 31 is 1 exactly when diff == 0.
    return (int)(((diff - 1U) >> 31) & 1U);
}

int ct_eq([]const unsigned char a_s, []const unsigned char b_s) {
    if (a_s.len != b_s.len) { return 0; }
    unsigned long n = a_s.len;
    const &unsigned char a = crypto_in_(a_s, 0UL);
    const &unsigned char b = crypto_in_(b_s, n);
    return ct_eq_n_(a, b, n);
}

int ct_is_zero_n_(const &unsigned char a, unsigned long n) {
    unsigned int acc = 0U;
    unsafe {
        const unsigned char* pa = (const unsigned char*)a;
        unsigned long i = 0UL;
        while (i < n) { acc = acc | (unsigned int)pa[i]; i = i + 1UL; }
    }
    return (int)(((acc - 1U) >> 31) & 1U);
}

int ct_is_zero([]const unsigned char a_s) {
    unsigned long n = a_s.len;
    const &unsigned char a = crypto_in_(a_s, 0UL);
    return ct_is_zero_n_(a, n);
}

void ct_copy_if_n_(unsigned int bit, &unsigned char dst, const &unsigned char src, unsigned long n) {
    unsigned char m = (unsigned char)(0U - (bit & 1U));
    unsafe {
        unsigned char* d = (unsigned char*)dst;
        const unsigned char* s = (const unsigned char*)src;
        unsigned long i = 0UL;
        while (i < n) { d[i] = (unsigned char)(d[i] ^ (m & (d[i] ^ s[i]))); i = i + 1UL; }
    }
}

void ct_copy_if(unsigned int bit, []unsigned char dst_s, []const unsigned char src_s) {
    if (dst_s.len != src_s.len) { crypto_buf_short_(); }
    unsigned long n = dst_s.len;
    &unsigned char dst = crypto_out_(dst_s, 0UL);
    const &unsigned char src = crypto_in_(src_s, n);
    ct_copy_if_n_(bit, dst, src, n);
}

unsigned int ct_eq_u32(unsigned int a, unsigned int b) {
    unsigned int x = a ^ b;
    // x | -x has its top bit set exactly when x != 0.
    unsigned int nz = (x | (0U - x)) >> 31;
    return nz - 1U;
}

} // namespace std
