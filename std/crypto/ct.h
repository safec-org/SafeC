// SafeC Standard Library — constant-time helpers for cryptographic code.
//
// Comparisons and selections whose running time doesn't depend on the
// (secret) bytes involved: no early exit, no branch on the data. Used by
// std's AEADs, RSA/ECDSA and TLS for tags, padding and key material.
// Freestanding-safe.
#pragma once
#include <std/crypto/buf.h>

namespace std {

// 1 if a[0..n) == b[0..n), looking at every byte.
int  ct_eq([]const unsigned char a, []const unsigned char b);
// internal: the (reference, length) core of ct_eq()
int  ct_eq_n_(const &unsigned char a, const &unsigned char b, unsigned long n);
// 1 if every byte of a[0..n) is zero, looking at every byte.
int  ct_is_zero([]const unsigned char a);
// internal: the (reference, length) core of ct_is_zero()
int  ct_is_zero_n_(const &unsigned char a, unsigned long n);
// dst[0..n) = bit ? src[0..n) : dst[0..n) ('bit' is 0 or 1), touching
// every byte either way.
void ct_copy_if(unsigned int bit, []unsigned char dst, []const unsigned char src);
// internal: the (reference, length) core of ct_copy_if()
void ct_copy_if_n_(unsigned int bit, &unsigned char dst, const &unsigned char src, unsigned long n);
// 0xFFFFFFFF when a == b, 0 otherwise (no branch).
unsigned int ct_eq_u32(unsigned int a, unsigned int b);

} // namespace std
