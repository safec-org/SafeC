// SafeC Standard Library — Bit manipulation implementation
// Uses GCC/Clang compiler builtins for hardware-accelerated operations.
#pragma once
#include "bit.h"

const int popcount32(unsigned int x) {
    unsafe { return __builtin_popcount(x); }
}

const int popcount64(unsigned long long x) {
    unsafe { return __builtin_popcountll(x); }
}

const int clz32(unsigned int x) {
    if (x == (unsigned int)0) return 32;
    unsafe { return __builtin_clz(x); }
}

const int clz64(unsigned long long x) {
    if (x == (unsigned long long)0) return 64;
    unsafe { return __builtin_clzll(x); }
}

const int ctz32(unsigned int x) {
    if (x == (unsigned int)0) return 32;
    unsafe { return __builtin_ctz(x); }
}

const int ctz64(unsigned long long x) {
    if (x == (unsigned long long)0) return 64;
    unsafe { return __builtin_ctzll(x); }
}

const int bsf32(unsigned int x) {
    if (x == (unsigned int)0) return -1;
    unsafe { return __builtin_ctz(x); }
}

const int bsf64(unsigned long long x) {
    if (x == (unsigned long long)0) return -1;
    unsafe { return __builtin_ctzll(x); }
}

const int bsr32(unsigned int x) {
    if (x == (unsigned int)0) return -1;
    return 31 - clz32(x);
}

const int bsr64(unsigned long long x) {
    if (x == (unsigned long long)0) return -1;
    return 63 - clz64(x);
}

const unsigned int rotl32(unsigned int x, int n) {
    int shift = n & 31;
    if (shift == 0) return x;
    return (x << shift) | (x >> (32 - shift));
}

const unsigned int rotr32(unsigned int x, int n) {
    int shift = n & 31;
    if (shift == 0) return x;
    return (x >> shift) | (x << (32 - shift));
}

const unsigned long long rotl64(unsigned long long x, int n) {
    int shift = n & 63;
    if (shift == 0) return x;
    return (x << shift) | (x >> (64 - shift));
}

const unsigned long long rotr64(unsigned long long x, int n) {
    int shift = n & 63;
    if (shift == 0) return x;
    return (x >> shift) | (x << (64 - shift));
}

const unsigned int bswap32(unsigned int x) {
    unsafe { return __builtin_bswap32(x); }
}

const unsigned long long bswap64(unsigned long long x) {
    unsafe { return __builtin_bswap64(x); }
}

const int is_pow2(unsigned long long x) {
    return x > (unsigned long long)0 && (x & (x - (unsigned long long)1)) == (unsigned long long)0;
}

const unsigned long long next_pow2(unsigned long long x) {
    if (x == (unsigned long long)0) return (unsigned long long)1;
    x = x - (unsigned long long)1;
    x = x | (x >> (unsigned long long)1);
    x = x | (x >> (unsigned long long)2);
    x = x | (x >> (unsigned long long)4);
    x = x | (x >> (unsigned long long)8);
    x = x | (x >> (unsigned long long)16);
    x = x | (x >> (unsigned long long)32);
    return x + (unsigned long long)1;
}

const int ilog2(unsigned long long x) {
    if (x == (unsigned long long)0) return -1;
    return 63 - clz64(x);
}

const unsigned int bits32(unsigned int x, int lo, int hi) {
    int width = hi - lo + 1;
    unsigned int mask = (unsigned int)((1 << width) - 1);
    return (x >> lo) & mask;
}

const unsigned long long bits64(unsigned long long x, int lo, int hi) {
    int width = hi - lo + 1;
    unsigned long long mask = ((unsigned long long)1 << width) - (unsigned long long)1;
    return (x >> lo) & mask;
}

const unsigned int set_bits32(unsigned int x, int lo, int hi, unsigned int val) {
    int width = hi - lo + 1;
    unsigned int mask = (unsigned int)((1 << width) - 1);
    return (x & ~(mask << lo)) | ((val & mask) << lo);
}
