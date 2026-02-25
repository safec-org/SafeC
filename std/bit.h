// SafeC Standard Library — Bit manipulation (C23 <stdbit.h> + compiler builtins)
// All functions are const-eligible (pure, no side effects).
#pragma once

// ── Population count ──────────────────────────────────────────────────────────

// Count set bits in a 32-bit value.
const int popcount32(unsigned int x);

// Count set bits in a 64-bit value.
const int popcount64(unsigned long long x);

// ── Count leading zeros ───────────────────────────────────────────────────────

// Number of leading 0-bits before the most significant 1-bit.
// Returns 32 for x == 0.
const int clz32(unsigned int x);

// Returns 64 for x == 0.
const int clz64(unsigned long long x);

// ── Count trailing zeros ──────────────────────────────────────────────────────

// Number of trailing 0-bits after the least significant 1-bit.
// Returns 32 for x == 0.
const int ctz32(unsigned int x);

// Returns 64 for x == 0.
const int ctz64(unsigned long long x);

// ── Bit-scan (find first/last set bit) ───────────────────────────────────────

// Index of the least-significant set bit (0-based).  Returns -1 for x == 0.
const int bsf32(unsigned int x);
const int bsf64(unsigned long long x);

// Index of the most-significant set bit (0-based).  Returns -1 for x == 0.
const int bsr32(unsigned int x);
const int bsr64(unsigned long long x);

// ── Rotate ────────────────────────────────────────────────────────────────────

// Rotate x left/right by n bits (n is taken modulo 32/64).
const unsigned int  rotl32(unsigned int x, int n);
const unsigned int  rotr32(unsigned int x, int n);
const unsigned long long rotl64(unsigned long long x, int n);
const unsigned long long rotr64(unsigned long long x, int n);

// ── Byte-swap (endian reversal) ───────────────────────────────────────────────

const unsigned int       bswap32(unsigned int x);
const unsigned long long bswap64(unsigned long long x);

// ── Power-of-two helpers ──────────────────────────────────────────────────────

// Return 1 if x is a power of two (x > 0).
const int is_pow2(unsigned long long x);

// Round x up to the next power of two (returns x if already a power of two).
// Returns 1 for x == 0.
const unsigned long long next_pow2(unsigned long long x);

// Integer log2 (floor).  Returns -1 for x == 0.
const int ilog2(unsigned long long x);

// ── Bit field helpers ─────────────────────────────────────────────────────────

// Extract bits [hi:lo] (inclusive) from x.
const unsigned int  bits32(unsigned int x, int lo, int hi);
const unsigned long long bits64(unsigned long long x, int lo, int hi);

// Set bits [hi:lo] to val in x, return new value.
const unsigned int  set_bits32(unsigned int x, int lo, int hi, unsigned int val);
