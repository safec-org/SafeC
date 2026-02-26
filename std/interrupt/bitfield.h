// SafeC Standard Library — Bitfield Helpers
// Pure functions for register bit manipulation. Freestanding-safe.
#pragma once

// Extract bits [hi:lo] (inclusive) from a 32-bit value.
unsigned int bf_extract32(unsigned int val, int lo, int hi);

// Insert `field` into bits [hi:lo] of `val`. Returns modified value.
unsigned int bf_insert32(unsigned int val, int lo, int hi, unsigned int field);

// Set bits [hi:lo] to all 1s.
unsigned int bf_set32(unsigned int val, int lo, int hi);

// Clear bits [hi:lo] to all 0s.
unsigned int bf_clear32(unsigned int val, int lo, int hi);

// Toggle bits [hi:lo].
unsigned int bf_toggle32(unsigned int val, int lo, int hi);

// Check if bit `n` is set. Returns 1 if set.
int bf_test32(unsigned int val, int n);

// Return a mask with bits [hi:lo] set.
unsigned int bf_mask32(int lo, int hi);

// Count set bits in a 32-bit value.
int bf_popcount32(unsigned int val);
