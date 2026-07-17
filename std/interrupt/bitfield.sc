// SafeC Standard Library — Bitfield Helpers
// All functions are pure — no side effects, no MMIO access.
#pragma once
#include <std/interrupt/bitfield.h>

namespace std {

inline unsigned int bf_mask32(int lo, int hi) {
    unsigned int width = (unsigned int)(hi - lo + 1);
    unsigned int mask = ((unsigned int)1 << width) - (unsigned int)1;
    return mask << lo;
}

inline unsigned int bf_extract32(unsigned int val, int lo, int hi) {
    unsigned int mask = bf_mask32(lo, hi);
    return (val & mask) >> lo;
}

inline unsigned int bf_insert32(unsigned int val, int lo, int hi, unsigned int field) {
    unsigned int mask = bf_mask32(lo, hi);
    val = val & ~mask;
    val = val | ((field << lo) & mask);
    return val;
}

inline unsigned int bf_set32(unsigned int val, int lo, int hi) {
    return val | bf_mask32(lo, hi);
}

inline unsigned int bf_clear32(unsigned int val, int lo, int hi) {
    return val & ~bf_mask32(lo, hi);
}

inline unsigned int bf_toggle32(unsigned int val, int lo, int hi) {
    return val ^ bf_mask32(lo, hi);
}

inline int bf_test32(unsigned int val, int n) {
    if ((val & ((unsigned int)1 << n)) != (unsigned int)0) {
        return 1;
    }
    return 0;
}

inline int bf_popcount32(unsigned int val) {
    int count = 0;
    while (val != (unsigned int)0) {
        count = count + (int)(val & (unsigned int)1);
        val = val >> 1;
    }
    return count;
}

} // namespace std
