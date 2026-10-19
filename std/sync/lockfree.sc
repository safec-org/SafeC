// SafeC Standard Library — LFQueue<T> (generic; see lockfree.h). This file
// holds the capacity-rounding helpers the header's constructors share.
#pragma once
#include <std/sync/lockfree.h>

namespace std {

unsigned long lfq_floor_pow2_(unsigned long n) {
    if (n == 0UL) return 1UL;
    unsigned long p = 1UL;
    while (p * 2UL <= n) { p = p * 2UL; }
    return p;
}

unsigned long lfq_ceil_pow2_(unsigned long n) {
    unsigned long p = 1UL;
    while (p < n) { p = p * 2UL; }
    return p;
}

} // namespace std
