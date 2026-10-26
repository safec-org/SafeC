// SafeC Standard Library — HashMap<K,V> / StrMap<V>. The containers are
// generic and live entirely in map.h; this file holds the one non-generic
// helper they share.
#pragma once
#include <std/collections/map.h>

namespace std {

// FNV-1a over the bytes up to the NUL, plus a murmur3 finalizer.
unsigned int map_hash_str_(const &char key) {
    unsigned int h = 2166136261U;
    unsafe {
        const char* s = key;
        while (*s != 0) {
            h = (h ^ (unsigned int)(unsigned char)(*s)) * 16777619U;
            s = s + 1;
        }
    }
    h = h ^ (h >> 16); h = h * 2246822507U;
    h = h ^ (h >> 13); h = h * 3266489909U;
    return h ^ (h >> 16);
}

unsigned long map_round_cap_(unsigned long n) {
    unsigned long c = 16UL;
    while (c < n) { c = c * 2UL; }
    return c;
}

} // namespace std
