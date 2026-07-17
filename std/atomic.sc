// SafeC Standard Library — Atomics (C11 <stdatomic.h>-equivalent surface)
//
// atomic_load/store/fetch_*/exchange/cas below are SafeC compiler builtins
// (see Sema::checkCall's "Atomic built-ins" handling), not C library
// functions — unlike real C, they need no declaration from a system
// <stdatomic.h> at all, so this file (unlike real C11 code) works
// unmodified in --freestanding mode, where no hosted header is available.
#pragma once
#include <std/atomic.h>

namespace std {

inline int atomic_load_int(const int* addr) {
    unsafe { return atomic_load((const _Atomic int*)addr); }
}

inline void atomic_store_int(int* addr, int val) {
    unsafe { atomic_store((_Atomic int*)addr, val); }
}

inline int atomic_fetch_add_int(int* addr, int delta) {
    unsafe { return atomic_fetch_add((_Atomic int*)addr, delta); }
}

inline int atomic_fetch_sub_int(int* addr, int delta) {
    unsafe { return atomic_fetch_sub((_Atomic int*)addr, delta); }
}

inline int atomic_fetch_and_int(int* addr, int mask) {
    unsafe { return atomic_fetch_and((_Atomic int*)addr, mask); }
}

inline int atomic_fetch_or_int(int* addr, int mask) {
    unsafe { return atomic_fetch_or((_Atomic int*)addr, mask); }
}

inline int atomic_fetch_xor_int(int* addr, int mask) {
    unsafe { return atomic_fetch_xor((_Atomic int*)addr, mask); }
}

inline int atomic_exchange_int(int* addr, int val) {
    unsafe { return atomic_exchange((_Atomic int*)addr, val); }
}

inline int atomic_cas_int(int* addr, int* expected, int desired) {
    unsafe {
        return atomic_compare_exchange_strong(
            (_Atomic int*)addr, expected, desired);
    }
}

inline long long atomic_load_ll(const long long* addr) {
    unsafe { return atomic_load((const _Atomic long long*)addr); }
}

inline void atomic_store_ll(long long* addr, long long val) {
    unsafe { atomic_store((_Atomic long long*)addr, val); }
}

inline long long atomic_fetch_add_ll(long long* addr, long long delta) {
    unsafe { return atomic_fetch_add((_Atomic long long*)addr, delta); }
}

inline long long atomic_fetch_sub_ll(long long* addr, long long delta) {
    unsafe { return atomic_fetch_sub((_Atomic long long*)addr, delta); }
}

inline long long atomic_exchange_ll(long long* addr, long long val) {
    unsafe { return atomic_exchange((_Atomic long long*)addr, val); }
}

inline int atomic_cas_ll(long long* addr, long long* expected, long long desired) {
    unsafe {
        return atomic_compare_exchange_strong(
            (_Atomic long long*)addr, expected, desired);
    }
}

inline void atomic_thread_fence() {
    unsafe { atomic_fence(); }
}

} // namespace std
