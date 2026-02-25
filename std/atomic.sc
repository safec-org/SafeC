// SafeC Standard Library — Atomics (C11 <stdatomic.h>)
#pragma once
#include "atomic.h"
#include <stdatomic.h>

int atomic_load_int(const int* addr) {
    unsafe { return atomic_load((const _Atomic int*)addr); }
}

void atomic_store_int(int* addr, int val) {
    unsafe { atomic_store((_Atomic int*)addr, val); }
}

int atomic_fetch_add_int(int* addr, int delta) {
    unsafe { return atomic_fetch_add((_Atomic int*)addr, delta); }
}

int atomic_fetch_sub_int(int* addr, int delta) {
    unsafe { return atomic_fetch_sub((_Atomic int*)addr, delta); }
}

int atomic_fetch_and_int(int* addr, int mask) {
    unsafe { return atomic_fetch_and((_Atomic int*)addr, mask); }
}

int atomic_fetch_or_int(int* addr, int mask) {
    unsafe { return atomic_fetch_or((_Atomic int*)addr, mask); }
}

int atomic_fetch_xor_int(int* addr, int mask) {
    unsafe { return atomic_fetch_xor((_Atomic int*)addr, mask); }
}

int atomic_exchange_int(int* addr, int val) {
    unsafe { return atomic_exchange((_Atomic int*)addr, val); }
}

int atomic_cas_int(int* addr, int* expected, int desired) {
    unsafe {
        return atomic_compare_exchange_strong(
            (_Atomic int*)addr, expected, desired);
    }
}

long long atomic_load_ll(const long long* addr) {
    unsafe { return atomic_load((const _Atomic long long*)addr); }
}

void atomic_store_ll(long long* addr, long long val) {
    unsafe { atomic_store((_Atomic long long*)addr, val); }
}

long long atomic_fetch_add_ll(long long* addr, long long delta) {
    unsafe { return atomic_fetch_add((_Atomic long long*)addr, delta); }
}

long long atomic_fetch_sub_ll(long long* addr, long long delta) {
    unsafe { return atomic_fetch_sub((_Atomic long long*)addr, delta); }
}

long long atomic_exchange_ll(long long* addr, long long val) {
    unsafe { return atomic_exchange((_Atomic long long*)addr, val); }
}

int atomic_cas_ll(long long* addr, long long* expected, long long desired) {
    unsafe {
        return atomic_compare_exchange_strong(
            (_Atomic long long*)addr, expected, desired);
    }
}

void atomic_thread_fence() {
    unsafe { atomic_thread_fence(memory_order_seq_cst); }
}
