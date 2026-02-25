// SafeC Standard Library — Atomics (C11 <stdatomic.h>)
// Lock-free operations on shared integers and pointers.
// All functions require unsafe{} at the call site only when storing results
// in raw pointers; the operations themselves are memory-safe.
#pragma once

// ── atomic_int (32-bit signed) ────────────────────────────────────────────────

// Atomically load *addr (sequential consistency).
int  atomic_load_int(const int* addr);

// Atomically store val into *addr.
void atomic_store_int(int* addr, int val);

// Atomically add delta to *addr; return the old value.
int  atomic_fetch_add_int(int* addr, int delta);

// Atomically subtract delta from *addr; return the old value.
int  atomic_fetch_sub_int(int* addr, int delta);

// Atomically AND *addr with mask; return the old value.
int  atomic_fetch_and_int(int* addr, int mask);

// Atomically OR *addr with mask; return the old value.
int  atomic_fetch_or_int(int* addr, int mask);

// Atomically XOR *addr with mask; return the old value.
int  atomic_fetch_xor_int(int* addr, int mask);

// Atomically exchange *addr with val; return the old value.
int  atomic_exchange_int(int* addr, int val);

// CAS: if *addr == expected, store desired and return 1; else update
// *expected to the current value and return 0.
int  atomic_cas_int(int* addr, int* expected, int desired);

// ── atomic_long (64-bit signed) ───────────────────────────────────────────────

long long  atomic_load_ll(const long long* addr);
void       atomic_store_ll(long long* addr, long long val);
long long  atomic_fetch_add_ll(long long* addr, long long delta);
long long  atomic_fetch_sub_ll(long long* addr, long long delta);
long long  atomic_exchange_ll(long long* addr, long long val);
int        atomic_cas_ll(long long* addr, long long* expected, long long desired);

// ── Memory fence ──────────────────────────────────────────────────────────────

// Full sequential-consistency fence.
void atomic_thread_fence();
