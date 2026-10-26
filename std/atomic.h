// SafeC Standard Library — Atomics (C11 <stdatomic.h>)
// Lock-free operations on shared integers and pointers.
// All functions require unsafe{} at the call site only when storing results
// in raw pointers; the operations themselves are memory-safe.
#pragma once

// ── atomic_int (32-bit signed) ────────────────────────────────────────────────

// Atomically load *addr (sequential consistency).
namespace std {

int  atomic_load_int(const &int addr);

// Atomically store val into *addr.
void atomic_store_int(&int addr, int val);

// Atomically add delta to *addr; return the old value.
int  atomic_fetch_add_int(&int addr, int delta);

// Atomically subtract delta from *addr; return the old value.
int  atomic_fetch_sub_int(&int addr, int delta);

// Atomically AND *addr with mask; return the old value.
int  atomic_fetch_and_int(&int addr, int mask);

// Atomically OR *addr with mask; return the old value.
int  atomic_fetch_or_int(&int addr, int mask);

// Atomically XOR *addr with mask; return the old value.
int  atomic_fetch_xor_int(&int addr, int mask);

// Atomically exchange *addr with val; return the old value.
int  atomic_exchange_int(&int addr, int val);

// CAS: if *addr == expected, store desired and return 1; else update
// *expected to the current value and return 0.
int  atomic_cas_int(&int addr, &int expected, int desired);

// ── atomic_long (64-bit signed) ───────────────────────────────────────────────

long long  atomic_load_ll(const &long long addr);
void       atomic_store_ll(&long long addr, long long val);
long long  atomic_fetch_add_ll(&long long addr, long long delta);
long long  atomic_fetch_sub_ll(&long long addr, long long delta);
long long  atomic_exchange_ll(&long long addr, long long val);
int        atomic_cas_ll(&long long addr, &long long expected, long long desired);

// ── Memory fence ──────────────────────────────────────────────────────────────

// Full sequential-consistency fence.
void atomic_thread_fence();

// ── Typed atomics ─────────────────────────────────────────────────────────────
// An int / long long every access to which is atomic: Sync, so threads may
// share one through a mutable reference ('&static AtomicInt', or a local
// lent to spawn_scoped threads) where a plain int would race.
//   static struct AtomicInt hits;            // zero-initialized
//   hits.fetch_add(1);
struct AtomicInt {
    int v_;
    unsafe impl Send + Sync;
    static struct AtomicInt new(int v);
    int  load() const;
    void store(int v);
    int  fetch_add(int d);     // returns the old value
    int  fetch_sub(int d);
    int  exchange(int v);
    int  compare_exchange(&int expected, int desired);   // 1 if stored
};

struct AtomicLong {
    long long v_;
    unsafe impl Send + Sync;
    static struct AtomicLong new(long long v);
    long long load() const;
    void      store(long long v);
    long long fetch_add(long long d);
    long long fetch_sub(long long d);
    long long exchange(long long v);
    int       compare_exchange(&long long expected, long long desired);
};


} // namespace std
