#pragma once
// SafeC Standard Library — allocation traits shared by every allocator.
//
// std/alloc has four allocators over a buffer of their own (or one you
// hand them); each suits a different allocation pattern:
//
//   BumpAllocator  (bump.h)  arena: O(1) alloc, frees everything at once
//                            with reset(); dealloc of one block is a no-op.
//   PoolAllocator  (pool.h)  variable sizes, first fit, coalescing frees.
//   SlabAllocator  (slab.h)  one fixed object size, O(1) alloc and free.
//   TlsfAllocator  (tlsf.h)  variable sizes, O(1) alloc and free with
//                            bounded fragmentation (real-time use).
//
// The interface is split into one trait per capability, composed into the
// two roles generic code asks for, and the typed helpers are written once
// against the smallest trait they need:
//
//   Allocate        alloc(size, align)
//   Deallocate      dealloc(ptr)
//   Resettable      reset(): frees every block at once
//   Allocator       Allocate + Deallocate: BumpAllocator (no-op dealloc),
//                   PoolAllocator, SlabAllocator, TlsfAllocator,
//                   SecureAllocator (crypto/secure_alloc.h),
//                   CountingAllocator<A> (below)
//   ArenaAllocator  Allocate + Resettable: allocate freely, then reset() —
//                   BumpAllocator, PoolAllocator, SlabAllocator,
//                   TlsfAllocator
//
// Traits are structural: a type conforms by having the methods, so your own
// allocator works with alloc_t / alloc_array_t / dealloc_t (and with any
// generic<A: Allocator> code) as soon as it defines alloc and dealloc with
// these signatures. Each call is resolved at compile time (no dispatch at
// run time).
//
//   struct BumpAllocator arena = BumpAllocator::new(4096UL);
//   ?&heap Packet p = alloc_t(&arena);              // T = Packet, from the target
//   ?&heap float xs = alloc_array_t(&arena, 64UL);  // 64 floats
//   ...
//   arena.destroy();
//
// Alignment each allocator honours ('align' must be a power of two; 0 is
// taken as 1; anything else gives null):
//   bump  any power of two (measured on the address, padding the arena).
//   pool  up to 8; null beyond.
//   slab  up to the largest power of two dividing its object size, at most
//         8; null beyond, and null when size > its object size.
//   tlsf  up to 8; null beyond.
// The bounds for pool, slab and tlsf assume an 8-aligned backing buffer:
// always true for X::new; for X::over it's the buffer you pass.
#include <std/mem.h>
#include <std/traits.h>   // Resettable

namespace std {

trait Allocate {
    // 'size' bytes aligned to 'align' (a power of two), uninitialised; null
    // when the allocator can't (out of space, or an alignment or size it
    // doesn't provide).
    ?&heap unsigned char alloc(unsigned long size, unsigned long align);
}

trait Deallocate {
    // Gives back a block alloc returned; null is a no-op. A no-op on an
    // allocator that can't free one block (bump: reset() frees them all).
    void dealloc(?&heap unsigned char ptr);
}

trait Allocator : Allocate + Deallocate {}

trait ArenaAllocator : Allocate + Resettable {}

// ── Typed allocation (recommended) ───────────────────────────────────────────
// One T / n Ts, aligned for T; null when the allocator can't. The memory is
// uninitialised. T is inferred from the target ('?&heap Packet p =
// alloc_t(&pool);'). alloc_array_t aborts with a diagnostic if n * sizeof(T)
// overflows (checked_mul_size).
generic<T, A: Allocate>
?&heap T alloc_t(&A a) {
    unsafe { return (?&heap T)a.alloc(sizeof(T), alignof(T)); }
}

generic<T, A: Allocate>
?&heap T alloc_array_t(&A a, unsigned long n) {
    unsafe { return (?&heap T)a.alloc(checked_mul_size(n, sizeof(T)), alignof(T)); }
}

// Gives back a block from alloc_t / alloc_array_t on the same allocator.
generic<T, A: Deallocate>
void dealloc_t(&A a, &heap T obj) {
    unsafe { a.dealloc((?&heap unsigned char)obj); }
}

// ── CountingAllocator: usage statistics over any Allocator ──────────────────
// Forwards every call to the allocator it borrows and counts them: blocks
// handed out and given back, failed requests, bytes requested. It is an
// Allocator itself, so it slots in wherever the allocator it wraps would
// (leak checks in tests, sizing an arena or pool from a real workload).
//
//   struct PoolAllocator pool = PoolAllocator::new(65536UL);
//   struct CountingAllocator<PoolAllocator> c = CountingAllocator::over(&pool);
//   run_workload(&c);                       // any generic<A: Allocator> code
//   if (c.live() != 0UL) { ... }            // blocks never given back
generic<A: Allocator>
struct CountingAllocator {
    &A            inner;      // the allocator doing the work (not owned)
    unsigned long allocs;     // alloc calls that returned a block
    unsigned long frees;      // dealloc calls with a non-null block
    unsigned long failures;   // alloc calls that returned null
    unsigned long bytes;      // bytes requested by the successful allocs

    static struct CountingAllocator<A> over(&A inner);
    ?&heap unsigned char alloc(unsigned long size, unsigned long align);
    void          dealloc(?&heap unsigned char ptr);
    // Blocks handed out and not given back yet.
    unsigned long live() const;
};

generic<A: Allocator>
struct CountingAllocator<A> CountingAllocator::over(&A inner) {
    struct CountingAllocator<A> c;
    c.inner = inner;
    c.allocs = 0UL;
    c.frees = 0UL;
    c.failures = 0UL;
    c.bytes = 0UL;
    return c;
}

generic<A: Allocator>
?&heap unsigned char CountingAllocator::alloc(unsigned long size, unsigned long align) {
    ?&heap unsigned char p = self.inner.alloc(size, align);
    if (p.is_null()) {
        self.failures = self.failures + 1UL;
    } else {
        self.allocs = self.allocs + 1UL;
        self.bytes = self.bytes + size;
    }
    return p;
}

generic<A: Allocator>
void CountingAllocator::dealloc(?&heap unsigned char ptr) {
    if (!ptr.is_null()) { self.frees = self.frees + 1UL; }
    self.inner.dealloc(ptr);
}

generic<A: Allocator>
unsigned long CountingAllocator::live() const { return self.allocs - self.frees; }

} // namespace std
