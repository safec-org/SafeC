#pragma once
// SafeC Standard Library — one allocation interface over every allocator.
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
// Each one conforms to the trait below, and the typed helpers are written
// once against it. Traits are structural: a type conforms by having the
// methods, so your own allocator works with alloc_t / alloc_array_t /
// dealloc_t (and with any generic<A: Allocator> code) as soon as it defines
// alloc and dealloc with these signatures. Each call is resolved at compile
// time (no dispatch at run time).
//
//   struct BumpAllocator arena = bump_new(4096UL);
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
// always true for *_new; for *_init it's the buffer you pass.
#include <std/mem.h>

namespace std {

trait Allocator {
    // 'size' bytes aligned to 'align' (a power of two), uninitialised; null
    // when the allocator can't (out of space, or an alignment or size it
    // doesn't provide).
    ?&heap void alloc(unsigned long size, unsigned long align);
    // Gives back a block alloc returned; null is a no-op. A no-op on an
    // allocator that can't free one block (bump: reset() frees them all).
    void        dealloc(?&heap void ptr);
}

// ── Typed allocation (recommended) ───────────────────────────────────────────
// One T / n Ts, aligned for T; null when the allocator can't. The memory is
// uninitialised. T is inferred from the target ('?&heap Packet p =
// alloc_t(&pool);'). alloc_array_t aborts with a diagnostic if n * sizeof(T)
// overflows (checked_mul_size).
generic<T, A: Allocator>
?&heap T alloc_t(&A a) {
    unsafe { return (?&heap T)a.alloc(sizeof(T), alignof(T)); }
}

generic<T, A: Allocator>
?&heap T alloc_array_t(&A a, unsigned long n) {
    unsafe { return (?&heap T)a.alloc(checked_mul_size(n, sizeof(T)), alignof(T)); }
}

// Gives back a block from alloc_t / alloc_array_t on the same allocator.
generic<T, A: Allocator>
void dealloc_t(&A a, &heap T obj) {
    unsafe { a.dealloc((?&heap void)obj); }
}

} // namespace std
