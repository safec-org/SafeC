// SafeC Standard Library — Slab Allocator
// Fixed-size object pool. O(1) alloc/free via embedded free-list.
// Conforms to Allocator (allocator.h): typed allocation is alloc_t /
// dealloc_t there.
#pragma once
#include <std/alloc/allocator.h>

namespace std {

struct SlabAllocator {
    &heap void    base;       // heap-backed backing buffer
    int           owned;   // 1 if SlabAllocator::new allocated base (destroy() frees it)
    void*         free_head;  // head of embedded free-list (raw — unsafe arithmetic)
    unsigned long obj_size;   // slot size: the requested size rounded up to a multiple of 8
    unsigned long cap;        // max number of objects
    unsigned long used;       // currently allocated count
    // 1 if each slot has an extra 8-byte double-free-detection tag ahead
    // of it (SlabAllocator::new, where this allocator sized+owns 'base' itself, so
    // growing the per-slot stride is safe) — 0 if not (SlabAllocator::over, where
    // 'base' is caller-provided and sized to the documented obj_size*count
    // contract, which adding a tag would silently overrun).
    int           has_tags;

    // A slab over a user-provided buffer of obj_size*count bytes. Slots are
    // obj_size rounded up to a multiple of 8 (at least 8, to hold the
    // free-list link) and start 8-byte aligned, so 'cap' can come out below
    // 'count' when that rounding doesn't fit. No double-free detection (see
    // has_tags) — the buffer size contract can't grow to fit a tag.
    static struct SlabAllocator over(&unsigned char buffer, unsigned long obj_size, unsigned long count);

    // A heap-backed slab of `count` objects (double-free-detecting).
    static struct SlabAllocator new(unsigned long obj_size, unsigned long count);

    // Allocate one object (all `obj_size` bytes of it) for `size` bytes
    // aligned to `align`. null if the slab is full, size > obj_size, or
    // `align` is above max_align() or not a power of two (0 is taken as 1).
    ?&heap void   alloc(unsigned long size, unsigned long align);

    // Return an object to the pool (null: no-op). Aborts with a diagnostic
    // on double-free (SlabAllocator::new-created allocators only — see has_tags above).
    void          dealloc(?&heap void ptr);

    // Return the number of free slots remaining.
    unsigned long available() const;

    // The largest alignment alloc honours: the largest power of two dividing
    // obj_size, at most 8 (given an 8-aligned buffer: always for SlabAllocator::new).
    unsigned long max_align() const;

    // Free the backing buffer (only valid for heap-backed allocators from SlabAllocator::new).
    void          destroy();

    // Internal: build embedded free-list through the buffer.
    void          build_freelist_();

    // Internal: bytes between consecutive slots' payload pointers.
    unsigned long slot_stride_() const;
};

} // namespace std
