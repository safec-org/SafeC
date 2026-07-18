// SafeC Standard Library — Slab Allocator
// Fixed-size object pool. O(1) alloc/free via embedded free-list.
#pragma once

namespace std {

struct SlabAllocator {
    &heap void    base;       // heap-backed backing buffer
    void*         free_head;  // head of embedded free-list (raw — unsafe arithmetic)
    unsigned long obj_size;   // size of each object (>= sizeof(void*))
    unsigned long cap;        // max number of objects
    unsigned long used;       // currently allocated count
    // 1 if each slot has an extra 8-byte double-free-detection tag ahead
    // of it (slab_new, where this allocator sized+owns 'base' itself, so
    // growing the per-slot stride is safe) — 0 if not (slab_init, where
    // 'base' is caller-provided and sized to the documented obj_size*count
    // contract, which adding a tag would silently overrun).
    int           has_tags;

    // Allocate one object. Returns NULL if pool is exhausted.
    &heap void    alloc();

    // Return an object to the pool. Aborts with a diagnostic on double-free
    // (slab_new-created allocators only — see has_tags above).
    void          dealloc(void* ptr);

    // Return the number of free slots remaining.
    unsigned long available() const;

    // Free the backing buffer (only valid for heap-backed allocators from slab_new).
    void          destroy();

    // Internal: build embedded free-list through the buffer.
    void          build_freelist_();

    // Internal: bytes between consecutive slots' payload pointers.
    unsigned long slot_stride_() const;
};

// Initialize over a user-provided heap buffer of exactly obj_size*count
// bytes. obj_size must be >= 8 (pointer size). No double-free detection
// (see has_tags) — the buffer size contract can't grow to fit a tag.
struct SlabAllocator slab_init(&heap void buffer, unsigned long obj_size, unsigned long count);

// Create a heap-backed slab allocator (double-free-detecting).
struct SlabAllocator slab_new(unsigned long obj_size, unsigned long count);

} // namespace std
