// SafeC Standard Library — Secure Allocator
// Zeroes memory on free to prevent heap inspection / data remanence.
// Uses a slab allocator so object sizes are fixed — no fragmentation metadata
// can leak size information.
// Conforms to Allocator (std/alloc/allocator.h): typed allocation is
// alloc_t / dealloc_t there, and the zeroing on dealloc applies to them too.
//
//   struct SecureAllocator sa = std::SecureAllocator::new(32UL, 4UL);
//   ?&heap Key k = std::alloc_t(&sa);   // null if sizeof(Key) > 32 or full
//   ...
//   std::dealloc_t(&sa, k);             // zeroed, then back on the free list
//   sa.destroy();                       // zeroes every slot, frees the buffer
#pragma once
#include <std/alloc/slab.h>

namespace std {

struct SecureAllocator {
    struct SlabAllocator slab;   // underlying slab

    // A secure allocator of `count` slots of `obj_size` bytes each
    // (heap-backed, double-free-detecting: SlabAllocator::new).
    static struct SecureAllocator new(unsigned long obj_size, unsigned long count);

    // One slot for `size` bytes aligned to `align` (a power of two; 0 is
    // taken as 1). null when every slot is in use, size > obj_size, or
    // `align` is above slab.max_align() (at most 8).
    ?&heap void alloc(unsigned long size, unsigned long align);

    // Zero and free an object slot (null: no-op).
    void       dealloc(?&heap void ptr);

    // Return number of available slots.
    unsigned long available() const;

    // Zero all memory and destroy the backing buffer.
    void       drop();
    void       destroy() consume;   // = drop()
};

} // namespace std
