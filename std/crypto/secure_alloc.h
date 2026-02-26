// SafeC Standard Library — Secure Allocator
// Zeroes memory on free to prevent heap inspection / data remanence.
// Uses a slab allocator so object sizes are fixed — no fragmentation metadata
// can leak size information.
#pragma once
#include "../alloc/slab.h"

struct SecureAllocator {
    struct SlabAllocator slab;   // underlying slab

    // Allocate one object slot.  Returns NULL on OOM.
    &heap void alloc();

    // Zero and free an object slot.
    void       dealloc(void* ptr);

    // Return number of available slots.
    unsigned long available() const;

    // Zero all memory and destroy the backing buffer.
    void       destroy();
};

// Initialise a secure allocator for objects of `obj_size` bytes, `count` slots.
struct SecureAllocator secure_alloc_new(unsigned long obj_size,
                                        unsigned long count);
