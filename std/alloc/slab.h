// SafeC Standard Library — Slab Allocator
// Fixed-size object pool. O(1) alloc/free via embedded free-list.
#pragma once

struct SlabAllocator {
    &heap void    base;       // heap-backed backing buffer
    void*         free_head;  // head of embedded free-list (raw — unsafe arithmetic)
    unsigned long obj_size;   // size of each object (>= sizeof(void*))
    unsigned long cap;        // max number of objects
    unsigned long used;       // currently allocated count

    // Allocate one object. Returns NULL if pool is exhausted.
    &heap void    alloc();

    // Return an object to the pool.
    void          dealloc(void* ptr);

    // Return the number of free slots remaining.
    unsigned long available() const;

    // Free the backing buffer (only valid for heap-backed allocators from slab_new).
    void          destroy();

    // Internal: build embedded free-list through the buffer.
    void          build_freelist_();
};

// Initialize over a user-provided heap buffer. obj_size must be >= 8 (pointer size).
struct SlabAllocator slab_init(&heap void buffer, unsigned long obj_size, unsigned long count);

// Create a heap-backed slab allocator.
struct SlabAllocator slab_new(unsigned long obj_size, unsigned long count);
