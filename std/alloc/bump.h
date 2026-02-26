// SafeC Standard Library — Bump Allocator
// O(1) alloc, O(1) reset. No individual free — reset reclaims all.
#pragma once

struct BumpAllocator {
    &heap void    base;    // heap-backed backing buffer
    unsigned long used;    // bytes currently allocated
    unsigned long cap;     // total buffer capacity in bytes

    // Allocate `size` bytes with `align` alignment. Returns NULL on OOM.
    // `align` must be a power of two.
    &heap void    alloc(unsigned long size, unsigned long align);

    // Reset the allocator — all previous allocations become invalid.
    void          reset();

    // Return the number of bytes still available.
    unsigned long remaining() const;

    // Free the backing buffer (only valid for heap-backed allocators from bump_new).
    void          destroy();
};

// Initialize a bump allocator over an existing heap buffer.
struct BumpAllocator bump_init(&heap void buffer, unsigned long cap);

// Create a heap-backed bump allocator with the given capacity.
struct BumpAllocator bump_new(unsigned long cap);
