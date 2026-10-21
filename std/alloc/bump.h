// SafeC Standard Library — Bump Allocator
// O(1) alloc, O(1) reset. No individual free — reset reclaims all.
// Conforms to Allocator (allocator.h): typed allocation is alloc_t /
// alloc_array_t there.
#pragma once
#include <std/mem.h>
#include <std/alloc/allocator.h>

namespace std {

struct BumpAllocator {
    &heap void    base;    // heap-backed backing buffer
    int           owned;   // 1 if bump_new allocated base (destroy() frees it)
    unsigned long used;    // bytes currently allocated
    unsigned long cap;     // total buffer capacity in bytes

    // Allocate `size` bytes whose address is a multiple of `align` (any
    // power of two; 0 is taken as 1), padding the arena as needed. null
    // when the arena is exhausted or `align` isn't a power of two.
    ?&heap void   alloc(unsigned long size, unsigned long align);

    // A no-op: an arena can't give back one block. reset() frees them all.
    void          dealloc(?&heap void ptr);

    // Reset the allocator — all previous allocations become invalid.
    void          reset();

    // Return the number of bytes still available.
    unsigned long remaining() const;

    // Free the backing buffer (only valid for heap-backed allocators from bump_new).
    void          destroy();
};

// Initialize a bump allocator over an existing heap buffer.
struct BumpAllocator bump_init(&unsigned char buffer, unsigned long cap);

// Create a heap-backed bump allocator with the given capacity.
struct BumpAllocator bump_new(unsigned long cap);

} // namespace std
