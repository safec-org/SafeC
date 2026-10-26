// SafeC Standard Library — Bump Allocator
// O(1) alloc, O(1) reset. No individual free — reset reclaims all.
// Conforms to ArenaAllocator (allocator.h), and to Allocator with a no-op
// dealloc: typed allocation is alloc_t / alloc_array_t there.
#pragma once
#include <std/mem.h>
#include <std/alloc/allocator.h>

namespace std {

struct BumpAllocator {
    &heap unsigned char base;    // heap-backed backing buffer
    int           owned;   // 1 if BumpAllocator::new allocated base (destroy() frees it)
    unsigned long used;    // bytes currently allocated
    unsigned long cap;     // total buffer capacity in bytes

    // An arena over an existing heap buffer of `cap` bytes (not freed by destroy()).
    static struct BumpAllocator over([]unsigned char buffer);

    // A heap-backed arena of `cap` bytes (destroy() frees it).
    static struct BumpAllocator new(unsigned long cap);

    // Allocate `size` bytes whose address is a multiple of `align` (any
    // power of two; 0 is taken as 1), padding the arena as needed. null
    // when the arena is exhausted or `align` isn't a power of two.
    ?&heap unsigned char alloc(unsigned long size, unsigned long align);

    // A no-op: an arena can't give back one block. reset() frees them all.
    void          dealloc(?&heap unsigned char ptr);

    // Reset the allocator — all previous allocations become invalid.
    void          reset();

    // Return the number of bytes still available.
    unsigned long remaining() const;

    // Free the backing buffer (only valid for heap-backed allocators from BumpAllocator::new).
    void          drop();
    void          destroy() consume;   // = drop()
};

} // namespace std
