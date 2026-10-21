// SafeC Standard Library — Pool Allocator
// Variable-size first-fit allocator with coalescing on free.
// Each allocation is preceded by a PoolBlock header.
// Conforms to Allocator (allocator.h): typed allocation is alloc_t /
// alloc_array_t / dealloc_t there.
#pragma once
#include <std/mem.h>
#include <std/alloc/allocator.h>

namespace std {

struct PoolBlock {
    unsigned long size;       // usable size (excluding header)
    int           is_free;    // 1 = free, 0 = allocated
    void*         next;       // next PoolBlock* in the list (raw — unsafe only)
};

struct PoolAllocator {
    &heap void    base;       // heap-backed managed memory
    int           owned;   // 1 if pool_new allocated base (destroy() frees it)
    unsigned long cap;        // total buffer size
    void*         head;       // first PoolBlock* in the list (raw — unsafe only)

    // Allocate `size` bytes aligned to `align`. Every block is 8-aligned
    // (given an 8-aligned buffer: always for pool_new), so align is honoured
    // up to 8 (0 is taken as 1). null if no free block fits, or `align` is
    // above 8 or not a power of two.
    ?&heap void   alloc(unsigned long size, unsigned long align);

    // Free a previously allocated pointer (null: no-op). Coalesces adjacent
    // free blocks. Aborts with a diagnostic on a double free.
    void          dealloc(?&heap void ptr);

    // Return total free bytes (sum of all free blocks).
    unsigned long available() const;

    // Free the backing buffer (only valid for heap-backed allocators from pool_new).
    void          destroy();

    // Internal: return the fixed header size in bytes.
    unsigned long header_size_() const;
};

// Initialize over a user-provided heap buffer (8-aligned).
struct PoolAllocator pool_init(&unsigned char buffer, unsigned long cap);

// Create a heap-backed pool allocator.
struct PoolAllocator pool_new(unsigned long cap);

} // namespace std
