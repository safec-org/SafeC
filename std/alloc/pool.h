// SafeC Standard Library — Pool Allocator
// Variable-size first-fit allocator with coalescing on free.
// Each allocation is preceded by a PoolBlock header.
// Conforms to Allocator and ArenaAllocator (allocator.h): typed allocation
// is alloc_t / alloc_array_t / dealloc_t there.
#pragma once
#include <std/mem.h>
#include <std/alloc/allocator.h>

namespace std {

struct PoolBlock {
    unsigned long size;       // usable size (excluding header)
    int           is_free;    // 1 = free, 0 = allocated
    ?&PoolBlock   next;       // next block in the list (walked inside unsafe)
};

struct PoolAllocator {
    &heap void    base;       // heap-backed managed memory
    int           owned;   // 1 if PoolAllocator::new allocated base (destroy() frees it)
    unsigned long cap;        // total buffer size
    ?&PoolBlock   head;       // first block in the list (walked inside unsafe)

    // A pool over a user-provided heap buffer (8-aligned) of `cap` bytes.
    static struct PoolAllocator over([]unsigned char buffer);

    // A heap-backed pool of `cap` bytes.
    static struct PoolAllocator new(unsigned long cap);

    // Allocate `size` bytes aligned to `align`. Every block is 8-aligned
    // (given an 8-aligned buffer: always for PoolAllocator::new), so align is honoured
    // up to 8 (0 is taken as 1). null if no free block fits, or `align` is
    // above 8 or not a power of two.
    ?&heap void   alloc(unsigned long size, unsigned long align);

    // Free a previously allocated pointer (null: no-op). Coalesces adjacent
    // free blocks. Aborts with a diagnostic on a double free.
    void          dealloc(?&heap void ptr);

    // Free every block at once: the whole buffer is one free block again
    // (blocks handed out before become invalid).
    void          reset();

    // Return total free bytes (sum of all free blocks).
    unsigned long available() const;

    // Free the backing buffer (only valid for heap-backed allocators from PoolAllocator::new).
    void          drop();
    void          destroy() consume;   // = drop()

    // Internal: return the fixed header size in bytes.
    unsigned long header_size_() const;
};

} // namespace std
