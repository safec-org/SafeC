// SafeC Standard Library — Pool Allocator
// Variable-size first-fit allocator with coalescing on free.
// Each allocation is preceded by a PoolBlock header.
#pragma once
#include <std/mem.h>

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

    // Allocate `size` bytes. Returns NULL if no suitable block found.
    &heap void    alloc(unsigned long size);

    // Free a previously allocated pointer. Coalesces adjacent free blocks.
    void          dealloc(?&heap void ptr);

    // Return total free bytes (sum of all free blocks).
    unsigned long available() const;

    // Free the backing buffer (only valid for heap-backed allocators from pool_new).
    void          destroy();

    // Internal: return the fixed header size in bytes.
    unsigned long header_size_() const;
};

// Initialize over a user-provided heap buffer.
struct PoolAllocator pool_init(&unsigned char buffer, unsigned long cap);

// Create a heap-backed pool allocator.
struct PoolAllocator pool_new(unsigned long cap);

// ── Typed allocation (recommended) ───────────────────────────────────────────
generic<T>
?&heap T pool_alloc_t(&PoolAllocator p) {
    unsafe { return (?&heap T)p.alloc(sizeof(T)); }
}

generic<T>
void pool_free_t(&PoolAllocator p, &heap T obj) {
    unsafe { p.dealloc((&heap void)obj); }
}

} // namespace std
