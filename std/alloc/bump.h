// SafeC Standard Library — Bump Allocator
// O(1) alloc, O(1) reset. No individual free — reset reclaims all.
#pragma once
#include <std/mem.h>

namespace std {

struct BumpAllocator {
    &heap void    base;    // heap-backed backing buffer
    int           owned;   // 1 if bump_new allocated base (destroy() frees it)
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
struct BumpAllocator bump_init(&unsigned char buffer, unsigned long cap);

// Create a heap-backed bump allocator with the given capacity.
struct BumpAllocator bump_new(unsigned long cap);

// ── Typed allocation (recommended) ───────────────────────────────────────────
// One T / n Ts, aligned for T; null when the arena is exhausted. The memory
// is uninitialised and lives until reset()/destroy().
generic<T>
?&heap T bump_alloc_t(&BumpAllocator a) {
    unsafe { return (?&heap T)a.alloc(sizeof(T), alignof(T)); }
}

generic<T>
?&heap T bump_alloc_array_t(&BumpAllocator a, unsigned long n) {
    unsafe { return (?&heap T)a.alloc(checked_mul_size(n, sizeof(T)), alignof(T)); }
}

} // namespace std
