// SafeC Standard Library — TLSF (Two-Level Segregated Fit) Allocator
// O(1) real-time allocator with bounded fragmentation.
// Uses a two-level bitmap index for fast size-class lookup.
// Conforms to Allocator (allocator.h): typed allocation is alloc_t /
// alloc_array_t / dealloc_t there.
#pragma once
#include <std/mem.h>
#include <std/alloc/allocator.h>

// First-level index: log2 buckets. Second-level: linear subdivisions.
// SL_COUNT = 16 subdivisions per FL bucket.
// FL_MAX = 32 covers allocations up to 4 GB.

#define TLSF_FL_MAX   32
#define TLSF_SL_COUNT 16

namespace std {

struct TlsfBlock {
    unsigned long size;       // block size (excluding header), low bit = free flag
    void*         prev_phys;  // previous physical block (for coalescing)
    void*         next_free;  // next free block in this size class
    void*         prev_free;  // prev free block in this size class

    // Query free bit (bit 0 of size field).
    int           is_free_() const;

    // Get actual usable size (mask off free bit).
    unsigned long block_size_() const;

    // Set free bit.
    void          set_free_();

    // Clear free bit (mark as used).
    void          set_used_();
};

struct TlsfAllocator {
    &heap void    base;                                       // heap-backed managed memory
    int           owned;   // 1 if tlsf_new allocated base (destroy() frees it)
    unsigned long cap;
    unsigned int  fl_bitmap;                                  // first-level bitmap
    unsigned int  sl_bitmap[TLSF_FL_MAX];                     // second-level bitmaps
    void*         blocks[TLSF_FL_MAX * TLSF_SL_COUNT];       // free-list heads (raw ptrs)

    // Allocate `size` bytes aligned to `align`. O(1). Every block is
    // 8-aligned (given an 8-aligned buffer: always for tlsf_new), so align
    // is honoured up to 8 (0 is taken as 1). null when no free block fits,
    // or `align` is above 8 or not a power of two.
    ?&heap void   alloc(unsigned long size, unsigned long align);

    // Free a previously allocated pointer (null: no-op). O(1). Aborts with
    // a diagnostic on a double free.
    void          dealloc(?&heap void ptr);

    // Free the backing buffer (only valid for heap-backed allocators).
    void          destroy();

    // Internal helpers.
    void          insert_(&TlsfBlock blk);
    void          remove_(&TlsfBlock blk);
    void*         find_(unsigned long size);
};

// Initialize over a user-provided heap buffer.
struct TlsfAllocator tlsf_init(&unsigned char buffer, unsigned long cap);

// Create a heap-backed TLSF allocator.
struct TlsfAllocator tlsf_new(unsigned long cap);

// Compute first-level and second-level indices for a given size.
void tlsf_mapping_(unsigned long size, &int fl, &int sl);

} // namespace std
