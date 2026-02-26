// SafeC Standard Library — TLSF (Two-Level Segregated Fit) Allocator
// O(1) real-time allocator with bounded fragmentation.
// Uses a two-level bitmap index for fast size-class lookup.
#pragma once

// First-level index: log2 buckets. Second-level: linear subdivisions.
// SL_COUNT = 16 subdivisions per FL bucket.
// FL_MAX = 32 covers allocations up to 4 GB.

#define TLSF_FL_MAX   32
#define TLSF_SL_COUNT 16

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
    unsigned long cap;
    unsigned int  fl_bitmap;                                  // first-level bitmap
    unsigned int  sl_bitmap[TLSF_FL_MAX];                     // second-level bitmaps
    void*         blocks[TLSF_FL_MAX * TLSF_SL_COUNT];       // free-list heads (raw ptrs)

    // Allocate `size` bytes. Returns NULL on failure. O(1).
    &heap void    alloc(unsigned long size);

    // Free a previously allocated pointer. O(1).
    void          free(&heap void ptr);

    // Free the backing buffer (only valid for heap-backed allocators).
    void          destroy();

    // Internal helpers.
    void          insert_(&stack TlsfBlock blk);
    void          remove_(&stack TlsfBlock blk);
    void*         find_(unsigned long size);
};

// Initialize over a user-provided heap buffer.
struct TlsfAllocator tlsf_init(&heap void buffer, unsigned long cap);

// Create a heap-backed TLSF allocator.
struct TlsfAllocator tlsf_new(unsigned long cap);

// Compute first-level and second-level indices for a given size.
void tlsf_mapping_(unsigned long size, int* fl, int* sl);
