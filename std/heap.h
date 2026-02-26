// SafeC Standard Library — Global Freestanding Heap
// Unified heap interface backed by:
//   Freestanding: static HEAP_SIZE-byte buffer managed by TLSF (O(1), real-time-safe).
//   Hosted:       thin malloc/free/realloc wrappers.
// Override buffer size with -DHEAP_SIZE=<bytes> at compile time.
#pragma once

#ifndef HEAP_SIZE
#define HEAP_SIZE 1048576  // 1 MiB default
#endif

// Initialize the global heap.  Must be called once before any heap_alloc().
// In hosted mode this is a no-op.
void          heap_init();

// Allocate `size` bytes from the global heap.  Returns NULL on failure.
&heap void    heap_alloc(unsigned long size);

// Free a block returned by heap_alloc() or heap_realloc().
void          heap_free(&heap void ptr);

// Resize an allocation.  Copies up to min(old_size, new_size) bytes.
// Returns NULL on failure; original block is NOT freed in that case.
&heap void    heap_realloc(&heap void ptr, unsigned long new_size);

// Return total heap capacity in bytes.
// In hosted mode returns 0 (unbounded).
unsigned long heap_capacity();
