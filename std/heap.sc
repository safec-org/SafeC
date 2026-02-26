// SafeC Standard Library — Global Freestanding Heap
#pragma once
#include "heap.h"

#ifdef __SAFEC_FREESTANDING__
// ── Freestanding path: static buffer + TLSF ──────────────────────────────────
#include "alloc/tlsf.h"

extern void* memcpy(void* dst, const void* src, unsigned long n);

// Backing store: static array aligned to 8 bytes.
static unsigned long heap_buf_[(HEAP_SIZE + 7) / 8];  // word-aligned storage

// TLSF allocator operating over heap_buf_.
static struct TlsfAllocator heap_tlsf_;
static int heap_ready_ = 0;

// TLSF block header size (32 bytes: size + prev_phys + next_free + prev_free).
unsigned long heap_hdr_() {
    return (unsigned long)32;
}

void heap_init() {
    if (heap_ready_ != 0) { return; }
    unsafe {
        // Cast static storage to &heap void for tlsf_init.
        &heap void buf = (&heap void)(void*)heap_buf_;
        heap_tlsf_ = tlsf_init(buf, (unsigned long)HEAP_SIZE);
    }
    heap_ready_ = 1;
}

&heap void heap_alloc(unsigned long size) {
    if (heap_ready_ == 0) { return (&heap void)0; }
    return heap_tlsf_.alloc(size);
}

void heap_free(&heap void ptr) {
    if (heap_ready_ == 0) { return; }
    if ((void*)ptr == (void*)0) { return; }
    heap_tlsf_.free(ptr);
}

&heap void heap_realloc(&heap void ptr, unsigned long new_size) {
    if (heap_ready_ == 0) { return (&heap void)0; }
    if ((void*)ptr == (void*)0) { return heap_tlsf_.alloc(new_size); }

    &heap void new_ptr = heap_tlsf_.alloc(new_size);
    if ((void*)new_ptr == (void*)0) { return (&heap void)0; }

    unsafe {
        // Read old block size from the TLSF header sitting 32 bytes before data.
        struct TlsfBlock* blk = (struct TlsfBlock*)((unsigned long)ptr - heap_hdr_());
        unsigned long old_size = blk->block_size_();
        unsigned long copy_n   = old_size < new_size ? old_size : new_size;
        memcpy((void*)new_ptr, (void*)ptr, copy_n);
    }
    heap_tlsf_.free(ptr);
    return new_ptr;
}

unsigned long heap_capacity() {
    return (unsigned long)HEAP_SIZE;
}

#else
// ── Hosted path: malloc/free/realloc wrappers ─────────────────────────────────

extern void* malloc(unsigned long size);
extern void  free(void* ptr);
extern void* realloc(void* ptr, unsigned long new_size);
extern void* memcpy(void* dst, const void* src, unsigned long n);

void heap_init() { }  // no-op

&heap void heap_alloc(unsigned long size) {
    unsafe { return (&heap void)malloc(size); }
}

void heap_free(&heap void ptr) {
    unsafe { free((void*)ptr); }
}

&heap void heap_realloc(&heap void ptr, unsigned long new_size) {
    unsafe { return (&heap void)realloc((void*)ptr, new_size); }
}

unsigned long heap_capacity() {
    return (unsigned long)0;  // unbounded
}

#endif
