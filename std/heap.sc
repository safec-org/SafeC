// SafeC Standard Library — Global Freestanding Heap
#pragma once
#include <std/stddef.h>
#include <std/heap.h>
#include <std/alloc/tlsf.h>

namespace std {

#ifdef __SAFEC_FREESTANDING__
// ── Freestanding path: static buffer + TLSF ──────────────────────────────────

extern void* memcpy(void* dst, const void* src, size_t n);

// Backing store: static array aligned to 8 bytes.
static unsigned long heap_buf_[(HEAP_SIZE + 7) / 8];  // word-aligned storage

// TLSF allocator operating over heap_buf_.
static struct TlsfAllocator heap_tlsf_;
static int heap_ready_ = 0;

// TLSF block header size (32 bytes: size + prev_phys + next_free + prev_free).
inline unsigned long heap_hdr_() {
    return (unsigned long)32;
}

inline void heap_init() {
    if (heap_ready_ != 0) { return; }
    unsafe {
        heap_tlsf_ = TlsfAllocator::over((&unsigned char)(void*)heap_buf_, (unsigned long)HEAP_SIZE);
    }
    heap_ready_ = 1;
}

// TLSF block, 8-aligned (all TLSF guarantees); a null address when it can't.
static &heap void heap_tlsf_alloc_(unsigned long size) {
    void* p = heap_tlsf_.alloc(size, 8UL);
    unsafe { return (&heap void)p; }
}

&heap void heap_alloc(unsigned long size) {
    if (heap_ready_ == 0) { return (&heap void)0; }
    return heap_tlsf_alloc_(size);
}

inline void heap_free(&heap void ptr) {
    if (heap_ready_ == 0) { return; }
    unsafe { if ((void*)ptr == (void*)0) { return; } }
    heap_tlsf_.dealloc(ptr);
}

&heap void heap_realloc(&heap void ptr, unsigned long new_size) {
    if (heap_ready_ == 0) { return (&heap void)0; }
    unsafe { if ((void*)ptr == (void*)0) { return heap_tlsf_alloc_(new_size); } }

    &heap void new_ptr = heap_tlsf_alloc_(new_size);
    unsafe { if ((void*)new_ptr == (void*)0) { return (&heap void)0; } }

    unsafe {
        // Read old block size from the TLSF header sitting 32 bytes before data.
        struct TlsfBlock* blk = (struct TlsfBlock*)((unsigned long)ptr - heap_hdr_());
        unsigned long old_size = blk->block_size_();
        unsigned long copy_n   = old_size < new_size ? old_size : new_size;
        memcpy((void*)new_ptr, (void*)ptr, copy_n);
    }
    heap_tlsf_.dealloc(ptr);
    return new_ptr;
}

inline unsigned long heap_capacity() {
    return (unsigned long)HEAP_SIZE;
}

#else
// ── Hosted path: malloc/free/realloc wrappers ─────────────────────────────────

extern void* malloc(size_t size);
extern void  free(void* ptr);
extern void* realloc(void* ptr, size_t new_size);
extern void* memcpy(void* dst, const void* src, size_t n);

void heap_init() { }  // no-op

&heap void heap_alloc(unsigned long size) {
    unsafe { return (&heap void)malloc(size); }
}

inline void heap_free(&heap void ptr) {
    unsafe { free((void*)ptr); }
}

&heap void heap_realloc(&heap void ptr, unsigned long new_size) {
    unsafe { return (&heap void)realloc((void*)ptr, new_size); }
}

inline unsigned long heap_capacity() {
    return (unsigned long)0;  // unbounded
}

#endif

} // namespace std
