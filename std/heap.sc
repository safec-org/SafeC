// SafeC Standard Library — Global Freestanding Heap
#pragma once
#include <std/stddef.h>
#include <std/heap.h>
#include <std/alloc/tlsf.h>

namespace std {

#ifdef __SAFEC_FREESTANDING__
// ── Freestanding path: static buffer + TLSF ──────────────────────────────────

extern unsigned char* memcpy(unsigned char* dst, const unsigned char* src, size_t n);

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
        heap_tlsf_ = TlsfAllocator::over(((unsigned char*)(unsigned char*)heap_buf_)[0..(unsigned long)HEAP_SIZE]);
    }
    heap_ready_ = 1;
}

// TLSF block, 8-aligned (all TLSF guarantees); a null address when it can't.
static &heap unsigned char heap_tlsf_alloc_(unsigned long size) {
    unsigned char* p = heap_tlsf_.alloc(size, 8UL);
    unsafe { return (&heap unsigned char)p; }
}

&heap unsigned char heap_alloc(unsigned long size) {
    if (heap_ready_ == 0) { return (&heap unsigned char)0; }
    return heap_tlsf_alloc_(size);
}

inline void heap_free(&heap unsigned char ptr) {
    if (heap_ready_ == 0) { return; }
    unsafe { if ((unsigned char*)ptr == (unsigned char*)0) { return; } }
    heap_tlsf_.dealloc(ptr);
}

&heap unsigned char heap_realloc(&heap unsigned char ptr, unsigned long new_size) {
    if (heap_ready_ == 0) { return (&heap unsigned char)0; }
    unsafe { if ((unsigned char*)ptr == (unsigned char*)0) { return heap_tlsf_alloc_(new_size); } }

    &heap unsigned char new_ptr = heap_tlsf_alloc_(new_size);
    unsafe { if ((unsigned char*)new_ptr == (unsigned char*)0) { return (&heap unsigned char)0; } }

    unsafe {
        // Read old block size from the TLSF header sitting 32 bytes before data.
        struct TlsfBlock* blk = (struct TlsfBlock*)((unsigned long)ptr - heap_hdr_());
        unsigned long old_size = blk->block_size_();
        unsigned long copy_n   = old_size < new_size ? old_size : new_size;
        memcpy((unsigned char*)new_ptr, (unsigned char*)ptr, copy_n);
    }
    heap_tlsf_.dealloc(ptr);
    return new_ptr;
}

inline unsigned long heap_capacity() {
    return (unsigned long)HEAP_SIZE;
}

#else
// ── Hosted path: malloc/free/realloc wrappers ─────────────────────────────────

extern unsigned char* malloc(size_t size);
extern void  free(unsigned char* ptr);
extern unsigned char* realloc(unsigned char* ptr, size_t new_size);
extern unsigned char* memcpy(unsigned char* dst, const unsigned char* src, size_t n);

void heap_init() { }  // no-op

&heap unsigned char heap_alloc(unsigned long size) {
    unsafe { return (&heap unsigned char)malloc(size); }
}

inline void heap_free(&heap unsigned char ptr) {
    unsafe { free((unsigned char*)ptr); }
}

&heap unsigned char heap_realloc(&heap unsigned char ptr, unsigned long new_size) {
    unsafe { return (&heap unsigned char)realloc((unsigned char*)ptr, new_size); }
}

inline unsigned long heap_capacity() {
    return (unsigned long)0;  // unbounded
}

#endif

} // namespace std
