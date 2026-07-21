// SafeC Standard Library — Slab Allocator
#pragma once
#include <std/alloc/slab.h>
#include <std/mem.h>
#include <std/stderr_compat.h>

namespace std {

extern void* malloc(unsigned long size);
extern void  free(void* ptr);
// Same rationale as mem.sc/pool.sc for not using panic.h/panic_at here.
extern void  abort();
extern int   fprintf(void* stream, const char* fmt, ...);

static void slab_abort_(const char* msg) {
    unsafe { fprintf(SAFEC_STDERR_, "std::alloc (slab) fatal: %s\n", msg); }
    unsafe { abort(); }
}

#define SLAB_TAG_SIZE_    ((unsigned long)8)
#define SLAB_LIVE_MAGIC_  ((unsigned long)0x5A1B5A1B5A1B5A1B)
#define SLAB_FREE_MAGIC_  ((unsigned long)0xF5EEF5EEF5EEF5EE)

inline unsigned long SlabAllocator::slot_stride_() const {
    if (self.has_tags) { return self.obj_size + SLAB_TAG_SIZE_; }
    return self.obj_size;
}

// Build free-list through the buffer: each free slot stores a pointer to the
// next, and (if has_tags) each slot's tag is initialized to "free".
inline void SlabAllocator::build_freelist_() {
    unsigned long stride    = self.slot_stride_();
    unsigned long payloadOff = (unsigned long)0;
    if (self.has_tags) { payloadOff = SLAB_TAG_SIZE_; }

    unsigned long i = (unsigned long)0;
    unsafe {
        while (i < self.cap - (unsigned long)1) {
            void* slotBase = (void*)((unsigned long)self.base + i * stride);
            void* slot     = (void*)((unsigned long)slotBase + payloadOff);
            void* nextBase = (void*)((unsigned long)self.base + (i + (unsigned long)1) * stride);
            void* next     = (void*)((unsigned long)nextBase + payloadOff);
            if (self.has_tags) { *(unsigned long*)slotBase = SLAB_FREE_MAGIC_; }
            *(void**)slot = next;
            i = i + (unsigned long)1;
        }
        // Last slot points to NULL
        void* lastBase = (void*)((unsigned long)self.base + i * stride);
        void* last     = (void*)((unsigned long)lastBase + payloadOff);
        if (self.has_tags) { *(unsigned long*)lastBase = SLAB_FREE_MAGIC_; }
        *(void**)last = (void*)0;
        self.free_head = (void*)((unsigned long)self.base + payloadOff);
    }
}

inline struct SlabAllocator slab_init(&heap void buffer, unsigned long obj_size, unsigned long count) {
    struct SlabAllocator a;
    a.base = buffer;
    // Ensure minimum object size can hold a pointer
    if (obj_size < (unsigned long)8) {
        a.obj_size = (unsigned long)8;
    } else {
        a.obj_size = obj_size;
    }
    a.cap       = count;
    a.used      = (unsigned long)0;
    a.free_head = (void*)0;
    a.has_tags  = 0;
    a.build_freelist_();
    return a;
}

inline struct SlabAllocator slab_new(unsigned long obj_size, unsigned long count) {
    struct SlabAllocator a;
    if (obj_size < (unsigned long)8) {
        a.obj_size = (unsigned long)8;
    } else {
        a.obj_size = obj_size;
    }
    a.cap      = count;
    a.used     = (unsigned long)0;
    a.has_tags = 1;
    unsafe { a.base = (&heap void)malloc(checked_mul_size(a.obj_size + SLAB_TAG_SIZE_, count)); }
    a.free_head = (void*)0;
    a.build_freelist_();
    return a;
}

&heap void SlabAllocator::alloc() {
    if (self.free_head == (void*)0) {
        return (&heap void)0;
    }
    unsafe {
        void* obj    = self.free_head;
        self.free_head = *(void**)obj;
        self.used    = self.used + (unsigned long)1;
        if (self.has_tags) {
            unsigned long* tag = (unsigned long*)((unsigned long)obj - SLAB_TAG_SIZE_);
            *tag = SLAB_LIVE_MAGIC_;
        }
        return (&heap void)obj;
    }
}

void SlabAllocator::dealloc(void* ptr) {
    if (ptr == (void*)0) { return; }
    unsafe {
        if (self.has_tags) {
            unsigned long* tag = (unsigned long*)((unsigned long)ptr - SLAB_TAG_SIZE_);
            if (*tag == SLAB_FREE_MAGIC_) {
                slab_abort_("dealloc() called twice on the same pointer (double free)");
                return;
            }
            if (*tag != SLAB_LIVE_MAGIC_) {
                slab_abort_("dealloc() called on a pointer alloc() never returned "
                            "(mismatched allocator or corrupted heap)");
                return;
            }
            *tag = SLAB_FREE_MAGIC_;
        }
        *(void**)ptr   = self.free_head;
        self.free_head = ptr;
        self.used      = self.used - (unsigned long)1;
    }
}

inline unsigned long SlabAllocator::available() const {
    return self.cap - self.used;
}

inline void SlabAllocator::destroy() {
    unsafe { free((void*)self.base); }
    self.free_head = (void*)0;
    self.used      = (unsigned long)0;
    self.cap       = (unsigned long)0;
}

} // namespace std
