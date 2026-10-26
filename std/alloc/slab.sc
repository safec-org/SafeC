// SafeC Standard Library — Slab Allocator
#pragma once
#include <std/stddef.h>
#include <std/alloc/slab.h>
#include <std/mem.h>
#include <std/stderr_compat.h>

namespace std {

extern unsigned char* malloc(size_t size);
extern void  free(unsigned char* ptr);
// Same rationale as mem.sc/pool.sc for not using panic.h/panic_at here.
extern void  abort();
extern int   fprintf(CFile_* stream, const char* fmt, ...);

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
    if (self.cap == (unsigned long)0) { self.free_head = null; return; }  // cap - 1 would wrap
    unsafe {
        while (i < self.cap - (unsigned long)1) {
            unsigned char* slotBase = (unsigned char*)((unsigned long)self.base + i * stride);
            unsigned char* slot     = (unsigned char*)((unsigned long)slotBase + payloadOff);
            unsigned char* nextBase = (unsigned char*)((unsigned long)self.base + (i + (unsigned long)1) * stride);
            unsigned char* next     = (unsigned char*)((unsigned long)nextBase + payloadOff);
            if (self.has_tags) { *(unsigned long*)slotBase = SLAB_FREE_MAGIC_; }
            *(unsigned char**)slot = next;
            i = i + (unsigned long)1;
        }
        // Last slot points to NULL
        unsigned char* lastBase = (unsigned char*)((unsigned long)self.base + i * stride);
        unsigned char* last     = (unsigned char*)((unsigned long)lastBase + payloadOff);
        if (self.has_tags) { *(unsigned long*)lastBase = SLAB_FREE_MAGIC_; }
        *(unsigned char**)last = (unsigned char*)0;
        self.free_head = (?&SlabLink_)((unsigned long)self.base + payloadOff);
    }
}

// Slot size: at least one pointer (free slots hold the free-list link) and
// a multiple of 8, so every slot — and the link inside it — stays aligned.
static unsigned long slab_slot_size_(unsigned long obj_size) {
    unsigned long sz = obj_size < 8UL ? 8UL : obj_size;
    return (sz + 7UL) & ~7UL;
}

// Rounding the slot size (and aligning the buffer's start) can fit fewer
// slots than buffer.len / obj_size; cap says how many there are.
inline struct SlabAllocator SlabAllocator::over([]unsigned char buffer, unsigned long obj_size) {
    struct SlabAllocator a;
    unsigned long bytes = buffer.len;
    unsigned long addr;
    unsafe { addr = (unsigned long)buffer.ptr; }
    unsigned long skip = ((addr + 7UL) & ~7UL) - addr;
    unsafe { a.base = (&heap unsigned char)(addr + skip); }
    a.owned = 0;
    a.obj_size  = slab_slot_size_(obj_size);
    a.cap       = bytes > skip ? (bytes - skip) / a.obj_size : 0UL;
    a.used      = (unsigned long)0;
    a.free_head = null;
    a.has_tags  = 0;
    a.build_freelist_();
    return a;
}

inline struct SlabAllocator SlabAllocator::new(unsigned long obj_size, unsigned long count) {
    struct SlabAllocator a;
    a.obj_size = slab_slot_size_(obj_size);
    a.cap      = count;
    a.used     = (unsigned long)0;
    a.has_tags = 1;
    unsafe { a.base = (&heap unsigned char)malloc(checked_mul_size(a.obj_size + SLAB_TAG_SIZE_, count)); }
    a.owned = 1;
    a.free_head = null;
    a.build_freelist_();
    return a;
}

inline unsigned long SlabAllocator::max_align() const {
    // Slot payloads sit at base + i*stride (+ the 8-byte tag), and the tag
    // doesn't change the low bits below 8: obj_size's lowest set bit, capped.
    unsigned long low = self.obj_size & (~self.obj_size + 1UL);
    if (low == 0UL || low > 8UL) { return 8UL; }
    return low;
}

?&heap unsigned char SlabAllocator::alloc(unsigned long size, unsigned long align) {
    if ((align & (align - 1UL)) != 0UL || align > self.max_align()) { return null; }
    if (size > self.obj_size) { return null; }
    if (self.free_head.is_null()) {
        return null;
    }
    unsafe {
        struct SlabLink_* obj = (struct SlabLink_*)self.free_head;
        self.free_head = obj->next;
        self.used    = self.used + (unsigned long)1;
        if (self.has_tags) {
            unsigned long* tag = (unsigned long*)((unsigned long)obj - SLAB_TAG_SIZE_);
            *tag = SLAB_LIVE_MAGIC_;
        }
        return (?&heap unsigned char)obj;
    }
}

void SlabAllocator::dealloc(?&heap unsigned char ptr_ref) {
    unsigned char* ptr = ptr_ref;
    if (ptr == (unsigned char*)0) { return; }
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
        struct SlabLink_* link = (struct SlabLink_*)ptr;
        link->next     = self.free_head;
        self.free_head = (?&SlabLink_)link;
        self.used      = self.used - (unsigned long)1;
    }
}

inline unsigned long SlabAllocator::available() const {
    return self.cap - self.used;
}

inline void SlabAllocator::reset() {
    self.used = (unsigned long)0;
    self.build_freelist_();
}

inline void SlabAllocator::destroy() { self.drop(); }

inline void SlabAllocator::drop() {
    if (self.owned) { unsafe { free((unsigned char*)self.base); } }
    self.free_head = null;
    self.used      = (unsigned long)0;
    self.cap       = (unsigned long)0;
}

} // namespace std
