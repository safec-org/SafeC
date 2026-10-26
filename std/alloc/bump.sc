// SafeC Standard Library — Bump Allocator
#pragma once
#include <std/stddef.h>
#include <std/alloc/bump.h>

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);

inline struct BumpAllocator BumpAllocator::over([]unsigned char buffer_s) {
    unsigned long cap = buffer_s.len;
    &unsigned char buffer;
    unsafe { buffer = (&unsigned char)buffer_s.ptr; }
    struct BumpAllocator a;
    unsafe { a.base = (&heap void)buffer; }
    a.owned = 0;
    a.used = (unsigned long)0;
    a.cap  = cap;
    return a;
}

inline struct BumpAllocator BumpAllocator::new(unsigned long cap) {
    struct BumpAllocator a;
    unsafe { a.base = (&heap void)malloc(cap); }
    a.owned = 1;
    a.used = (unsigned long)0;
    a.cap  = cap;
    return a;
}

?&heap void BumpAllocator::alloc(unsigned long size, unsigned long align) {
    if (align == 0UL) { align = 1UL; }
    if ((align & (align - 1UL)) != 0UL) { return null; }
    unsigned long mask = align - 1UL;
    // Align the address, not the offset: the buffer itself may be less
    // aligned than 'align' (malloc gives 16, a caller's buffer maybe less).
    unsigned long base;
    unsafe { base = (unsigned long)self.base; }
    unsigned long cur = base + self.used;
    if (mask > ~0UL - cur) { return null; }
    unsigned long aligned = ((cur + mask) & ~mask) - base;   // offset in the arena
    // 'aligned + size > self.cap' would wrap around for a large enough
    // 'size', silently passing the check and handing back an out-of-bounds
    // pointer; comparing against 'self.cap - aligned' instead can't
    // overflow, since 'aligned' is already checked <= self.cap here.
    if (aligned > self.cap || size > self.cap - aligned) {
        return null;
    }
    unsafe {
        ?&heap void ptr = (?&heap void)(base + aligned);
        self.used = aligned + size;
        return ptr;
    }
}

inline void BumpAllocator::dealloc(?&heap void ptr) {
    // An arena frees its blocks all at once (reset/destroy).
}

inline void BumpAllocator::reset() {
    self.used = (unsigned long)0;
}

inline unsigned long BumpAllocator::remaining() const {
    return self.cap - self.used;
}

inline void BumpAllocator::destroy() { self.drop(); }

inline void BumpAllocator::drop() {
    if (self.owned) { unsafe { free((void*)self.base); } }
    self.used = (unsigned long)0;
    self.cap  = (unsigned long)0;
}

} // namespace std
