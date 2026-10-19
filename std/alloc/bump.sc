// SafeC Standard Library — Bump Allocator
#pragma once
#include <std/stddef.h>
#include <std/alloc/bump.h>

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);

inline struct BumpAllocator bump_init(&unsigned char buffer, unsigned long cap) {
    struct BumpAllocator a;
    unsafe { a.base = (&heap void)buffer; }
    a.owned = 0;
    a.used = (unsigned long)0;
    a.cap  = cap;
    return a;
}

inline struct BumpAllocator bump_new(unsigned long cap) {
    struct BumpAllocator a;
    unsafe { a.base = (&heap void)malloc(cap); }
    a.owned = 1;
    a.used = (unsigned long)0;
    a.cap  = cap;
    return a;
}

&heap void BumpAllocator::alloc(unsigned long size, unsigned long align) {
    unsigned long mask    = align - (unsigned long)1;
    unsigned long aligned = (self.used + mask) & ~mask;
    // 'aligned + size > self.cap' would wrap around for a large enough
    // 'size', silently passing the check and handing back an out-of-bounds
    // pointer; comparing against 'self.cap - aligned' instead can't
    // overflow, since 'aligned' is already checked <= self.cap here.
    if (aligned > self.cap || size > self.cap - aligned) {
        return (&heap void)0;
    }
    unsafe {
        &heap void ptr = (&heap void)((unsigned long)self.base + aligned);
        self.used = aligned + size;
        return ptr;
    }
}

inline void BumpAllocator::reset() {
    self.used = (unsigned long)0;
}

inline unsigned long BumpAllocator::remaining() const {
    return self.cap - self.used;
}

inline void BumpAllocator::destroy() {
    if (self.owned) { unsafe { free((void*)self.base); } }
    self.used = (unsigned long)0;
    self.cap  = (unsigned long)0;
}

} // namespace std
