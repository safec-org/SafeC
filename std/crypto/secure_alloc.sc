// SafeC Standard Library — Secure Allocator
#pragma once
#include "secure_alloc.h"

extern void* memset(void* p, int v, unsigned long n);

struct SecureAllocator secure_alloc_new(unsigned long obj_size,
                                        unsigned long count) {
    struct SecureAllocator sa;
    sa.slab = slab_new(obj_size, count);
    return sa;
}

&heap void SecureAllocator::alloc() {
    return self.slab.alloc();
}

void SecureAllocator::dealloc(void* ptr) {
    if (ptr == (void*)0) { return; }
    // Zero the slot before returning it to the free list.
    unsafe { memset(ptr, 0, self.slab.obj_size); }
    self.slab.dealloc(ptr);
}

unsigned long SecureAllocator::available() const {
    return self.slab.available();
}

void SecureAllocator::destroy() {
    // Zero all backing memory before freeing.
    unsafe {
        if ((void*)self.slab.base != (void*)0) {
            unsigned long total = self.slab.obj_size * self.slab.cap;
            memset((void*)self.slab.base, 0, total);
        }
    }
    self.slab.destroy();
}
