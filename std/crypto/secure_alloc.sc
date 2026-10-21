// SafeC Standard Library — Secure Allocator
#pragma once
#include <std/stddef.h>
#include <std/crypto/secure_alloc.h>

namespace std {

extern void* memset(void* p, int v, size_t n);

struct SecureAllocator secure_alloc_new(unsigned long obj_size,
                                        unsigned long count) {
    struct SecureAllocator sa;
    sa.slab = slab_new(obj_size, count);
    return sa;
}

&heap void SecureAllocator::alloc() {
    void* p = self.slab.alloc(self.slab.obj_size, 1UL);
    unsafe { return (&heap void)p; }
}

inline void SecureAllocator::dealloc(?&heap void ptr) {
    if (ptr.is_null()) { return; }
    // Zero the slot before returning it to the free list.
    unsafe { memset(ptr, 0, self.slab.obj_size); }
    self.slab.dealloc(ptr);
}

inline unsigned long SecureAllocator::available() const {
    return self.slab.available();
}

inline void SecureAllocator::destroy() {
    // Zero all backing memory before freeing.
    unsafe {
        if ((void*)self.slab.base != (void*)0) {
            unsigned long total = self.slab.obj_size * self.slab.cap;
            memset((void*)self.slab.base, 0, total);
        }
    }
    self.slab.destroy();
}

} // namespace std
