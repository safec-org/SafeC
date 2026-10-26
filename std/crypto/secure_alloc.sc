// SafeC Standard Library — Secure Allocator
#pragma once
#include <std/stddef.h>
#include <std/crypto/secure_alloc.h>

namespace std {

extern void* memset(void* p, int v, size_t n);

struct SecureAllocator SecureAllocator::new(unsigned long obj_size, unsigned long count) {
    struct SecureAllocator sa;
    sa.slab = SlabAllocator::new(obj_size, count);
    return sa;
}

inline ?&heap void SecureAllocator::alloc(unsigned long size, unsigned long align) {
    return self.slab.alloc(size, align);
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

inline void SecureAllocator::destroy() { self.drop(); }

inline void SecureAllocator::drop() {
    // Zero all backing memory before freeing.
    unsafe {
        if ((void*)self.slab.base != (void*)0) {
            // Every slot's whole stride (with SlabAllocator::new that is
            // obj_size plus its 8-byte tag), not just obj_size * cap.
            unsigned long total = self.slab.slot_stride_() * self.slab.cap;
            memset((void*)self.slab.base, 0, total);
        }
    }
    self.slab.destroy();
}

} // namespace std
