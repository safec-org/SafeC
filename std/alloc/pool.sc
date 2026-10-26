// SafeC Standard Library — Pool Allocator
#pragma once
#include <std/stddef.h>
#include <std/alloc/pool.h>
#include <std/stderr_compat.h>

namespace std {

extern void* malloc(size_t size);
extern void  free(void* ptr);
// Not '#include <std/panic.h>' (function-like macros need
// --compat-preprocessor — see mem.sc's identical comment) and not an
// 'extern panic_at' either, to avoid making every PoolAllocator user also
// link a --compat-preprocessor build of panic.sc: same inline
// diagnostic-then-abort mem.sc's alloc_abort_ uses.
extern void  abort();
extern int   fprintf(void* stream, const char* fmt, ...);

static void pool_abort_(const char* msg) {
    unsafe { fprintf(SAFEC_STDERR_, "std::alloc (pool) fatal: %s\n", msg); }
    unsafe { abort(); }
}

inline unsigned long PoolAllocator::header_size_() const {
    // sizeof(PoolBlock) = size(ul) + is_free(int) + next(ptr) — 24 bytes conservative
    return (unsigned long)24;
}

// The whole buffer as one free block. cap < hdr means the buffer can't even
// hold one block header: writing one would overrun it, and 'cap - hdr'
// would underflow. The pool is left empty (permanently exhausted) instead.
inline void PoolAllocator::reset() {
    unsigned long hdr = (unsigned long)24;
    if (self.cap < hdr) {
        self.head = (void*)0;
        return;
    }
    unsafe {
        struct PoolBlock* first = (struct PoolBlock*)self.base;
        first->size    = self.cap - hdr;
        first->is_free = 1;
        first->next    = (void*)0;
        self.head = (void*)first;
    }
}

inline struct PoolAllocator PoolAllocator::over([]unsigned char buffer_s) {
    unsigned long cap = buffer_s.len;
    &unsigned char buffer;
    unsafe { buffer = (&unsigned char)buffer_s.ptr; }
    struct PoolAllocator a;
    unsafe { a.base = (&heap void)buffer; }
    a.owned = 0;
    a.cap  = cap;
    a.reset();
    return a;
}

inline struct PoolAllocator PoolAllocator::new(unsigned long cap) {
    struct PoolAllocator a;
    unsigned long hdr = (unsigned long)24;
    // Always malloc at least enough room for one block header, regardless
    // of what 'cap' the caller asked for — otherwise a small 'cap' still
    // underflows 'size' below into a huge value while the actual backing
    // buffer stays tiny, corrupting the very first real allocation.
    unsigned long allocCap = cap;
    if (allocCap < hdr) { allocCap = hdr; }
    unsafe { a.base = (&heap void)malloc(allocCap); }
    a.owned = 1;
    a.cap = allocCap;
    a.reset();
    return a;
}

?&heap void PoolAllocator::alloc(unsigned long size, unsigned long align) {
    // Every block header sits at an 8-aligned offset (24-byte headers,
    // sizes rounded to 8), so payloads are 8-aligned: alignments up to 8
    // come free; above that would need padding blocks, so null instead.
    if ((align & (align - 1UL)) != 0UL || align > 8UL) { return null; }
    if (size > self.cap) { return null; }   // also keeps the rounding below from wrapping
    // Align size to 8 bytes
    unsigned long aligned = (size + (unsigned long)7) & ~(unsigned long)7;
    unsigned long hdr     = self.header_size_();

    unsafe {
        struct PoolBlock* cur = (struct PoolBlock*)self.head;
        while (cur != (struct PoolBlock*)0) {
            if (cur->is_free == 1 && cur->size >= aligned) {
                // Split if remaining space can hold another block + 8 bytes
                if (cur->size >= aligned + hdr + (unsigned long)8) {
                    struct PoolBlock* split = (struct PoolBlock*)((unsigned long)cur + hdr + aligned);
                    split->size    = cur->size - aligned - hdr;
                    split->is_free = 1;
                    split->next    = cur->next;
                    cur->size = aligned;
                    cur->next = (void*)split;
                }
                cur->is_free = 0;
                return (?&heap void)((unsigned long)cur + hdr);
            }
            cur = (struct PoolBlock*)cur->next;
        }
    }
    return null;
}

void PoolAllocator::dealloc(?&heap void ptr_ref) {
    void* ptr = ptr_ref;
    if (ptr == (void*)0) { return; }
    unsigned long hdr = self.header_size_();
    unsafe {
        struct PoolBlock* blk = (struct PoolBlock*)((unsigned long)ptr - hdr);
        // Unlike mem.sc's alloc()/dealloc(), no quarantine is needed here
        // to make this reliable: a pool block's 'is_free' flag lives in
        // the pool's own backing buffer for the pool's whole lifetime —
        // freeing it never hands the memory back to a system allocator
        // that might immediately overwrite it with its own bookkeeping,
        // so the flag is always readable as whatever this allocator itself
        // last wrote.
        if (blk->is_free == 1) {
            pool_abort_("dealloc() called twice on the same pointer (double free)");
            return;
        }
        blk->is_free = 1;

        // Coalesce adjacent free blocks
        struct PoolBlock* cur = (struct PoolBlock*)self.head;
        while (cur != (struct PoolBlock*)0) {
            if (cur->is_free == 1 && cur->next != (void*)0) {
                struct PoolBlock* nxt = (struct PoolBlock*)cur->next;
                if (nxt->is_free == 1) {
                    cur->size = cur->size + hdr + nxt->size;
                    cur->next = nxt->next;
                    // Don't advance — check again in case of triple coalesce
                } else {
                    cur = (struct PoolBlock*)cur->next;
                }
            } else {
                cur = (struct PoolBlock*)cur->next;
            }
        }
    }
}

inline unsigned long PoolAllocator::available() const {
    unsigned long total = (unsigned long)0;
    unsafe {
        struct PoolBlock* cur = (struct PoolBlock*)self.head;
        while (cur != (struct PoolBlock*)0) {
            if (cur->is_free == 1) {
                total = total + cur->size;
            }
            cur = (struct PoolBlock*)cur->next;
        }
    }
    return total;
}

inline void PoolAllocator::destroy() { self.drop(); }

inline void PoolAllocator::drop() {
    if (self.owned) { unsafe { free((void*)self.base); } }
    self.head = (void*)0;
    self.cap  = (unsigned long)0;
}

} // namespace std
