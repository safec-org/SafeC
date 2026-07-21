// SafeC Standard Library — Pool Allocator
#pragma once
#include <std/alloc/pool.h>
#include <std/stderr_compat.h>

namespace std {

extern void* malloc(unsigned long size);
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

inline struct PoolAllocator pool_init(&heap void buffer, unsigned long cap) {
    struct PoolAllocator a;
    a.base = buffer;
    a.cap  = cap;
    unsigned long hdr = (unsigned long)24;
    unsafe {
        struct PoolBlock* first = (struct PoolBlock*)buffer;
        // cap < hdr means the caller's buffer can't even hold one block
        // header; 'cap - hdr' would underflow to a huge value and the pool
        // would believe it owns far more memory than it actually does, so
        // clamp to an empty (permanently exhausted) pool instead.
        if (cap >= hdr) {
            first->size = cap - hdr;
        } else {
            first->size = (unsigned long)0;
        }
        first->is_free = 1;
        first->next    = (void*)0;
        a.head = (void*)first;
    }
    return a;
}

inline struct PoolAllocator pool_new(unsigned long cap) {
    struct PoolAllocator a;
    unsigned long hdr = (unsigned long)24;
    // Always malloc at least enough room for one block header, regardless
    // of what 'cap' the caller asked for — otherwise a small 'cap' still
    // underflows 'size' below into a huge value while the actual backing
    // buffer stays tiny, corrupting the very first real allocation.
    unsigned long allocCap = cap;
    if (allocCap < hdr) { allocCap = hdr; }
    unsafe { a.base = (&heap void)malloc(allocCap); }
    a.cap = allocCap;
    unsafe {
        struct PoolBlock* first = (struct PoolBlock*)a.base;
        first->size    = allocCap - hdr;
        first->is_free = 1;
        first->next    = (void*)0;
        a.head = (void*)first;
    }
    return a;
}

&heap void PoolAllocator::alloc(unsigned long size) {
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
                return (&heap void)((unsigned long)cur + hdr);
            }
            cur = (struct PoolBlock*)cur->next;
        }
    }
    return (&heap void)0;
}

void PoolAllocator::dealloc(void* ptr) {
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

inline void PoolAllocator::destroy() {
    unsafe { free((void*)self.base); }
    self.head = (void*)0;
    self.cap  = (unsigned long)0;
}

} // namespace std
