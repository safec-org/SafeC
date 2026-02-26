// SafeC Standard Library — Pool Allocator
#pragma once
#include "pool.h"

extern void* malloc(unsigned long size);
extern void  free(void* ptr);

unsigned long PoolAllocator::header_size_() const {
    // sizeof(PoolBlock) = size(ul) + is_free(int) + next(ptr) — 24 bytes conservative
    return (unsigned long)24;
}

struct PoolAllocator pool_init(&heap void buffer, unsigned long cap) {
    struct PoolAllocator a;
    a.base = buffer;
    a.cap  = cap;
    unsafe {
        struct PoolBlock* first = (struct PoolBlock*)buffer;
        first->size    = cap - (unsigned long)24;
        first->is_free = 1;
        first->next    = (void*)0;
        a.head = (void*)first;
    }
    return a;
}

struct PoolAllocator pool_new(unsigned long cap) {
    struct PoolAllocator a;
    unsafe { a.base = (&heap void)malloc(cap); }
    a.cap = cap;
    unsafe {
        struct PoolBlock* first = (struct PoolBlock*)a.base;
        first->size    = cap - (unsigned long)24;
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
    unsigned long hdr = self.header_size_();
    unsafe {
        struct PoolBlock* blk = (struct PoolBlock*)((unsigned long)ptr - hdr);
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

unsigned long PoolAllocator::available() const {
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

void PoolAllocator::destroy() {
    unsafe { free((void*)self.base); }
    self.head = (void*)0;
    self.cap  = (unsigned long)0;
}
