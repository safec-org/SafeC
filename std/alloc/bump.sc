// SafeC Standard Library — Bump Allocator
#pragma once
#include "bump.h"

extern void* malloc(unsigned long size);
extern void  free(void* ptr);

struct BumpAllocator bump_init(&heap void buffer, unsigned long cap) {
    struct BumpAllocator a;
    a.base = buffer;
    a.used = (unsigned long)0;
    a.cap  = cap;
    return a;
}

struct BumpAllocator bump_new(unsigned long cap) {
    struct BumpAllocator a;
    unsafe { a.base = (&heap void)malloc(cap); }
    a.used = (unsigned long)0;
    a.cap  = cap;
    return a;
}

&heap void BumpAllocator::alloc(unsigned long size, unsigned long align) {
    unsigned long mask    = align - (unsigned long)1;
    unsigned long aligned = (self.used + mask) & ~mask;
    if (aligned + size > self.cap) {
        return (&heap void)0;
    }
    unsafe {
        &heap void ptr = (&heap void)((unsigned long)self.base + aligned);
        self.used = aligned + size;
        return ptr;
    }
}

void BumpAllocator::reset() {
    self.used = (unsigned long)0;
}

unsigned long BumpAllocator::remaining() const {
    return self.cap - self.used;
}

void BumpAllocator::destroy() {
    unsafe { free((void*)self.base); }
    self.used = (unsigned long)0;
    self.cap  = (unsigned long)0;
}
