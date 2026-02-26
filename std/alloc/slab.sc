// SafeC Standard Library — Slab Allocator
#pragma once
#include "slab.h"

extern void* malloc(unsigned long size);
extern void  free(void* ptr);

// Build free-list through the buffer: each free slot stores a pointer to the next.
void SlabAllocator::build_freelist_() {
    unsigned long i = (unsigned long)0;
    unsafe {
        while (i < self.cap - (unsigned long)1) {
            void* slot = (void*)((unsigned long)self.base + i * self.obj_size);
            void* next = (void*)((unsigned long)self.base + (i + (unsigned long)1) * self.obj_size);
            *(void**)slot = next;
            i = i + (unsigned long)1;
        }
        // Last slot points to NULL
        void* last = (void*)((unsigned long)self.base + i * self.obj_size);
        *(void**)last = (void*)0;
        self.free_head = (void*)self.base;
    }
}

struct SlabAllocator slab_init(&heap void buffer, unsigned long obj_size, unsigned long count) {
    struct SlabAllocator a;
    a.base = buffer;
    // Ensure minimum object size can hold a pointer
    if (obj_size < (unsigned long)8) {
        a.obj_size = (unsigned long)8;
    } else {
        a.obj_size = obj_size;
    }
    a.cap       = count;
    a.used      = (unsigned long)0;
    a.free_head = (void*)0;
    a.build_freelist_();
    return a;
}

struct SlabAllocator slab_new(unsigned long obj_size, unsigned long count) {
    struct SlabAllocator a;
    if (obj_size < (unsigned long)8) {
        a.obj_size = (unsigned long)8;
    } else {
        a.obj_size = obj_size;
    }
    a.cap  = count;
    a.used = (unsigned long)0;
    unsafe { a.base = (&heap void)malloc(a.obj_size * count); }
    a.free_head = (void*)0;
    a.build_freelist_();
    return a;
}

&heap void SlabAllocator::alloc() {
    if (self.free_head == (void*)0) {
        return (&heap void)0;
    }
    unsafe {
        void* obj    = self.free_head;
        self.free_head = *(void**)obj;
        self.used    = self.used + (unsigned long)1;
        return (&heap void)obj;
    }
}

void SlabAllocator::dealloc(void* ptr) {
    unsafe {
        *(void**)ptr   = self.free_head;
        self.free_head = ptr;
        self.used      = self.used - (unsigned long)1;
    }
}

unsigned long SlabAllocator::available() const {
    return self.cap - self.used;
}

void SlabAllocator::destroy() {
    unsafe { free((void*)self.base); }
    self.free_head = (void*)0;
    self.used      = (unsigned long)0;
    self.cap       = (unsigned long)0;
}
