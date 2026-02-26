// SafeC Standard Library — Lock-free SPSC Ring Buffer Queue
#pragma once
#include "lockfree.h"

extern void* malloc(unsigned long size);
extern void  free(void* ptr);
extern void* memcpy(void* dst, const void* src, unsigned long n);

struct LFQueue lfq_init(&heap void buffer, unsigned long elem_size, unsigned long cap) {
    struct LFQueue q;
    q.buffer    = buffer;
    q.elem_size = elem_size;
    q.cap       = cap;
    q.head      = (long long)0;
    q.tail      = (long long)0;
    return q;
}

struct LFQueue lfq_new(unsigned long elem_size, unsigned long cap) {
    struct LFQueue q;
    unsafe { q.buffer = (&heap void)malloc(elem_size * cap); }
    q.elem_size = elem_size;
    q.cap       = cap;
    q.head      = (long long)0;
    q.tail      = (long long)0;
    return q;
}

int LFQueue::enqueue(const void* elem) {
    unsafe {
        long long h      = atomic_load(&self.head);
        long long t      = atomic_load(&self.tail);
        long long next_h = (h + (long long)1) & (long long)(self.cap - (unsigned long)1);
        if (next_h == t) {
            return 0; // full
        }
        unsigned long offset = (unsigned long)h * self.elem_size;
        void* dst = (void*)((unsigned long)self.buffer + offset);
        memcpy(dst, elem, self.elem_size);
        atomic_store(&self.head, next_h);
        return 1;
    }
}

int LFQueue::dequeue(void* out) {
    unsafe {
        long long h = atomic_load(&self.head);
        long long t = atomic_load(&self.tail);
        if (t == h) {
            return 0; // empty
        }
        unsigned long offset = (unsigned long)t * self.elem_size;
        void* src = (void*)((unsigned long)self.buffer + offset);
        memcpy(out, src, self.elem_size);
        long long next_t = (t + (long long)1) & (long long)(self.cap - (unsigned long)1);
        atomic_store(&self.tail, next_t);
        return 1;
    }
}

int LFQueue::is_empty() const {
    unsafe {
        long long h = atomic_load(&self.head);
        long long t = atomic_load(&self.tail);
        if (h == t) { return 1; }
        return 0;
    }
}

int LFQueue::is_full() const {
    unsafe {
        long long h      = atomic_load(&self.head);
        long long t      = atomic_load(&self.tail);
        long long next_h = (h + (long long)1) & (long long)(self.cap - (unsigned long)1);
        if (next_h == t) { return 1; }
        return 0;
    }
}

unsigned long LFQueue::len() const {
    unsafe {
        long long h = atomic_load(&self.head);
        long long t = atomic_load(&self.tail);
        return (unsigned long)((h - t + (long long)self.cap) & (long long)(self.cap - (unsigned long)1));
    }
}

void LFQueue::destroy() {
    unsafe { free((void*)self.buffer); }
    self.cap = (unsigned long)0;
}

generic<T>
int lfq_enqueue_t(&stack LFQueue q, T val) {
    return q.enqueue((const void*)&val);
}

generic<T>
int lfq_dequeue_t(&stack LFQueue q, T* out) {
    return q.dequeue((void*)out);
}
