// SafeC Standard Library — Lock-free SPSC Ring Buffer Queue
#pragma once
#include <std/sync/lockfree.h>
#include <std/mem.h>

namespace std {

extern void* malloc(unsigned long size);
extern void  free(void* ptr);
extern void* memcpy(void* dst, const void* src, unsigned long n);

// LFQueue's head/tail indexing uses '& (cap - 1)' throughout this file
// (see lfq_push/lfq_pop/lfq_len below) instead of '% cap' — only correct
// when cap is a power of two, silently producing wrong (not just slower)
// indices otherwise. Neither constructor validated that until now.
static unsigned long lfq_floor_pow2_(unsigned long n) {
    if (n == (unsigned long)0) { return (unsigned long)1; }
    unsigned long p = (unsigned long)1;
    while (p * (unsigned long)2 <= n) { p = p * (unsigned long)2; }
    return p;
}

static unsigned long lfq_ceil_pow2_(unsigned long n) {
    if (n <= (unsigned long)1) { return (unsigned long)1; }
    unsigned long p = (unsigned long)1;
    while (p < n) { p = p * (unsigned long)2; }
    return p;
}

inline struct LFQueue lfq_init(&heap void buffer, unsigned long elem_size, unsigned long cap) {
    struct LFQueue q;
    q.buffer    = buffer;
    q.elem_size = elem_size;
    // Round DOWN, never up: 'buffer' is caller-provided and sized for the
    // caller's original 'cap', so this may only ever use less of it, never
    // more (rounding up here could read/write past the caller's buffer).
    q.cap       = lfq_floor_pow2_(cap);
    q.head      = (long long)0;
    q.tail      = (long long)0;
    return q;
}

inline struct LFQueue lfq_new(unsigned long elem_size, unsigned long cap) {
    struct LFQueue q;
    // Round UP: this allocator mallocs its own buffer, so it can just size
    // it to match — no reason to hand back less capacity than requested.
    unsigned long roundedCap = lfq_ceil_pow2_(cap);
    unsafe { q.buffer = (&heap void)malloc(checked_mul_size(elem_size, roundedCap)); }
    q.elem_size = elem_size;
    q.cap       = roundedCap;
    q.head      = (long long)0;
    q.tail      = (long long)0;
    return q;
}

inline int LFQueue::enqueue(const void* elem) {
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

inline int LFQueue::dequeue(void* out) {
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

inline int LFQueue::is_empty() const {
    unsafe {
        long long h = atomic_load(&self.head);
        long long t = atomic_load(&self.tail);
        if (h == t) { return 1; }
        return 0;
    }
}

inline int LFQueue::is_full() const {
    unsafe {
        long long h      = atomic_load(&self.head);
        long long t      = atomic_load(&self.tail);
        long long next_h = (h + (long long)1) & (long long)(self.cap - (unsigned long)1);
        if (next_h == t) { return 1; }
        return 0;
    }
}

inline unsigned long LFQueue::len() const {
    unsafe {
        long long h = atomic_load(&self.head);
        long long t = atomic_load(&self.tail);
        return (unsigned long)((h - t + (long long)self.cap) & (long long)(self.cap - (unsigned long)1));
    }
}

inline void LFQueue::destroy() {
    unsafe { free((void*)self.buffer); }
    self.cap = (unsigned long)0;
}

generic<T>
int lfq_enqueue_t(&stack LFQueue q, T val) {
    unsafe { return q.enqueue((const void*)&val); }
}

generic<T>
int lfq_dequeue_t(&stack LFQueue q, T* out) {
    unsafe { return q.dequeue((void*)out); }
}

} // namespace std
