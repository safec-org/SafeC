// SafeC Standard Library — MPSC Ring Buffer implementation (see mpsc.h)
#pragma once
#include <std/sync/mpsc.h>
#include <std/sync/spinlock.sc>

namespace std {

extern void* malloc(unsigned long size);
extern void  free(void* ptr);
extern void* memcpy(void* dst, const void* src, unsigned long n);

inline struct MpscQueue mpsc_new(unsigned long elem_size, unsigned long cap) {
    struct MpscQueue q;
    unsafe { q.buffer = (&heap void)malloc(elem_size * cap); }
    q.elem_size = elem_size;
    q.cap       = cap;
    q.head      = 0UL;
    q.tail      = 0UL;
    q.count     = 0UL;
    q.lock      = spinlock_init();
    return q;
}

inline int MpscQueue::enqueue(const void* elem) {
    self.lock.lock();
    if (self.count == self.cap) {
        self.lock.unlock();
        return 0; // full
    }
    unsafe {
        unsigned long offset = self.tail * self.elem_size;
        void* dst = (void*)((unsigned long)self.buffer + offset);
        memcpy(dst, elem, self.elem_size);
    }
    self.tail  = (self.tail + 1UL) % self.cap;
    self.count = self.count + 1UL;
    self.lock.unlock();
    return 1;
}

inline int MpscQueue::dequeue(void* out) {
    self.lock.lock();
    if (self.count == 0UL) {
        self.lock.unlock();
        return 0; // empty
    }
    unsafe {
        unsigned long offset = self.head * self.elem_size;
        void* src = (void*)((unsigned long)self.buffer + offset);
        memcpy(out, src, self.elem_size);
    }
    self.head  = (self.head + 1UL) % self.cap;
    self.count = self.count - 1UL;
    self.lock.unlock();
    return 1;
}

inline int MpscQueue::is_empty() const {
    return self.count == 0UL ? 1 : 0;
}

inline int MpscQueue::is_full() const {
    return self.count == self.cap ? 1 : 0;
}

inline unsigned long MpscQueue::len() const {
    return self.count;
}

inline void MpscQueue::destroy() {
    unsafe { free((void*)self.buffer); }
    self.cap = 0UL;
}

generic<T>
int mpsc_enqueue_t(&stack MpscQueue q, T val) {
    unsafe { return q.enqueue((const void*)&val); }
}

generic<T>
int mpsc_dequeue_t(&stack MpscQueue q, T* out) {
    unsafe { return q.dequeue((void*)out); }
}

} // namespace std
