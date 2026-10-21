// SafeC Standard Library — LFQueue<T>: lock-free SPSC ring buffer queue
//
//   struct LFQueue<long long> q = std::LFQueue::new(8UL);   // cap rounded up to a power of two
//   q.enqueue(5LL);              // producer thread; 0 if full
//   ?long long v = q.dequeue();  // consumer thread; null if empty
//   q.destroy();
//
// Exactly one producer and one consumer (see mpsc.h for several producers).
// head/tail are monotonically increasing element counters updated with
// atomic stores; only the slot index is 'counter & (cap - 1)'. That tells
// "full" (head - tail == cap) apart from "empty" (head == tail) without
// sacrificing a slot, so all 'cap' elements are usable.
#pragma once
#include <std/mem.h>

namespace std {

generic<T>
struct LFQueue {
    ?&heap T      buffer;
    unsigned long cap;        // power of two
    long long     head;       // elements ever enqueued (producer, atomic)
    long long     tail;       // elements ever dequeued (consumer, atomic)
    int           owned;      // 1 if LFQueue::new allocated the buffer

    // Uses a caller-provided buffer of at least 'cap' elements; cap is rounded
    // DOWN to a power of two so the queue never touches past the buffer.
    static struct LFQueue<T> over(&heap T buffer, unsigned long cap);

    // Allocates its own buffer; cap is rounded UP to a power of two. A failed
    // allocation yields a queue with cap 0 (every enqueue reports full).
    static struct LFQueue<T> new(unsigned long cap);

    int           enqueue(T val);   // 1 on success, 0 if full
    ?T            dequeue();        // null if empty
    int           is_empty() const;
    int           is_full() const;
    unsigned long length() const;
    void          destroy();        // frees the buffer if LFQueue::new allocated it
};

unsigned long lfq_floor_pow2_(unsigned long n);   // lockfree.sc
unsigned long lfq_ceil_pow2_(unsigned long n);

generic<T>
struct LFQueue<T> LFQueue::over(&heap T buffer, unsigned long cap) {
    struct LFQueue<T> q;
    q.buffer = buffer;
    q.cap    = lfq_floor_pow2_(cap);
    q.head   = 0LL;
    q.tail   = 0LL;
    q.owned  = 0;
    return q;
}

generic<T>
struct LFQueue<T> LFQueue::new(unsigned long cap) {
    struct LFQueue<T> q;
    unsigned long rounded = lfq_ceil_pow2_(cap);
    q.buffer = null;
    unsafe { q.buffer = (?&heap T)alloc_raw_(checked_mul_size(rounded, sizeof(T))); }
    q.cap   = q.buffer.is_null() ? 0UL : rounded;
    q.head  = 0LL;
    q.tail  = 0LL;
    q.owned = 1;
    return q;
}

generic<T>
int LFQueue::enqueue(T val) {
    unsafe {
        long long h = atomic_load(&self.head);
        long long t = atomic_load(&self.tail);
        if ((unsigned long)(h - t) >= self.cap) return 0;   // full
        self.buffer[(unsigned long)h & (self.cap - 1UL)] = val;
        atomic_store(&self.head, h + 1LL);                   // publish after the write
    }
    return 1;
}

generic<T>
?T LFQueue::dequeue() {
    T v;
    unsafe {
        long long h = atomic_load(&self.head);
        long long t = atomic_load(&self.tail);
        if (t == h) return null;                             // empty
        v = self.buffer[(unsigned long)t & (self.cap - 1UL)];
        atomic_store(&self.tail, t + 1LL);                   // release the slot after the read
    }
    return v;
}

generic<T>
int LFQueue::is_empty() const {
    unsafe { return atomic_load(&self.head) == atomic_load(&self.tail); }
}

generic<T>
int LFQueue::is_full() const {
    unsafe { return (unsigned long)(atomic_load(&self.head) - atomic_load(&self.tail)) >= self.cap; }
}

generic<T>
unsigned long LFQueue::length() const {
    unsafe { return (unsigned long)(atomic_load(&self.head) - atomic_load(&self.tail)); }
}

generic<T>
void LFQueue::destroy() {
    if (self.owned) { unsafe { dealloc_raw_((void*)self.buffer); } }
    self.buffer = null;
    self.cap = 0UL;
}

} // namespace std
