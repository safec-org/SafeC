// SafeC Standard Library — MPSC (Multi-Producer / Single-Consumer) Ring Buffer
//
// std::LFQueue (lockfree.h) is the SPSC case: lock-free, but only correct
// with exactly one producer and one consumer — concurrent producers race
// on the same 'head' index there with no coordination at all. MpscQueue is
// the multi-producer sibling: still a bounded ring buffer, still a
// non-blocking API (enqueue/dequeue return immediately with a full/empty
// indicator rather than parking the calling thread — same convention as
// LFQueue, unlike std::sync::channel.h's Channel, which blocks), but
// correctness across multiple concurrent producers comes from a
// std::Spinlock guarding every mutation (both enqueue and dequeue), not
// from a lock-free algorithm. That's a deliberate simplicity-over-
// cleverness choice: a correct, easily-verified spinlock-guarded queue
// over a hand-rolled lock-free MPSC ring buffer (which needs real CAS-
// based coordination between producers to be correct at all).
//
// "Single-consumer" is a contract this type does not itself enforce (the
// spinlock would still serialize concurrent dequeue() calls correctly) —
// it's a naming/intent signal for callers, the same way LFQueue's SPSC
// contract is documented rather than mechanically checked.
#pragma once
//
//   struct MpscQueue<int> q = std::MpscQueue::new(64UL);
//   q.enqueue(7);              // any thread; 0 if full
//   ?int v = q.dequeue();      // consumer; null if empty
//   q.destroy();
#include <std/sync/spinlock.h>
#include <std/mem.h>

namespace std {

generic<T>
struct MpscQueue {
    ?&heap T      buffer;
    unsigned long cap;        // capacity in elements
    unsigned long head;       // consumer reads here
    unsigned long tail;       // producers write here
    unsigned long count;      // pending element count
    struct Spinlock lock;     // guards head/tail/count and the buffer contents

    // Heap-allocates its own buffer. 'cap' need not be a power of two (unlike
    // LFQueue) — the spinlock already serializes index updates. A failed
    // allocation yields a queue with cap 0 (every enqueue reports full).
    static struct MpscQueue<T> new(unsigned long cap);

    int           enqueue(T val);   // 1 on success, 0 if full
    ?T            dequeue();        // null if empty
    int           is_empty() const;
    int           is_full() const;
    unsigned long length() const;
    void          destroy();        // frees the buffer
};

generic<T>
struct MpscQueue<T> MpscQueue::new(unsigned long cap) {
    struct MpscQueue<T> q;
    q.buffer = null;
    unsafe { q.buffer = (?&heap T)alloc_raw_(checked_mul_size(cap, sizeof(T))); }
    q.cap   = q.buffer.is_null() ? 0UL : cap;
    q.head  = 0UL;
    q.tail  = 0UL;
    q.count = 0UL;
    q.lock  = Spinlock::new();
    return q;
}

generic<T>
int MpscQueue::enqueue(T val) {
    self.lock.lock();
    if (self.count == self.cap) { self.lock.unlock(); return 0; }
    unsafe { self.buffer[self.tail] = val; }
    self.tail  = self.tail + 1UL == self.cap ? 0UL : self.tail + 1UL;
    self.count = self.count + 1UL;
    self.lock.unlock();
    return 1;
}

generic<T>
?T MpscQueue::dequeue() {
    self.lock.lock();
    if (self.count == 0UL) { self.lock.unlock(); return null; }
    T v;
    unsafe { v = self.buffer[self.head]; }
    self.head  = self.head + 1UL == self.cap ? 0UL : self.head + 1UL;
    self.count = self.count - 1UL;
    self.lock.unlock();
    return v;
}

generic<T>
int MpscQueue::is_empty() const { return self.count == 0UL; }

generic<T>
int MpscQueue::is_full() const { return self.count == self.cap; }

generic<T>
unsigned long MpscQueue::length() const { return self.count; }

generic<T>
void MpscQueue::destroy() {
    unsafe { dealloc_raw_((void*)self.buffer); }
    self.buffer = null;
    self.cap = 0UL;
}

} // namespace std
