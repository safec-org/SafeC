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
#include <std/sync/spinlock.h>

namespace std {

struct MpscQueue {
    &heap void    buffer;
    unsigned long cap;        // capacity in elements
    unsigned long elem_size;
    unsigned long head;       // consumer reads here
    unsigned long tail;       // producers write here
    unsigned long count;      // pending element count
    struct Spinlock lock;     // guards head/tail/count and the buffer contents

    // Copies 'elem' into the queue. Returns 1 on success, 0 if full.
    int           enqueue(const void* elem);

    // Copies the front element into 'out' and removes it. Returns 1 on
    // success, 0 if empty.
    int           dequeue(void* out);

    int           is_empty() const;
    int           is_full() const;
    unsigned long len() const;

    // Frees the backing buffer.
    void          destroy();
};

// Heap-allocates its own backing buffer. 'cap' need not be a power of two
// (unlike LFQueue) — the spinlock already serializes index updates, so
// there's no lock-free-algorithm reason to require it.
struct MpscQueue mpsc_new(unsigned long elem_size, unsigned long cap);

// ── Typed generic wrappers ──────────────────────────────────────────────────
generic<T>
int mpsc_enqueue_t(&stack MpscQueue q, T val);

generic<T>
int mpsc_dequeue_t(&stack MpscQueue q, T* out);

} // namespace std
