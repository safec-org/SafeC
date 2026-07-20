#pragma once
// SafeC Standard Library — Queue (FIFO, circular buffer)
// Amortized O(1) enqueue/dequeue. Grows automatically when full.

namespace std {

struct Queue {
    void*         data;
    unsigned long head;      // index of front element
    unsigned long tail;      // index where next element will be written
    unsigned long len;       // current element count
    unsigned long cap;       // total allocated slots
    unsigned long elem_size;
};

// ── Lifecycle ─────────────────────────────────────────────────────────────────
struct Queue queue_new(unsigned long elem_size);
struct Queue queue_with_cap(unsigned long elem_size, unsigned long cap);
void         queue_free(&Queue q);

// ── Core operations ───────────────────────────────────────────────────────────
int           queue_enqueue(&Queue q, const void* elem);
int           queue_dequeue(&Queue q, void* out);
void*         queue_front(&Queue q);   // peek front; NULL if empty
void*         queue_back(&Queue q);    // peek back; NULL if empty
unsigned long queue_len(&Queue q);
int           queue_is_empty(&Queue q);
void          queue_clear(&Queue q);

// ── Typed generic wrappers ────────────────────────────────────────────────────
generic<T>
int queue_enqueue_t(&Queue q, T val);

generic<T>
T* queue_front_t(&Queue q);

generic<T>
int queue_dequeue_t(&Queue q, T* out);

} // namespace std
