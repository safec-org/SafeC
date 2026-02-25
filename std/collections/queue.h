#pragma once
// SafeC Standard Library — Queue (FIFO, circular buffer)
// Amortized O(1) enqueue/dequeue. Grows automatically when full.

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
void         queue_free(struct Queue* q);

// ── Core operations ───────────────────────────────────────────────────────────
int           queue_enqueue(struct Queue* q, const void* elem);
int           queue_dequeue(struct Queue* q, void* out);
void*         queue_front(struct Queue* q);   // peek front; NULL if empty
void*         queue_back(struct Queue* q);    // peek back; NULL if empty
unsigned long queue_len(struct Queue* q);
int           queue_is_empty(struct Queue* q);
void          queue_clear(struct Queue* q);

// ── Typed generic wrappers ────────────────────────────────────────────────────
generic<T>
int queue_enqueue_t(struct Queue* q, T val);

generic<T>
T* queue_front_t(struct Queue* q);

generic<T>
int queue_dequeue_t(struct Queue* q, T* out);
