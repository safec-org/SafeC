// SafeC Standard Library — Lock-free SPSC Ring Buffer Queue
// Single-Producer Single-Consumer bounded queue using atomic head/tail.
#pragma once

struct LFQueue {
    &heap void    buffer;     // heap-backed element storage
    unsigned long cap;        // capacity (must be power of 2)
    unsigned long elem_size;  // size of each element in bytes
    long long     head;       // write index (producer, atomic)
    long long     tail;       // read index (consumer, atomic)

    // Enqueue an element (raw bytes). Returns 1 on success, 0 if full.
    int           enqueue(const void* elem);

    // Dequeue an element (raw bytes). Copies to `out`. Returns 1 on success, 0 if empty.
    int           dequeue(void* out);

    // Check if queue is empty.
    int           is_empty() const;

    // Check if queue is full.
    int           is_full() const;

    // Return current element count.
    unsigned long len() const;

    // Free heap-backed queue.
    void          destroy();
};

// Initialize over a user-provided heap buffer. cap must be a power of 2.
struct LFQueue lfq_init(&heap void buffer, unsigned long elem_size, unsigned long cap);

// Create a heap-backed lock-free queue.
struct LFQueue lfq_new(unsigned long elem_size, unsigned long cap);

// ── Typed generic wrappers ──────────────────────────────────────────────────
generic<T>
int lfq_enqueue_t(&stack LFQueue q, T val);

generic<T>
int lfq_dequeue_t(&stack LFQueue q, T* out);
