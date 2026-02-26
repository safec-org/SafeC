// SafeC Standard Library — Lock-Free Ring Buffer (real-time audio / DSP safe)
// Single-producer / single-consumer; power-of-two capacity; no dynamic allocation.
// Uses atomic load/store for head and tail to guarantee correct ordering.
// Freestanding-safe.
#pragma once

#define RING_DEFAULT_CAP  4096   // bytes; override before including if needed

struct RingBuffer {
    &static unsigned char buf;   // backing store (set by ring_init)
    unsigned long     cap;    // capacity in bytes (must be power of two)
    unsigned long     mask;   // cap - 1
    volatile unsigned long head;   // write position (producer)
    volatile unsigned long tail;   // read position (consumer)

    // Return number of bytes currently readable.
    unsigned long readable() const;

    // Return number of bytes that can be written without blocking.
    unsigned long writable() const;

    // Return 1 if empty.
    int  is_empty() const;

    // Return 1 if full.
    int  is_full() const;

    // Write up to `len` bytes from `data`.  Returns bytes actually written.
    unsigned long write(const &stack unsigned char data, unsigned long len);

    // Read up to `len` bytes into `out`.  Returns bytes actually read.
    unsigned long read(&stack unsigned char out, unsigned long len);

    // Peek at `len` bytes without consuming them.  Returns bytes copied.
    unsigned long peek(&stack unsigned char out, unsigned long len) const;

    // Discard `len` bytes.  Returns bytes discarded.
    unsigned long discard(unsigned long len);

    // Reset (empty) the buffer.  Safe only when no concurrent access.
    void  clear();
};

// Initialise a RingBuffer with an existing backing store.
// `cap` must be a power of two.
struct RingBuffer ring_init(&static unsigned char buf, unsigned long cap);

// Static-capacity helper: declares a RingBuffer backed by a static array of `N` bytes.
// Usage: RING_STATIC(my_ring, 1024);
#define RING_STATIC(name, N)                          \
    static unsigned char name##_storage_[N];          \
    static struct RingBuffer name = { name##_storage_, N, N-1, 0, 0 }
