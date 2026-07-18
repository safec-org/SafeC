// SafeC Standard Library — Ring Buffer Implementation
#pragma once
#include <std/collections/ringbuffer.h>

namespace std {

// RingBuffer's '& self.mask' indexing (readable()/write()/etc. below) only
// gives correct results when cap is a power of two; nothing previously
// checked that. Rounds DOWN (never up) since 'buf' is caller-provided and
// sized for the caller's original 'cap' — this may only ever use less of
// it, never more.
static unsigned long ringbuf_floor_pow2_(unsigned long n) {
    if (n == (unsigned long)0) { return (unsigned long)1; }
    unsigned long p = (unsigned long)1;
    while (p * (unsigned long)2 <= n) { p = p * (unsigned long)2; }
    return p;
}

inline struct RingBuffer ring_init(&static unsigned char buf, unsigned long cap) {
    struct RingBuffer rb;
    unsigned long roundedCap = ringbuf_floor_pow2_(cap);
    rb.buf  = buf;
    rb.cap  = roundedCap;
    rb.mask = roundedCap - (unsigned long)1;
    rb.head = (unsigned long)0;
    rb.tail = (unsigned long)0;
    return rb;
}

inline unsigned long RingBuffer::readable() const {
    return self.head - self.tail;
}

inline unsigned long RingBuffer::writable() const {
    return self.cap - (self.head - self.tail);
}

inline int RingBuffer::is_empty() const {
    if (self.head == self.tail) { return 1; }
    return 0;
}

inline int RingBuffer::is_full() const {
    if ((self.head - self.tail) >= self.cap) { return 1; }
    return 0;
}

inline unsigned long RingBuffer::write(const &stack unsigned char data, unsigned long len) {
    unsigned long space = self.writable();
    if (len > space) { len = space; }
    unsigned long head = self.head;
    unsafe {
        unsigned long i = (unsigned long)0;
        while (i < len) {
            self.buf[(head + i) & self.mask] = data[i];
            i = i + (unsigned long)1;
        }
    }
    // Release store: ensure writes visible before head update.
    unsafe { asm volatile ("" : : : "memory"); }
    self.head = head + len;
    return len;
}

inline unsigned long RingBuffer::read(&stack unsigned char out, unsigned long len) {
    unsigned long avail = self.readable();
    if (len > avail) { len = avail; }
    unsigned long tail = self.tail;
    unsafe {
        unsigned long i = (unsigned long)0;
        while (i < len) {
            out[i] = self.buf[(tail + i) & self.mask];
            i = i + (unsigned long)1;
        }
    }
    unsafe { asm volatile ("" : : : "memory"); }
    self.tail = tail + len;
    return len;
}

inline unsigned long RingBuffer::peek(&stack unsigned char out, unsigned long len) const {
    unsigned long avail = self.readable();
    if (len > avail) { len = avail; }
    unsigned long tail = self.tail;
    unsafe {
        unsigned long i = (unsigned long)0;
        while (i < len) {
            out[i] = self.buf[(tail + i) & self.mask];
            i = i + (unsigned long)1;
        }
    }
    return len;
}

inline unsigned long RingBuffer::discard(unsigned long len) {
    unsigned long avail = self.readable();
    if (len > avail) { len = avail; }
    self.tail = self.tail + len;
    return len;
}

inline void RingBuffer::clear() {
    self.head = (unsigned long)0;
    self.tail = (unsigned long)0;
}

} // namespace std
