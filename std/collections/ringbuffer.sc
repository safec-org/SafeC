// SafeC Standard Library — Ring Buffer Implementation
#pragma once
#include <std/collections/ringbuffer.h>

namespace std {

// RingBuffer's '& self.mask' indexing (length()/write()/etc. below) only
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

inline struct RingBuffer RingBuffer::over([]unsigned char buf) {
    struct RingBuffer rb;
    unsigned long roundedCap = buf.len == 0UL ? 0UL : ringbuf_floor_pow2_(buf.len);
    rb.buf  = buf;
    rb.cap  = roundedCap;
    rb.mask = roundedCap == 0UL ? 0UL : roundedCap - (unsigned long)1;
    rb.head = (unsigned long)0;
    rb.tail = (unsigned long)0;
    return rb;
}

inline unsigned long RingBuffer::length() const {
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

// write()/read()/peek() copy in at most two contiguous runs (the part up to
// the physical end of 'buf', then the wrapped remainder from its start)
// instead of one loop that recomputes '(pos + i) & mask' every iteration.
// Both do exactly the same bytes in the same order and stay allocation-free
// and dependency-free (no memcpy call, so this file's "Freestanding-safe"
// guarantee for real-time audio/DSP use is unaffected) — the only change is
// that each run is now a straight, unwrapped index sequence, which the
// auto-vectorizer can turn into real SIMD loads/stores; the masked version
// couldn't be vectorized as well with a data-dependent wraparound computed
// fresh every byte.
inline unsigned long RingBuffer::write([]const unsigned char data) {
    unsigned long len = data.len;
    unsigned long space = self.writable();
    if (len > space) { len = space; }
    unsigned long head = self.head;
    unsafe {
        unsigned long start = head & self.mask;
        unsigned long firstChunk = self.cap - start;
        if (firstChunk > len) { firstChunk = len; }
        unsigned long i = (unsigned long)0;
        while (i < firstChunk) {
            self.buf[start + i] = data[i];
            i = i + (unsigned long)1;
        }
        unsigned long remaining = len - firstChunk;
        i = (unsigned long)0;
        while (i < remaining) {
            self.buf[i] = data[firstChunk + i];
            i = i + (unsigned long)1;
        }
    }
    // Release store: ensure writes visible before head update.
    unsafe { asm volatile ("" : : : "memory"); }
    self.head = head + len;
    return len;
}

inline unsigned long RingBuffer::read([]unsigned char out) {
    unsigned long len = out.len;
    unsigned long avail = self.length();
    if (len > avail) { len = avail; }
    unsigned long tail = self.tail;
    unsafe {
        unsigned long start = tail & self.mask;
        unsigned long firstChunk = self.cap - start;
        if (firstChunk > len) { firstChunk = len; }
        unsigned long i = (unsigned long)0;
        while (i < firstChunk) {
            out[i] = self.buf[start + i];
            i = i + (unsigned long)1;
        }
        unsigned long remaining = len - firstChunk;
        i = (unsigned long)0;
        while (i < remaining) {
            out[firstChunk + i] = self.buf[i];
            i = i + (unsigned long)1;
        }
    }
    unsafe { asm volatile ("" : : : "memory"); }
    self.tail = tail + len;
    return len;
}

inline unsigned long RingBuffer::peek([]unsigned char out) const {
    unsigned long len = out.len;
    unsigned long avail = self.length();
    if (len > avail) { len = avail; }
    unsigned long tail = self.tail;
    unsafe {
        unsigned long start = tail & self.mask;
        unsigned long firstChunk = self.cap - start;
        if (firstChunk > len) { firstChunk = len; }
        unsigned long i = (unsigned long)0;
        while (i < firstChunk) {
            out[i] = self.buf[start + i];
            i = i + (unsigned long)1;
        }
        unsigned long remaining = len - firstChunk;
        i = (unsigned long)0;
        while (i < remaining) {
            out[firstChunk + i] = self.buf[i];
            i = i + (unsigned long)1;
        }
    }
    return len;
}

inline unsigned long RingBuffer::discard(unsigned long len) {
    unsigned long avail = self.length();
    if (len > avail) { len = avail; }
    self.tail = self.tail + len;
    return len;
}

inline int RingBuffer::enqueue(unsigned char b) {
    unsigned char one[1];
    one[0] = b;
    return self.write(one[0..1]) == (unsigned long)1 ? 1 : 0;
}

inline ?unsigned char RingBuffer::dequeue() {
    unsigned char one[1];
    one[0] = (unsigned char)0;
    if (self.read(one[0..1]) == (unsigned long)0) { return null; }
    return one[0];
}

inline void RingBuffer::clear() {
    self.head = (unsigned long)0;
    self.tail = (unsigned long)0;
}

} // namespace std
