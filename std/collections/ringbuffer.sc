// SafeC Standard Library — Ring Buffer Implementation
#pragma once
#include "ringbuffer.h"

struct RingBuffer ring_init(&static unsigned char buf, unsigned long cap) {
    struct RingBuffer rb;
    rb.buf  = buf;
    rb.cap  = cap;
    rb.mask = cap - (unsigned long)1;
    rb.head = (unsigned long)0;
    rb.tail = (unsigned long)0;
    return rb;
}

unsigned long RingBuffer::readable() const {
    return self.head - self.tail;
}

unsigned long RingBuffer::writable() const {
    return self.cap - (self.head - self.tail);
}

int RingBuffer::is_empty() const {
    if (self.head == self.tail) { return 1; }
    return 0;
}

int RingBuffer::is_full() const {
    if ((self.head - self.tail) >= self.cap) { return 1; }
    return 0;
}

unsigned long RingBuffer::write(const &stack unsigned char data, unsigned long len) {
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

unsigned long RingBuffer::read(&stack unsigned char out, unsigned long len) {
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

unsigned long RingBuffer::peek(&stack unsigned char out, unsigned long len) const {
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

unsigned long RingBuffer::discard(unsigned long len) {
    unsigned long avail = self.readable();
    if (len > avail) { len = avail; }
    self.tail = self.tail + len;
    return len;
}

void RingBuffer::clear() {
    self.head = (unsigned long)0;
    self.tail = (unsigned long)0;
}
