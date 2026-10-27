#pragma once
// SafeC Standard Library — Queue<T> (FIFO, circular buffer)
//
//   struct Queue<int> q = std::Queue::new();
//   q.enqueue(1); q.enqueue(2);
//   ?int first = q.dequeue();   // 1
//   q.free();
//
// Amortized O(1) enqueue/dequeue; the ring grows (doubling, power-of-two
// capacity) when full. front()/back() hand out '?&T' into the ring, valid
// until the next enqueue()/free().
//
// A Queue owns its elements: it is a Drop type, and T may own something
// itself (String, Vec, ...). enqueue() takes the element by value (one it
// can't store is dropped), dequeue() moves it out, and clear()/drop() drop
// every element still queued.
#include <std/mem.h>

namespace std {

generic<T>
struct Queue {
    ?&heap T      data;
    unsigned long head;      // index of the front element
    unsigned long len;       // element count
    unsigned long cap;       // 0 or a power of two

    static struct Queue<T> new();                     // empty, no allocation
    static struct Queue<T> with_cap(unsigned long cap); // room for cap elements

    int           enqueue(T value);   // 0 on OOM
    ?T            dequeue();          // null if empty
    ?&T           front() const;      // null if empty
    ?&T           back() const;       // null if empty
    unsigned long length() const;
    int           is_empty() const;
    void          clear();            // drop every element, keep the buffer
    int           reserve(unsigned long n);
    void          drop();             // drops the elements, frees the buffer (Drop)
    void          free() consume;   // = drop()
};

generic<T>
struct Queue<T> Queue::new() {
    struct Queue<T> q;
    q.data = null;
    q.head = 0UL;
    q.len = 0UL;
    q.cap = 0UL;
    return q;
}

generic<T>
struct Queue<T> Queue::with_cap(unsigned long cap) {
    struct Queue<T> q = Queue::new();
    q.reserve(cap);
    return q;
}

// Re-lays the ring out linearly (front at index 0) in a buffer of
// max(8, n) slots rounded up to a power of two.
generic<T>
int Queue::reserve(unsigned long n) {
    if (n <= self.cap) return 1;
    unsigned long c = 8UL;
    while (c < n) { c = c * 2UL; }
    ?&heap T grown = null;
    unsafe { grown = (?&heap T)alloc_raw_(checked_mul_size(c, sizeof(T))); }
    if (grown.is_null()) return 0;
    unsafe {
        unsigned long mask = self.cap - 1UL;
        unsigned long i = 0UL;
        while (i < self.len) { grown[i] = self.data[(self.head + i) & mask]; i = i + 1UL; }
        dealloc(self.data);
    }
    self.data = grown;
    self.head = 0UL;
    self.cap = c;
    return 1;
}

generic<T>
int Queue::enqueue(T value) {
    if (self.len == self.cap) {
        if (!self.reserve(self.cap == 0UL ? 8UL : self.cap * 2UL)) { drop(value); return 0; }
    }
    unsafe { self.data[(self.head + self.len) & (self.cap - 1UL)] = value; }
    self.len = self.len + 1UL;
    return 1;
}

generic<T>
?T Queue::dequeue() {
    if (self.len == 0UL) return null;
    unsigned long at = self.head;
    self.head = (self.head + 1UL) & (self.cap - 1UL);
    self.len = self.len - 1UL;
    unsafe { return self.data[at]; }
}

generic<T>
?&T Queue::front() const {
    if (self.len == 0UL) return null;
    unsafe { return (?&T)&self.data[self.head]; }
}

generic<T>
?&T Queue::back() const {
    if (self.len == 0UL) return null;
    unsafe { return (?&T)&self.data[(self.head + self.len - 1UL) & (self.cap - 1UL)]; }
}

generic<T>
unsigned long Queue::length() const { return self.len; }

generic<T>
int Queue::is_empty() const { return self.len == 0UL; }

generic<T>
void Queue::clear() {
    if const (needs_drop(T)) {
        unsigned long i = 0UL;
        while (i < self.len) {
            unsafe { drop(self.data[(self.head + i) & (self.cap - 1UL)]); }
            i = i + 1UL;
        }
    }
    self.head = 0UL;
    self.len = 0UL;
}

generic<T>
void Queue::free() { self.drop(); }

generic<T>
void Queue::drop() {
    self.clear();
    unsafe { dealloc(self.data); }
    self.data = null;
    self.head = 0UL;
    self.len = 0UL;
    self.cap = 0UL;
}

} // namespace std
