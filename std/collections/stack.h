#pragma once
// SafeC Standard Library — Stack<T> (LIFO)
//
//   struct Stack<int> s = std::Stack::new();
//   s.push(1); s.push(2);
//   ?int top = s.pop();      // 2
//   s.free();
//
// Backed by one growing heap buffer: O(1) amortized push/pop. peek() hands
// out a '?&T' into that buffer, valid until the next push()/free().
//
// A Stack owns its elements: it is a Drop type, and T may own something
// itself (String, Vec, ...). push() takes the element by value (one it
// can't store is dropped), pop() moves it out, and clear()/drop() drop
// every element still on the stack.
#include <std/mem.h>

namespace std {

generic<T>
struct Stack {
    ?&heap T      data;
    unsigned long top;       // number of elements
    unsigned long cap;

    static struct Stack<T> new();                     // empty, no allocation
    static struct Stack<T> with_cap(unsigned long cap); // room for cap elements

    int           push(T value);   // 0 on OOM
    ?T            pop();           // null if empty
    ?&T           peek() const;    // null if empty
    unsigned long length() const;
    int           is_empty() const;
    void          clear();         // drop every element, keep the buffer
    int           reserve(unsigned long n);
    void          drop();          // drops the elements, frees the buffer (Drop)
    void          free() consume;   // = drop()
};

generic<T>
struct Stack<T> Stack::new() {
    struct Stack<T> s;
    s.data = null;
    s.top = 0UL;
    s.cap = 0UL;
    return s;
}

generic<T>
struct Stack<T> Stack::with_cap(unsigned long cap) {
    struct Stack<T> s = Stack::new();
    s.reserve(cap);
    return s;
}

generic<T>
int Stack::reserve(unsigned long n) {
    if (n <= self.cap) return 1;
    ?&heap T grown = null;
    unsafe { grown = (?&heap T)realloc_buf_raw_((unsigned char*)self.data, checked_mul_size(n, sizeof(T))); }
    if (grown.is_null()) return 0;
    self.data = grown;
    self.cap = n;
    return 1;
}

generic<T>
int Stack::push(T value) {
    if (self.top == self.cap) {
        if (!self.reserve(self.cap < 8UL ? 8UL : self.cap * 2UL)) { drop(value); return 0; }
    }
    unsafe { self.data[self.top] = value; }
    self.top = self.top + 1UL;
    return 1;
}

generic<T>
?T Stack::pop() {
    if (self.top == 0UL) return null;
    self.top = self.top - 1UL;
    unsafe { return self.data[self.top]; }
}

generic<T>
?&T Stack::peek() const {
    if (self.top == 0UL) return null;
    unsafe { return (?&T)&self.data[self.top - 1UL]; }
}

generic<T>
unsigned long Stack::length() const { return self.top; }

generic<T>
int Stack::is_empty() const { return self.top == 0UL; }

generic<T>
void Stack::clear() {
    if const (needs_drop(T)) {
        unsigned long i = 0UL;
        while (i < self.top) { unsafe { drop(self.data[i]); } i = i + 1UL; }
    }
    self.top = 0UL;
}

generic<T>
void Stack::free() { self.drop(); }

generic<T>
void Stack::drop() {
    self.clear();
    unsafe { dealloc(self.data); }
    self.data = null;
    self.top = 0UL;
    self.cap = 0UL;
}

} // namespace std
