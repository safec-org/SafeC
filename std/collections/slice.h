// SafeC Standard Library — Slice<T>: a bounds-checked view of T elements
//
//   int arr[8];
//   struct Slice<int> s = std::slice_of(&arr[0], 8UL);
//   int third = s.get(2UL).default(0);
//   struct Slice<int> mid = s.sub(2UL, 5UL);   // elements 2,3,4
//
// A Slice doesn't own its elements unless it came from slice_alloc() (then
// free() releases them). The arr_* helpers do the same bounds-checked work on
// a bare (reference, length) pair.
#pragma once
#include <std/mem.h>
#include <std/panic.h>

namespace std {

generic<T>
struct Slice {
    ?&T           ptr;   // first element; null for an empty slice
    unsigned long len;   // element count

    int             in_bounds(unsigned long idx) const;
    ?&T             get(unsigned long idx) const;     // null if out of range
    &T              at(unsigned long idx) const;      // panics if out of range
    int             set(unsigned long idx, T val);    // 0 if out of range
    struct Slice<T> sub(unsigned long start, unsigned long end) const; // [start, end); empty if invalid
    unsigned long   length() const;
    int             is_empty() const;
    void            free();                            // only for slice_alloc() slices
};

generic<T>
struct Slice<T> slice_of(&T first, unsigned long len) {
    struct Slice<T> s;
    s.ptr = first;
    s.len = len;
    return s;
}

// Zero-initialised heap slice of 'len' elements; empty (null) on OOM.
generic<T>
struct Slice<T> slice_alloc(unsigned long len) {
    struct Slice<T> s;
    s.ptr = null;
    s.len = 0UL;
    ?&T p = null;
    unsafe { p = (?&T)alloc_zeroed_raw_(checked_mul_size(len, sizeof(T))); }
    if (!p.is_null()) { s.ptr = p; s.len = len; }
    return s;
}

generic<T>
int Slice::in_bounds(unsigned long idx) const { return idx < self.len; }

generic<T>
?&T Slice::get(unsigned long idx) const {
    if (idx >= self.len) return null;
    match (self.ptr) {
        case some(p): unsafe { return (?&T)&p[idx]; }
        case null: { }
    }
    return null;
}

generic<T>
&T Slice::at(unsigned long idx) const {
    return expect_ref(self.get(idx), "Slice::at: index out of range");
}

generic<T>
int Slice::set(unsigned long idx, T val) {
    match (self.get(idx)) {
        case some(e): { *e = val; return 1; }
        case null: { }
    }
    return 0;
}

generic<T>
struct Slice<T> Slice::sub(unsigned long start, unsigned long end) const {
    struct Slice<T> out;
    out.ptr = null;
    out.len = 0UL;
    if (start < end && end <= self.len) {
        out.ptr = self.get(start);
        out.len = end - start;
    }
    return out;
}

generic<T>
unsigned long Slice::length() const { return self.len; }

generic<T>
int Slice::is_empty() const { return self.len == 0UL; }

generic<T>
void Slice::free() {
    unsafe { dealloc_raw_((void*)self.ptr); }
    self.ptr = null;
    self.len = 0UL;
}

// ── Bounds-checked helpers on a (first element, length) pair ────────────────

generic<T>
?&T arr_at(&T first, unsigned long len, unsigned long idx) {
    if (idx >= len) return null;
    unsafe { return (?&T)&first[idx]; }
}

generic<T>
int arr_set(&T first, unsigned long len, unsigned long idx, T val) {
    if (idx >= len) return 0;
    unsafe { first[idx] = val; }
    return 1;
}

generic<T>
void arr_fill(&T first, unsigned long len, T val) {
    unsafe {
        unsigned long i = 0UL;
        while (i < len) { first[i] = val; i = i + 1UL; }
    }
}

generic<T>
void arr_copy(&T dst, const &T src, unsigned long len) {
    unsafe {
        unsigned long i = 0UL;
        while (i < len) { dst[i] = src[i]; i = i + 1UL; }
    }
}

// Smallest / largest element by '<'; null for an empty array.
generic<T>
?T arr_min(const &T first, unsigned long len) {
    if (len == 0UL) return null;
    T best = *first;
    unsafe {
        unsigned long i = 1UL;
        while (i < len) { if (first[i] < best) best = first[i]; i = i + 1UL; }
    }
    return best;
}

generic<T>
?T arr_max(const &T first, unsigned long len) {
    if (len == 0UL) return null;
    T best = *first;
    unsafe {
        unsigned long i = 1UL;
        while (i < len) { if (first[i] > best) best = first[i]; i = i + 1UL; }
    }
    return best;
}

generic<T>
void arr_reverse(&T first, unsigned long len) {
    if (len < 2UL) return;
    unsafe {
        unsigned long lo = 0UL;
        unsigned long hi = len - 1UL;
        while (lo < hi) {
            T tmp = first[lo];
            first[lo] = first[hi];
            first[hi] = tmp;
            lo = lo + 1UL;
            hi = hi - 1UL;
        }
    }
}

} // namespace std
