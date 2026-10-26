// SafeC Standard Library — Slice<T>: a bounds-checked view of T elements
//
//   int arr[8];
//   struct Slice<int> s = std::Slice::from(arr[0..8]);
//   int third = s.get(2UL).default(0);
//   struct Slice<int> mid = s.sub(2UL, 5UL);   // elements 2,3,4
//   []int raw = s.as_slice();                  // back to the built-in slice
//
// The language's own []T is the buffer type std's APIs take and return;
// Slice<T> is a library view on top of it with checked get/set/sub, and
// the way to view a heap array from 'new T[n]' (Slice::of(first, n): the
// caller vouches for n) or allocate one (Slice::alloc). A Slice doesn't own
// its elements unless it came from Slice::alloc() (then free() releases
// them). The arr_* helpers below work on a []T directly.
#pragma once
#include <std/mem.h>
#include <std/panic.h>

namespace std {

generic<T>
struct Slice {
    ?&T           ptr;   // first element; null for an empty slice
    unsigned long len;   // element count

    static struct Slice<T> from([]T s);                      // view of a built-in slice
    static struct Slice<T> of(&T first, unsigned long len);  // view of len elements at first (a 'new T[len]')
    static struct Slice<T> alloc(unsigned long len);         // zeroed heap slice; empty on OOM

    int             in_bounds(unsigned long idx) const;
    ?&T             get(unsigned long idx) const;     // null if out of range
    &T              at(unsigned long idx) const;      // panics if out of range
    int             set(unsigned long idx, T val);    // 0 if out of range
    struct Slice<T> sub(unsigned long start, unsigned long end) const; // [start, end); empty if invalid
    unsigned long   length() const;
    int             is_empty() const;
    []T             as_slice() const;                  // the same elements as a built-in slice
    void            free();                            // only for Slice::alloc() slices
};

generic<T>
struct Slice<T> Slice::from([]T s) {
    struct Slice<T> v;
    v.ptr = null;
    v.len = 0UL;
    if (s.len != 0UL) { unsafe { v.ptr = (?&T)s.ptr; } v.len = s.len; }
    return v;
}

generic<T>
[]T Slice::as_slice() const {
    unsafe { return ((T*)self.ptr)[0..(self.ptr.is_null() ? 0UL : self.len)]; }
}

generic<T>
struct Slice<T> Slice::of(&T first, unsigned long len) {
    struct Slice<T> s;
    s.ptr = first;
    s.len = len;
    return s;
}

// Zero-initialised heap slice of 'len' elements; empty (null) on OOM.
generic<T>
struct Slice<T> Slice::alloc(unsigned long len) {
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
    unsafe { dealloc((?&heap T)self.ptr); }
    self.ptr = null;
    self.len = 0UL;
}

// ── Helpers on a built-in slice ([]T) ───────────────────────────────────
// s[i] already traps when out of range; arr_at / arr_set are the
// non-panicking forms.

generic<T>
?&T arr_at([]T s, unsigned long idx) {
    if (idx >= s.len) return null;
    return &s[idx];
}

generic<T>
int arr_set([]T s, unsigned long idx, T val) {
    if (idx >= s.len) return 0;
    s[idx] = val;
    return 1;
}

generic<T>
void arr_fill([]T s, T val) {
    unsigned long i = 0UL;
    while (i < s.len) { s[i] = val; i = i + 1UL; }
}

// Copies min(dst.len, src.len) elements; returns that count.
generic<T>
unsigned long arr_copy([]T dst, []const T src) {
    unsigned long n = dst.len < src.len ? dst.len : src.len;
    unsigned long i = 0UL;
    while (i < n) { dst[i] = src[i]; i = i + 1UL; }
    return n;
}

// Smallest / largest element by '<'; null for an empty slice.
generic<T>
?T arr_min([]const T s) {
    if (s.len == 0UL) return null;
    T best = s[0];
    unsigned long i = 1UL;
    while (i < s.len) { if (s[i] < best) best = s[i]; i = i + 1UL; }
    return best;
}

generic<T>
?T arr_max([]const T s) {
    if (s.len == 0UL) return null;
    T best = s[0];
    unsigned long i = 1UL;
    while (i < s.len) { if (s[i] > best) best = s[i]; i = i + 1UL; }
    return best;
}

generic<T>
void arr_reverse([]T s) {
    if (s.len < 2UL) return;
    unsigned long lo = 0UL;
    unsigned long hi = s.len - 1UL;
    while (lo < hi) {
        T tmp = s[lo];
        s[lo] = s[hi];
        s[hi] = tmp;
        lo = lo + 1UL;
        hi = hi - 1UL;
    }
}

} // namespace std
