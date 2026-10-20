#pragma once
// SafeC Standard Library — Vec<T> (dynamic array)
//
//   struct Vec<int> v = std::vec_new();        // T inferred from the target
//   v.push(3);
//   match (v.get(0UL)) { case null: ... case some(x): *x = 4; }
//   ?int last = v.pop();
//   v.free();
//
// Elements are stored by value in one contiguous heap buffer. Element access
// hands out '?&T' (null when out of range) — a reference into the buffer,
// valid until the next call that can reallocate it (push/insert/extend/
// reserve/shrink/free). Copies are shallow: a Vec of structs that own heap
// memory (String, Vec, ...) doesn't free those on clear()/free().
#include <std/mem.h>
#include <std/panic.h>
#include <std/cmp.h>

namespace std {

generic<T>
struct Vec {
    ?&heap T      data;  // element buffer; null while cap == 0
    unsigned long len;   // element count
    unsigned long cap;   // allocated capacity, in elements

    // ── Capacity ────────────────────────────────────────────────────────────
    int           reserve(unsigned long new_cap);   // ensure cap >= new_cap; 0 on OOM
    void          shrink();                          // cap := len
    unsigned long length() const;
    unsigned long total_capacity() const;
    int           is_empty() const;

    // ── Element access ──────────────────────────────────────────────────────
    ?&T           get(unsigned long idx) const;      // null if out of range
    &T            at(unsigned long idx) const;       // bounds-checked: panics if out of range
    int           set(unsigned long idx, T value);    // 0 if out of range
    ?&T           front();
    ?&T           back();

    // ── Mutation ────────────────────────────────────────────────────────────
    int           push(T value);                      // 0 on OOM
    ?T            pop();                              // null if empty
    int           insert(unsigned long idx, T value); // idx <= len; shifts right
    ?T            remove(unsigned long idx);          // shifts left; null if out of range
    void          clear();                            // len := 0, keeps the buffer
    int           extend(const &T items, unsigned long count); // append a C array

    // ── Algorithms ──────────────────────────────────────────────────────────
    void          reverse();
    void          sort(fn int(const &T, const &T) cmp);  // ascending by cmp (<0, 0, >0); see vec_sort/vec_sort_by for inlined orderings
    long long     find(const &T key, fn int(const &T, const &T) cmp) const; // -1 if absent
    int           contains(const &T key, fn int(const &T, const &T) cmp) const;
    struct Vec<T> clone() const;                      // shallow element copy
    void          foreach(fn void(&T, unsigned long) func);
    struct Vec<T> filter(fn int(const &T) pred) const;

    void          free();
};

generic<T>
struct Vec<T> vec_new() {
    struct Vec<T> v;
    v.data = null;
    v.len = 0UL;
    v.cap = 0UL;
    return v;
}

generic<T>
struct Vec<T> vec_with_cap(unsigned long cap) {
    struct Vec<T> v = vec_new();
    v.reserve(cap);
    return v;
}

// Copies 'len' elements starting at 'items'.
generic<T>
struct Vec<T> vec_from_arr(const &T items, unsigned long len) {
    struct Vec<T> v = vec_with_cap(len);
    v.extend(items, len);
    return v;
}

// ── Capacity ──────────────────────────────────────────────────────────────────

generic<T>
int Vec::reserve(unsigned long new_cap) {
    if (new_cap <= self.cap) return 1;
    unsigned long bytes = checked_mul_size(new_cap, sizeof(T));
    ?&heap T grown = null;
    unsafe { grown = (?&heap T)realloc_buf_raw_((void*)self.data, bytes); }
    if (grown.is_null()) return 0;
    self.data = grown;
    self.cap = new_cap;
    return 1;
}

generic<T>
void Vec::shrink() {
    if (self.len == self.cap) return;
    if (self.len == 0UL) {
        unsafe { dealloc_raw_((void*)self.data); }
        self.data = null;
        self.cap = 0UL;
        return;
    }
    ?&heap T smaller = null;
    unsafe { smaller = (?&heap T)realloc_buf_raw_((void*)self.data, self.len * sizeof(T)); }
    if (smaller.is_null()) return;   // keep the larger buffer
    self.data = smaller;
    self.cap = self.len;
}

generic<T>
unsigned long Vec::length() const { return self.len; }

generic<T>
unsigned long Vec::total_capacity() const { return self.cap; }

generic<T>
int Vec::is_empty() const { return self.len == 0UL; }

// ── Element access ────────────────────────────────────────────────────────────

generic<T>
?&T Vec::get(unsigned long idx) const {
    if (idx >= self.len) return null;
    unsafe { return (?&T)&self.data[idx]; }
}

generic<T>
&T Vec::at(unsigned long idx) const {
    if (idx >= self.len) panic_at("Vec::at: index out of range", "vec.h", 0);
    match (self.data) {
        case some(buf): unsafe { return &buf[idx]; }
        case null: { }
    }
    panic_at("Vec::at: empty vector", "vec.h", 0);
    return self.at(idx);   // unreachable
}

generic<T>
int Vec::set(unsigned long idx, T value) {
    if (idx >= self.len) return 0;
    unsafe { self.data[idx] = value; }
    return 1;
}

generic<T>
?&T Vec::front() { return self.get(0UL); }

generic<T>
?&T Vec::back() {
    if (self.len == 0UL) return null;
    return self.get(self.len - 1UL);
}

// ── Mutation ──────────────────────────────────────────────────────────────────

// The growth path is one out-of-line, cold call (realloc_array_raw_) that
// returns the new buffer, so a push loop's hot path is a compare, a store
// and an increment, with the buffer, length and capacity in registers.
generic<T>
int Vec::push(T value) {
    if (self.len == self.cap) {
        unsigned long next = self.cap < 4UL ? 4UL : self.cap * 2UL;
        ?&heap T grown = null;
        unsafe { grown = (?&heap T)realloc_array_raw_((void*)self.data, next, sizeof(T)); }
        if (grown.is_null()) return 0;
        self.data = grown;
        self.cap = next;
    }
    unsafe { self.data[self.len] = value; }
    self.len = self.len + 1UL;
    return 1;
}

generic<T>
?T Vec::pop() {
    if (self.len == 0UL) return null;
    self.len = self.len - 1UL;
    unsafe { return self.data[self.len]; }
}

generic<T>
int Vec::insert(unsigned long idx, T value) {
    if (idx > self.len) return 0;
    if (self.len == self.cap) {
        unsigned long next = self.cap < 4UL ? 4UL : self.cap * 2UL;
        ?&heap T grown = null;
        unsafe { grown = (?&heap T)realloc_array_raw_((void*)self.data, next, sizeof(T)); }
        if (grown.is_null()) return 0;
        self.data = grown;
        self.cap = next;
    }
    unsafe {
        unsigned long i = self.len;
        while (i > idx) { self.data[i] = self.data[i - 1UL]; i = i - 1UL; }
        self.data[idx] = value;
    }
    self.len = self.len + 1UL;
    return 1;
}

generic<T>
?T Vec::remove(unsigned long idx) {
    if (idx >= self.len) return null;
    unsafe {
        T out = self.data[idx];
        unsigned long i = idx;
        while (i + 1UL < self.len) { self.data[i] = self.data[i + 1UL]; i = i + 1UL; }
        self.len = self.len - 1UL;
        return out;
    }
}

generic<T>
void Vec::clear() { self.len = 0UL; }

generic<T>
int Vec::extend(const &T items, unsigned long count) {
    if (count == 0UL) return 1;
    if (self.len + count > self.cap) {
        unsigned long next = self.cap * 2UL;
        if (next < self.len + count) next = self.len + count;
        if (!self.reserve(next)) return 0;
    }
    unsafe {
        unsigned long i = 0UL;
        while (i < count) { self.data[self.len + i] = items[i]; i = i + 1UL; }
    }
    self.len = self.len + count;
    return 1;
}

// ── Algorithms ────────────────────────────────────────────────────────────────

generic<T>
void Vec::reverse() {
    if (self.len < 2UL) return;
    unsafe {
        unsigned long i = 0UL;
        unsigned long j = self.len - 1UL;
        while (i < j) {
            T t = self.data[i];
            self.data[i] = self.data[j];
            self.data[j] = t;
            i = i + 1UL;
            j = j - 1UL;
        }
    }
}

// Introsort over [lo, hi): median-of-three quicksort, insertion sort below 16
// elements, heapsort once recursion gets too deep (O(n log n) worst case).
// Element moves are whole-T copies, not byte loops, and no libc is needed.
generic<T, C>
void vec_insertion_sort_(&T a, long long lo, long long hi, const &C ord) {
    unsafe {
        long long i = lo + 1LL;
        while (i < hi) {
            T x = a[i];
            long long j = i - 1LL;
            while (j >= lo && ord.cmp(&a[j], &x) > 0) { a[j + 1LL] = a[j]; j = j - 1LL; }
            a[j + 1LL] = x;
            i = i + 1LL;
        }
    }
}

generic<T, C>
void vec_sift_down_(&T a, long long lo, long long root, long long n, const &C ord) {
    unsafe {
        while (1) {
            long long child = root * 2LL + 1LL;
            if (child >= n) return;
            if (child + 1LL < n && ord.cmp(&a[lo + child], &a[lo + child + 1LL]) < 0) child = child + 1LL;
            if (ord.cmp(&a[lo + root], &a[lo + child]) >= 0) return;
            T t = a[lo + root];
            a[lo + root] = a[lo + child];
            a[lo + child] = t;
            root = child;
        }
    }
}

generic<T, C>
void vec_heap_sort_(&T a, long long lo, long long hi, const &C ord) {
    long long n = hi - lo;
    long long i = n / 2LL;
    while (i > 0LL) { i = i - 1LL; vec_sift_down_(a, lo, i, n, ord); }
    long long end = n;
    unsafe {
        while (end > 1LL) {
            end = end - 1LL;
            T t = a[lo];
            a[lo] = a[lo + end];
            a[lo + end] = t;
            vec_sift_down_(a, lo, 0LL, end, ord);
        }
    }
}

generic<T, C>
void vec_introsort_(&T a, long long lo, long long hi, int depth, const &C ord) {
    while (hi - lo > 16LL) {
        if (depth == 0) { vec_heap_sort_(a, lo, hi, ord); return; }
        depth = depth - 1;
        unsafe {
            long long mid = lo + (hi - lo) / 2LL;
            // Median of three into a[mid], then use it as the pivot.
            if (ord.cmp(&a[mid], &a[lo]) < 0) { T t = a[mid]; a[mid] = a[lo]; a[lo] = t; }
            if (ord.cmp(&a[hi - 1LL], &a[lo]) < 0) { T t = a[hi - 1LL]; a[hi - 1LL] = a[lo]; a[lo] = t; }
            if (ord.cmp(&a[hi - 1LL], &a[mid]) < 0) { T t = a[hi - 1LL]; a[hi - 1LL] = a[mid]; a[mid] = t; }
            T pivot = a[mid];
            long long i = lo;
            long long j = hi - 1LL;
            while (1) {
                while (ord.cmp(&a[i], &pivot) < 0) i = i + 1LL;
                while (ord.cmp(&pivot, &a[j]) < 0) j = j - 1LL;
                if (i >= j) break;
                T t = a[i]; a[i] = a[j]; a[j] = t;
                i = i + 1LL;
                j = j - 1LL;
            }
            // Recurse into the smaller half, loop on the larger.
            if (j + 1LL - lo < hi - (j + 1LL)) {
                vec_introsort_(a, lo, j + 1LL, depth, ord);
                lo = j + 1LL;
            } else {
                vec_introsort_(a, j + 1LL, hi, depth, ord);
                hi = j + 1LL;
            }
        }
    }
    vec_insertion_sort_(a, lo, hi, ord);
}

// Sorts by a compile-time ordering type (Less<T>, Greater<T>, or your own
// struct with 'int cmp(const &T, const &T) const'): every comparison is a
// direct call the optimizer inlines, unlike sort()'s function pointer.
generic<T, C>
void vec_sort_by(&Vec<T> v, C ord) {
    if (v.len < 2UL) return;
    int depth = 0;
    unsigned long n = v.len;
    while (n > 1UL) { n = n >> 1UL; depth = depth + 2; }
    match (v.data) {
        case null: return;
        case some(buf): vec_introsort_(buf, 0LL, (long long)v.len, depth, &ord);
    }
}

// Ascending natural order ('<'), comparisons inlined.
generic<T>
void vec_sort(&Vec<T> v) {
    struct Less<T> ord = order_less();
    vec_sort_by(v, ord);
}

generic<T>
void Vec::sort(fn int(const &T, const &T) cmp) {
    struct FnOrder<T> ord = order_fn(cmp);
    vec_sort_by(self, ord);
}

generic<T>
long long Vec::find(const &T key, fn int(const &T, const &T) cmp) const {
    unsafe {
        unsigned long i = 0UL;
        while (i < self.len) {
            if (cmp(&self.data[i], key) == 0) return (long long)i;
            i = i + 1UL;
        }
    }
    return -1LL;
}

generic<T>
int Vec::contains(const &T key, fn int(const &T, const &T) cmp) const {
    return self.find(key, cmp) >= 0LL;
}

generic<T>
struct Vec<T> Vec::clone() const {
    struct Vec<T> out = vec_with_cap(self.len);
    match (self.data) {
        case null: { }
        case some(buf): out.extend(buf, self.len);
    }
    return out;
}

generic<T>
void Vec::foreach(fn void(&T, unsigned long) func) {
    unsafe {
        unsigned long i = 0UL;
        while (i < self.len) { func(&self.data[i], i); i = i + 1UL; }
    }
}

generic<T>
struct Vec<T> Vec::filter(fn int(const &T) pred) const {
    struct Vec<T> out = vec_new();
    unsafe {
        unsigned long i = 0UL;
        while (i < self.len) {
            if (pred(&self.data[i])) out.push(self.data[i]);
            i = i + 1UL;
        }
    }
    return out;
}

generic<T>
void Vec::free() {
    unsafe { dealloc_raw_((void*)self.data); }
    self.data = null;
    self.len = 0UL;
    self.cap = 0UL;
}

// Maps every element through 'f' into a new Vec<U>.
generic<T, U>
struct Vec<U> vec_map(const &Vec<T> v, fn U(const &T) f) {
    struct Vec<U> out = vec_with_cap(v.len);
    unsafe {
        unsigned long i = 0UL;
        while (i < v.len) { out.push(f(&v.data[i])); i = i + 1UL; }
    }
    return out;
}

} // namespace std
