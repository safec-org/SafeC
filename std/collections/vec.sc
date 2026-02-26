// SafeC Standard Library — Vec implementation
#include "vec.h"
#include "../mem.h"

// ── Lifecycle ─────────────────────────────────────────────────────────────────
struct Vec vec_new(unsigned long elem_size) {
    struct Vec v;
    unsafe { v.data = (&heap void)0; }
    v.len      = 0UL;
    v.cap      = 0UL;
    v.elem_size = elem_size;
    return v;
}

struct Vec vec_with_cap(unsigned long elem_size, unsigned long cap) {
    struct Vec v;
    v.elem_size = elem_size;
    v.len       = 0UL;
    v.cap       = cap;
    unsafe {
        v.data = (&heap void)alloc(elem_size * cap);
        if ((void*)v.data == (void*)0) { v.cap = 0UL; }
    }
    return v;
}

void Vec::free() {
    unsafe {
        if ((void*)self.data != (void*)0) { dealloc((void*)self.data); }
    }
    unsafe { self.data = (&heap void)0; }
    self.len = 0UL;
    self.cap = 0UL;
}

// ── Capacity ──────────────────────────────────────────────────────────────────
unsigned long Vec::length() const { return self.len; }
unsigned long Vec::capacity() const { return self.cap; }
int           Vec::is_empty() const { return self.len == 0UL; }

int Vec::reserve(unsigned long new_cap) {
    if (new_cap <= self.cap) { return 1; }
    unsafe {
        void* nd = realloc_buf((void*)self.data, new_cap * self.elem_size);
        if (nd == (void*)0) { return 0; }
        self.data = (&heap void)nd;
        self.cap  = new_cap;
    }
    return 1;
}

void Vec::shrink() {
    if (self.len == 0UL) {
        unsafe {
            if ((void*)self.data != (void*)0) { dealloc((void*)self.data); }
            self.data = (&heap void)0;
        }
        self.cap = 0UL;
        return;
    }
    unsafe {
        void* nd = realloc_buf((void*)self.data, self.len * self.elem_size);
        if (nd != (void*)0) { self.data = (&heap void)nd; self.cap = self.len; }
    }
}

// ── Internal grow helper ──────────────────────────────────────────────────────
int Vec::grow_() {
    unsigned long new_cap = self.cap == 0UL ? 4UL : self.cap * 2UL;
    return self.reserve(new_cap);
}

// ── Element access ────────────────────────────────────────────────────────────
&heap void Vec::get_raw(unsigned long idx) {
    if (idx >= self.len) { return (&heap void)0; }
    unsafe { return (&heap void)((char*)self.data + idx * self.elem_size); }
}

int Vec::set_raw(unsigned long idx, const void* elem) {
    if (idx >= self.len) { return 0; }
    unsafe {
        char* dst = (char*)self.data + idx * self.elem_size;
        safe_memcpy((void*)dst, elem, self.elem_size);
    }
    return 1;
}

&heap void Vec::front_raw() {
    if (self.len == 0UL) { return (&heap void)0; }
    return self.data;
}

&heap void Vec::back_raw() {
    if (self.len == 0UL) { return (&heap void)0; }
    unsafe { return (&heap void)((char*)self.data + (self.len - 1UL) * self.elem_size); }
}

// ── Mutation ──────────────────────────────────────────────────────────────────
int Vec::push(const void* elem) {
    if (self.len == self.cap) {
        if (!self.grow_()) { return 0; }
    }
    unsafe {
        char* dst = (char*)self.data + self.len * self.elem_size;
        safe_memcpy((void*)dst, elem, self.elem_size);
    }
    self.len = self.len + 1UL;
    return 1;
}

int Vec::pop(void* out) {
    if (self.len == 0UL) { return 0; }
    self.len = self.len - 1UL;
    if (out != (void*)0) {
        unsafe {
            char* src = (char*)self.data + self.len * self.elem_size;
            safe_memcpy(out, (const void*)src, self.elem_size);
        }
    }
    return 1;
}

int Vec::insert(unsigned long idx, const void* elem) {
    if (idx > self.len) { return 0; }
    if (self.len == self.cap) {
        if (!self.grow_()) { return 0; }
    }
    unsafe {
        char* base = (char*)self.data;
        unsigned long bytes = (self.len - idx) * self.elem_size;
        if (bytes > 0UL)
            safe_memmove((void*)(base + (idx + 1UL) * self.elem_size),
                         (const void*)(base + idx * self.elem_size), bytes);
        safe_memcpy((void*)(base + idx * self.elem_size), elem, self.elem_size);
    }
    self.len = self.len + 1UL;
    return 1;
}

int Vec::remove(unsigned long idx, void* out) {
    if (idx >= self.len) { return 0; }
    if (out != (void*)0) {
        unsafe {
            char* src = (char*)self.data + idx * self.elem_size;
            safe_memcpy(out, (const void*)src, self.elem_size);
        }
    }
    unsafe {
        char* base = (char*)self.data;
        unsigned long bytes = (self.len - idx - 1UL) * self.elem_size;
        if (bytes > 0UL)
            safe_memmove((void*)(base + idx * self.elem_size),
                         (const void*)(base + (idx + 1UL) * self.elem_size), bytes);
    }
    self.len = self.len - 1UL;
    return 1;
}

void Vec::clear() { self.len = 0UL; }

int Vec::extend(const void* arr, unsigned long count) {
    if (!self.reserve(self.len + count)) { return 0; }
    unsafe {
        char* dst = (char*)self.data + self.len * self.elem_size;
        safe_memcpy((void*)dst, arr, count * self.elem_size);
    }
    self.len = self.len + count;
    return 1;
}

// ── Algorithms ────────────────────────────────────────────────────────────────
void Vec::reverse() {
    if (self.len <= 1UL) { return; }
    unsafe {
        char* tmp = (char*)alloc(self.elem_size);
        if (tmp == (char*)0) { return; }
        unsigned long lo = 0UL;
        unsigned long hi = self.len - 1UL;
        while (lo < hi) {
            char* a = (char*)self.data + lo * self.elem_size;
            char* b = (char*)self.data + hi * self.elem_size;
            safe_memcpy((void*)tmp, (const void*)a, self.elem_size);
            safe_memcpy((void*)a,   (const void*)b, self.elem_size);
            safe_memcpy((void*)b,   (const void*)tmp, self.elem_size);
            lo = lo + 1UL;
            hi = hi - 1UL;
        }
        dealloc((void*)tmp);
    }
}

extern void qsort(void* base, unsigned long n, unsigned long sz, void* cmp);

void Vec::sort(void* cmp) {
    if (self.len <= 1UL) { return; }
    unsafe { qsort((void*)self.data, self.len, self.elem_size, cmp); }
}

long long Vec::find(const void* key, void* cmp) const {
    unsafe {
        int (*cmpfn)(const void*, const void*) = (int (*)(const void*, const void*))cmp;
        int i = 0;
        while ((unsigned long)i < self.len) {
            char* elem = (char*)self.data + (unsigned long)i * self.elem_size;
            if (cmpfn(key, (const void*)elem) == 0) { return (long long)i; }
            i = i + 1;
        }
    }
    return -1LL;
}

int Vec::contains(const void* key, void* cmp) const {
    return self.find(key, cmp) >= 0LL;
}

struct Vec Vec::clone() const {
    struct Vec c = vec_with_cap(self.elem_size, self.len);
    if (self.len > 0UL && (void*)c.data != (void*)0) {
        unsafe { safe_memcpy((void*)c.data, (const void*)self.data, self.len * self.elem_size); }
        c.len = self.len;
    }
    return c;
}

void Vec::foreach(void* fn) {
    unsafe {
        void (*f)(void*, unsigned long) = (void (*)(void*, unsigned long))fn;
        unsigned long i = 0UL;
        while (i < self.len) {
            f((void*)((char*)self.data + i * self.elem_size), i);
            i = i + 1UL;
        }
    }
}

struct Vec Vec::filter(void* pred) const {
    struct Vec out = vec_new(self.elem_size);
    unsafe {
        int (*p)(const void*) = (int (*)(const void*))pred;
        unsigned long i = 0UL;
        while (i < self.len) {
            void* elem = (void*)((char*)self.data + i * self.elem_size);
            if (p((const void*)elem)) { out.push((const void*)elem); }
            i = i + 1UL;
        }
    }
    return out;
}

struct Vec Vec::map_raw(unsigned long out_elem_size, void* fn) const {
    struct Vec out = vec_with_cap(out_elem_size, self.len);
    unsafe {
        void (*f)(const void*, void*) = (void (*)(const void*, void*))fn;
        void* tmp = alloc(out_elem_size);
        unsigned long i = 0UL;
        while (i < self.len) {
            void* src = (void*)((char*)self.data + i * self.elem_size);
            f((const void*)src, tmp);
            out.push((const void*)tmp);
            i = i + 1UL;
        }
        dealloc(tmp);
    }
    return out;
}

// ── Typed generic wrappers ────────────────────────────────────────────────────
generic<T>
int vec_push_t(&stack Vec v, T val) {
    return v.push((const void*)&val);
}

generic<T>
T* vec_at(&stack Vec v, unsigned long idx) {
    return (T*)v.get_raw(idx);
}

generic<T>
int vec_pop_t(&stack Vec v, T* out) {
    return v.pop((void*)out);
}

generic<T>
struct Vec vec_from_arr(T* arr, unsigned long len) {
    struct Vec v = vec_new((unsigned long)sizeof(T));
    v.extend((const void*)arr, len);
    return v;
}
