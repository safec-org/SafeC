// SafeC Standard Library — Generic Slice implementation
#pragma once
#include "slice.h"

extern void* malloc(unsigned long size);
extern void* calloc(unsigned long n, unsigned long size);
extern void  free(void* ptr);
extern void* memcpy(void* dst, const void* src, unsigned long n);

// ── Slice methods ─────────────────────────────────────────────────────────────

int Slice::in_bounds(unsigned long idx) const {
    return idx < self.len;
}

void* Slice::get_raw(unsigned long idx) const {
    if (idx >= self.len) { return (void*)0; }
    unsafe { return (void*)((char*)self.ptr + idx * self.elem_size); }
}

int Slice::set_raw(unsigned long idx, const void* val) {
    if (idx >= self.len) { return 0; }
    unsafe { memcpy((char*)self.ptr + idx * self.elem_size, val, self.elem_size); }
    return 1;
}

struct Slice Slice::sub(unsigned long start, unsigned long end) const {
    struct Slice out;
    if (start > end || end > self.len) {
        out.ptr      = (void*)0;
        out.len      = (unsigned long)0;
        out.elem_size = self.elem_size;
        return out;
    }
    out.len       = end - start;
    out.elem_size = self.elem_size;
    unsafe { out.ptr = (void*)((char*)self.ptr + start * self.elem_size); }
    return out;
}

unsigned long Slice::length() const { return self.len; }
int           Slice::is_empty() const { return self.len == (unsigned long)0; }

void Slice::free() {
    unsafe { free(self.ptr); self.ptr = (void*)0; self.len = (unsigned long)0; }
}

// ── Construction ──────────────────────────────────────────────────────────────

struct Slice slice_void_from(void* ptr, unsigned long len, unsigned long elem_size) {
    struct Slice s;
    s.ptr       = ptr;
    s.len       = len;
    s.elem_size = elem_size;
    return s;
}

struct Slice slice_void_alloc(unsigned long len, unsigned long elem_size) {
    struct Slice s;
    s.len       = len;
    s.elem_size = elem_size;
    unsafe { s.ptr = calloc(len, elem_size); }
    return s;
}

// ── Generic typed array operations ────────────────────────────────────────────

generic<T>
struct Slice slice_of(&stack T ptr, unsigned long len) {
    struct Slice s;
    unsafe {
        s.ptr       = (void*)ptr;
        s.len       = len;
        s.elem_size = sizeof(T);
    }
    return s;
}

generic<T>
T* arr_at(T* ptr, unsigned long len, unsigned long idx) {
    if (idx >= len) { return (T*)0; }
    unsafe { return ptr + idx; }
}

generic<T>
void arr_set(T* ptr, unsigned long len, unsigned long idx, T val) {
    if (idx < len) { unsafe { ptr[idx] = val; } }
}

generic<T>
T arr_get(T* ptr, unsigned long idx) {
    unsafe { return ptr[idx]; }
}

generic<T>
void arr_fill(T* ptr, unsigned long len, T val) {
    unsigned long i = (unsigned long)0;
    while (i < len) { unsafe { ptr[i] = val; } i = i + (unsigned long)1; }
}

generic<T>
void arr_copy(T* dst, T* src, unsigned long len) {
    unsigned long i = (unsigned long)0;
    while (i < len) {
        unsafe { dst[i] = src[i]; }
        i = i + (unsigned long)1;
    }
}

generic<T>
T arr_min(T* ptr, unsigned long len) {
    T best = arr_get(ptr, (unsigned long)0);
    unsigned long i = (unsigned long)1;
    while (i < len) {
        T v = arr_get(ptr, i);
        if (v < best) { best = v; }
        i = i + (unsigned long)1;
    }
    return best;
}

generic<T>
T arr_max(T* ptr, unsigned long len) {
    T best = arr_get(ptr, (unsigned long)0);
    unsigned long i = (unsigned long)1;
    while (i < len) {
        T v = arr_get(ptr, i);
        if (v > best) { best = v; }
        i = i + (unsigned long)1;
    }
    return best;
}

generic<T>
void arr_reverse(T* ptr, unsigned long len) {
    unsigned long lo = (unsigned long)0;
    unsigned long hi = len - (unsigned long)1;
    while (lo < hi) {
        T tmp = arr_get(ptr, lo);
        arr_set(ptr, len, lo, arr_get(ptr, hi));
        arr_set(ptr, len, hi, tmp);
        lo = lo + (unsigned long)1;
        hi = hi - (unsigned long)1;
    }
}
