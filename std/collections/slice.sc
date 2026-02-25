// SafeC Standard Library — Generic Slice implementation
#pragma once
#include "slice.h"
#include <stdlib.h>
#include <string.h>

// ── Generic typed array operations ────────────────────────────────────────────
// All generics operate on T* directly; T is inferred from the pointer argument.

// Build a Slice from a typed pointer.  sizeof(T) fills in elem_size.
generic<T>
struct Slice slice_of(T* ptr, unsigned long len) {
    struct Slice s;
    s.ptr  = (void*)ptr;
    s.len  = len;
    unsafe { s.elem_size = sizeof(T); }
    return s;
}

// Bounds-checked pointer to element at idx.  Returns null on out-of-bounds.
generic<T>
T* arr_at(T* ptr, unsigned long len, unsigned long idx) {
    if (idx >= len) return (T*)0;
    unsafe { return ptr + idx; }
}

// Bounds-checked element set.
generic<T>
void arr_set(T* ptr, unsigned long len, unsigned long idx, T val) {
    if (idx < len) { unsafe { ptr[idx] = val; } }
}

// Unchecked element get (caller must ensure idx < len).
generic<T>
T arr_get(T* ptr, unsigned long idx) {
    unsafe { return ptr[idx]; }
}

// Fill every element with val.
generic<T>
void arr_fill(T* ptr, unsigned long len, T val) {
    unsigned long i = (unsigned long)0;
    while (i < len) { unsafe { ptr[i] = val; } i = i + (unsigned long)1; }
}

// Copy len elements from src to dst (non-overlapping).
generic<T>
void arr_copy(T* dst, T* src, unsigned long len) {
    unsigned long i = (unsigned long)0;
    while (i < len) {
        unsafe { dst[i] = src[i]; }
        i = i + (unsigned long)1;
    }
}

// Return minimum element (linear scan).
generic<T>
T arr_min(T* ptr, unsigned long len) {
    T best = arr_get(ptr, (unsigned long)0);
    unsigned long i = (unsigned long)1;
    while (i < len) {
        T v = arr_get(ptr, i);
        if (v < best) best = v;
        i = i + (unsigned long)1;
    }
    return best;
}

// Return maximum element (linear scan).
generic<T>
T arr_max(T* ptr, unsigned long len) {
    T best = arr_get(ptr, (unsigned long)0);
    unsigned long i = (unsigned long)1;
    while (i < len) {
        T v = arr_get(ptr, i);
        if (v > best) best = v;
        i = i + (unsigned long)1;
    }
    return best;
}

// Reverse elements in-place.
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

// ── Type-erased Slice API ─────────────────────────────────────────────────────

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

void slice_free(struct Slice* s) {
    unsafe { free(s->ptr); s->ptr = (void*)0; s->len = (unsigned long)0; }
}

const int slice_in_bounds(struct Slice s, unsigned long idx) {
    return idx < s.len;
}

void* slice_get_raw(struct Slice s, unsigned long idx) {
    if (idx >= s.len) return (void*)0;
    unsafe { return (void*)((char*)s.ptr + idx * s.elem_size); }
}

int slice_set_raw(struct Slice s, unsigned long idx, const void* val) {
    if (idx >= s.len) return 0;
    unsafe { memcpy((char*)s.ptr + idx * s.elem_size, val, s.elem_size); }
    return 1;
}

struct Slice slice_sub(struct Slice s, unsigned long start, unsigned long end) {
    struct Slice out;
    if (start > end || end > s.len) {
        out.ptr = (void*)0; out.len = (unsigned long)0; out.elem_size = s.elem_size;
        return out;
    }
    out.len       = end - start;
    out.elem_size = s.elem_size;
    unsafe { out.ptr = (void*)((char*)s.ptr + start * s.elem_size); }
    return out;
}

const unsigned long slice_len(struct Slice s) { return s.len; }
const int slice_is_empty(struct Slice s) { return s.len == (unsigned long)0; }
