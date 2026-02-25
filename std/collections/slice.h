// SafeC Standard Library — Generic Slice
// A Slice wraps a typed T* pointer + a length, providing bounds-checked access.
// Because SafeC does not support generic structs, the struct stores void* and
// elem_size; generic<T> functions provide type-safe construction and access.
#pragma once

// ── Type-erased fat pointer ────────────────────────────────────────────────────

struct Slice {
    void*         ptr;       // pointer to first element
    unsigned long len;       // number of elements
    unsigned long elem_size; // stride in bytes
};

// ── Generic construction ──────────────────────────────────────────────────────
// T is inferred from the typed pointer argument.

// generic<T> struct Slice slice_of(T* ptr, unsigned long len);
// generic<T> struct Slice slice_alloc_t(unsigned long len);
// generic<T> T* slice_at(T* ptr, unsigned long len, unsigned long idx);
// generic<T> void arr_set(T* ptr, unsigned long len, unsigned long idx, T val);
// generic<T> T    arr_get(T* ptr, unsigned long idx);
// generic<T> void arr_fill(T* ptr, unsigned long len, T val);
// generic<T> void arr_copy(T* dst, T* src, unsigned long len);
// generic<T> T    arr_min(T* ptr, unsigned long len);
// generic<T> T    arr_max(T* ptr, unsigned long len);
// generic<T> void arr_reverse(T* ptr, unsigned long len);
//
// NOTE: The above generics are defined in slice.sc and expanded at each call
// site via monomorphization.  Include slice.sc along with slice.h.

// ── Type-erased Slice API (no generics needed) ────────────────────────────────

struct Slice  slice_void_from(void* ptr, unsigned long len, unsigned long elem_size);
struct Slice  slice_void_alloc(unsigned long len, unsigned long elem_size);
void          slice_free(struct Slice* s);
int           slice_in_bounds(struct Slice s, unsigned long idx);
void*         slice_get_raw(struct Slice s, unsigned long idx);
int           slice_set_raw(struct Slice s, unsigned long idx, const void* val);
struct Slice  slice_sub(struct Slice s, unsigned long start, unsigned long end);
unsigned long slice_len(struct Slice s);
int           slice_is_empty(struct Slice s);
