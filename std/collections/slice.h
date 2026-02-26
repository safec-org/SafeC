// SafeC Standard Library — Generic Slice
// A Slice wraps a typed T* pointer + a length, providing bounds-checked access.
// Because SafeC does not support generic structs, the struct stores void* and
// elem_size; generic<T> functions provide type-safe construction and access.
#pragma once

// ── Type-erased fat pointer ────────────────────────────────────────────────────

struct Slice {
    void*         ptr;       // pointer to first element (may point into any region)
    unsigned long len;       // number of elements
    unsigned long elem_size; // stride in bytes

    // ── Methods ────────────────────────────────────────────────────────────────
    int           in_bounds(unsigned long idx) const;
    void*         get_raw(unsigned long idx) const;     // NULL if OOB
    int           set_raw(unsigned long idx, const void* val);
    struct Slice  sub(unsigned long start, unsigned long end) const;
    unsigned long length() const;
    int           is_empty() const;
    void          free();
};

// ── Construction ──────────────────────────────────────────────────────────────
struct Slice  slice_void_from(void* ptr, unsigned long len, unsigned long elem_size);
struct Slice  slice_void_alloc(unsigned long len, unsigned long elem_size);

// ── Generic typed construction ────────────────────────────────────────────────
// T is inferred from the typed pointer argument.

generic<T>
struct Slice slice_of(&stack T ptr, unsigned long len);

generic<T>
T* arr_at(T* ptr, unsigned long len, unsigned long idx);

generic<T>
void arr_set(T* ptr, unsigned long len, unsigned long idx, T val);

generic<T>
T arr_get(T* ptr, unsigned long idx);

generic<T>
void arr_fill(T* ptr, unsigned long len, T val);

generic<T>
void arr_copy(T* dst, T* src, unsigned long len);

generic<T>
T arr_min(T* ptr, unsigned long len);

generic<T>
T arr_max(T* ptr, unsigned long len);

generic<T>
void arr_reverse(T* ptr, unsigned long len);
