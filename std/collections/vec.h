#pragma once
// SafeC Standard Library — Vec (dynamic array)
// Type-erased dynamic array. Use generic<T> wrappers for typed access.

struct Vec {
    &heap void    data;      // heap-backed element buffer
    unsigned long len;       // current element count
    unsigned long cap;       // allocated element capacity
    unsigned long elem_size; // size of each element in bytes

    // ── Capacity ────────────────────────────────────────────────────────────
    int           reserve(unsigned long new_cap);  // ensure capacity >= new_cap
    void          shrink();                         // shrink cap to len
    unsigned long length() const;                  // element count
    unsigned long capacity() const;                // allocated capacity
    int           is_empty() const;

    // ── Element access (raw/type-erased) ────────────────────────────────────
    &heap void    get_raw(unsigned long idx);       // NULL if OOB
    int           set_raw(unsigned long idx, const void* elem);
    &heap void    front_raw();                      // first element
    &heap void    back_raw();                       // last element

    // ── Mutation ────────────────────────────────────────────────────────────
    int           push(const void* elem);           // append; grows if needed
    int           pop(void* out);                   // remove last; write to out
    int           insert(unsigned long idx, const void* elem); // shift-right insert
    int           remove(unsigned long idx, void* out);        // shift-left remove
    void          clear();                          // set len=0 (keep buffer)
    int           extend(const void* arr, unsigned long count); // append array

    // ── Algorithms ──────────────────────────────────────────────────────────
    void          reverse();
    void          sort(void* cmp);                  // cmp: int(*)(const void*, const void*)
    long long     find(const void* key, void* cmp) const; // -1 if not found
    int           contains(const void* key, void* cmp) const;
    struct Vec    clone() const;                    // deep copy
    void          foreach(void* fn);               // fn: void(*)(void* elem, unsigned long idx)
    struct Vec    filter(void* pred) const;        // pred: int(*)(const void*)
    struct Vec    map_raw(unsigned long out_elem_size, void* fn) const; // fn: void(*)(const void* in, void* out)

    // ── Lifecycle ────────────────────────────────────────────────────────────
    void          free();

    // ── Internal ────────────────────────────────────────────────────────────
    int           grow_();
};

// Constructor free functions.
struct Vec vec_new(unsigned long elem_size);
struct Vec vec_with_cap(unsigned long elem_size, unsigned long cap);

// ── Typed generic wrappers ────────────────────────────────────────────────────
generic<T>
int vec_push_t(&stack Vec v, T val);

generic<T>
T* vec_at(&stack Vec v, unsigned long idx);

generic<T>
int vec_pop_t(&stack Vec v, T* out);

generic<T>
struct Vec vec_from_arr(T* arr, unsigned long len);
