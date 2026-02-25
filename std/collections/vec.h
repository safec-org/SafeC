#pragma once
// SafeC Standard Library — Vec (dynamic array)
// Type-erased dynamic array. Use generic<T> wrappers for typed access.

struct Vec {
    void*         data;      // heap-allocated element buffer
    unsigned long len;       // current element count
    unsigned long cap;       // allocated element capacity
    unsigned long elem_size; // size of each element in bytes
};

// ── Lifecycle ─────────────────────────────────────────────────────────────────
struct Vec vec_new(unsigned long elem_size);
struct Vec vec_with_cap(unsigned long elem_size, unsigned long cap);
void       vec_free(struct Vec* v);

// ── Capacity ──────────────────────────────────────────────────────────────────
int           vec_reserve(struct Vec* v, unsigned long new_cap); // ensure capacity >= new_cap
void          vec_shrink(struct Vec* v);                          // shrink cap to len
unsigned long vec_len(struct Vec* v);
unsigned long vec_cap(struct Vec* v);
int           vec_is_empty(struct Vec* v);

// ── Element access (raw/type-erased) ─────────────────────────────────────────
void* vec_get_raw(struct Vec* v, unsigned long idx);           // NULL if OOB
int   vec_set_raw(struct Vec* v, unsigned long idx, const void* elem);
void* vec_front_raw(struct Vec* v);                            // first element
void* vec_back_raw(struct Vec* v);                             // last element

// ── Mutation ──────────────────────────────────────────────────────────────────
int  vec_push(struct Vec* v, const void* elem);                // append; grows if needed
int  vec_pop(struct Vec* v, void* out);                        // remove last; write to out
int  vec_insert(struct Vec* v, unsigned long idx, const void* elem); // shift-right insert
int  vec_remove(struct Vec* v, unsigned long idx, void* out);  // shift-left remove
void vec_clear(struct Vec* v);                                  // set len=0 (keep buffer)
int  vec_extend(struct Vec* v, const void* arr, unsigned long count); // append array

// ── Algorithms ────────────────────────────────────────────────────────────────
void      vec_reverse(struct Vec* v);
void      vec_sort(struct Vec* v, void* cmp);        // cmp: int(*)(const void*, const void*)
long long vec_find(struct Vec* v, const void* key, void* cmp); // -1 if not found
int       vec_contains(struct Vec* v, const void* key, void* cmp);
struct Vec vec_clone(struct Vec* v);                  // deep copy
void      vec_foreach(struct Vec* v, void* fn);       // fn: void(*)(void* elem, unsigned long idx)
struct Vec vec_filter(struct Vec* v, void* pred);     // pred: int(*)(const void*)
struct Vec vec_map_raw(struct Vec* v, unsigned long out_elem_size, void* fn); // fn: void(*)(const void* in, void* out)

// ── Typed generic wrappers ────────────────────────────────────────────────────
generic<T>
int vec_push_t(struct Vec* v, T val);

generic<T>
T* vec_at(struct Vec* v, unsigned long idx);

generic<T>
int vec_pop_t(struct Vec* v, T* out);

generic<T>
struct Vec vec_from_arr(T* arr, unsigned long len);
