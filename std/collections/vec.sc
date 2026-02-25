// SafeC Standard Library — Vec implementation
#include "vec.h"
#include "../mem.h"

// ── Lifecycle ─────────────────────────────────────────────────────────────────
struct Vec vec_new(unsigned long elem_size) {
    struct Vec v;
    v.data = (void*)0;
    v.len = 0UL;
    v.cap = 0UL;
    v.elem_size = elem_size;
    return v;
}

struct Vec vec_with_cap(unsigned long elem_size, unsigned long cap) {
    struct Vec v;
    v.elem_size = elem_size;
    v.len = 0UL;
    v.cap = cap;
    unsafe {
        v.data = alloc(elem_size * cap);
        if (v.data == (void*)0) v.cap = 0UL;
    }
    return v;
}

void vec_free(struct Vec* v) {
    unsafe {
        if (v->data != (void*)0) dealloc(v->data);
    }
    v->data = (void*)0;
    v->len = 0UL;
    v->cap = 0UL;
}

// ── Capacity ──────────────────────────────────────────────────────────────────
unsigned long vec_len(struct Vec* v)    { return v->len; }
unsigned long vec_cap(struct Vec* v)    { return v->cap; }
int           vec_is_empty(struct Vec* v) { return v->len == 0UL; }

int vec_reserve(struct Vec* v, unsigned long new_cap) {
    if (new_cap <= v->cap) return 1;
    unsafe {
        void* nd = realloc_buf(v->data, new_cap * v->elem_size);
        if (nd == (void*)0) return 0;
        v->data = nd;
        v->cap = new_cap;
    }
    return 1;
}

void vec_shrink(struct Vec* v) {
    if (v->len == 0UL) {
        unsafe { if (v->data != (void*)0) dealloc(v->data); }
        v->data = (void*)0;
        v->cap = 0UL;
        return;
    }
    unsafe {
        void* nd = realloc_buf(v->data, v->len * v->elem_size);
        if (nd != (void*)0) { v->data = nd; v->cap = v->len; }
    }
}

// ── Internal grow helper ──────────────────────────────────────────────────────
int vec_grow_(struct Vec* v) {
    unsigned long new_cap = v->cap == 0UL ? 4UL : v->cap * 2UL;
    return vec_reserve(v, new_cap);
}

// ── Element access ────────────────────────────────────────────────────────────
void* vec_get_raw(struct Vec* v, unsigned long idx) {
    if (idx >= v->len) return (void*)0;
    unsafe { return (void*)((char*)v->data + idx * v->elem_size); }
}

int vec_set_raw(struct Vec* v, unsigned long idx, const void* elem) {
    if (idx >= v->len) return 0;
    unsafe {
        char* dst = (char*)v->data + idx * v->elem_size;
        safe_memcpy((void*)dst, elem, v->elem_size);
    }
    return 1;
}

void* vec_front_raw(struct Vec* v) { return v->len > 0UL ? v->data : (void*)0; }
void* vec_back_raw(struct Vec* v) {
    if (v->len == 0UL) return (void*)0;
    unsafe { return (void*)((char*)v->data + (v->len - 1UL) * v->elem_size); }
}

// ── Mutation ──────────────────────────────────────────────────────────────────
int vec_push(struct Vec* v, const void* elem) {
    if (v->len == v->cap) {
        if (!vec_grow_(v)) return 0;
    }
    unsafe {
        char* dst = (char*)v->data + v->len * v->elem_size;
        safe_memcpy((void*)dst, elem, v->elem_size);
    }
    v->len = v->len + 1UL;
    return 1;
}

int vec_pop(struct Vec* v, void* out) {
    if (v->len == 0UL) return 0;
    v->len = v->len - 1UL;
    if (out != (void*)0) {
        unsafe {
            char* src = (char*)v->data + v->len * v->elem_size;
            safe_memcpy(out, (const void*)src, v->elem_size);
        }
    }
    return 1;
}

int vec_insert(struct Vec* v, unsigned long idx, const void* elem) {
    if (idx > v->len) return 0;
    if (v->len == v->cap) {
        if (!vec_grow_(v)) return 0;
    }
    unsafe {
        // Shift elements [idx, len) right by one position
        char* base = (char*)v->data;
        unsigned long bytes = (v->len - idx) * v->elem_size;
        if (bytes > 0UL)
            safe_memmove((void*)(base + (idx + 1UL) * v->elem_size),
                         (const void*)(base + idx * v->elem_size), bytes);
        safe_memcpy((void*)(base + idx * v->elem_size), elem, v->elem_size);
    }
    v->len = v->len + 1UL;
    return 1;
}

int vec_remove(struct Vec* v, unsigned long idx, void* out) {
    if (idx >= v->len) return 0;
    if (out != (void*)0) {
        unsafe {
            char* src = (char*)v->data + idx * v->elem_size;
            safe_memcpy(out, (const void*)src, v->elem_size);
        }
    }
    unsafe {
        char* base = (char*)v->data;
        unsigned long bytes = (v->len - idx - 1UL) * v->elem_size;
        if (bytes > 0UL)
            safe_memmove((void*)(base + idx * v->elem_size),
                         (const void*)(base + (idx + 1UL) * v->elem_size), bytes);
    }
    v->len = v->len - 1UL;
    return 1;
}

void vec_clear(struct Vec* v) { v->len = 0UL; }

int vec_extend(struct Vec* v, const void* arr, unsigned long count) {
    if (!vec_reserve(v, v->len + count)) return 0;
    unsafe {
        char* dst = (char*)v->data + v->len * v->elem_size;
        safe_memcpy((void*)dst, arr, count * v->elem_size);
    }
    v->len = v->len + count;
    return 1;
}

// ── Algorithms ────────────────────────────────────────────────────────────────
void vec_reverse(struct Vec* v) {
    if (v->len <= 1UL) return;
    unsafe {
        char* tmp = (char*)alloc(v->elem_size);
        if (tmp == (char*)0) return;
        unsigned long lo = 0UL;
        unsigned long hi = v->len - 1UL;
        while (lo < hi) {
            char* a = (char*)v->data + lo * v->elem_size;
            char* b = (char*)v->data + hi * v->elem_size;
            safe_memcpy((void*)tmp, (const void*)a, v->elem_size);
            safe_memcpy((void*)a,   (const void*)b, v->elem_size);
            safe_memcpy((void*)b,   (const void*)tmp, v->elem_size);
            lo = lo + 1UL;
            hi = hi - 1UL;
        }
        dealloc((void*)tmp);
    }
}

extern void qsort(void* base, unsigned long n, unsigned long sz, void* cmp);

void vec_sort(struct Vec* v, void* cmp) {
    if (v->len <= 1UL) return;
    unsafe { qsort(v->data, v->len, v->elem_size, cmp); }
}

long long vec_find(struct Vec* v, const void* key, void* cmp) {
    // cmp: int(*)(const void*, const void*)
    // We use it via a type-punned function pointer call in unsafe
    unsafe {
        // Cast cmp to a function pointer and call it
        int i = 0;
        while ((unsigned long)i < v->len) {
            char* elem = (char*)v->data + (unsigned long)i * v->elem_size;
            // call cmp(key, elem) via indirect call
            int (*cmpfn)(const void*, const void*) = (int (*)(const void*, const void*))cmp;
            if (cmpfn(key, (const void*)elem) == 0) return (long long)i;
            i = i + 1;
        }
    }
    return -1LL;
}

int vec_contains(struct Vec* v, const void* key, void* cmp) {
    return vec_find(v, key, cmp) >= 0LL;
}

struct Vec vec_clone(struct Vec* v) {
    struct Vec c = vec_with_cap(v->elem_size, v->len);
    if (v->len > 0UL && c.data != (void*)0) {
        unsafe { safe_memcpy(c.data, (const void*)v->data, v->len * v->elem_size); }
        c.len = v->len;
    }
    return c;
}

void vec_foreach(struct Vec* v, void* fn) {
    unsafe {
        void (*f)(void*, unsigned long) = (void (*)(void*, unsigned long))fn;
        unsigned long i = 0UL;
        while (i < v->len) {
            f((void*)((char*)v->data + i * v->elem_size), i);
            i = i + 1UL;
        }
    }
}

struct Vec vec_filter(struct Vec* v, void* pred) {
    struct Vec out = vec_new(v->elem_size);
    unsafe {
        int (*p)(const void*) = (int (*)(const void*))pred;
        unsigned long i = 0UL;
        while (i < v->len) {
            void* elem = (void*)((char*)v->data + i * v->elem_size);
            if (p((const void*)elem)) vec_push(&out, (const void*)elem);
            i = i + 1UL;
        }
    }
    return out;
}

struct Vec vec_map_raw(struct Vec* v, unsigned long out_elem_size, void* fn) {
    struct Vec out = vec_with_cap(out_elem_size, v->len);
    unsafe {
        void (*f)(const void*, void*) = (void (*)(const void*, void*))fn;
        void* tmp = alloc(out_elem_size);
        unsigned long i = 0UL;
        while (i < v->len) {
            void* src = (void*)((char*)v->data + i * v->elem_size);
            f((const void*)src, tmp);
            vec_push(&out, (const void*)tmp);
            i = i + 1UL;
        }
        dealloc(tmp);
    }
    return out;
}

// ── Typed generic wrappers ────────────────────────────────────────────────────
generic<T>
int vec_push_t(struct Vec* v, T val) {
    return vec_push(v, (const void*)&val);
}

generic<T>
T* vec_at(struct Vec* v, unsigned long idx) {
    return (T*)vec_get_raw(v, idx);
}

generic<T>
int vec_pop_t(struct Vec* v, T* out) {
    return vec_pop(v, (void*)out);
}

generic<T>
struct Vec vec_from_arr(T* arr, unsigned long len) {
    struct Vec v = vec_new((unsigned long)sizeof(T));
    vec_extend(&v, (const void*)arr, len);
    return v;
}
