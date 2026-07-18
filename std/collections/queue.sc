// SafeC Standard Library — Queue (circular buffer) implementation
#pragma once
#include <std/collections/queue.h>
#include <std/mem.h>

namespace std {

inline struct Queue queue_new(unsigned long elem_size) {
    struct Queue q;
    q.data = (void*)0;
    q.head = 0UL; q.tail = 0UL; q.len = 0UL; q.cap = 0UL;
    q.elem_size = elem_size;
    return q;
}

inline struct Queue queue_with_cap(unsigned long elem_size, unsigned long cap) {
    struct Queue q;
    q.elem_size = elem_size;
    q.head = 0UL; q.tail = 0UL; q.len = 0UL; q.cap = cap;
    unsafe { q.data = alloc(checked_mul_size(elem_size, cap)); }
    if (q.data == (void*)0) q.cap = 0UL;
    return q;
}

inline void queue_free(struct Queue* q) {
    unsafe {
        if (q->data != (void*)0) dealloc(q->data);
        q->data = (void*)0;
        q->head = 0UL; q->tail = 0UL; q->len = 0UL; q->cap = 0UL;
    }
}

inline unsigned long queue_len(struct Queue* q)     { unsafe { return q->len; } }
inline int           queue_is_empty(struct Queue* q){ unsafe { return q->len == 0UL; } }

inline void queue_clear(struct Queue* q) {
    unsafe { q->head = 0UL; q->tail = 0UL; q->len = 0UL; }
}

inline void* queue_front(struct Queue* q) {
    unsafe {
        if (q->len == 0UL) return (void*)0;
        return (void*)((char*)q->data + q->head * q->elem_size);
    }
}

inline void* queue_back(struct Queue* q) {
    unsafe {
        if (q->len == 0UL) return (void*)0;
        unsigned long back_idx = (q->tail == 0UL) ? (q->cap - 1UL) : (q->tail - 1UL);
        return (void*)((char*)q->data + back_idx * q->elem_size);
    }
}

// Grow: linearize circular buffer into a new allocation of double capacity
int queue_grow_(struct Queue* q) {
    unsafe {
        unsigned long new_cap = q->cap == 0UL ? 8UL : q->cap * 2UL;
        void* nd = alloc(checked_mul_size(new_cap, q->elem_size));
        if (nd == (void*)0) return 0;
        // Copy elements in logical order
        if (q->len > 0UL) {
            if (q->head < q->tail) {
                // Contiguous
                safe_memcpy(nd, (const void*)((char*)q->data + q->head * q->elem_size),
                            q->len * q->elem_size);
            } else {
                // Wrapped
                unsigned long first_part = q->cap - q->head;
                safe_memcpy(nd,
                    (const void*)((char*)q->data + q->head * q->elem_size),
                    first_part * q->elem_size);
                safe_memcpy((void*)((char*)nd + first_part * q->elem_size),
                    (const void*)q->data,
                    q->tail * q->elem_size);
            }
        }
        if (q->data != (void*)0) dealloc(q->data);
        q->data = nd;
        q->head = 0UL;
        q->tail = q->len;
        q->cap  = new_cap;
    }
    return 1;
}

inline int queue_enqueue(struct Queue* q, const void* elem) {
    unsafe {
        if (q->len == q->cap) {
            if (!queue_grow_(q)) return 0;
        }
        char* dst = (char*)q->data + q->tail * q->elem_size;
        safe_memcpy((void*)dst, elem, q->elem_size);
        q->tail = (q->tail + 1UL) % q->cap;
        q->len = q->len + 1UL;
        return 1;
    }
}

inline int queue_dequeue(struct Queue* q, void* out) {
    unsafe {
        if (q->len == 0UL) return 0;
        if (out != (void*)0) {
            char* src = (char*)q->data + q->head * q->elem_size;
            safe_memcpy(out, (const void*)src, q->elem_size);
        }
        q->head = (q->head + 1UL) % q->cap;
        q->len = q->len - 1UL;
        return 1;
    }
}

generic<T>
inline int queue_enqueue_t(struct Queue* q, T val) {
    unsafe { return queue_enqueue(q, (const void*)&val); }
}

generic<T>
inline T* queue_front_t(struct Queue* q) {
    return (T*)queue_front(q);
}

generic<T>
inline int queue_dequeue_t(struct Queue* q, T* out) {
    unsafe { return queue_dequeue(q, (void*)out); }
}

} // namespace std
