// SafeC Standard Library — Stack implementation
#pragma once
#include <std/collections/stack.h>
#include <std/mem.h>

namespace std {

inline struct Stack stack_new(unsigned long elem_size) {
    struct Stack s;
    s.data = (void*)0;
    s.top = 0UL;
    s.cap = 0UL;
    s.elem_size = elem_size;
    return s;
}

inline struct Stack stack_with_cap(unsigned long elem_size, unsigned long cap) {
    struct Stack s;
    s.elem_size = elem_size;
    s.top = 0UL;
    s.cap = cap;
    unsafe { s.data = alloc(elem_size * cap); }
    if (s.data == (void*)0) s.cap = 0UL;
    return s;
}

inline void stack_free(struct Stack* s) {
    unsafe {
        if (s->data != (void*)0) dealloc(s->data);
        s->data = (void*)0;
        s->top = 0UL;
        s->cap = 0UL;
    }
}

inline unsigned long stack_len(struct Stack* s)    { unsafe { return s->top; } }
inline int           stack_is_empty(struct Stack* s) { unsafe { return s->top == 0UL; } }
inline void          stack_clear(struct Stack* s)  { unsafe { s->top = 0UL; } }

inline void* stack_peek(struct Stack* s) {
    unsafe {
        if (s->top == 0UL) return (void*)0;
        return (void*)((char*)s->data + (s->top - 1UL) * s->elem_size);
    }
}

inline int stack_push(struct Stack* s, const void* elem) {
    unsafe {
        if (s->top == s->cap) {
            unsigned long new_cap = s->cap == 0UL ? 8UL : s->cap * 2UL;
            void* nd = realloc_buf(s->data, new_cap * s->elem_size);
            if (nd == (void*)0) return 0;
            s->data = nd;
            s->cap = new_cap;
        }
        char* dst = (char*)s->data + s->top * s->elem_size;
        safe_memcpy((void*)dst, elem, s->elem_size);
        s->top = s->top + 1UL;
        return 1;
    }
}

inline int stack_pop(struct Stack* s, void* out) {
    unsafe {
        if (s->top == 0UL) return 0;
        s->top = s->top - 1UL;
        if (out != (void*)0) {
            char* src = (char*)s->data + s->top * s->elem_size;
            safe_memcpy(out, (const void*)src, s->elem_size);
        }
        return 1;
    }
}

generic<T>
inline int stack_push_t(struct Stack* s, T val) {
    return stack_push(s, (const void*)&val);
}

generic<T>
inline T* stack_peek_t(struct Stack* s) {
    return (T*)stack_peek(s);
}

generic<T>
inline int stack_pop_t(struct Stack* s, T* out) {
    return stack_pop(s, (void*)out);
}

} // namespace std
