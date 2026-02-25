#pragma once
// SafeC Standard Library — Stack (LIFO)
// Backed by a growing array. O(1) amortized push/pop.

struct Stack {
    void*         data;
    unsigned long top;       // number of elements
    unsigned long cap;
    unsigned long elem_size;
};

// ── Lifecycle ─────────────────────────────────────────────────────────────────
struct Stack stack_new(unsigned long elem_size);
struct Stack stack_with_cap(unsigned long elem_size, unsigned long cap);
void         stack_free(struct Stack* s);

// ── Core operations ───────────────────────────────────────────────────────────
int           stack_push(struct Stack* s, const void* elem);  // returns 1 on success
int           stack_pop(struct Stack* s, void* out);          // returns 0 if empty
void*         stack_peek(struct Stack* s);                    // NULL if empty
unsigned long stack_len(struct Stack* s);
int           stack_is_empty(struct Stack* s);
void          stack_clear(struct Stack* s);

// ── Typed generic wrappers ────────────────────────────────────────────────────
generic<T>
int stack_push_t(struct Stack* s, T val);

generic<T>
T* stack_peek_t(struct Stack* s);

generic<T>
int stack_pop_t(struct Stack* s, T* out);
