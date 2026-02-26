// SafeC Standard Library — Static-capacity Vector and Map
// All storage is on the stack / in static memory — zero heap allocation.
// Capacity is a compile-time constant baked into the struct via macros.
// Freestanding-safe.
#pragma once

// ── StaticVec ─────────────────────────────────────────────────────────────────
// Usage:
//   STATIC_VEC_DECL(IntVec, int, 64);   // declare a type
//   struct IntVec v;
//   static_vec_push(&v, 42);
// Or use the generic inline helpers below.

#define STATIC_VEC_DECL(TypeName, ElemType, Cap)          \
    struct TypeName {                                      \
        ElemType      data[(Cap)];                        \
        unsigned long len;                                \
        unsigned long cap;                                \
    }

// Generic operations — work with any STATIC_VEC_DECL type via sizeof(elem).
// Pass &vec and the value by pointer for type safety.

// Initialise a StaticVec (zero len; cap = Cap).  Must call before use.
#define STATIC_VEC_INIT(vec, Cap)   do { (vec)->len = 0; (vec)->cap = (Cap); } while(0)

// Push one element.  Returns 1 on success, 0 if full.
#define STATIC_VEC_PUSH(vec, val)                                         \
    ((vec)->len < (vec)->cap                                              \
     ? ((vec)->data[(vec)->len++] = (val), 1)                            \
     : 0)

// Pop last element into *out.  Returns 1 on success, 0 if empty.
#define STATIC_VEC_POP(vec, out)                                          \
    ((vec)->len > 0                                                       \
     ? (*(out) = (vec)->data[--(vec)->len], 1)                           \
     : 0)

// Peek at last element without removing.
#define STATIC_VEC_TOP(vec)  ((vec)->data[(vec)->len - 1])

// Access element by index (unchecked).
#define STATIC_VEC_AT(vec, i)  ((vec)->data[(i)])

// Number of elements.
#define STATIC_VEC_LEN(vec)  ((vec)->len)

// Is empty?
#define STATIC_VEC_EMPTY(vec) ((vec)->len == 0)

// ── StaticMap ─────────────────────────────────────────────────────────────────
// Simple open-addressing hash map with compile-time capacity.
// Keys are unsigned long; values are void* (cast as needed).

#define STATIC_MAP_DECL(TypeName, Cap)                    \
    struct TypeName {                                     \
        unsigned long keys[(Cap)];                        \
        void*         vals[(Cap)];                        \
        int           used[(Cap)];                        \
        unsigned long count;                              \
        unsigned long cap;                                \
    }

#define STATIC_MAP_INIT(m, Cap)  do {                     \
    (m)->count = 0; (m)->cap = (Cap);                     \
    unsigned long _i = 0;                                 \
    while (_i < (Cap)) { (m)->used[_i] = 0; _i++; }      \
} while(0)

// Insert key/val.  Returns 1 on success, 0 if full.
static inline int static_map_insert_(unsigned long* keys, void** vals,
                                      int* used, unsigned long cap,
                                      unsigned long* count,
                                      unsigned long key, void* val) {
    unsigned long h = key % cap;
    unsigned long i = 0;
    while (i < cap) {
        unsigned long slot = (h + i) % cap;
        if (!used[slot] || keys[slot] == key) {
            keys[slot] = key;
            vals[slot] = val;
            if (!used[slot]) { used[slot] = 1; (*count)++; }
            return 1;
        }
        i++;
    }
    return 0;
}

static inline void* static_map_get_(unsigned long* keys, void** vals,
                                     int* used, unsigned long cap,
                                     unsigned long key) {
    unsigned long h = key % cap;
    unsigned long i = 0;
    while (i < cap) {
        unsigned long slot = (h + i) % cap;
        if (!used[slot]) return (void*)0;
        if (keys[slot] == key) return vals[slot];
        i++;
    }
    return (void*)0;
}

#define STATIC_MAP_INSERT(m, key, val) \
    static_map_insert_((m)->keys, (m)->vals, (m)->used, (m)->cap, &(m)->count, (key), (void*)(val))

#define STATIC_MAP_GET(m, key) \
    static_map_get_((m)->keys, (m)->vals, (m)->used, (m)->cap, (key))

#define STATIC_MAP_LEN(m) ((m)->count)
