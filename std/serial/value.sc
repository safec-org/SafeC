// SafeC Standard Library — Serializer core implementation (see value.h)
#pragma once
#include <std/serial/value.h>
#include <std/collections/vec.sc>
#include <std/mem.sc>
#include <std/str.sc>

namespace std {

static char* value_strdup_(const char* s) {
    unsigned long n = str_len(s);
    char* buf;
    unsafe { buf = (char*)alloc(n + 1UL); }
    if (buf == (char*)0) { return (char*)0; }
    str_copy(buf, s, n + 1UL);
    return buf;
}

inline struct Value value_null() {
    struct Value v;
    v.kind = VAL_NULL;
    v.int_val = 0LL;
    v.float_val = 0.0;
    v.bool_val = 0;
    unsafe { v.str_val = (char*)0; }
    v.arr_val = vec_new(sizeof(struct Value));
    v.obj_val = vec_new(sizeof(struct ObjectEntry));
    return v;
}

inline struct Value value_bool(int b) {
    struct Value v = value_null();
    v.kind = VAL_BOOL;
    v.bool_val = b;
    return v;
}

inline struct Value value_int(long long n) {
    struct Value v = value_null();
    v.kind = VAL_INT;
    v.int_val = n;
    return v;
}

inline struct Value value_float(double f) {
    struct Value v = value_null();
    v.kind = VAL_FLOAT;
    v.float_val = f;
    return v;
}

inline struct Value value_string(const char* s) {
    struct Value v = value_null();
    v.kind = VAL_STRING;
    v.str_val = value_strdup_(s);
    return v;
}

inline struct Value value_array() {
    struct Value v = value_null();
    v.kind = VAL_ARRAY;
    return v;
}

inline struct Value value_object() {
    struct Value v = value_null();
    v.kind = VAL_OBJECT;
    return v;
}

inline void value_array_push(struct Value* arr, struct Value v) {
    unsafe { arr->arr_val.push((const void*)&v); }
}

inline void value_object_set(struct Value* obj, const char* key, struct Value v) {
    struct ObjectEntry e;
    e.key = value_strdup_(key);
    // Store the value on the heap so ObjectEntry::val is a stable pointer
    // (obj_val is a Vec of ObjectEntry *by value* — the Value itself has
    // to live somewhere that survives the Vec growing/reallocating).
    unsafe {
        struct Value* heapVal = (struct Value*)alloc(sizeof(struct Value));
        *heapVal = v;
        e.val = heapVal;
        obj->obj_val.push((const void*)&e);
    }
}

struct Value value_clone(const struct Value* v) {
    struct Value out = value_null();
    int kind;
    unsafe { kind = v->kind; }
    out.kind = kind;
    unsafe {
        out.int_val   = v->int_val;
        out.float_val = v->float_val;
        out.bool_val  = v->bool_val;
    }
    if (kind == VAL_STRING) {
        unsafe {
            if (v->str_val != (char*)0) { out.str_val = value_strdup_(v->str_val); }
        }
    } else if (kind == VAL_ARRAY) {
        unsigned long i = 0UL;
        unsigned long n;
        unsafe { n = v->arr_val.length(); }
        while (i < n) {
            struct Value* elem;
            unsafe { elem = (struct Value*)v->arr_val.get_raw(i); }
            value_array_push(&out, value_clone(elem));
            i = i + 1UL;
        }
    } else if (kind == VAL_OBJECT) {
        unsigned long i = 0UL;
        unsigned long n;
        unsafe { n = v->obj_val.length(); }
        while (i < n) {
            struct ObjectEntry* e;
            unsafe { e = (struct ObjectEntry*)v->obj_val.get_raw(i); }
            unsafe { value_object_set(&out, e->key, value_clone(e->val)); }
            i = i + 1UL;
        }
    }
    return out;
}

void value_free(struct Value* v) {
    int kind;
    unsafe { kind = v->kind; }
    if (kind == VAL_STRING) {
        unsafe { if (v->str_val != (char*)0) { dealloc((void*)v->str_val); } }
    } else if (kind == VAL_ARRAY) {
        unsigned long i = 0UL;
        unsigned long n;
        unsafe { n = v->arr_val.length(); }
        while (i < n) {
            struct Value* elem;
            unsafe { elem = (struct Value*)v->arr_val.get_raw(i); }
            value_free(elem);
            i = i + 1UL;
        }
        unsafe { v->arr_val.free(); }
    } else if (kind == VAL_OBJECT) {
        unsigned long i = 0UL;
        unsigned long n;
        unsafe { n = v->obj_val.length(); }
        while (i < n) {
            struct ObjectEntry* e;
            unsafe {
                e = (struct ObjectEntry*)v->obj_val.get_raw(i);
                value_free(e->val);
                dealloc((void*)e->val);
                dealloc((void*)e->key);
            }
            i = i + 1UL;
        }
        unsafe { v->obj_val.free(); }
    }
}

inline int Value::kind_of() const { return self.kind; }
inline int Value::is_null() const { return self.kind == VAL_NULL; }
inline int Value::as_bool() const { return self.bool_val; }
inline long long Value::as_int() const {
    if (self.kind == VAL_FLOAT) { return (long long)self.float_val; }
    return self.int_val;
}
inline double Value::as_float() const {
    if (self.kind == VAL_INT) { return (double)self.int_val; }
    return self.float_val;
}
inline const char* Value::as_string() const {
    unsafe {
        if (self.str_val == (char*)0) { return ""; }
        return (const char*)self.str_val;
    }
}

inline unsigned long Value::array_len() const { return self.arr_val.length(); }
inline struct Value* Value::array_at(unsigned long idx) const {
    unsafe { return (struct Value*)self.arr_val.get_raw(idx); }
}
inline unsigned long Value::object_len() const { return self.obj_val.length(); }

struct Value* Value::object_get(const char* key) const {
    unsigned long i = 0UL;
    unsigned long n = self.obj_val.length();
    while (i < n) {
        struct ObjectEntry* e;
        unsafe { e = (struct ObjectEntry*)self.obj_val.get_raw(i); }
        int matched;
        unsafe { matched = str_eq(e->key, key); }
        if (matched != 0) {
            struct Value* result;
            unsafe { result = e->val; }
            return result;
        }
        i = i + 1UL;
    }
    unsafe { return (struct Value*)0; }
}

} // namespace std
