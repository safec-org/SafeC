// SafeC Standard Library — Serializer core implementation (see value.h)
#pragma once
#include <std/serial/value.h>
#include <std/collections/vec.h>
#include <std/mem.h>
#include <std/str.h>

namespace std {

static char* value_strdup_(const char* s) {
    unsigned long n; unsafe { n = str_len(s); }
    char* buf;
    unsafe { buf = (char*)alloc_raw_(n + 1UL); }
    if (buf == (char*)0) { return (char*)0; }
    unsafe { str_copy(buf, s, n + 1UL); }
    return buf;
}

inline struct Value Value::null() {
    struct Value v;
    v.kind = VAL_NULL;
    v.int_val = 0LL;
    v.float_val = 0.0;
    v.bool_val = 0;
    unsafe { v.str_val = (char*)0; }
    v.arr_val = Vec::new();
    v.obj_val = Vec::new();
    return v;
}

inline struct Value Value::from_bool(int b) {
    struct Value v = Value::null();
    v.kind = VAL_BOOL;
    v.bool_val = b;
    return v;
}

inline struct Value Value::from_int(long long n) {
    struct Value v = Value::null();
    v.kind = VAL_INT;
    v.int_val = n;
    return v;
}

inline struct Value Value::from_float(double f) {
    struct Value v = Value::null();
    v.kind = VAL_FLOAT;
    v.float_val = f;
    return v;
}

inline struct Value Value::from_str(const &char s) {
    struct Value v = Value::null();
    v.kind = VAL_STRING;
    v.str_val = value_strdup_(s);
    return v;
}

inline struct Value Value::array() {
    struct Value v = Value::null();
    v.kind = VAL_ARRAY;
    return v;
}

inline struct Value Value::object() {
    struct Value v = Value::null();
    v.kind = VAL_OBJECT;
    return v;
}

inline void Value::array_push(struct Value v) {
    self.arr_val.push(v);
}

inline void Value::object_set(const &char key, struct Value v) {
    struct ObjectEntry e;
    e.key = value_strdup_(key);
    // Store the value on the heap so ObjectEntry::val is a stable pointer
    // (obj_val is a Vec of ObjectEntry *by value* — the Value itself has
    // to live somewhere that survives the Vec growing/reallocating).
    unsafe {
        struct Value* heapVal = (struct Value*)alloc_raw_(sizeof(struct Value));
        *heapVal = v;
        e.val = heapVal;
        self.obj_val.push(e);
    }
}

struct Value Value::clone() const {
    struct Value out = Value::null();
    int kind = self.kind;
    out.kind = kind;
    out.int_val   = self.int_val;
    out.float_val = self.float_val;
    out.bool_val  = self.bool_val;
    if (kind == VAL_STRING) {
        unsafe {
            if (self.str_val != (char*)0) { out.str_val = value_strdup_(self.str_val); }
        }
    } else if (kind == VAL_ARRAY) {
        unsigned long i = 0UL;
        unsigned long n = self.arr_val.length();
        while (i < n) {
            &Value elem;
            unsafe { elem = self.arr_val.at(i); }
            out.array_push(elem.clone());
            i = i + 1UL;
        }
    } else if (kind == VAL_OBJECT) {
        unsigned long i = 0UL;
        unsigned long n = self.obj_val.length();
        while (i < n) {
            struct ObjectEntry* e;
            unsafe { e = self.obj_val.at(i); }
            unsafe {
                match (e->key) {
                    case some(k): { out.object_set(k, e->val.clone()); }
                    case null:    { }
                }
            }
            i = i + 1UL;
        }
    }
    return out;
}

void Value::free() {
    int kind = self.kind;
    if (kind == VAL_STRING) {
        unsafe { if (self.str_val != (char*)0) { dealloc_raw_((void*)self.str_val); } }
    } else if (kind == VAL_ARRAY) {
        unsigned long i = 0UL;
        unsigned long n = self.arr_val.length();
        while (i < n) {
            &Value elem;
            unsafe { elem = self.arr_val.at(i); }
            elem.free();
            i = i + 1UL;
        }
        self.arr_val.free();
    } else if (kind == VAL_OBJECT) {
        unsigned long i = 0UL;
        unsigned long n = self.obj_val.length();
        while (i < n) {
            struct ObjectEntry* e;
            unsafe {
                e = self.obj_val.at(i);
                e->val.free();
                dealloc_raw_((void*)e->val);
                dealloc_raw_((void*)e->key);
            }
            i = i + 1UL;
        }
        self.obj_val.free();
    }
}

// Forwarding wrappers kept for std/ml (see value.h).
// kept for std/ml until its refactor moves to the static method
inline struct Value value_array() { return Value::array(); }
// kept for std/ml until its refactor moves to the static method
inline struct Value value_object() { return Value::object(); }
// kept for std/ml until its refactor moves to the static method
inline struct Value value_string(const &char s) { return Value::from_str(s); }
inline void value_array_push(&Value arr, struct Value v) { arr.array_push(v); }
inline void value_object_set(&Value obj, const &char key, struct Value v) { obj.object_set(key, v); }
inline struct Value value_clone(const &Value v) { return v.clone(); }
inline void value_free(&Value v) { v.free(); }

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
inline const &char Value::as_string() const {
    unsafe {
        if (self.str_val == (char*)0) { return ""; }
        return (const char*)self.str_val;
    }
}

inline unsigned long Value::array_len() const { return self.arr_val.length(); }
inline &Value Value::array_at(unsigned long idx) const {
    unsafe { return self.arr_val.at(idx); }
}
inline unsigned long Value::object_len() const { return self.obj_val.length(); }

?&Value Value::object_get(const &char key) const {
    unsigned long i = 0UL;
    unsigned long n = self.obj_val.length();
    while (i < n) {
        struct ObjectEntry* e;
        unsafe { e = self.obj_val.at(i); }
        int matched;
        matched = 0;
        unsafe {
            match (e->key) {
                case some(k): { matched = str_eq(k, key); }
                case null:    { }
            }
        }
        if (matched != 0) {
            &Value result;
            unsafe { result = e->val; }
            return result;
        }
        i = i + 1UL;
    }
    unsafe { return (struct Value*)0; }
}

} // namespace std
