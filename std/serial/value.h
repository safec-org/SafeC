// SafeC Standard Library — Serializer core: a generic, format-agnostic
// value tree (Null/Bool/Int/Float/String/Array/Object).
//
// This is the core the format backends (json.sc, xml.sc, html.sc) all
// build on: each backend only needs 'struct Value -> text' (and, where it
// makes sense, 'text -> struct Value') — none of them touch a specific
// struct's fields directly. A struct becomes serializable by writing a
// 'to_value()' method that builds one of these (see json.h's top comment
// for a worked example) — SafeC has no field-name/field-type reflection
// today (only fieldcount(T), a count with no names attached), so this is
// deliberately a manual, trait-shaped pattern rather than a #[derive]-style
// automatic one; std/serial/derive.md (n/a here) would be the place to
// revisit that if reflection ever grows field names.
//
// Ownership: every Value that holds heap data (String, Array, Object) OWNS
// it — value_free() recursively releases the whole tree, and
// value_array_push()/value_object_set() take ownership of the Value they're
// given (don't free it yourself afterward; do call value_clone() first if
// you still need your own copy).
#pragma once
#include <std/collections/vec.h>

namespace std {

// Value::kind — deliberately plain ints (see collections/map.sc's MapEntry
// state field for the same convention elsewhere in std) rather than
// SafeC's tagged-union feature: 'arr_val'/'obj_val' are same-size Vec
// headers regardless of what they hold, so there's no payload-size
// mismatch a tagged union would actually save memory on, and this keeps
// Value a plain flat struct — the simpler, more established shape for
// something meant to be pushed into Vec elements by raw memcpy.
#define VAL_NULL   0
#define VAL_BOOL   1
#define VAL_INT    2
#define VAL_FLOAT  3
#define VAL_STRING 4
#define VAL_ARRAY  5
#define VAL_OBJECT 6

// One key/value pair inside a VAL_OBJECT's obj_val Vec.
struct ObjectEntry {
    char*   key;  // heap-allocated, NUL-terminated, owned
    &Value  val;  // heap-allocated, owned, always present once inserted
};

struct Value {
    int          kind;      // one of the VAL_* constants above
    long long    int_val;   // VAL_INT
    double       float_val; // VAL_FLOAT
    int          bool_val;  // VAL_BOOL
    char*        str_val;   // VAL_STRING — heap-allocated, NUL-terminated, owned
    struct Vec   arr_val;   // VAL_ARRAY  — elements are 'struct Value' by value
    struct Vec   obj_val;   // VAL_OBJECT — elements are 'struct ObjectEntry' by value

    int kind_of() const;
    int is_null() const;
    int as_bool() const;
    long long as_int() const;
    double as_float() const;
    const char* as_string() const;

    // Array/Object element access (read-only views — the returned
    // reference aliases storage owned by this Value, don't free it
    // separately). 'array_at' trusts 'idx' is in range (see array_len());
    // 'object_get' is the one nullable lookup — no entry for 'key' is a
    // real, expected outcome, not a caller error.
    unsigned long   array_len() const;
    &Value          array_at(unsigned long idx) const;
    unsigned long   object_len() const;
    // Empty (null case) if the key isn't present.
    ?&Value         object_get(const char* key) const;
};

struct Value value_null();
struct Value value_bool(int v);
struct Value value_int(long long v);
struct Value value_float(double v);
struct Value value_string(const char* s);   // copies s
struct Value value_array();                 // empty; grow with value_array_push
struct Value value_object();                // empty; grow with value_object_set

// Both take ownership of 'v' (see the file-level ownership note above).
void value_array_push(&Value arr, struct Value v);
void value_object_set(&Value obj, const char* key, struct Value v);

struct Value value_clone(const &Value v);
void value_free(&Value v);

} // namespace std
