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
// it — v.free() recursively releases the whole tree, and
// v.array_push()/v.object_set() take ownership of the Value they're given
// (don't free it yourself afterward; do call x.clone() first if you still
// need your own copy).
#pragma once
#include <std/collections/vec.h>
#include <std/collections/string.h>

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

// A text format for Value trees. Each backend is a small struct whose
// methods are the format's writer and parser: Json (json.h), Yaml (yaml.h),
// Csv (csv.h), Xml (xml.h; carries the root element's tag) and Html
// (html.h). Value::encode / Value::decode take any of them, and so does
// generic code bounded by Format:
//
//   struct String text = v.encode(Json::new());
//   struct Value back = Value::decode(Xml::new("doc"), text.as_ptr(), &ok);
trait Format {
    void         write(const &Value v, &String out) const;   // appends v's text to out
    struct Value parse(const &char text, ?&int ok) const;    // *ok: 1, or 0 on malformed input
}

// One key/value pair inside a VAL_OBJECT's obj_val Vec.
struct ObjectEntry {
    ?&heap char  key;  // heap-allocated, NUL-terminated, owned — empty (null)
                        // only on an allocation failure inside Value::object_set()
    &Value       val;  // heap-allocated, owned, always present once inserted
};

struct Value {
    int          kind;      // one of the VAL_* constants above
    long long    int_val;   // VAL_INT
    double       float_val; // VAL_FLOAT
    int          bool_val;  // VAL_BOOL
    ?&heap char  str_val;   // VAL_STRING — heap-allocated, NUL-terminated, owned;
                             // empty (null) for every other 'kind'
    struct Vec<struct Value>       arr_val;   // VAL_ARRAY  — elements by value
    struct Vec<struct ObjectEntry> obj_val;   // VAL_OBJECT — key/value entries by value

    // ── Construction ────────────────────────────────────────────────────────
    static struct Value null();
    static struct Value from_bool(int v);
    static struct Value from_int(long long v);
    static struct Value from_float(double v);
    static struct Value from_str(const &char s);   // copies s
    static struct Value array();                   // empty; grow with v.array_push
    static struct Value object();                  // empty; grow with v.object_set

    int kind_of() const;
    int is_null() const;
    int as_bool() const;
    long long as_int() const;
    double as_float() const;
    const &char as_string() const;

    // Array/Object element access (read-only views — the returned
    // reference aliases storage owned by this Value, don't free it
    // separately). 'array_at' trusts 'idx' is in range (see array_len());
    // 'object_get' is the one nullable lookup — no entry for 'key' is a
    // real, expected outcome, not a caller error.
    unsigned long   array_len() const;
    &Value          array_at(unsigned long idx) const;
    unsigned long   object_len() const;
    // The idx-th entry of an object, in insertion order (for iterating a
    // VAL_OBJECT's keys — e.g. a YAML/JSON map whose order matters): its
    // key ("" if the entry's key allocation failed) and its value. Like
    // array_at, both trust 'idx' < object_len().
    const &char     object_key_at(unsigned long idx) const;
    &Value          object_value_at(unsigned long idx) const;
    // Empty (null case) if the key isn't present.
    ?&Value         object_get(const &char key) const;

    // Array/Object building. Both take ownership of 'v' (see the
    // file-level ownership note above).
    void            array_push(struct Value v);
    void            object_set(const &char key, struct Value v);

    // Deep copy / recursive release of the whole tree.
    struct Value    clone() const;
    void            drop();
    void            free() consume;   // = drop()

    // ── Text formats (any Format) ───────────────────────────────────────────
    // A fresh String holding this tree in format f.
    generic<F: Format> struct String encode(F f) const;
    // Parses text in format f; *ok (if non-null) is 1 on success, else 0.
    generic<F: Format> static struct Value decode(F f, const &char text, ?&int ok);
};

generic<F: Format>
struct String Value::encode(F f) const {
    struct String out = String::new();
    f.write(self, &out);
    return out;
}

generic<F: Format>
struct Value Value::decode(F f, const &char text, ?&int ok) { return f.parse(text, ok); }

// Kept for std/ml until its refactor moves to the methods above; forward to
// Value::array() / Value::object() / Value::from_str(s) /
// arr.array_push(v) / obj.object_set(key, v) / v.clone() / v.free().
struct Value value_array();
struct Value value_object();
struct Value value_string(const &char s);
void value_array_push(&Value arr, struct Value v);
void value_object_set(&Value obj, const &char key, struct Value v);
struct Value value_clone(const &Value v);
void value_free(&Value v);

} // namespace std
