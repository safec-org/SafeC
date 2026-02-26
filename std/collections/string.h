#pragma once
// SafeC Standard Library — String (mutable, heap-allocated)
// A growable UTF-8 byte string with rich manipulation methods.

struct String {
    char*         data; // NUL-terminated heap buffer
    unsigned long len;  // byte length (not including NUL)
    unsigned long cap;  // allocated buffer size (including NUL slot)

    // ── Access ──────────────────────────────────────────────────────────────
    unsigned long  length() const;
    int            is_empty() const;
    const char*    as_ptr() const;          // NUL-terminated C string
    int            char_at(unsigned long idx) const; // -1 if OOB
    void           set_char(unsigned long idx, char c);

    // ── Append ──────────────────────────────────────────────────────────────
    int            push_char(char c);
    int            push(const char* cstr);
    int            push_str(&stack String other);
    int            push_int(long long v);
    int            push_uint(unsigned long long v);
    int            push_float(double v, int decimals);
    int            push_bool(int v);

    // ── Modification ────────────────────────────────────────────────────────
    void           clear();
    void           truncate(unsigned long new_len);
    int            insert(unsigned long idx, const char* cstr);
    int            remove_range(unsigned long start, unsigned long end);
    void           replace_char(char from, char to);
    int            replace(const char* from, const char* to);      // first occurrence
    int            replace_all(const char* from, const char* to);

    // ── Search ──────────────────────────────────────────────────────────────
    long long      index_of(const char* needle) const;
    long long      last_index_of(const char* needle) const;
    int            contains(const char* needle) const;
    int            starts_with(const char* prefix) const;
    int            ends_with(const char* suffix) const;
    int            count(const char* needle) const;

    // ── Transformation (return new String) ──────────────────────────────────
    struct String  substr(unsigned long start, unsigned long end) const;
    struct String  to_upper() const;
    struct String  to_lower() const;
    struct String  trim() const;
    struct String  trim_left() const;
    struct String  trim_right() const;

    // ── Comparison ──────────────────────────────────────────────────────────
    int            eq(&stack String other) const;
    int            eq_cstr(const char* other) const;
    int            cmp(&stack String other) const;     // <0, 0, >0
    int            lt(&stack String other) const;
    int            gt(&stack String other) const;

    // ── Conversion ──────────────────────────────────────────────────────────
    long long      parse_int(int* ok) const;
    double         parse_float(int* ok) const;

    // ── Lifecycle ────────────────────────────────────────────────────────────
    struct String  clone() const;
    void           free();

    // ── Internal ────────────────────────────────────────────────────────────
    int            reserve_(unsigned long need);
};

// ── Constructor free functions ────────────────────────────────────────────────
struct String string_new();
struct String string_from(const char* s);
struct String string_from_n(const char* s, unsigned long n);
struct String string_with_cap(unsigned long cap);
struct String string_repeat(const char* s, unsigned long n);
struct String string_join(const char* sep, &stack String parts, unsigned long count);
