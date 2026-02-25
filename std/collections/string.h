#pragma once
// SafeC Standard Library — String (mutable, heap-allocated)
// A growable UTF-8 byte string with rich manipulation methods.

struct String {
    char*         data; // NUL-terminated buffer
    unsigned long len;  // byte length (not including NUL)
    unsigned long cap;  // allocated buffer size (including NUL slot)
};

// ── Lifecycle ─────────────────────────────────────────────────────────────────
struct String string_new();
struct String string_from(const char* s);
struct String string_from_n(const char* s, unsigned long n);
struct String string_with_cap(unsigned long cap);
struct String string_repeat(const char* s, unsigned long n);
struct String string_clone(const struct String* s);
void          string_free(struct String* s);

// ── Access ────────────────────────────────────────────────────────────────────
unsigned long  string_len(const struct String* s);
int            string_is_empty(const struct String* s);
const char*    string_as_ptr(const struct String* s);   // NUL-terminated C string
int            string_char_at(const struct String* s, unsigned long idx); // -1 if OOB
void           string_set_char(struct String* s, unsigned long idx, char c);

// ── Append ────────────────────────────────────────────────────────────────────
int string_push_char(struct String* s, char c);
int string_push(struct String* s, const char* cstr);
int string_push_str(struct String* s, const struct String* other);
int string_push_int(struct String* s, long long v);
int string_push_uint(struct String* s, unsigned long long v);
int string_push_float(struct String* s, double v, int decimals);
int string_push_bool(struct String* s, int v);

// ── Modification ──────────────────────────────────────────────────────────────
void string_clear(struct String* s);
void string_truncate(struct String* s, unsigned long new_len);
int  string_insert(struct String* s, unsigned long idx, const char* cstr);
int  string_remove_range(struct String* s, unsigned long start, unsigned long end);
void string_replace_char(struct String* s, char from, char to);
int  string_replace(struct String* s, const char* from, const char* to); // replaces first occurrence
int  string_replace_all(struct String* s, const char* from, const char* to);

// ── Search ────────────────────────────────────────────────────────────────────
long long string_index_of(const struct String* s, const char* needle);
long long string_last_index_of(const struct String* s, const char* needle);
int       string_contains(const struct String* s, const char* needle);
int       string_starts_with(const struct String* s, const char* prefix);
int       string_ends_with(const struct String* s, const char* suffix);
int       string_count(const struct String* s, const char* needle); // count occurrences

// ── Transformation (return new String) ───────────────────────────────────────
struct String string_substr(const struct String* s, unsigned long start, unsigned long end);
struct String string_to_upper(const struct String* s);
struct String string_to_lower(const struct String* s);
struct String string_trim(const struct String* s);
struct String string_trim_left(const struct String* s);
struct String string_trim_right(const struct String* s);
struct String string_join(const char* sep, const struct String* parts, unsigned long count);

// ── Comparison ────────────────────────────────────────────────────────────────
int string_eq(const struct String* a, const struct String* b);
int string_eq_cstr(const struct String* s, const char* other);
int string_cmp(const struct String* a, const struct String* b); // <0, 0, >0
int string_lt(const struct String* a, const struct String* b);
int string_gt(const struct String* a, const struct String* b);

// ── Conversion ────────────────────────────────────────────────────────────────
long long string_parse_int(const struct String* s, int* ok);
double    string_parse_float(const struct String* s, int* ok);
