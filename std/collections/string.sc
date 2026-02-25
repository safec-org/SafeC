// SafeC Standard Library — String implementation
#include "string.h"
#include "../mem.h"
#include "../str.h"
#include "../fmt.h"
#include "../convert.h"

// Internal: ensure at least `need` bytes of capacity (not counting NUL).
int string_reserve_(struct String* s, unsigned long need) {
    unsigned long needed_cap = need + 1UL;
    if (needed_cap <= s->cap) return 1;
    unsigned long new_cap = s->cap < 16UL ? 16UL : s->cap;
    while (new_cap < needed_cap) new_cap = new_cap * 2UL;
    unsafe {
        char* nd = (char*)realloc_buf((void*)s->data, new_cap);
        if (nd == (char*)0) return 0;
        s->data = nd;
        s->cap = new_cap;
    }
    return 1;
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────
struct String string_new() {
    struct String s;
    unsafe { s.data = (char*)alloc(16UL); }
    if (s.data != (char*)0) {
        s.data[0] = 0;
        s.cap = 16UL;
    } else {
        s.cap = 0UL;
    }
    s.len = 0UL;
    return s;
}

struct String string_from(const char* cstr) {
    struct String s = string_new();
    if (cstr != (const char*)0) string_push(&s, cstr);
    return s;
}

struct String string_from_n(const char* cstr, unsigned long n) {
    struct String s;
    s.len = 0UL;
    unsafe {
        s.data = (char*)alloc(n + 1UL);
        if (s.data == (char*)0) { s.cap = 0UL; return s; }
        s.cap = n + 1UL;
        safe_memcpy((void*)s.data, (const void*)cstr, n);
        s.data[n] = 0;
        s.len = n;
    }
    return s;
}

struct String string_with_cap(unsigned long cap) {
    struct String s;
    s.len = 0UL;
    s.cap = cap + 1UL;
    unsafe {
        s.data = (char*)alloc(s.cap);
        if (s.data != (char*)0) s.data[0] = 0;
        else s.cap = 0UL;
    }
    return s;
}

struct String string_repeat(const char* cstr, unsigned long n) {
    unsigned long slen = str_len(cstr);
    unsigned long total = slen * n;
    struct String s = string_with_cap(total);
    unsigned long i = 0UL;
    while (i < n) { string_push(&s, cstr); i = i + 1UL; }
    return s;
}

struct String string_clone(const struct String* src) {
    return string_from_n(src->data, src->len);
}

void string_free(struct String* s) {
    unsafe { if (s->data != (char*)0) dealloc((void*)s->data); }
    s->data = (char*)0;
    s->len = 0UL;
    s->cap = 0UL;
}

// ── Access ────────────────────────────────────────────────────────────────────
unsigned long string_len(const struct String* s)    { return s->len; }
int string_is_empty(const struct String* s)          { return s->len == 0UL; }
const char* string_as_ptr(const struct String* s)    { return (const char*)s->data; }

int string_char_at(const struct String* s, unsigned long idx) {
    if (idx >= s->len) return -1;
    return (int)(unsigned char)s->data[idx];
}

void string_set_char(struct String* s, unsigned long idx, char c) {
    if (idx < s->len) s->data[idx] = c;
}

// ── Append ────────────────────────────────────────────────────────────────────
int string_push_char(struct String* s, char c) {
    if (!string_reserve_(s, s->len + 1UL)) return 0;
    s->data[s->len] = c;
    s->len = s->len + 1UL;
    s->data[s->len] = 0;
    return 1;
}

int string_push(struct String* s, const char* cstr) {
    if (cstr == (const char*)0) return 1;
    unsigned long clen = str_len(cstr);
    if (!string_reserve_(s, s->len + clen)) return 0;
    unsafe { safe_memcpy((void*)(s->data + s->len), (const void*)cstr, clen + 1UL); }
    s->len = s->len + clen;
    return 1;
}

int string_push_str(struct String* s, const struct String* other) {
    return string_push(s, (const char*)other->data);
}

int string_push_int(struct String* s, long long v) {
    char buf[32];
    unsafe { fmt_int((char*)buf, 32UL, v); }
    return string_push(s, (const char*)buf);
}

int string_push_uint(struct String* s, unsigned long long v) {
    char buf[32];
    unsafe { fmt_uint((char*)buf, 32UL, v); }
    return string_push(s, (const char*)buf);
}

int string_push_float(struct String* s, double v, int decimals) {
    char buf[64];
    unsafe { fmt_float((char*)buf, 64UL, v, decimals); }
    return string_push(s, (const char*)buf);
}

int string_push_bool(struct String* s, int v) {
    return string_push(s, v ? "true" : "false");
}

// ── Modification ──────────────────────────────────────────────────────────────
void string_clear(struct String* s) {
    s->len = 0UL;
    if (s->data != (char*)0) s->data[0] = 0;
}

void string_truncate(struct String* s, unsigned long new_len) {
    if (new_len >= s->len) return;
    s->len = new_len;
    if (s->data != (char*)0) s->data[new_len] = 0;
}

int string_insert(struct String* s, unsigned long idx, const char* cstr) {
    if (idx > s->len) idx = s->len;
    unsigned long clen = str_len(cstr);
    if (!string_reserve_(s, s->len + clen)) return 0;
    unsafe {
        // Shift existing chars right
        safe_memmove((void*)(s->data + idx + clen),
                     (const void*)(s->data + idx), s->len - idx + 1UL);
        safe_memcpy((void*)(s->data + idx), (const void*)cstr, clen);
    }
    s->len = s->len + clen;
    return 1;
}

int string_remove_range(struct String* s, unsigned long start, unsigned long end) {
    if (start >= s->len || start >= end) return 0;
    if (end > s->len) end = s->len;
    unsigned long removed = end - start;
    unsafe {
        safe_memmove((void*)(s->data + start),
                     (const void*)(s->data + end), s->len - end + 1UL);
    }
    s->len = s->len - removed;
    return 1;
}

void string_replace_char(struct String* s, char from, char to) {
    unsigned long i = 0UL;
    while (i < s->len) {
        if (s->data[i] == from) s->data[i] = to;
        i = i + 1UL;
    }
}

int string_replace(struct String* s, const char* from, const char* to) {
    long long pos = string_index_of(s, from);
    if (pos < 0LL) return 0;
    unsigned long from_len = str_len(from);
    unsigned long to_len = str_len(to);
    string_remove_range(s, (unsigned long)pos, (unsigned long)pos + from_len);
    string_insert(s, (unsigned long)pos, to);
    return 1;
}

int string_replace_all(struct String* s, const char* from, const char* to) {
    int count = 0;
    unsigned long from_len = str_len(from);
    if (from_len == 0UL) return 0;
    unsigned long to_len = str_len(to);
    unsigned long i = 0UL;
    while (i + from_len <= s->len) {
        long long pos = string_index_of(s, from);
        if (pos < 0LL || (unsigned long)pos < i) break;
        string_remove_range(s, (unsigned long)pos, (unsigned long)pos + from_len);
        string_insert(s, (unsigned long)pos, to);
        i = (unsigned long)pos + to_len;
        count = count + 1;
    }
    return count;
}

// ── Search ────────────────────────────────────────────────────────────────────
long long string_index_of(const struct String* s, const char* needle) {
    if (needle == (const char*)0 || s->data == (char*)0) return -1LL;
    const char* found = str_find((const char*)s->data, needle);
    if (found == (const char*)0) return -1LL;
    unsafe { return (long long)(found - (const char*)s->data); }
}

long long string_last_index_of(const struct String* s, const char* needle) {
    if (needle == (const char*)0 || s->data == (char*)0) return -1LL;
    unsigned long nlen = str_len(needle);
    if (nlen == 0UL) return (long long)s->len;
    long long result = -1LL;
    unsigned long i = 0UL;
    while (i + nlen <= s->len) {
        unsafe {
            if (safe_memcmp((const void*)(s->data + i), (const void*)needle, nlen) == 0)
                result = (long long)i;
        }
        i = i + 1UL;
    }
    return result;
}

int string_contains(const struct String* s, const char* needle) {
    return string_index_of(s, needle) >= 0LL;
}

int string_starts_with(const struct String* s, const char* prefix) {
    unsigned long plen = str_len(prefix);
    if (plen > s->len) return 0;
    return safe_memcmp((const void*)s->data, (const void*)prefix, plen) == 0;
}

int string_ends_with(const struct String* s, const char* suffix) {
    unsigned long slen = str_len(suffix);
    if (slen > s->len) return 0;
    unsafe {
        return safe_memcmp((const void*)(s->data + s->len - slen),
                           (const void*)suffix, slen) == 0;
    }
}

int string_count(const struct String* s, const char* needle) {
    unsigned long nlen = str_len(needle);
    if (nlen == 0UL) return 0;
    int count = 0;
    unsigned long i = 0UL;
    while (i + nlen <= s->len) {
        unsafe {
            if (safe_memcmp((const void*)(s->data + i), (const void*)needle, nlen) == 0) {
                count = count + 1;
                i = i + nlen;
            } else {
                i = i + 1UL;
            }
        }
    }
    return count;
}

// ── Transformation ────────────────────────────────────────────────────────────
struct String string_substr(const struct String* s, unsigned long start, unsigned long end) {
    if (start >= s->len) return string_new();
    if (end > s->len) end = s->len;
    if (start >= end) return string_new();
    return string_from_n((const char*)(s->data + start), end - start);
}

struct String string_to_upper(const struct String* s) {
    struct String out = string_clone(s);
    unsigned long i = 0UL;
    while (i < out.len) {
        char c = out.data[i];
        if (c >= 'a' && c <= 'z') out.data[i] = c - (char)32;
        i = i + 1UL;
    }
    return out;
}

struct String string_to_lower(const struct String* s) {
    struct String out = string_clone(s);
    unsigned long i = 0UL;
    while (i < out.len) {
        char c = out.data[i];
        if (c >= 'A' && c <= 'Z') out.data[i] = c + (char)32;
        i = i + 1UL;
    }
    return out;
}

struct String string_trim_left(const struct String* s) {
    unsigned long start = 0UL;
    while (start < s->len) {
        char c = s->data[start];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') start = start + 1UL;
        else break;
    }
    return string_from_n((const char*)(s->data + start), s->len - start);
}

struct String string_trim_right(const struct String* s) {
    unsigned long end = s->len;
    while (end > 0UL) {
        char c = s->data[end - 1UL];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') end = end - 1UL;
        else break;
    }
    return string_from_n((const char*)s->data, end);
}

struct String string_trim(const struct String* s) {
    struct String tmp = string_trim_left(s);
    struct String out = string_trim_right(&tmp);
    string_free(&tmp);
    return out;
}

struct String string_join(const char* sep, const struct String* parts, unsigned long count) {
    struct String out = string_new();
    unsigned long i = 0UL;
    while (i < count) {
        if (i > 0UL && sep != (const char*)0) string_push(&out, sep);
        string_push_str(&out, &parts[i]);
        i = i + 1UL;
    }
    return out;
}

// ── Comparison ────────────────────────────────────────────────────────────────
int string_eq(const struct String* a, const struct String* b) {
    if (a->len != b->len) return 0;
    return safe_memcmp((const void*)a->data, (const void*)b->data, a->len) == 0;
}
int string_eq_cstr(const struct String* s, const char* other) {
    return str_cmp((const char*)s->data, other) == 0;
}
int string_cmp(const struct String* a, const struct String* b) {
    return str_cmp((const char*)a->data, (const char*)b->data);
}
int string_lt(const struct String* a, const struct String* b) { return string_cmp(a, b) < 0; }
int string_gt(const struct String* a, const struct String* b) { return string_cmp(a, b) > 0; }

// ── Conversion ────────────────────────────────────────────────────────────────
long long string_parse_int(const struct String* s, int* ok) {
    return str_to_int((const char*)s->data, ok);
}
double string_parse_float(const struct String* s, int* ok) {
    return str_to_float((const char*)s->data, ok);
}
