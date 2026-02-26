// SafeC Standard Library — String implementation
#include "string.h"
#include "../mem.h"
#include "../str.h"
#include "../fmt.h"
#include "../convert.h"

// ── Internal: ensure at least `need` bytes of capacity (not counting NUL) ─────
int String::reserve_(unsigned long need) {
    unsigned long needed_cap = need + 1UL;
    if (needed_cap <= self.cap) { return 1; }
    unsigned long new_cap = self.cap < 16UL ? 16UL : self.cap;
    while (new_cap < needed_cap) { new_cap = new_cap * 2UL; }
    unsafe {
        char* nd = (char*)realloc_buf((void*)self.data, new_cap);
        if (nd == (char*)0) { return 0; }
        self.data = nd;
        self.cap  = new_cap;
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
    if (cstr != (const char*)0) { s.push(cstr); }
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
        if (s.data != (char*)0) { s.data[0] = 0; }
        else { s.cap = 0UL; }
    }
    return s;
}

struct String string_repeat(const char* cstr, unsigned long n) {
    unsigned long slen  = str_len(cstr);
    unsigned long total = slen * n;
    struct String s = string_with_cap(total);
    unsigned long i = 0UL;
    while (i < n) { s.push(cstr); i = i + 1UL; }
    return s;
}

struct String string_join(const char* sep, &stack String parts, unsigned long count) {
    struct String out = string_new();
    unsigned long i = 0UL;
    while (i < count) {
        if (i > 0UL && sep != (const char*)0) { out.push(sep); }
        unsafe {
            // Index into the parts array
            struct String* arr = (struct String*)parts;
            out.push_str(arr[i]);
        }
        i = i + 1UL;
    }
    return out;
}

struct String String::clone() const {
    return string_from_n(self.data, self.len);
}

void String::free() {
    unsafe { if (self.data != (char*)0) { dealloc((void*)self.data); } }
    self.data = (char*)0;
    self.len  = 0UL;
    self.cap  = 0UL;
}

// ── Access ────────────────────────────────────────────────────────────────────
unsigned long String::length() const  { return self.len; }
int           String::is_empty() const { return self.len == 0UL; }
const char*   String::as_ptr() const  { return (const char*)self.data; }

int String::char_at(unsigned long idx) const {
    if (idx >= self.len) { return -1; }
    return (int)(unsigned char)self.data[idx];
}

void String::set_char(unsigned long idx, char c) {
    if (idx < self.len) { self.data[idx] = c; }
}

// ── Append ────────────────────────────────────────────────────────────────────
int String::push_char(char c) {
    if (!self.reserve_(self.len + 1UL)) { return 0; }
    self.data[self.len] = c;
    self.len = self.len + 1UL;
    self.data[self.len] = 0;
    return 1;
}

int String::push(const char* cstr) {
    if (cstr == (const char*)0) { return 1; }
    unsigned long clen = str_len(cstr);
    if (!self.reserve_(self.len + clen)) { return 0; }
    unsafe { safe_memcpy((void*)(self.data + self.len), (const void*)cstr, clen + 1UL); }
    self.len = self.len + clen;
    return 1;
}

int String::push_str(&stack String other) {
    return self.push((const char*)other.data);
}

int String::push_int(long long v) {
    char buf[32];
    unsafe { fmt_int((char*)buf, 32UL, v); }
    return self.push((const char*)buf);
}

int String::push_uint(unsigned long long v) {
    char buf[32];
    unsafe { fmt_uint((char*)buf, 32UL, v); }
    return self.push((const char*)buf);
}

int String::push_float(double v, int decimals) {
    char buf[64];
    unsafe { fmt_float((char*)buf, 64UL, v, decimals); }
    return self.push((const char*)buf);
}

int String::push_bool(int v) {
    return self.push(v ? "true" : "false");
}

// ── Modification ──────────────────────────────────────────────────────────────
void String::clear() {
    self.len = 0UL;
    if (self.data != (char*)0) { self.data[0] = 0; }
}

void String::truncate(unsigned long new_len) {
    if (new_len >= self.len) { return; }
    self.len = new_len;
    if (self.data != (char*)0) { self.data[new_len] = 0; }
}

int String::insert(unsigned long idx, const char* cstr) {
    if (idx > self.len) { idx = self.len; }
    unsigned long clen = str_len(cstr);
    if (!self.reserve_(self.len + clen)) { return 0; }
    unsafe {
        safe_memmove((void*)(self.data + idx + clen),
                     (const void*)(self.data + idx), self.len - idx + 1UL);
        safe_memcpy((void*)(self.data + idx), (const void*)cstr, clen);
    }
    self.len = self.len + clen;
    return 1;
}

int String::remove_range(unsigned long start, unsigned long end) {
    if (start >= self.len || start >= end) { return 0; }
    if (end > self.len) { end = self.len; }
    unsigned long removed = end - start;
    unsafe {
        safe_memmove((void*)(self.data + start),
                     (const void*)(self.data + end), self.len - end + 1UL);
    }
    self.len = self.len - removed;
    return 1;
}

void String::replace_char(char from, char to) {
    unsigned long i = 0UL;
    while (i < self.len) {
        if (self.data[i] == from) { self.data[i] = to; }
        i = i + 1UL;
    }
}

int String::replace(const char* from, const char* to) {
    long long pos = self.index_of(from);
    if (pos < 0LL) { return 0; }
    unsigned long from_len = str_len(from);
    self.remove_range((unsigned long)pos, (unsigned long)pos + from_len);
    self.insert((unsigned long)pos, to);
    return 1;
}

int String::replace_all(const char* from, const char* to) {
    int count = 0;
    unsigned long from_len = str_len(from);
    if (from_len == 0UL) { return 0; }
    unsigned long to_len = str_len(to);
    unsigned long i = 0UL;
    while (i + from_len <= self.len) {
        long long pos = self.index_of(from);
        if (pos < 0LL || (unsigned long)pos < i) { break; }
        self.remove_range((unsigned long)pos, (unsigned long)pos + from_len);
        self.insert((unsigned long)pos, to);
        i = (unsigned long)pos + to_len;
        count = count + 1;
    }
    return count;
}

// ── Search ────────────────────────────────────────────────────────────────────
long long String::index_of(const char* needle) const {
    if (needle == (const char*)0 || self.data == (char*)0) { return -1LL; }
    const char* found = str_find((const char*)self.data, needle);
    if (found == (const char*)0) { return -1LL; }
    unsafe { return (long long)(found - (const char*)self.data); }
}

long long String::last_index_of(const char* needle) const {
    if (needle == (const char*)0 || self.data == (char*)0) { return -1LL; }
    unsigned long nlen = str_len(needle);
    if (nlen == 0UL) { return (long long)self.len; }
    long long result = -1LL;
    unsigned long i = 0UL;
    while (i + nlen <= self.len) {
        unsafe {
            if (safe_memcmp((const void*)(self.data + i), (const void*)needle, nlen) == 0) {
                result = (long long)i;
            }
        }
        i = i + 1UL;
    }
    return result;
}

int String::contains(const char* needle) const {
    return self.index_of(needle) >= 0LL;
}

int String::starts_with(const char* prefix) const {
    unsigned long plen = str_len(prefix);
    if (plen > self.len) { return 0; }
    return safe_memcmp((const void*)self.data, (const void*)prefix, plen) == 0;
}

int String::ends_with(const char* suffix) const {
    unsigned long slen = str_len(suffix);
    if (slen > self.len) { return 0; }
    unsafe {
        return safe_memcmp((const void*)(self.data + self.len - slen),
                           (const void*)suffix, slen) == 0;
    }
}

int String::count(const char* needle) const {
    unsigned long nlen = str_len(needle);
    if (nlen == 0UL) { return 0; }
    int cnt = 0;
    unsigned long i = 0UL;
    while (i + nlen <= self.len) {
        unsafe {
            if (safe_memcmp((const void*)(self.data + i), (const void*)needle, nlen) == 0) {
                cnt = cnt + 1;
                i = i + nlen;
            } else {
                i = i + 1UL;
            }
        }
    }
    return cnt;
}

// ── Transformation ────────────────────────────────────────────────────────────
struct String String::substr(unsigned long start, unsigned long end) const {
    if (start >= self.len) { return string_new(); }
    if (end > self.len) { end = self.len; }
    if (start >= end) { return string_new(); }
    return string_from_n((const char*)(self.data + start), end - start);
}

struct String String::to_upper() const {
    struct String out = self.clone();
    unsigned long i = 0UL;
    while (i < out.len) {
        char c = out.data[i];
        if (c >= 'a' && c <= 'z') { out.data[i] = c - (char)32; }
        i = i + 1UL;
    }
    return out;
}

struct String String::to_lower() const {
    struct String out = self.clone();
    unsigned long i = 0UL;
    while (i < out.len) {
        char c = out.data[i];
        if (c >= 'A' && c <= 'Z') { out.data[i] = c + (char)32; }
        i = i + 1UL;
    }
    return out;
}

struct String String::trim_left() const {
    unsigned long start = 0UL;
    while (start < self.len) {
        char c = self.data[start];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { start = start + 1UL; }
        else { break; }
    }
    return string_from_n((const char*)(self.data + start), self.len - start);
}

struct String String::trim_right() const {
    unsigned long end = self.len;
    while (end > 0UL) {
        char c = self.data[end - 1UL];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { end = end - 1UL; }
        else { break; }
    }
    return string_from_n((const char*)self.data, end);
}

struct String String::trim() const {
    struct String tmp = self.trim_left();
    struct String out = tmp.trim_right();
    tmp.free();
    return out;
}

// ── Comparison ────────────────────────────────────────────────────────────────
int String::eq(&stack String other) const {
    if (self.len != other.len) { return 0; }
    return safe_memcmp((const void*)self.data, (const void*)other.data, self.len) == 0;
}

int String::eq_cstr(const char* other) const {
    return str_cmp((const char*)self.data, other) == 0;
}

int String::cmp(&stack String other) const {
    return str_cmp((const char*)self.data, (const char*)other.data);
}

int String::lt(&stack String other) const { return self.cmp(other) < 0; }
int String::gt(&stack String other) const { return self.cmp(other) > 0; }

// ── Conversion ────────────────────────────────────────────────────────────────
long long String::parse_int(int* ok) const {
    return str_to_int((const char*)self.data, ok);
}

double String::parse_float(int* ok) const {
    return str_to_float((const char*)self.data, ok);
}
