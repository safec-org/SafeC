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
        void* raw = realloc_buf((void*)self.data, new_cap);
        if (raw == (void*)0) { return 0; }
        self.data = (&heap char)raw;
        self.cap  = new_cap;
    }
    return 1;
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────
struct String string_new() {
    struct String s;
    s.len = 0UL;
    unsafe {
        void* raw = alloc(16UL);
        if (raw == (void*)0) {
            s.data = (&heap char)0;
            s.cap = 0UL;
            return s;
        }
        s.data = (&heap char)raw;
    }
    unsafe { s.data[0] = (char)0; }
    s.cap = 16UL;
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
        void* raw = alloc(n + 1UL);
        if (raw == (void*)0) {
            s.data = (&heap char)0;
            s.cap = 0UL;
            return s;
        }
        s.data = (&heap char)raw;
        s.cap = n + 1UL;
        safe_memcpy((void*)s.data, (const void*)cstr, n);
    }
    unsafe { s.data[n] = (char)0; }
    s.len = n;
    return s;
}

struct String string_with_cap(unsigned long cap) {
    struct String s;
    s.len = 0UL;
    s.cap = cap + 1UL;
    unsafe {
        void* raw = alloc(s.cap);
        if (raw == (void*)0) {
            s.data = (&heap char)0;
            s.cap = 0UL;
            return s;
        }
        s.data = (&heap char)raw;
    }
    unsafe { s.data[0] = (char)0; }
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
            struct String* arr = (struct String*)parts;
            out.push_str((&stack String)arr[i]);
        }
        i = i + 1UL;
    }
    return out;
}

struct String String::clone() const {
    unsafe { return string_from_n((const char*)self.data, self.len); }
}

void String::free() {
    unsafe {
        if (self.cap > 0UL) { dealloc((void*)self.data); }
        self.data = (&heap char)0;
    }
    self.len  = 0UL;
    self.cap  = 0UL;
}

// ── Access ────────────────────────────────────────────────────────────────────
unsigned long String::length() const  { return self.len; }
int           String::is_empty() const { return self.len == 0UL; }

const char* String::as_ptr() const {
    unsafe { return (const char*)self.data; }
}

int String::char_at(unsigned long idx) const {
    if (idx >= self.len) { return -1; }
    unsafe { return (int)(unsigned char)self.data[idx]; }
}

void String::set_char(unsigned long idx, char c) {
    if (idx < self.len) { unsafe { self.data[idx] = c; } }
}

// ── Capacity ──────────────────────────────────────────────────────────────────
int String::reserve(unsigned long additional) {
    return self.reserve_(self.len + additional);
}

void String::shrink_to_fit() {
    if (self.len + 1UL >= self.cap) { return; }
    unsigned long new_cap = self.len + 1UL;
    unsafe {
        void* raw = realloc_buf((void*)self.data, new_cap);
        if (raw == (void*)0) { return; }
        self.data = (&heap char)raw;
        self.cap  = new_cap;
    }
}

// ── Append ────────────────────────────────────────────────────────────────────
int String::push_char(char c) {
    if (!self.reserve_(self.len + 1UL)) { return 0; }
    unsafe { self.data[self.len] = c; }
    self.len = self.len + 1UL;
    unsafe { self.data[self.len] = (char)0; }
    return 1;
}

int String::push(const char* cstr) {
    if (cstr == (const char*)0) { return 1; }
    unsigned long clen = str_len(cstr);
    if (!self.reserve_(self.len + clen)) { return 0; }
    unsafe { safe_memcpy((void*)((char*)self.data + self.len), (const void*)cstr, clen + 1UL); }
    self.len = self.len + clen;
    return 1;
}

int String::push_str(&stack String other) {
    unsafe { return self.push((const char*)other.data); }
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
    if (self.cap > 0UL) { unsafe { self.data[0] = (char)0; } }
}

void String::truncate(unsigned long new_len) {
    if (new_len >= self.len) { return; }
    self.len = new_len;
    if (self.cap > 0UL) { unsafe { self.data[new_len] = (char)0; } }
}

int String::insert(unsigned long idx, const char* cstr) {
    if (idx > self.len) { idx = self.len; }
    unsigned long clen = str_len(cstr);
    if (!self.reserve_(self.len + clen)) { return 0; }
    unsafe {
        safe_memmove((void*)((char*)self.data + idx + clen),
                     (const void*)((char*)self.data + idx), self.len - idx + 1UL);
        safe_memcpy((void*)((char*)self.data + idx), (const void*)cstr, clen);
    }
    self.len = self.len + clen;
    return 1;
}

int String::remove_range(unsigned long start, unsigned long end) {
    if (start >= self.len || start >= end) { return 0; }
    if (end > self.len) { end = self.len; }
    unsigned long removed = end - start;
    unsafe {
        safe_memmove((void*)((char*)self.data + start),
                     (const void*)((char*)self.data + end), self.len - end + 1UL);
    }
    self.len = self.len - removed;
    return 1;
}

void String::replace_char(char from, char to) {
    unsigned long i = 0UL;
    while (i < self.len) {
        unsafe { if (self.data[i] == from) { self.data[i] = to; } }
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
    int cnt = 0;
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
        cnt = cnt + 1;
    }
    return cnt;
}

void String::reverse() {
    if (self.len < 2UL) { return; }
    unsigned long lo = 0UL;
    unsigned long hi = self.len - 1UL;
    while (lo < hi) {
        unsafe {
            char tmp = self.data[lo];
            self.data[lo] = self.data[hi];
            self.data[hi] = tmp;
        }
        lo = lo + 1UL;
        hi = hi - 1UL;
    }
}

int String::pop_char() {
    if (self.len == 0UL) { return -1; }
    self.len = self.len - 1UL;
    unsafe {
        int c = (int)(unsigned char)self.data[self.len];
        self.data[self.len] = (char)0;
        return c;
    }
}

// ── Search ────────────────────────────────────────────────────────────────────
long long String::index_of(const char* needle) const {
    if (needle == (const char*)0 || self.cap == 0UL) { return -1LL; }
    unsafe {
        const char* found = str_find((const char*)self.data, needle);
        if (found == (const char*)0) { return -1LL; }
        return (long long)(found - (const char*)self.data);
    }
}

long long String::last_index_of(const char* needle) const {
    if (needle == (const char*)0 || self.cap == 0UL) { return -1LL; }
    unsigned long nlen = str_len(needle);
    if (nlen == 0UL) { return (long long)self.len; }
    long long result = -1LL;
    unsigned long i = 0UL;
    while (i + nlen <= self.len) {
        unsafe {
            if (safe_memcmp((const void*)((char*)self.data + i), (const void*)needle, nlen) == 0) {
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
    unsafe { return safe_memcmp((const void*)self.data, (const void*)prefix, plen) == 0; }
}

int String::ends_with(const char* suffix) const {
    unsigned long slen = str_len(suffix);
    if (slen > self.len) { return 0; }
    unsafe {
        return safe_memcmp((const void*)((char*)self.data + self.len - slen),
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
            if (safe_memcmp((const void*)((char*)self.data + i), (const void*)needle, nlen) == 0) {
                cnt = cnt + 1;
                i = i + nlen;
            } else {
                i = i + 1UL;
            }
        }
    }
    return cnt;
}

long long String::find_char(char c) const {
    unsigned long i = 0UL;
    while (i < self.len) {
        unsafe { if (self.data[i] == c) { return (long long)i; } }
        i = i + 1UL;
    }
    return -1LL;
}

long long String::rfind_char(char c) const {
    if (self.len == 0UL) { return -1LL; }
    unsigned long i = self.len;
    while (i > 0UL) {
        i = i - 1UL;
        unsafe { if (self.data[i] == c) { return (long long)i; } }
    }
    return -1LL;
}

// ── Transformation ────────────────────────────────────────────────────────────
struct String String::substr(unsigned long start, unsigned long end) const {
    if (start >= self.len) { return string_new(); }
    if (end > self.len) { end = self.len; }
    if (start >= end) { return string_new(); }
    unsafe { return string_from_n((const char*)((char*)self.data + start), end - start); }
}

struct String String::to_upper() const {
    struct String out = self.clone();
    unsigned long i = 0UL;
    while (i < out.len) {
        unsafe {
            char c = out.data[i];
            if (c >= 'a' && c <= 'z') { out.data[i] = (char)(c - (char)32); }
        }
        i = i + 1UL;
    }
    return out;
}

struct String String::to_lower() const {
    struct String out = self.clone();
    unsigned long i = 0UL;
    while (i < out.len) {
        unsafe {
            char c = out.data[i];
            if (c >= 'A' && c <= 'Z') { out.data[i] = (char)(c + (char)32); }
        }
        i = i + 1UL;
    }
    return out;
}

struct String String::trim_left() const {
    unsigned long start = 0UL;
    while (start < self.len) {
        char c;
        unsafe { c = self.data[start]; }
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { start = start + 1UL; }
        else { break; }
    }
    unsafe { return string_from_n((const char*)((char*)self.data + start), self.len - start); }
}

struct String String::trim_right() const {
    unsigned long end = self.len;
    while (end > 0UL) {
        char c;
        unsafe { c = self.data[end - 1UL]; }
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { end = end - 1UL; }
        else { break; }
    }
    unsafe { return string_from_n((const char*)self.data, end); }
}

struct String String::trim() const {
    struct String tmp = self.trim_left();
    struct String out = tmp.trim_right();
    tmp.free();
    return out;
}

struct String String::pad_left(unsigned long width, char fill) const {
    if (self.len >= width) { return self.clone(); }
    unsigned long pad = width - self.len;
    struct String out = string_with_cap(width);
    unsigned long i = 0UL;
    while (i < pad) { out.push_char(fill); i = i + 1UL; }
    unsafe { out.push((const char*)self.data); }
    return out;
}

struct String String::pad_right(unsigned long width, char fill) const {
    if (self.len >= width) { return self.clone(); }
    unsigned long pad = width - self.len;
    struct String out = string_with_cap(width);
    unsafe { out.push((const char*)self.data); }
    unsigned long i = 0UL;
    while (i < pad) { out.push_char(fill); i = i + 1UL; }
    return out;
}

struct String String::strip_prefix(const char* prefix) const {
    unsigned long plen = str_len(prefix);
    if (plen > self.len) { return self.clone(); }
    unsafe {
        if (safe_memcmp((const void*)self.data, (const void*)prefix, plen) != 0) {
            return self.clone();
        }
        return string_from_n((const char*)((char*)self.data + plen), self.len - plen);
    }
}

struct String String::strip_suffix(const char* suffix) const {
    unsigned long slen = str_len(suffix);
    if (slen > self.len) { return self.clone(); }
    unsafe {
        if (safe_memcmp((const void*)((char*)self.data + self.len - slen), (const void*)suffix, slen) != 0) {
            return self.clone();
        }
        return string_from_n((const char*)self.data, self.len - slen);
    }
}

struct String String::repeat(unsigned long n) const {
    if (n == 0UL) { return string_new(); }
    struct String out = string_with_cap(self.len * n);
    unsigned long i = 0UL;
    while (i < n) {
        unsafe { out.push((const char*)self.data); }
        i = i + 1UL;
    }
    return out;
}

struct String String::capitalize() const {
    if (self.len == 0UL) { return string_new(); }
    struct String out = self.clone();
    unsafe {
        char first = out.data[0];
        if (first >= 'a' && first <= 'z') { out.data[0] = (char)(first - (char)32); }
    }
    // Lowercase the rest
    unsigned long i = 1UL;
    while (i < out.len) {
        unsafe {
            char c = out.data[i];
            if (c >= 'A' && c <= 'Z') { out.data[i] = (char)(c + (char)32); }
        }
        i = i + 1UL;
    }
    return out;
}

// ── Split ─────────────────────────────────────────────────────────────────────
unsigned long String::split(const char* delim, &stack String out, unsigned long max) const {
    if (max == 0UL || self.cap == 0UL) { return 0UL; }
    unsigned long dlen = str_len(delim);
    if (dlen == 0UL) {
        // Empty delimiter: return clone in first slot
        unsafe {
            struct String* arr = (struct String*)out;
            arr[0] = self.clone();
        }
        return 1UL;
    }
    unsigned long cnt = 0UL;
    unsigned long start = 0UL;
    while (start <= self.len && cnt < max) {
        // If this is the last slot, take remainder
        if (cnt == max - 1UL) {
            unsafe {
                struct String* arr = (struct String*)out;
                arr[cnt] = string_from_n((const char*)((char*)self.data + start), self.len - start);
            }
            cnt = cnt + 1UL;
            break;
        }
        // Search for delimiter
        long long pos = -1LL;
        if (start + dlen <= self.len) {
            unsigned long j = start;
            while (j + dlen <= self.len) {
                unsafe {
                    if (safe_memcmp((const void*)((char*)self.data + j), (const void*)delim, dlen) == 0) {
                        pos = (long long)j;
                        break;
                    }
                }
                j = j + 1UL;
            }
        }
        if (pos < 0LL) {
            // No more delimiters — take the rest
            unsafe {
                struct String* arr = (struct String*)out;
                arr[cnt] = string_from_n((const char*)((char*)self.data + start), self.len - start);
            }
            cnt = cnt + 1UL;
            break;
        }
        // Extract segment before delimiter
        unsafe {
            struct String* arr = (struct String*)out;
            arr[cnt] = string_from_n((const char*)((char*)self.data + start), (unsigned long)pos - start);
        }
        cnt = cnt + 1UL;
        start = (unsigned long)pos + dlen;
    }
    // If string ends with delimiter and we have room, add empty string
    if (start == self.len && cnt < max && cnt > 0UL) {
        unsafe {
            struct String* arr = (struct String*)out;
            arr[cnt] = string_new();
        }
        cnt = cnt + 1UL;
    }
    return cnt;
}

unsigned long String::split_lines(&stack String out, unsigned long max) const {
    if (max == 0UL || self.cap == 0UL) { return 0UL; }
    unsigned long cnt = 0UL;
    unsigned long start = 0UL;
    while (start <= self.len && cnt < max) {
        if (cnt == max - 1UL) {
            unsafe {
                struct String* arr = (struct String*)out;
                arr[cnt] = string_from_n((const char*)((char*)self.data + start), self.len - start);
            }
            cnt = cnt + 1UL;
            break;
        }
        // Find \n or \r\n
        unsigned long j = start;
        long long pos = -1LL;
        unsigned long skip = 1UL; // how many bytes the line ending occupies
        while (j < self.len) {
            unsafe {
                if (self.data[j] == '\n') {
                    pos = (long long)j;
                    skip = 1UL;
                    break;
                }
                if (self.data[j] == '\r') {
                    pos = (long long)j;
                    if (j + 1UL < self.len && self.data[j + 1UL] == '\n') {
                        skip = 2UL;
                    } else {
                        skip = 1UL;
                    }
                    break;
                }
            }
            j = j + 1UL;
        }
        if (pos < 0LL) {
            unsafe {
                struct String* arr = (struct String*)out;
                arr[cnt] = string_from_n((const char*)((char*)self.data + start), self.len - start);
            }
            cnt = cnt + 1UL;
            break;
        }
        unsafe {
            struct String* arr = (struct String*)out;
            arr[cnt] = string_from_n((const char*)((char*)self.data + start), (unsigned long)pos - start);
        }
        cnt = cnt + 1UL;
        start = (unsigned long)pos + skip;
    }
    return cnt;
}

unsigned long String::split_whitespace(&stack String out, unsigned long max) const {
    if (max == 0UL || self.cap == 0UL) { return 0UL; }
    unsigned long cnt = 0UL;
    unsigned long i = 0UL;
    // Skip leading whitespace
    while (i < self.len) {
        char c;
        unsafe { c = self.data[i]; }
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { i = i + 1UL; }
        else { break; }
    }
    while (i < self.len && cnt < max) {
        // If last slot, take remainder (trimmed of trailing whitespace)
        if (cnt == max - 1UL) {
            unsigned long end = self.len;
            while (end > i) {
                char c;
                unsafe { c = self.data[end - 1UL]; }
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { end = end - 1UL; }
                else { break; }
            }
            unsafe {
                struct String* arr = (struct String*)out;
                arr[cnt] = string_from_n((const char*)((char*)self.data + i), end - i);
            }
            cnt = cnt + 1UL;
            break;
        }
        // Find end of current token
        unsigned long start = i;
        while (i < self.len) {
            char c;
            unsafe { c = self.data[i]; }
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { break; }
            i = i + 1UL;
        }
        unsafe {
            struct String* arr = (struct String*)out;
            arr[cnt] = string_from_n((const char*)((char*)self.data + start), i - start);
        }
        cnt = cnt + 1UL;
        // Skip whitespace between tokens
        while (i < self.len) {
            char c;
            unsafe { c = self.data[i]; }
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { i = i + 1UL; }
            else { break; }
        }
    }
    return cnt;
}

// ── Comparison ────────────────────────────────────────────────────────────────
int String::eq(&stack String other) const {
    if (self.len != other.len) { return 0; }
    unsafe { return safe_memcmp((const void*)self.data, (const void*)other.data, self.len) == 0; }
}

int String::eq_cstr(const char* other) const {
    unsafe { return str_cmp((const char*)self.data, other) == 0; }
}

int String::cmp(&stack String other) const {
    unsafe { return str_cmp((const char*)self.data, (const char*)other.data); }
}

int String::lt(&stack String other) const { return self.cmp(other) < 0; }
int String::gt(&stack String other) const { return self.cmp(other) > 0; }

int String::eq_ignore_case(&stack String other) const {
    if (self.len != other.len) { return 0; }
    unsigned long i = 0UL;
    while (i < self.len) {
        unsafe {
            char a = self.data[i];
            char b = other.data[i];
            if (a >= 'A' && a <= 'Z') { a = (char)(a + (char)32); }
            if (b >= 'A' && b <= 'Z') { b = (char)(b + (char)32); }
            if (a != b) { return 0; }
        }
        i = i + 1UL;
    }
    return 1;
}

int String::eq_cstr_ignore_case(const char* other) const {
    unsigned long olen = str_len(other);
    if (self.len != olen) { return 0; }
    unsigned long i = 0UL;
    while (i < self.len) {
        unsafe {
            char a = self.data[i];
            char b = other[i];
            if (a >= 'A' && a <= 'Z') { a = (char)(a + (char)32); }
            if (b >= 'A' && b <= 'Z') { b = (char)(b + (char)32); }
            if (a != b) { return 0; }
        }
        i = i + 1UL;
    }
    return 1;
}

// ── Query ─────────────────────────────────────────────────────────────────────
int String::is_ascii() const {
    unsigned long i = 0UL;
    while (i < self.len) {
        unsafe { if ((unsigned char)self.data[i] > 127) { return 0; } }
        i = i + 1UL;
    }
    return 1;
}

int String::is_numeric() const {
    if (self.len == 0UL) { return 0; }
    unsigned long i = 0UL;
    while (i < self.len) {
        char c;
        unsafe { c = self.data[i]; }
        if (c < '0' || c > '9') { return 0; }
        i = i + 1UL;
    }
    return 1;
}

int String::is_alphanumeric() const {
    if (self.len == 0UL) { return 0; }
    unsigned long i = 0UL;
    while (i < self.len) {
        char c;
        unsafe { c = self.data[i]; }
        int ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
        if (!ok) { return 0; }
        i = i + 1UL;
    }
    return 1;
}

// ── Conversion ────────────────────────────────────────────────────────────────
long long String::parse_int(int* ok) const {
    unsafe { return str_to_int((const char*)self.data, ok); }
}

double String::parse_float(int* ok) const {
    unsafe { return str_to_float((const char*)self.data, ok); }
}
