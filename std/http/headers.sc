// SafeC Standard Library — HttpHeaders implementation (see headers.h).
#pragma once
#include <std/http/headers.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>

namespace std {

static char hh_lower_(char c) {
    if (c >= 'A' && c <= 'Z') { return (char)(c + 32); }
    return c;
}

int http_name_eq(const &char a, const &char b) {
    unsafe {
        unsigned long i = 0UL;
        while (a[i] != (char)0 && hh_lower_(a[i]) == hh_lower_(b[i])) { i = i + 1UL; }
        return a[i] == (char)0 && b[i] == (char)0;
    }
}

struct String http_header_param(const &char value, const &char name) {
    struct String out = String::new();
    unsafe {
        const char* s = value;
        unsigned long i = 0UL;
        // Skip the main value.
        while (s[i] != (char)0 && s[i] != ';') {
            if (s[i] == '"') { i = i + 1UL; while (s[i] != (char)0 && s[i] != '"') { i = i + 1UL; } }
            if (s[i] != (char)0) { i = i + 1UL; }
        }
        while (s[i] == ';') {
            i = i + 1UL;
            while (hh_is_blank_(s[i])) { i = i + 1UL; }
            unsigned long ka = i;
            while (s[i] != (char)0 && s[i] != '=' && s[i] != ';') { i = i + 1UL; }
            unsigned long kb = i;
            while (kb > ka && hh_is_blank_(s[kb - 1UL])) { kb = kb - 1UL; }
            int same = 1;
            unsigned long k = 0UL;
            while (ka + k < kb) {
                if (name[k] == (char)0 || hh_lower_(s[ka + k]) != hh_lower_(name[k])) { same = 0; break; }
                k = k + 1UL;
            }
            if (same && name[k] != (char)0) { same = 0; }
            if (s[i] != '=') { continue; }
            i = i + 1UL;
            while (hh_is_blank_(s[i])) { i = i + 1UL; }
            struct String v = String::new();
            if (s[i] == '"') {
                i = i + 1UL;
                while (s[i] != (char)0 && s[i] != '"') {
                    if (s[i] == '\\' && s[i + 1UL] != (char)0) { i = i + 1UL; }
                    v.push_char(s[i]);
                    i = i + 1UL;
                }
                if (s[i] == '"') { i = i + 1UL; }
                while (s[i] != (char)0 && s[i] != ';') { i = i + 1UL; }
            } else {
                unsigned long va = i;
                while (s[i] != (char)0 && s[i] != ';') { i = i + 1UL; }
                unsigned long vb = i;
                while (vb > va && hh_is_blank_(s[vb - 1UL])) { vb = vb - 1UL; }
                v.push_n(&s[va], vb - va);
            }
            if (same) { out.free(); return v; }
            v.free();
        }
    }
    return out;
}

struct String http_header_main_value(const &char value) {
    struct String out = String::new();
    unsafe {
        unsigned long i = 0UL;
        while (hh_is_blank_(value[i])) { i = i + 1UL; }
        while (value[i] != (char)0 && value[i] != ';') { out.push_char(hh_lower_(value[i])); i = i + 1UL; }
        while (out.length() > 0UL && hh_is_blank_(out.as_ptr()[out.length() - 1UL])) { out.truncate(out.length() - 1UL); }
    }
    return out;
}

struct HttpHeaders HttpHeaders::new() {
    struct HttpHeaders h;
    h.fields = Vec::new();
    return h;
}

static int hh_is_blank_(char c) { return c == ' ' || c == '\t'; }

struct HttpHeaders HttpHeaders::parse(const &char raw) {
    struct HttpHeaders h = HttpHeaders::new();
    unsafe {
        const char* s = raw;
        unsigned long pos = 0UL;
        while (s[pos] != (char)0) {
            unsigned long end = pos;
            while (s[end] != (char)0 && s[end] != '\n') { end = end + 1UL; }
            unsigned long next = s[end] == '\n' ? end + 1UL : end;
            unsigned long e = end;
            if (e > pos && s[e - 1UL] == '\r') { e = e - 1UL; }
            if (e > pos && hh_is_blank_(s[pos]) && h.fields.length() > 0UL) {
                // obs-fold: continuation of the previous field's value.
                unsigned long a = pos;
                while (a < e && hh_is_blank_(s[a])) { a = a + 1UL; }
                unsigned long b = e;
                while (b > a && hh_is_blank_(s[b - 1UL])) { b = b - 1UL; }
                if (b > a) {
                    &HttpHeaderField last = h.fields.at(h.fields.length() - 1UL);
                    last.value.push_char(' ');
                    last.value.push_n(&s[a], b - a);
                }
            } else {
                unsigned long colon = pos;
                while (colon < e && s[colon] != ':') { colon = colon + 1UL; }
                if (colon < e) {
                    unsigned long ka = pos;
                    unsigned long kb = colon;
                    while (ka < kb && hh_is_blank_(s[ka])) { ka = ka + 1UL; }
                    while (kb > ka && hh_is_blank_(s[kb - 1UL])) { kb = kb - 1UL; }
                    unsigned long va = colon + 1UL;
                    unsigned long vb = e;
                    while (va < vb && hh_is_blank_(s[va])) { va = va + 1UL; }
                    while (vb > va && hh_is_blank_(s[vb - 1UL])) { vb = vb - 1UL; }
                    if (kb > ka) {
                        struct HttpHeaderField f;
                        f.name = String::from_n(&s[ka], kb - ka);
                        f.value = String::from_n(&s[va], vb - va);
                        h.fields.push(f);
                    }
                }
            }
            pos = next;
        }
    }
    return h;
}

unsigned long HttpHeaders::length() const { return self.fields.length(); }

const ?&char HttpHeaders::get_nth(const &char name, unsigned long n) const {
    unsigned long seen = 0UL;
    unsigned long i = 0UL;
    while (i < self.fields.length()) {
        &HttpHeaderField f = self.fields.at(i);
        if (http_name_eq(f.name.as_ptr(), name)) {
            if (seen == n) { return f.value.as_ptr(); }
            seen = seen + 1UL;
        }
        i = i + 1UL;
    }
    return null;
}

const ?&char HttpHeaders::get(const &char name) const { return self.get_nth(name, 0UL); }

const &char HttpHeaders::get_or(const &char name, const &char fallback) const {
    match (self.get_nth(name, 0UL)) {
        case some(v): { return v; }
        case null: { return fallback; }
    }
    return fallback;
}

unsigned long HttpHeaders::count(const &char name) const {
    unsigned long n = 0UL;
    unsigned long i = 0UL;
    while (i < self.fields.length()) {
        if (http_name_eq(self.fields.at(i).name.as_ptr(), name)) { n = n + 1UL; }
        i = i + 1UL;
    }
    return n;
}

struct String HttpHeaders::get_joined(const &char name) const {
    struct String out = String::new();
    unsigned long i = 0UL;
    while (i < self.fields.length()) {
        &HttpHeaderField f = self.fields.at(i);
        if (http_name_eq(f.name.as_ptr(), name)) {
            if (!out.is_empty()) { out.push(", "); }
            out.push_str(&f.value);
        }
        i = i + 1UL;
    }
    return out;
}

int HttpHeaders::has(const &char name) const { return self.count(name) > 0UL; }

// Is 'token' one of the comma-separated items of value v?
static int hh_list_has_(const char* v, const char* token) {
    unsafe {
        unsigned long i = 0UL;
        unsigned long tl = 0UL;
        while (token[tl] != (char)0) { tl = tl + 1UL; }
        while (v[i] != (char)0) {
            while (v[i] == ' ' || v[i] == '\t' || v[i] == ',') { i = i + 1UL; }
            unsigned long a = i;
            while (v[i] != (char)0 && v[i] != ',') { i = i + 1UL; }
            unsigned long b = i;
            while (b > a && (v[b - 1UL] == ' ' || v[b - 1UL] == '\t')) { b = b - 1UL; }
            if (b - a == tl && tl > 0UL) {
                unsigned long k = 0UL;
                while (k < tl && hh_lower_(v[a + k]) == hh_lower_(token[k])) { k = k + 1UL; }
                if (k == tl) { return 1; }
            }
        }
    }
    return 0;
}

int HttpHeaders::contains_token(const &char name, const &char token) const {
    unsigned long i = 0UL;
    while (i < self.fields.length()) {
        &HttpHeaderField f = self.fields.at(i);
        if (http_name_eq(f.name.as_ptr(), name) && hh_list_has_(f.value.as_ptr(), token)) { return 1; }
        i = i + 1UL;
    }
    return 0;
}

const ?&char HttpHeaders::name_at(unsigned long i) const {
    if (i >= self.fields.length()) { return null; }
    return self.fields.at(i).name.as_ptr();
}

const ?&char HttpHeaders::value_at(unsigned long i) const {
    if (i >= self.fields.length()) { return null; }
    return self.fields.at(i).value.as_ptr();
}

void HttpHeaders::append(const &char name, const &char value) {
    struct HttpHeaderField f;
    f.name = String::from(name);
    f.value = String::from(value);
    self.fields.push(f);
}

void HttpHeaders::set(const &char name, const &char value) {
    int done = 0;
    unsigned long i = 0UL;
    while (i < self.fields.length()) {
        &HttpHeaderField f = self.fields.at(i);
        if (http_name_eq(f.name.as_ptr(), name)) {
            if (!done) {
                f.value.free();
                f.value = String::from(value);
                done = 1;
            } else {
                f.name.free();
                f.value.free();
                self.fields.remove(i);
                continue;
            }
        }
        i = i + 1UL;
    }
    if (!done) { self.append(name, value); }
}

unsigned long HttpHeaders::remove(const &char name) {
    unsigned long n = 0UL;
    unsigned long i = 0UL;
    while (i < self.fields.length()) {
        &HttpHeaderField f = self.fields.at(i);
        if (http_name_eq(f.name.as_ptr(), name)) {
            f.name.free();
            f.value.free();
            self.fields.remove(i);
            n = n + 1UL;
            continue;
        }
        i = i + 1UL;
    }
    return n;
}

void HttpHeaders::clear() {
    unsigned long i = 0UL;
    while (i < self.fields.length()) {
        &HttpHeaderField f = self.fields.at(i);
        f.name.free();
        f.value.free();
        i = i + 1UL;
    }
    self.fields.clear();
}

void HttpHeaders::write_to(&String out) const {
    unsigned long i = 0UL;
    while (i < self.fields.length()) {
        &HttpHeaderField f = self.fields.at(i);
        out.push_str(&f.name);
        out.push(": ");
        out.push_str(&f.value);
        out.push("\r\n");
        i = i + 1UL;
    }
}

struct String HttpHeaders::to_raw() const {
    struct String out = String::new();
    unsigned long i = 0UL;
    while (i < self.fields.length()) {
        &HttpHeaderField f = self.fields.at(i);
        if (i > 0UL) { out.push("\r\n"); }
        out.push_str(&f.name);
        out.push(": ");
        out.push_str(&f.value);
        i = i + 1UL;
    }
    return out;
}

struct HttpHeaders HttpHeaders::clone() const {
    struct HttpHeaders h = HttpHeaders::new();
    unsigned long i = 0UL;
    while (i < self.fields.length()) {
        &HttpHeaderField f = self.fields.at(i);
        h.append(f.name.as_ptr(), f.value.as_ptr());
        i = i + 1UL;
    }
    return h;
}

void HttpHeaders::free() { self.drop(); }

void HttpHeaders::drop() {
    self.clear();
    self.fields.free();
}

} // namespace std
