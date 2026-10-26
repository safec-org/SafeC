// SafeC Standard Library — HTTP cookies implementation (see cookie.h).
#pragma once
#include <std/http/cookie.h>
#include <std/http/http.h>
#include <std/http/headers.h>
#include <std/http/date.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>

namespace std {

static int ck_blank_(char c) { return c == ' ' || c == '\t'; }

static char ck_lower_(char c) {
    if (c >= 'A' && c <= 'Z') { return (char)(c + 32); }
    return c;
}

// s[a..b) equals 'word' ignoring case.
static int ck_word_is_(const char* s, unsigned long a, unsigned long b, const char* word) {
    unsafe {
        unsigned long i = 0UL;
        while (a + i < b) {
            if (word[i] == (char)0 || ck_lower_(s[a + i]) != ck_lower_(word[i])) { return 0; }
            i = i + 1UL;
        }
        return word[i] == (char)0;
    }
}

static void ck_parse_into_(&CookieList list, const char* s) {
    unsafe {
        unsigned long i = 0UL;
        while (s[i] != (char)0) {
            while (ck_blank_(s[i]) || s[i] == ';') { i = i + 1UL; }
            unsigned long a = i;
            while (s[i] != (char)0 && s[i] != ';') { i = i + 1UL; }
            unsigned long b = i;
            while (b > a && ck_blank_(s[b - 1UL])) { b = b - 1UL; }
            unsigned long eq = a;
            while (eq < b && s[eq] != '=') { eq = eq + 1UL; }
            if (eq < b && eq > a) {
                unsigned long ne = eq;
                while (ne > a && ck_blank_(s[ne - 1UL])) { ne = ne - 1UL; }
                unsigned long va = eq + 1UL;
                while (va < b && ck_blank_(s[va])) { va = va + 1UL; }
                unsigned long vb = b;
                if (vb - va >= 2UL && s[va] == '"' && s[vb - 1UL] == '"') { va = va + 1UL; vb = vb - 1UL; }
                struct CookiePair p;
                p.name = String::from_n_(&s[a], ne - a);
                p.value = String::from_n_(&s[va], vb - va);
                list.items.push(p);
            }
        }
    }
}

struct CookieList CookieList::parse(const &char cookie_header) {
    struct CookieList l;
    l.items = Vec::new();
    ck_parse_into_(&l, cookie_header);
    return l;
}

struct CookieList CookieList::from_request(const &HttpRequest req) {
    struct CookieList l;
    l.items = Vec::new();
    struct HttpHeaders h = HttpHeaders::parse(req.headers.as_ptr());
    unsigned long i = 0UL;
    while (i < h.length()) {
        match (h.name_at(i)) {
            case some(n): {
                if (http_name_eq(n, "cookie")) {
                    match (h.value_at(i)) { case some(v): { ck_parse_into_(&l, v); } case null: { } }
                }
            }
            case null: { }
        }
        i = i + 1UL;
    }
    h.free();
    return l;
}

unsigned long CookieList::length() const { return self.items.length(); }

const ?&char CookieList::get(const &char name) const {
    unsigned long i = 0UL;
    while (i < self.items.length()) {
        &CookiePair p = self.items.at(i);
        if (p.name.eq_cstr(name)) { return p.value.as_ptr(); }
        i = i + 1UL;
    }
    return null;
}

const ?&char CookieList::name_at(unsigned long i) const {
    if (i >= self.items.length()) { return null; }
    return self.items.at(i).name.as_ptr();
}

const ?&char CookieList::value_at(unsigned long i) const {
    if (i >= self.items.length()) { return null; }
    return self.items.at(i).value.as_ptr();
}

void CookieList::free() {
    unsigned long i = 0UL;
    while (i < self.items.length()) {
        &CookiePair p = self.items.at(i);
        p.name.free();
        p.value.free();
        i = i + 1UL;
    }
    self.items.free();
}

struct SetCookie SetCookie::new(const &char name, const &char value) {
    struct SetCookie c;
    c.name = String::from(name);
    c.value = String::from(value);
    c.path = String::new();
    c.domain = String::new();
    c.max_age = -1LL;
    c.expires = 0LL;
    c.secure = 0;
    c.http_only = 0;
    c.same_site = COOKIE_SAMESITE_UNSET;
    c.partitioned = 0;
    return c;
}

struct SetCookie SetCookie::removal(const &char name, const &char path) {
    struct SetCookie c = SetCookie::new(name, "");
    c.path.push(path);
    c.max_age = 0LL;
    c.expires = 1LL;
    return c;
}

struct SetCookie SetCookie::parse(const &char header_value, ?&int ok) {
    struct SetCookie c = SetCookie::new("", "");
    int good = 0;
    unsafe {
        const char* s = header_value;
        unsigned long i = 0UL;
        int first = 1;
        while (s[i] != (char)0) {
            while (ck_blank_(s[i])) { i = i + 1UL; }
            unsigned long a = i;
            while (s[i] != (char)0 && s[i] != ';') { i = i + 1UL; }
            unsigned long b = i;
            if (s[i] == ';') { i = i + 1UL; }
            while (b > a && ck_blank_(s[b - 1UL])) { b = b - 1UL; }
            unsigned long eq = a;
            while (eq < b && s[eq] != '=') { eq = eq + 1UL; }
            unsigned long ke = eq;
            while (ke > a && ck_blank_(s[ke - 1UL])) { ke = ke - 1UL; }
            unsigned long va = eq < b ? eq + 1UL : b;
            while (va < b && ck_blank_(s[va])) { va = va + 1UL; }
            if (first) {
                first = 0;
                if (eq >= b || ke == a) { break; }
                c.name.push_n_(&s[a], ke - a);
                unsigned long vb = b;
                if (vb - va >= 2UL && s[va] == '"' && s[vb - 1UL] == '"') { va = va + 1UL; vb = vb - 1UL; }
                c.value.push_n_(&s[va], vb - va);
                good = 1;
                continue;
            }
            struct String val = String::from_n_(&s[va], b - va);
            if (ck_word_is_(s, a, ke, "path")) { c.path.clear(); c.path.push_str(&val); }
            else if (ck_word_is_(s, a, ke, "domain")) {
                c.domain.clear();
                if (val.starts_with(".")) { c.domain.push(&val.as_ptr()[1]); } else { c.domain.push_str(&val); }
            }
            else if (ck_word_is_(s, a, ke, "max-age")) { int pok = 0; long long v = val.parse_int(&pok); if (pok) { c.max_age = v; } }
            else if (ck_word_is_(s, a, ke, "expires")) { int dok = 0; long long t = http_date_parse(val.as_ptr(), &dok); if (dok) { c.expires = t; } }
            else if (ck_word_is_(s, a, ke, "secure")) { c.secure = 1; }
            else if (ck_word_is_(s, a, ke, "httponly")) { c.http_only = 1; }
            else if (ck_word_is_(s, a, ke, "partitioned")) { c.partitioned = 1; }
            else if (ck_word_is_(s, a, ke, "samesite")) {
                if (val.eq_cstr_ignore_case("lax")) { c.same_site = COOKIE_SAMESITE_LAX; }
                else if (val.eq_cstr_ignore_case("strict")) { c.same_site = COOKIE_SAMESITE_STRICT; }
                else if (val.eq_cstr_ignore_case("none")) { c.same_site = COOKIE_SAMESITE_NONE; }
            }
            val.free();
        }
    }
    if (ok != (int*)0) { unsafe { *ok = good; } }
    return c;
}

struct String SetCookie::to_string() const {
    struct String out = String::new();
    out.push_str(&self.name);
    out.push_char('=');
    out.push_str(&self.value);
    if (!self.path.is_empty()) { out.push("; Path="); out.push_str(&self.path); }
    if (!self.domain.is_empty()) { out.push("; Domain="); out.push_str(&self.domain); }
    if (self.max_age >= 0LL) { out.push("; Max-Age="); out.push_int(self.max_age); }
    if (self.expires != 0LL) {
        struct String d = http_date_format(self.expires);
        out.push("; Expires=");
        out.push_str(&d);
        d.free();
    }
    if (self.secure || self.same_site == COOKIE_SAMESITE_NONE) { out.push("; Secure"); }
    if (self.http_only) { out.push("; HttpOnly"); }
    if (self.same_site == COOKIE_SAMESITE_LAX) { out.push("; SameSite=Lax"); }
    if (self.same_site == COOKIE_SAMESITE_STRICT) { out.push("; SameSite=Strict"); }
    if (self.same_site == COOKIE_SAMESITE_NONE) { out.push("; SameSite=None"); }
    if (self.partitioned) { out.push("; Partitioned"); }
    return out;
}

void SetCookie::free() {
    self.name.free();
    self.value.free();
    self.path.free();
    self.domain.free();
}

void http_set_cookie(&HttpResponse resp, const &SetCookie c) {
    struct String v = c.to_string();
    http_headers_add(&resp.headers, "Set-Cookie", v.as_ptr());
    v.free();
}

} // namespace std
