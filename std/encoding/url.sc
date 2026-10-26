// SafeC Standard Library — URL implementation (see url.h)
#pragma once
#include <std/encoding/url.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>

namespace std {

// ── Percent-encoding ───────────────────────────────────────────────────────

static int url_is_unreserved_(char c) {
    if (c >= 'A' && c <= 'Z') { return 1; }
    if (c >= 'a' && c <= 'z') { return 1; }
    if (c >= '0' && c <= '9') { return 1; }
    return c == '-' || c == '_' || c == '.' || c == '~';
}

// RFC 3986 sub-delims plus ':' '@' '/': what a path keeps unescaped.
static int url_is_path_char_(char c) {
    if (url_is_unreserved_(c)) { return 1; }
    return c == '!' || c == '$' || c == '&' || c == '\'' || c == '(' || c == ')' ||
           c == '*' || c == '+' || c == ',' || c == ';' || c == '=' || c == ':' ||
           c == '@' || c == '/';
}

static char url_hex_digit_(int v) {
    if (v < 10) { return (char)((int)'0' + v); }
    return (char)((int)'A' + (v - 10));
}

static void url_push_escape_(&String out, char c) {
    unsigned char b;
    unsafe { b = (unsigned char)c; }
    out.push_char('%');
    out.push_char(url_hex_digit_(((int)b >> 4) & 0xF));
    out.push_char(url_hex_digit_((int)b & 0xF));
}

// mode 0: component, 1: path, 2: form
static struct String url_encode_mode_(const &char s, int mode) {
    struct String out = String::new();
    unsigned long i = 0UL;
    while (1) {
        char c;
        unsafe { c = s[i]; }
        if (c == (char)0) { break; }
        int keep = 0;
        if (mode == 1) { keep = url_is_path_char_(c); }
        else { keep = url_is_unreserved_(c) || (mode == 2 && c == '*'); }
        if (keep) {
            out.push_char(c);
        } else if (mode == 2 && c == ' ') {
            out.push_char('+');
        } else {
            url_push_escape_(&out, c);
        }
        i = i + 1UL;
    }
    return out;
}

struct String url_encode(const &char s) { return url_encode_mode_(s, 0); }
struct String url_encode_path(const &char s) { return url_encode_mode_(s, 1); }
struct String form_encode(const &char s) { return url_encode_mode_(s, 2); }

static int url_hex_val_(char c, int* ok) {
    if (c >= '0' && c <= '9') { return (int)c - (int)'0'; }
    if (c >= 'a' && c <= 'f') { return (int)c - (int)'a' + 10; }
    if (c >= 'A' && c <= 'F') { return (int)c - (int)'A' + 10; }
    unsafe { *ok = 0; }
    return 0;
}

static int url_is_hex_(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

struct String url_decode(const &char s, ?&int ok) {
    struct String out = String::new();
    int good = 1;
    unsigned long i = 0UL;
    while (1) {
        char c;
        unsafe { c = s[i]; }
        if (c == (char)0) { break; }
        if (c == '%') {
            char h1;
            char h2;
            unsafe { h1 = s[i + 1UL]; }
            if (h1 == (char)0) { good = 0; break; }
            unsafe { h2 = s[i + 2UL]; }
            if (h2 == (char)0) { good = 0; break; }
            int hv1 = url_hex_val_(h1, &good);
            int hv2 = url_hex_val_(h2, &good);
            if (good == 0) { break; }
            unsafe { out.push_char((char)((hv1 << 4) | hv2)); }
            i = i + 3UL;
        } else {
            unsafe { out.push_char(c); }
            i = i + 1UL;
        }
    }
    if (ok != (int*)0) { unsafe { *ok = good; } }
    return out;
}

// Decodes s[0..n) into 'out' (form rules when 'plus' is set: '+' = space);
// malformed escapes are kept as-is.
static void url_decode_lenient_(&String out, const char* s, unsigned long n, int plus) {
    unsigned long i = 0UL;
    unsafe {
        while (i < n) {
            char c = s[i];
            if (c == '%' && i + 2UL < n &&
                url_is_hex_(s[i + 1UL]) && url_is_hex_(s[i + 2UL])) {
                int g = 1;
                int hv = (url_hex_val_(s[i + 1UL], &g) << 4) | url_hex_val_(s[i + 2UL], &g);
                out.push_char((char)hv);
                i = i + 3UL;
                continue;
            }
            if (plus && c == '+') { out.push_char(' '); } else { out.push_char(c); }
            i = i + 1UL;
        }
    }
}

struct String form_decode(const &char s) {
    struct String out = String::new();
    unsigned long n = 0UL;
    unsafe { while (s[n] != (char)0) { n = n + 1UL; } }
    url_decode_lenient_(&out, s, n, 1);
    return out;
}

// ── QueryParams ────────────────────────────────────────────────────────────

static int url_cstr_eq_(const char* a, const char* b) {
    unsafe {
        unsigned long i = 0UL;
        while (a[i] != (char)0 && a[i] == b[i]) { i = i + 1UL; }
        return a[i] == b[i];
    }
}

struct QueryParams QueryParams::new() {
    struct QueryParams q;
    q.pairs = Vec::new();
    return q;
}

struct QueryParams QueryParams::parse_bytes([]const char qs) {
    unsigned long n = qs.len;
    struct QueryParams q = QueryParams::new();
    unsigned long i = 0UL;
    unsafe {
        if (n > 0UL && qs[0] == '?') { i = 1UL; }
        while (i < n) {
            unsigned long end = i;
            while (end < n && qs[end] != '&') { end = end + 1UL; }
            if (end > i) {
                unsigned long eq = i;
                while (eq < end && qs[eq] != '=') { eq = eq + 1UL; }
                struct UrlQueryPair p;
                p.key = String::new();
                p.value = String::new();
                url_decode_lenient_(&p.key, &qs[i], eq - i, 1);
                if (eq < end) { url_decode_lenient_(&p.value, &qs[eq + 1UL], end - eq - 1UL, 1); }
                q.pairs.push(p);
            }
            i = end + 1UL;
        }
    }
    return q;
}

struct QueryParams QueryParams::parse(const &char qs) {
    unsigned long n = 0UL;
    unsafe { while (qs[n] != (char)0) { n = n + 1UL; } }
    []const char bytes;
    unsafe { bytes = ((const char*)qs)[0..n]; }
    return QueryParams::parse_bytes(bytes);
}

unsigned long QueryParams::length() const { return self.pairs.length(); }

const ?&char QueryParams::get_nth(const &char key, unsigned long n) const {
    unsigned long seen = 0UL;
    unsigned long i = 0UL;
    while (i < self.pairs.length()) {
        &UrlQueryPair p = self.pairs.at(i);
        if (url_cstr_eq_(p.key.as_ptr(), key)) {
            if (seen == n) { return p.value.as_ptr(); }
            seen = seen + 1UL;
        }
        i = i + 1UL;
    }
    return null;
}

const ?&char QueryParams::get(const &char key) const { return self.get_nth(key, 0UL); }

const &char QueryParams::get_or(const &char key, const &char fallback) const {
    const ?&char v = self.get_nth(key, 0UL);
    match (v) {
        case some(s): { return s; }
        case null: { return fallback; }
    }
    return fallback;
}

unsigned long QueryParams::count(const &char key) const {
    unsigned long n = 0UL;
    unsigned long i = 0UL;
    while (i < self.pairs.length()) {
        if (url_cstr_eq_(self.pairs.at(i).key.as_ptr(), key)) { n = n + 1UL; }
        i = i + 1UL;
    }
    return n;
}

int QueryParams::has(const &char key) const { return self.count(key) > 0UL; }

const ?&char QueryParams::key_at(unsigned long i) const {
    if (i >= self.pairs.length()) { return null; }
    return self.pairs.at(i).key.as_ptr();
}

const ?&char QueryParams::value_at(unsigned long i) const {
    if (i >= self.pairs.length()) { return null; }
    return self.pairs.at(i).value.as_ptr();
}

void QueryParams::append(const &char key, const &char value) {
    struct UrlQueryPair p;
    p.key = String::from(key);
    p.value = String::from(value);
    self.pairs.push(p);
}

void QueryParams::set(const &char key, const &char value) {
    int done = 0;
    unsigned long i = 0UL;
    while (i < self.pairs.length()) {
        &UrlQueryPair p = self.pairs.at(i);
        if (url_cstr_eq_(p.key.as_ptr(), key)) {
            if (!done) {
                std::replace(&p.value, String::from(value)).drop();
                done = 1;
            } else {
                drop(self.pairs.remove(i));
                continue;
            }
        }
        i = i + 1UL;
    }
    if (!done) { self.append(key, value); }
}

unsigned long QueryParams::remove(const &char key) {
    unsigned long n = 0UL;
    unsigned long i = 0UL;
    while (i < self.pairs.length()) {
        &UrlQueryPair p = self.pairs.at(i);
        if (url_cstr_eq_(p.key.as_ptr(), key)) {
            drop(self.pairs.remove(i));
            n = n + 1UL;
            continue;
        }
        i = i + 1UL;
    }
    return n;
}

void QueryParams::sort() {
    // Insertion sort: stable, and query strings are short.
    unsigned long n = self.pairs.length();
    unsigned long i = 1UL;
    while (i < n) {
        unsigned long j = i;
        while (j > 0UL) {
            &UrlQueryPair a = self.pairs.at(j - 1UL);
            &UrlQueryPair b = self.pairs.at(j);
            if (a.key.cmp(&b.key) <= 0) { break; }
            std::swap(a, b);
            j = j - 1UL;
        }
        i = i + 1UL;
    }
}

struct String QueryParams::to_string() const {
    struct String out = String::new();
    unsigned long i = 0UL;
    while (i < self.pairs.length()) {
        &UrlQueryPair p = self.pairs.at(i);
        if (i > 0UL) { out.push_char('&'); }
        struct String k = form_encode(p.key.as_ptr());
        struct String v = form_encode(p.value.as_ptr());
        out.push_str(&k);
        out.push_char('=');
        out.push_str(&v);
        k.free();
        v.free();
        i = i + 1UL;
    }
    return out;
}

struct QueryParams QueryParams::clone() const {
    struct QueryParams q = QueryParams::new();
    unsigned long i = 0UL;
    while (i < self.pairs.length()) {
        &UrlQueryPair p = self.pairs.at(i);
        q.append(p.key.as_ptr(), p.value.as_ptr());
        i = i + 1UL;
    }
    return q;
}

void QueryParams::free() { self.drop(); }

void QueryParams::drop() {
    self.pairs.drop();   // and each pair's strings
}

// ── Url ────────────────────────────────────────────────────────────────────

struct Url Url::new() {
    struct Url u;
    u.scheme = String::new();
    u.username = String::new();
    u.password = String::new();
    u.host = String::new();
    u.port = -1;
    u.path = String::new();
    u.query = String::new();
    u.fragment = String::new();
    u.has_authority = 0;
    u.has_query = 0;
    u.has_fragment = 0;
    return u;
}

static char url_lower_(char c) {
    if (c >= 'A' && c <= 'Z') { return (char)(c + 32); }
    return c;
}

static int url_is_alpha_(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
static int url_is_digit_(char c) { return c >= '0' && c <= '9'; }

static void url_push_range_(&String out, const char* s, unsigned long a, unsigned long b, int lower) {
    unsafe {
        unsigned long i = a;
        while (i < b) {
            out.push_char(lower ? url_lower_(s[i]) : s[i]);
            i = i + 1UL;
        }
    }
}

struct Url Url::parse(const &char text, ?&int ok) {
    struct Url u = Url::new();
    int good = 1;
    const char* s = text;
    unsigned long n = 0UL;
    unsafe {
        while (s[n] != (char)0) {
            unsigned char b = (unsigned char)s[n];
            if (b <= (unsigned char)0x20 || b == (unsigned char)0x7F) { good = 0; }
            n = n + 1UL;
        }
    }
    unsigned long pos = 0UL;
    // scheme: ALPHA *( ALPHA / DIGIT / "+" / "-" / "." ) ":" — only if the
    // ':' comes before any '/', '?', '#'.
    if (good) {
        unsafe {
            unsigned long i = 0UL;
            while (i < n && s[i] != ':' && s[i] != '/' && s[i] != '?' && s[i] != '#') { i = i + 1UL; }
            if (i < n && s[i] == ':' && i > 0UL) {
                int valid = url_is_alpha_(s[0]);
                unsigned long k = 1UL;
                while (k < i) {
                    char c = s[k];
                    if (!(url_is_alpha_(c) || url_is_digit_(c) || c == '+' || c == '-' || c == '.')) { valid = 0; }
                    k = k + 1UL;
                }
                if (valid) {
                    url_push_range_(&u.scheme, s, 0UL, i, 1);
                    pos = i + 1UL;
                } else {
                    good = 0;
                }
            }
        }
    }
    // authority
    if (good) {
        unsafe {
            if (pos + 1UL < n && s[pos] == '/' && s[pos + 1UL] == '/') {
                u.has_authority = 1;
                pos = pos + 2UL;
                unsigned long aend = pos;
                while (aend < n && s[aend] != '/' && s[aend] != '?' && s[aend] != '#') { aend = aend + 1UL; }
                // userinfo: up to the last '@'
                unsigned long at = aend;
                unsigned long k = pos;
                while (k < aend) { if (s[k] == '@') { at = k; } k = k + 1UL; }
                unsigned long hs = pos;
                if (at < aend) {
                    unsigned long colon = pos;
                    while (colon < at && s[colon] != ':') { colon = colon + 1UL; }
                    url_push_range_(&u.username, s, pos, colon, 0);
                    if (colon < at) { url_push_range_(&u.password, s, colon + 1UL, at, 0); }
                    hs = at + 1UL;
                }
                unsigned long he = aend;
                unsigned long portStart = aend;
                if (hs < aend && s[hs] == '[') {
                    unsigned long rb = hs;
                    while (rb < aend && s[rb] != ']') { rb = rb + 1UL; }
                    if (rb >= aend) {
                        good = 0;
                    } else {
                        url_push_range_(&u.host, s, hs + 1UL, rb, 1);
                        if (rb + 1UL < aend) {
                            if (s[rb + 1UL] != ':') { good = 0; }
                            portStart = rb + 2UL;
                        }
                    }
                } else {
                    unsigned long c = hs;
                    while (c < aend && s[c] != ':') { c = c + 1UL; }
                    he = c;
                    if (c < aend) { portStart = c + 1UL; }
                    url_push_range_(&u.host, s, hs, he, 1);
                }
                if (good && portStart < aend) {
                    long port = 0L;
                    unsigned long p = portStart;
                    while (p < aend) {
                        if (!url_is_digit_(s[p])) { good = 0; break; }
                        port = port * 10L + (long)((int)s[p] - (int)'0');
                        if (port > 65535L) { good = 0; break; }
                        p = p + 1UL;
                    }
                    if (good) { u.port = (int)port; }
                }
                pos = aend;
            }
        }
    }
    if (good) {
        unsafe {
            unsigned long pe = pos;
            while (pe < n && s[pe] != '?' && s[pe] != '#') { pe = pe + 1UL; }
            url_push_range_(&u.path, s, pos, pe, 0);
            pos = pe;
            if (pos < n && s[pos] == '?') {
                u.has_query = 1;
                unsigned long qe = pos + 1UL;
                while (qe < n && s[qe] != '#') { qe = qe + 1UL; }
                url_push_range_(&u.query, s, pos + 1UL, qe, 0);
                pos = qe;
            }
            if (pos < n && s[pos] == '#') {
                u.has_fragment = 1;
                url_push_range_(&u.fragment, s, pos + 1UL, n, 0);
            }
        }
    }
    if (!good) {
        u.free();
        u = Url::new();
    }
    if (ok != (int*)0) { unsafe { *ok = good; } }
    return u;
}

static int url_host_needs_brackets_(const &String host) {
    return host.find_char(':') >= 0LL;
}

struct String Url::to_string() const {
    struct String out = String::new();
    if (!self.scheme.is_empty()) {
        out.push_str(&self.scheme);
        out.push_char(':');
    }
    if (self.has_authority) {
        out.push("//");
        if (!self.username.is_empty() || !self.password.is_empty()) {
            out.push_str(&self.username);
            if (!self.password.is_empty()) {
                out.push_char(':');
                out.push_str(&self.password);
            }
            out.push_char('@');
        }
        if (url_host_needs_brackets_(&self.host)) {
            out.push_char('[');
            out.push_str(&self.host);
            out.push_char(']');
        } else {
            out.push_str(&self.host);
        }
        if (self.port >= 0) {
            out.push_char(':');
            out.push_int((long long)self.port);
        }
    }
    out.push_str(&self.path);
    if (self.has_query) {
        out.push_char('?');
        out.push_str(&self.query);
    }
    if (self.has_fragment) {
        out.push_char('#');
        out.push_str(&self.fragment);
    }
    return out;
}

// RFC 3986 §5.2.4, over an input buffer consumed from the front.
struct String url_remove_dot_segments(const &char path) {
    struct String in = String::from(path);
    struct String out = String::new();
    while (!in.is_empty()) {
        if (in.starts_with("../")) { in.remove_range(0UL, 3UL); continue; }
        if (in.starts_with("./")) { in.remove_range(0UL, 2UL); continue; }
        if (in.starts_with("/./")) { in.remove_range(0UL, 2UL); continue; }
        if (in.eq_cstr("/.")) { in.truncate(1UL); continue; }
        if (in.starts_with("/../") || in.eq_cstr("/..")) {
            if (in.eq_cstr("/..")) { in.truncate(1UL); } else { in.remove_range(0UL, 3UL); }
            long long last = out.rfind_char('/');
            if (last < 0LL) { out.clear(); } else { out.truncate((unsigned long)last); }
            continue;
        }
        if (in.eq_cstr(".") || in.eq_cstr("..")) { in.clear(); continue; }
        // Move the first segment (with its leading '/', if any) to out.
        unsigned long end = 0UL;
        if (in.char_at(0UL) == (int)'/') { end = 1UL; }
        while (end < in.length() && in.char_at(end) != (int)'/') { end = end + 1UL; }
        unsafe { out.push_n_(in.as_ptr(), end); }
        in.remove_range(0UL, end);
    }
    in.free();
    return out;
}

// RFC 3986 §5.2.3
static struct String url_merge_paths_(const &Url base, const &String refPath) {
    struct String out = String::new();
    if (base.has_authority && base.path.is_empty()) {
        out.push_char('/');
        out.push_str(refPath);
        return out;
    }
    long long last = base.path.rfind_char('/');
    if (last >= 0LL) {
        unsafe { out.push_n_(base.path.as_ptr(), (unsigned long)last + 1UL); }
    }
    out.push_str(refPath);
    return out;
}

static void url_copy_authority_(&Url dst, const &Url src) {
    std::replace(&dst.username, src.username.clone()).drop();
    std::replace(&dst.password, src.password.clone()).drop();
    std::replace(&dst.host, src.host.clone()).drop();
    dst.port = src.port;
    dst.has_authority = src.has_authority;
}

struct Url Url::resolve(const &char refText, ?&int ok) const {
    int pok = 0;
    struct Url r = Url::parse(refText, &pok);
    if (!pok || self.scheme.is_empty()) {
        r.free();
        if (ok != (int*)0) { unsafe { *ok = 0; } }
        return Url::new();
    }
    struct Url t = Url::new();
    if (!r.scheme.is_empty()) {
        t.scheme.push_str(&r.scheme);
        url_copy_authority_(&t, &r);
        struct String p = url_remove_dot_segments(r.path.as_ptr());
        t.path.free(); t.path = p;
        t.query.push_str(&r.query); t.has_query = r.has_query;
    } else {
        if (r.has_authority) {
            url_copy_authority_(&t, &r);
            struct String p = url_remove_dot_segments(r.path.as_ptr());
            t.path.free(); t.path = p;
            t.query.push_str(&r.query); t.has_query = r.has_query;
        } else {
            if (r.path.is_empty()) {
                t.path.push_str(&self.path);
                if (r.has_query) {
                    t.query.push_str(&r.query); t.has_query = 1;
                } else {
                    t.query.push_str(&self.query); t.has_query = self.has_query;
                }
            } else {
                if (r.path.starts_with("/")) {
                    struct String p = url_remove_dot_segments(r.path.as_ptr());
                    t.path.free(); t.path = p;
                } else {
                    struct String merged = url_merge_paths_(self, &r.path);
                    struct String p = url_remove_dot_segments(merged.as_ptr());
                    merged.free();
                    t.path.free(); t.path = p;
                }
                t.query.push_str(&r.query); t.has_query = r.has_query;
            }
            url_copy_authority_(&t, self);
        }
        t.scheme.push_str(&self.scheme);
    }
    t.fragment.push_str(&r.fragment);
    t.has_fragment = r.has_fragment;
    r.free();
    if (ok != (int*)0) { unsafe { *ok = 1; } }
    return t;
}

int Url::port_or_default() const {
    if (self.port >= 0) { return self.port; }
    if (self.scheme.eq_cstr("http") || self.scheme.eq_cstr("ws")) { return 80; }
    if (self.scheme.eq_cstr("https") || self.scheme.eq_cstr("wss")) { return 443; }
    if (self.scheme.eq_cstr("ftp")) { return 21; }
    return -1;
}

struct String Url::host_port() const {
    struct String out = String::new();
    if (url_host_needs_brackets_(&self.host)) {
        out.push_char('[');
        out.push_str(&self.host);
        out.push_char(']');
    } else {
        out.push_str(&self.host);
    }
    if (self.port >= 0) {
        out.push_char(':');
        out.push_int((long long)self.port);
    }
    return out;
}

struct String Url::origin() const {
    struct String out = self.scheme.clone();
    out.push("://");
    struct String hp = self.host_port();
    out.push_str(&hp);
    hp.free();
    return out;
}

struct String Url::request_target() const {
    struct String out = String::new();
    if (self.path.is_empty()) { out.push_char('/'); } else { out.push_str(&self.path); }
    if (self.has_query) {
        out.push_char('?');
        out.push_str(&self.query);
    }
    return out;
}

struct QueryParams Url::query_params() const {
    return QueryParams::parse_bytes(self.query.as_bytes());
}

void Url::set_query_params(const &QueryParams params) {
    std::replace(&self.query, params.to_string()).drop();
    self.has_query = params.length() > 0UL;
}

struct String Url::path_decoded() const {
    struct String out = String::new();
    url_decode_lenient_(&out, self.path.as_ptr(), self.path.length(), 0);
    return out;
}

int Url::is_absolute() const { return !self.scheme.is_empty(); }

struct Url Url::clone() const {
    struct Url u;
    u.scheme = self.scheme.clone();
    u.username = self.username.clone();
    u.password = self.password.clone();
    u.host = self.host.clone();
    u.port = self.port;
    u.path = self.path.clone();
    u.query = self.query.clone();
    u.fragment = self.fragment.clone();
    u.has_authority = self.has_authority;
    u.has_query = self.has_query;
    u.has_fragment = self.has_fragment;
    return u;
}

void Url::free() { self.drop(); }

void Url::drop() {
    self.scheme.free();
    self.username.free();
    self.password.free();
    self.host.free();
    self.path.free();
    self.query.free();
    self.fragment.free();
}

} // namespace std
