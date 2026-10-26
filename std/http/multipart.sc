// SafeC Standard Library — multipart/form-data (see multipart.h).
#pragma once
#include <std/http/multipart.h>
#include <std/http/http.h>
#include <std/http/headers.h>
#include <std/crypto/rng.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>

namespace std {

// Index of 'needle' (length m) in hay[from..n), or -1.
static long long mp_find_(const char* hay, unsigned long n, unsigned long from, const char* needle, unsigned long m) {
    if (m == 0UL || n < m) { return -1LL; }
    unsafe {
        unsigned long i = from;
        while (i + m <= n) {
            if (hay[i] == needle[0]) {
                unsigned long k = 1UL;
                while (k < m && hay[i + k] == needle[k]) { k = k + 1UL; }
                if (k == m) { return (long long)i; }
            }
            i = i + 1UL;
        }
    }
    return -1LL;
}

struct MultipartForm MultipartForm::parse(const &char content_type, const &unsigned char body,
                                          unsigned long len, ?&int ok) {
    struct MultipartForm f;
    f.parts = Vec::new();
    int good = 1;
    struct String boundary = http_header_param(content_type, "boundary");
    if (boundary.is_empty() || boundary.length() > 200UL) { good = 0; }
    struct String delim = String::from("--");
    delim.push_str(&boundary);
    const char* b;
    unsafe { b = (const char*)body; }
    unsigned long dl = delim.length();
    long long pos = good ? mp_find_(b, len, 0UL, delim.as_ptr(), dl) : -1LL;
    if (pos < 0LL) { good = 0; }
    int closed = 0;
    while (good) {
        unsigned long p = (unsigned long)pos + dl;
        // "--" after the delimiter closes the body.
        unsafe {
            if (p + 1UL < len && b[p] == '-' && b[p + 1UL] == '-') { closed = 1; break; }
            while (p < len && (b[p] == ' ' || b[p] == '\t')) { p = p + 1UL; }
            if (p + 1UL < len && b[p] == '\r' && b[p + 1UL] == '\n') { p = p + 2UL; }
            else if (p < len && b[p] == '\n') { p = p + 1UL; }
            else { good = 0; break; }
        }
        long long hend = mp_find_(b, len, p, "\r\n\r\n", 4UL);
        unsigned long bodyStart = 0UL;
        if (hend >= 0LL) { bodyStart = (unsigned long)hend + 4UL; }
        else {
            hend = mp_find_(b, len, p, "\n\n", 2UL);
            if (hend < 0LL) { good = 0; break; }
            bodyStart = (unsigned long)hend + 2UL;
        }
        // The next delimiter, which must start a line.
        struct String nextDelim = String::from("\r\n");
        nextDelim.push_str(&delim);
        long long next = mp_find_(b, len, bodyStart, nextDelim.as_ptr(), nextDelim.length());
        unsigned long bodyEnd = 0UL;
        if (next >= 0LL) {
            bodyEnd = (unsigned long)next;
            pos = next + 2LL;
        } else {
            struct String nd2 = String::from("\n");
            nd2.push_str(&delim);
            next = mp_find_(b, len, bodyStart, nd2.as_ptr(), nd2.length());
            nd2.free();
            if (next < 0LL) { nextDelim.free(); good = 0; break; }
            bodyEnd = (unsigned long)next;
            pos = next + 1LL;
        }
        nextDelim.free();
        struct MultipartPart part;
        struct String rawHead = String::new();
        unsafe { rawHead.push_n_(&b[p], (unsigned long)hend - p); }
        part.headers = HttpHeaders::parse(rawHead.as_ptr());
        rawHead.free();
        const &char cd = part.headers.get_or("Content-Disposition", "");
        part.name = http_header_param(cd, "name");
        part.filename = http_header_param(cd, "filename");
        part.has_filename = 0;
        {
            struct String lc = String::from(cd);
            struct String low = lc.to_lower();
            part.has_filename = low.contains("filename=");
            low.free();
            lc.free();
        }
        part.content_type = String::from(part.headers.get_or("Content-Type", "text/plain"));
        part.data = String::new();
        unsafe { part.data.push_n_(&b[bodyStart], bodyEnd - bodyStart); }
        f.parts.push(part);
    }
    if (!closed) { good = 0; }
    boundary.free();
    delim.free();
    if (!good) {
        f.free();
        f.parts = Vec::new();
    }
    if (ok != (int*)0) { unsafe { *ok = good; } }
    return f;
}

struct MultipartForm MultipartForm::from_request(const &HttpRequest req, ?&int ok) {
    struct String ct = req.header("Content-Type");
    struct MultipartForm f;
    unsafe { f = MultipartForm::parse(ct.as_ptr(), (const &unsigned char)req.body.as_ptr(), req.body.length(), ok); }
    ct.free();
    return f;
}

unsigned long MultipartForm::length() const { return self.parts.length(); }

?&MultipartPart MultipartForm::get(const &char name) {
    unsigned long i = 0UL;
    while (i < self.parts.length()) {
        &MultipartPart p = self.parts.at(i);
        if (p.name.eq_cstr(name)) { return p; }
        i = i + 1UL;
    }
    return null;
}

?&MultipartPart MultipartForm::part_at(unsigned long i) {
    if (i >= self.parts.length()) { return null; }
    return self.parts.at(i);
}

const ?&char MultipartForm::value(const &char name) const {
    unsigned long i = 0UL;
    while (i < self.parts.length()) {
        &MultipartPart p = self.parts.at(i);
        if (p.name.eq_cstr(name)) { return p.data.as_ptr(); }
        i = i + 1UL;
    }
    return null;
}

void MultipartForm::free() { self.drop(); }

void MultipartForm::drop() {
    self.parts.drop();   // and each part
}

struct MultipartBuilder MultipartBuilder::new() {
    struct MultipartBuilder m;
    m.boundary = String::from("----SafeCFormBoundary");
    unsigned char rnd[12];
    struct RngCtx rng;
    int rok;
    unsafe { rok = rng.init(); }
    int i = 0;
    while (i < 12) { rnd[i] = (unsigned char)(i * 37 + 11); i = i + 1; }
    if (rok) { unsafe { rng.fill_n_(&rnd[0], 12UL); } }
    const &char digits = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    i = 0;
    while (i < 12) {
        unsafe { m.boundary.push_char(digits[(int)rnd[i] % 62]); }
        i = i + 1;
    }
    m.body = String::new();
    return m;
}

static void mp_quote_into_(&String out, const &char s) {
    out.push_char('"');
    unsafe {
        unsigned long i = 0UL;
        while (s[i] != (char)0) {
            char c = s[i];
            if (c == '"') { out.push("%22"); }
            else if (c == '\r') { out.push("%0D"); }
            else if (c == '\n') { out.push("%0A"); }
            else { out.push_char(c); }
            i = i + 1UL;
        }
    }
    out.push_char('"');
}

void MultipartBuilder::add_field(const &char name, const &char value) {
    self.body.push("--");
    self.body.push_str(&self.boundary);
    self.body.push("\r\nContent-Disposition: form-data; name=");
    mp_quote_into_(&self.body, name);
    self.body.push("\r\n\r\n");
    self.body.push(value);
    self.body.push("\r\n");
}

void MultipartBuilder::add_file_n_(const &char name, const &char filename, const &char content_type,
                                const &unsigned char data, unsigned long len) {
    self.body.push("--");
    self.body.push_str(&self.boundary);
    self.body.push("\r\nContent-Disposition: form-data; name=");
    mp_quote_into_(&self.body, name);
    self.body.push("; filename=");
    mp_quote_into_(&self.body, filename);
    self.body.push("\r\nContent-Type: ");
    self.body.push(content_type);
    self.body.push("\r\n\r\n");
    unsafe { self.body.push_n_((const char*)data, len); }
    self.body.push("\r\n");
}

void MultipartBuilder::add_file(const &char name, const &char filename, const &char content_type, []const unsigned char data_s) {
    unsigned long len = data_s.len;
    const &unsigned char data = crypto_in_(data_s, 0UL);
    self.add_file_n_(name, filename, content_type, data, len);
}

struct String MultipartBuilder::content_type() const {
    struct String s = String::from("multipart/form-data; boundary=");
    s.push_str(&self.boundary);
    return s;
}

void MultipartBuilder::finish() {
    self.body.push("--");
    self.body.push_str(&self.boundary);
    self.body.push("--\r\n");
}

void MultipartBuilder::free() { self.drop(); }

void MultipartBuilder::drop() {
    self.boundary.free();
    self.body.free();
}

} // namespace std
