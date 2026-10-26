// SafeC Standard Library — HTTP/1.1 implementation (see http.h)
// safeguard: link windows -lws2_32
#pragma once
#include <std/stddef.h>
#include <std/http/http.h>
#include <std/collections/string.h>
#include <std/mem.h>
#include <std/fmt.h>
#include <std/convert.h>
#include <std/str.h>
#include <std/sched/io_nb.h>
#include <std/io_traits.h>
#include <std/io_buf.h>
#include <std/sched/reactor.h>
#include <std/thread.h>
#include <std/encoding/url.h>
#include <std/encoding/deflate.h>
#include <std/http/headers.h>
#include <std/http/date.h>
#include <std/tls/tls.h>
extern void* malloc(size_t size);
extern void  free(void* ptr);
// A backend for io_nb.h's declarations (io_nb_bsd.sc / io_nb_linux.sc /
// io_nb_win32.sc) must be included by the caller before this file, the
// same way std/ipc/uds.h's callers pick uds_bsd.sc/uds_linux.sc themselves
// — see io_nb.h's top comment for the backend list. http.sc only ever
// calls the portable function signatures, never anything backend-specific.
// http_serve_reactor (below) needs the same treatment for its reactor
// backend — reactor_kqueue.sc/reactor_epoll.sc/reactor_win32.sc — also the
// caller's job to include first.

namespace std {

// A connected socket's fd is not a CRT file descriptor on Windows — it's a
// Winsock SOCKET (truncated to int here, same convention io_nb_win32.sc's
// tcp_accept_nb/tcp_listen_nb already use), and Winsock sockets only
// support recv/send/closesocket, not plain read/write/close (those work
// only for CRT-wrapped handles, not raw SOCKETs). POSIX has no such split
// — read/write/close work on any fd, sockets included. Plain wrapper
// functions, not macros: function-like macros need --compat-preprocessor
// (same reason mem.sc/pool.sc avoid panic.h's PANIC macro), and a real
// function works in safe mode with no extra link dependency either way.
#ifdef _WIN32
extern int recv(unsigned long long fd, void* buf, int count, int flags);
extern int send(unsigned long long fd, const void* buf, int count, int flags);
extern int closesocket(unsigned long long fd);
extern int WSAStartup(unsigned short versionRequested, void* wsaData);
static long sock_read_(int fd, void* buf, unsigned long count) {
    unsafe { return (long)recv((unsigned long long)fd, buf, (int)count, 0); }
}
static long sock_write_(int fd, const void* buf, unsigned long count) {
    unsafe { return (long)send((unsigned long long)fd, buf, (int)count, 0); }
}
static int sock_close_(int fd) {
    unsafe { return closesocket((unsigned long long)fd); }
}
#else
// read/write's real POSIX return type is ssize_t (8 bytes on every
// 64-bit target this runs on) — declared 'long' here to match, not
// 'int'. This isn't just pedantry: std/sched/reactor_epoll.sc declares
// this same libc 'read' symbol (for draining signalfd) with the correct
// 'long' return type, and a caller that also includes both reactor_*.sc
// and this file (see http_serve_reactor below) hits a real duplicate-
// extern-declaration mismatch if the two disagree.
extern ssize_t read(int fd, void* buf, size_t count);
extern ssize_t write(int fd, const void* buf, size_t count);
extern int close(int fd);
static long sock_read_(int fd, void* buf, unsigned long count) {
    unsafe { return read(fd, buf, count); }
}
#ifdef __linux__
// send() with MSG_NOSIGNAL rather than write(): a write to a socket whose
// peer has already closed raises SIGPIPE, whose default action kills the
// whole process. BSD/macOS opt out per socket instead (SO_NOSIGPIPE, set
// by io_nb_bsd.sc on every socket it hands out).
extern ssize_t send(int fd, const void* buf, size_t count, int flags);
#define HTTP_MSG_NOSIGNAL_ 0x4000
static long sock_write_(int fd, const void* buf, unsigned long count) {
    unsafe { return send(fd, buf, count, HTTP_MSG_NOSIGNAL_); }
}
#else
static long sock_write_(int fd, const void* buf, unsigned long count) {
    unsafe { return write(fd, buf, count); }
}
#endif
static int sock_close_(int fd) {
    unsafe { return close(fd); }
}
#endif

// 'struct addrinfo' (<netdb.h>/<ws2tcpip.h>). The first four ints and
// ai_next agree everywhere; Linux puts ai_addr before ai_canonname, BSD/
// macOS and Windows the other way round. (Windows' ai_addrlen is a size_t
// rather than a 4-byte socklen_t, but on 64-bit targets the pointers that
// follow land at the same offsets either way.)
struct AddrInfo_ {
    int    ai_flags;
    int    ai_family;
    int    ai_socktype;
    int    ai_protocol;
    unsigned long ai_addrlen;
#ifdef __linux__
    void*  ai_addr;
    char*  ai_canonname;
#else
    char*  ai_canonname;
    void*  ai_addr;
#endif
    struct AddrInfo_* ai_next;
};
extern int  getaddrinfo(const char* node, const char* service, const void* hints, void** res);
extern void freeaddrinfo(void* res);

#ifdef _WIN32
extern int getsockname(unsigned long long fd, void* addr, int* len);
extern int getpeername(unsigned long long fd, void* addr, int* len);
extern int shutdown(unsigned long long fd, int how);
static int sock_name_(int fd, unsigned char* sa, int peer) {
    int len = 16;
    unsafe {
        if (peer) { return getpeername((unsigned long long)fd, (void*)sa, &len); }
        return getsockname((unsigned long long)fd, (void*)sa, &len);
    }
}
static int sock_shutdown_(int fd, int how) {
    unsafe { return shutdown((unsigned long long)fd, how); }
}
#else
extern int getsockname(int fd, void* addr, unsigned int* len);
extern int getpeername(int fd, void* addr, unsigned int* len);
extern int shutdown(int fd, int how);
static int sock_name_(int fd, unsigned char* sa, int peer) {
    unsigned int len = 16U;
    unsafe {
        if (peer) { return getpeername(fd, (void*)sa, &len); }
        return getsockname(fd, (void*)sa, &len);
    }
}
static int sock_shutdown_(int fd, int how) {
    unsafe { return shutdown(fd, how); }
}
#endif
// SHUT_RD is 0 and SHUT_RDWR 2 on every platform (SD_RECEIVE/SD_BOTH on
// Winsock).
#define HTTP_SHUT_RD_   0

// sockaddr_in: the port sits at byte offset 2 and the IPv4 address at 4 on
// every platform (Linux/Windows: 2-byte family; BSD: length + family byte).
static int sock_addr_of_(int fd, int peer, &unsigned int addr, &unsigned short port) {
    unsigned char sa[16];
    int i = 0;
    while (i < 16) { sa[i] = (unsigned char)0; i = i + 1; }
    if (sock_name_(fd, &sa[0], peer) != 0) { return 0; }
    unsafe {
        *port = (unsigned short)(((unsigned int)sa[2] << 8) | (unsigned int)sa[3]);
        *addr = *(unsigned int*)&sa[4];
    }
    return 1;
}

static char http_lower_(char c) {
    if (c >= 'A' && c <= 'Z') { return (char)(c + 32); }
    return c;
}

// ── Header block parsing ──────────────────────────────────────────────────

// Header-line iteration over a raw "Key: value" block: lines end in "\r\n"
// or a bare "\n". Sets [*ks,*ke) to the trimmed key and [*vs,*ve) to the
// trimmed value of the line starting at 'pos'; *le to the end of the line
// including its terminator; returns the next line's start, or 0 when
// there's no line left. *ks == *ke for a line with no ':'.
static unsigned long http_next_header_(const &char h, unsigned long len, unsigned long pos,
                                       &unsigned long ks, &unsigned long ke,
                                       &unsigned long vs, &unsigned long ve) {
    if (pos >= len) {
        *ks = len; *ke = len; *vs = len; *ve = len;
        return 0UL;
    }
    unsafe {
        unsigned long end = pos;
        while (end < len && h[end] != '\n') { end = end + 1UL; }
        unsigned long next = end < len ? end + 1UL : len;
        if (end > pos && h[end - 1UL] == '\r') { end = end - 1UL; }
        unsigned long colon = pos;
        while (colon < end && h[colon] != ':') { colon = colon + 1UL; }
        *ks = pos; *ke = pos; *vs = end; *ve = end;
        if (colon < end) {
            unsigned long a = pos;
            unsigned long b = colon;
            while (a < b && (h[a] == ' ' || h[a] == '\t')) { a = a + 1UL; }
            while (b > a && (h[b - 1UL] == ' ' || h[b - 1UL] == '\t')) { b = b - 1UL; }
            unsigned long c = colon + 1UL;
            unsigned long d = end;
            while (c < d && (h[c] == ' ' || h[c] == '\t')) { c = c + 1UL; }
            while (d > c && (h[d - 1UL] == ' ' || h[d - 1UL] == '\t')) { d = d - 1UL; }
            *ks = a; *ke = b; *vs = c; *ve = d;
        }
        return next == pos ? 0UL : next;
    }
}

// Case-insensitive compare of h[ks..ke) with NUL-terminated 'name'.
static int http_key_is_(const &char h, unsigned long ks, unsigned long ke, const &char name) {
    unsafe {
        unsigned long i = 0UL;
        while (ks + i < ke) {
            if (name[i] == (char)0 || http_lower_(h[ks + i]) != http_lower_(name[i])) { return 0; }
            i = i + 1UL;
        }
        return name[i] == (char)0;
    }
}

// Single pass over the raw block, no per-line allocation.
struct String http_header_get(const &char raw_headers, const &char name) {
    unsigned long len = str_len(raw_headers);
    unsigned long pos = 0UL;
    unsigned long ks = 0UL;
    unsigned long ke = 0UL;
    unsigned long vs = 0UL;
    unsigned long ve = 0UL;
    while (1) {
        unsigned long next = http_next_header_(raw_headers, len, pos, &ks, &ke, &vs, &ve);
        if (ke > ks && http_key_is_(raw_headers, ks, ke, name)) {
            unsafe { return String::from_n(&raw_headers[vs], ve - vs); }
        }
        if (next == 0UL) { break; }
        pos = next;
    }
    return String::new();
}

static int http_header_present_(const &char raw, const &char name) {
    unsigned long len = str_len(raw);
    unsigned long pos = 0UL;
    unsigned long ks = 0UL;
    unsigned long ke = 0UL;
    unsigned long vs = 0UL;
    unsigned long ve = 0UL;
    while (1) {
        unsigned long next = http_next_header_(raw, len, pos, &ks, &ke, &vs, &ve);
        if (ke > ks && http_key_is_(raw, ks, ke, name)) { return 1; }
        if (next == 0UL) { break; }
        pos = next;
    }
    return 0;
}

// Is 'token' an item of the comma-separated list v[a..b)?
static int http_list_has_(const char* v, unsigned long a, unsigned long b, const char* token) {
    unsafe {
        unsigned long tl = 0UL;
        while (token[tl] != (char)0) { tl = tl + 1UL; }
        unsigned long i = a;
        while (i < b) {
            while (i < b && (v[i] == ' ' || v[i] == '\t' || v[i] == ',')) { i = i + 1UL; }
            unsigned long s = i;
            while (i < b && v[i] != ',') { i = i + 1UL; }
            unsigned long e = i;
            while (e > s && (v[e - 1UL] == ' ' || v[e - 1UL] == '\t')) { e = e - 1UL; }
            if (e - s == tl && tl > 0UL) {
                unsigned long k = 0UL;
                while (k < tl && http_lower_(v[s + k]) == http_lower_(token[k])) { k = k + 1UL; }
                if (k == tl) { return 1; }
            }
        }
    }
    return 0;
}

int http_header_has_token(const &char raw_headers, const &char name, const &char token) {
    unsigned long len = str_len(raw_headers);
    unsigned long pos = 0UL;
    unsigned long ks = 0UL;
    unsigned long ke = 0UL;
    unsigned long vs = 0UL;
    unsigned long ve = 0UL;
    while (1) {
        unsigned long next = http_next_header_(raw_headers, len, pos, &ks, &ke, &vs, &ve);
        if (ke > ks && http_key_is_(raw_headers, ks, ke, name)) {
            unsafe { if (http_list_has_(raw_headers, vs, ve, token)) { return 1; } }
        }
        if (next == 0UL) { break; }
        pos = next;
    }
    return 0;
}

// Rebuilds 'raw' without the lines named 'name'; if 'value' is non-null,
// one "name: value" line takes the first removed line's place (or goes at
// the end). Returns the number of lines removed.
static unsigned long http_headers_rewrite_(&String raw, const &char name, const ?&char value) {
    struct String out = String::with_cap(raw.length() + 32UL);
    const char* h = raw.as_ptr();
    unsigned long len = raw.length();
    unsigned long pos = 0UL;
    unsigned long ks = 0UL;
    unsigned long ke = 0UL;
    unsigned long vs = 0UL;
    unsigned long ve = 0UL;
    unsigned long removed = 0UL;
    int placed = 0;
    const char* vp = (const char*)0;
    match (value) { case some(v): { vp = v; } case null: { } }
    unsafe {
        while (len > 0UL) {
            unsigned long next = http_next_header_(h, len, pos, &ks, &ke, &vs, &ve);
            unsigned long lineEnd = next == 0UL ? len : next;
            unsigned long contentEnd = lineEnd;
            while (contentEnd > pos && (h[contentEnd - 1UL] == '\n' || h[contentEnd - 1UL] == '\r')) {
                contentEnd = contentEnd - 1UL;
            }
            if (ke > ks && http_key_is_(h, ks, ke, name)) {
                removed = removed + 1UL;
                if (!placed && vp != (const char*)0) {
                    if (!out.is_empty()) { out.push("\r\n"); }
                    out.push(name);
                    out.push(": ");
                    out.push(vp);
                    placed = 1;
                }
            } else if (contentEnd > pos) {
                if (!out.is_empty()) { out.push("\r\n"); }
                out.push_n(&h[pos], contentEnd - pos);
            }
            if (next == 0UL) { break; }
            pos = next;
        }
        if (!placed && vp != (const char*)0) {
            if (!out.is_empty()) { out.push("\r\n"); }
            out.push(name);
            out.push(": ");
            out.push(vp);
        }
    }
    std::replace(raw, out).drop();
    return removed;
}

void http_headers_set(&String raw, const &char name, const &char value) {
    http_headers_rewrite_(raw, name, value);
}

void http_headers_add(&String raw, const &char name, const &char value) {
    unsafe {
        while (raw.length() > 0UL && (raw.as_ptr()[raw.length() - 1UL] == '\n' ||
                                      raw.as_ptr()[raw.length() - 1UL] == '\r')) {
            raw.truncate(raw.length() - 1UL);
        }
    }
    if (!raw.is_empty()) { raw.push("\r\n"); }
    raw.push(name);
    raw.push(": ");
    raw.push(value);
}

unsigned long http_headers_remove(&String raw, const &char name) {
    if (!http_header_present_(raw.as_ptr(), name)) { return 0UL; }
    return http_headers_rewrite_(raw, name, null);
}

// ── Request / response values ─────────────────────────────────────────────

void HttpResponse::free() { self.drop(); }

void HttpResponse::drop() {
    self.headers.free();
    self.body.free();
}

void HttpRequest::free() { self.drop(); }

void HttpRequest::drop() {
    self.method.free();
    self.path.free();
    self.headers.free();
    self.body.free();
}

struct HttpResponse HttpResponse::new(int status) {
    struct HttpResponse r;
    r.status = status;
    r.headers = String::new();
    r.body = String::new();
    return r;
}

struct HttpResponse HttpResponse::bytes(int status, const &char content_type,
                                        const ?&unsigned char data, unsigned long len) {
    struct HttpResponse r = HttpResponse::new(status);
    http_headers_add(&r.headers, "Content-Type", content_type);
    if (len > 0UL) { unsafe { r.body.push_n((const char*)data, len); } }
    return r;
}

struct HttpResponse HttpResponse::text(int status, const &char body) {
    struct HttpResponse r = HttpResponse::new(status);
    http_headers_add(&r.headers, "Content-Type", "text/plain; charset=utf-8");
    r.body.push(body);
    return r;
}

struct HttpResponse HttpResponse::html(int status, const &char body) {
    struct HttpResponse r = HttpResponse::new(status);
    http_headers_add(&r.headers, "Content-Type", "text/html; charset=utf-8");
    r.body.push(body);
    return r;
}

struct HttpResponse HttpResponse::json(int status, const &char body) {
    struct HttpResponse r = HttpResponse::new(status);
    http_headers_add(&r.headers, "Content-Type", "application/json");
    r.body.push(body);
    return r;
}

static void http_html_escape_into_(&String out, const &char s) {
    unsafe {
        unsigned long i = 0UL;
        while (s[i] != (char)0) {
            char c = s[i];
            if (c == '&') { out.push("&amp;"); }
            else if (c == '<') { out.push("&lt;"); }
            else if (c == '>') { out.push("&gt;"); }
            else if (c == '"') { out.push("&quot;"); }
            else { out.push_char(c); }
            i = i + 1UL;
        }
    }
}

struct HttpResponse HttpResponse::redirect(int status, const &char location) {
    if (status != 301 && status != 302 && status != 303 && status != 307 && status != 308) { status = 302; }
    struct HttpResponse r = HttpResponse::new(status);
    http_headers_add(&r.headers, "Location", location);
    http_headers_add(&r.headers, "Content-Type", "text/html; charset=utf-8");
    r.body.push("<a href=\"");
    http_html_escape_into_(&r.body, location);
    r.body.push("\">");
    r.body.push(http_reason_phrase(status));
    r.body.push("</a>\n");
    return r;
}

struct HttpResponse HttpResponse::error(int status) {
    struct HttpResponse r = HttpResponse::new(status);
    http_headers_add(&r.headers, "Content-Type", "text/plain; charset=utf-8");
    r.body.push_int((long long)status);
    r.body.push_char(' ');
    r.body.push(http_reason_phrase(status));
    r.body.push_char('\n');
    return r;
}

void HttpResponse::set_header(const &char name, const &char value) { http_headers_set(&self.headers, name, value); }
void HttpResponse::add_header(const &char name, const &char value) { http_headers_add(&self.headers, name, value); }
unsigned long HttpResponse::remove_header(const &char name) { return http_headers_remove(&self.headers, name); }
struct String HttpResponse::header(const &char name) const { return http_header_get(self.headers.as_ptr(), name); }
int HttpResponse::has_header(const &char name) const { return http_header_present_(self.headers.as_ptr(), name); }
struct HttpHeaders HttpResponse::header_map() const { return HttpHeaders::parse(self.headers.as_ptr()); }

void HttpResponse::set_body(const &char body) {
    self.body.clear();
    self.body.push(body);
}

void HttpResponse::set_body_bytes(const &unsigned char data, unsigned long len) {
    self.body.clear();
    unsafe { self.body.push_n((const char*)data, len); }
}

struct HttpRequest HttpRequest::new(const &char method, const &char path) {
    struct HttpRequest r;
    r.method = String::from(method);
    r.path = String::from(path);
    r.headers = String::new();
    r.body = String::new();
    return r;
}

struct String HttpRequest::header(const &char name) const { return http_header_get(self.headers.as_ptr(), name); }
int HttpRequest::has_header(const &char name) const { return http_header_present_(self.headers.as_ptr(), name); }
struct HttpHeaders HttpRequest::header_map() const { return HttpHeaders::parse(self.headers.as_ptr()); }

struct String HttpRequest::path_only() const {
    long long q = self.path.find_char('?');
    long long h = self.path.find_char('#');
    unsigned long end = self.path.length();
    if (q >= 0LL) { end = (unsigned long)q; }
    if (h >= 0LL && (unsigned long)h < end) { end = (unsigned long)h; }
    return self.path.substr(0UL, end);
}

struct String HttpRequest::query_string() const {
    long long q = self.path.find_char('?');
    if (q < 0LL) { return String::new(); }
    long long h = self.path.find_char('#');
    unsigned long end = self.path.length();
    if (h > q) { end = (unsigned long)h; }
    return self.path.substr((unsigned long)q + 1UL, end);
}

struct QueryParams HttpRequest::query() const {
    struct String qs = self.query_string();
    struct QueryParams q = QueryParams::parse_n(qs.as_ptr(), qs.length());
    qs.free();
    return q;
}

struct QueryParams HttpRequest::form() const {
    return QueryParams::parse_n(self.body.as_ptr(), self.body.length());
}

int HttpRequest::content_type_is(const &char mime) const {
    struct String ct = http_header_get(self.headers.as_ptr(), "Content-Type");
    long long semi = ct.find_char(';');
    if (semi >= 0LL) { ct.truncate((unsigned long)semi); }
    struct String t = ct.trim();
    int eq = t.eq_cstr_ignore_case(mime);
    t.free();
    ct.free();
    return eq;
}

struct String HttpRequest::cookie(const &char name) const {
    struct String out = String::new();
    const char* h = self.headers.as_ptr();
    unsigned long len = self.headers.length();
    unsigned long pos = 0UL;
    unsigned long ks = 0UL;
    unsigned long ke = 0UL;
    unsigned long vs = 0UL;
    unsigned long ve = 0UL;
    unsigned long nl = str_len(name);
    unsafe {
        while (len > 0UL) {
            unsigned long next = http_next_header_(h, len, pos, &ks, &ke, &vs, &ve);
            if (ke > ks && http_key_is_(h, ks, ke, "cookie")) {
                unsigned long i = vs;
                while (i < ve) {
                    while (i < ve && (h[i] == ' ' || h[i] == ';')) { i = i + 1UL; }
                    unsigned long a = i;
                    while (i < ve && h[i] != ';') { i = i + 1UL; }
                    unsigned long b = i;
                    while (b > a && h[b - 1UL] == ' ') { b = b - 1UL; }
                    unsigned long eq = a;
                    while (eq < b && h[eq] != '=') { eq = eq + 1UL; }
                    unsigned long ne = eq;
                    while (ne > a && h[ne - 1UL] == ' ') { ne = ne - 1UL; }
                    if (eq < b && ne - a == nl) {
                        unsigned long k = 0UL;
                        while (k < nl && h[a + k] == name[k]) { k = k + 1UL; }
                        if (k == nl) {
                            unsigned long va = eq + 1UL;
                            while (va < b && h[va] == ' ') { va = va + 1UL; }
                            unsigned long vb = b;
                            if (vb - va >= 2UL && h[va] == '"' && h[vb - 1UL] == '"') { va = va + 1UL; vb = vb - 1UL; }
                            out.push_n(&h[va], vb - va);
                            return out;
                        }
                    }
                }
            }
            if (next == 0UL) { break; }
            pos = next;
        }
    }
    return out;
}

struct HttpRequest HttpRequest::clone() const {
    struct HttpRequest r;
    r.method = self.method.clone();
    r.path = self.path.clone();
    r.headers = self.headers.clone();
    r.body = self.body.clone();
    return r;
}

// ── Host resolution ────────────────────────────────────────────────────────

static int parse_dotted_quad_(const char* host, unsigned int* addr_out) {
    int parts[4];
    int partIdx = 0;
    int cur = -1;
    unsigned long i = 0UL;
    while (1) {
        char c;
        unsafe { c = host[i]; }
        if (c == (char)0 || c == '.') {
            if (cur < 0 || cur > 255) { return 0; }
            if (partIdx > 3) { return 0; }
            unsafe { parts[partIdx] = cur; }
            partIdx = partIdx + 1;
            cur = -1;
            if (c == (char)0) { break; }
            i = i + 1UL;
            continue;
        }
        if (c < '0' || c > '9') { return 0; }
        if (cur < 0) { cur = 0; }
        cur = cur * 10 + ((int)c - (int)'0');
        i = i + 1UL;
    }
    if (partIdx != 4) { return 0; }
    unsafe {
        *addr_out = sched_ipv4((unsigned char)parts[0], (unsigned char)parts[1],
                                (unsigned char)parts[2], (unsigned char)parts[3]);
    }
    return 1;
}

int http_resolve_host(const &char host, &unsigned int addr_out) {
    if (parse_dotted_quad_(host, addr_out)) { return 1; }

    // getaddrinfo, not gethostbyname: gethostbyname returns a pointer into
    // one static buffer shared by every thread, so two concurrent
    // http_request calls on different threads could read each other's
    // (or a half-written) result.
#ifdef _WIN32
    // Winsock refuses every call, getaddrinfo included, until WSAStartup
    // has run; the socket helpers only do that once they create a socket,
    // which here comes after resolution. Safe to repeat (reference-counted).
    unsigned char wsaData[512];
    unsafe { WSAStartup((unsigned short)0x0202, (void*)&wsaData[0]); }
#endif
    struct AddrInfo_ hints;
    hints.ai_flags = 0;
    hints.ai_family = SCHED_AF_INET;
    hints.ai_socktype = SCHED_SOCK_STREAM;
    hints.ai_protocol = 0;
    hints.ai_addrlen = 0UL;
    hints.ai_addr = (void*)0;
    hints.ai_canonname = (char*)0;
    hints.ai_next = (struct AddrInfo_*)0;
    void* resPtr = (void*)0;
    int rc;
    unsafe { rc = getaddrinfo(host, (const char*)0, (const void*)&hints, &resPtr); }
    if (rc != 0 || resPtr == (void*)0) { return 0; }
    void* sa;
    unsafe { sa = ((struct AddrInfo_*)resPtr)->ai_addr; }
    int found = 0;
    if (sa != (void*)0) {
        // sockaddr_in.sin_addr sits at byte offset 4 on every platform
        // (Linux/Windows: 2-byte family + port; BSD: len + family + port),
        // already in network byte order.
        unsigned int netAddr;
        unsafe { netAddr = *(unsigned int*)((unsigned char*)sa + 4); }
        unsafe { *addr_out = netAddr; }
        found = 1;
    }
    unsafe { freeaddrinfo(resPtr); }
    return found;
}

// ── TcpStream ─────────────────────────────────────────────────────────────

struct TcpStream TcpStream::from_fd(int fd) {
    struct TcpStream s;
    s.fd = fd;
    s.tls = null;
    return s;
}

struct TcpStream TcpStream::over_tls(&heap TlsStream t) {
    struct TcpStream s;
    s.fd = t.fd;
    s.tls = t;
    return s;
}

int TcpStream::is_tls() const { return !self.tls.is_null(); }

long long TcpStream::read(&unsigned char buf, unsigned long n) {
    match (self.tls) {
        case some(t): {
            long long r = t.read(buf, n);
            return r < 0LL ? -1LL : r;
        }
        case null: { }
    }
    unsafe { return (long long)sock_read_(self.fd, (void*)buf, n); }
}

long long TcpStream::write(const &unsigned char buf, unsigned long n) {
    match (self.tls) {
        case some(t): {
            long long r = t.write(buf, n);
            return r < 0LL ? -1LL : r;
        }
        case null: { }
    }
    unsafe { return (long long)sock_write_(self.fd, (const void*)buf, n); }
}

int TcpStream::close() {
    match (self.tls) {
        case some(t): {
            int r = t.close();
            std::dealloc(t);
            self.tls = null;
            self.fd = -1;
            return r;
        }
        case null: { }
    }
    unsafe { return sock_close_(self.fd); }
}

// The whole of 'text' to the stream; 1 on success.
static int write_text_(&TcpStream s, const &String text) {
    unsafe { return write_all(s, (const &unsigned char)text.as_ptr(), text.length()); }
}

static int write_cstr_(&TcpStream s, const &char text) {
    unsafe { return write_all(s, (const &unsigned char)text, str_len(text)); }
}

struct TcpStream http_open_stream(const &char host, unsigned short port, int tls,
                                  const ?&TlsConfig cfg, int connect_timeout_ms,
                                  int io_timeout_ms, ?&TlsError err) {
    struct TcpStream out = TcpStream::from_fd(-1);
    struct TlsError e = TlsError::none();
    unsigned int addr = 0U;
    int fd = -1;
    if (!http_resolve_host(host, &addr)) {
        e = TlsError::make(TLS_ERR_IO, "cannot resolve the host name");
    } else {
        // Waits for the handshake itself rather than relying on the first
        // blocking write to do it: Linux's write() does wait on a
        // still-connecting socket, but BSD and Winsock fail it with
        // ENOTCONN, and none of them bound the wait below the kernel's own
        // SYN-retry limit (~130s on Linux).
        unsafe { fd = tcp_connect_timeout(addr, port, connect_timeout_ms); }
        if (fd < 0) { e = TlsError::make(TLS_ERR_IO, "TCP connect failed"); }
    }
    if (fd >= 0) {
        int t = io_timeout_ms;
        if (tls && t <= 0) { t = connect_timeout_ms; }
        if (t > 0 && sock_set_io_timeout(fd, t) != 0) {
            e = TlsError::make(TLS_ERR_IO, "cannot set the socket timeout");
            unsafe { sock_close_(fd); }
            fd = -1;
        }
    }
    if (fd >= 0 && !tls) {
        out = TcpStream::from_fd(fd);
    } else if (fd >= 0) {
        struct TlsConfig defaults = TlsConfig::new();
        defaults.set_alpn("http/1.1");
        struct Result<TlsStream, TlsError> r;
        match (cfg) {
            case some(c): { r = TlsStream::connect(fd, host, c); }
            case null: { r = TlsStream::connect(fd, host, &defaults); }
        }
        defaults.free();
        if (!r.ok()) {
            e = r.error;
            unsafe { sock_close_(fd); }
        } else {
            ?&heap TlsStream h = std::alloc(sizeof(struct TlsStream));
            match (h) {
                case some(hp): {
                    *hp = r.value;
                    if (io_timeout_ms <= 0) { sock_set_io_timeout(fd, 0); }
                    out = TcpStream::over_tls(hp);
                }
                case null: {
                    r.value.close();
                    e = TlsError::make(TLS_ERR_NO_MEMORY, "out of memory");
                }
            }
        }
    }
    match (err) {
        case some(ep): { *ep = e; }
        case null: { }
    }
    return out;
}

// ── Status codes ──────────────────────────────────────────────────────────

const &char http_reason_phrase(int status) {
    if (status == 100) { return "Continue"; }
    if (status == 101) { return "Switching Protocols"; }
    if (status == 103) { return "Early Hints"; }
    if (status == 200) { return "OK"; }
    if (status == 201) { return "Created"; }
    if (status == 202) { return "Accepted"; }
    if (status == 203) { return "Non-Authoritative Information"; }
    if (status == 204) { return "No Content"; }
    if (status == 205) { return "Reset Content"; }
    if (status == 206) { return "Partial Content"; }
    if (status == 300) { return "Multiple Choices"; }
    if (status == 301) { return "Moved Permanently"; }
    if (status == 302) { return "Found"; }
    if (status == 303) { return "See Other"; }
    if (status == 304) { return "Not Modified"; }
    if (status == 307) { return "Temporary Redirect"; }
    if (status == 308) { return "Permanent Redirect"; }
    if (status == 400) { return "Bad Request"; }
    if (status == 401) { return "Unauthorized"; }
    if (status == 402) { return "Payment Required"; }
    if (status == 403) { return "Forbidden"; }
    if (status == 404) { return "Not Found"; }
    if (status == 405) { return "Method Not Allowed"; }
    if (status == 406) { return "Not Acceptable"; }
    if (status == 407) { return "Proxy Authentication Required"; }
    if (status == 408) { return "Request Timeout"; }
    if (status == 409) { return "Conflict"; }
    if (status == 410) { return "Gone"; }
    if (status == 411) { return "Length Required"; }
    if (status == 412) { return "Precondition Failed"; }
    if (status == 413) { return "Content Too Large"; }
    if (status == 414) { return "URI Too Long"; }
    if (status == 415) { return "Unsupported Media Type"; }
    if (status == 416) { return "Range Not Satisfiable"; }
    if (status == 417) { return "Expectation Failed"; }
    if (status == 418) { return "I'm a teapot"; }
    if (status == 421) { return "Misdirected Request"; }
    if (status == 422) { return "Unprocessable Content"; }
    if (status == 425) { return "Too Early"; }
    if (status == 426) { return "Upgrade Required"; }
    if (status == 428) { return "Precondition Required"; }
    if (status == 429) { return "Too Many Requests"; }
    if (status == 431) { return "Request Header Fields Too Large"; }
    if (status == 451) { return "Unavailable For Legal Reasons"; }
    if (status == 500) { return "Internal Server Error"; }
    if (status == 501) { return "Not Implemented"; }
    if (status == 502) { return "Bad Gateway"; }
    if (status == 503) { return "Service Unavailable"; }
    if (status == 504) { return "Gateway Timeout"; }
    if (status == 505) { return "HTTP Version Not Supported"; }
    return "Unknown";
}

int http_status_is_informational(int status) { return status >= 100 && status < 200; }
int http_status_is_success(int status) { return status >= 200 && status < 300; }
int http_status_is_redirect(int status) {
    return status == 301 || status == 302 || status == 303 || status == 307 || status == 308;
}
int http_status_is_client_error(int status) { return status >= 400 && status < 500; }
int http_status_is_server_error(int status) { return status >= 500 && status < 600; }

static int http_status_has_body_(int status) {
    return !(status < 200 || status == 204 || status == 304);
}

struct String http_build_response(int status, const &char content_type,
                                   const ?&unsigned char body, unsigned long body_len) {
    const char* ctype;
    if (content_type) { ctype = content_type; } else { ctype = "text/plain"; }
    struct String out = String::new();
    unsafe {
        out.push("HTTP/1.1 ");
        out.push_int((long long)status);
        out.push_char(' ');
        out.push(http_reason_phrase(status));
        out.push("\r\nContent-Type: ");
        out.push(ctype);
        out.push("\r\nContent-Length: ");
        out.push_uint((unsigned long long)body_len);
        out.push("\r\nConnection: close\r\n\r\n");
        if (body_len > 0UL) { out.push_n((const char*)body, body_len); }
    }
    return out;
}

// Serializes a handler's HttpResponse: status line, every header the
// handler set except the framing ones this server owns (Content-Length,
// Transfer-Encoding, Connection), a Date and a default Content-Type if it
// set none, then the framing. 'head' omits the body (the Content-Length
// still describes it); 'keep' picks the Connection header.
static struct String http_render_response_(&HttpResponse resp, int head, int keep) {
    struct String out = String::with_cap(200UL + resp.headers.length() + resp.body.length());
    unsafe {
        out.push("HTTP/1.1 ");
        out.push_int((long long)resp.status);
        out.push_char(' ');
        out.push(http_reason_phrase(resp.status));
        out.push("\r\n");
        const char* h = resp.headers.as_ptr();
        unsigned long len = resp.headers.length();
        unsigned long pos = 0UL;
        unsigned long ks = 0UL;
        unsigned long ke = 0UL;
        unsigned long vs = 0UL;
        unsigned long ve = 0UL;
        int hasType = 0;
        int hasDate = 0;
        long long headLen = -1LL;
        while (len > 0UL) {
            unsigned long next = http_next_header_(h, len, pos, &ks, &ke, &vs, &ve);
            if (head && ke > ks && http_key_is_(h, ks, ke, "content-length")) {
                long long v = 0LL;
                unsigned long k = vs;
                while (k < ve && h[k] >= '0' && h[k] <= '9') { v = v * 10LL + (long long)((int)h[k] - (int)'0'); k = k + 1UL; }
                if (k == ve && ve > vs) { headLen = v; }
            }
            if (ke > ks && !http_key_is_(h, ks, ke, "content-length") &&
                !http_key_is_(h, ks, ke, "connection") &&
                !http_key_is_(h, ks, ke, "transfer-encoding")) {
                if (http_key_is_(h, ks, ke, "content-type")) { hasType = 1; }
                if (http_key_is_(h, ks, ke, "date")) { hasDate = 1; }
                out.push_n(&h[ks], ke - ks);
                out.push(": ");
                out.push_n(&h[vs], ve - vs);
                out.push("\r\n");
            }
            if (next == 0UL) { break; }
            pos = next;
        }
        if (!hasDate) {
            struct String d = http_date_now();
            out.push("Date: ");
            out.push_str(&d);
            out.push("\r\n");
            d.free();
        }
        int bodyAllowed = http_status_has_body_(resp.status);
        if (!hasType && bodyAllowed) { out.push("Content-Type: text/plain\r\n"); }
        if (bodyAllowed) {
            // HEAD: a handler that skipped building the body may state its
            // length with an explicit Content-Length.
            unsigned long long clen = (unsigned long long)resp.body.length();
            if (head && resp.body.length() == 0UL && headLen >= 0LL) { clen = (unsigned long long)headLen; }
            out.push("Content-Length: ");
            out.push_uint(clen);
            out.push("\r\n");
        }
        out.push(keep ? "Connection: keep-alive\r\n\r\n" : "Connection: close\r\n\r\n");
        if (!head && bodyAllowed) { out.push_n(resp.body.as_ptr(), resp.body.length()); }
    }
    return out;
}

// ── Chunked coding ────────────────────────────────────────────────────────

static char http_hex_digit_(unsigned int v) {
    if (v < 10U) { return (char)((int)'0' + (int)v); }
    return (char)((int)'a' + (int)(v - 10U));
}

static void http_push_hex_(&String out, unsigned long v) {
    char tmp[20];
    int n = 0;
    if (v == 0UL) { out.push_char('0'); return; }
    while (v > 0UL && n < 20) {
        unsafe { tmp[n] = http_hex_digit_((unsigned int)(v & 15UL)); }
        v = v >> 4;
        n = n + 1;
    }
    while (n > 0) {
        n = n - 1;
        unsafe { out.push_char(tmp[n]); }
    }
}

void http_chunked_encode(&String out, const ?&unsigned char data, unsigned long len) {
    if (len == 0UL) { return; }
    http_push_hex_(out, len);
    out.push("\r\n");
    unsafe { out.push_n((const char*)data, len); }
    out.push("\r\n");
}

void http_chunked_end(&String out) { out.push("0\r\n\r\n"); }

static int http_hex_val_(char c) {
    if (c >= '0' && c <= '9') { return (int)c - (int)'0'; }
    if (c >= 'a' && c <= 'f') { return (int)c - (int)'a' + 10; }
    if (c >= 'A' && c <= 'F') { return (int)c - (int)'A' + 10; }
    return -1;
}

// Parses a chunk-size line's leading hex number (extensions after ';'
// ignored). -1 on no digits / overflow.
static long long http_parse_chunk_size_(const char* s, unsigned long n) {
    long long v = 0LL;
    unsigned long i = 0UL;
    unsafe {
        while (i < n && (s[i] == ' ' || s[i] == '\t')) { i = i + 1UL; }
        unsigned long start = i;
        while (i < n) {
            int d = http_hex_val_(s[i]);
            if (d < 0) { break; }
            if (v > 0x0FFFFFFFFFFFFFFLL) { return -1LL; }
            v = v * 16LL + (long long)d;
            i = i + 1UL;
        }
        if (i == start) { return -1LL; }
        while (i < n && (s[i] == ' ' || s[i] == '\t')) { i = i + 1UL; }
        if (i < n && s[i] != ';') { return -1LL; }
    }
    return v;
}

int http_chunked_decode(const &unsigned char data, unsigned long len, &String out,
                        ?&unsigned long consumed) {
    const char* s;
    unsafe { s = (const char*)data; }
    unsigned long pos = 0UL;
    while (1) {
        unsigned long eol = pos;
        unsafe { while (eol < len && s[eol] != '\n') { eol = eol + 1UL; } }
        if (eol >= len) { return 0; }
        unsigned long lineEnd = eol;
        unsafe { if (lineEnd > pos && s[lineEnd - 1UL] == '\r') { lineEnd = lineEnd - 1UL; } }
        long long size;
        unsafe { size = http_parse_chunk_size_(&s[pos], lineEnd - pos); }
        if (size < 0LL) { return 0; }
        pos = eol + 1UL;
        if (size == 0LL) {
            // Trailer section: lines until an empty one.
            while (1) {
                unsigned long e = pos;
                unsafe { while (e < len && s[e] != '\n') { e = e + 1UL; } }
                if (e >= len) { return 0; }
                unsigned long le = e;
                unsafe { if (le > pos && s[le - 1UL] == '\r') { le = le - 1UL; } }
                int empty = le == pos;
                pos = e + 1UL;
                if (empty) { break; }
            }
            if (consumed != (unsigned long*)0) { unsafe { *consumed = pos; } }
            return 1;
        }
        if ((unsigned long)size > len - pos) { return 0; }
        unsafe { out.push_n(&s[pos], (unsigned long)size); }
        pos = pos + (unsigned long)size;
        unsafe {
            if (pos + 1UL < len && s[pos] == '\r' && s[pos + 1UL] == '\n') { pos = pos + 2UL; }
            else if (pos < len && s[pos] == '\n') { pos = pos + 1UL; }
            else { return 0; }
        }
    }
    return 0;
}

// ── Reading heads and bodies from a connection ────────────────────────────

// Reads through the next "\r\n\r\n" (or "\n\n") into 'out', at most 'max'
// bytes. 1 on success; 0 if the stream ended or failed (*any set when at
// least one byte had arrived); -1 if the head exceeded 'max'.
static int http_read_head_(&BufReader<TcpStream> in, &String out, unsigned long max, &int any) {
    int state = 0;   // 0 text, 1 saw \n, 2 saw \n\r
    *any = 0;
    while (1) {
        if (in.pos == in.len) {
            long long k = in.inner.read(&in.buf[0], (unsigned long)IO_BUF_SIZE);
            if (k <= 0LL) { return 0; }
            in.pos = 0UL;
            in.len = (unsigned long)k;
        }
        unsigned long i = in.pos;
        while (i < in.len) {
            unsigned char c = in.buf[i];
            i = i + 1UL;
            // Skip blank lines before a request/status line (RFC 9112 §2.2).
            if (*any == 0 && out.is_empty() && (c == (unsigned char)'\r' || c == (unsigned char)'\n')) {
                in.pos = i;
                continue;
            }
            *any = 1;
            if (c == (unsigned char)'\n') {
                if (state == 1 || state == 2) {
                    unsafe { out.push_n((const &char)(const char*)&in.buf[in.pos], i - in.pos); }
                    in.pos = i;
                    return 1;
                }
                state = 1;
            } else if (c == (unsigned char)'\r') {
                if (state == 1) { state = 2; } else if (state != 2) { state = 0; }
            } else {
                state = 0;
            }
            if (out.length() + (i - in.pos) > max) { return -1; }
        }
        unsafe { out.push_n((const &char)(const char*)&in.buf[in.pos], in.len - in.pos); }
        in.pos = in.len;
    }
    return 0;
}

// Reads one line (through '\n') of at most 'max' bytes; the terminator is
// not kept. 1 ok, 0 eof/error/too long.
static int http_read_line_(&BufReader<TcpStream> in, &String out, unsigned long max) {
    while (1) {
        if (in.pos == in.len) {
            long long k = in.inner.read(&in.buf[0], (unsigned long)IO_BUF_SIZE);
            if (k <= 0LL) { return 0; }
            in.pos = 0UL;
            in.len = (unsigned long)k;
        }
        unsigned long i = in.pos;
        while (i < in.len) {
            if (in.buf[i] == (unsigned char)'\n') {
                unsafe { out.push_n((const &char)(const char*)&in.buf[in.pos], i - in.pos); }
                in.pos = i + 1UL;
                if (out.length() > 0UL) {
                    unsafe { if (out.as_ptr()[out.length() - 1UL] == '\r') { out.truncate(out.length() - 1UL); } }
                }
                return 1;
            }
            i = i + 1UL;
            if (out.length() + (i - in.pos) > max) { return 0; }
        }
        unsafe { out.push_n((const &char)(const char*)&in.buf[in.pos], in.len - in.pos); }
        in.pos = in.len;
    }
    return 0;
}

// Appends exactly 'want' bytes. 1 ok, 0 if the stream ended first.
static int http_read_n_(&BufReader<TcpStream> in, &String out, unsigned long want) {
    unsigned long remaining = want;
    unsigned char chunk[4096];
    if (!out.reserve(want)) { return 0; }
    while (remaining > 0UL) {
        unsigned long ask = (remaining < 4096UL) ? remaining : 4096UL;
        long long n = in.read(&chunk[0], ask);
        if (n <= 0LL) { return 0; }
        unsafe { out.push_n((const char*)&chunk[0], (unsigned long)n); }
        remaining = remaining - (unsigned long)n;
    }
    return 1;
}

// Appends everything until the peer closes, failing past 'max' (0 = none).
static int http_read_to_eof_(&BufReader<TcpStream> in, &String out, unsigned long max) {
    unsigned char chunk[4096];
    while (1) {
        long long n = in.read(&chunk[0], 4096UL);
        if (n < 0LL) { return 0; }
        if (n == 0LL) { return 1; }
        if (max != 0UL && out.length() + (unsigned long)n > max) { return 0; }
        unsafe { out.push_n((const char*)&chunk[0], (unsigned long)n); }
    }
    return 0;
}

// Decodes a chunked body straight from the stream. Returns 1 ok, 0 on a
// malformed/truncated body, -1 when it exceeds 'max' (0 = none). Trailer
// fields are read and dropped.
static int http_read_chunked_(&BufReader<TcpStream> in, &String out, unsigned long max) {
    while (1) {
        struct String line = String::new();
        if (!http_read_line_(in, &line, 8192UL)) { line.free(); return 0; }
        long long size = http_parse_chunk_size_(line.as_ptr(), line.length());
        line.free();
        if (size < 0LL) { return 0; }
        if (size == 0LL) {
            while (1) {
                struct String t = String::new();
                if (!http_read_line_(in, &t, 8192UL)) { t.free(); return 0; }
                int empty = t.is_empty();
                t.free();
                if (empty) { return 1; }
            }
        }
        if (max != 0UL && out.length() + (unsigned long)size > max) { return -1; }
        if (!http_read_n_(in, out, (unsigned long)size)) { return 0; }
        struct String crlf = String::new();
        int ok = http_read_line_(in, &crlf, 2UL) && crlf.is_empty();
        crlf.free();
        if (!ok) { return 0; }
    }
    return 0;
}

// Content-Length of a header block: -1 if absent, -2 if malformed or
// conflicting.
static long long http_content_length_(const &char raw) {
    unsigned long len = str_len(raw);
    unsigned long pos = 0UL;
    unsigned long ks = 0UL;
    unsigned long ke = 0UL;
    unsigned long vs = 0UL;
    unsigned long ve = 0UL;
    long long found = -1LL;
    unsafe {
        while (1) {
            unsigned long next = http_next_header_(raw, len, pos, &ks, &ke, &vs, &ve);
            if (ke > ks && http_key_is_(raw, ks, ke, "content-length")) {
                long long v = 0LL;
                unsigned long i = vs;
                if (i == ve) { return -2LL; }
                while (i < ve) {
                    char c = raw[i];
                    if (c == ',' || c == ' ') { i = i + 1UL; continue; }   // "5, 5" (repeated value)
                    if (c < '0' || c > '9') { return -2LL; }
                    if (v > 100000000000000LL) { return -2LL; }
                    v = v * 10LL + (long long)((int)c - (int)'0');
                    i = i + 1UL;
                    if (i < ve && raw[i] == ',') {
                        if (found >= 0LL && found != v) { return -2LL; }
                        found = v;
                        v = 0LL;
                    }
                }
                if (found >= 0LL && found != v) { return -2LL; }
                found = v;
            }
            if (next == 0UL) { break; }
            pos = next;
        }
    }
    return found;
}

static int http_is_chunked_(const &char raw) {
    return http_header_has_token(raw, "Transfer-Encoding", "chunked");
}

// ── Client ────────────────────────────────────────────────────────────────

// One request/response exchange on a fresh connection.
static struct HttpResponse http_exchange_(const &char host, unsigned short port, const &char hostHeader,
                                          const &char method, const &char target,
                                          const ?&char extra_headers,
                                          const ?&unsigned char body, unsigned long body_len,
                                          int connect_timeout_ms, int io_timeout_ms,
                                          unsigned long max_body, int tls, const ?&TlsConfig tlscfg,
                                          ?&TlsError tlserr, ?&int ok) {
    struct HttpResponse resp = HttpResponse::new(0);
    int good = 1;

    struct TcpStream s = http_open_stream(host, port, tls, tlscfg, connect_timeout_ms, io_timeout_ms, tlserr);
    int fd = s.fd;
    if (fd < 0) { good = 0; }

    int isHead = 0;
    unsafe { isHead = str_eq(method, "HEAD"); }
    if (good) {
        struct String req = String::new();
        int userFraming = 0;
        int userHost = 0;
        match (extra_headers) {
            case some(h): {
                userFraming = http_header_present_(h, "Transfer-Encoding") || http_header_present_(h, "Content-Length");
                userHost = http_header_present_(h, "Host");
            }
            case null: { }
        }
        unsafe {
            req.push(method);
            req.push_char(' ');
            req.push(target);
            req.push(" HTTP/1.1\r\n");
            if (!userHost) {
                req.push("Host: ");
                req.push(hostHeader);
                req.push("\r\n");
            }
            req.push("Connection: close\r\n");
            match (extra_headers) {
                case some(h): {
                    req.push(h);
                    if (req.length() > 0UL && req.as_ptr()[req.length() - 1UL] != '\n') { req.push("\r\n"); }
                }
                case null: { }
            }
            if (!userFraming && (body_len > 0UL || str_eq(method, "POST") || str_eq(method, "PUT") ||
                                 str_eq(method, "PATCH"))) {
                req.push("Content-Length: ");
                req.push_uint((unsigned long long)body_len);
                req.push("\r\n");
            }
            req.push("\r\n");
            if (body_len > 0UL) { req.push_n((const char*)body, body_len); }
        }
        if (!write_text_(&s, &req)) { good = 0; }
        req.free();
    }

    if (good) {
        struct BufReader<TcpStream> in = BufReader::over(&s);
        // Interim 1xx responses (100 Continue, 103 Early Hints) precede
        // the real one.
        while (good) {
            struct String headerBlock = String::new();
            int any = 0;
            if (http_read_head_(&in, &headerBlock, 1048576UL, &any) != 1) {
                good = 0;
                headerBlock.free();
                break;
            }
            long firstLineEnd = headerBlock.index_of("\n");
            struct String statusLine = (firstLineEnd >= 0L)
                ? headerBlock.substr(0UL, (unsigned long)firstLineEnd)
                : headerBlock.clone();
            struct String restHeaders = (firstLineEnd >= 0L)
                ? headerBlock.substr((unsigned long)firstLineEnd + 1UL, headerBlock.length())
                : String::new();
            headerBlock.free();
            // Trim the blank line(s) closing the block.
            unsafe {
                while (restHeaders.length() > 0UL &&
                       (restHeaders.as_ptr()[restHeaders.length() - 1UL] == '\n' ||
                        restHeaders.as_ptr()[restHeaders.length() - 1UL] == '\r')) {
                    restHeaders.truncate(restHeaders.length() - 1UL);
                }
            }
            int statusCode = 0;
            if (statusLine.starts_with("HTTP/") && statusLine.length() >= 12UL) {
                int pok = 0;
                struct String code = statusLine.substr(9UL, 12UL);
                statusCode = (int)code.parse_int(&pok);
                code.free();
                if (!pok) { statusCode = 0; }
            }
            statusLine.free();
            if (statusCode < 100 || statusCode > 999) { restHeaders.free(); good = 0; break; }
            if (statusCode < 200 && statusCode != 101) { restHeaders.free(); continue; }
            resp.status = statusCode;
            resp.headers.free();
            resp.headers = restHeaders;
            break;
        }
        if (good && !isHead && http_status_has_body_(resp.status)) {
            if (http_is_chunked_(resp.headers.as_ptr())) {
                if (http_read_chunked_(&in, &resp.body, max_body) != 1) { good = 0; }
            } else {
                long long cl = http_content_length_(resp.headers.as_ptr());
                if (cl == -2LL) {
                    good = 0;
                } else if (cl >= 0LL) {
                    if (max_body != 0UL && (unsigned long)cl > max_body) { good = 0; }
                    else if (!http_read_n_(&in, &resp.body, (unsigned long)cl)) { good = 0; }
                } else {
                    // No framing: the body runs until the server closes.
                    if (!http_read_to_eof_(&in, &resp.body, max_body)) { good = 0; }
                }
            }
        }
    }

    if (fd >= 0) { s.close(); }
    if (ok != (int*)0) { unsafe { *ok = good; } }
    return resp;
}

struct HttpResponse http_request(const &char host, unsigned short port,
                                  const &char method, const &char path,
                                  const ?&char extra_headers,
                                  const ?&unsigned char body, unsigned long body_len,
                                  ?&int ok) {
    return http_request_timeout(host, port, method, path, extra_headers, body, body_len,
                                HTTP_DEFAULT_CONNECT_TIMEOUT_MS, 0, ok);
}

struct HttpResponse http_request_timeout(const &char host, unsigned short port,
                                          const &char method, const &char path,
                                          const ?&char extra_headers,
                                          const ?&unsigned char body, unsigned long body_len,
                                          int connect_timeout_ms, int io_timeout_ms,
                                          ?&int ok) {
    struct String hostHeader = String::from(host);
    if (port != (unsigned short)80) {
        hostHeader.push_char(':');
        hostHeader.push_uint((unsigned long long)port);
    }
    struct HttpResponse r = http_exchange_(host, port, hostHeader.as_ptr(), method, path, extra_headers,
                                           body, body_len, connect_timeout_ms, io_timeout_ms, 0UL, 0, null, null, ok);
    hostHeader.free();
    return r;
}

struct HttpResponse http_get(const &char host, unsigned short port,
                              const &char path, ?&int ok) {
    return http_request(host, port, "GET", path, null,
                         null, 0UL, ok);
}

struct HttpResponse http_post(const &char host, unsigned short port, const &char path,
                               const &char content_type,
                               const ?&unsigned char body, unsigned long body_len,
                               ?&int ok) {
    const char* ctype;
    if (content_type) { ctype = content_type; } else { ctype = "application/octet-stream"; }
    struct String hdr = String::new();
    unsafe {
        hdr.push("Content-Type: ");
        hdr.push(ctype);
        hdr.push("\r\n");
    }
    struct HttpResponse r = http_request(host, port, "POST", path, hdr.as_ptr(),
                                          body, body_len, ok);
    hdr.free();
    return r;
}

struct HttpClientOptions HttpClientOptions::defaults() {
    struct HttpClientOptions o;
    o.max_redirects = 5;
    o.connect_timeout_ms = HTTP_DEFAULT_CONNECT_TIMEOUT_MS;
    o.io_timeout_ms = 30000;
    o.decompress = 1;
    o.max_body_bytes = 64UL * 1024UL * 1024UL;
    o.tls = null;
    o.tls_error = null;
    return o;
}

int http_decode_body(&HttpResponse resp, unsigned long max_out) {
    struct String enc = http_header_get(resp.headers.as_ptr(), "Content-Encoding");
    struct String e = enc.trim();
    enc.free();
    int result = 1;
    if (e.is_empty() || e.eq_cstr_ignore_case("identity")) {
        e.free();
        return 1;
    }
    struct Vec<unsigned char> out = Vec::new();
    int ok = 0;
    unsafe {
        const &unsigned char data = (const &unsigned char)resp.body.as_ptr();
        if (e.eq_cstr_ignore_case("gzip") || e.eq_cstr_ignore_case("x-gzip")) {
            out = gzip_decompress(data, resp.body.length(), max_out, &ok);
        } else if (e.eq_cstr_ignore_case("deflate")) {
            // "deflate" is meant to be zlib-wrapped, but some servers send
            // raw DEFLATE; accept both.
            out = zlib_decompress(data, resp.body.length(), max_out, &ok);
            if (!ok) {
                out.free();
                out = inflate_raw(data, resp.body.length(), max_out, null, &ok);
            }
        }
    }
    if (ok) {
        resp.body.clear();
        match (out.data) {
            case some(d): { unsafe { resp.body.push_n((const char*)d, out.len); } }
            case null: { }
        }
        http_headers_remove(&resp.headers, "Content-Encoding");
        http_headers_remove(&resp.headers, "Content-Length");
    } else {
        result = 0;
    }
    out.free();
    e.free();
    return result;
}

struct HttpResponse http_fetch(const &char method, const &char url,
                               const ?&char extra_headers,
                               const ?&unsigned char body, unsigned long body_len,
                               const ?&HttpClientOptions optsIn, ?&String final_url,
                               ?&int ok) {
    struct HttpClientOptions opts = HttpClientOptions::defaults();
    match (optsIn) { case some(o): { opts = *o; } case null: { } }
    int pok = 0;
    struct Url cur = Url::parse(url, &pok);
    struct String curMethod = String::from(method);
    int sendBody = 1;
    int redirects = 0;
    int good = pok;
    struct HttpResponse resp = HttpResponse::new(0);
    while (good) {
        int https = cur.scheme.eq_cstr("https");
        if ((!https && !cur.scheme.eq_cstr("http")) || cur.host.is_empty()) { good = 0; break; }
        int port = cur.port_or_default();
        if (port <= 0 || port > 65535) { good = 0; break; }
        struct String target = cur.request_target();
        struct String hostHeader = cur.host_port();
        struct String hdrs = String::new();
        int userUa = 0;
        int userAe = 0;
        match (extra_headers) {
            case some(h): {
                hdrs.push(h);
                userUa = http_header_present_(h, "User-Agent");
                userAe = http_header_present_(h, "Accept-Encoding");
            }
            case null: { }
        }
        if (!hdrs.is_empty() && !hdrs.ends_with("\n")) { hdrs.push("\r\n"); }
        if (!userUa) { hdrs.push("User-Agent: SafeC-std/http\r\n"); }
        if (opts.decompress && !userAe) { hdrs.push("Accept-Encoding: gzip, deflate\r\n"); }
        resp.free();
        int xok = 0;
        if (sendBody) {
            resp = http_exchange_(cur.host.as_ptr(), (unsigned short)port, hostHeader.as_ptr(),
                                  curMethod.as_ptr(), target.as_ptr(), hdrs.as_ptr(), body, body_len,
                                  opts.connect_timeout_ms, opts.io_timeout_ms, opts.max_body_bytes,
                                  https, opts.tls, opts.tls_error, &xok);
        } else {
            resp = http_exchange_(cur.host.as_ptr(), (unsigned short)port, hostHeader.as_ptr(),
                                  curMethod.as_ptr(), target.as_ptr(), hdrs.as_ptr(), null, 0UL,
                                  opts.connect_timeout_ms, opts.io_timeout_ms, opts.max_body_bytes,
                                  https, opts.tls, opts.tls_error, &xok);
        }
        target.free();
        hostHeader.free();
        hdrs.free();
        if (!xok) { good = 0; break; }
        if (http_status_is_redirect(resp.status) && opts.max_redirects > 0) {
            struct String loc = http_header_get(resp.headers.as_ptr(), "Location");
            if (loc.is_empty()) { loc.free(); break; }
            if (redirects >= opts.max_redirects) { loc.free(); good = 0; break; }
            int rok = 0;
            struct Url next = cur.resolve(loc.as_ptr(), &rok);
            loc.free();
            if (!rok) { next.free(); good = 0; break; }
            next.has_fragment = 0;
            cur.free();
            cur = next;
            if (resp.status == 301 || resp.status == 302 || resp.status == 303) {
                if (!curMethod.eq_cstr("HEAD")) {
                    curMethod.free();
                    curMethod = String::from("GET");
                }
                sendBody = 0;
            }
            redirects = redirects + 1;
            continue;
        }
        break;
    }
    if (good && opts.decompress) {
        if (!http_decode_body(&resp, opts.max_body_bytes)) { good = 0; }
    }
    match (final_url) {
        case some(fu): {
            std::replace(fu, cur.to_string()).drop();
        }
        case null: { }
    }
    cur.free();
    curMethod.free();
    if (ok != (int*)0) { unsafe { *ok = good; } }
    return resp;
}

struct HttpResponse http_get_url(const &char url, ?&int ok) {
    return http_fetch("GET", url, null, null, 0UL, null, null, ok);
}

struct HttpResponse http_post_url(const &char url, const &char content_type,
                                  const ?&unsigned char body, unsigned long body_len, ?&int ok) {
    struct String hdr = String::from("Content-Type: ");
    hdr.push(content_type);
    hdr.push("\r\n");
    struct HttpResponse r = http_fetch("POST", url, hdr.as_ptr(), body, body_len, null, null, ok);
    hdr.free();
    return r;
}

// ── Server: request parsing ───────────────────────────────────────────────

// Result of reading one request off a connection.
#define HTTP_READ_OK_       1
#define HTTP_READ_CLOSED_   0    // connection closed / timed out between requests
#define HTTP_READ_ERROR_   -1    // malformed or over a limit: answer *status, then close

struct HttpReqInfo_ {
    int versionMinor;     // HTTP/1.x
    int wantsClose;       // the client asked for Connection: close (or 1.0 without keep-alive)
    int status;           // error status for HTTP_READ_ERROR_
};

// Turns an absolute-form target ("http://host/p?q") into origin-form.
static void http_origin_form_(&String path) {
    if (!path.starts_with("http://") && !path.starts_with("https://")) { return; }
    int ok = 0;
    struct Url u = Url::parse(path.as_ptr(), &ok);
    if (ok) {
        struct String t = u.request_target();
        std::replace(path, t).drop();
    }
    u.free();
}

static int http_read_request_(&BufReader<TcpStream> in, const &HttpServerConfig cfg, &TcpStream out,
                              &HttpRequest req, &HttpReqInfo_ info) {
    info.versionMinor = 1;
    info.wantsClose = 0;
    info.status = 400;
    struct String head = String::new();
    int any = 0;
    int rc = http_read_head_(in, &head, cfg.max_header_bytes, &any);
    if (rc == -1) {
        head.free();
        info.status = 431;
        return HTTP_READ_ERROR_;
    }
    if (rc == 0) {
        head.free();
        if (any) { info.status = 400; return HTTP_READ_ERROR_; }
        return HTTP_READ_CLOSED_;
    }
    long firstLineEnd = head.index_of("\n");
    struct String requestLine = head.substr(0UL, firstLineEnd >= 0L ? (unsigned long)firstLineEnd : head.length());
    if (requestLine.ends_with("\r")) { requestLine.truncate(requestLine.length() - 1UL); }
    struct String rest = firstLineEnd >= 0L ? head.substr((unsigned long)firstLineEnd + 1UL, head.length()) : String::new();
    head.free();
    unsafe {
        while (rest.length() > 0UL && (rest.as_ptr()[rest.length() - 1UL] == '\n' ||
                                       rest.as_ptr()[rest.length() - 1UL] == '\r')) {
            rest.truncate(rest.length() - 1UL);
        }
    }
    // "METHOD SP request-target SP HTTP-version"
    struct String parts[4];
    unsigned long np = requestLine.split(" ", &parts[0], 4UL);
    int good = np == 3UL;
    if (good) {
        if (parts[2].eq_cstr("HTTP/1.1")) { info.versionMinor = 1; }
        else if (parts[2].eq_cstr("HTTP/1.0")) { info.versionMinor = 0; }
        else if (parts[2].starts_with("HTTP/")) { info.status = 505; good = 0; }
        else { good = 0; }
        if (parts[0].is_empty() || parts[1].is_empty()) { good = 0; }
    }
    if (good) {
        req.method.free();
        req.method = parts[0].clone();
        req.path.free();
        req.path = parts[1].clone();
        http_origin_form_(&req.path);
    }
    unsigned long pi = 0UL;
    while (pi < np) { unsafe { parts[pi].free(); } pi = pi + 1UL; }   // split filled the first np
    unsafe { forget(parts); }                                          // ... all freed above
    requestLine.free();
    req.headers.free();
    req.headers = rest;
    if (!good) { return HTTP_READ_ERROR_; }

    const &char h = req.headers.as_ptr();
    if (info.versionMinor == 0) {
        info.wantsClose = !http_header_has_token(h, "Connection", "keep-alive");
    } else {
        info.wantsClose = http_header_has_token(h, "Connection", "close");
    }
    if (info.versionMinor == 1 && !http_header_present_(h, "Host")) { info.status = 400; return HTTP_READ_ERROR_; }

    int chunked = 0;
    if (http_header_present_(h, "Transfer-Encoding")) {
        // Only "chunked" (last) is understood; anything else is 501, and
        // Transfer-Encoding with Content-Length is a smuggling hazard (400).
        struct String te = http_header_get(h, "Transfer-Encoding");
        struct String tl = te.to_lower();
        struct String tt = tl.trim();
        int onlyChunked = tt.eq_cstr("chunked");
        tt.free();
        tl.free();
        te.free();
        if (!onlyChunked) { info.status = 501; return HTTP_READ_ERROR_; }
        if (http_header_present_(h, "Content-Length")) { info.status = 400; return HTTP_READ_ERROR_; }
        chunked = 1;
    }
    long long cl = http_content_length_(h);
    if (cl == -2LL) { info.status = 400; return HTTP_READ_ERROR_; }
    if (!chunked && cl > 0LL && (unsigned long)cl > cfg.max_body_bytes) { info.status = 413; return HTTP_READ_ERROR_; }
    if ((chunked || cl > 0LL) && info.versionMinor == 1 && http_header_has_token(h, "Expect", "100-continue")) {
        if (!write_cstr_(out, "HTTP/1.1 100 Continue\r\n\r\n")) { return HTTP_READ_CLOSED_; }
    }
    if (chunked) {
        int r = http_read_chunked_(in, &req.body, cfg.max_body_bytes);
        if (r == -1) { info.status = 413; return HTTP_READ_ERROR_; }
        if (r != 1) { info.status = 400; return HTTP_READ_ERROR_; }
    } else if (cl > 0LL) {
        if (!http_read_n_(in, &req.body, (unsigned long)cl)) { return HTTP_READ_CLOSED_; }
    }
    return HTTP_READ_OK_;
}

// ── Server: connections ───────────────────────────────────────────────────

struct TcpStream HttpConn::detach(?&String early) {
    self.detached = 1;
    match (early) {
        case some(e): {
            unsafe {
                struct BufReader<TcpStream>* r = (struct BufReader<TcpStream>*)self.reader_;
                if (r != (struct BufReader<TcpStream>*)0 && r->pos < r->len) {
                    e.push_n((const char*)&r->buf[r->pos], r->len - r->pos);
                    r->pos = r->len;
                }
            }
        }
        case null: { }
    }
    struct TcpStream s = TcpStream::from_fd(self.fd);
    s.tls = self.tls;
    return s;
}

struct String HttpConn::peer_string() const {
    struct String out = String::new();
    unsigned int a = self.peer_addr;
    unsafe {
        unsigned char* b = (unsigned char*)&a;
        out.push_uint((unsigned long long)b[0]); out.push_char('.');
        out.push_uint((unsigned long long)b[1]); out.push_char('.');
        out.push_uint((unsigned long long)b[2]); out.push_char('.');
        out.push_uint((unsigned long long)b[3]);
    }
    out.push_char(':');
    out.push_uint((unsigned long long)self.peer_port);
    return out;
}

struct HttpServerConfig HttpServerConfig::defaults(unsigned short port) {
    struct HttpServerConfig c;
    c.port = port;
    c.threads = 8;
    c.keep_alive = 1;
    c.idle_timeout_ms = 5000;
    c.io_timeout_ms = 30000;
    c.max_requests_per_conn = 1000;
    c.max_header_bytes = 64UL * 1024UL;
    c.max_body_bytes = 16UL * 1024UL * 1024UL;
    c.tls = null;
    return c;
}

struct HttpServerShared_ {
    struct HttpServerConfig cfg;
    int listenFd;
    unsigned short port;
    HttpDispatchFn_ dispatch;
    void* ctx;
    int stopping;
    int running;           // worker threads alive
    int serving;           // serve() between start and return
    unsigned long long mtx;
    int* activeFds;        // per worker: the connection it is serving, or -1
    unsigned long long* tids;
    int has_tls;           // HTTPS: every connection starts with acceptor.accept()
    struct TlsAcceptor acceptor;
};

// Serves requests on one accepted connection until it closes, the client
// or the config asks for close, the idle timeout passes, or the server
// stops. Closes fd unless a handler detached it.
static void http_serve_conn_(struct HttpServerShared_* sh, int fd) {
    struct HttpServerConfig cfg;
    unsafe { cfg = sh->cfg; }
    struct HttpConn conn;
    conn.fd = fd;
    conn.peer_addr = 0U;
    conn.peer_port = (unsigned short)0;
    conn.request_index = 0UL;
    conn.keep_alive = 0;
    conn.detached = 0;
    conn.tls = null;
    sock_addr_of_(fd, 1, &conn.peer_addr, &conn.peer_port);
    if (cfg.io_timeout_ms > 0) { sock_set_io_timeout(fd, cfg.io_timeout_ms); }
    struct TcpStream s = TcpStream::from_fd(fd);
    int hasTls = 0;
    unsafe { hasTls = sh->has_tls; }
    if (hasTls) {
        // HTTPS: the TLS handshake first (bounded by the I/O timeout).
        if (cfg.io_timeout_ms <= 0) { sock_set_io_timeout(fd, cfg.idle_timeout_ms > 0 ? cfg.idle_timeout_ms : 30000); }
        struct Result<TlsStream, TlsError> tr;
        unsafe { tr = sh->acceptor.accept(fd); }
        if (!tr.ok()) {
            unsafe { sock_close_(fd); }
            return;
        }
        ?&heap TlsStream h = std::alloc(sizeof(struct TlsStream));
        match (h) {
            case some(hp): {
                *hp = tr.value;
                s = TcpStream::over_tls(hp);
                conn.tls = hp;
            }
            case null: {
                tr.value.close();
                return;
            }
        }
        if (cfg.io_timeout_ms <= 0) { sock_set_io_timeout(fd, 0); }
    }
    struct BufReader<TcpStream> in = BufReader::over(&s);
    unsafe { conn.reader_ = (void*)&in; }
    while (1) {
        if (conn.request_index > 0UL && in.buffered() == 0UL) {
            // Between requests: the idle limit applies.
            sock_set_io_timeout(fd, cfg.idle_timeout_ms);
        }
        struct HttpRequest req = HttpRequest::new("", "");
        struct HttpReqInfo_ info;
        int rc = http_read_request_(&in, &cfg, &s, &req, &info);
        if (conn.request_index > 0UL) { sock_set_io_timeout(fd, cfg.io_timeout_ms); }
        if (rc == HTTP_READ_CLOSED_) { req.free(); break; }
        if (rc == HTTP_READ_ERROR_) {
            struct HttpResponse er = HttpResponse::error(info.status);
            struct String t = http_render_response_(&er, 0, 0);
            write_text_(&s, &t);
            t.free();
            er.free();
            req.free();
            break;
        }
        int stopping;
        unsafe { stopping = sh->stopping; }
        int keep = cfg.keep_alive && !info.wantsClose && !stopping &&
                   (cfg.max_requests_per_conn <= 0 ||
                    conn.request_index + 1UL < (unsigned long)cfg.max_requests_per_conn);
        conn.keep_alive = keep;
        struct HttpResponse resp;
        unsafe { resp = sh->dispatch(sh->ctx, &req, &conn); }
        if (conn.detached) {
            resp.free();
            req.free();
            return;
        }
        keep = keep && conn.keep_alive;
        int head = req.method.eq_cstr("HEAD");
        struct String text = http_render_response_(&resp, head, keep);
        int wrote = write_text_(&s, &text);
        text.free();
        resp.free();
        req.free();
        if (!wrote || !keep) { break; }
        conn.request_index = conn.request_index + 1UL;
    }
    s.close();
}

struct HttpWorkerArg_ {
    struct HttpServerShared_* sh;
    int index;
};

static ?&HttpWorkerArg_ http_server_worker_(?&HttpWorkerArg_ argPtr) {
    struct HttpWorkerArg_* a;
    unsafe { a = (struct HttpWorkerArg_*)argPtr; }
    struct HttpServerShared_* sh;
    int idx;
    unsafe {
        sh = a->sh;
        idx = a->index;
        dealloc_raw_((void*)a);
    }
    while (1) {
        int stopping;
        unsafe { stopping = sh->stopping; }
        if (stopping) { break; }
        int clientFd;
        unsafe { clientFd = tcp_accept_nb(sh->listenFd); }
        unsafe { stopping = sh->stopping; }
        if (stopping) {
            if (clientFd >= 0) { unsafe { sock_close_(clientFd); } }
            break;
        }
        if (clientFd < 0) {
            thread_sleep_ms(1UL);
            continue;
        }
        unsafe { fd_set_blocking(clientFd); }
        unsafe {
            mutex_lock(&sh->mtx);
            sh->activeFds[idx] = clientFd;
            mutex_unlock(&sh->mtx);
        }
        http_serve_conn_(sh, clientFd);
        unsafe {
            mutex_lock(&sh->mtx);
            sh->activeFds[idx] = -1;
            mutex_unlock(&sh->mtx);
        }
    }
    unsafe {
        mutex_lock(&sh->mtx);
        sh->running = sh->running - 1;
        mutex_unlock(&sh->mtx);
    }
    return null;
}

struct HttpServer HttpServer::bind(const &HttpServerConfig cfg, ?&int ok) {
    struct HttpServer srv;
    srv.port = (unsigned short)0;
    unsafe { srv.shared_ = (void*)0; }
    int good = 1;
    if (cfg.threads < 1) { good = 0; }
    int fd = -1;
    if (good) {
        unsafe { fd = tcp_listen_nb(cfg.port); }
        if (fd < 0) { good = 0; }
    }
    if (good) {
        unsafe { fd_set_blocking(fd); }
        unsigned int a = 0U;
        unsigned short p = cfg.port;
        if (sock_addr_of_(fd, 0, &a, &p)) { srv.port = p; } else { srv.port = cfg.port; }
        struct HttpServerShared_* sh;
        unsafe {
            sh = (struct HttpServerShared_*)alloc_zeroed_raw_(sizeof(struct HttpServerShared_));
            if (sh == (struct HttpServerShared_*)0) {
                good = 0;
            } else {
                sh->cfg = *cfg;
                sh->listenFd = fd;
                sh->port = srv.port;
                sh->stopping = 0;
                sh->running = 0;
                sh->serving = 0;
                mutex_init(&sh->mtx);
                sh->activeFds = (int*)alloc_raw_(checked_mul_size((unsigned long)cfg.threads, sizeof(int)));
                sh->tids = (unsigned long long*)alloc_zeroed_raw_(
                    checked_mul_size((unsigned long)cfg.threads, sizeof(unsigned long long)));
                int i = 0;
                while (i < cfg.threads) { sh->activeFds[i] = -1; i = i + 1; }
                sh->has_tls = 0;
                srv.shared_ = (void*)sh;
                match (cfg.tls) {
                    case some(tc): {
                        struct Result<TlsAcceptor, TlsError> ar = TlsAcceptor::new(tc);
                        if (ar.ok()) {
                            sh->acceptor = ar.value;
                            sh->has_tls = 1;
                        } else {
                            good = 0;
                            dealloc_raw_((void*)sh->activeFds);
                            dealloc_raw_((void*)sh->tids);
                            dealloc_raw_((void*)sh);
                            srv.shared_ = (void*)0;
                        }
                    }
                    case null: { }
                }
            }
        }
        if (!good) { unsafe { sock_close_(fd); } }
    }
    if (ok != (int*)0) { unsafe { *ok = good; } }
    return srv;
}

static struct HttpResponse http_plain_dispatch_(void* ctx, &HttpRequest req, &HttpConn conn) {
    HttpHandler h;
    unsafe { h = (HttpHandler)ctx; }
    return h(req);
}

// Spawns the workers and waits for them; closes the listening socket at
// the end.
static int http_server_run_(struct HttpServerShared_* sh) {
    int n;
    unsafe { n = sh->cfg.threads; }
    int i = 0;
    int failed = 0;
    while (i < n) {
        struct HttpWorkerArg_* a;
        unsafe {
            a = (struct HttpWorkerArg_*)alloc_raw_(sizeof(struct HttpWorkerArg_));
            a->sh = sh;
            a->index = i;
            mutex_lock(&sh->mtx);
            sh->running = sh->running + 1;
            mutex_unlock(&sh->mtx);
            if (thread_create(&sh->tids[i], http_server_worker_, (?&HttpWorkerArg_)a) != 0) {
                mutex_lock(&sh->mtx);
                sh->running = sh->running - 1;
                sh->stopping = 1;
                mutex_unlock(&sh->mtx);
                dealloc_raw_((void*)a);
                failed = 1;
                break;
            }
        }
        i = i + 1;
    }
    int k = 0;
    while (k < i) {
        unsafe { thread_join(sh->tids[k]); }
        k = k + 1;
    }
    unsafe {
        mutex_lock(&sh->mtx);
        sock_close_(sh->listenFd);
        sh->listenFd = -1;
        sh->serving = 0;
        mutex_unlock(&sh->mtx);
    }
    return failed ? -1 : 0;
}

int HttpServer::serve_raw_(HttpDispatchFn_ dispatch, void* ctx) {
    struct HttpServerShared_* sh;
    unsafe { sh = (struct HttpServerShared_*)self.shared_; }
    if (sh == (struct HttpServerShared_*)0) { return -1; }
    unsafe {
        mutex_lock(&sh->mtx);
        if (sh->serving || sh->listenFd < 0) { mutex_unlock(&sh->mtx); return -1; }
        sh->serving = 1;
        sh->dispatch = dispatch;
        sh->ctx = ctx;
        mutex_unlock(&sh->mtx);
    }
    return http_server_run_(sh);
}

int HttpServer::serve(HttpHandler handler) {
    void* ctx;
    unsafe { ctx = (void*)handler; }
    return self.serve_raw_(http_plain_dispatch_, ctx);
}

static ?&HttpServerShared_ http_server_bg_(?&HttpServerShared_ arg) {
    struct HttpServerShared_* sh;
    unsafe { sh = (struct HttpServerShared_*)arg; }
    http_server_run_(sh);
    return null;
}

int HttpServer::start_raw_(HttpDispatchFn_ dispatch, void* ctx) {
    struct HttpServerShared_* sh;
    unsafe { sh = (struct HttpServerShared_*)self.shared_; }
    if (sh == (struct HttpServerShared_*)0) { return -1; }
    unsafe {
        mutex_lock(&sh->mtx);
        if (sh->serving || sh->listenFd < 0) { mutex_unlock(&sh->mtx); return -1; }
        sh->serving = 1;
        sh->dispatch = dispatch;
        sh->ctx = ctx;
        mutex_unlock(&sh->mtx);
        unsigned long long tid = 0ULL;
        if (thread_create(&tid, http_server_bg_, (?&HttpServerShared_)sh) != 0) {
            mutex_lock(&sh->mtx);
            sh->serving = 0;
            mutex_unlock(&sh->mtx);
            return -1;
        }
        thread_detach(tid);
    }
    return 0;
}

int HttpServer::start(HttpHandler handler) {
    void* ctx;
    unsafe { ctx = (void*)handler; }
    return self.start_raw_(http_plain_dispatch_, ctx);
}

int HttpServer::is_running() const {
    struct HttpServerShared_* sh;
    unsafe { sh = (struct HttpServerShared_*)self.shared_; }
    if (sh == (struct HttpServerShared_*)0) { return 0; }
    int r;
    unsafe { r = sh->serving && !sh->stopping; }
    return r;
}

void HttpServer::stop() {
    struct HttpServerShared_* sh;
    unsafe { sh = (struct HttpServerShared_*)self.shared_; }
    if (sh == (struct HttpServerShared_*)0) { return; }
    int n;
    int fromWorker = 0;
    unsafe {
        mutex_lock(&sh->mtx);
        if (sh->stopping) { mutex_unlock(&sh->mtx); return; }
        sh->stopping = 1;
        n = sh->cfg.threads;
        // Wake connections idling between requests: a read-side shutdown
        // makes their blocked recv return 0. A request already being
        // handled still gets its response written.
        int i = 0;
        unsigned long long me = thread_self();
        while (i < n) {
            if (sh->activeFds[i] >= 0) { sock_shutdown_(sh->activeFds[i], HTTP_SHUT_RD_); }
            if (sh->tids[i] == me) { fromWorker = 1; }
            i = i + 1;
        }
        mutex_unlock(&sh->mtx);
    }
    // Wake every worker blocked in accept() with a throwaway connection.
    unsigned int lo = sched_ipv4((unsigned char)127, (unsigned char)0, (unsigned char)0, (unsigned char)1);
    int k = 0;
    while (k < n) {
        int wfd;
        unsafe { wfd = tcp_connect_timeout(lo, sh->port, 500); }
        if (wfd >= 0) { unsafe { sock_close_(wfd); } }
        k = k + 1;
    }
    if (fromWorker) { return; }
    // Wait for serve() (or the start() thread) to wind down.
    int waited = 0;
    while (waited < 60000) {
        int serving;
        unsafe { serving = sh->serving; }
        if (!serving) { break; }
        thread_sleep_ms(2UL);
        waited = waited + 2;
    }
}

int http_serve_config(const &HttpServerConfig cfg, HttpHandler handler) {
    int ok = 0;
    struct HttpServer srv = HttpServer::bind(cfg, &ok);
    if (!ok) { return -1; }
    return srv.serve(handler);
}

// The original entry points: one request per connection.
static struct HttpServerConfig http_legacy_config_(unsigned short port, int threads) {
    struct HttpServerConfig c = HttpServerConfig::defaults(port);
    c.threads = threads;
    c.keep_alive = 0;
    c.io_timeout_ms = 0;
    c.max_body_bytes = 1024UL * 1024UL * 1024UL;
    c.max_header_bytes = 1024UL * 1024UL;
    return c;
}

int http_serve(unsigned short port, HttpHandler handler) {
    struct HttpServerConfig c = http_legacy_config_(port, 1);
    return http_serve_config(&c, handler);
}

int http_serve_threaded(unsigned short port, HttpHandler handler, int numThreads) {
    if (numThreads < 1) { return -1; }
    struct HttpServerConfig c = http_legacy_config_(port, numThreads);
    return http_serve_config(&c, handler);
}

// ── Streaming responses ───────────────────────────────────────────────────

struct HttpChunkedWriter HttpChunkedWriter::begin(&HttpConn conn, int status, const ?&char headers) {
    struct HttpChunkedWriter w;
    w.s = conn.detach(null);
    w.ok = 1;
    struct HttpResponse r = HttpResponse::new(status);
    match (headers) { case some(h): { r.headers.push(h); } case null: { } }
    http_headers_remove(&r.headers, "Content-Length");
    http_headers_remove(&r.headers, "Transfer-Encoding");
    struct String text = String::new();
    text.push("HTTP/1.1 ");
    text.push_int((long long)status);
    text.push_char(' ');
    text.push(http_reason_phrase(status));
    text.push("\r\n");
    if (!r.headers.is_empty()) {
        text.push_str(&r.headers);
        text.push("\r\n");
    }
    text.push("Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n");
    unsafe { w.ok = write_all(&w.s, (const &unsigned char)text.as_ptr(), text.length()); }
    text.free();
    r.free();
    return w;
}

int HttpChunkedWriter::write(const &unsigned char data, unsigned long len) {
    if (!self.ok) { return 0; }
    if (len == 0UL) { return 1; }
    struct String t = String::new();
    http_chunked_encode(&t, data, len);
    unsafe { self.ok = write_all(&self.s, (const &unsigned char)t.as_ptr(), t.length()); }
    t.free();
    return self.ok;
}

int HttpChunkedWriter::write_str(const &char text) {
    unsafe { return self.write((const &unsigned char)text, str_len(text)); }
}

int HttpChunkedWriter::finish() {
    if (self.ok) {
        unsafe { self.ok = write_all(&self.s, (const &unsigned char)"0\r\n\r\n", 5UL); }
    }
    self.s.close();
    return self.ok;
}

// ── Reactor-based server ─────────────────────────────────────────────────────
//
// http_serve/http_serve_threaded are a small, fixed-size pool of OS
// threads each blocked on one connection's accept/read/write at a time —
// simple, but the concurrency ceiling is exactly the thread count, and
// each blocked thread costs a full OS stack and a context switch to wake.
// http_serve_reactor instead runs 'numThreads' OS threads (typically one
// per core, not one per connection), each driving its own
// std::TaskScheduler/std::Reactor pair (see std/sched/reactor.h) so a
// single thread can hold hundreds of connections open at once without
// blocking on any one of them — the same non-blocking-I/O-plus-event-loop
// shape Node/nginx/Rust's async runtimes use, built on the cooperative
// task scheduler this codebase already had but nothing (including this
// module, until now) actually drove with real socket I/O.
//
// Caller-selected backend, same convention as io_nb.h and this file's own
// choice of io_nb_bsd.sc/io_nb_linux.sc/io_nb_win32.sc: before including
// http.sc, also include whichever of std/sched/reactor_kqueue.sc /
// reactor_epoll.sc / reactor_win32.sc matches your target.

static struct HttpRequest http_parse_request_from_buf_(struct String* buf, long headerEnd, int* ok) {
    struct HttpRequest req;
    req.method  = String::new();
    req.path    = String::new();
    req.headers = String::new();
    req.body    = String::new();
    int good = 1;

    unsafe {
        struct String headerBlock = buf->substr(0UL, (unsigned long)headerEnd);
        long firstLineEnd = headerBlock.index_of("\r\n");
        struct String requestLine = (firstLineEnd >= 0L)
            ? headerBlock.substr(0UL, (unsigned long)firstLineEnd)
            : headerBlock.clone();
        struct String restHeaders = (firstLineEnd >= 0L)
            ? headerBlock.substr((unsigned long)firstLineEnd + 2UL, headerBlock.length())
            : String::new();
        headerBlock.free();

        struct String parts[8];
        unsigned long np = requestLine.split(" ", parts, 8UL);
        if (np >= 2UL) {
            req.method.free();
            req.method = parts[0].clone();
            req.path.free();
            req.path = parts[1].clone();
        } else {
            good = 0;
        }
        unsigned long pi = 0UL;
        while (pi < np) { parts[pi].free(); pi = pi + 1UL; }
        requestLine.free();

        req.headers.free();
        req.headers = restHeaders;

        // By the time this is called, the task's read phases have already
        // accumulated the full body (Content-Length bytes past headerEnd) —
        // no I/O needed here, pure parsing. Bytes past Content-Length that
        // came in the same read aren't part of this body (the same cut the
        // blocking servers make).
        unsigned long bodyEnd = buf->length();
        struct String clStr = http_header_get(req.headers.as_ptr(), "Content-Length");
        if (!clStr.is_empty()) {
            int clok = 0;
            unsigned long cl = (unsigned long)clStr.parse_int(&clok);
            if (clok && (unsigned long)headerEnd + cl < bodyEnd) bodyEnd = (unsigned long)headerEnd + cl;
        }
        clStr.free();
        req.body.free();
        req.body = buf->substr((unsigned long)headerEnd, bodyEnd);

        if (ok != (int*)0) { *ok = good; }
    }
    return req;
}

#define HTTP_CONN_PHASE_READ_HEADERS_ 0
#define HTTP_CONN_PHASE_READ_BODY_    1
#define HTTP_CONN_PHASE_DISPATCH_     2
#define HTTP_CONN_PHASE_WRITE_        3

struct HttpConnState_ {
    int fd;
    HttpHandler handler;
    struct TaskScheduler* sched;
    struct Reactor* reactor;
    int phase;
    int registeredFilter; // 0 (none yet), SCHED_READ, or SCHED_WRITE — whichever
                           // this fd is currently reactor-registered for, so
                           // phase transitions know whether to remove()/add()
    struct String readBuf;
    long headerEnd;        // -1 until "\r\n\r\n" found
    unsigned long bodyWant; // Content-Length target; 0 if no body expected
    struct String respText;
    unsigned long sent;
};

static void http_conn_cleanup_(struct HttpConnState_* st) {
    unsafe {
        if (st->registeredFilter != 0) {
            st->reactor->remove(st->fd, st->registeredFilter);
        }
        sock_close_(st->fd);
        st->readBuf.free();
        st->respText.free();
        free(st);
    }
}

// Task function: int(?&HttpConnState_ arg, int resume_point) — see std/sync/task.h.
// Always yields '1' (task.sc only checks zero-vs-nonzero; the actual
// resume point lives in st->phase, not the return value) until the
// connection is fully handled, then returns 0 (TASK_DONE). Entirely
// pointer-driven (st is heap-allocated, shared across yields/resumes), so
// the whole body runs inside one unsafe block rather than one per access.
static int http_conn_task_(?&HttpConnState_ argPtr, int resume_point) {
    unsafe {
        struct HttpConnState_* st = (struct HttpConnState_*)argPtr;

        if (st->phase == HTTP_CONN_PHASE_READ_HEADERS_) {
            char chunk[4096];
            while (1) {
                long n = sock_read_(st->fd, (void*)chunk, 4096UL);
                if (n > 0L) {
                    st->readBuf.push_n(chunk, (unsigned long)n);
                    long idx = st->readBuf.index_of("\r\n\r\n");
                    if (idx >= 0L) {
                        st->headerEnd = idx + 4L;
                        st->phase = HTTP_CONN_PHASE_READ_BODY_;
                        break;
                    }
                    continue; // more may already be available; avoid a needless yield
                }
                if (n == 0L) { http_conn_cleanup_(st); return 0; } // peer closed early
                if (sock_would_block()) {
                    if (st->registeredFilter != SCHED_READ) {
                        st->reactor->add(st->fd, SCHED_READ);
                        st->registeredFilter = SCHED_READ;
                    }
                    st->sched->await_fd(st->fd, SCHED_READ);
                    return 1;
                }
                http_conn_cleanup_(st); return 0; // real error
            }
        }

        if (st->phase == HTTP_CONN_PHASE_READ_BODY_) {
            struct String headerBlockForCL = st->readBuf.substr(0UL, (unsigned long)st->headerEnd);
            struct String clStr = http_header_get(headerBlockForCL.as_ptr(), "Content-Length");
            headerBlockForCL.free();
            if (!clStr.is_empty()) {
                int clok = 0;
                st->bodyWant = (unsigned long)clStr.parse_int(&clok);
            }
            clStr.free();

            char chunk[4096];
            while (st->readBuf.length() - (unsigned long)st->headerEnd < st->bodyWant) {
                unsigned long haveBody = st->readBuf.length() - (unsigned long)st->headerEnd;
                unsigned long want = st->bodyWant - haveBody;
                unsigned long ask = want < 4096UL ? want : 4096UL;
                long n = sock_read_(st->fd, (void*)chunk, ask);
                if (n > 0L) {
                    st->readBuf.push_n(chunk, (unsigned long)n);
                    continue;
                }
                if (n == 0L) { http_conn_cleanup_(st); return 0; }
                if (sock_would_block()) {
                    if (st->registeredFilter != SCHED_READ) {
                        st->reactor->add(st->fd, SCHED_READ);
                        st->registeredFilter = SCHED_READ;
                    }
                    st->sched->await_fd(st->fd, SCHED_READ);
                    return 1;
                }
                http_conn_cleanup_(st); return 0;
            }
            st->phase = HTTP_CONN_PHASE_DISPATCH_;
        }

        if (st->phase == HTTP_CONN_PHASE_DISPATCH_) {
            int reqOk = 0;
            struct HttpRequest req = http_parse_request_from_buf_(&st->readBuf, st->headerEnd, &reqOk);
            if (reqOk) {
                struct HttpResponse resp = st->handler(&req);
                st->respText = http_render_response_(&resp, req.method.eq_cstr("HEAD"), 0);
                resp.free();
            } else {
                st->respText = http_build_response(400, "text/plain", (const unsigned char*)"", 0UL);
            }
            req.free();
            st->sent = 0UL;
            if (st->registeredFilter == SCHED_READ) {
                st->reactor->remove(st->fd, SCHED_READ);
                st->registeredFilter = 0;
            }
            st->phase = HTTP_CONN_PHASE_WRITE_;
        }

        // HTTP_CONN_PHASE_WRITE_
        while (st->sent < st->respText.length()) {
            long n = sock_write_(st->fd,
                (const void*)&st->respText.as_ptr()[st->sent],
                st->respText.length() - st->sent);
            if (n > 0L) {
                st->sent = st->sent + (unsigned long)n;
                continue;
            }
            if (sock_would_block()) {
                if (st->registeredFilter != SCHED_WRITE) {
                    st->reactor->add(st->fd, SCHED_WRITE);
                    st->registeredFilter = SCHED_WRITE;
                }
                st->sched->await_fd(st->fd, SCHED_WRITE);
                return 1;
            }
            http_conn_cleanup_(st); return 0; // write error
        }
        http_conn_cleanup_(st);
        return 0; // done
    }
}

// Spawns a connection-handling task for an already-accepted fd on the
// given scheduler/reactor — the one piece of setup shared by both ways a
// connection can reach a worker thread: the acceptor dispatching directly
// into its own scheduler (target == the acceptor's own thread), and a
// worker draining a freshly-received fd out of its dispatch queue (see
// http_worker_loop_ below). 'sched'/'reactor' must belong to the thread
// calling this — HttpConnState_ pins them for this connection's whole
// lifetime, and TaskScheduler/Reactor are not safe to touch from more
// than one thread.
static void http_spawn_conn_(struct TaskScheduler* sched, struct Reactor* reactor,
                              HttpHandler handler, int fd) {
    unsafe {
        struct HttpConnState_* st = (struct HttpConnState_*)malloc(sizeof(struct HttpConnState_));
        if (st == (struct HttpConnState_*)0) {
            sock_close_(fd);
            return;
        }
        st->fd = fd;
        st->handler = handler;
        st->sched = sched;
        st->reactor = reactor;
        st->phase = HTTP_CONN_PHASE_READ_HEADERS_;
        st->registeredFilter = 0;
        st->readBuf = String::new();
        st->headerEnd = -1L;
        st->bodyWant = 0UL;
        st->respText = String::new();
        st->sent = 0UL;
        int idx = sched->spawn(http_conn_task_, (?&HttpConnState_)st);
        if (idx < 0) {
            // Scheduler at capacity (TASK_MAX concurrent connections on
            // this thread already) — drop this one rather than leak the
            // fd or block whoever's calling this.
            sock_close_(fd);
            st->readBuf.free();
            st->respText.free();
            free(st);
        }
    }
}

// Bounded MPSC-ish handoff queue: the acceptor (always thread 0 — see
// http_serve_reactor) pushes newly-accepted fds bound for some *other*
// thread here; that thread drains it from inside its own event loop
// (http_worker_loop_). One of these per worker thread, so it's really
// single-producer/single-consumer per queue despite the acceptor writing
// into all of them — each queue's mutex only ever contends against that
// one worker's own drain, never another queue's.
#define HTTP_DISPATCH_CAP_ 256

struct HttpDispatchQ_ {
    int fds[HTTP_DISPATCH_CAP_];
    int head;
    int tail;
    int count;
    unsigned long long mtx;
};

struct HttpReactorShared_ {
    HttpHandler handler;
    int listenFd;
    int numThreads;
    struct HttpDispatchQ_* queues; // array[numThreads], one per worker
    struct TaskScheduler** scheds; // array[numThreads]; scheds[i] is null
                                     // until thread i has published its own
                                     // &sched (see http_reactor_worker_) --
                                     // the acceptor's load comparison
                                     // (http_accept_task_) skips any index
                                     // still null rather than dereference it.
};

struct HttpAcceptState_ {
    int listenFd;
    HttpHandler handler;
    struct TaskScheduler* sched;
    struct Reactor* reactor;
    int registered;
    struct HttpReactorShared_* shared;
};

// Picks which thread a freshly-accepted connection should go to: the one
// currently carrying the least work, not just "whoever's next in line".
// Round-robin looks fair but isn't — it assumes every connection costs
// the same, so a thread that's fallen behind (a slower client, a bigger
// request body, anything that keeps a connection alive longer) just
// keeps getting handed the same 1-in-N share anyway and falls further
// behind. Load here is each candidate thread's own in-flight task count
// (TaskScheduler::active_count() — connections it's actively serving)
// plus whatever's still sitting in its dispatch queue waiting to be
// picked up (shared->queues[i].count) — the second term matters because
// active_count() alone can't see work that's been routed to a thread but
// not yet drained into its scheduler. Both reads are plain, unlocked
// loads of another thread's data: correctness only needs a *reasonable*
// pick, not a perfectly current one, so a value that's a few microseconds
// stale is fine — it just means the odd connection goes to the
// second-least-loaded thread instead of the least-loaded one, not a bug.
// scheds[i] is null until thread i finishes its own setup (see
// http_reactor_worker_); skipped here rather than dereferenced so the
// acceptor never has to wait for every thread to be ready before it can
// start accepting.
static int http_pick_target_(struct HttpReactorShared_* shared) {
    unsafe {
        int best = 0;
        int bestLoad = -1;
        int i = 0;
        while (i < shared->numThreads) {
            struct TaskScheduler* s = shared->scheds[i];
            if (s != (struct TaskScheduler*)0) {
                int load = s->active_count() + shared->queues[i].count;
                if (bestLoad < 0 || load < bestLoad) {
                    bestLoad = load;
                    best = i;
                }
            }
            i = i + 1;
        }
        return best;
    }
}

// Task function for the one long-lived "accept new connections" task,
// which runs only on thread 0 (see http_reactor_worker_) — every reactor
// thread still gets its own TaskScheduler/Reactor for *serving*
// connections, but there is exactly one listening socket and exactly one
// task anywhere that ever calls accept() on it, so there is no
// thundering herd of threads racing each other for the same new
// connection. What this task does with each accepted fd is hand it to
// the least-loaded thread (see http_pick_target_): thread 0's own
// scheduler directly if that's the pick, otherwise the target thread's
// dispatch queue (see HttpDispatchQ_) for that thread to pick up itself
// — never touches another thread's TaskScheduler/Reactor directly.
// Never returns 0 (TASK_DONE) — it's meant to run for the
// lifetime of the server, the same way http_serve's/http_serve_threaded's
// own accept loops never return.
static int http_accept_task_(?&HttpAcceptState_ argPtr, int resume_point) {
    unsafe {
        struct HttpAcceptState_* ac = (struct HttpAcceptState_*)argPtr;

        while (1) {
            int clientFd = tcp_accept_nb(ac->listenFd);
            if (clientFd >= 0) {
                int target = http_pick_target_(ac->shared);
                if (target == 0) {
                    http_spawn_conn_(ac->sched, ac->reactor, ac->handler, clientFd);
                } else {
                    struct HttpDispatchQ_* q = ac->shared->queues + target;
                    std::mutex_lock(&q->mtx);
                    if (q->count < HTTP_DISPATCH_CAP_) {
                        q->fds[q->tail] = clientFd;
                        q->tail = (q->tail + 1) % HTTP_DISPATCH_CAP_;
                        q->count = q->count + 1;
                        std::mutex_unlock(&q->mtx);
                    } else {
                        std::mutex_unlock(&q->mtx);
                        // Target thread isn't draining fast enough to keep
                        // up — drop rather than block the acceptor or
                        // leak the fd, same policy as hitting the
                        // scheduler's own TASK_MAX capacity above.
                        sock_close_(clientFd);
                    }
                }
                continue; // more may be pending; drain before yielding
            }
            if (sock_would_block()) {
                if (ac->registered == 0) {
                    ac->reactor->add(ac->listenFd, SCHED_READ);
                    ac->registered = 1;
                }
                ac->sched->await_fd(ac->listenFd, SCHED_READ);
                return 1;
            }
            // Some other accept() failure — yield and retry rather than
            // spin or crash the acceptor over one bad call.
            ac->sched->await_fd(ac->listenFd, SCHED_READ);
            return 1;
        }
    }
}

// Every worker thread's main loop (including thread 0, which also runs
// the accept task above alongside this). Deliberately not std::reactor_
// run(): that helper returns once the scheduler has zero active tasks,
// which is exactly the normal resting state for a worker that's simply
// waiting for its next dispatched connection — this loop must keep
// going forever instead, the same way every other http_serve* entry
// point never returns. It also can't block indefinitely in reactor->poll
// the way reactor_run does when idle: an infinite wait would only wake
// for activity on fds *this* thread already owns, so it would never
// notice a new fd the acceptor just pushed into myQueue. Polling with a
// short bound instead (1ms) when there's nothing else to do bounds the
// worst-case latency between "acceptor dispatched a connection to an
// idle thread" and "that thread starts serving it" to ~1ms, without
// needing a cross-thread wakeup primitive (eventfd/self-pipe/etc. — real
// options, just more machinery than this needs). Measured why it has to
// be *this* short, not just "short": at 20ms it looked fine on paper but
// tanked measured throughput by ~10x the moment load was split across
// more than one thread (c=50 against N=2 dropped from ~35k req/s to
// ~4.4k) — this handler's whole request/response cycle is sub-millisecond,
// so any idle-to-busy gap anywhere near that scale is a huge relative
// tax paid on every such transition, and with per-thread traffic thinned
// by round-robin dispatch, idle-to-busy transitions are frequent, not
// rare. 1ms brought every thread count back in line with single-thread
// throughput (minus the expected small per-thread overhead — see
// http_serve_reactor's own comment). Once a thread has any connections
// of its own, its normal poll(sched, 0) path (below) stays fully
// responsive as before; the bound only ever applies while idle, and idle
// polling this fast costs a negligible, bounded amount of CPU (at most
// 1000 no-op wakeups/sec on a thread with literally nothing to do).
static void http_worker_loop_(struct TaskScheduler* sched, struct Reactor* reactor,
                               struct HttpDispatchQ_* myQueue, HttpHandler handler) {
    unsafe {
        while (1) {
            if (std::mutex_trylock(&myQueue->mtx) == 0) {
                while (myQueue->count > 0) {
                    int fd = myQueue->fds[myQueue->head];
                    myQueue->head = (myQueue->head + 1) % HTTP_DISPATCH_CAP_;
                    myQueue->count = myQueue->count - 1;
                    http_spawn_conn_(sched, reactor, handler, fd);
                }
                std::mutex_unlock(&myQueue->mtx);
            }
            // A busy myQueue->mtx (trylock failed) just means the
            // acceptor is mid-push right now — whatever it's adding will
            // still be there next iteration, no need to wait for it here.

            sched->tick();
            if (sched->has_ready() != 0) {
                reactor->poll(sched, 0LL);
            } else {
                reactor->poll(sched, 1LL);
            }
        }
    }
}

struct HttpReactorThreadCtx_ {
    struct HttpReactorShared_* shared;
    int selfIndex;
};

static ?&HttpReactorThreadCtx_ http_reactor_worker_(?&HttpReactorThreadCtx_ argPtr) {
    unsafe {
        struct HttpReactorThreadCtx_* ctx = (struct HttpReactorThreadCtx_*)argPtr;
        struct HttpReactorShared_* shared = ctx->shared;

        struct Reactor reactor = Reactor::new();
        if (reactor.init() != 0) { return null; }
        struct TaskScheduler sched = TaskScheduler::new();

        struct HttpDispatchQ_* myQueue = shared->queues + ctx->selfIndex;

        // Published for http_pick_target_'s load comparison (see
        // HttpReactorShared_) — must happen before this thread does
        // anything else, so the acceptor never sees stale garbage here,
        // only "not ready yet" (null, handled by http_pick_target_) or a
        // valid, live pointer.
        shared->scheds[ctx->selfIndex] = (struct TaskScheduler*)&sched;

        struct HttpAcceptState_* ac = (struct HttpAcceptState_*)0;
        if (ctx->selfIndex == 0) {
            ac = (struct HttpAcceptState_*)malloc(sizeof(struct HttpAcceptState_));
            ac->listenFd = shared->listenFd;
            ac->handler = shared->handler;
            ac->sched = (struct TaskScheduler*)&sched;
            ac->reactor = (struct Reactor*)&reactor;
            ac->registered = 0;
            ac->shared = shared;
            sched.spawn(http_accept_task_, (?&HttpAcceptState_)ac);
        }

        http_worker_loop_(&sched, &reactor, myQueue, shared->handler);

        reactor.close_();
        if (ac != (struct HttpAcceptState_*)0) { free(ac); }
        free(ctx);
        return null;
    }
}

// Same contract as http_serve()/http_serve_threaded(), but each of
// 'numThreads' OS threads runs a non-blocking reactor event loop instead
// of blocking one-connection-at-a-time — see the file-header comment
// above for why this scales further under concurrent load. Requires a
// reactor backend (reactor_kqueue.sc/reactor_epoll.sc/reactor_win32.sc)
// to already be included by the caller, same as io_nb's own backend
// selection. Returns -1 immediately on setup failure; does not return
// otherwise.
//
// One listening socket, shared by every platform (Windows included —
// there's no SO_REUSEPORT-style trick needed here at all). Only thread 0
// ever calls accept() on it; every other thread only ever serves
// connections thread 0 hands it through its HttpDispatchQ_ (see
// http_accept_task_/http_worker_loop_ above). That's what actually fixes
// the old design's thundering herd — every thread's epoll/kqueue/WSAPoll
// set racing accept() on one shared fd made throughput *worse* with more
// threads, not just flat — without needing a second listening socket per
// thread or a platform-specific fallback.
int http_serve_reactor(unsigned short port, HttpHandler handler, int numThreads) {
    if (numThreads < 1) { return -1; }
    int listenFd;
    unsafe { listenFd = tcp_listen_nb(port); }
    if (listenFd < 0) { return -1; }

    struct HttpReactorShared_* shared;
    unsafe { shared = (struct HttpReactorShared_*)malloc(sizeof(struct HttpReactorShared_)); }
    unsafe {
        shared->handler = handler;
        shared->listenFd = listenFd;
        shared->numThreads = numThreads;
        shared->queues = (struct HttpDispatchQ_*)malloc(
            std::checked_mul_size((unsigned long)numThreads, sizeof(struct HttpDispatchQ_)));
        shared->scheds = (struct TaskScheduler**)malloc(
            std::checked_mul_size((unsigned long)numThreads, sizeof(struct TaskScheduler*)));
    }
    int qi = 0;
    while (qi < numThreads) {
        unsafe {
            shared->queues[qi].head = 0;
            shared->queues[qi].tail = 0;
            shared->queues[qi].count = 0;
            std::mutex_init(&shared->queues[qi].mtx);
            // Null until the corresponding thread publishes its own
            // &sched (see http_reactor_worker_) — http_pick_target_
            // skips any index still null.
            shared->scheds[qi] = (struct TaskScheduler*)0;
        }
        qi = qi + 1;
    }

    unsigned long long tid = 0ULL;
    int i = 0;
    while (i < numThreads) {
        struct HttpReactorThreadCtx_* tctx;
        unsafe {
            tctx = (struct HttpReactorThreadCtx_*)malloc(sizeof(struct HttpReactorThreadCtx_));
            tctx->shared = shared;
            tctx->selfIndex = i;
            if (thread_create(&tid, http_reactor_worker_, (?&HttpReactorThreadCtx_)tctx) != 0) { return -1; }
        }
        i = i + 1;
    }
    unsafe { thread_join(tid); }
    return 0;
}

} // namespace std
