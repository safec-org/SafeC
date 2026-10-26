// SafeC Standard Library — Server-Sent Events (see sse.h).
#pragma once
#include <std/http/sse.h>
#include <std/http/http.h>
#include <std/encoding/url.h>
#include <std/collections/string.h>
#include <std/sched/io_nb.h>
#include <std/io_traits.h>
#include <std/str.h>

namespace std {

// ── Server ────────────────────────────────────────────────────────────────

static int sse_write_(&SseStream st, const &String text) {
    if (!st.ok) { return 0; }
    unsafe { st.ok = write_all(&st.s, (const &unsigned char)text.as_ptr(), text.length()); }
    return st.ok;
}

struct SseStream SseStream::start(&HttpConn conn, const ?&char extra_headers) {
    struct SseStream st;
    st.s = conn.detach(null);
    st.ok = 1;
    // Streams sit idle between events: no I/O timeout.
    sock_set_io_timeout(st.s.fd, 0);
    struct String head = String::from("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                                      "Cache-Control: no-cache\r\nConnection: close\r\n");
    match (extra_headers) {
        case some(h): {
            head.push(h);
            if (!head.ends_with("\n")) { head.push("\r\n"); }
        }
        case null: { }
    }
    head.push("\r\n");
    sse_write_(&st, &head);
    head.free();
    return st;
}

static void sse_push_field_lines_(&String out, const &char field, const &char text) {
    unsafe {
        unsigned long i = 0UL;
        while (1) {
            out.push(field);
            out.push(": ");
            while (text[i] != (char)0 && text[i] != '\n' && text[i] != '\r') { out.push_char(text[i]); i = i + 1UL; }
            out.push_char('\n');
            if (text[i] == (char)0) { break; }
            if (text[i] == '\r' && text[i + 1UL] == '\n') { i = i + 1UL; }
            i = i + 1UL;
        }
    }
}

int SseStream::send_event(const ?&char event, const &char data, const ?&char id) {
    struct String out = String::new();
    match (event) { case some(e): { out.push("event: "); out.push(e); out.push_char('\n'); } case null: { } }
    match (id) { case some(v): { out.push("id: "); out.push(v); out.push_char('\n'); } case null: { } }
    sse_push_field_lines_(&out, "data", data);
    out.push_char('\n');
    int r = sse_write_(self, &out);
    out.free();
    return r;
}

int SseStream::send(const &char data) { return self.send_event(null, data, null); }

int SseStream::comment(const &char text) {
    struct String out = String::new();
    sse_push_field_lines_(&out, "", text);
    out.push_char('\n');
    int r = sse_write_(self, &out);
    out.free();
    return r;
}

int SseStream::retry(int ms) {
    struct String out = String::from("retry: ");
    out.push_int((long long)ms);
    out.push("\n\n");
    int r = sse_write_(self, &out);
    out.free();
    return r;
}

void SseStream::close() {
    self.s.close();
    self.ok = 0;
}

// ── Parser ────────────────────────────────────────────────────────────────

void SseEvent::free() {
    self.event.free();
    self.data.free();
    self.id.free();
}

struct SseParser SseParser::new() {
    struct SseParser p;
    p.buf = String::new();
    p.data = String::new();
    p.event = String::new();
    p.last_id = String::new();
    p.has_data = 0;
    p.retry = -1;
    return p;
}

void SseParser::feed(const &unsigned char bytes, unsigned long len) {
    unsafe { self.buf.push_n((const char*)bytes, len); }
}

int SseParser::next(&SseEvent out) {
    while (1) {
        // One line (LF, CRLF or a lone CR).
        unsigned long n = self.buf.length();
        const char* b = self.buf.as_ptr();
        unsigned long i = 0UL;
        unsafe { while (i < n && b[i] != '\n' && b[i] != '\r') { i = i + 1UL; } }
        if (i >= n) { return 0; }
        unsigned long next = i + 1UL;
        unsafe {
            if (b[i] == '\r') {
                if (i + 1UL >= n) { return 0; }   // might be the CR of a CRLF
                if (b[i + 1UL] == '\n') { next = i + 2UL; }
            }
        }
        struct String line = self.buf.substr(0UL, i);
        self.buf.remove_range(0UL, next);
        if (line.is_empty()) {
            line.free();
            if (!self.has_data) {
                self.event.clear();
                continue;
            }
            out.event = self.event.is_empty() ? String::from("message") : self.event.clone();
            out.data = self.data.clone();
            out.id = self.last_id.clone();
            out.retry = self.retry;
            self.data.clear();
            self.event.clear();
            self.has_data = 0;
            self.retry = -1;
            return 1;
        }
        if (line.starts_with(":")) { line.free(); continue; }
        long long colon = line.find_char(':');
        struct String field = colon >= 0LL ? line.substr(0UL, (unsigned long)colon) : line.clone();
        struct String value = String::new();
        if (colon >= 0LL) {
            unsigned long vs = (unsigned long)colon + 1UL;
            if (line.char_at(vs) == (int)' ') { vs = vs + 1UL; }
            value.free();
            value = line.substr(vs, line.length());
        }
        if (field.eq_cstr("data")) {
            if (self.has_data) { self.data.push_char('\n'); }
            self.data.push_str(&value);
            self.has_data = 1;
        } else if (field.eq_cstr("event")) {
            self.event.clear();
            self.event.push_str(&value);
        } else if (field.eq_cstr("id")) {
            if (value.find_char((char)0) < 0LL) {
                self.last_id.clear();
                self.last_id.push_str(&value);
            }
        } else if (field.eq_cstr("retry")) {
            int pok = 0;
            long long v = value.parse_int(&pok);
            if (pok && value.is_numeric()) { self.retry = (int)v; }
        }
        field.free();
        value.free();
        line.free();
    }
    return 0;
}

void SseParser::free() {
    self.buf.free();
    self.data.free();
    self.event.free();
    self.last_id.free();
}

// ── Client ────────────────────────────────────────────────────────────────

struct SseClient SseClient::connect(const &char url, const ?&char last_event_id, int timeout_ms, ?&int ok) {
    return SseClient::connect_tls(url, last_event_id, null, timeout_ms, ok);
}

struct SseClient SseClient::connect_tls(const &char url, const ?&char last_event_id, const ?&TlsConfig cfg,
                                        int timeout_ms, ?&int ok) {
    struct SseClient c;
    c.fd = -1;
    c.tls = null;
    c.chunked = 0;
    c.chunk_left = 0UL;
    c.eof = 0;
    c.parser = SseParser::new();
    c.raw = String::new();
    int good = 1;
    int pok = 0;
    struct Url u = Url::parse(url, &pok);
    int tls = pok && u.scheme.eq_cstr("https");
    if (!pok || (!tls && !u.scheme.eq_cstr("http")) || u.host.is_empty()) { good = 0; }
    int fd = -1;
    struct TcpStream s = TcpStream::from_fd(-1);
    if (good) {
        s = http_open_stream(u.host.as_ptr(), (unsigned short)u.port_or_default(), tls, cfg,
                             timeout_ms, timeout_ms, null);
        fd = s.fd;
        if (fd < 0) { good = 0; }
    }
    if (good) {
        struct String req = String::from("GET ");
        struct String target = u.request_target();
        struct String hp = u.host_port();
        req.push_str(&target);
        req.push(" HTTP/1.1\r\nHost: ");
        req.push_str(&hp);
        req.push("\r\nAccept: text/event-stream\r\nCache-Control: no-cache\r\n");
        match (last_event_id) {
            case some(id): { req.push("Last-Event-ID: "); req.push(id); req.push("\r\n"); }
            case null: { }
        }
        req.push("\r\n");
        target.free();
        hp.free();
        unsafe { if (!write_all(&s, (const &unsigned char)req.as_ptr(), req.length())) { good = 0; } }
        req.free();
    }
    if (good) {
        // The response head; what follows it goes to the event decoder.
        struct String head = String::new();
        long long end = -1LL;
        unsigned char buf[4096];
        while (end < 0LL && head.length() < 65536UL) {
            long long n = s.read(&buf[0], 4096UL);
            if (n <= 0LL) { break; }
            unsafe { head.push_n((const char*)&buf[0], (unsigned long)n); }
            end = head.index_of("\r\n\r\n");
        }
        if (end < 0LL || !head.starts_with("HTTP/1.") || head.length() < 12UL || head.char_at(9UL) != (int)'2') {
            good = 0;
        } else {
            struct String hdrs = head.substr(0UL, (unsigned long)end);
            struct String ct = http_header_get(hdrs.as_ptr(), "Content-Type");
            struct String ctl = ct.to_lower();
            if (!ctl.starts_with("text/event-stream")) { good = 0; }
            ctl.free();
            ct.free();
            c.chunked = http_header_has_token(hdrs.as_ptr(), "Transfer-Encoding", "chunked");
            hdrs.free();
            struct String rest = head.substr((unsigned long)end + 4UL, head.length());
            c.raw.push_str(&rest);
            rest.free();
        }
        head.free();
    }
    if (good) {
        if (timeout_ms > 0) { sock_set_io_timeout(fd, 0); }
        c.fd = fd;
        c.tls = s.tls;
    } else if (fd >= 0) {
        s.close();
    }
    u.free();
    if (ok != (int*)0) { unsafe { *ok = good; } }
    return c;
}

// Moves decoded bytes from 'raw' into the parser; 0 on malformed chunking.
static int sse_drain_raw_(&SseClient c) {
    if (!c.chunked) {
        unsafe { c.parser.feed((const &unsigned char)c.raw.as_ptr(), c.raw.length()); }
        c.raw.clear();
        return 1;
    }
    while (!c.raw.is_empty()) {
        if (c.chunk_left > 0UL) {
            unsigned long take = c.raw.length() < c.chunk_left ? c.raw.length() : c.chunk_left;
            unsafe { c.parser.feed((const &unsigned char)c.raw.as_ptr(), take); }
            c.raw.remove_range(0UL, take);
            c.chunk_left = c.chunk_left - take;
            continue;
        }
        // Expect "\r\n" after data (or a size line).
        if (c.raw.starts_with("\r\n")) { c.raw.remove_range(0UL, 2UL); continue; }
        long long eol = c.raw.index_of("\r\n");
        if (eol < 0LL) { return 1; }
        unsigned long size = 0UL;
        unsigned long i = 0UL;
        int any = 0;
        while (i < (unsigned long)eol) {
            int ch = c.raw.char_at(i);
            int d = -1;
            if (ch >= (int)'0' && ch <= (int)'9') { d = ch - (int)'0'; }
            else if (ch >= (int)'a' && ch <= (int)'f') { d = ch - (int)'a' + 10; }
            else if (ch >= (int)'A' && ch <= (int)'F') { d = ch - (int)'A' + 10; }
            if (d < 0) { break; }
            size = size * 16UL + (unsigned long)d;
            any = 1;
            i = i + 1UL;
        }
        if (!any) { return 0; }
        c.raw.remove_range(0UL, (unsigned long)eol + 2UL);
        if (size == 0UL) { c.eof = 1; c.raw.clear(); return 1; }
        c.chunk_left = size;
    }
    return 1;
}

int SseClient::next(&SseEvent out) {
    if (self.fd < 0) { return 0; }
    while (1) {
        if (!sse_drain_raw_(self)) { return 0; }
        if (self.parser.next(out)) { return 1; }
        if (self.eof) { return 0; }
        unsigned char buf[4096];
        struct TcpStream s = TcpStream::from_fd(self.fd);
        s.tls = self.tls;
        long long n = s.read(&buf[0], 4096UL);
        if (n <= 0LL) {
            self.eof = 1;
            // A final event without its blank line is dropped, as
            // EventSource does.
            return 0;
        }
        unsafe { self.raw.push_n((const char*)&buf[0], (unsigned long)n); }
    }
    return 0;
}

void SseClient::close() {
    if (self.fd >= 0) {
        struct TcpStream s = TcpStream::from_fd(self.fd);
        s.tls = self.tls;
        s.close();
        self.fd = -1;
        self.tls = null;
    }
    self.parser.free();
    self.raw.free();
}

} // namespace std
