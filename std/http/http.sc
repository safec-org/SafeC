// SafeC Standard Library — HTTP/1.1 implementation (see http.h)
#pragma once
#include <std/http/http.h>
#include <std/collections/string.sc>
#include <std/mem.sc>
#include <std/fmt.sc>
#include <std/convert.sc>
#include <std/str.sc>
#include <std/sched/io_nb.h>
#include <std/thread.h>
#include <std/thread.sc>
// A backend for io_nb.h's declarations (io_nb_bsd.sc / io_nb_linux.sc /
// io_nb_win32.sc) must be included by the caller before this file, the
// same way std/ipc/uds.h's callers pick uds_bsd.sc/uds_linux.sc themselves
// — see io_nb.h's top comment for the backend list. http.sc only ever
// calls the portable function signatures, never anything backend-specific.

namespace std {

extern int read(int fd, void* buf, unsigned long count);
extern long write(int fd, const void* buf, unsigned long count);
extern int close(int fd);

// POSIX 'struct hostent' (<netdb.h>) — stable layout across BSD/Linux.
struct HostEnt_ {
    char*  h_name;
    char** h_aliases;
    int    h_addrtype;
    int    h_length;
    char** h_addr_list;
};
extern void* gethostbyname(const char* name);

void HttpResponse::free() {
    self.headers.free();
    self.body.free();
}

void HttpRequest::free() {
    self.method.free();
    self.path.free();
    self.headers.free();
    self.body.free();
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

int http_resolve_host(const char* host, unsigned int* addr_out) {
    if (parse_dotted_quad_(host, addr_out)) { return 1; }

    void* hePtr;
    unsafe { hePtr = gethostbyname(host); }
    if (hePtr == (void*)0) { return 0; }
    struct HostEnt_* he;
    unsafe { he = (struct HostEnt_*)hePtr; }
    char** addrList;
    unsafe { addrList = he->h_addr_list; }
    if (addrList == (char**)0) { return 0; }
    char* first;
    unsafe { first = addrList[0]; }
    if (first == (char*)0) { return 0; }
    unsigned int netAddr = 0U;
    int k = 0;
    while (k < 4) {
        unsigned char b;
        unsafe { b = (unsigned char)first[k]; }
        netAddr = netAddr | (((unsigned int)b) << (k * 8));
        k = k + 1;
    }
    unsafe { *addr_out = netAddr; }
    return 1;
}

// ── Blocking socket I/O helpers ───────────────────────────────────────────

// Reads bytes into 'out' until 'terminator' (e.g. "\r\n\r\n") is found
// within what's been read so far, or the connection closes. Returns the
// byte offset (into 'out') right after the terminator on success, or -1 if
// the connection closed before the terminator appeared.
static long read_until_(int fd, struct String* out, const char* terminator) {
    unsigned long termLen = str_len(terminator);
    char chunk[512];
    while (1) {
        long n;
        unsafe { n = read(fd, (void*)chunk, 512UL); }
        if (n <= 0L) { return -1L; }
        unsafe { out->push_n(chunk, (unsigned long)n); }
        long idx = out->index_of(terminator);
        if (idx >= 0L) { return idx + (long)termLen; }
    }
}

// Reads exactly 'want' more bytes, appending to 'out'. Returns 1 on
// success, 0 if the connection closed early.
static int read_n_(int fd, struct String* out, unsigned long want) {
    unsigned long remaining = want;
    char chunk[4096];
    while (remaining > 0UL) {
        unsigned long ask = (remaining < 4096UL) ? remaining : 4096UL;
        long n;
        unsafe { n = read(fd, (void*)chunk, ask); }
        if (n <= 0L) { return 0; }
        unsafe { out->push_n(chunk, (unsigned long)n); }
        remaining = remaining - (unsigned long)n;
    }
    return 1;
}

static int write_all_(int fd, const char* data, unsigned long len) {
    unsigned long sent = 0UL;
    while (sent < len) {
        long n;
        unsafe { n = write(fd, (const void*)(data + sent), len - sent); }
        if (n <= 0L) { return 0; }
        sent = sent + (unsigned long)n;
    }
    return 1;
}

// ── Header parsing ────────────────────────────────────────────────────────

struct String http_header_get(const char* raw_headers, const char* name) {
    struct String result = string_new();
    unsigned long nameLen = str_len(name);
    struct String headersStr = string_from(raw_headers);
    struct String lines[256];
    unsigned long lineCount = headersStr.split("\r\n", lines, 256UL);
    unsigned long i = 0UL;
    while (i < lineCount) {
        const char* line = lines[i].as_ptr();
        long colon;
        unsafe { colon = (long)string_from(line).index_of(":"); }
        if (colon >= 0L) {
            struct String key = lines[i].substr(0UL, (unsigned long)colon);
            struct String keyTrim = key.trim();
            key.free();
            if (keyTrim.length() == nameLen && keyTrim.eq_cstr_ignore_case(name)) {
                struct String val = lines[i].substr((unsigned long)colon + 1UL, lines[i].length());
                struct String valTrim = val.trim();
                val.free();
                result.free();
                result = valTrim;
                keyTrim.free();
                break;
            }
            keyTrim.free();
        }
        i = i + 1UL;
    }
    unsigned long j = 0UL;
    while (j < lineCount) { lines[j].free(); j = j + 1UL; }
    headersStr.free();
    return result;
}

// ── Response building (server side) ───────────────────────────────────────

const char* http_reason_phrase(int status) {
    if (status == 200) { return "OK"; }
    if (status == 201) { return "Created"; }
    if (status == 202) { return "Accepted"; }
    if (status == 204) { return "No Content"; }
    if (status == 301) { return "Moved Permanently"; }
    if (status == 302) { return "Found"; }
    if (status == 304) { return "Not Modified"; }
    if (status == 400) { return "Bad Request"; }
    if (status == 401) { return "Unauthorized"; }
    if (status == 403) { return "Forbidden"; }
    if (status == 404) { return "Not Found"; }
    if (status == 405) { return "Method Not Allowed"; }
    if (status == 408) { return "Request Timeout"; }
    if (status == 409) { return "Conflict"; }
    if (status == 429) { return "Too Many Requests"; }
    if (status == 500) { return "Internal Server Error"; }
    if (status == 501) { return "Not Implemented"; }
    if (status == 502) { return "Bad Gateway"; }
    if (status == 503) { return "Service Unavailable"; }
    return "Unknown";
}

struct String http_build_response(int status, const char* content_type,
                                   const unsigned char* body, unsigned long body_len) {
    const char* ctype;
    if (content_type) { ctype = content_type; } else { ctype = "text/plain"; }
    struct String out = string_new();
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

// ── Client ────────────────────────────────────────────────────────────────

struct HttpResponse http_request(const char* host, unsigned short port,
                                  const char* method, const char* path,
                                  const char* extra_headers,
                                  const unsigned char* body, unsigned long body_len,
                                  int* ok) {
    struct HttpResponse resp;
    resp.status  = 0;
    resp.headers = string_new();
    resp.body    = string_new();
    int good = 1;

    unsigned int addr = 0U;
    if (!http_resolve_host(host, &addr)) { good = 0; }

    int fd = -1;
    if (good) {
        unsafe { fd = tcp_connect_nb(addr, port); }
        if (fd < 0) { good = 0; }
    }
    if (good) {
        // A blocking-mode I/O call on a socket with connect() still in
        // flight waits for the connection to finish (POSIX-guaranteed —
        // see fd_set_blocking's doc comment in io_nb.h), so no separate
        // "wait for connect to complete" step is needed here.
        unsafe { fd_set_blocking(fd); }
    }

    if (good) {
        struct String req = string_new();
        unsafe {
            req.push(method);
            req.push_char(' ');
            req.push(path);
            req.push(" HTTP/1.1\r\nHost: ");
            req.push(host);
            req.push("\r\nConnection: close\r\n");
            if (extra_headers != (const char*)0) { req.push(extra_headers); }
            if (body_len > 0UL) {
                req.push("Content-Length: ");
                req.push_uint((unsigned long long)body_len);
                req.push("\r\n");
            }
            req.push("\r\n");
            if (body_len > 0UL) { req.push_n((const char*)body, body_len); }
        }
        if (!write_all_(fd, req.as_ptr(), req.length())) { good = 0; }
        req.free();
    }

    if (good) {
        struct String buf = string_new();
        long headerEnd = read_until_(fd, &buf, "\r\n\r\n");
        if (headerEnd < 0L) {
            good = 0;
        } else {
            struct String headerBlock = buf.substr(0UL, (unsigned long)headerEnd);
            long firstLineEnd = headerBlock.index_of("\r\n");
            struct String statusLine = (firstLineEnd >= 0L)
                ? headerBlock.substr(0UL, (unsigned long)firstLineEnd)
                : headerBlock.clone();
            struct String restHeaders = (firstLineEnd >= 0L)
                ? headerBlock.substr((unsigned long)firstLineEnd + 2UL, headerBlock.length())
                : string_new();
            headerBlock.free();

            // Status line: "HTTP/1.1 200 OK"
            struct String parts[8];
            unsigned long np = statusLine.split(" ", parts, 8UL);
            int statusCode = 0;
            if (np >= 2UL) {
                int pok = 0;
                statusCode = (int)parts[1].parse_int(&pok);
            }
            unsigned long pi = 0UL;
            while (pi < np) { parts[pi].free(); pi = pi + 1UL; }
            statusLine.free();

            resp.status = statusCode;
            resp.headers.free();
            resp.headers = restHeaders;

            struct String clStr = http_header_get(resp.headers.as_ptr(), "Content-Length");
            unsigned long contentLen = 0UL;
            if (!clStr.is_empty()) {
                int clok = 0;
                contentLen = (unsigned long)clStr.parse_int(&clok);
            }
            clStr.free();

            unsigned long already = buf.length() - (unsigned long)headerEnd;
            struct String alreadyBody = buf.substr((unsigned long)headerEnd, buf.length());
            resp.body.free();
            resp.body = alreadyBody;
            if (contentLen > already) {
                if (!read_n_(fd, &resp.body, contentLen - already)) {
                    good = 0;
                }
            }
        }
        buf.free();
    }

    if (fd >= 0) { unsafe { close(fd); } }
    if (ok != (int*)0) { unsafe { *ok = good; } }
    return resp;
}

struct HttpResponse http_get(const char* host, unsigned short port,
                              const char* path, int* ok) {
    return http_request(host, port, "GET", path, (const char*)0,
                         (const unsigned char*)0, 0UL, ok);
}

struct HttpResponse http_post(const char* host, unsigned short port, const char* path,
                               const char* content_type,
                               const unsigned char* body, unsigned long body_len,
                               int* ok) {
    const char* ctype;
    if (content_type) { ctype = content_type; } else { ctype = "application/octet-stream"; }
    struct String hdr = string_new();
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

// ── Server ────────────────────────────────────────────────────────────────

static struct HttpRequest http_parse_request_(int fd, int* ok) {
    struct HttpRequest req;
    req.method  = string_new();
    req.path    = string_new();
    req.headers = string_new();
    req.body    = string_new();
    int good = 1;

    struct String buf = string_new();
    long headerEnd = read_until_(fd, &buf, "\r\n\r\n");
    if (headerEnd < 0L) {
        good = 0;
    } else {
        struct String headerBlock = buf.substr(0UL, (unsigned long)headerEnd);
        long firstLineEnd = headerBlock.index_of("\r\n");
        struct String requestLine = (firstLineEnd >= 0L)
            ? headerBlock.substr(0UL, (unsigned long)firstLineEnd)
            : headerBlock.clone();
        struct String restHeaders = (firstLineEnd >= 0L)
            ? headerBlock.substr((unsigned long)firstLineEnd + 2UL, headerBlock.length())
            : string_new();
        headerBlock.free();

        // Request line: "GET /path HTTP/1.1"
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

        if (good) {
            struct String clStr = http_header_get(req.headers.as_ptr(), "Content-Length");
            unsigned long contentLen = 0UL;
            if (!clStr.is_empty()) {
                int clok = 0;
                contentLen = (unsigned long)clStr.parse_int(&clok);
            }
            clStr.free();

            unsigned long already = buf.length() - (unsigned long)headerEnd;
            req.body.free();
            req.body = buf.substr((unsigned long)headerEnd, buf.length());
            if (contentLen > already) {
                if (!read_n_(fd, &req.body, contentLen - already)) { good = 0; }
            }
        }
    }
    buf.free();
    if (ok != (int*)0) { unsafe { *ok = good; } }
    return req;
}

int http_serve(unsigned short port, HttpHandler handler) {
    int listenFd;
    unsafe { listenFd = tcp_listen_nb(port); }
    if (listenFd < 0) { return -1; }
    unsafe { fd_set_blocking(listenFd); }

    while (1) {
        int clientFd;
        unsafe { clientFd = tcp_accept_nb(listenFd); }
        if (clientFd < 0) { continue; }
        unsafe { fd_set_blocking(clientFd); }

        int reqOk = 0;
        struct HttpRequest req = http_parse_request_(clientFd, &reqOk);
        if (reqOk) {
            struct HttpResponse resp = handler(&req);
            struct String contentType = http_header_get(resp.headers.as_ptr(), "Content-Type");
            struct String respText;
            unsafe {
                respText = http_build_response(
                    resp.status, contentType.as_ptr(),
                    (const unsigned char*)resp.body.as_ptr(), resp.body.length());
            }
            contentType.free();
            write_all_(clientFd, respText.as_ptr(), respText.length());
            respText.free();
            resp.free();
        }
        req.free();
        unsafe { close(clientFd); }
    }
}

// ── Threaded server ──────────────────────────────────────────────────────────

struct __HttpThreadCtx {
    int         listenFd;
    HttpHandler handler;
};

static void* __http_worker_main(void* argPtr) {
    struct __HttpThreadCtx* ctx;
    unsafe { ctx = (struct __HttpThreadCtx*)argPtr; }

    while (1) {
        int clientFd;
        unsafe { clientFd = tcp_accept_nb(ctx->listenFd); }
        if (clientFd < 0) { continue; }
        unsafe { fd_set_blocking(clientFd); }

        int reqOk = 0;
        struct HttpRequest req = http_parse_request_(clientFd, &reqOk);
        if (reqOk) {
            struct HttpResponse resp;
            unsafe { resp = ctx->handler(&req); }
            struct String contentType = http_header_get(resp.headers.as_ptr(), "Content-Type");
            struct String respText;
            unsafe {
                respText = http_build_response(
                    resp.status, contentType.as_ptr(),
                    (const unsigned char*)resp.body.as_ptr(), resp.body.length());
            }
            contentType.free();
            write_all_(clientFd, respText.as_ptr(), respText.length());
            respText.free();
            resp.free();
        }
        req.free();
        unsafe { close(clientFd); }
    }
    return (void*)0;
}

int http_serve_threaded(unsigned short port, HttpHandler handler, int numThreads) {
    if (numThreads < 1) { return -1; }
    int listenFd;
    unsafe { listenFd = tcp_listen_nb(port); }
    if (listenFd < 0) { return -1; }
    unsafe { fd_set_blocking(listenFd); }

    struct __HttpThreadCtx* ctx;
    unsafe { ctx = (struct __HttpThreadCtx*)malloc(sizeof(struct __HttpThreadCtx)); }
    unsafe { ctx->listenFd = listenFd; ctx->handler = handler; }

    unsigned long long tid = 0ULL;
    int i = 0;
    while (i < numThreads) {
        unsafe {
            if (thread_create(&tid, (void*)__http_worker_main, (void*)ctx) != 0) { return -1; }
        }
        i = i + 1;
    }
    // Every worker loops forever (same "does not return" contract as
    // http_serve) — joining the last one blocks the calling thread for
    // the process's lifetime, same as http_serve's own infinite while(1).
    unsafe { thread_join(tid); }
    return 0;
}

} // namespace std
