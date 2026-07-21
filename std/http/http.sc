// SafeC Standard Library — HTTP/1.1 implementation (see http.h)
#pragma once
#include <std/http/http.h>
#include <std/collections/string.sc>
#include <std/mem.sc>
#include <std/fmt.sc>
#include <std/convert.sc>
#include <std/str.sc>
#include <std/sched/io_nb.h>
#include <std/sched/reactor.h>
#include <std/thread.h>
#include <std/thread.sc>
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
extern long read(int fd, void* buf, unsigned long count);
extern long write(int fd, const void* buf, unsigned long count);
extern int close(int fd);
static long sock_read_(int fd, void* buf, unsigned long count) {
    unsafe { return read(fd, buf, count); }
}
static long sock_write_(int fd, const void* buf, unsigned long count) {
    unsafe { return write(fd, buf, count); }
}
static int sock_close_(int fd) {
    unsafe { return close(fd); }
}
#endif

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
        unsafe { n = sock_read_(fd, (void*)chunk, 512UL); }
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
        unsafe { n = sock_read_(fd, (void*)chunk, ask); }
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
        unsafe { n = sock_write_(fd, (const void*)(data + sent), len - sent); }
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

    if (fd >= 0) { unsafe { sock_close_(fd); } }
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
        unsafe { sock_close_(clientFd); }
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
        unsafe { sock_close_(clientFd); }
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

// ── Reactor-based server ─────────────────────────────────────────────────────
// Gated on SAFEC_REACTOR_BACKEND_INCLUDED_ (defined by reactor_kqueue.sc/
// reactor_epoll.sc/reactor_win32.sc — see their own comment on this
// macro): Reactor::add/remove/poll and TaskScheduler::spawn_task/await_fd
// below are real symbol references, not just declarations, so a caller
// that includes http.sc *without* a reactor backend (http_serve/
// http_serve_threaded's ordinary callers, unchanged by any of this) would
// otherwise fail to link over functions it never calls and code it never
// asked for — SafeC doesn't dead-strip unused functions before linking.
// Callers that DO want http_serve_reactor already have to include a
// reactor backend first for the exact same reason http_serve/
// http_serve_threaded already require an io_nb backend first (see the
// file-header comment above); this just extends that same opt-in
// convention to the reactor dependency specifically.
#ifdef SAFEC_REACTOR_BACKEND_INCLUDED_
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
    req.method  = string_new();
    req.path    = string_new();
    req.headers = string_new();
    req.body    = string_new();
    int good = 1;

    unsafe {
        struct String headerBlock = buf->substr(0UL, (unsigned long)headerEnd);
        long firstLineEnd = headerBlock.index_of("\r\n");
        struct String requestLine = (firstLineEnd >= 0L)
            ? headerBlock.substr(0UL, (unsigned long)firstLineEnd)
            : headerBlock.clone();
        struct String restHeaders = (firstLineEnd >= 0L)
            ? headerBlock.substr((unsigned long)firstLineEnd + 2UL, headerBlock.length())
            : string_new();
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
        // no I/O needed here, pure parsing.
        req.body.free();
        req.body = buf->substr((unsigned long)headerEnd, buf->length());

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

// Task function: int(void* arg, int resume_point) — see std/sync/task.h.
// Always yields '1' (task.sc only checks zero-vs-nonzero; the actual
// resume point lives in st->phase, not the return value) until the
// connection is fully handled, then returns 0 (TASK_DONE). Entirely
// pointer-driven (st is heap-allocated, shared across yields/resumes), so
// the whole body runs inside one unsafe block rather than one per access.
static int http_conn_task_(void* argPtr, int resume_point) {
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
                struct String contentType = http_header_get(resp.headers.as_ptr(), "Content-Type");
                st->respText = http_build_response(
                    resp.status, contentType.as_ptr(),
                    (const unsigned char*)resp.body.as_ptr(), resp.body.length());
                contentType.free();
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
                (const void*)(st->respText.as_ptr() + st->sent),
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
        st->readBuf = string_new();
        st->headerEnd = -1L;
        st->bodyWant = 0UL;
        st->respText = string_new();
        st->sent = 0UL;
        int idx = sched->spawn_task((void*)http_conn_task_, (void*)st);
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
static int http_accept_task_(void* argPtr, int resume_point) {
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

static void* http_reactor_worker_(void* argPtr) {
    unsafe {
        struct HttpReactorThreadCtx_* ctx = (struct HttpReactorThreadCtx_*)argPtr;
        struct HttpReactorShared_* shared = ctx->shared;

        struct Reactor reactor = reactor_init();
        if (reactor.init() != 0) { return (void*)0; }
        struct TaskScheduler sched = task_sched_init();

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
            sched.spawn_task((void*)http_accept_task_, (void*)ac);
        }

        http_worker_loop_(&sched, &reactor, myQueue, shared->handler);

        reactor.close_();
        if (ac != (struct HttpAcceptState_*)0) { free(ac); }
        free(ctx);
        return (void*)0;
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
            if (thread_create(&tid, (void*)http_reactor_worker_, (void*)tctx) != 0) { return -1; }
        }
        i = i + 1;
    }
    unsafe { thread_join(tid); }
    return 0;
}
#endif // SAFEC_REACTOR_BACKEND_INCLUDED_

} // namespace std
