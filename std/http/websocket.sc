// SafeC Standard Library — WebSocket implementation (see websocket.h)
#pragma once
#include <std/stddef.h>
#include <std/http/websocket.h>
#include <std/http/http.h>
#include <std/collections/string.h>
#include <std/mem.h>
#include <std/fmt.h>
#include <std/convert.h>
#include <std/str.h>
#include <std/sched/io_nb.h>
#include <std/crypto/sha1.h>
#include <std/crypto/sha1.h>
#include <std/crypto/rng.h>
#include <std/crypto/rng.h>
#include <std/encoding/base64.h>
#include <std/encoding/base64.h>
// See websocket.h's top comment: a std/sched/io_nb_*.sc backend and
// std/http/http.sc must both already be included by the caller before this
// file (http.sc supplies http_resolve_host/http_header_get, reused below).

namespace std {

// Same platform split as http.sc's sock_read_/sock_write_/sock_close_
// (file-local there, so repeated here): Winsock sockets only support
// recv/send/closesocket, and on Linux writes go through send() with
// MSG_NOSIGNAL so a peer that has gone away can't kill the process with
// SIGPIPE (io_nb_bsd.sc sets SO_NOSIGPIPE on BSD/macOS sockets instead).
#ifdef _WIN32
extern int recv(unsigned long long fd, void* buf, int count, int flags);
extern int send(unsigned long long fd, const void* buf, int count, int flags);
extern int closesocket(unsigned long long fd);
static long ws_sock_read_(int fd, void* buf, unsigned long count) {
    unsafe { return (long)recv((unsigned long long)fd, buf, (int)count, 0); }
}
static long ws_sock_write_(int fd, const void* buf, unsigned long count) {
    unsafe { return (long)send((unsigned long long)fd, buf, (int)count, 0); }
}
static int ws_sock_close_(int fd) {
    unsafe { return closesocket((unsigned long long)fd); }
}
#else
extern ssize_t read(int fd, void* buf, size_t count);
extern int close(int fd);
static long ws_sock_read_(int fd, void* buf, unsigned long count) {
    unsafe { return read(fd, buf, count); }
}
#ifdef __linux__
extern ssize_t send(int fd, const void* buf, size_t count, int flags);
static long ws_sock_write_(int fd, const void* buf, unsigned long count) {
    unsafe { return send(fd, buf, count, 0x4000); } // MSG_NOSIGNAL
}
#else
extern ssize_t write(int fd, const void* buf, size_t count);
static long ws_sock_write_(int fd, const void* buf, unsigned long count) {
    unsafe { return write(fd, buf, count); }
}
#endif
static int ws_sock_close_(int fd) {
    unsafe { return close(fd); }
}
#endif

void WsMessage::free() {
    if (self.data != (unsigned char*)0) {
        unsafe { dealloc_raw_((void*)self.data); }
    }
}

// RFC 6455 section 1.3's fixed GUID, concatenated with the client's
// Sec-WebSocket-Key before SHA-1 hashing to derive Sec-WebSocket-Accept.
static const char* ws_guid_() {
    return "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
}

static struct String ws_compute_accept_(const char* key) {
    struct String combined = string_new();
    unsafe {
        combined.push(key);
        combined.push(ws_guid_());
    }
    unsigned char digest[20];
    unsafe {
        sha1((const unsigned char*)combined.as_ptr(), combined.length(), digest);
    }
    combined.free();
    return base64_encode(digest, 20UL);
}

// ── Shared blocking I/O helpers (see http.sc's identically-shaped, but
// file-local, versions — duplicated here rather than exposed from http.h
// since these are small and this module shouldn't need to reach into
// http.sc's request/response-specific internals). ──

static long ws_read_until_(int fd, struct String* out, const char* terminator) {
    unsigned long termLen; unsafe { termLen = str_len(terminator); }
    char chunk[512];
    while (1) {
        long n;
        unsafe { n = ws_sock_read_(fd, (void*)chunk, 512UL); }
        if (n <= 0L) { return -1L; }
        unsafe { out->push_n(chunk, (unsigned long)n); }
        long idx; unsafe { idx = out->index_of(terminator); }
        if (idx >= 0L) { return idx + (long)termLen; }
    }
}

static int ws_write_all_(int fd, const char* data, unsigned long len) {
    unsigned long sent = 0UL;
    while (sent < len) {
        long n;
        unsafe { n = ws_sock_write_(fd, (const void*)(data + sent), len - sent); }
        if (n <= 0L) { return 0; }
        sent = sent + (unsigned long)n;
    }
    return 1;
}

static int ws_read_n_(int fd, unsigned char* out, unsigned long want) {
    unsigned long got = 0UL;
    while (got < want) {
        long n;
        unsafe { n = ws_sock_read_(fd, (void*)(out + got), want - got); }
        if (n <= 0L) { return 0; }
        got = got + (unsigned long)n;
    }
    return 1;
}

// ── Client handshake ───────────────────────────────────────────────────────

struct WsConn ws_connect(const &char host, unsigned short port,
                          const &char path, ?&int ok) {
    return ws_connect_timeout(host, port, path, WS_DEFAULT_TIMEOUT_MS, ok);
}

struct WsConn ws_connect_timeout(const &char host, unsigned short port,
                                  const &char path, int timeout_ms, ?&int ok) {
    struct WsConn conn;
    conn.fd = -1;
    conn.is_server = 0;
    int good = 1;

    unsigned int addr = 0U;
    if (!http_resolve_host(host, &addr)) { good = 0; }

    int fd = -1;
    if (good) {
        unsafe { fd = tcp_connect_timeout(addr, port, timeout_ms); }
        if (fd < 0) { good = 0; }
    }
    if (good && timeout_ms > 0) {
        if (sock_set_io_timeout(fd, timeout_ms) != 0) { good = 0; }
    }

    struct String keyB64 = string_new();
    if (good) {
        unsigned char keyRaw[16];
        struct RngCtx rng;
        int rngOk;
        unsafe { rngOk = rng.init(); }
        if (!rngOk) {
            good = 0;
        } else {
            unsafe { rng.fill(&keyRaw[0], 16UL); }
            struct String encoded = base64_encode(keyRaw, 16UL);
            keyB64.free();
            keyB64 = encoded;
        }
    }

    if (good) {
        struct String req = string_new();
        unsafe {
            req.push("GET ");
            req.push(path);
            req.push(" HTTP/1.1\r\nHost: ");
            req.push(host);
            req.push("\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                      "Sec-WebSocket-Key: ");
            req.push(keyB64.as_ptr());
            req.push("\r\nSec-WebSocket-Version: 13\r\n\r\n");
        }
        if (!ws_write_all_(fd, req.as_ptr(), req.length())) { good = 0; }
        req.free();
    }

    if (good) {
        struct String buf = string_new();
        long headerEnd = ws_read_until_(fd, &buf, "\r\n\r\n");
        if (headerEnd < 0L) {
            good = 0;
        } else {
            struct String headerBlock = buf.substr(0UL, (unsigned long)headerEnd);
            if (headerBlock.index_of("101") < 0L) { good = 0; }
            struct String accept = http_header_get(headerBlock.as_ptr(), "Sec-WebSocket-Accept");
            struct String expected = ws_compute_accept_(keyB64.as_ptr());
            if (!accept.eq_cstr(expected.as_ptr())) { good = 0; }
            accept.free();
            expected.free();
            headerBlock.free();
        }
        buf.free();
    }
    keyB64.free();

    // The handshake timeout must not outlive it: an open WebSocket can sit
    // idle between messages indefinitely, and recv() blocks until one
    // arrives.
    if (good && timeout_ms > 0) {
        if (sock_set_io_timeout(fd, 0) != 0) { good = 0; }
    }
    if (good) {
        conn.fd = fd;
    } else if (fd >= 0) {
        unsafe { ws_sock_close_(fd); }
    }
    if (ok != (int*)0) { unsafe { *ok = good; } }
    return conn;
}

// ── Server handshake ───────────────────────────────────────────────────────

int ws_listen(unsigned short port) {
    int fd;
    unsafe { fd = tcp_listen_nb(port); }
    if (fd < 0) { return -1; }
    unsafe { fd_set_blocking(fd); }
    return fd;
}

int WsConn::accept(int listenFd) {
    int fd;
    unsafe { fd = tcp_accept_nb(listenFd); }
    if (fd < 0) { return 0; }
    unsafe { fd_set_blocking(fd); }

    int good = 1;
    struct String buf = string_new();
    long headerEnd = ws_read_until_(fd, &buf, "\r\n\r\n");
    struct String key = string_new();
    if (headerEnd < 0L) {
        good = 0;
    } else {
        struct String headerBlock = buf.substr(0UL, (unsigned long)headerEnd);
        struct String foundKey = http_header_get(headerBlock.as_ptr(), "Sec-WebSocket-Key");
        headerBlock.free();
        if (foundKey.is_empty()) {
            good = 0;
        } else {
            key.free();
            key = foundKey;
        }
        if (good == 0) { foundKey.free(); }
    }
    buf.free();

    if (good) {
        struct String accept = ws_compute_accept_(key.as_ptr());
        struct String resp = string_new();
        unsafe {
            resp.push("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                       "Connection: Upgrade\r\nSec-WebSocket-Accept: ");
            resp.push(accept.as_ptr());
            resp.push("\r\n\r\n");
        }
        if (!ws_write_all_(fd, resp.as_ptr(), resp.length())) { good = 0; }
        resp.free();
        accept.free();
    }
    key.free();

    if (!good) {
        unsafe { ws_sock_close_(fd); }
        return 0;
    }
    unsafe {
        self.fd = fd;
        self.is_server = 1;
    }
    return 1;
}

// ── Framing ───────────────────────────────────────────────────────────────

static int ws_write_frame_(struct WsConn* conn, int opcode,
                            const unsigned char* data, unsigned long len) {
    int fd;
    int mustMask;
    unsafe { fd = conn->fd; mustMask = (conn->is_server != 0) ? 0 : 1; }

    unsigned char head[10];
    unsigned long headLen = 0UL;
    unsafe {
        head[0] = (unsigned char)(0x80U | (unsigned int)opcode); // FIN=1
        if (len < 126UL) {
            head[1] = (unsigned char)((mustMask ? 0x80U : 0U) | len);
            headLen = 2UL;
        } else if (len <= 0xFFFFUL) {
            head[1] = (unsigned char)((mustMask ? 0x80U : 0U) | 126U);
            head[2] = (unsigned char)((len >> 8) & 0xFFUL);
            head[3] = (unsigned char)(len & 0xFFUL);
            headLen = 4UL;
        } else {
            head[1] = (unsigned char)((mustMask ? 0x80U : 0U) | 127U);
            int i = 0;
            while (i < 8) {
                head[2 + i] = (unsigned char)((len >> (56 - i * 8)) & 0xFFUL);
                i = i + 1;
            }
            headLen = 10UL;
        }
    }
    if (!ws_write_all_(fd, (const char*)head, headLen)) { return 0; }

    if (mustMask) {
        unsigned char maskKey[4];
        struct RngCtx rng;
        int rngOk;
        unsafe { rngOk = rng.init(); }
        if (!rngOk) { return 0; }
        unsafe { rng.fill(&maskKey[0], 4UL); }
        if (!ws_write_all_(fd, (const char*)maskKey, 4UL)) { return 0; }

        if (len > 0UL) {
            unsigned char* masked;
            unsafe { masked = (unsigned char*)alloc_raw_(len); }
            if (masked == (unsigned char*)0) { return 0; }
            unsigned long i = 0UL;
            while (i < len) {
                unsigned char b;
                unsigned char mk;
                unsafe { b = data[i]; mk = maskKey[i % 4UL]; }
                unsafe { masked[i] = (unsigned char)((unsigned int)b ^ (unsigned int)mk); }
                i = i + 1UL;
            }
            int wrote;
            unsafe { wrote = ws_write_all_(fd, (const char*)masked, len); }
            unsafe { dealloc_raw_((void*)masked); }
            if (!wrote) { return 0; }
        }
    } else {
        if (len > 0UL) {
            int ok;
            unsafe { ok = ws_write_all_(fd, (const char*)data, len); }
            if (!ok) { return 0; }
        }
    }
    return 1;
}

int WsConn::send_text(const &char text) {
    unsigned long len = str_len(text);
    const unsigned char* udata;
    unsafe { udata = (const unsigned char*)text; }
    return ws_write_frame_(self, WS_OPCODE_TEXT, udata, len);
}

int WsConn::send_binary(const &unsigned char data, unsigned long len) {
    return ws_write_frame_(self, WS_OPCODE_BINARY, data, len);
}

int WsConn::send_ping() {
    return ws_write_frame_(self, WS_OPCODE_PING, (const unsigned char*)0, 0UL);
}

int WsConn::send_pong() {
    return ws_write_frame_(self, WS_OPCODE_PONG, (const unsigned char*)0, 0UL);
}

int WsConn::send_close() {
    return ws_write_frame_(self, WS_OPCODE_CLOSE, (const unsigned char*)0, 0UL);
}

int WsConn::recv(&WsMessage out) {
    int fd = self.fd;

    unsigned char head[2];
    if (!ws_read_n_(fd, head, 2UL)) { return 0; }
    int opcode;
    int maskBit;
    unsigned long len7;
    unsafe {
        opcode = (int)(head[0] & 0x0FU);
        maskBit = (int)((head[1] >> 7) & 0x1U);
        len7 = (unsigned long)(head[1] & 0x7FU);
    }

    unsigned long payloadLen = len7;
    if (len7 == 126UL) {
        unsigned char ext[2];
        if (!ws_read_n_(fd, ext, 2UL)) { return 0; }
        unsafe { payloadLen = ((unsigned long)ext[0] << 8) | (unsigned long)ext[1]; }
    } else if (len7 == 127UL) {
        unsigned char ext[8];
        if (!ws_read_n_(fd, ext, 8UL)) { return 0; }
        unsigned long v = 0UL;
        int i = 0;
        while (i < 8) {
            unsigned char b;
            unsafe { b = ext[i]; }
            v = (v << 8) | (unsigned long)b;
            i = i + 1;
        }
        payloadLen = v;
    }

    unsigned char maskKey[4];
    if (maskBit != 0) {
        if (!ws_read_n_(fd, maskKey, 4UL)) { return 0; }
    }

    unsigned char* payload = (unsigned char*)0;
    if (payloadLen > 0UL) {
        unsafe { payload = (unsigned char*)alloc_raw_(payloadLen); }
        if (payload == (unsigned char*)0) { return 0; }
        if (!ws_read_n_(fd, payload, payloadLen)) {
            unsafe { dealloc_raw_((void*)payload); }
            return 0;
        }
        if (maskBit != 0) {
            unsigned long i = 0UL;
            while (i < payloadLen) {
                unsigned char b;
                unsigned char mk;
                unsafe { b = payload[i]; mk = maskKey[i % 4UL]; }
                unsafe { payload[i] = (unsigned char)((unsigned int)b ^ (unsigned int)mk); }
                i = i + 1UL;
            }
        }
    }

    unsafe {
        out->opcode = opcode;
        out->data   = payload;
        out->len    = payloadLen;
    }
    return 1;
}

void WsConn::close() {
    self.send_close();
    int fd = self.fd;
    if (fd >= 0) { unsafe { ws_sock_close_(fd); } }
}

} // namespace std
