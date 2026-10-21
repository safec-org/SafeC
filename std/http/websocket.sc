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
#include <std/io_traits.h>
#include <std/io_buf.h>
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
    struct String combined = String::new();
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

// ── Client handshake ───────────────────────────────────────────────────────

struct WsConn WsConn::connect(const &char host, unsigned short port,
                              const &char path, ?&int ok) {
    return WsConn::connect_timeout(host, port, path, WS_DEFAULT_TIMEOUT_MS, ok);
}

struct WsConn WsConn::connect_timeout(const &char host, unsigned short port,
                                      const &char path, int timeout_ms, ?&int ok) {
    struct WsConn conn;
    conn.fd = -1;
    conn.is_server = 0;
    conn.early = null;
    conn.earlyLen = 0UL;
    conn.earlyPos = 0UL;
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

    struct String keyB64 = String::new();
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
        struct String req = String::new();
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
        struct TcpStream s = TcpStream::from_fd(fd);
        int sent;
        unsafe { sent = write_all(&s, (const &unsigned char)req.as_ptr(), req.length()); }
        if (!sent) { good = 0; }
        req.free();
    }

    if (good) {
        struct TcpStream s = TcpStream::from_fd(fd);
        struct BufReader<TcpStream> in = BufReader::over(&s);
        struct String headerBlock = String::new();
        if (!in.read_until(&headerBlock, "\r\n\r\n")) {
            good = 0;
        } else {
            if (headerBlock.index_of("101") < 0L) { good = 0; }
            struct String accept = http_header_get(headerBlock.as_ptr(), "Sec-WebSocket-Accept");
            struct String expected = ws_compute_accept_(keyB64.as_ptr());
            if (!accept.eq_cstr(expected.as_ptr())) { good = 0; }
            accept.free();
            expected.free();
        }
        headerBlock.free();
        if (good && !ws_keep_early_(&conn, &in)) { good = 0; }
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
        struct TcpStream s = TcpStream::from_fd(fd);
        s.close();
    }
    if (ok != (int*)0) { unsafe { *ok = good; } }
    return conn;
}

// ── Bytes that arrived with the handshake ─────────────────────────────────

// Keeps what 'in' read past the handshake (the start of the first frame,
// when the peer sent it in the same segment) so recv() doesn't lose it.
static int ws_keep_early_(&WsConn c, &BufReader<TcpStream> in) {
    c.early = null;
    c.earlyLen = 0UL;
    c.earlyPos = 0UL;
    unsigned long n = in.buffered();
    if (n == 0UL) return 1;
    ?&heap unsigned char buf = (?&heap unsigned char)alloc_raw_(n);
    match (buf) {
        case null: { return 0; }
        case some(b): {
            long long got = in.read(b, n);
            if (got != (long long)n) { unsafe { dealloc_raw_((void*)b); } return 0; }
            c.early = b;
            c.earlyLen = n;
        }
    }
    return 1;
}

// A Reader over a connection: the early bytes first, then the socket.
struct WsSource_ {
    &WsConn conn;
    long long read(&unsigned char out, unsigned long n);
};

long long WsSource_::read(&unsigned char out, unsigned long n) {
    &WsConn c = self.conn;
    if (c.earlyPos < c.earlyLen) {
        unsigned long left = c.earlyLen - c.earlyPos;
        unsigned long k = n < left ? n : left;
        match (c.early) {
            case some(b): { unsafe { safe_memcpy_raw_((void*)out, (const void*)&b[c.earlyPos], k); } }
            case null: { return -1LL; }
        }
        c.earlyPos = c.earlyPos + k;
        if (c.earlyPos == c.earlyLen) {
            match (c.early) { case some(b): { unsafe { dealloc_raw_((void*)b); } } case null: { } }
            c.early = null;
            c.earlyLen = 0UL;
            c.earlyPos = 0UL;
        }
        return (long long)k;
    }
    struct TcpStream s = TcpStream::from_fd(c.fd);
    return s.read(out, n);
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
    struct TcpStream s = TcpStream::from_fd(fd);
    struct BufReader<TcpStream> in = BufReader::over(&s);
    struct String headerBlock = String::new();
    struct String key = String::new();
    if (!in.read_until(&headerBlock, "\r\n\r\n")) {
        good = 0;
    } else {
        struct String foundKey = http_header_get(headerBlock.as_ptr(), "Sec-WebSocket-Key");
        if (foundKey.is_empty()) {
            good = 0;
        } else {
            key.free();
            key = foundKey;
        }
        if (good == 0) { foundKey.free(); }
    }
    headerBlock.free();

    if (good) {
        struct String accept = ws_compute_accept_(key.as_ptr());
        struct String resp = String::new();
        unsafe {
            resp.push("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                       "Connection: Upgrade\r\nSec-WebSocket-Accept: ");
            resp.push(accept.as_ptr());
            resp.push("\r\n\r\n");
        }
        int sent;
        unsafe { sent = write_all(&s, (const &unsigned char)resp.as_ptr(), resp.length()); }
        if (!sent) { good = 0; }
        resp.free();
        accept.free();
    }
    key.free();

    if (!good) {
        s.close();
        return 0;
    }
    unsafe {
        self.fd = fd;
        self.is_server = 1;
    }
    if (!ws_keep_early_(self, &in)) { s.close(); self.fd = -1; return 0; }
    return 1;
}

// ── Framing ───────────────────────────────────────────────────────────────

#define WS_MASK_CHUNK_ 16384

static int ws_write_frame_(struct WsConn* conn, int opcode,
                            const unsigned char* data, unsigned long len) {
    int fd;
    int mustMask;
    unsafe { fd = conn->fd; mustMask = (conn->is_server != 0) ? 0 : 1; }
    struct TcpStream s = TcpStream::from_fd(fd);

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
    // Header, mask key and a small payload leave in one write; a large
    // payload streams through without a copy of the whole of it.
    struct BufWriter<TcpStream> out = BufWriter::over(&s);
    if (!write_all(&out, &head[0], headLen)) { return 0; }

    if (mustMask) {
        unsigned char maskKey[4];
        struct RngCtx rng;
        int rngOk;
        unsafe { rngOk = rng.init(); }
        if (!rngOk) { return 0; }
        unsafe { rng.fill(&maskKey[0], 4UL); }
        if (!write_all(&out, &maskKey[0], 4UL)) { return 0; }

        // Masked a chunk at a time (a multiple of 4, so the key lines up).
        unsigned char chunk[WS_MASK_CHUNK_];
        unsigned long i = 0UL;
        while (i < len) {
            unsigned long n = len - i;
            if (n > (unsigned long)WS_MASK_CHUNK_) { n = (unsigned long)WS_MASK_CHUNK_; }
            unsigned long k = 0UL;
            unsafe {
                while (k < n) {
                    chunk[k] = (unsigned char)((unsigned int)data[i + k] ^ (unsigned int)maskKey[k & 3UL]);
                    k = k + 1UL;
                }
            }
            if (!write_all(&out, &chunk[0], n)) { return 0; }
            i = i + n;
        }
    } else if (len > 0UL) {
        int ok;
        unsafe { ok = write_all(&out, (const &unsigned char)data, len); }
        if (!ok) { return 0; }
    }
    return out.flush() == 0;
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
    struct WsSource_ s;
    s.conn = self;

    unsigned char head[2];
    if (!read_exact(&s, &head[0], 2UL)) { return 0; }
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
        if (!read_exact(&s, &ext[0], 2UL)) { return 0; }
        unsafe { payloadLen = ((unsigned long)ext[0] << 8) | (unsigned long)ext[1]; }
    } else if (len7 == 127UL) {
        unsigned char ext[8];
        if (!read_exact(&s, &ext[0], 8UL)) { return 0; }
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
        if (!read_exact(&s, &maskKey[0], 4UL)) { return 0; }
    }

    unsigned char* payload = (unsigned char*)0;
    if (payloadLen > 0UL) {
        unsafe { payload = (unsigned char*)alloc_raw_(payloadLen); }
        if (payload == (unsigned char*)0) { return 0; }
        int got;
        unsafe { got = read_exact(&s, (&unsigned char)payload, payloadLen); }
        if (!got) {
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
    if (self.fd >= 0) {
        struct TcpStream s = TcpStream::from_fd(self.fd);
        s.close();
    }
    match (self.early) { case some(b): { unsafe { dealloc_raw_((void*)b); } } case null: { } }
    self.early = null;
    self.earlyLen = 0UL;
    self.earlyPos = 0UL;
}

} // namespace std
