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
#include <std/encoding/url.h>
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
    return base64_encode(digest[0..20]);
}

// ── Client handshake ───────────────────────────────────────────────────────

struct WsConn WsConn::connect(const &char host, unsigned short port,
                              const &char path, ?&int ok) {
    return WsConn::connect_timeout(host, port, path, WS_DEFAULT_TIMEOUT_MS, ok);
}

// The connection's stream (plain TCP, or its TLS session for wss://).
static struct TcpStream ws_stream_(const &WsConn c) {
    struct TcpStream s = TcpStream::from_fd(c.fd);
    s.tls = c.tls;
    return s;
}

struct WsConn WsConn::connect_timeout(const &char host, unsigned short port,
                                      const &char path, int timeout_ms, ?&int ok) {
    return WsConn::connect_tls(host, port, path, 0, null, timeout_ms, ok);
}

struct WsConn WsConn::connect_tls(const &char host, unsigned short port, const &char path,
                                  int tls, const ?&TlsConfig cfg, int timeout_ms, ?&int ok) {
    struct WsConn conn;
    conn.fd = -1;
    conn.tls = null;
    conn.is_server = 0;
    conn.early = null;
    conn.earlyLen = 0UL;
    conn.earlyPos = 0UL;
    int good = 1;

    // Bounded by timeout_ms throughout the opening handshake (TLS included).
    struct TcpStream s = http_open_stream(host, port, tls, cfg, timeout_ms, timeout_ms, null);
    int fd = s.fd;
    if (fd < 0) { good = 0; }

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
            struct String encoded = base64_encode(keyRaw[0..16]);
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
        int sent;
        unsafe { sent = write_all(&s, req.as_bytes()); }
        if (!sent) { good = 0; }
        req.free();
    }

    if (good) {
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
        conn.tls = s.tls;
    } else if (fd >= 0) {
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
            long long got;
            unsafe { got = in.read(((unsigned char*)b)[0..n]); }
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
    long long read([]unsigned char out);
};

long long WsSource_::read([]unsigned char out_s) {
    unsigned long n = out_s.len;
    if (n == 0UL) { return 0LL; }
    &unsigned char out;
    unsafe { out = (&unsigned char)out_s.ptr; }
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
    struct TcpStream s = ws_stream_(c);
    return s.read(out_s);
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
        unsafe { sent = write_all(&s, resp.as_bytes()); }
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
        self.tls = null;
        self.is_server = 1;
    }
    if (!ws_keep_early_(self, &in)) { s.close(); self.fd = -1; return 0; }
    return 1;
}

// ── Framing ───────────────────────────────────────────────────────────────

#define WS_MASK_CHUNK_ 16384

static int ws_write_frame_(struct WsConn* conn, int opcode,
                            const unsigned char* data, unsigned long len) {
    int mustMask;
    struct TcpStream s;
    unsafe {
        mustMask = (conn->is_server != 0) ? 0 : 1;
        s = ws_stream_((&WsConn)conn);
    }

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
    if (!write_all(&out, head[0..headLen])) { return 0; }

    if (mustMask) {
        unsigned char maskKey[4];
        struct RngCtx rng;
        int rngOk;
        unsafe { rngOk = rng.init(); }
        if (!rngOk) { return 0; }
        unsafe { rng.fill(&maskKey[0], 4UL); }
        if (!write_all(&out, maskKey[0..4UL])) { return 0; }

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
            if (!write_all(&out, chunk[0..n])) { return 0; }
            i = i + n;
        }
    } else if (len > 0UL) {
        int ok;
        unsafe { ok = write_all(&out, ((const unsigned char*)data)[0..len]); }
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

static int ws_recv_frame_(&WsConn conn, &WsMessage out, &int finOut) {
    struct WsSource_ s;
    s.conn = conn;

    unsigned char head[2];
    if (!read_exact(&s, head[0..2UL])) { return 0; }
    int opcode;
    int maskBit;
    unsigned long len7;
    unsafe {
        *finOut = (int)((head[0] >> 7) & 0x1U);
        opcode = (int)(head[0] & 0x0FU);
        maskBit = (int)((head[1] >> 7) & 0x1U);
        len7 = (unsigned long)(head[1] & 0x7FU);
    }

    unsigned long payloadLen = len7;
    if (len7 == 126UL) {
        unsigned char ext[2];
        if (!read_exact(&s, ext[0..2UL])) { return 0; }
        unsafe { payloadLen = ((unsigned long)ext[0] << 8) | (unsigned long)ext[1]; }
    } else if (len7 == 127UL) {
        unsigned char ext[8];
        if (!read_exact(&s, ext[0..8UL])) { return 0; }
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
        if (!read_exact(&s, maskKey[0..4UL])) { return 0; }
    }

    unsigned char* payload = (unsigned char*)0;
    if (payloadLen > 0UL) {
        unsafe { payload = (unsigned char*)alloc_raw_(payloadLen); }
        if (payload == (unsigned char*)0) { return 0; }
        int got;
        unsafe { got = read_exact(&s, ((unsigned char*)payload)[0..payloadLen]); }
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

int WsConn::recv(&WsMessage out) {
    int fin = 1;
    return ws_recv_frame_(self, out, &fin);
}

int WsConn::recv_message(&WsMessage out, unsigned long max_len) {
    struct String acc = String::new();
    int msgOp = -1;
    while (1) {
        struct WsMessage f;
        int fin = 1;
        if (!ws_recv_frame_(self, &f, &fin)) { acc.free(); return 0; }
        if (f.opcode == WS_OPCODE_PING) {
            // Answered automatically, with the same payload (RFC 6455 5.5.3).
            const unsigned char* d = (const unsigned char*)0;
            unsafe { d = (const unsigned char*)f.data; }
            ws_write_frame_(self, WS_OPCODE_PONG, d, f.len);
            f.free();
            continue;
        }
        if (f.opcode == WS_OPCODE_PONG) { f.free(); continue; }
        if (f.opcode == WS_OPCODE_CLOSE) {
            acc.free();
            *out = f;
            return 1;
        }
        if (f.opcode == WS_OPCODE_CONTINUATION) {
            if (msgOp < 0) { f.free(); acc.free(); return 0; }
        } else {
            if (msgOp >= 0) { f.free(); acc.free(); return 0; }   // new message inside a fragmented one
            msgOp = f.opcode;
        }
        if (max_len != 0UL && acc.length() + f.len > max_len) { f.free(); acc.free(); return 0; }
        if (f.len > 0UL) { unsafe { acc.push_n((const char*)f.data, f.len); } }
        f.free();
        if (fin) { break; }
    }
    unsigned char* payload = (unsigned char*)0;
    if (acc.length() > 0UL) {
        unsafe {
            payload = (unsigned char*)alloc_raw_(acc.length());
            if (payload == (unsigned char*)0) { acc.free(); return 0; }
            safe_memcpy_raw_((void*)payload, (const void*)acc.as_ptr(), acc.length());
        }
    }
    unsafe {
        out->opcode = msgOp;
        out->data = payload;
        out->len = acc.length();
    }
    acc.free();
    return 1;
}

int ws_is_upgrade(const &HttpRequest req) {
    return req.method.eq_cstr("GET") &&
           http_header_has_token(req.headers.as_ptr(), "Upgrade", "websocket") &&
           http_header_has_token(req.headers.as_ptr(), "Connection", "upgrade") &&
           req.has_header("Sec-WebSocket-Key");
}

struct WsConn WsConn::accept_upgrade(const &HttpRequest req, &HttpConn conn, ?&int ok) {
    struct WsConn c;
    c.fd = -1;
    c.tls = null;
    c.is_server = 1;
    c.early = null;
    c.earlyLen = 0UL;
    c.earlyPos = 0UL;
    int good = ws_is_upgrade(req);
    struct String ver = req.header("Sec-WebSocket-Version");
    if (!ver.eq_cstr("13")) { good = 0; }
    ver.free();
    if (!good) {
        if (ok != (int*)0) { unsafe { *ok = 0; } }
        return c;
    }
    struct String key = req.header("Sec-WebSocket-Key");
    struct String accept = ws_compute_accept_(key.as_ptr());
    key.free();
    struct String early = String::new();
    struct TcpStream s = conn.detach(&early);
    struct String resp = String::from("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                                      "Connection: Upgrade\r\nSec-WebSocket-Accept: ");
    resp.push_str(&accept);
    resp.push("\r\n\r\n");
    accept.free();
    int sent;
    unsafe { sent = write_all(&s, resp.as_bytes()); }
    resp.free();
    // The server's I/O timeout must not cut an idle WebSocket off.
    sock_set_io_timeout(s.fd, 0);
    if (!sent) {
        s.close();
        early.free();
        if (ok != (int*)0) { unsafe { *ok = 0; } }
        return c;
    }
    c.fd = s.fd;
    c.tls = s.tls;
    if (early.length() > 0UL) {
        ?&heap unsigned char buf = (?&heap unsigned char)alloc_raw_(early.length());
        match (buf) {
            case some(b): {
                unsafe { safe_memcpy_raw_((void*)b, (const void*)early.as_ptr(), early.length()); }
                c.early = b;
                c.earlyLen = early.length();
            }
            case null: { }
        }
    }
    early.free();
    if (ok != (int*)0) { unsafe { *ok = 1; } }
    return c;
}

struct WsConn WsConn::connect_url(const &char url, int timeout_ms, ?&int ok) {
    return WsConn::connect_url_tls(url, null, timeout_ms, ok);
}

struct WsConn WsConn::connect_url_tls(const &char url, const ?&TlsConfig cfg, int timeout_ms, ?&int ok) {
    int pok = 0;
    struct Url u = Url::parse(url, &pok);
    int tls = pok && (u.scheme.eq_cstr("wss") || u.scheme.eq_cstr("https"));
    if (pok && !tls && !u.scheme.eq_cstr("ws") && !u.scheme.eq_cstr("http")) { pok = 0; }
    int port = u.port_or_default();
    if (u.host.is_empty() || port <= 0) { pok = 0; }
    struct WsConn c;
    if (!pok) {
        c.fd = -1;
        c.tls = null;
        c.is_server = 0;
        c.early = null;
        c.earlyLen = 0UL;
        c.earlyPos = 0UL;
        u.free();
        if (ok != (int*)0) { unsafe { *ok = 0; } }
        return c;
    }
    struct String target = u.request_target();
    c = WsConn::connect_tls(u.host.as_ptr(), (unsigned short)port, target.as_ptr(), tls, cfg, timeout_ms, ok);
    target.free();
    u.free();
    return c;
}

void WsConn::close() {
    self.send_close();
    if (self.fd >= 0) {
        struct TcpStream s = ws_stream_(self);
        s.close();
        self.fd = -1;
        self.tls = null;
    }
    match (self.early) { case some(b): { unsafe { dealloc_raw_((void*)b); } } case null: { } }
    self.early = null;
    self.earlyLen = 0UL;
    self.earlyPos = 0UL;
}

} // namespace std
