// SafeC Standard Library — WebSocket (RFC 6455), client + server.
//
// Built on the same blocking-socket model as std/http/http.h (see that
// header's backend-selection note — a std/sched/io_nb_*.sc backend must be
// included before this file too) plus std::sha1 (RFC 6455's handshake is
// specified in terms of SHA-1, not a general recommendation — see
// std/crypto/sha1.h's own note) and std::base64_encode/base64_decode.
//
// Scope: masking (client-to-server, per the RFC's mandatory requirement)
// and unmasking are fully handled. WsConn::recv returns single frames
// (fragments and pings included, as before); WsConn::recv_message
// reassembles fragmented messages and answers pings itself. A server can
// accept on its own listening socket (ws_listen + accept) or upgrade a
// request on an HttpServer/Router connection (accept_upgrade), so HTTP and
// WebSocket endpoints share one port. wss:// runs over TLS (std/tls:
// connect_url / connect_tls on the client; accept_upgrade on an HTTPS
// HttpServer). No extensions (permessage-deflate).
#pragma once
#include <std/collections/string.h>
#include <std/http/http.h>

namespace std {

#define WS_OPCODE_CONTINUATION 0x0
#define WS_OPCODE_TEXT         0x1
#define WS_OPCODE_BINARY       0x2
#define WS_OPCODE_CLOSE        0x8
#define WS_OPCODE_PING         0x9
#define WS_OPCODE_PONG         0xA

// Opening-handshake limit WsConn::connect uses (see connect_timeout).
#define WS_DEFAULT_TIMEOUT_MS 30000

struct WsMessage {
    int                    opcode;  // one of the WS_OPCODE_* constants above
    ?&heap unsigned char   data;    // heap-owned; empty (null)/0-length for e.g. a bare ping
    unsigned long          len;

    void free();
};

struct WsConn {
    int fd;
    ?&heap TlsStream tls;   // the TLS session of a wss:// connection, else null
    int is_server; // 1 = accepted server-side (frames we send are unmasked,
                   // frames we receive must be masked); 0 = client-side
                   // (the reverse)
    // Bytes the peer sent right behind its handshake (in the same read as
    // the handshake's last line): the start of its first frame, which
    // recv() consumes before reading the socket. Heap-owned; null when
    // nothing came early. close() frees it.
    ?&heap unsigned char early;
    unsigned long        earlyLen;
    unsigned long        earlyPos;

    // ── Client ────────────────────────────────────────────────────────────
    // Connects to ws://host:port/path — performs the TCP connect, sends the
    // HTTP Upgrade request with a fresh random Sec-WebSocket-Key, and verifies
    // the server's Sec-WebSocket-Accept matches the RFC 6455 formula before
    // returning. On any failure (connect, non-101 response, key mismatch),
    // '*ok' is set to 0.
    static struct WsConn connect(const &char host, unsigned short port,
                                 const &char path, ?&int ok);

    // connect with an explicit limit: 'timeout_ms' bounds the TCP connect
    // and each send/receive of the opening handshake (a server that accepts
    // but never answers the Upgrade fails after that long). It is cleared once
    // the handshake succeeds, so recv() on the returned connection still
    // waits indefinitely for the next message. <= 0 means no limit. connect
    // uses WS_DEFAULT_TIMEOUT_MS. DNS resolution is not covered (see
    // http_request_timeout).
    static struct WsConn connect_timeout(const &char host, unsigned short port,
                                         const &char path, int timeout_ms, ?&int ok);
    // connect_timeout over TLS when 'tls' is set (wss://; 'cfg' null:
    // TlsConfig::new()'s defaults — system roots, host name verified).
    static struct WsConn connect_tls(const &char host, unsigned short port, const &char path,
                                     int tls, const ?&TlsConfig cfg, int timeout_ms, ?&int ok);

    // ── Server ────────────────────────────────────────────────────────────
    // Blocks until one client connects to 'listenFd' (from ws_listen) and
    // completes the WebSocket handshake (reads its HTTP Upgrade request,
    // validates it, sends back the 101 response with the computed
    // Sec-WebSocket-Accept). On success, fills this connection in place
    // and returns 1; on a connection that fails to complete the
    // handshake, returns 0 (the underlying fd is already closed).
    int accept(int listenFd);

    // ── Framing ───────────────────────────────────────────────────────────
    int send_text(const &char text);
    int send_binary(const &unsigned char data, unsigned long len);
    int send_ping();
    int send_pong();
    // Sends a close frame. Does not close the underlying socket — call
    // close() (below) to do both.
    int send_close();

    // Connects to a ws:// or wss:// URL ("wss://host:port/path?query"; wss
    // defaults to port 443 and TLS with TlsConfig::new()'s defaults), with
    // connect_timeout's limit. connect_url_tls takes the TLS settings.
    static struct WsConn connect_url(const &char url, int timeout_ms, ?&int ok);
    static struct WsConn connect_url_tls(const &char url, const ?&TlsConfig cfg, int timeout_ms, ?&int ok);

    // ── Server, on an HTTP server's connection ────────────────────────────
    // Completes the WebSocket handshake for a request an HttpServer/Router
    // handler received (check ws_is_upgrade(req) first): writes the 101
    // response, takes the socket over (conn.detach — the server then
    // leaves the connection alone) and returns the WsConn. *ok = 0 if the
    // request isn't a valid version-13 upgrade (the handler should then
    // answer it normally, e.g. 400/426). The handler may serve the
    // WebSocket right there (blocking that worker) or hand it to a thread.
    static struct WsConn accept_upgrade(const &HttpRequest req, &HttpConn conn, ?&int ok);

    // Reads one whole *message*: continuation frames are reassembled, pings
    // are answered with a pong automatically and pongs skipped, so 'out' is
    // a WS_OPCODE_TEXT/BINARY message or a WS_OPCODE_CLOSE frame. 0 on a
    // connection error, a protocol error, or a message over 'max_len'
    // bytes (0 = no limit).
    int recv_message(&WsMessage out, unsigned long max_len);

    // Reads one complete frame (see the file-level comment on
    // fragmentation; recv_message reassembles). Returns 1 with 'out' filled on success, 0 on a
    // connection error/EOF/malformed frame. A WS_OPCODE_CLOSE message is
    // returned like any other (not treated specially) — check for it and
    // call close() yourself.
    int recv(&WsMessage out);

    // Sends a close frame (best-effort — ignores failure, since the peer
    // may already be gone) and closes the socket.
    void close();
};

// ── Server ────────────────────────────────────────────────────────────────

// Creates a blocking listening socket on 'port'. Returns the fd, or -1 on
// failure.
int ws_listen(unsigned short port);

// 1 if 'req' asks for a WebSocket upgrade (GET with Upgrade: websocket,
// Connection: upgrade and a Sec-WebSocket-Key).
int ws_is_upgrade(const &HttpRequest req);

} // namespace std
