// SafeC Standard Library — WebSocket (RFC 6455), client + server.
//
// Built on the same blocking-socket model as std/http/http.h (see that
// header's backend-selection note — a std/sched/io_nb_*.sc backend must be
// included before this file too) plus std::sha1 (RFC 6455's handshake is
// specified in terms of SHA-1, not a general recommendation — see
// std/crypto/sha1.h's own note) and std::base64_encode/base64_decode.
//
// Scope: masking (client-to-server, per the RFC's mandatory requirement)
// and unmasking are fully handled. Message fragmentation (FIN=0
// continuation frames) is not reassembled — ws_recv treats every frame as
// a complete message; a peer that fragments large messages will need
// application-level handling on top of this (most peers, including
// browsers for reasonably-sized messages, don't fragment by default). Ping/
// pong frames are surfaced to the caller via WsMessage's opcode rather than
// answered automatically — respond with ws_send_pong yourself if the peer
// expects it.
#pragma once
#include <std/collections/string.h>

namespace std {

#define WS_OPCODE_CONTINUATION 0x0
#define WS_OPCODE_TEXT         0x1
#define WS_OPCODE_BINARY       0x2
#define WS_OPCODE_CLOSE        0x8
#define WS_OPCODE_PING         0x9
#define WS_OPCODE_PONG         0xA

struct WsConn {
    int fd;
    int is_server; // 1 = accepted server-side (frames we send are unmasked,
                   // frames we receive must be masked); 0 = client-side
                   // (the reverse)
};

struct WsMessage {
    int            opcode;  // one of the WS_OPCODE_* constants above
    unsigned char* data;    // heap-owned; NULL/0-length for e.g. a bare ping
    unsigned long  len;

    void free();
};

// ── Client ────────────────────────────────────────────────────────────────

// Connects to ws://host:port/path — performs the TCP connect, sends the
// HTTP Upgrade request with a fresh random Sec-WebSocket-Key, and verifies
// the server's Sec-WebSocket-Accept matches the RFC 6455 formula before
// returning. On any failure (connect, non-101 response, key mismatch),
// '*ok' is set to 0.
struct WsConn ws_connect(const char* host, unsigned short port,
                          const char* path, int* ok);

// ── Server ────────────────────────────────────────────────────────────────

// Creates a blocking listening socket on 'port'. Returns the fd, or -1 on
// failure.
int ws_listen(unsigned short port);

// Blocks until one client connects and completes the WebSocket handshake
// (reads its HTTP Upgrade request, validates it, sends back the 101
// response with the computed Sec-WebSocket-Accept). On success, fills
// '*out' and returns 1; on a connection that fails to complete the
// handshake, returns 0 (the underlying fd is already closed).
int ws_accept(int listenFd, struct WsConn* out);

// ── Framing ───────────────────────────────────────────────────────────────

int ws_send_text(struct WsConn* conn, const char* text);
int ws_send_binary(struct WsConn* conn, const unsigned char* data, unsigned long len);
int ws_send_ping(struct WsConn* conn);
int ws_send_pong(struct WsConn* conn);
// Sends a close frame. Does not close the underlying socket — call
// ws_close (below) to do both.
int ws_send_close(struct WsConn* conn);

// Reads one complete frame (see the file-level comment on fragmentation).
// Returns 1 with '*out' filled on success, 0 on a connection error/EOF/
// malformed frame. A WS_OPCODE_CLOSE message is returned like any other
// (not treated specially) — check for it and call ws_close yourself.
int ws_recv(struct WsConn* conn, struct WsMessage* out);

// Sends a close frame (best-effort — ignores failure, since the peer may
// already be gone) and closes the socket.
void ws_close(struct WsConn* conn);

} // namespace std
