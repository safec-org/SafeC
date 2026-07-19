// SafeC Standard Library — RPC over HTTP (gRPC-inspired)
//
// Not wire-compatible with real gRPC: real gRPC requires HTTP/2 (its
// framing and multiplexing model, plus trailer-carried status), which this
// module doesn't implement — std::http (see std/http/http.h) is HTTP/1.1
// only. What IS reused from gRPC: the length-prefixed message envelope (1
// compression-flag byte + 4-byte big-endian length + message bytes) and
// the "/Service/Method" path-routing convention, both carried inside a
// plain HTTP/1.1 POST request/response instead of an HTTP/2 stream. A real
// grpc-go/grpc-python/etc. client or server cannot talk to this one, and
// vice versa. If genuine gRPC wire compatibility is a hard requirement,
// this module isn't it — treat it as a same-shaped, HTTP/1.1-only RPC
// mechanism instead.
//
// Message payloads are opaque bytes as far as this module is concerned —
// serialize/deserialize them yourself, e.g. with std::pb_write_*/
// std::PbReader (see std/serial/protobuf.h) for a real protobuf message,
// or any other format. Builds on std::http (see http.h's backend-selection
// note — a std/sched/io_nb_*.sc backend must be included by the caller
// before this file too).
#pragma once
#include <std/collections/string.h>

namespace std {

struct RpcResult {
    int            ok;      // 1 = success, 0 = failure (see error_message)
    unsigned char* data;    // heap-owned; response bytes on success
    unsigned long  len;
    struct String  error_message; // meaningful when ok == 0

    void free();
};

// A server-side method handler: given the raw (already-unframed) request
// bytes, return the raw response bytes to frame and send back. Set
// RpcResult::ok = 0 (with error_message) to report a failure — rpc_serve
// maps that to an HTTP error response rather than a 200.
typedef fn struct RpcResult(const unsigned char* data, unsigned long len) RpcMethodHandler;

// ── Client ────────────────────────────────────────────────────────────────

// Calls 'path' (e.g. "/Greeter/SayHello") at host:port with
// 'request_data'. On a transport-level failure (connect, non-200
// response, malformed framing), the returned RpcResult has ok = 0 and
// error_message set; a handler-reported failure (server called
// RpcResult with ok=0) is surfaced the same way, with the handler's own
// error_message forwarded.
struct RpcResult rpc_call(const char* host, unsigned short port, const char* path,
                           const unsigned char* request_data, unsigned long request_len);

// ── Server ────────────────────────────────────────────────────────────────

// Registers 'handler' to be dispatched to for requests whose path exactly
// matches 'path'. Must be called (for every method the service exposes)
// before rpc_serve() — there is no dynamic registration after the accept
// loop starts.
void rpc_register(const char* path, RpcMethodHandler handler);

// Blocking accept loop (like http_serve): for each request, looks up the
// registered handler for its path, unframes the body, calls the handler,
// frames its result, and sends the response back. A path with no
// registered handler gets HTTP 404. Returns -1 immediately if the
// listening socket can't be created; does not return otherwise.
int rpc_serve(unsigned short port);

} // namespace std
