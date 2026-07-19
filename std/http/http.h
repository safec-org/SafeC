// SafeC Standard Library — HTTP/1.1 client and server (hosted, POSIX
// sockets), built on std::tcp_connect_nb/tcp_listen_nb/tcp_accept_nb (see
// std/sched/io_nb.h) for the portable BSD/Linux/Windows socket setup, then
// flipped back to blocking mode here for simple synchronous request/
// response semantics — this is not the same non-blocking-reactor model
// std/sched/reactor.h pairs those functions with.
//
// Like io_nb.h itself, this header has no backend .sc of its own — before
// including std/http/http.sc, first include whichever of
// std/sched/io_nb_bsd.sc / io_nb_linux.sc / io_nb_win32.sc matches your
// target (see io_nb.h's top comment), the same way std/ipc/uds.h's callers
// pick uds_bsd.sc/uds_linux.sc themselves. http.sc only calls io_nb.h's
// portable function signatures, never anything backend-specific.
//
// Scope: request/response bodies are Content-Length-delimited; chunked
// transfer-encoding is not supported (a chunked response's body is
// returned as-is, undecoded — check for a "Transfer-Encoding" header
// yourself via http_header_get if that matters to your use case). No TLS
// (see std/crypto/tls.h separately if you need HTTPS — this module speaks
// plain HTTP only). The server is single-threaded/one-request-at-a-time;
// combine with std::spawn per accepted connection for concurrency (see
// reference/concurrency.md).
#pragma once
#include <std/collections/string.h>

namespace std {

struct HttpResponse {
    int status;
    struct String headers; // raw header block, one "Key: value" per line,
                            // CRLF-separated, no leading/trailing blank line
    struct String body;

    void free();
};

struct HttpRequest {
    struct String method;
    struct String path;
    struct String headers; // raw header block, same shape as HttpResponse's
    struct String body;

    void free();
};

// Resolves 'host' (a numeric IPv4 address or a hostname, via
// gethostbyname) to a network-byte-order IPv4 address suitable for
// tcp_connect_nb (see std/sched/io_nb.h). Returns 1 on success, 0 if the
// address doesn't parse and hostname resolution also failed. Exposed
// separately (not just used internally by http_request) since
// std/http/websocket.h's client side needs the same resolution step.
int http_resolve_host(const char* host, unsigned int* addr_out);

// Case-insensitive header lookup in a raw "Key: value\r\nKey2: value2..."
// block (as stored in HttpResponse::headers / HttpRequest::headers).
// Returns an owned copy of the value (leading/trailing whitespace
// trimmed), or an empty String if the header isn't present.
struct String http_header_get(const char* raw_headers, const char* name);

// Performs a blocking HTTP/1.1 request. 'host' may be a hostname (resolved
// via gethostbyname) or a numeric IPv4 address. 'extra_headers' may be
// NULL/empty, or a caller-built block of additional "Key: value\r\n" lines
// (each line already CRLF-terminated) — Host, Content-Length, and
// Connection: close are added automatically and don't need to be included.
// 'body'/'body_len' may be NULL/0 for a bodyless request (GET, etc.).
// On any failure (DNS resolution, connect, or a response that doesn't
// parse as HTTP), '*ok' (if non-NULL) is set to 0.
struct HttpResponse http_request(const char* host, unsigned short port,
                                  const char* method, const char* path,
                                  const char* extra_headers,
                                  const unsigned char* body, unsigned long body_len,
                                  int* ok);

// Convenience wrappers over http_request.
struct HttpResponse http_get(const char* host, unsigned short port,
                              const char* path, int* ok);
struct HttpResponse http_post(const char* host, unsigned short port, const char* path,
                               const char* content_type,
                               const unsigned char* body, unsigned long body_len,
                               int* ok);

// A request handler: given the parsed request, return the response to
// send back.
//   struct HttpResponse my_handler(struct HttpRequest* req) { ... }
//   std::http_serve(8080, my_handler);
typedef fn struct HttpResponse(struct HttpRequest*) HttpHandler;

// Binds 'port' and serves HTTP/1.1 requests in a blocking accept loop,
// calling 'handler' for each parsed request and writing its returned
// response back to the client, until the process exits. Returns -1
// immediately if the listening socket can't be created or bound (check
// stderr/errno-equivalent via the platform's usual means); does not
// return otherwise.
int http_serve(unsigned short port, HttpHandler handler);

// Builds a complete "HTTP/1.1 <status> <reason>\r\n...\r\n\r\n<body>" text
// block ready to send to a client. 'status' must be one of the codes
// http_reason_phrase (below) recognizes (100–599 range; unrecognized
// codes get the reason phrase "Unknown").
struct String http_build_response(int status, const char* content_type,
                                   const unsigned char* body, unsigned long body_len);

// Standard reason phrase for a status code (e.g. 404 -> "Not Found"),
// "Unknown" for anything not in the common set.
const char* http_reason_phrase(int status);

} // namespace std
