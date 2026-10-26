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
// Scope: HTTP/1.1 (and 1.0) over plain TCP. Bodies may be Content-Length-
// delimited, chunked (decoded on both sides; see http_chunked_encode /
// HttpChunkedWriter for sending one) or, for responses, close-delimited.
// Servers support persistent connections (keep-alive), "Expect:
// 100-continue", HEAD, request-size limits and a clean stop (HttpServer);
// the client can follow redirects and decode gzip/deflate bodies
// (http_fetch). HTTPS: http_fetch & co. take https:// URLs (TLS through
// std/tls — the OS's TLS library, or SafeC's own; certificates verified
// against the system roots by default, see HttpClientOptions::tls), and
// HttpServer serves HTTPS when its config names a TlsServerConfig.
// Routing, middleware, static files, cookies, multipart forms and
// Server-Sent Events live in the sibling headers (router.h,
// static_files.h, cookie.h, multipart.h, sse.h).
//
// The legacy entry points (http_serve, http_serve_threaded) keep their
// original one-request-per-connection behavior; HttpServer is the
// configurable server, and http_serve_reactor the non-blocking one.
#pragma once
#include <std/collections/string.h>
#include <std/encoding/url.h>
#include <std/http/headers.h>
#include <std/tls/tls.h>

namespace std {

struct HttpResponse {
    int status;
    struct String headers; // raw header block, one "Key: value" per line,
                            // CRLF-separated, no leading/trailing blank line
    struct String body;

    // ── Building ──────────────────────────────────────────────────────────
    // Empty headers and body.
    static struct HttpResponse new(int status);
    // 'body' with Content-Type text/plain; charset=utf-8 / text/html;
    // charset=utf-8 / application/json.
    static struct HttpResponse text(int status, const &char body);
    static struct HttpResponse html(int status, const &char body);
    static struct HttpResponse json(int status, const &char body);
    // 'len' raw bytes (may be null/0) with the given Content-Type.
    static struct HttpResponse bytes(int status, const &char content_type,
                                     const ?&unsigned char data, unsigned long len);
    // A redirect (301, 302, 303, 307 or 308; anything else becomes 302)
    // to 'location', with a short HTML body linking to it.
    static struct HttpResponse redirect(int status, const &char location);
    // 'status' with its reason phrase as a text/plain body ("404 Not
    // Found") — the default error page.
    static struct HttpResponse error(int status);

    // Header edits on the raw block (see http_headers_set & co. below).
    void          set_header(const &char name, const &char value);
    void          add_header(const &char name, const &char value);
    unsigned long remove_header(const &char name);
    // Owned copy of the first value for 'name' (empty if absent).
    struct String header(const &char name) const;
    int           has_header(const &char name) const;
    // The header block as an HttpHeaders (caller frees).
    struct HttpHeaders header_map() const;
    // Replaces the body (Content-Type untouched).
    void          set_body(const &char body);
    void          set_body_bytes(const &unsigned char data, unsigned long len);
    void drop();

    void free() consume;   // = drop()
};

struct HttpRequest {
    struct String method;
    struct String path;    // the request-target as sent: "/a/b?x=1"
    struct String headers; // raw header block, same shape as HttpResponse's
    struct String body;

    // A request with this method and target and no headers or body — for
    // tests and for calling a Router directly.
    static struct HttpRequest new(const &char method, const &char path);

    // Owned copy of the first value for 'name' (any case; empty if absent).
    struct String header(const &char name) const;
    int           has_header(const &char name) const;
    struct HttpHeaders header_map() const;
    // 'path' up to the '?' (still percent-encoded), and the part after it
    // (empty when there is none).
    struct String path_only() const;
    struct String query_string() const;
    // The decoded query parameters.
    struct QueryParams query() const;
    // The body parsed as application/x-www-form-urlencoded (regardless of
    // Content-Type — check content_type_is() first if it matters).
    struct QueryParams form() const;
    // 1 if the Content-Type's media type (before any ';') equals 'mime',
    // ignoring case: req.content_type_is("application/json").
    int           content_type_is(const &char mime) const;
    // The value of cookie 'name' from the Cookie header(s); empty if unset.
    struct String cookie(const &char name) const;
    struct HttpRequest clone() const;
    void drop();

    void free() consume;   // = drop()
};

// A connected, blocking TCP socket — optionally with TLS on it: the
// connection http_request and WsConn talk over, for your own protocols on a
// socket from tcp_connect_timeout / tcp_accept_nb (flipped to blocking with
// fd_set_blocking — see std/sched/io_nb.h) or http_open_stream. It conforms
// to io_traits.h's Stream (Reader + Writer), so read_exact / write_all /
// copy_stream and io_buf.h's BufReader / BufWriter work on it (http and
// WsConn read their header blocks through a BufReader<TcpStream>). With
// 'tls' set, reads and writes are the TLS stream's (application data) and a
// TLS_WANT_* (a socket timeout) reads as -1.
// Writes never raise SIGPIPE: a peer that has gone away is a -1 result.
struct TcpStream {
    int fd;
    ?&heap TlsStream tls;    // null: plain TCP

    // Wraps an open socket fd. A TcpStream is a handle, not an owner (std
    // makes short-lived streams over fds that something else closes), so
    // it is not a Drop type: whoever owns the fd calls close() once.
    static struct TcpStream from_fd(int fd);
    // Wraps an established TLS stream (heap-allocated: the TcpStream owns
    // it from then on, and close() sends close_notify, frees it and closes
    // its socket).
    static struct TcpStream over_tls(&heap TlsStream t);
    int is_tls() const;

    // At most n bytes; the count read, 0 once the peer closed, -1 on error.
    long long read(&unsigned char buf, unsigned long n);
    // At most n bytes; the count written, -1 on error.
    long long write(const &unsigned char buf, unsigned long n);
    // Closes the socket. 0 on success.
    int close();
};

// Resolves 'host' (a numeric IPv4 address or a hostname, via
// getaddrinfo — safe to call from several threads at once) to a network-byte-order IPv4 address suitable for
// tcp_connect_nb (see std/sched/io_nb.h). Returns 1 on success, 0 if the
// address doesn't parse and hostname resolution also failed. Exposed
// separately (not just used internally by http_request) since
// std/http/websocket.h's client side needs the same resolution step.
int http_resolve_host(const &char host, &unsigned int addr_out);

// Resolves and connects to host:port and, when 'tls' is set, runs a TLS
// client handshake over the socket ('cfg' null: TlsConfig::new()'s
// defaults — the system roots, host name verified — offering ALPN
// "http/1.1"): the connection http_fetch, WsConn and SseClient make for
// https:// and wss:// URLs. 'connect_timeout_ms' bounds the TCP connect,
// 'io_timeout_ms' each send/receive afterwards (the TLS handshake is bounded
// by the connect timeout when io_timeout_ms is <= 0). On failure the result
// has fd -1 and *err (if non-null) says why (code TLS_ERR_IO for DNS/
// connect failures).
struct TcpStream http_open_stream(const &char host, unsigned short port, int tls,
                                  const ?&TlsConfig cfg, int connect_timeout_ms,
                                  int io_timeout_ms, ?&TlsError err);

// Case-insensitive header lookup in a raw "Key: value\r\nKey2: value2..."
// block (as stored in HttpResponse::headers / HttpRequest::headers).
// Returns an owned copy of the value (leading/trailing whitespace
// trimmed), or an empty String if the header isn't present.
struct String http_header_get(const &char raw_headers, const &char name);

// Edits on a raw header block (HttpRequest/HttpResponse::headers):
// set replaces every line with this name by one "Name: value" line (at
// the first one's position, or appended); add appends a line, keeping any
// existing ones; remove deletes every line with this name and returns how
// many there were. Names compare case-insensitively.
void          http_headers_set(&String raw, const &char name, const &char value);
void          http_headers_add(&String raw, const &char name, const &char value);
unsigned long http_headers_remove(&String raw, const &char name);
// 1 if header 'name' holds a comma-separated list containing 'token'
// (case-insensitive) — e.g. http_header_has_token(h, "Connection", "close").
int           http_header_has_token(const &char raw_headers, const &char name, const &char token);

// Performs a blocking HTTP/1.1 request. 'host' may be a hostname (resolved
// via getaddrinfo) or a numeric IPv4 address. 'extra_headers' may be
// NULL/empty, or a caller-built block of additional "Key: value\r\n" lines
// (each line already CRLF-terminated) — Host, Content-Length, and
// Connection: close are added automatically and don't need to be included.
// 'body'/'body_len' may be NULL/0 for a bodyless request (GET, etc.).
// On any failure (DNS resolution, connect, or a response that doesn't
// parse as HTTP), '*ok' (if non-NULL) is set to 0. The TCP connect gives
// up after HTTP_DEFAULT_CONNECT_TIMEOUT_MS; reading the response has no
// time limit (use http_request_timeout to bound it).
struct HttpResponse http_request(const &char host, unsigned short port,
                                  const &char method, const &char path,
                                  const ?&char extra_headers,
                                  const ?&unsigned char body, unsigned long body_len,
                                  ?&int ok);

#define HTTP_DEFAULT_CONNECT_TIMEOUT_MS 30000

// http_request with explicit limits. 'connect_timeout_ms' bounds the TCP
// connect (see std::tcp_connect_timeout); 'io_timeout_ms' bounds each
// individual send/receive on the socket (SO_SNDTIMEO/SO_RCVTIMEO), so a
// server that accepts but never answers fails the request after that long
// rather than hanging; it is not a limit on the whole transfer. Either
// value <= 0 means no limit. DNS resolution (getaddrinfo) is not
// covered by either: it is bounded only by the system resolver's own
// timeout/attempts settings.
struct HttpResponse http_request_timeout(const &char host, unsigned short port,
                                          const &char method, const &char path,
                                          const ?&char extra_headers,
                                          const ?&unsigned char body, unsigned long body_len,
                                          int connect_timeout_ms, int io_timeout_ms,
                                          ?&int ok);

// Convenience wrappers over http_request.
struct HttpResponse http_get(const &char host, unsigned short port,
                              const &char path, ?&int ok);
struct HttpResponse http_post(const &char host, unsigned short port, const &char path,
                               const &char content_type,
                               const ?&unsigned char body, unsigned long body_len,
                               ?&int ok);

// A request handler: given the parsed request, return the response to
// send back.
//   struct HttpResponse my_handler(&HttpRequest req) { ... }
//   std::http_serve(8080, my_handler);
typedef fn struct HttpResponse(&HttpRequest) HttpHandler;

// Binds 'port' and serves HTTP/1.1 requests in a blocking accept loop,
// calling 'handler' for each parsed request and writing its returned
// response back to the client, until the process exits. Returns -1
// immediately if the listening socket can't be created or bound (check
// stderr/errno-equivalent via the platform's usual means); does not
// return otherwise.
int http_serve(unsigned short port, HttpHandler handler);

// Same contract as http_serve(), but 'numThreads' worker threads (see
// std/thread.h) all call accept() on the same listening socket instead
// of one — the OS kernel fairly and safely distributes incoming
// connections across whichever threads are currently blocked in
// accept(), so this needs no manual work queue/condvar dispatch (the
// standard "thread-per-listener" pattern nginx's worker model and
// classic prefork Apache both use). Each connection is still handled
// entirely on the thread that accepted it, one at a time — 'handler'
// itself is called concurrently from up to 'numThreads' OS threads, so
// it (and anything it touches — shared globals, etc.) must be safe to
// call that way. Returns -1 immediately on setup failure (bad
// 'numThreads', listen/bind failure, or a worker thread failing to
// start); does not return otherwise.
int http_serve_threaded(unsigned short port, HttpHandler handler, int numThreads);

// Same contract as http_serve()/http_serve_threaded(), but each of
// 'numThreads' OS threads runs a non-blocking std::Reactor/TaskScheduler
// event loop (see std/sched/reactor.h) instead of blocking one connection
// at a time — many concurrent connections per thread rather than one,
// the same shape async runtimes elsewhere use for higher throughput
// under concurrent load. Requires a reactor backend
// (reactor_kqueue.sc/reactor_epoll.sc/reactor_win32.sc) to already be
// included by the caller, same convention as io_nb.h's own backend
// selection. Returns -1 immediately on setup failure; does not return
// otherwise.
int http_serve_reactor(unsigned short port, HttpHandler handler, int numThreads);

// Builds a complete "HTTP/1.1 <status> <reason>\r\n...\r\n\r\n<body>" text
// block ready to send to a client. 'status' must be one of the codes
// http_reason_phrase (below) recognizes (100–599 range; unrecognized
// codes get the reason phrase "Unknown").
struct String http_build_response(int status, const &char content_type,
                                   const ?&unsigned char body, unsigned long body_len);

// Standard reason phrase for a status code (e.g. 404 -> "Not Found"),
// "Unknown" for anything not in the IANA registry's common set.
const &char http_reason_phrase(int status);

// Status classes: 1xx, 2xx, 3xx (a redirect: 301/302/303/307/308), 4xx,
// 5xx.
int http_status_is_informational(int status);
int http_status_is_success(int status);
int http_status_is_redirect(int status);
int http_status_is_client_error(int status);
int http_status_is_server_error(int status);

// ── Chunked transfer coding (RFC 9112 §7.1) ───────────────────────────────

// Appends 'len' bytes as one chunk ("<hex len>\r\n<data>\r\n") to 'out';
// len 0 appends nothing (the terminator is http_chunked_end's job).
void http_chunked_encode(&String out, const ?&unsigned char data, unsigned long len);
// Appends the last-chunk + empty trailer ("0\r\n\r\n").
void http_chunked_end(&String out);
// Decodes a complete chunked body held in memory (chunk extensions and
// trailer fields are skipped) into 'out'. *consumed (if non-null) gets the
// number of input bytes used. 0 on a malformed or truncated encoding.
int  http_chunked_decode(const &unsigned char data, unsigned long len, &String out,
                         ?&unsigned long consumed);

// ── Client: URLs, redirects, decompression ────────────────────────────────

struct HttpClientOptions {
    int           max_redirects;      // redirects followed (0: none); default 5
    int           connect_timeout_ms; // default HTTP_DEFAULT_CONNECT_TIMEOUT_MS
    int           io_timeout_ms;      // per send/receive; 0 = none (default 30000)
    int           decompress;         // send Accept-Encoding: gzip, deflate and
                                      // decode the body (default 1)
    unsigned long max_body_bytes;     // larger response bodies fail; 0 = no limit
                                      // (default 64 MiB)
    ?&TlsConfig   tls;                // https:// settings (trust, client cert,
                                      // backend); null = TlsConfig::new()'s defaults
    ?&TlsError    tls_error;          // if set, receives why a TLS connection failed

    static struct HttpClientOptions defaults();
};

// A request to an absolute http:// or https:// URL (default ports 80 and
// 443; https through std/tls with opts.tls's settings, the server's
// certificate verified against the system roots by default): resolves and
// connects to its host and port, sends method + request-target, and follows
// redirects
// (Location resolved against the current URL; 301/302/303 turn a non-GET/
// HEAD request into a bodyless GET, 307/308 resend it unchanged) up to
// opts.max_redirects. 'extra_headers' as for http_request. A gzip/deflate
// Content-Encoding is decoded when opts.decompress is set (the
// Content-Encoding header is then removed from the response). 'opts' may
// be null (defaults()). *ok = 0 on a bad or non-http(s) URL, transport or
// TLS failure (opts.tls_error says which), malformed
// response, too many redirects or a body over the limit; the status of a
// completed exchange is reported even when it is 4xx/5xx (ok = 1).
// 'final_url', if non-null, receives the URL the response came from.
struct HttpResponse http_fetch(const &char method, const &char url,
                               const ?&char extra_headers,
                               const ?&unsigned char body, unsigned long body_len,
                               const ?&HttpClientOptions opts, ?&String final_url,
                               ?&int ok);
// http_fetch conveniences with default options.
struct HttpResponse http_get_url(const &char url, ?&int ok);
struct HttpResponse http_post_url(const &char url, const &char content_type,
                                  const ?&unsigned char body, unsigned long body_len, ?&int ok);

// Decodes resp.body according to its Content-Encoding (gzip, x-gzip,
// deflate — zlib or raw — or identity) in place and drops the header.
// 0 (body untouched) on an unknown coding or corrupt data.
int http_decode_body(&HttpResponse resp, unsigned long max_out);

// ── Server: configuration, connections, lifecycle ─────────────────────────

struct HttpServerConfig {
    unsigned short port;            // 0 = any free port (see HttpServer::port)
    int           threads;          // worker threads, each serving one connection at a time
    int           keep_alive;       // persistent connections (HTTP/1.1 default, 1.0 opt-in)
    int           idle_timeout_ms;  // how long a kept-alive connection may sit idle
    int           io_timeout_ms;    // per send/receive while a request is in flight; 0 = none
    int           max_requests_per_conn; // 0 = unlimited
    unsigned long max_header_bytes; // request line + headers; larger gets 431
    unsigned long max_body_bytes;   // Content-Length or chunked total; larger gets 413
    // HTTPS: when set, bind() builds a TlsAcceptor from it (bind fails if
    // it can't — try TlsAcceptor::new yourself for the reason) and every
    // connection starts with a TLS handshake (bounded by io_timeout_ms).
    // Only needed during bind(). null = plain HTTP (the default).
    ?&TlsServerConfig tls;

    // port, 8 threads, keep-alive on (5 s idle), 30 s I/O timeout, 1000
    // requests per connection, 64 KiB of headers, 16 MiB of body.
    static struct HttpServerConfig defaults(unsigned short port);
};

// The connection a request arrived on, as router handlers and middleware
// see it (std/http/router.h). A handler that takes the socket over —
// Server-Sent Events, a WebSocket upgrade, a hand-written streaming
// response — calls detach(): the server then neither writes the returned
// response nor closes the socket.
struct HttpConn {
    int            fd;
    unsigned int   peer_addr;      // IPv4, network byte order (0 if unknown)
    unsigned short peer_port;
    unsigned long  request_index;  // 0 for the connection's first request
    int            keep_alive;     // the server will keep the connection open after
                                   // this response; set to 0 to close it instead
    int            detached;
    void*          reader_;        // internal: the connection's BufReader<TcpStream>
    ?&heap TlsStream tls;          // the TLS session on an HTTPS server's connection, else null

    // Takes over the (blocking) socket. Bytes the client already sent past
    // this request (pipelined data, a WebSocket frame right behind the
    // handshake) are appended to 'early' if it is non-null.
    struct TcpStream detach(?&String early);
    // "a.b.c.d:port".
    struct String peer_string() const;
};

// Dispatch entry the server loops call (std/http/router.h's Router uses
// it); a plain HttpHandler is adapted to it. Internal: ctx is the router.
typedef fn struct HttpResponse(void* ctx, &HttpRequest req, &HttpConn conn) HttpDispatchFn_;

// A bound, configurable blocking server: threads × one connection at a
// time each, with keep-alive. bind() opens the socket; serve() (blocking)
// or start() (background thread) runs it; stop() ends it from any thread.
//
//   struct HttpServer srv = std::HttpServer::bind(&cfg, &ok);
//   srv.start(handler);              // returns at once; srv.port is the bound port
//   ...
//   srv.stop();                      // in-flight requests finish, then the workers exit
struct HttpServer {
    unsigned short port;                // the bound port (useful with config port 0)
    void*          shared_;             // internal server state

    static struct HttpServer bind(const &HttpServerConfig cfg, ?&int ok);
    // Runs the worker threads and blocks until stop(). 0 after a stop, -1
    // if the server isn't bound or a worker thread can't start.
    int  serve(HttpHandler handler);
    // serve() on a background thread; returns 0 once the workers run.
    int  start(HttpHandler handler);
    int  serve_raw_(HttpDispatchFn_ dispatch, void* ctx);
    int  start_raw_(HttpDispatchFn_ dispatch, void* ctx);
    // Stops accepting, wakes idle keep-alive connections, waits for the
    // workers to finish their current request, and closes the listening
    // socket. Safe to call from a handler's thread too (then it doesn't
    // wait for that worker). Idempotent.
    void stop();
    int  is_running() const;
};

// HttpServer::bind + serve in one call; returns only on a setup failure
// (-1) — there is no handle to stop it with.
int http_serve_config(const &HttpServerConfig cfg, HttpHandler handler);

// Streams a response of unknown length over a detached connection with
// chunked transfer coding: begin() writes the status line and headers
// (plus Transfer-Encoding: chunked), write() sends a chunk, finish() the
// last chunk and closes the socket.
struct HttpChunkedWriter {
    struct TcpStream s;
    int              ok;      // 0 once any write failed

    static struct HttpChunkedWriter begin(&HttpConn conn, int status, const ?&char headers);
    int write(const &unsigned char data, unsigned long len);
    int write_str(const &char text);
    int finish();
};

} // namespace std
