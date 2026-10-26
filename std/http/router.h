// SafeC Standard Library — HTTP routing and middleware.
//
// A Router maps (method, path pattern) to handlers — patterns as in
// std/http/route.h (":id" captures a segment, "*" / "*rest" the rest of
// the path) — runs a middleware chain around them, and serves itself on an
// HttpServer (std/http/http.h):
//
//   struct HttpResponse get_user(&HttpRequest req, &HttpCtx ctx) {
//       return std::HttpResponse::text(200, ctx.param_or("id", "?"));
//   }
//
//   struct Router r = std::Router::new();
//   r.use_logger();
//   r.use_cors("*", "GET, POST", "Content-Type");
//   r.get("/users/:id", get_user);
//   r.static_dir("/assets", "./public");
//   struct HttpServerConfig cfg = std::HttpServerConfig::defaults(8080);
//   r.listen(&cfg);                                  // or r.start(&srv) + srv.stop()
//
// Matching. The most specific matching route wins (literal segments beat
// ":param", which beats "*"), whatever the registration order. A path
// that matches some route but not for this method gets 405 with an Allow
// header (HEAD falls back to a GET route; OPTIONS without a route of its
// own answers 204 with Allow); no match at all gets the not-found handler
// (a plain 404 by default).
//
// Middleware is fn(req, ctx, next): call next.run(req, ctx) to continue
// down the chain (and eventually to the route), or return a response of
// your own to short-circuit. Middleware runs in registration order for
// every request, matched or not, and sees ctx.params of the matched route.
//
// Handlers are plain function pointers (no closures), so per-application
// state lives in globals or in what the handler can reach from the
// request; ctx.state (set_state) carries one pointer-sized value for the
// whole router.
#pragma once
#include <std/http/http.h>
#include <std/http/route.h>
#include <std/http/static_files.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>

namespace std {

// Per-request context passed to route handlers and middleware.
struct HttpCtx {
    &HttpConn conn;                    // the connection (peer, keep-alive, detach)
    struct Vec<struct RouteParam> params;
    int       route_index;             // index of the matched route, -1 if none
    long long start_ns;                // when dispatch began (monotonic clock)
    ?&Erased  router_;                 // internal: the Router

    // A captured path parameter (":name" / "*name"; "*" for a bare
    // wildcard), percent-decoded, or null.
    const ?&char param(const &char name) const;
    const &char  param_or(const &char name, const &char fallback) const;
    // The pattern of the matched route ("" if none).
    const &char  pattern() const;
};

typedef fn struct HttpResponse(&HttpRequest req, &HttpCtx ctx) RouteHandler;

// The rest of the middleware chain.
struct HttpNext {
    ?&Erased router_;   // internal: the Router
    int      index_;    // internal: next middleware to run

    struct HttpResponse run(&HttpRequest req, &HttpCtx ctx);
};

typedef fn struct HttpResponse(&HttpRequest req, &HttpCtx ctx, &HttpNext next) Middleware;

struct HttpRoute_ {
    struct String method;     // "" = any method
    struct String pattern;
    RouteHandler  handler;
    int           kind;       // 0 handler, 1 static directory
    struct String dir;        // kind 1
};

struct Router {
    struct Vec<struct HttpRoute_> routes;
    struct Vec<Middleware>        middleware;
    RouteHandler                  not_found;      // set_not_found; plain 404 when unset
    int                           has_not_found;
    struct StaticOptions          static_opts;    // for static_dir routes
    // Built-in middleware settings (use_cors / use_body_limit / use_gzip).
    struct String                 cors_origin;
    struct String                 cors_methods;
    struct String                 cors_headers;
    unsigned long                 body_limit;
    unsigned long                 gzip_min;
    ?&Erased                      state;          // set_state / router_state (erased, see std/erased.h)

    static struct Router new();

    // Registers 'handler' for 'method' ("GET", ...; "*" = any) and
    // 'pattern'. Later registrations of the same method+pattern replace
    // earlier ones.
    void route(const &char method, const &char pattern, RouteHandler handler);
    void get(const &char pattern, RouteHandler handler);
    void post(const &char pattern, RouteHandler handler);
    void put(const &char pattern, RouteHandler handler);
    void patch(const &char pattern, RouteHandler handler);
    void del(const &char pattern, RouteHandler handler);       // DELETE
    void options(const &char pattern, RouteHandler handler);
    void any(const &char pattern, RouteHandler handler);
    // Serves files under 'dir' for GET/HEAD below URL 'prefix' ("/assets"
    // -> "/assets/*"), through http_serve_dir with static_opts.
    void static_dir(const &char prefix, const &char dir);
    void set_not_found(RouteHandler handler);

    // Appends a middleware to the chain.
    void use(Middleware mw);
    // Built-ins:
    // one log line per request through std/log.h (LOG_LEVEL_INFO, tag
    // "http"): "GET /path -> 200 (1.2 ms, 512 B)".
    void use_logger();
    // CORS: answers preflights (std/http/cors.h) and adds
    // Access-Control-Allow-Origin & co. to every response.
    void use_cors(const &char origin, const &char methods, const &char headers);
    // 413 for request bodies over 'max_bytes' (on top of the server's own
    // max_body_bytes).
    void use_body_limit(unsigned long max_bytes);
    // gzip-compresses compressible responses of at least 'min_bytes' when
    // the client sends Accept-Encoding: gzip (adds Vary: Accept-Encoding).
    void use_gzip(unsigned long min_bytes);
    // Recovers nothing (SafeC has no exceptions) but turns a handler's
    // status 0 response into 500 — a cheap guard for "forgot to set
    // status".
    void use_status_guard();

    // Runs the chain for one request (what the server calls; also handy
    // in tests: build an HttpRequest::new and an HttpConn by hand).
    struct HttpResponse dispatch(&HttpRequest req, &HttpConn conn);

    // Serves this router on a bound server (blocking / background). The
    // Router must outlive the server.
    int serve(&HttpServer srv);
    int start(&HttpServer srv);
    // HttpServer::bind + serve; -1 if the bind fails.
    int listen(const &HttpServerConfig cfg);

    void free();
};

// Stores/reads the router's one application pointer (shared by every
// request; synchronize it yourself if handlers mutate what it points at).
generic<T>
void router_set_state(&Router r, ?&T state) { unsafe { r.state = (?&Erased)state; } }
generic<T>
?&T router_state(&HttpCtx ctx) {
    unsafe {
        struct Router* r = (struct Router*)ctx.router_;
        return (?&T)r->state;
    }
}

// A ready-made HttpConn for calling Router::dispatch outside a server
// (fd -1, nothing detachable).
struct HttpConn http_conn_none();

} // namespace std
