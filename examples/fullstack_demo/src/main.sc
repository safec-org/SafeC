// fullstack_demo — entry point
//
// Demonstrates SafeC's Cargo-style [features] (see Package.toml): this
// file's shape changes depending on which of "frontend" / "backend" /
// "fullstack" was enabled at build time —
//
//   safeguard build                                        # fullstack (default)
//   safeguard build --features backend  --no-default-features   # JSON API only
//   safeguard build --features frontend --no-default-features   # static HTML only
//
// Frontend rendering itself lives in src/pages.scx, SafeC's JSX/TSX/RSX-
// style HTML-string templating language. The backend is a std::Router
// (std/http/router.h) on a keep-alive std::HttpServer: HTML pages, a JSON
// API, and the WASM client's files from static/ (run from this
// directory so ./static resolves).
extern int printf(const char* fmt, ...);

#include <std/mem.h>
#include <std/collections/string.h>

#ifdef SAFEC_FEATURE_FRONTEND
// Defined in src/pages.scx (transpiled to plain SafeC before safec sees it).
struct String render_home(const char* visitor_name);
struct String render_about();
struct String render_counter();
struct String render_not_found(const char* path);
#endif

#ifdef SAFEC_FEATURE_BACKEND
#include <std/http/http.h>
#include <std/http/router.h>

static struct HttpResponse api_status(&HttpRequest req, &HttpCtx ctx) {
#ifdef SAFEC_FEATURE_FRONTEND
    return std::HttpResponse::json(200, "{\"status\":\"ok\",\"service\":\"fullstack_demo\",\"mode\":\"fullstack\"}");
#else
    return std::HttpResponse::json(200, "{\"status\":\"ok\",\"service\":\"fullstack_demo\",\"mode\":\"backend-only\"}");
#endif
}

#ifdef SAFEC_FEATURE_FRONTEND
static struct HttpResponse page(struct String body) {
    struct HttpResponse r = std::HttpResponse::html(200, body.as_ptr());
    body.free();
    return r;
}

static struct HttpResponse home(&HttpRequest req, &HttpCtx ctx) { return page(render_home("web visitor")); }
static struct HttpResponse about(&HttpRequest req, &HttpCtx ctx) { return page(render_about()); }
static struct HttpResponse counter(&HttpRequest req, &HttpCtx ctx) { return page(render_counter()); }

static struct HttpResponse not_found(&HttpRequest req, &HttpCtx ctx) {
    struct String path = req.path_only();
    struct String body = render_not_found(path.as_ptr());
    path.free();
    struct HttpResponse r = std::HttpResponse::html(404, body.as_ptr());
    body.free();
    return r;
}
#endif
#endif

int main() {
#ifdef SAFEC_FEATURE_BACKEND
    struct Router app = std::Router::new();
    app.use_logger();
    app.use_gzip(1024UL);
    app.get("/api/status", api_status);
#ifdef SAFEC_FEATURE_FRONTEND
    app.get("/", home);
    app.get("/about", about);
    app.get("/counter", counter);
    app.static_dir("/static", "static");
    app.set_not_found(not_found);
    unsafe { printf("fullstack_demo: listening on http://127.0.0.1:8123 (mode=fullstack)\n"); }
#else
    unsafe { printf("fullstack_demo: listening on http://127.0.0.1:8123 (mode=backend-only)\n"); }
#endif
    struct HttpServerConfig cfg = std::HttpServerConfig::defaults((unsigned short)8123);
    int rc = app.listen(&cfg);
    app.free();
    return rc == 0 ? 0 : 1;
#else
    struct String page = render_home("static build");
    unsafe { printf("%s\n", page.as_ptr()); }
    page.free();
    return 0;
#endif
}
