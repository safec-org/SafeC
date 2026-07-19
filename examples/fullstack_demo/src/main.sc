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
// style HTML-string templating language.
extern int printf(const char* fmt, ...);

#include <std/mem.sc>
#include <std/fmt.sc>
#include <std/convert.sc>
#include <std/str.sc>
#include <std/collections/string.h>
#ifndef SAFEC_FEATURE_BACKEND
// When backend is off, nothing else in this link pulls in String's
// implementation (http.sc normally does) — bring it in here explicitly so
// main.o and pages.scx's generated .o aren't both left with unresolved
// references and nothing that defines them. (When backend IS on, http.sc's
// own '#include <std/collections/string.sc>' already covers this — adding
// it again here too would make main.o strongly define every String method
// a second time next to pages.scx's generated .o, which is exactly the
// duplicate-symbol trap scx's own auto-injected includes avoid; see
// ScxTranspiler.cpp's comment on the same issue.)
#include <std/collections/string.sc>
#endif

#ifdef SAFEC_FEATURE_FRONTEND
// Defined in src/pages.scx (transpiled to plain SafeC before safec sees it).
struct String render_home(const char* visitor_name);
struct String render_about();
struct String render_not_found(const char* path);
#endif

#ifdef SAFEC_FEATURE_BACKEND
#include <std/sched/io_nb_bsd.sc>
#include <std/http/http.h>
#include <std/http/http.sc>

struct HttpResponse handle(struct HttpRequest* req) {
    struct String body;
    const char* contentType;

#ifdef SAFEC_FEATURE_FRONTEND
    int isHome = 0;
    int isAbout = 0;
    unsafe {
        isHome  = req->path.eq_cstr("/");
        isAbout = req->path.eq_cstr("/about");
    }
    if (isHome) {
        body = render_home("web visitor");
    } else if (isAbout) {
        body = render_about();
    } else {
        const char* p;
        unsafe { p = req->path.as_ptr(); }
        body = render_not_found(p);
    }
    contentType = "text/html";
#else
    body = std::string_from(
        "{\"status\":\"ok\",\"service\":\"fullstack_demo\",\"mode\":\"backend-only\"}");
    contentType = "application/json";
#endif

    struct HttpResponse resp;
    resp.status = 200;
    resp.headers = std::string_new();
    unsafe {
        resp.headers.push("Content-Type: ");
        resp.headers.push(contentType);
        resp.headers.push("\r\n");
    }
    resp.body = body;
    return resp;
}
#endif

int main() {
#ifdef SAFEC_FEATURE_BACKEND
#ifdef SAFEC_FEATURE_FRONTEND
    unsafe { printf("fullstack_demo: listening on 8123 (mode=fullstack)\n"); }
#else
    unsafe { printf("fullstack_demo: listening on 8123 (mode=backend-only)\n"); }
#endif
    std::http_serve((unsigned short)8123, handle);
#else
    struct String page = render_home("static build");
    unsafe { printf("%s\n", page.as_ptr()); }
    page.free();
#endif
    return 0;
}
