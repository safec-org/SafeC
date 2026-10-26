// SafeC Standard Library — HTTP router and middleware (see router.h).
#pragma once
#include <std/http/router.h>
#include <std/http/http.h>
#include <std/http/route.h>
#include <std/http/static_files.h>
#include <std/http/cors.h>
#include <std/encoding/deflate.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>
#include <std/log.h>
#include <std/time.h>
#include <std/str.h>

namespace std {

// ── HttpCtx / HttpNext ────────────────────────────────────────────────────

const ?&char HttpCtx::param(const &char name) const { return http_route_param(&self.params, name); }

const &char HttpCtx::param_or(const &char name, const &char fallback) const {
    match (http_route_param(&self.params, name)) {
        case some(v): { return v; }
        case null: { }
    }
    return fallback;
}

const &char HttpCtx::pattern() const {
    if (self.route_index < 0) { return ""; }
    struct Router* r;
    unsafe {
        r = (struct Router*)self.router_;
        return r->routes.at((unsigned long)self.route_index).pattern.as_ptr();
    }
}

static struct HttpResponse router_final_(struct Router* r, &HttpRequest req, &HttpCtx ctx);

struct HttpResponse HttpNext::run(&HttpRequest req, &HttpCtx ctx) {
    struct Router* r;
    unsafe { r = (struct Router*)self.router_; }
    unsigned long n;
    unsafe { n = r->middleware.length(); }
    if ((unsigned long)self.index_ < n) {
        Middleware mw;
        unsafe { mw = *r->middleware.at((unsigned long)self.index_); }
        struct HttpNext next;
        next.router_ = self.router_;
        next.index_ = self.index_ + 1;
        return mw(req, ctx, &next);
    }
    return router_final_(r, req, ctx);
}

// ── Router ────────────────────────────────────────────────────────────────

struct Router Router::new() {
    struct Router r;
    r.routes = Vec::new();
    r.middleware = Vec::new();
    r.has_not_found = 0;
    r.static_opts = StaticOptions::defaults();
    r.cors_origin = String::new();
    r.cors_methods = String::new();
    r.cors_headers = String::new();
    r.body_limit = 0UL;
    r.gzip_min = 0UL;
    unsafe { r.state = (void*)0; }
    return r;
}

static void router_add_(&Router rr, const &char method, const &char pattern, RouteHandler handler,
                        int kind, const &char dir) {
    unsigned long i = 0UL;
    while (i < rr.routes.length()) {
        &HttpRoute_ rt = rr.routes.at(i);
        if (rt.kind == kind && rt.method.eq_cstr(method) && rt.pattern.eq_cstr(pattern)) {
            rt.handler = handler;
            rt.dir.clear();
            rt.dir.push(dir);
            return;
        }
        i = i + 1UL;
    }
    struct HttpRoute_ rt;
    rt.method = String::from(method);
    rt.pattern = String::from(pattern);
    rt.handler = handler;
    rt.kind = kind;
    rt.dir = String::from(dir);
    rr.routes.push(rt);
}

static struct HttpResponse router_static_placeholder_(&HttpRequest req, &HttpCtx ctx) {
    return HttpResponse::error(404);
}

void Router::route(const &char method, const &char pattern, RouteHandler handler) {
    if (str_eq(method, "*")) { router_add_(self, "", pattern, handler, 0, ""); }
    else { router_add_(self, method, pattern, handler, 0, ""); }
}
void Router::get(const &char pattern, RouteHandler handler) { self.route("GET", pattern, handler); }
void Router::post(const &char pattern, RouteHandler handler) { self.route("POST", pattern, handler); }
void Router::put(const &char pattern, RouteHandler handler) { self.route("PUT", pattern, handler); }
void Router::patch(const &char pattern, RouteHandler handler) { self.route("PATCH", pattern, handler); }
void Router::del(const &char pattern, RouteHandler handler) { self.route("DELETE", pattern, handler); }
void Router::options(const &char pattern, RouteHandler handler) { self.route("OPTIONS", pattern, handler); }
void Router::any(const &char pattern, RouteHandler handler) { self.route("*", pattern, handler); }

void Router::static_dir(const &char prefix, const &char dir) {
    struct String pat = String::from(prefix);
    while (pat.length() > 0UL && pat.ends_with("/")) { pat.truncate(pat.length() - 1UL); }
    pat.push("/*");
    router_add_(self, "GET", pat.as_ptr(), router_static_placeholder_, 1, dir);
    pat.free();
}

void Router::set_not_found(RouteHandler handler) {
    self.not_found = handler;
    self.has_not_found = 1;
}

void Router::use(Middleware mw) { self.middleware.push(mw); }

static int router_method_ok_(&HttpRoute_ rt, &HttpRequest req) {
    if (rt.method.is_empty()) { return 1; }
    if (rt.method.eq(&req.method)) { return 1; }
    return req.method.eq_cstr("HEAD") && rt.method.eq_cstr("GET");
}

static void router_allow_add_(&String allow, const &char m) {
    if (allow.contains(m)) { return; }
    if (!allow.is_empty()) { allow.push(", "); }
    allow.push(m);
}

// Matches the request; on a hit fills ctx.params / route_index. Returns
// 1 matched, 0 no path match, -1 path matched but not the method (allow
// filled).
static int router_match_(&Router rr, &HttpRequest req, &HttpCtx ctx, &String allow) {
    struct String path = req.path_only();
    long long bestScore = 0LL;
    int best = -1;
    int pathHit = 0;
    struct Vec<struct RouteParam> tmp = Vec::new();
    unsigned long i = 0UL;
    while (i < rr.routes.length()) {
        &HttpRoute_ rt = rr.routes.at(i);
        long long score = http_route_match(rt.pattern.as_ptr(), path.as_ptr(), &tmp);
        if (score > 0LL) {
            pathHit = 1;
            if (router_method_ok_(rt, req)) {
                // Exact method beats the HEAD->GET fallback and "any".
                long long s2 = score * 4LL + (rt.method.eq(&req.method) ? 2LL : (rt.method.is_empty() ? 1LL : 0LL));
                if (s2 > bestScore) {
                    bestScore = s2;
                    best = (int)i;
                    http_route_params_clear(&ctx.params);
                    unsigned long k = 0UL;
                    std::swap(&ctx.params, &tmp);   // ctx.params was just emptied
                    tmp.clear();
                }
            } else {
                if (rt.method.is_empty()) { router_allow_add_(allow, "*"); }
                else {
                    router_allow_add_(allow, rt.method.as_ptr());
                    if (rt.method.eq_cstr("GET")) { router_allow_add_(allow, "HEAD"); }
                }
            }
            http_route_params_clear(&tmp);
        }
        i = i + 1UL;
    }
    tmp.free();
    path.free();
    if (best >= 0) {
        ctx.route_index = best;
        return 1;
    }
    return pathHit ? -1 : 0;
}

static struct HttpResponse router_final_(struct Router* rp, &HttpRequest req, &HttpCtx ctx) {
    &Router rr;
    unsafe { rr = (&Router)rp; }
    if (ctx.route_index >= 0) {
        &HttpRoute_ rt = rr.routes.at((unsigned long)ctx.route_index);
        if (rt.kind == 1) {
            const &char rest = ctx.param_or("*", "");
            struct String sub = String::from("/");
            sub.push(rest);
            struct String qs = req.query_string();
            struct HttpResponse r = http_serve_dir(req, rt.dir.as_ptr(), sub.as_ptr(), &rr.static_opts);
            // A directory redirect is relative to the mount, not the root.
            if (r.status == 301) {
                struct String loc = req.path_only();
                if (!loc.ends_with("/")) { loc.push_char('/'); }
                if (!qs.is_empty()) { loc.push_char('?'); loc.push_str(&qs); }
                r.set_header("Location", loc.as_ptr());
                loc.free();
            }
            qs.free();
            sub.free();
            return r;
        }
        return rt.handler(req, ctx);
    }
    struct String allow = String::new();
    int m;
    {
        // Re-run the method check for the 405/OPTIONS answer.
        struct HttpCtx probe;
        probe.conn = ctx.conn;
        probe.params = Vec::new();
        probe.route_index = -1;
        probe.start_ns = 0LL;
        unsafe { probe.router_ = (void*)rp; }
        m = router_match_(rr, req, &probe, &allow);
        http_route_params_clear(&probe.params);
        probe.params.free();
    }
    if (m == -1) {
        if (req.method.eq_cstr("OPTIONS")) {
            router_allow_add_(&allow, "OPTIONS");
            struct HttpResponse r = HttpResponse::new(204);
            r.set_header("Allow", allow.as_ptr());
            allow.free();
            return r;
        }
        struct HttpResponse r = HttpResponse::error(405);
        r.set_header("Allow", allow.as_ptr());
        allow.free();
        return r;
    }
    allow.free();
    if (rr.has_not_found) { return rr.not_found(req, ctx); }
    return HttpResponse::error(404);
}

struct HttpResponse Router::dispatch(&HttpRequest req, &HttpConn conn) {
    struct HttpCtx ctx;
    ctx.conn = conn;
    ctx.params = Vec::new();
    ctx.route_index = -1;
    ctx.start_ns = time_mono_ns();
    unsafe { ctx.router_ = (void*)self; }
    struct String allow = String::new();
    router_match_(self, req, &ctx, &allow);
    allow.free();
    struct HttpNext next;
    unsafe { next.router_ = (void*)self; }
    next.index_ = 0;
    struct HttpResponse resp = next.run(req, &ctx);
    http_route_params_clear(&ctx.params);
    ctx.params.free();
    return resp;
}

static struct HttpResponse router_server_dispatch_(void* ctx, &HttpRequest req, &HttpConn conn) {
    struct Router* r;
    unsafe { r = (struct Router*)ctx; }
    unsafe { return r->dispatch(req, conn); }
}

int Router::serve(&HttpServer srv) {
    void* me;
    unsafe { me = (void*)self; }
    return srv.serve_raw_(router_server_dispatch_, me);
}

int Router::start(&HttpServer srv) {
    void* me;
    unsafe { me = (void*)self; }
    return srv.start_raw_(router_server_dispatch_, me);
}

int Router::listen(const &HttpServerConfig cfg) {
    int ok = 0;
    struct HttpServer srv = HttpServer::bind(cfg, &ok);
    if (!ok) { return -1; }
    return self.serve(&srv);
}

void Router::free() { self.drop(); }

void Router::drop() {
    unsigned long i = 0UL;
    while (i < self.routes.length()) {
        &HttpRoute_ rt = self.routes.at(i);
        rt.method.free();
        rt.pattern.free();
        rt.dir.free();
        i = i + 1UL;
    }
    self.routes.free();
    self.middleware.free();
    self.cors_origin.free();
    self.cors_methods.free();
    self.cors_headers.free();
}

struct HttpConn http_conn_none() {
    struct HttpConn c;
    c.fd = -1;
    c.peer_addr = 0U;
    c.peer_port = (unsigned short)0;
    c.request_index = 0UL;
    c.keep_alive = 0;
    c.detached = 0;
    unsafe { c.reader_ = (void*)0; }
    return c;
}

// ── Built-in middleware ───────────────────────────────────────────────────

static struct Router* router_of_(&HttpCtx ctx) {
    unsafe { return (struct Router*)ctx.router_; }
}

static struct HttpResponse mw_logger_(&HttpRequest req, &HttpCtx ctx, &HttpNext next) {
    struct HttpResponse resp = next.run(req, ctx);
    long long us = (time_mono_ns() - ctx.start_ns) / 1000LL;
    struct String line = String::new();
    line.push_str(&req.method);
    line.push_char(' ');
    line.push_str(&req.path);
    line.push(" -> ");
    line.push_int((long long)resp.status);
    line.push(" (");
    line.push_int(us / 1000LL);
    line.push_char('.');
    line.push_int((us / 100LL) % 10LL);
    line.push(" ms, ");
    line.push_uint((unsigned long long)resp.body.length());
    line.push(" B)");
    log_write(LOG_LEVEL_INFO, "http", line.as_ptr(), "router.sc", 0);
    line.free();
    return resp;
}

void Router::use_logger() { self.use(mw_logger_); }

static struct HttpResponse mw_cors_(&HttpRequest req, &HttpCtx ctx, &HttpNext next) {
    struct Router* r = router_of_(ctx);
    unsafe {
        if (cors_is_preflight(req)) {
            return cors_preflight_response(r->cors_origin.as_ptr(), r->cors_methods.as_ptr(),
                                           r->cors_headers.as_ptr());
        }
        struct HttpResponse resp = next.run(req, ctx);
        cors_apply_headers(&resp, r->cors_origin.as_ptr());
        return resp;
    }
}

void Router::use_cors(const &char origin, const &char methods, const &char headers) {
    self.cors_origin.clear();
    self.cors_origin.push(origin);
    self.cors_methods.clear();
    self.cors_methods.push(methods);
    self.cors_headers.clear();
    self.cors_headers.push(headers);
    self.use(mw_cors_);
}

static struct HttpResponse mw_body_limit_(&HttpRequest req, &HttpCtx ctx, &HttpNext next) {
    struct Router* r = router_of_(ctx);
    unsigned long lim;
    unsafe { lim = r->body_limit; }
    if (lim != 0UL && req.body.length() > lim) {
        ctx.conn.keep_alive = 0;
        return HttpResponse::error(413);
    }
    return next.run(req, ctx);
}

void Router::use_body_limit(unsigned long max_bytes) {
    self.body_limit = max_bytes;
    self.use(mw_body_limit_);
}

// Does Accept-Encoding allow gzip (present and not "q=0")?
static int router_accepts_gzip_(&HttpRequest req) {
    struct String ae = req.header("Accept-Encoding");
    struct String low = ae.to_lower();
    ae.free();
    int ok = 0;
    struct String parts[32];
    unsigned long n = low.split(",", &parts[0], 32UL);
    unsigned long i = 0UL;
    while (i < n) {
        struct String t = parts[i].trim();
        if (t.starts_with("gzip") || t.starts_with("*")) {
            ok = t.index_of("q=0") < 0LL || t.index_of("q=0.") >= 0LL;
            if (t.index_of("q=0.0") >= 0LL && t.index_of("q=0.0") == t.length() - 5UL) { ok = 0; }
        }
        t.free();
        unsafe { parts[i].free(); }   // the first n slots hold Strings
        i = i + 1UL;
    }
    unsafe { forget(parts); }        // each filled slot was freed above
    low.free();
    return ok;
}

static struct HttpResponse mw_gzip_(&HttpRequest req, &HttpCtx ctx, &HttpNext next) {
    struct HttpResponse resp = next.run(req, ctx);
    struct Router* r = router_of_(ctx);
    unsigned long minb;
    unsafe { minb = r->gzip_min; }
    if (resp.status != 200 || resp.body.length() < minb || resp.has_header("Content-Encoding")) { return resp; }
    struct String ct = resp.header("Content-Type");
    int comp = http_mime_is_compressible(ct.as_ptr());
    ct.free();
    if (!comp) { return resp; }
    resp.add_header("Vary", "Accept-Encoding");
    if (!router_accepts_gzip_(req)) { return resp; }
    struct Vec<unsigned char> z;
    unsafe { z = gzip_compress((const &unsigned char)resp.body.as_ptr(), resp.body.length(), DEFLATE_DEFAULT); }
    if (z.len > 0UL && z.len < resp.body.length()) {
        match (z.data) {
            case some(d): {
                resp.set_body_bytes(d, z.len);
                resp.set_header("Content-Encoding", "gzip");
                resp.remove_header("ETag");    // the representation changed
            }
            case null: { }
        }
    }
    z.free();
    return resp;
}

void Router::use_gzip(unsigned long min_bytes) {
    self.gzip_min = min_bytes;
    self.use(mw_gzip_);
}

static struct HttpResponse mw_status_guard_(&HttpRequest req, &HttpCtx ctx, &HttpNext next) {
    struct HttpResponse resp = next.run(req, ctx);
    if (resp.status < 100 || resp.status > 999) {
        resp.free();
        return HttpResponse::error(500);
    }
    return resp;
}

void Router::use_status_guard() { self.use(mw_status_guard_); }

} // namespace std
