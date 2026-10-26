// std/http's web-stack layer: HttpHeaders, request/response helpers,
// dates, cookies, chunked coding, route patterns, Router + middleware,
// static files, multipart, Server-Sent Events, WebSocket upgrades, the
// keep-alive HttpServer and the URL client (redirects, gzip) — over real
// loopback connections where a network is involved.
#include <std/test/test.h>
#include <std/mem.h>
#include <std/str.h>
#include <std/thread.h>
#include <std/time.h>
#include <std/io_file.h>
#include <std/io_traits.h>
#include <std/io_buf.h>
#include <std/sched/io_nb.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>
#include <std/encoding/url.h>
#include <std/encoding/deflate.h>
#include <std/http/http.h>
#include <std/http/headers.h>
#include <std/http/date.h>
#include <std/http/cookie.h>
#include <std/http/route.h>
#include <std/http/router.h>
#include <std/http/static_files.h>
#include <std/http/multipart.h>
#include <std/http/sse.h>
#include <std/http/websocket.h>

// ── Pure helpers ───────────────────────────────────────────────────────────

void t_headers() {
    struct HttpHeaders h = HttpHeaders::parse("Content-Type: text/html\r\nSet-Cookie: a=1\r\nX-Folded: one\r\n  two\r\nset-cookie: b=2\r\nbad line\r\nConnection: keep-alive, Upgrade");
    ASSERT_EQ(h.length(), 5);
    ASSERT_STR_EQ(h.get_or("content-type", ""), "text/html");
    ASSERT_EQ(h.count("SET-COOKIE"), 2);
    match (h.get_nth("set-cookie", 1UL)) { case some(v): { ASSERT_STR_EQ(v, "b=2"); } case null: { ASSERT_TRUE(0); } }
    ASSERT_STR_EQ(h.get_or("x-folded", ""), "one two");
    struct String j = h.get_joined("Set-Cookie");
    ASSERT_STR_EQ(j.as_ptr(), "a=1, b=2");
    j.free();
    ASSERT_TRUE(h.contains_token("connection", "upgrade"));
    ASSERT_FALSE(h.contains_token("connection", "close"));
    h.set("Set-Cookie", "c=3");
    ASSERT_EQ(h.count("set-cookie"), 1);
    ASSERT_EQ(h.remove("x-folded"), 1);
    h.append("X-New", "v");
    struct String raw = h.to_raw();
    ASSERT_STR_EQ(raw.as_ptr(), "Content-Type: text/html\r\nSet-Cookie: c=3\r\nConnection: keep-alive, Upgrade\r\nX-New: v");
    raw.free();
    struct HttpHeaders c = h.clone();
    ASSERT_EQ(c.length(), 4);
    c.free();
    h.free();

    // Raw-block edits.
    struct String blk = String::from("A: 1\r\nB: 2\r\na: 3");
    http_headers_set(&blk, "a", "9");
    ASSERT_STR_EQ(blk.as_ptr(), "a: 9\r\nB: 2");
    http_headers_add(&blk, "C", "x");
    ASSERT_EQ(http_headers_remove(&blk, "b"), 1);
    ASSERT_STR_EQ(blk.as_ptr(), "a: 9\r\nC: x");
    ASSERT_TRUE(http_header_has_token(blk.as_ptr(), "c", "X"));
    blk.free();

    struct String p = http_header_param("multipart/form-data; charset=utf-8; boundary=\"ab\\\"c\"", "Boundary");
    ASSERT_STR_EQ(p.as_ptr(), "ab\"c");
    p.free();
    p = http_header_param("form-data; name=field1; filename=\"a b.txt\"", "filename");
    ASSERT_STR_EQ(p.as_ptr(), "a b.txt");
    p.free();
    p = http_header_main_value(" Text/HTML ; charset=x");
    ASSERT_STR_EQ(p.as_ptr(), "text/html");
    p.free();
}

void t_request_response_helpers() {
    struct HttpRequest req = HttpRequest::new("POST", "/search/a%20b?q=safe+c&page=2#frag");
    http_headers_add(&req.headers, "Content-Type", "application/x-www-form-urlencoded; charset=utf-8");
    http_headers_add(&req.headers, "Cookie", "sid=abc; theme=\"dark\"");
    req.body.push("name=Ada+L&lang=sc");
    struct String po = req.path_only();
    ASSERT_STR_EQ(po.as_ptr(), "/search/a%20b");
    po.free();
    struct String qs = req.query_string();
    ASSERT_STR_EQ(qs.as_ptr(), "q=safe+c&page=2");
    qs.free();
    struct QueryParams q = req.query();
    ASSERT_STR_EQ(q.get_or("q", ""), "safe c");
    q.free();
    struct QueryParams f = req.form();
    ASSERT_STR_EQ(f.get_or("name", ""), "Ada L");
    f.free();
    ASSERT_TRUE(req.content_type_is("application/x-www-form-urlencoded"));
    ASSERT_FALSE(req.content_type_is("application/json"));
    struct String ck = req.cookie("theme");
    ASSERT_STR_EQ(ck.as_ptr(), "dark");
    ck.free();
    ck = req.cookie("nope");
    ASSERT_STR_EQ(ck.as_ptr(), "");
    ck.free();
    struct HttpRequest copy = req.clone();
    ASSERT_TRUE(copy.body.eq(&req.body));
    copy.free();
    req.free();

    struct HttpResponse r = HttpResponse::json(201, "{\"ok\":true}");
    ASSERT_EQ(r.status, 201);
    struct String ct = r.header("content-type");
    ASSERT_STR_EQ(ct.as_ptr(), "application/json");
    ct.free();
    r.set_header("X-A", "1");
    r.add_header("X-A", "2");
    ASSERT_EQ(r.remove_header("x-a"), 2);
    ASSERT_FALSE(r.has_header("X-A"));
    r.free();
    r = HttpResponse::redirect(307, "/next?a=<b>");
    ASSERT_EQ(r.status, 307);
    struct String loc = r.header("Location");
    ASSERT_STR_EQ(loc.as_ptr(), "/next?a=<b>");
    loc.free();
    ASSERT_TRUE(r.body.contains("&lt;b&gt;"));
    r.free();
    r = HttpResponse::redirect(200, "/x");
    ASSERT_EQ(r.status, 302);
    r.free();
    r = HttpResponse::error(404);
    ASSERT_STR_EQ(r.body.as_ptr(), "404 Not Found\n");
    r.free();
    ASSERT_STR_EQ(http_reason_phrase(413), "Content Too Large");
    ASSERT_STR_EQ(http_reason_phrase(308), "Permanent Redirect");
    ASSERT_TRUE(http_status_is_redirect(303));
    ASSERT_FALSE(http_status_is_redirect(304));
    ASSERT_TRUE(http_status_is_client_error(404));
    ASSERT_TRUE(http_status_is_server_error(503));
    ASSERT_TRUE(http_status_is_success(204));
}

void t_dates() {
    struct String d = http_date_format(784111777LL);
    ASSERT_STR_EQ(d.as_ptr(), "Sun, 06 Nov 1994 08:49:37 GMT");
    d.free();
    d = http_date_format(0LL);
    ASSERT_STR_EQ(d.as_ptr(), "Thu, 01 Jan 1970 00:00:00 GMT");
    d.free();
    d = http_date_format(951782400LL);   // leap day 2000
    ASSERT_STR_EQ(d.as_ptr(), "Tue, 29 Feb 2000 00:00:00 GMT");
    d.free();
    int ok = 0;
    ASSERT_EQ(http_date_parse("Sun, 06 Nov 1994 08:49:37 GMT", &ok), 784111777LL);
    ASSERT_TRUE(ok);
    ASSERT_EQ(http_date_parse("Sunday, 06-Nov-94 08:49:37 GMT", &ok), 784111777LL);
    ASSERT_TRUE(ok);
    ASSERT_EQ(http_date_parse("Sun Nov  6 08:49:37 1994", &ok), 784111777LL);
    ASSERT_TRUE(ok);
    http_date_parse("Sun, 06 Nov 1994 08:49 GMT", &ok);
    ASSERT_FALSE(ok);
    http_date_parse("yesterday", &ok);
    ASSERT_FALSE(ok);
    long long y = 0LL;
    int m = 0;
    int dd = 0;
    http_civil_from_days(http_days_from_civil(2024LL, 12, 31), &y, &m, &dd);
    ASSERT_EQ(y, 2024);
    ASSERT_EQ(m, 12);
    ASSERT_EQ(dd, 31);
    struct String now = http_date_now();
    ASSERT_TRUE(now.ends_with(" GMT"));
    now.free();
}

void t_cookies() {
    struct CookieList l = CookieList::parse("a=1; b=\"two\"; junk; c = 3 ");
    ASSERT_EQ(l.length(), 3);
    match (l.get("b")) { case some(v): { ASSERT_STR_EQ(v, "two"); } case null: { ASSERT_TRUE(0); } }
    match (l.get("c")) { case some(v): { ASSERT_STR_EQ(v, "3"); } case null: { ASSERT_TRUE(0); } }
    ASSERT_TRUE(l.get("junk").is_null());
    l.free();

    struct HttpRequest req = HttpRequest::new("GET", "/");
    http_headers_add(&req.headers, "Cookie", "x=1");
    http_headers_add(&req.headers, "cookie", "y=2");
    l = CookieList::from_request(&req);
    ASSERT_EQ(l.length(), 2);
    l.free();
    req.free();

    struct SetCookie c = SetCookie::new("sid", "abc123");
    c.path.push("/");
    c.domain.push("example.com");
    c.max_age = 3600LL;
    c.expires = 784111777LL;
    c.http_only = 1;
    c.same_site = COOKIE_SAMESITE_LAX;
    struct String s = c.to_string();
    ASSERT_STR_EQ(s.as_ptr(), "sid=abc123; Path=/; Domain=example.com; Max-Age=3600; Expires=Sun, 06 Nov 1994 08:49:37 GMT; HttpOnly; SameSite=Lax");
    int ok = 0;
    struct SetCookie back = SetCookie::parse(s.as_ptr(), &ok);
    ASSERT_TRUE(ok);
    ASSERT_STR_EQ(back.name.as_ptr(), "sid");
    ASSERT_STR_EQ(back.domain.as_ptr(), "example.com");
    ASSERT_EQ(back.max_age, 3600);
    ASSERT_EQ(back.expires, 784111777LL);
    ASSERT_TRUE(back.http_only);
    ASSERT_FALSE(back.secure);
    ASSERT_EQ(back.same_site, COOKIE_SAMESITE_LAX);
    back.free();
    s.free();
    c.free();
    c = SetCookie::new("t", "v");
    c.same_site = COOKIE_SAMESITE_NONE;
    s = c.to_string();
    ASSERT_STR_EQ(s.as_ptr(), "t=v; Secure; SameSite=None");   // None requires Secure
    s.free();
    c.free();
    struct SetCookie bad = SetCookie::parse("; Path=/", &ok);
    ASSERT_FALSE(ok);
    bad.free();

    struct HttpResponse r = HttpResponse::new(200);
    struct SetCookie del = SetCookie::removal("sid", "/");
    http_set_cookie(&r, &del);
    del.free();
    struct String v = r.header("Set-Cookie");
    ASSERT_STR_EQ(v.as_ptr(), "sid=; Path=/; Max-Age=0; Expires=Thu, 01 Jan 1970 00:00:01 GMT");
    v.free();
    r.free();
}

void t_chunked_coding() {
    struct String enc = String::new();
    unsafe {
        http_chunked_encode(&enc, str_bytes("hello "));
        http_chunked_encode(&enc, str_bytes(""));
        http_chunked_encode(&enc, str_bytes("world, this is longer"));
    }
    http_chunked_end(&enc);
    ASSERT_STR_EQ(enc.as_ptr(), "6\r\nhello \r\n15\r\nworld, this is longer\r\n0\r\n\r\n");
    struct String dec = String::new();
    unsigned long used = 0UL;
    unsafe { ASSERT_TRUE(http_chunked_decode(enc.as_bytes(), &dec, &used)); }
    ASSERT_STR_EQ(dec.as_ptr(), "hello world, this is longer");
    ASSERT_EQ(used, enc.length());
    dec.free();
    enc.free();
    // Extensions and trailers are skipped; truncation fails.
    const &char ext = "3;name=v\r\nabc\r\n0\r\nX-Trailer: 1\r\n\r\nextra";
    dec = String::new();
    unsafe { ASSERT_TRUE(http_chunked_decode(str_bytes(ext), &dec, &used)); }
    ASSERT_STR_EQ(dec.as_ptr(), "abc");
    ASSERT_EQ(used, str_len(ext) - 5UL);
    dec.clear();
    unsafe { ASSERT_FALSE(http_chunked_decode(str_bytes("5\r\nab"), &dec, null)); }
    unsafe { ASSERT_FALSE(http_chunked_decode(str_bytes("zz\r\n"), &dec, null)); }
    dec.free();
}

void t_route_patterns() {
    struct Vec<RouteParam> p = Vec::new();
    ASSERT_TRUE(http_route_match("/", "/", &p) > 0LL);
    ASSERT_TRUE(http_route_match("/users/:id", "/users/42/", &p) > 0LL);
    match (http_route_param(&p, "id")) { case some(v): { ASSERT_STR_EQ(v, "42"); } case null: { ASSERT_TRUE(0); } }
    http_route_params_clear(&p);
    ASSERT_EQ(http_route_match("/users/:id", "/users", &p), 0);
    ASSERT_EQ(http_route_match("/users/:id", "/users/1/x", &p), 0);
    ASSERT_EQ(p.length(), 0);
    ASSERT_TRUE(http_route_match("/files/*path", "/files/a/b%20c.txt", &p) > 0LL);
    match (http_route_param(&p, "path")) { case some(v): { ASSERT_STR_EQ(v, "a/b c.txt"); } case null: { ASSERT_TRUE(0); } }
    http_route_params_clear(&p);
    ASSERT_TRUE(http_route_match("/files/*", "/files", &p) > 0LL);
    match (http_route_param(&p, "*")) { case some(v): { ASSERT_STR_EQ(v, ""); } case null: { ASSERT_TRUE(0); } }
    http_route_params_clear(&p);
    long long lit = http_route_match("/users/me", "/users/me", &p);
    http_route_params_clear(&p);
    long long par = http_route_match("/users/:id", "/users/me", &p);
    http_route_params_clear(&p);
    long long wild = http_route_match("/users/*", "/users/me", &p);
    http_route_params_clear(&p);
    ASSERT_TRUE(lit > par);
    ASSERT_TRUE(par > wild);
    ASSERT_TRUE(wild > 0LL);
    p.free();
}

// ── Router (dispatched directly, no network) ───────────────────────────────

static struct String g_trace;

static struct HttpResponse h_user(&HttpRequest req, &HttpCtx ctx) {
    struct String b = String::from("user ");
    b.push(ctx.param_or("id", "?"));
    struct HttpResponse r = HttpResponse::text(200, b.as_ptr());
    b.free();
    return r;
}
static struct HttpResponse h_me(&HttpRequest req, &HttpCtx ctx) { return HttpResponse::text(200, "me"); }
static struct HttpResponse h_create(&HttpRequest req, &HttpCtx ctx) {
    return HttpResponse::json(201, req.body.as_ptr());
}
static struct HttpResponse h_big(&HttpRequest req, &HttpCtx ctx) {
    struct String s = String::repeat_str("compress me please ", 200UL);
    struct HttpResponse r = HttpResponse::text(200, s.as_ptr());
    s.free();
    return r;
}
static struct HttpResponse h_missing(&HttpRequest req, &HttpCtx ctx) { return HttpResponse::text(404, "custom 404"); }
static struct HttpResponse h_zero(&HttpRequest req, &HttpCtx ctx) { return HttpResponse::new(0); }

static struct HttpResponse mw_trace_a(&HttpRequest req, &HttpCtx ctx, &HttpNext next) {
    g_trace.push("a>");
    struct HttpResponse r = next.run(req, ctx);
    g_trace.push("<a");
    r.add_header("X-Mw", "a");
    return r;
}
static struct HttpResponse mw_trace_b(&HttpRequest req, &HttpCtx ctx, &HttpNext next) {
    g_trace.push("b>");
    if (req.has_header("X-Block")) { return HttpResponse::text(403, "blocked"); }
    struct HttpResponse r = next.run(req, ctx);
    g_trace.push("<b");
    return r;
}

static struct HttpResponse call(&Router r, const &char method, const &char path, const ?&char headers, const ?&char body) {
    struct HttpRequest req = HttpRequest::new(method, path);
    match (headers) { case some(h): { req.headers.push(h); } case null: { } }
    match (body) { case some(b): { req.body.push(b); } case null: { } }
    struct HttpConn conn = http_conn_none();
    struct HttpResponse resp = r.dispatch(&req, &conn);
    req.free();
    return resp;
}

void t_router_dispatch() {
    g_trace = String::new();
    struct Router r = Router::new();
    r.use(mw_trace_a);
    r.use(mw_trace_b);
    r.get("/users/:id", h_user);
    r.get("/users/me", h_me);
    r.post("/users", h_create);
    r.get("/big", h_big);
    r.get("/zero", h_zero);
    r.use_gzip(100UL);
    r.use_body_limit(16UL);
    r.use_status_guard();

    struct HttpResponse resp = call(&r, "GET", "/users/42?x=1", null, null);
    ASSERT_EQ(resp.status, 200);
    ASSERT_STR_EQ(resp.body.as_ptr(), "user 42");
    ASSERT_STR_EQ(g_trace.as_ptr(), "a>b><b<a");
    struct String mw = resp.header("X-Mw");
    ASSERT_STR_EQ(mw.as_ptr(), "a");
    mw.free();
    resp.free();
    resp = call(&r, "GET", "/users/me", null, null);   // literal beats :id regardless of order
    ASSERT_STR_EQ(resp.body.as_ptr(), "me");
    resp.free();
    resp = call(&r, "HEAD", "/users/me", null, null);  // HEAD falls back to GET
    ASSERT_EQ(resp.status, 200);
    resp.free();
    resp = call(&r, "POST", "/users", null, "{\"n\":1}");
    ASSERT_EQ(resp.status, 201);
    resp.free();
    resp = call(&r, "POST", "/users", null, "0123456789abcdefg");   // over the body limit
    ASSERT_EQ(resp.status, 413);
    resp.free();
    resp = call(&r, "DELETE", "/users/7", null, null);
    ASSERT_EQ(resp.status, 405);
    struct String allow = resp.header("Allow");
    ASSERT_STR_EQ(allow.as_ptr(), "GET, HEAD");
    allow.free();
    resp.free();
    resp = call(&r, "OPTIONS", "/users", null, null);
    ASSERT_EQ(resp.status, 204);
    allow = resp.header("Allow");
    ASSERT_STR_EQ(allow.as_ptr(), "POST, OPTIONS");
    allow.free();
    resp.free();
    resp = call(&r, "GET", "/nowhere", null, null);
    ASSERT_EQ(resp.status, 404);
    resp.free();
    r.set_not_found(h_missing);
    resp = call(&r, "GET", "/nowhere", null, null);
    ASSERT_STR_EQ(resp.body.as_ptr(), "custom 404");
    resp.free();
    g_trace.clear();
    resp = call(&r, "GET", "/users/1", "X-Block: 1", null);   // short-circuit
    ASSERT_EQ(resp.status, 403);
    ASSERT_STR_EQ(g_trace.as_ptr(), "a>b><a");
    resp.free();
    resp = call(&r, "GET", "/zero", null, null);
    ASSERT_EQ(resp.status, 500);
    resp.free();

    // gzip: only when asked for, and the result inflates back.
    resp = call(&r, "GET", "/big", null, null);
    ASSERT_FALSE(resp.has_header("Content-Encoding"));
    unsigned long plain = resp.body.length();
    resp.free();
    resp = call(&r, "GET", "/big", "Accept-Encoding: deflate, gzip;q=0.8", null);
    struct String ce = resp.header("Content-Encoding");
    ASSERT_STR_EQ(ce.as_ptr(), "gzip");
    ce.free();
    struct String vary = resp.header("Vary");
    ASSERT_STR_EQ(vary.as_ptr(), "Accept-Encoding");
    vary.free();
    ASSERT_TRUE(resp.body.length() < plain / 4UL);
    ASSERT_TRUE(http_decode_body(&resp, 0UL));
    ASSERT_EQ(resp.body.length(), plain);
    ASSERT_FALSE(resp.has_header("Content-Encoding"));
    resp.free();
    resp = call(&r, "GET", "/big", "Accept-Encoding: gzip;q=0", null);
    ASSERT_FALSE(resp.has_header("Content-Encoding"));
    resp.free();
    r.free();

    // CORS middleware.
    struct Router c = Router::new();
    c.use_cors("https://app.example", "GET, POST", "Content-Type");
    c.get("/api", h_me);
    resp = call(&c, "OPTIONS", "/api", "Origin: https://app.example\r\nAccess-Control-Request-Method: POST", null);
    ASSERT_EQ(resp.status, 204);
    struct String am = resp.header("Access-Control-Allow-Methods");
    ASSERT_STR_EQ(am.as_ptr(), "GET, POST");
    am.free();
    resp.free();
    resp = call(&c, "GET", "/api", "Origin: https://app.example", null);
    struct String ao = resp.header("Access-Control-Allow-Origin");
    ASSERT_STR_EQ(ao.as_ptr(), "https://app.example");
    ao.free();
    resp.free();
    c.free();
    g_trace.free();
}

void t_multipart_roundtrip() {
    struct MultipartBuilder b = MultipartBuilder::new();
    b.add_field("title", "Hello \"there\"");
    unsigned char bin[5] = { (unsigned char)0, (unsigned char)1, (unsigned char)13, (unsigned char)10, (unsigned char)255 };
    b.add_file("upload", "data.bin", "application/octet-stream", bin[0..5UL]);
    b.add_field("empty", "");
    b.finish();
    struct String ct = b.content_type();
    int ok = 0;
    struct MultipartForm f;
    unsafe { f = MultipartForm::parse(ct.as_ptr(), (const &unsigned char)b.body.as_ptr(), b.body.length(), &ok); }
    ASSERT_TRUE(ok);
    ASSERT_EQ(f.length(), 3);
    match (f.value("title")) { case some(v): { ASSERT_STR_EQ(v, "Hello \"there\""); } case null: { ASSERT_TRUE(0); } }
    match (f.get("upload")) {
        case some(p): {
            ASSERT_STR_EQ(p.filename.as_ptr(), "data.bin");
            ASSERT_TRUE(p.has_filename);
            ASSERT_STR_EQ(p.content_type.as_ptr(), "application/octet-stream");
            ASSERT_EQ(p.data.length(), 5);
            ASSERT_EQ(p.data.char_at(2UL), 13);
            ASSERT_EQ(p.data.char_at(4UL), 255);
        }
        case null: { ASSERT_TRUE(0); }
    }
    match (f.value("empty")) { case some(v): { ASSERT_STR_EQ(v, ""); } case null: { ASSERT_TRUE(0); } }
    ASSERT_TRUE(f.get("absent").is_null());
    f.free();
    // A body without its closing delimiter fails.
    unsafe { f = MultipartForm::parse(ct.as_ptr(), (const &unsigned char)b.body.as_ptr(), b.body.length() - 8UL, &ok); }
    ASSERT_FALSE(ok);
    f.free();
    unsafe { f = MultipartForm::parse("multipart/form-data", (const &unsigned char)"x", 1UL, &ok); }
    ASSERT_FALSE(ok);
    f.free();
    ct.free();
    b.free();
}

void t_sse_parser() {
    struct SseParser p = SseParser::new();
    const &char text = ": comment\nretry: 1500\nevent: tick\nid: 7\ndata: one\ndata: two\n\ndata: plain\r\n\r\nevent: noData\n\ndata:x";
    unsafe { p.feed(str_bytes(text)); }
    struct SseEvent e;
    ASSERT_TRUE(p.next(&e));
    ASSERT_STR_EQ(e.event.as_ptr(), "tick");
    ASSERT_STR_EQ(e.data.as_ptr(), "one\ntwo");
    ASSERT_STR_EQ(e.id.as_ptr(), "7");
    ASSERT_EQ(e.retry, 1500);
    e.free();
    ASSERT_TRUE(p.next(&e));
    ASSERT_STR_EQ(e.event.as_ptr(), "message");
    ASSERT_STR_EQ(e.data.as_ptr(), "plain");
    ASSERT_STR_EQ(e.id.as_ptr(), "7");    // sticky
    ASSERT_EQ(e.retry, -1);
    e.free();
    ASSERT_FALSE(p.next(&e));             // "data:x" not terminated yet
    unsafe { p.feed(str_bytes("\n\n")); }
    ASSERT_TRUE(p.next(&e));
    ASSERT_STR_EQ(e.data.as_ptr(), "x");
    e.free();
    p.free();
}

void t_static_paths() {
    struct String out = String::new();
    ASSERT_TRUE(http_safe_join("/srv/www/", "/a/./b%20c/", 0, &out));
    ASSERT_STR_EQ(out.as_ptr(), "/srv/www/a/b c");
    ASSERT_FALSE(http_safe_join("/srv", "/a/../../etc/passwd", 0, &out));
    ASSERT_FALSE(http_safe_join("/srv", "/%2e%2e/x", 0, &out));
    ASSERT_FALSE(http_safe_join("/srv", "/a%00b", 0, &out));
    ASSERT_FALSE(http_safe_join("/srv", "/a\\b", 0, &out));
    ASSERT_FALSE(http_safe_join("/srv", "/.git/config", 0, &out));
    ASSERT_TRUE(http_safe_join("/srv", "/.well-known/x", 1, &out));
    out.free();
    ASSERT_STR_EQ(http_mime_type("index.HTML"), "text/html; charset=utf-8");
    ASSERT_STR_EQ(http_mime_type("/a/app.wasm"), "application/wasm");
    ASSERT_STR_EQ(http_mime_type("x.tar.gz"), "application/gzip");
    ASSERT_STR_EQ(http_mime_type("noext"), "application/octet-stream");
    ASSERT_STR_EQ(http_mime_type("dir.d/file"), "application/octet-stream");
    ASSERT_TRUE(http_mime_is_compressible("text/css; charset=utf-8"));
    ASSERT_TRUE(http_mime_is_compressible("image/svg+xml"));
    ASSERT_FALSE(http_mime_is_compressible("image/png"));
}

// ── Loopback servers ───────────────────────────────────────────────────────

extern unsigned int alarm(unsigned int seconds);
extern int remove(const char* path);

static struct Router g_app;
static struct HttpServer g_srv;
static unsigned short g_port = (unsigned short)0;
static struct String g_base;      // "http://127.0.0.1:<port>"
static int g_started = 0;

static struct HttpResponse a_echo(&HttpRequest req, &HttpCtx ctx) {
    struct HttpResponse r = HttpResponse::text(200, req.body.as_ptr());
    struct String n = String::new();
    n.push_uint((unsigned long long)ctx.conn.request_index);
    r.set_header("X-Request-Index", n.as_ptr());
    n.free();
    struct String m = String::from(req.method.as_ptr());
    r.set_header("X-Method", m.as_ptr());
    m.free();
    return r;
}
static struct HttpResponse a_redirect(&HttpRequest req, &HttpCtx ctx) {
    const &char n = ctx.param_or("n", "0");
    int ok = 0;
    struct String s = String::from(n);
    long long v = s.parse_int(&ok);
    s.free();
    if (v <= 0LL) { return HttpResponse::redirect(303, "/final?from=chain"); }
    struct String loc = String::from("../hop/");
    loc.push_int(v - 1LL);
    struct HttpResponse r = HttpResponse::redirect(302, loc.as_ptr());
    loc.free();
    return r;
}
static struct HttpResponse a_final(&HttpRequest req, &HttpCtx ctx) {
    struct String b = String::from(req.method.as_ptr());
    b.push_char(' ');
    b.push_str(&req.path);
    struct HttpResponse r = HttpResponse::text(200, b.as_ptr());
    b.free();
    return r;
}
static struct HttpResponse a_loop(&HttpRequest req, &HttpCtx ctx) { return HttpResponse::redirect(302, "/loop"); }
static struct HttpResponse a_big(&HttpRequest req, &HttpCtx ctx) {
    struct String s = String::repeat_str("lorem ipsum dolor ", 5000UL);
    struct HttpResponse r = HttpResponse::text(200, s.as_ptr());
    s.free();
    return r;
}
static struct HttpResponse a_upload(&HttpRequest req, &HttpCtx ctx) {
    int ok = 0;
    struct MultipartForm f = MultipartForm::from_request(req, &ok);
    struct String out = String::new();
    out.push_int((long long)ok);
    out.push_char(':');
    match (f.value("who")) { case some(v): { out.push(v); } case null: { } }
    out.push_char(':');
    match (f.get("file")) {
        case some(p): { out.push_str(&p.filename); out.push_char('='); out.push_uint((unsigned long long)p.data.length()); }
        case null: { }
    }
    f.free();
    struct HttpResponse r = HttpResponse::text(200, out.as_ptr());
    out.free();
    return r;
}
static struct HttpResponse a_cookie(&HttpRequest req, &HttpCtx ctx) {
    struct HttpResponse r = HttpResponse::text(200, "set");
    struct SetCookie c = SetCookie::new("session", "s1");
    c.path.push("/");
    c.http_only = 1;
    http_set_cookie(&r, &c);
    c.free();
    struct SetCookie c2 = SetCookie::new("pref", "x");
    http_set_cookie(&r, &c2);
    c2.free();
    return r;
}
static struct HttpResponse a_stream(&HttpRequest req, &HttpCtx ctx) {
    struct HttpChunkedWriter w = HttpChunkedWriter::begin(ctx.conn, 200, "Content-Type: text/plain");
    int i = 0;
    while (i < 5) {
        w.write_str("part;");
        i = i + 1;
    }
    w.finish();
    return HttpResponse::new(200);   // ignored: detached
}
static struct HttpResponse a_events(&HttpRequest req, &HttpCtx ctx) {
    struct SseStream s = SseStream::start(ctx.conn, null);
    s.retry(2000);
    s.comment("hello");
    s.send_event("greet", "line1\nline2", "1");
    s.send("plain");
    struct String last = req.header("Last-Event-ID");
    s.send_event("resume", last.as_ptr(), "2");
    last.free();
    s.close();
    return HttpResponse::new(200);
}
static struct HttpResponse a_ws(&HttpRequest req, &HttpCtx ctx) {
    if (!ws_is_upgrade(req)) { return HttpResponse::text(426, "upgrade required"); }
    int ok = 0;
    struct WsConn c = WsConn::accept_upgrade(req, ctx.conn, &ok);
    if (!ok) { return HttpResponse::error(400); }
    // A fragmented server->client message with a ping in between.
    unsafe {
        struct TcpStream s = TcpStream::from_fd(c.fd);
        unsigned char frames[13] = { (unsigned char)0x01, (unsigned char)3, (unsigned char)'h', (unsigned char)'e', (unsigned char)'l',
                                     (unsigned char)0x89, (unsigned char)0,
                                     (unsigned char)0x80, (unsigned char)2, (unsigned char)'l', (unsigned char)'o',
                                     (unsigned char)0, (unsigned char)0 };
        write_all(&s, frames[0..11UL]);
    }
    struct WsMessage m;
    while (c.recv_message(&m, 1048576UL)) {
        if (m.opcode == WS_OPCODE_CLOSE) { m.free(); break; }
        struct String echo = String::from("echo:");
        unsafe { echo.push_bytes(((const char*)m.data)[0..m.len]); }
        c.send_text(echo.as_ptr());
        echo.free();
        m.free();
    }
    c.close();
    return HttpResponse::new(200);
}
static struct HttpResponse a_close(&HttpRequest req, &HttpCtx ctx) {
    ctx.conn.keep_alive = 0;
    return HttpResponse::text(200, "bye");
}

static void write_file(const &char path, const &char text) {
    ?&File f = File::open(path, "wb");
    match (f) {
        case some(file): { unsafe { file.write(str_bytes(text)); } file.close(); }
        case null: { ASSERT_TRUE(0); }
    }
}

static void start_app() {
    if (g_started) return;
    g_started = 1;
    write_file("web_static_test.txt", "0123456789abcdefghij");
    write_file("web_static_test.css", "body { color: red; }");
    g_app = Router::new();
    g_app.any("/echo", a_echo);
    g_app.any("/hop/:n", a_redirect);
    g_app.any("/final", a_final);
    g_app.get("/loop", a_loop);
    g_app.get("/big", a_big);
    g_app.post("/upload", a_upload);
    g_app.get("/cookie", a_cookie);
    g_app.get("/stream", a_stream);
    g_app.get("/events", a_events);
    g_app.get("/ws", a_ws);
    g_app.get("/close", a_close);
    g_app.static_dir("/static", ".");
    g_app.use_gzip(512UL);
    struct HttpServerConfig cfg = HttpServerConfig::defaults((unsigned short)0);
    cfg.threads = 6;
    cfg.max_header_bytes = 4096UL;
    cfg.max_body_bytes = 65536UL;
    cfg.idle_timeout_ms = 2000;
    int ok = 0;
    g_srv = HttpServer::bind(&cfg, &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(g_srv.port != (unsigned short)0);
    g_port = g_srv.port;
    ASSERT_EQ(g_app.start(&g_srv), 0);
    g_base = String::from("http://127.0.0.1:");
    g_base.push_uint((unsigned long long)g_port);
}

static struct String url(const &char path) {
    struct String u = g_base.clone();
    u.push(path);
    return u;
}

// A raw connection to the test server.
static int raw_connect() {
    unsigned int lo = sched_ipv4((unsigned char)127, (unsigned char)0, (unsigned char)0, (unsigned char)1);
    int fd = tcp_connect_timeout(lo, g_port, 2000);
    if (fd >= 0) sock_set_io_timeout(fd, 3000);
    return fd;
}

static int raw_send(int fd, const &char text) {
    struct TcpStream s = TcpStream::from_fd(fd);
    unsafe { return write_all(&s, str_bytes(text)); }
}

// Reads one response (head + Content-Length body) from 'in'.
static int raw_response(&BufReader<TcpStream> in, &String head, &String body) {
    head.clear();
    body.clear();
    if (!in.read_until(head, "\r\n\r\n")) return 0;
    struct String cl = http_header_get(head.as_ptr(), "Content-Length");
    int ok = 0;
    long long n = cl.parse_int(&ok);
    cl.free();
    if (!ok) return 1;
    unsigned char buf[8192];
    if (n > 8192LL || !read_exact(in, buf[0..(unsigned long)n])) return 0;
    body.push_bytes(buf[0..(unsigned long)n]);
    return 1;
}

void t_server_keep_alive() {
    start_app();
    int fd = raw_connect();
    ASSERT_TRUE(fd >= 0);
    if (fd < 0) return;
    // Two pipelined requests, then a third after reading those.
    ASSERT_TRUE(raw_send(fd, "GET /echo HTTP/1.1\r\nHost: t\r\n\r\nPOST /echo HTTP/1.1\r\nHost: t\r\nContent-Length: 3\r\n\r\nabc"));
    struct TcpStream s = TcpStream::from_fd(fd);
    struct BufReader<TcpStream> in = BufReader::over(&s);
    struct String head = String::new();
    struct String body = String::new();
    ASSERT_TRUE(raw_response(&in, &head, &body));
    ASSERT_TRUE(head.starts_with("HTTP/1.1 200 OK"));
    ASSERT_TRUE(head.contains("Connection: keep-alive"));
    ASSERT_TRUE(head.contains("Date: "));
    ASSERT_TRUE(head.contains("X-Request-Index: 0"));
    ASSERT_TRUE(raw_response(&in, &head, &body));
    ASSERT_TRUE(head.contains("X-Request-Index: 1"));
    ASSERT_STR_EQ(body.as_ptr(), "abc");
    // Chunked request body with an extension and a trailer.
    ASSERT_TRUE(raw_send(fd, "POST /echo HTTP/1.1\r\nHost: t\r\nTransfer-Encoding: chunked\r\n\r\n4;x=y\r\nWiki\r\n5\r\npedia\r\n0\r\nX-T: 1\r\n\r\n"));
    ASSERT_TRUE(raw_response(&in, &head, &body));
    ASSERT_TRUE(head.contains("X-Request-Index: 2"));
    ASSERT_STR_EQ(body.as_ptr(), "Wikipedia");
    // HEAD: headers with the GET's length, no body.
    ASSERT_TRUE(raw_send(fd, "HEAD /static/web_static_test.txt HTTP/1.1\r\nHost: t\r\n\r\n"));
    head.clear();
    ASSERT_TRUE(in.read_until(&head, "\r\n\r\n"));
    ASSERT_TRUE(head.contains("Content-Length: 20"));
    // Expect: 100-continue gets an interim response before the body is sent.
    ASSERT_TRUE(raw_send(fd, "POST /echo HTTP/1.1\r\nHost: t\r\nContent-Length: 2\r\nExpect: 100-continue\r\n\r\n"));
    head.clear();
    ASSERT_TRUE(in.read_until(&head, "\r\n\r\n"));
    ASSERT_TRUE(head.starts_with("HTTP/1.1 100 Continue"));
    ASSERT_TRUE(raw_send(fd, "hi"));
    ASSERT_TRUE(raw_response(&in, &head, &body));
    ASSERT_STR_EQ(body.as_ptr(), "hi");
    // A handler that asks to close; then the server closes.
    ASSERT_TRUE(raw_send(fd, "GET /close HTTP/1.1\r\nHost: t\r\n\r\n"));
    ASSERT_TRUE(raw_response(&in, &head, &body));
    ASSERT_TRUE(head.contains("Connection: close"));
    unsigned char one[1];
    ASSERT_EQ(in.read(one[0..1UL]), 0);
    s.close();

    // HTTP/1.0 without keep-alive closes after one response.
    fd = raw_connect();
    ASSERT_TRUE(fd >= 0);
    if (fd < 0) return;
    struct TcpStream s2 = TcpStream::from_fd(fd);
    struct BufReader<TcpStream> in2 = BufReader::over(&s2);
    ASSERT_TRUE(raw_send(fd, "GET /echo HTTP/1.0\r\n\r\n"));
    ASSERT_TRUE(raw_response(&in2, &head, &body));
    ASSERT_TRUE(head.contains("Connection: close"));
    s2.close();
    head.free();
    body.free();
}

static int status_of(const &char request) {
    int fd = raw_connect();
    if (fd < 0) return -1;
    raw_send(fd, request);
    struct TcpStream s = TcpStream::from_fd(fd);
    struct BufReader<TcpStream> in = BufReader::over(&s);
    struct String head = String::new();
    int st = 0;
    if (in.read_until(&head, "\r\n\r\n") && head.length() > 12UL) {
        struct String code = head.substr(9UL, 12UL);
        int ok = 0;
        st = (int)code.parse_int(&ok);
        code.free();
    }
    head.free();
    s.close();
    return st;
}

void t_server_limits() {
    start_app();
    struct String hugeHeader = String::from("GET /echo HTTP/1.1\r\nHost: t\r\nX-Big: ");
    struct String pad = String::repeat_str("x", 5000UL);
    hugeHeader.push_str(&pad);
    hugeHeader.push("\r\n\r\n");
    ASSERT_EQ(status_of(hugeHeader.as_ptr()), 431);
    hugeHeader.free();
    pad.free();
    ASSERT_EQ(status_of("POST /echo HTTP/1.1\r\nHost: t\r\nContent-Length: 999999\r\n\r\n"), 413);
    ASSERT_EQ(status_of("GET /echo HTTP/2.0\r\nHost: t\r\n\r\n"), 505);
    ASSERT_EQ(status_of("GET /echo HTTP/1.1\r\n\r\n"), 400);                 // no Host
    ASSERT_EQ(status_of("garbage\r\n\r\n"), 400);
    ASSERT_EQ(status_of("POST /echo HTTP/1.1\r\nHost: t\r\nTransfer-Encoding: gzip\r\n\r\n"), 501);
    ASSERT_EQ(status_of("POST /echo HTTP/1.1\r\nHost: t\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\nab"), 400);
    ASSERT_EQ(status_of("DELETE /loop HTTP/1.1\r\nHost: t\r\n\r\n"), 405);
    // Absolute-form target.
    struct String abs = String::from("GET http://127.0.0.1/echo HTTP/1.1\r\nHost: t\r\n\r\n");
    ASSERT_EQ(status_of(abs.as_ptr()), 200);
    abs.free();
}

void t_client_fetch() {
    start_app();
    int ok = 0;
    struct String u = url("/hop/3");
    struct String final_url = String::new();
    struct HttpResponse r;
    unsafe { r = http_fetch("POST", u.as_ptr(), "Content-Type: text/plain", str_bytes("x"), null, &final_url, &ok); }
    ASSERT_TRUE(ok);
    ASSERT_EQ(r.status, 200);
    ASSERT_STR_EQ(r.body.as_ptr(), "GET /final?from=chain");   // 302/303 turn POST into GET
    ASSERT_TRUE(final_url.ends_with("/final?from=chain"));
    r.free();
    final_url.free();
    u.free();

    struct HttpClientOptions o = HttpClientOptions::defaults();
    o.max_redirects = 0;
    u = url("/hop/1");
    r = http_fetch("GET", u.as_ptr(), null, str_bytes(""), &o, null, &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ(r.status, 302);
    r.free();
    u.free();
    u = url("/loop");
    r = http_fetch("GET", u.as_ptr(), null, str_bytes(""), null, null, &ok);
    ASSERT_FALSE(ok);   // too many redirects
    r.free();
    u.free();

    // gzip on the wire, decoded by the client.
    u = url("/big");
    r = http_get_url(u.as_ptr(), &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ(r.body.length(), 90000);
    ASSERT_FALSE(r.has_header("Content-Encoding"));
    r.free();
    o = HttpClientOptions::defaults();
    o.decompress = 0;
    r = http_fetch("GET", u.as_ptr(), "Accept-Encoding: gzip", str_bytes(""), &o, null, &ok);
    ASSERT_TRUE(ok);
    struct String ce = r.header("Content-Encoding");
    ASSERT_STR_EQ(ce.as_ptr(), "gzip");
    ce.free();
    ASSERT_TRUE(r.body.length() < 2000UL);
    r.free();
    o.decompress = 1;
    o.max_body_bytes = 1000UL;
    r = http_fetch("GET", u.as_ptr(), null, str_bytes(""), &o, null, &ok);
    ASSERT_FALSE(ok);   // decoded body over the limit
    r.free();
    u.free();

    // Chunked streaming response.
    u = url("/stream");
    r = http_get_url(u.as_ptr(), &ok);
    ASSERT_TRUE(ok);
    ASSERT_STR_EQ(r.body.as_ptr(), "part;part;part;part;part;");
    r.free();
    u.free();

    // Cookies.
    u = url("/cookie");
    r = http_get_url(u.as_ptr(), &ok);
    struct HttpHeaders h = r.header_map();
    ASSERT_EQ(h.count("Set-Cookie"), 2);
    match (h.get("set-cookie")) {
        case some(v): {
            struct SetCookie c = SetCookie::parse(v, &ok);
            ASSERT_TRUE(ok);
            ASSERT_STR_EQ(c.value.as_ptr(), "s1");
            ASSERT_TRUE(c.http_only);
            c.free();
        }
        case null: { ASSERT_TRUE(0); }
    }
    h.free();
    r.free();
    u.free();

    // Multipart upload.
    struct MultipartBuilder mb = MultipartBuilder::new();
    mb.add_field("who", "Ada");
    struct String payload = String::repeat_str("\x01\x02\x03", 1000UL);
    unsafe { mb.add_file("file", "blob.bin", "application/octet-stream", payload.as_bytes()); }
    mb.finish();
    struct String ct = mb.content_type();
    struct String hdr = String::from("Content-Type: ");
    hdr.push_str(&ct);
    u = url("/upload");
    unsafe { r = http_fetch("POST", u.as_ptr(), hdr.as_ptr(), mb.body.as_bytes(), null, null, &ok); }
    ASSERT_TRUE(ok);
    ASSERT_STR_EQ(r.body.as_ptr(), "1:Ada:blob.bin=3000");
    r.free();
    u.free();
    hdr.free();
    ct.free();
    payload.free();
    mb.free();

    // Bad URLs (https:// is supported now: see test_tls.sc).
    r = http_get_url("ftp://example.com/", &ok);
    ASSERT_FALSE(ok);
    r.free();
    r = http_get_url("not a url", &ok);
    ASSERT_FALSE(ok);
    r.free();
}

void t_static_files() {
    start_app();
    int ok = 0;
    struct String u = url("/static/web_static_test.txt");
    struct HttpResponse r = http_get_url(u.as_ptr(), &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ(r.status, 200);
    ASSERT_STR_EQ(r.body.as_ptr(), "0123456789abcdefghij");
    struct String ct = r.header("Content-Type");
    ASSERT_STR_EQ(ct.as_ptr(), "text/plain; charset=utf-8");
    ct.free();
    struct String etag = r.header("ETag");
    struct String lm = r.header("Last-Modified");
    ASSERT_TRUE(etag.starts_with("\""));
    ASSERT_TRUE(lm.ends_with("GMT"));
    r.free();
    // Conditional GETs.
    struct String h = String::from("If-None-Match: ");
    h.push_str(&etag);
    r = http_fetch("GET", u.as_ptr(), h.as_ptr(), str_bytes(""), null, null, &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ(r.status, 304);
    ASSERT_EQ(r.body.length(), 0);
    r.free();
    h.clear();
    h.push("If-Modified-Since: ");
    h.push_str(&lm);
    r = http_fetch("GET", u.as_ptr(), h.as_ptr(), str_bytes(""), null, null, &ok);
    ASSERT_EQ(r.status, 304);
    r.free();
    // Ranges.
    r = http_fetch("GET", u.as_ptr(), "Range: bytes=2-5", str_bytes(""), null, null, &ok);
    ASSERT_EQ(r.status, 206);
    ASSERT_STR_EQ(r.body.as_ptr(), "2345");
    struct String cr = r.header("Content-Range");
    ASSERT_STR_EQ(cr.as_ptr(), "bytes 2-5/20");
    cr.free();
    r.free();
    r = http_fetch("GET", u.as_ptr(), "Range: bytes=-3", str_bytes(""), null, null, &ok);
    ASSERT_STR_EQ(r.body.as_ptr(), "hij");
    r.free();
    r = http_fetch("GET", u.as_ptr(), "Range: bytes=15-", str_bytes(""), null, null, &ok);
    ASSERT_STR_EQ(r.body.as_ptr(), "fghij");
    r.free();
    r = http_fetch("GET", u.as_ptr(), "Range: bytes=50-60", str_bytes(""), null, null, &ok);
    ASSERT_EQ(r.status, 416);
    r.free();
    h.clear();
    h.push("Range: bytes=0-1\r\nIf-Range: \"stale\"");
    r = http_fetch("GET", u.as_ptr(), h.as_ptr(), str_bytes(""), null, null, &ok);
    ASSERT_EQ(r.status, 200);   // validator mismatch: whole file
    r.free();
    h.free();
    etag.free();
    lm.free();
    u.free();

    u = url("/static/web_static_test.css");
    r = http_get_url(u.as_ptr(), &ok);
    ct = r.header("Content-Type");
    ASSERT_STR_EQ(ct.as_ptr(), "text/css; charset=utf-8");
    ct.free();
    r.free();
    u.free();
    u = url("/static/missing.txt");
    r = http_get_url(u.as_ptr(), &ok);
    ASSERT_EQ(r.status, 404);
    r.free();
    u.free();
    u = url("/static/%2e%2e/secret");
    r = http_get_url(u.as_ptr(), &ok);
    ASSERT_EQ(r.status, 403);
    r.free();
    u.free();
    // A directory without the slash redirects to it.
    struct HttpClientOptions o = HttpClientOptions::defaults();
    o.max_redirects = 0;
    u = url("/static/tests");
    r = http_fetch("GET", u.as_ptr(), null, str_bytes(""), &o, null, &ok);
    ASSERT_EQ(r.status, 301);
    struct String loc = r.header("Location");
    ASSERT_STR_EQ(loc.as_ptr(), "/static/tests/");
    loc.free();
    r.free();
    u.free();
}

void t_sse_stream() {
    start_app();
    int ok = 0;
    struct String u = url("/events");
    struct SseClient c = SseClient::connect(u.as_ptr(), "41", 2000, &ok);
    ASSERT_TRUE(ok);
    u.free();
    if (!ok) return;
    struct SseEvent e;
    ASSERT_TRUE(c.next(&e));
    ASSERT_STR_EQ(e.event.as_ptr(), "greet");
    ASSERT_STR_EQ(e.data.as_ptr(), "line1\nline2");
    ASSERT_STR_EQ(e.id.as_ptr(), "1");
    ASSERT_EQ(e.retry, 2000);
    e.free();
    ASSERT_TRUE(c.next(&e));
    ASSERT_STR_EQ(e.event.as_ptr(), "message");
    ASSERT_STR_EQ(e.data.as_ptr(), "plain");
    e.free();
    ASSERT_TRUE(c.next(&e));
    ASSERT_STR_EQ(e.event.as_ptr(), "resume");
    ASSERT_STR_EQ(e.data.as_ptr(), "41");
    e.free();
    ASSERT_FALSE(c.next(&e));   // the server closed the stream
    c.close();
}

void t_websocket_upgrade() {
    start_app();
    int ok = 0;
    struct String u = String::from("ws://127.0.0.1:");
    u.push_uint((unsigned long long)g_port);
    u.push("/ws");
    struct WsConn c = WsConn::connect_url(u.as_ptr(), 2000, &ok);
    u.free();
    ASSERT_TRUE(ok);
    if (!ok) return;
    struct WsMessage m;
    // "hel" + ping + "lo": reassembled, the ping answered automatically.
    ASSERT_TRUE(c.recv_message(&m, 0UL));
    ASSERT_EQ(m.opcode, WS_OPCODE_TEXT);
    ASSERT_EQ(m.len, 5);
    match (m.data) { case some(d): { unsafe { ASSERT_EQ(d[0], 'h'); ASSERT_EQ(d[4], 'o'); } } case null: { ASSERT_TRUE(0); } }
    m.free();
    ASSERT_TRUE(c.send_text("ping me"));
    ASSERT_TRUE(c.recv_message(&m, 0UL));
    ASSERT_EQ(m.len, 12);
    m.free();
    c.send_close();
    c.close();
    // Plain GET on the WebSocket route.
    struct String plain = url("/ws");
    struct HttpResponse r = http_get_url(plain.as_ptr(), &ok);
    ASSERT_EQ(r.status, 426);
    r.free();
    plain.free();
}

static struct HttpResponse plain_handler(&HttpRequest req) { return HttpResponse::text(200, "plain"); }

void t_server_stop() {
    struct HttpServerConfig cfg = HttpServerConfig::defaults((unsigned short)0);
    cfg.threads = 3;
    int ok = 0;
    struct HttpServer srv = HttpServer::bind(&cfg, &ok);
    ASSERT_TRUE(ok);
    if (!ok) return;
    ASSERT_EQ(srv.start(plain_handler), 0);
    ASSERT_TRUE(srv.is_running());
    // An idle keep-alive connection must not hold stop() up.
    unsigned int lo = sched_ipv4((unsigned char)127, (unsigned char)0, (unsigned char)0, (unsigned char)1);
    int idle = tcp_connect_timeout(lo, srv.port, 2000);
    ASSERT_TRUE(idle >= 0);
    struct HttpResponse r = http_get("127.0.0.1", srv.port, "/", &ok);
    ASSERT_TRUE(ok);
    ASSERT_STR_EQ(r.body.as_ptr(), "plain");
    r.free();
    struct TcpStream is = TcpStream::from_fd(idle);
    raw_send(idle, "GET / HTTP/1.1\r\nHost: t\r\n\r\n");
    unsigned char buf[512];
    ASSERT_TRUE(is.read(buf[0..512UL]) > 0LL);   // served; now idle on keep-alive
    long long t0 = time_mono_ns();
    srv.stop();
    ASSERT_TRUE((time_mono_ns() - t0) / 1000000LL < 1500LL);
    ASSERT_FALSE(srv.is_running());
    is.close();
    r = http_request_timeout("127.0.0.1", srv.port, "GET", "/", null, str_bytes(""), 500, 500, &ok);
    ASSERT_FALSE(ok);
    r.free();
    srv.stop();   // idempotent
}

static void add_net_tests(&TestSuite t) {
    // A networking regression should fail the run, not hang it.
    unsafe { alarm(90u); }
    t.add("server: keep-alive, pipelining, chunked body, HEAD, 100-continue", t_server_keep_alive);
    t.add("server: limits and malformed requests", t_server_limits);
    t.add("client: redirects, gzip, chunked, cookies, multipart", t_client_fetch);
    t.add("static files: etag, ranges, mime, traversal", t_static_files);
    t.add("server-sent events", t_sse_stream);
    t.add("websocket upgrade on the router", t_websocket_upgrade);
    t.add("server stop with idle connections", t_server_stop);
}

int main() {
    struct TestSuite t = TestSuite::new();
    t.add("headers map + raw block edits + params", t_headers);
    t.add("request/response helpers", t_request_response_helpers);
    t.add("http dates", t_dates);
    t.add("cookies", t_cookies);
    t.add("chunked coding", t_chunked_coding);
    t.add("route patterns", t_route_patterns);
    t.add("router dispatch + middleware", t_router_dispatch);
    t.add("multipart build/parse", t_multipart_roundtrip);
    t.add("sse parser", t_sse_parser);
    t.add("static path safety + mime", t_static_paths);
    add_net_tests(&t);
    t.run();
    t.print_summary();
    unsafe {
        remove("web_static_test.txt");
        remove("web_static_test.css");
    }
    return t.all_passed() ? 0 : 1;
}
