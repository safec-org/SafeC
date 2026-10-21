// std::http (client+server over loopback, header parsing, CORS, JWT,
// OAuth2, WebSocket) and std::rpc (rpc_call/rpc_serve, server functions).
// Each server runs on its own thread for the life of the process.
#include <std/test/test.h>
#include <std/mem.h>
#include <std/panic.h>
#include <std/str.h>
#include <std/thread.h>
#include <std/time.h>
#include <std/sched/io_nb.h>
#include <std/collections/string.h>
#include <std/serial/value.h>
#include <std/serial/json.h>
#include <std/http/http.h>
#include <std/http/cors.h>
#include <std/http/jwt.h>
#include <std/http/oauth2.h>
#include <std/http/websocket.h>
#include <std/rpc/rpc.h>
#include <std/rpc/server_fn.h>

static unsigned short g_http_port = (unsigned short)0;
static unsigned short g_rpc_port = (unsigned short)0;
static unsigned short g_ws_port = (unsigned short)0;
static unsigned short g_fn_port = (unsigned short)0;

static void pick_ports() {
    unsafe {
        // Spread across runs so a lingering TIME_WAIT socket from a previous
        // run can't collide.
        unsigned short base = (unsigned short)(20000ULL + thread_self() % 20000ULL);
        g_http_port = base;
        g_rpc_port = (unsigned short)(base + (unsigned short)1);
        g_ws_port = (unsigned short)(base + (unsigned short)2);
        g_fn_port = (unsigned short)(base + (unsigned short)3);
    }
}

static struct HttpResponse handler(&HttpRequest req) {
    struct HttpResponse r;
    unsafe {
        if (req.path.eq_cstr("/token") && req.method.eq_cstr("POST")) {
            const &char tok = "{\"access_token\":\"abc123\",\"refresh_token\":\"r1\",\"token_type\":\"Bearer\",\"expires_in\":3600}";
            r.status = 200;
            r.headers = string_from("Content-Type: application/json");
            r.body = string_from(tok);
            return r;
        }
        if (req.path.eq_cstr("/echo")) {
            r.status = 200;
            r.headers = string_from("Content-Type: text/plain\r\nX-Method: ");
            r.headers.push(req.method.as_ptr());
            r.body = req.body.clone();
            return r;
        }
        if (req.path.eq_cstr("/cors")) {
            r.status = 200;
            r.headers = string_new();
            r.body = string_from("ok");
            cors_apply_headers(&r, "https://a.example");
            return r;
        }
        r.status = 404;
        r.headers = string_new();
        r.body = string_from("nope");
    }
    return r;
}

static ?&int run_http(?&int arg) {
    unsafe { http_serve(g_http_port, handler); }
    return null;
}

static struct RpcResult rpc_upper(const &unsigned char data, unsigned long len) {
    struct RpcResult r;
    unsafe {
        ?&heap unsigned char out = (?&heap unsigned char)alloc(len);
        r.ok = 1;
        r.len = len;
        r.error_message = string_new();
        if (out.is_null()) { r.ok = 0; r.len = 0UL; r.data = null; return r; }
        &unsigned char o = std::expect_ref(out, "out is null");
        unsigned long i = 0UL;
        while (i < len) {
            unsigned char c = data[i];
            if (c >= (unsigned char)'a' && c <= (unsigned char)'z') c = (unsigned char)(c - 32);
            o[i] = c;
            i = i + 1UL;
        }
        r.data = out;
    }
    return r;
}

static ?&int run_rpc(?&int arg) {
    unsafe {
        rpc_register("/upper", rpc_upper);
        rpc_serve(g_rpc_port);
    }
    return null;
}

static struct Value fn_add(const &Value args) {
    long long a = 0LL;
    long long b = 0LL;
    unsafe {
        ?&Value av = args.object_get("a");
        ?&Value bv = args.object_get("b");
        if (!av.is_null()) a = av->as_int();
        if (!bv.is_null()) b = bv->as_int();
    }
    return value_int(a + b);
}

static ?&int run_fn(?&int arg) {
    unsafe {
        server_fn_register("add", fn_add);
        server_fn_serve(g_fn_port);
    }
    return null;
}

static ?&int run_ws(?&int arg) {
    unsafe {
        int lfd = ws_listen(g_ws_port);
        if (lfd < 0) return null;
        struct WsConn c;
        if (ws_accept(lfd, &c) != 1) return null;   // 1 = handshake completed
        struct WsMessage m;
        while (ws_recv(&c, &m) == 1) {
            if (m.opcode == WS_OPCODE_TEXT) {
                struct String echo = string_from("echo:");
                echo.push_n((const char*)m.data, m.len);
                ws_send_text(&c, echo.as_ptr());
                echo.free();
            } else if (m.opcode == WS_OPCODE_PING) {
                ws_send_pong(&c);
            } else if (m.opcode == WS_OPCODE_CLOSE) {
                m.free();
                break;
            }
            m.free();
        }
        ws_close(&c);
    }
    return null;
}

static int g_servers_started = 0;
static void start_servers() {
    if (g_servers_started) return;
    g_servers_started = 1;
    pick_ports();
    unsafe {
        unsigned long long t = 0ULL;
        thread_create(&t, run_http, null); thread_detach(t);
        thread_create(&t, run_rpc, null);  thread_detach(t);
        thread_create(&t, run_ws, null);   thread_detach(t);
        thread_create(&t, run_fn, null);   thread_detach(t);
    }
}

// GET with retries while the server thread comes up.
static struct HttpResponse get_retry(const &char path, &int ok) {
    struct HttpResponse r;
    int tries = 0;
    while (1) {
        unsafe { r = http_get("127.0.0.1", g_http_port, path, ok); }
        if (*ok || tries >= 50) return r;
        r.free();
        thread_sleep_ms(20UL);
        tries = tries + 1;
    }
    return r;
}

void t_http_roundtrip() {
    start_servers();
    unsafe {
        int ok = 0;
        struct HttpResponse r = get_retry("/echo", &ok);
        ASSERT_TRUE(ok);
        ASSERT_EQ(r.status, 200);
        struct String m = http_header_get(r.headers.as_ptr(), "x-method");   // case-insensitive
        ASSERT_STR_EQ(m.as_ptr(), "GET");
        m.free();
        r.free();

        const &char body = "posted payload";
        r = http_post("127.0.0.1", g_http_port, "/echo", "text/plain",
                      (const unsigned char*)body, str_len(body), &ok);
        ASSERT_TRUE(ok);
        ASSERT_STR_EQ(r.body.as_ptr(), "posted payload");
        r.free();

        r = http_get("127.0.0.1", g_http_port, "/cors", &ok);
        ASSERT_TRUE(ok);
        struct String acao = http_header_get(r.headers.as_ptr(), "Access-Control-Allow-Origin");
        ASSERT_STR_EQ(acao.as_ptr(), "https://a.example");   // handler headers reach the client
        acao.free();
        r.free();

        r = http_get("127.0.0.1", g_http_port, "/missing", &ok);
        ASSERT_TRUE(ok);
        ASSERT_EQ(r.status, 404);
        r.free();

        ASSERT_STR_EQ(http_reason_phrase(404), "Not Found");
        struct String raw = http_build_response(200, "text/plain", (const unsigned char*)"hi", 2UL);
        ASSERT_TRUE(raw.starts_with("HTTP/1.1 200"));
        ASSERT_TRUE(raw.ends_with("\r\n\r\nhi"));
        raw.free();
        unsigned int addr = 0u;
        ASSERT_EQ(http_resolve_host("127.0.0.1", &addr), 1);
        ASSERT_EQ(http_resolve_host("localhost", &addr), 1);
    }
}

void t_cors() {
    unsafe {
        struct HttpRequest req;
        req.method = string_from("OPTIONS");
        req.path = string_from("/api");
        req.headers = string_from("Origin: https://a.example\r\nAccess-Control-Request-Method: POST");
        req.body = string_new();
        ASSERT_TRUE(cors_is_preflight(&req));
        req.method.free();
        req.method = string_from("GET");
        ASSERT_FALSE(cors_is_preflight(&req));
        req.free();
        struct HttpResponse p = cors_preflight_response("https://a.example", "GET, POST", "Content-Type");
        struct String o = http_header_get(p.headers.as_ptr(), "Access-Control-Allow-Origin");
        ASSERT_STR_EQ(o.as_ptr(), "https://a.example");
        o.free();
        p.free();
    }
}

void t_jwt() {
    unsafe {
        struct Value claims = value_object();
        value_object_set(&claims, "sub", value_string("1234567890"));
        value_object_set(&claims, "admin", value_bool(1));
        struct String tok = jwt_sign(&claims, "secret");
        ASSERT_EQ(tok.count("."), 2);
        ASSERT_TRUE(tok.starts_with("eyJ"));   // base64url of '{"'
        int ok = 0;
        struct Value back = jwt_verify(tok.as_ptr(), "secret", &ok);
        ASSERT_TRUE(ok);
        ?&Value sub = back.object_get("sub");
        ASSERT_FALSE(sub.is_null());
        ASSERT_STR_EQ(sub->as_string(), "1234567890");
        value_free(&back);
        struct Value bad = jwt_verify(tok.as_ptr(), "wrong", &ok);
        ASSERT_FALSE(ok);
        value_free(&bad);
        // Tamper with the payload: signature must no longer verify.
        tok.set_char((unsigned long)tok.index_of(".") + 2UL, 'A');
        struct Value bad2 = jwt_verify(tok.as_ptr(), "secret", &ok);
        ASSERT_FALSE(ok);
        value_free(&bad2);
        tok.free();
        value_free(&claims);
    }
}

void t_oauth2() {
    start_servers();
    unsafe {
        int ok = 0;
        struct HttpResponse warm = get_retry("/echo", &ok);
        warm.free();
        struct OAuth2Token t = oauth2_exchange_code("127.0.0.1", g_http_port, "/token", "cid", "csecret",
                                                    "code42", "http://localhost/cb", &ok);
        ASSERT_TRUE(ok);
        ASSERT_STR_EQ(t.accessToken.as_ptr(), "abc123");
        ASSERT_STR_EQ(t.refreshToken.as_ptr(), "r1");
        ASSERT_EQ(t.expiresIn, 3600);
        oauth2_token_free(&t);
    }
}

void t_rpc() {
    start_servers();
    unsafe {
        struct RpcResult r;
        int tries = 0;
        while (1) {
            r = rpc_call("127.0.0.1", g_rpc_port, "/upper", (const unsigned char*)"hello rpc", 9UL);
            if (r.ok || tries >= 50) break;
            r.free();
            thread_sleep_ms(20UL);
            tries = tries + 1;
        }
        ASSERT_TRUE(r.ok);
        ASSERT_EQ(r.len, 9);
        match (r.data) {
            case null: { ASSERT_TRUE(0); }
            case some(d): {
                ASSERT_EQ(d[0], 'H');
                ASSERT_EQ(d[8], 'C');
            }
        }
        r.free();
        struct RpcResult miss = rpc_call("127.0.0.1", g_rpc_port, "/nope", (const unsigned char*)"x", 1UL);
        ASSERT_FALSE(miss.ok);
        miss.free();

        struct Value args = value_object();
        value_object_set(&args, "a", value_int(40LL));
        value_object_set(&args, "b", value_int(2LL));
        int ok = 0;
        struct Value res;
        tries = 0;
        while (1) {
            res = server_fn_call("127.0.0.1", g_fn_port, "add", &args, &ok);
            if (ok || tries >= 50) break;
            value_free(&res);
            thread_sleep_ms(20UL);
            tries = tries + 1;
        }
        ASSERT_TRUE(ok);
        ASSERT_EQ(res.as_int(), 42);
        value_free(&res);
        value_free(&args);
    }
}

void t_websocket() {
    start_servers();
    unsafe {
        int ok = 0;
        struct WsConn c;
        int tries = 0;
        while (1) {
            c = ws_connect("127.0.0.1", g_ws_port, "/", &ok);
            if (ok || tries >= 50) break;
            thread_sleep_ms(20UL);
            tries = tries + 1;
        }
        ASSERT_TRUE(ok);
        if (!ok) return;
        ASSERT_EQ(ws_send_text(&c, "hi there"), 1);
        struct WsMessage m;
        ASSERT_EQ(ws_recv(&c, &m), 1);
        ASSERT_EQ(m.opcode, WS_OPCODE_TEXT);
        ASSERT_EQ(m.len, 13);
        match (m.data) {
            case null: { ASSERT_TRUE(0); }
            case some(d): {
                ASSERT_EQ(d[0], 'e');
                ASSERT_EQ(d[12], 'e');
            }
        }
        m.free();
        // A payload over 125 bytes exercises the 16-bit extended length.
        struct String big = string_repeat("x", 300UL);
        ASSERT_EQ(ws_send_text(&c, big.as_ptr()), 1);
        ASSERT_EQ(ws_recv(&c, &m), 1);
        ASSERT_EQ(m.len, 305);
        m.free();
        big.free();
        ASSERT_EQ(ws_send_ping(&c), 1);
        ASSERT_EQ(ws_recv(&c, &m), 1);
        ASSERT_EQ(m.opcode, WS_OPCODE_PONG);
        m.free();
        ws_send_close(&c);
        ws_close(&c);
    }
}

extern int close(int fd);

static long long ms_since(long long t0) {
    return (time_mono_ns() - t0) / 1000000LL;
}

// connect()/handshake timeouts: none of these may take anywhere near the
// kernel's ~130s SYN-retry limit or block forever on a silent server.
void t_connect_timeout() {
    start_servers();
    unsafe {
        unsigned int lo = sched_ipv4((unsigned char)127, (unsigned char)0,
                                     (unsigned char)0, (unsigned char)1);
        unsigned short closedPort = (unsigned short)(g_http_port + (unsigned short)10);
        unsigned short silentPort = (unsigned short)(g_http_port + (unsigned short)11);

        // Refused: reported as a failure right away.
        long long t0 = time_mono_ns();
        int fd = tcp_connect_timeout(lo, closedPort, 2000);
        ASSERT_EQ(fd, -1);
        ASSERT_TRUE(ms_since(t0) < 1000LL);

        // Unanswered SYN (10.255.255.1 is normally blackholed; where it is
        // unroutable or rejected instead, connect fails even sooner).
        t0 = time_mono_ns();
        fd = tcp_connect_timeout(sched_ipv4((unsigned char)10, (unsigned char)255,
                                            (unsigned char)255, (unsigned char)1),
                                 (unsigned short)80, 300);
        ASSERT_EQ(fd, -1);
        ASSERT_TRUE(ms_since(t0) < 2000LL);

        // Success path returns a usable blocking socket.
        int ok = 0;
        struct HttpResponse warm = get_retry("/echo", &ok);   // server is up
        warm.free();
        fd = tcp_connect_timeout(lo, g_http_port, 2000);
        ASSERT_TRUE(fd >= 0);
        if (fd >= 0) close(fd);

        // A listener that never accept()s: the kernel completes the TCP
        // handshake, then nothing ever answers the request.
        int lfd = tcp_listen_nb(silentPort);
        ASSERT_TRUE(lfd >= 0);
        if (lfd < 0) return;

        t0 = time_mono_ns();
        struct HttpResponse r = http_request_timeout("127.0.0.1", silentPort, "GET", "/",
                                                     null, null, 0UL, 2000, 300, &ok);
        ASSERT_FALSE(ok);
        ASSERT_TRUE(ms_since(t0) < 2000LL);
        r.free();

        t0 = time_mono_ns();
        struct WsConn c = ws_connect_timeout("127.0.0.1", silentPort, "/", 300, &ok);
        ASSERT_FALSE(ok);
        ASSERT_EQ(c.fd, -1);
        ASSERT_TRUE(ms_since(t0) < 2000LL);
        close(lfd);

        // The same limits don't get in the way of a server that answers.
        r = http_request_timeout("127.0.0.1", g_http_port, "GET", "/echo",
                                 null, null, 0UL, 2000, 2000, &ok);
        ASSERT_TRUE(ok);
        ASSERT_EQ(r.status, 200);
        r.free();
    }
}

// Accepts connections and closes them at once without reading, so the
// client's next writes hit a reset connection.
static unsigned short g_closer_port = (unsigned short)0;
static ?&int run_closer(?&int arg) {
    unsafe {
        int lfd = tcp_listen_nb(g_closer_port);
        if (lfd < 0) return null;
        fd_set_blocking(lfd);
        while (1) {
            int cfd = tcp_accept_nb(lfd);
            if (cfd >= 0) close(cfd);
        }
    }
    return null;
}

// Writing to a peer that has closed must fail the request, not kill the
// process with SIGPIPE.
void t_write_to_closed_peer() {
    start_servers();
    unsafe {
        g_closer_port = (unsigned short)(g_http_port + (unsigned short)12);
        unsigned long long t = 0ULL;
        thread_create(&t, run_closer, null); thread_detach(t);
        unsigned long n = 4UL * 1024UL * 1024UL;
        struct String body = string_repeat("x", n);
        int ok = 1;
        int tries = 0;
        // Retry while the closer thread comes up (connect refused until then).
        while (1) {
            struct HttpResponse r = http_request_timeout("127.0.0.1", g_closer_port, "POST", "/",
                                                         null, (const unsigned char*)body.as_ptr(), n,
                                                         2000, 2000, &ok);
            r.free();
            tries = tries + 1;
            if (tries >= 20) break;
            thread_sleep_ms(20UL);
        }
        ASSERT_FALSE(ok);
        body.free();
    }
}

static ?&int resolve_worker(?&int arg) {
    unsafe {
        unsigned int want = sched_ipv4((unsigned char)127, (unsigned char)0,
                                       (unsigned char)0, (unsigned char)1);
        int i = 0;
        while (i < 200) {
            unsigned int a = 0U;
            if (!http_resolve_host("localhost", &a) || a != want) {
                match (arg) { case some(bad): { *bad = 1; } case null: { } }
            }
            i = i + 1;
        }
    }
    return null;
}

// Hostname resolution, including from several threads at once (the old
// gethostbyname path shared one static result buffer across threads).
void t_resolve() {
    start_servers();
    unsafe {
        unsigned int a = 0U;
        ASSERT_TRUE(http_resolve_host("localhost", &a));
        ASSERT_EQ(a, sched_ipv4((unsigned char)127, (unsigned char)0,
                                (unsigned char)0, (unsigned char)1));
        ASSERT_TRUE(http_resolve_host("10.1.2.3", &a));
        ASSERT_EQ(a, sched_ipv4((unsigned char)10, (unsigned char)1,
                                (unsigned char)2, (unsigned char)3));

        int bad = 0;
        unsigned long long th[4];
        int k = 0;
        while (k < 4) { thread_create(&th[k], resolve_worker, &bad); k = k + 1; }
        k = 0;
        while (k < 4) { thread_join(th[k]); k = k + 1; }
        ASSERT_EQ(bad, 0);

        int ok = 0;
        struct HttpResponse r = http_get("localhost", g_http_port, "/echo", &ok);
        ASSERT_TRUE(ok);
        ASSERT_EQ(r.status, 200);
        r.free();
    }
}

extern unsigned int alarm(unsigned int seconds);

int main() {
    // Watchdog: a networking regression should fail this test, not hang
    // the whole 'safeguard test' run (SIGALRM's default action terminates).
    unsafe { alarm(60u); }
    struct TestSuite t = test_suite_init();
    unsafe {
        t.add("http client/server", t_http_roundtrip);
        t.add("cors", t_cors);
        t.add("jwt hs256", t_jwt);
        t.add("oauth2 code exchange", t_oauth2);
        t.add("rpc + server_fn", t_rpc);
        t.add("websocket", t_websocket);
        t.add("connect/handshake timeouts", t_connect_timeout);
        t.add("write to closed peer", t_write_to_closed_peer);
        t.add("host resolution", t_resolve);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
