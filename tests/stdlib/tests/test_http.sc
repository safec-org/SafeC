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
#include <std/io_traits.h>
#include <std/io_buf.h>
#include <std/ipc/pipe.h>

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
            r.headers = String::from("Content-Type: application/json");
            r.body = String::from(tok);
            return r;
        }
        if (req.path.eq_cstr("/echo")) {
            r.status = 200;
            r.headers = String::from("Content-Type: text/plain\r\nX-Method: ");
            r.headers.push(req.method.as_ptr());
            r.body = req.body.clone();
            return r;
        }
        if (req.path.eq_cstr("/cors")) {
            r.status = 200;
            r.headers = String::new();
            r.body = String::from("ok");
            cors_apply_headers(&r, "https://a.example");
            return r;
        }
        r.status = 404;
        r.headers = String::new();
        r.body = String::from("nope");
    }
    return r;
}

static ?&int run_http(?&int arg) {
    unsafe { http_serve(g_http_port, handler); }
    return null;
}

static struct RpcResult rpc_upper([]const unsigned char data) {
    unsigned long len = data.len;
    struct RpcResult r;
    unsafe {
        ?&heap unsigned char out = (?&heap unsigned char)alloc(len);
        r.ok = 1;
        r.len = len;
        r.error_message = String::new();
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
    return Value::from_int(a + b);
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
        while (1) {   // one client at a time, for as long as tests connect
        struct WsConn c = WsConn::closed();
        if (c.accept(lfd) != 1) { c.drop(); continue; }   // 1 = handshake completed; a failed one is closed
        struct WsMessage m;
        while (c.recv(&m) == 1) {
            if (m.opcode == WS_OPCODE_TEXT) {
                struct String echo = String::from("echo:");
                echo.push_bytes(((const char*)m.data)[0..m.len]);
                c.send_text(echo.as_ptr());
                echo.free();
            } else if (m.opcode == WS_OPCODE_PING) {
                c.send_pong();
            } else if (m.opcode == WS_OPCODE_CLOSE) {
                m.free();
                break;
            }
            m.free();
        }
        c.close();
        }
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
        r = http_post("127.0.0.1", g_http_port, "/echo", "text/plain", str_bytes(body), &ok);
        ASSERT_TRUE(ok);
        ASSERT_STR_EQ(r.body.as_ptr(), "posted payload");
        r.free();

        // A body many reads long, both ways (server and client read it
        // after the header block through a BufReader<TcpStream>).
        struct String large = String::repeat_str("0123456789abcdef", 6250UL);   // 100000 bytes
        r = http_post("127.0.0.1", g_http_port, "/echo", "text/plain", large.as_bytes(), &ok);
        ASSERT_TRUE(ok);
        ASSERT_EQ(r.body.length(), 100000);
        ASSERT_TRUE(r.body.eq_cstr(large.as_ptr()));
        r.free();
        large.free();

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
        struct String raw = http_build_response(200, "text/plain", str_bytes("hi"));
        ASSERT_TRUE(raw.starts_with("HTTP/1.1 200"));
        ASSERT_TRUE(raw.ends_with("\r\n\r\nhi"));
        raw.free();
        unsigned int addr = 0u;
        ASSERT_EQ(http_resolve_host("127.0.0.1", &addr), 1);
        ASSERT_EQ(http_resolve_host("localhost", &addr), 1);
    }
}

// TcpStream through io_traits.h: a hand-written request with write_all,
// the status line with read_exact, the rest with copy_stream into a Pipe.
void t_tcp_stream() {
    start_servers();
    unsafe {
        int ok = 0;
        struct HttpResponse warm = get_retry("/echo", &ok);   // server is up
        warm.free();
        unsigned int lo = sched_ipv4((unsigned char)127, (unsigned char)0,
                                     (unsigned char)0, (unsigned char)1);
        int fd = tcp_connect_timeout(lo, g_http_port, 2000);
        ASSERT_TRUE(fd >= 0);
        if (fd < 0) return;
        struct TcpStream s = TcpStream::from_fd(fd);
        const &char req = "POST /echo HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello";
        ASSERT_EQ(write_all(&s, str_bytes(req)), 1);
        unsigned char status[12];
        ASSERT_EQ(read_exact(&s, status[0..12UL]), 1);
        ASSERT_EQ(status[9], '2');
        ASSERT_EQ(status[11], '0');
        struct Pipe p;
        ASSERT_EQ(p.init(), 0);
        long long rest = copy_stream(&s, &p);       // until the server closes
        ASSERT_TRUE(rest > 5LL);
        ASSERT_EQ(s.close(), 0);
        p.close_write();
        unsigned char tail[4096];
        long long n = p.read(tail[0..4096UL]);
        ASSERT_EQ(n, rest);
        if (n >= 5LL) { ASSERT_EQ(tail[n - 1LL], 'o'); }   // body "hello" last
        p.close_read();
    }
}

// A request/response exchange over any Stream, buffered both ways: the
// request goes out in pieces through a BufWriter<S> (one write on flush),
// the response's header block comes back with BufReader<S>::read_until and
// its Content-Length body with read_exact on the same BufReader. Returns
// the HTTP status (0 on failure); the body goes to 'body'.
generic<S: Stream>
int buffered_exchange(&S s, const &char path, const &char payload, &String body) {
    struct BufWriter<S> out = BufWriter::over(s);
    struct String hdr = String::new();
    hdr.push("POST ");
    hdr.push(path);
    hdr.push(" HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Length: ");
    hdr.push_uint((unsigned long long)str_len(payload));
    hdr.push("\r\n\r\n");
    int ok = 1;
    unsafe {
        ok = write_all(&out, hdr.as_bytes())
          && write_all(&out, str_bytes(payload))
          && out.flush() == 0;
    }
    hdr.free();
    if (!ok) return 0;

    struct BufReader<S> in = BufReader::over(s);
    struct String head = String::new();
    int status = 0;
    if (in.read_until(&head, "\r\n\r\n") && head.starts_with("HTTP/1.1 ")) {
        int pok = 0;
        struct String code = head.substr(9UL, 12UL);
        status = (int)code.parse_int(&pok);
        code.free();
        struct String cl = http_header_get(head.as_ptr(), "Content-Length");
        unsigned long n = (unsigned long)cl.parse_int(&pok);
        cl.free();
        unsigned char buf[256];
        if (n > 256UL || !read_exact(&in, buf[0..n])) {
            status = 0;
        } else {
            body.push_bytes(buf[0..n]);
        }
    }
    head.free();
    return status;
}

void t_tcp_buffered() {
    start_servers();
    unsafe {
        int ok = 0;
        struct HttpResponse warm = get_retry("/echo", &ok);
        warm.free();
        unsigned int lo = sched_ipv4((unsigned char)127, (unsigned char)0,
                                     (unsigned char)0, (unsigned char)1);
        int fd = tcp_connect_timeout(lo, g_http_port, 2000);
        ASSERT_TRUE(fd >= 0);
        if (fd < 0) return;
        struct TcpStream s = TcpStream::from_fd(fd);
        struct String body = String::new();
        ASSERT_EQ(buffered_exchange(&s, "/echo", "buffered hello", &body), 200);
        ASSERT_TRUE(body.eq_cstr("buffered hello"));
        body.free();
        ASSERT_EQ(s.close(), 0);
    }
}

void t_cors() {
    unsafe {
        struct HttpRequest req;
        req.method = String::from("OPTIONS");
        req.path = String::from("/api");
        req.headers = String::from("Origin: https://a.example\r\nAccess-Control-Request-Method: POST");
        req.body = String::new();
        ASSERT_TRUE(cors_is_preflight(&req));
        req.method.free();
        req.method = String::from("GET");
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
        struct Value claims = Value::object();
        claims.object_set("sub", Value::from_str("1234567890"));
        claims.object_set("admin", Value::from_bool(1));
        struct String tok = jwt_sign(&claims, "secret");
        ASSERT_EQ(tok.count("."), 2);
        ASSERT_TRUE(tok.starts_with("eyJ"));   // base64url of '{"'
        int ok = 0;
        struct Value back = jwt_verify(tok.as_ptr(), "secret", &ok);
        ASSERT_TRUE(ok);
        ?&Value sub = back.object_get("sub");
        ASSERT_FALSE(sub.is_null());
        ASSERT_STR_EQ(sub->as_string(), "1234567890");
        back.free();
        struct Value bad = jwt_verify(tok.as_ptr(), "wrong", &ok);
        ASSERT_FALSE(ok);
        bad.free();
        // Tamper with the payload: signature must no longer verify.
        tok.set_char((unsigned long)tok.index_of(".") + 2UL, 'A');
        struct Value bad2 = jwt_verify(tok.as_ptr(), "secret", &ok);
        ASSERT_FALSE(ok);
        bad2.free();
        tok.free();
        claims.free();
    }
}

void t_oauth2() {
    start_servers();
    unsafe {
        int ok = 0;
        struct HttpResponse warm = get_retry("/echo", &ok);
        warm.free();
        struct OAuth2Token t = OAuth2Token::exchange_code("127.0.0.1", g_http_port, "/token", "cid", "csecret",
                                                          "code42", "http://localhost/cb", &ok);
        ASSERT_TRUE(ok);
        ASSERT_STR_EQ(t.accessToken.as_ptr(), "abc123");
        ASSERT_STR_EQ(t.refreshToken.as_ptr(), "r1");
        ASSERT_EQ(t.expiresIn, 3600);
        t.free();
    }
}

void t_rpc() {
    start_servers();
    unsafe {
        struct RpcResult r;
        int tries = 0;
        while (1) {
            r = rpc_call("127.0.0.1", g_rpc_port, "/upper", str_bytes("hello rpc"));
            if (r.ok || tries >= 50) break;
            r.free();
            thread_sleep_ms(20UL);
            tries = tries + 1;
        }
        ASSERT_TRUE(r.ok);
        ASSERT_EQ(r.len, 9);
        ASSERT_EQ(r.bytes().len, 9);
        ASSERT_EQ(r.bytes()[1], 'E');
        match (r.data) {
            case null: { ASSERT_TRUE(0); }
            case some(d): {
                ASSERT_EQ(d[0], 'H');
                ASSERT_EQ(d[8], 'C');
            }
        }
        r.free();
        struct RpcResult miss = rpc_call("127.0.0.1", g_rpc_port, "/nope", str_bytes("x"));
        ASSERT_FALSE(miss.ok);
        miss.free();

        struct Value args = Value::object();
        args.object_set("a", Value::from_int(40LL));
        args.object_set("b", Value::from_int(2LL));
        int ok = 0;
        struct Value res;
        tries = 0;
        while (1) {
            res = server_fn_call("127.0.0.1", g_fn_port, "add", &args, &ok);
            if (ok || tries >= 50) break;
            res.free();
            thread_sleep_ms(20UL);
            tries = tries + 1;
        }
        ASSERT_TRUE(ok);
        ASSERT_EQ(res.as_int(), 42);
        res.free();
        args.free();
    }
}

// The client's first frame in the same write as its Upgrade request: the
// server reads both in one go, and the frame must still reach recv() (it
// used to be dropped with the handshake's read buffer).
void t_ws_frame_with_handshake() {
    start_servers();
    unsafe {
        int fd = -1;
        int tries = 0;
        while (tries < 50) {
            fd = tcp_connect_nb(sched_ipv4((unsigned char)127, (unsigned char)0, (unsigned char)0, (unsigned char)1),
                                g_ws_port);
            if (fd >= 0 && tcp_wait_connected(fd) == 0) break;
            if (fd >= 0) { struct TcpStream bad = TcpStream::from_fd(fd); bad.close(); fd = -1; }
            thread_sleep_ms(20UL);
            tries = tries + 1;
        }
        ASSERT_TRUE(fd >= 0);
        if (fd < 0) return;
        fd_set_blocking(fd);
        struct TcpStream s = TcpStream::from_fd(fd);
        struct String out = String::from("GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nUpgrade: websocket\r\n"
                                         "Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                                         "Sec-WebSocket-Version: 13\r\n\r\n");
        // FIN + text, masked, length 2, mask key 0 (payload unchanged), "hi"
        unsigned char frame[8];
        frame[0] = (unsigned char)0x81; frame[1] = (unsigned char)0x82;
        frame[2] = (unsigned char)0; frame[3] = (unsigned char)0; frame[4] = (unsigned char)0; frame[5] = (unsigned char)0;
        frame[6] = (unsigned char)'h'; frame[7] = (unsigned char)'i';
        out.push_bytes(frame[0..8]);
        ASSERT_TRUE(write_all(&s, out.as_bytes()));
        out.free();

        struct BufReader<TcpStream> in = BufReader::over(&s);
        struct String head = String::new();
        ASSERT_TRUE(in.read_until(&head, "\r\n\r\n"));
        ASSERT_TRUE(head.index_of("101") >= 0L);
        head.free();
        unsigned char reply[9];   // unmasked text frame "echo:hi"
        ASSERT_TRUE(read_exact(&in, reply[0..9UL]));
        ASSERT_EQ(reply[0], (unsigned char)0x81);
        ASSERT_EQ(reply[1], (unsigned char)7);
        ASSERT_EQ(reply[2], (unsigned char)'e');
        ASSERT_EQ(reply[7], (unsigned char)'h');
        ASSERT_EQ(reply[8], (unsigned char)'i');
        s.close();
    }
}

// A WsConn that never connected — WsConn::closed(), or a zeroed one after a
// failed accept() — is closed (fd -1): dropping it neither writes a close
// frame to nor closes fd 0. fd 0 is pointed at a pipe here to watch.
extern int dup(int fd);
extern int dup2(int fd, int fd2);

void t_ws_closed() {
    unsafe {
        struct WsConn z = WsConn::closed();
        ASSERT_EQ(z.fd, -1);
        ASSERT_EQ(z.send_text("x"), 0);
        z.drop();

        struct Pipe p;
        ASSERT_EQ(p.init(), 0);
        ASSERT_EQ(p.set_read_nonblocking(), 0);
        int saved = dup(0);
        ASSERT_TRUE(dup2((int)p.write_fd, 0) == 0);

        int lfd = tcp_listen_nb((unsigned short)0);   // nobody connects: accept fails at once
        ASSERT_TRUE(lfd >= 0);
        struct WsConn c;
        safe_memset_obj(&c, 0);                       // fd 0, as a zero-initialised conn
        ASSERT_EQ(c.accept(lfd), 0);
        ASSERT_EQ(c.fd, -1);
        c.drop();
        close(lfd);

        unsigned char buf[8];
        ASSERT_EQ(p.read(buf[0..8UL]), -1);           // nothing was written to fd 0 ...
        ASSERT_TRUE(dup2(0, 0) == 0);                 // ... and it is still open

        if (saved >= 0) { dup2(saved, 0); close(saved); } else { close(0); }
        p.close_read();
        p.close_write();
    }
}

void t_websocket() {
    start_servers();
    unsafe {
        int ok = 0;
        struct WsConn c;
        int tries = 0;
        while (1) {
            c = WsConn::connect("127.0.0.1", g_ws_port, "/", &ok);
            if (ok || tries >= 50) break;
            thread_sleep_ms(20UL);
            tries = tries + 1;
        }
        ASSERT_TRUE(ok);
        if (!ok) return;
        ASSERT_EQ(c.send_text("hi there"), 1);
        struct WsMessage m;
        ASSERT_EQ(c.recv(&m), 1);
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
        struct String big = String::repeat_str("x", 300UL);
        ASSERT_EQ(c.send_text(big.as_ptr()), 1);
        ASSERT_EQ(c.recv(&m), 1);
        ASSERT_EQ(m.len, 305);
        m.free();
        big.free();
        // Over 64 KB: the 64-bit length, and a masked payload several
        // mask chunks long.
        struct String huge = String::repeat_str("0123456789", 7000UL);
        ASSERT_EQ(c.send_text(huge.as_ptr()), 1);
        ASSERT_EQ(c.recv(&m), 1);
        ASSERT_EQ(m.len, 70005);
        match (m.data) {
            case null: { ASSERT_TRUE(0); }
            case some(d): {
                ASSERT_EQ(d[5], '0');
                ASSERT_EQ(d[5 + 16389], '9');
                ASSERT_EQ(d[70004], '9');
            }
        }
        m.free();
        huge.free();
        ASSERT_EQ(c.send_ping(), 1);
        ASSERT_EQ(c.recv(&m), 1);
        ASSERT_EQ(m.opcode, WS_OPCODE_PONG);
        m.free();
        c.send_close();
        c.close();
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
        int ok = 0;
        struct HttpResponse warm = get_retry("/echo", &ok);   // server is up
        warm.free();

        // Refused: reported as a failure right away.
        long long t0 = time_mono_ns();
        int fd = tcp_connect_timeout(lo, closedPort, 2000);
        ASSERT_EQ(fd, -1);
        ASSERT_TRUE(ms_since(t0) < 1000LL);

        // tcp_wait_connected (no time limit of its own) on the same two
        // outcomes.
        fd = tcp_connect_nb(lo, closedPort);
        ASSERT_TRUE(fd >= 0);
        if (fd >= 0) { ASSERT_EQ(tcp_wait_connected(fd), -1); close(fd); }
        fd = tcp_connect_nb(lo, g_http_port);
        ASSERT_TRUE(fd >= 0);
        if (fd >= 0) { ASSERT_EQ(tcp_wait_connected(fd), 0); close(fd); }

        // Unanswered SYN (10.255.255.1 is normally blackholed; where it is
        // unroutable or rejected instead, connect fails even sooner).
        t0 = time_mono_ns();
        fd = tcp_connect_timeout(sched_ipv4((unsigned char)10, (unsigned char)255,
                                            (unsigned char)255, (unsigned char)1),
                                 (unsigned short)80, 300);
        ASSERT_EQ(fd, -1);
        ASSERT_TRUE(ms_since(t0) < 2000LL);

        // Success path returns a usable blocking socket.
        fd = tcp_connect_timeout(lo, g_http_port, 2000);
        ASSERT_TRUE(fd >= 0);
        if (fd >= 0) close(fd);

        // A listener that never accept()s: the kernel completes the TCP
        // handshake, then nothing ever answers the request.
        int lfd = tcp_listen_nb(silentPort);
        ASSERT_TRUE(lfd >= 0);
        if (lfd < 0) return;

        t0 = time_mono_ns();
        struct HttpResponse r = http_request_timeout("127.0.0.1", silentPort, "GET", "/", null, str_bytes(""), 2000, 300, &ok);
        ASSERT_FALSE(ok);
        ASSERT_TRUE(ms_since(t0) < 2000LL);
        r.free();

        t0 = time_mono_ns();
        struct WsConn c = WsConn::connect_timeout("127.0.0.1", silentPort, "/", 300, &ok);
        ASSERT_FALSE(ok);
        ASSERT_EQ(c.fd, -1);
        ASSERT_TRUE(ms_since(t0) < 2000LL);
        close(lfd);

        // The same limits don't get in the way of a server that answers.
        r = http_request_timeout("127.0.0.1", g_http_port, "GET", "/echo", null, str_bytes(""), 2000, 2000, &ok);
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
        struct String body = String::repeat_str("x", n);
        int ok = 1;
        int tries = 0;
        // Retry while the closer thread comes up (connect refused until then).
        while (1) {
            struct HttpResponse r = http_request_timeout("127.0.0.1", g_closer_port, "POST", "/", null, ((const unsigned char*)body.as_ptr())[0..n], 2000, 2000, &ok);
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
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("http client/server", t_http_roundtrip);
        t.add("tcp stream (io traits)", t_tcp_stream);
        t.add("tcp stream through BufReader/BufWriter", t_tcp_buffered);
        t.add("cors", t_cors);
        t.add("jwt hs256", t_jwt);
        t.add("oauth2 code exchange", t_oauth2);
        t.add("rpc + server_fn", t_rpc);
        t.add("websocket", t_websocket);
        t.add("websocket: closed / failed-accept conn", t_ws_closed);
        t.add("websocket frame sent with the handshake", t_ws_frame_with_handshake);
        t.add("connect/handshake timeouts", t_connect_timeout);
        t.add("write to closed peer", t_write_to_closed_peer);
        t.add("host resolution", t_resolve);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
