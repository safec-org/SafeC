// The SChannel TLS backend (std/tls/tls_schannel_win32.sc) and the SafeC
// backend on Windows: against a TLS server on 127.0.0.1:$SAFEC_TLS_TEST_PORT
// (run.sh starts `openssl s_server` with tests/stdlib/fixtures/tls's
// chain-ec.pem when openssl is there), with the test root from
// $SAFEC_TLS_TEST_ROOT. Without those it checks only that the backends load.
#include <std/test/test.h>
#include <std/str.h>
#include <std/sys.h>
#include <std/sched/io_nb.h>
#include <std/tls/tls.h>
#include <std/tls/tls_backend.h>

extern int printf(const char* f, ...);
extern char* getenv(const char* name);
extern int atoi(const char* s);
extern int closesocket(unsigned long long fd);

static int g_port = 0;
static const char* g_root = (const char*)0;

static int connect_port() {
    int fd = tcp_connect_timeout(sched_ipv4((unsigned char)127, (unsigned char)0, (unsigned char)0, (unsigned char)1),
                                 (unsigned short)g_port, 5000);
    if (fd < 0) return -1;
    fd_set_blocking(fd);
    sock_set_io_timeout(fd, 10000);
    return fd;
}

// The code of the failure, 0 on a full request/response.
static int fetch(int backend, const &char host, int verify, const ?&char alpn) {
    struct TlsConfig cfg = TlsConfig::new();
    cfg.backend = backend;
    if (verify) {
        cfg.use_system_roots = 0;
        unsafe { cfg.set_ca_file((const &char)g_root); }
    } else {
        cfg.insecure_skip_verify();
    }
    match (alpn) { case some(a): { cfg.set_alpn(a); } case null: { } }
    int fd = connect_port();
    if (fd < 0) { cfg.free(); printf("    connect failed\n"); return TLS_ERR_IO; }
    struct Result<TlsStream, TlsError> r = TlsStream::connect(fd, host, &cfg);
    cfg.free();
    if (!r.ok()) {
        printf("    [%d] %s: %s\n", backend, host, r.error.message);
        unsafe { closesocket((unsigned long long)fd); }
        return r.error.code;
    }
    struct TlsStream t = r.value;
    printf("    [%s] %s %x %s alpn='%s'\n", t.backend_name(), host, t.version(), t.cipher(), t.alpn());
    int code = TLS_OK;
    unsafe {
        const &char req = "GET / HTTP/1.0\r\n\r\n";
        if (t.write(str_bytes(req)) <= 0LL) { code = TLS_ERR_IO; }
        unsigned char buf[8192];
        long long total = 0LL;
        while (total < 8000LL) {
            long long n = t.read(buf[(unsigned long)total..8000UL]);
            if (n <= 0LL) break;
            total = total + n;
        }
        buf[total] = (unsigned char)0;
        if (code == TLS_OK && str_ncmp((const char*)&buf[0], "HTTP/1.0 200", 12UL) != 0) { code = TLS_ERR_PROTOCOL; }
        match (alpn) {
            case some(a): { if (code == TLS_OK && !str_eq(t.alpn(), a)) { printf("    alpn mismatch\n"); code = TLS_ERR_PROTOCOL; } }
            case null: { }
        }
        struct X509Cert pc;
        if (code == TLS_OK && (!t.peer_certificate(&pc) || !str_eq((const char*)&pc.subject.cn[0], "localhost"))) { code = TLS_ERR_NO_CERT; }
    }
    t.close();
    return code;
}

void t_backends() {
    ASSERT_EQ(tls_backend_available(TLS_IMPL_SCHANNEL), 1);
    ASSERT_EQ(tls_backend_available(TLS_IMPL_OPENSSL), 0);
    ASSERT_EQ(tls_default_impl(), TLS_IMPL_SCHANNEL);
    // the Windows ROOT store reaches the SafeC backend
    ?&TlsTrust_ sys = tls_trust_system_();
    printf("    system roots for the SafeC backend: %d\n", sys.is_null() ? 0 : 1);
}

void t_schannel() {
    if (g_port == 0) { printf("    SKIP: no SAFEC_TLS_TEST_PORT\n"); return; }
    int sch = TLS_BACKEND_IMPL_BASE + TLS_IMPL_SCHANNEL;
    ASSERT_EQ(fetch(sch, "localhost", 1, null), TLS_OK);
    ASSERT_EQ(fetch(sch, "localhost", 1, "http/1.1"), TLS_OK);
    ASSERT_EQ(fetch(sch, "127.0.0.1", 0, null), TLS_OK);
    ASSERT_EQ(fetch(sch, "wrong.example", 1, null), TLS_ERR_CERT_HOSTNAME);
    // the SafeC backend on Windows
    ASSERT_EQ(fetch(TLS_BACKEND_SAFEC, "localhost", 1, "http/1.1"), TLS_OK);
}

int main() {
    unsafe {
        const char* p = getenv("SAFEC_TLS_TEST_PORT");
        if (p != (const char*)0) { g_port = atoi(p); }
        g_root = getenv("SAFEC_TLS_TEST_ROOT");
        if (g_root == (const char*)0) { g_port = 0; }
    }
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("backends", t_backends);
        t.add("schannel + safec against a local server", t_schannel);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
