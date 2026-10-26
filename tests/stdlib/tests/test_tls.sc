// std::tls — the TLS front end and its backends, over loopback:
//   - the RFC 8448 1-RTT key schedule (X25519 share, HKDF-Expand-Label)
//   - X.509 parsing, host matching and path validation against the test PKI
//     in fixtures/tls (made by fixtures/tls/make_pki.sh)
//   - SafeC server <-> SafeC client (ALPN, RSA/ECDSA keys, client
//     certificates, verification failures, a non-blocking handshake)
//   - the OpenSSL backend (when libssl loads) against SafeC, both ways
//   - `openssl s_server` / `s_client` interop: cipher suites, groups (HRR),
//     ALPN, SNI, expired / untrusted / self-signed / wrong-host certificates,
//     a TLS 1.2-only server (skipped without the openssl binary)
//   - https:// http_fetch, wss:// WebSocket and https SSE against an HTTPS
//     HttpServer
#include <std/test/test.h>
#include <std/mem.h>
#include <std/str.h>
#include <std/sys.h>
#include <std/thread.h>
#include <std/sched/io_nb.h>
#include <std/collections/string.h>
#include <std/crypto/x25519.h>
#include <std/crypto/hkdf.h>
#include <std/crypto/sha256.h>
#include <std/crypto/digest.h>
#include <std/crypto/x509.h>
#include <std/tls/tls.h>
#include <std/tls/tls_backend.h>
#include <std/tls/tls_safec.h>
#include <std/http/http.h>
#include <std/http/router.h>
#include <std/http/websocket.h>
#include <std/http/sse.h>

extern int printf(const char* f, ...);
extern int sprintf(char* b, const char* f, ...);
extern int setenv(const char* name, const char* value, int overwrite);
extern int unsetenv(const char* name);
extern int close(int fd);

#define FX "fixtures/tls/"
// The OpenSSL backend, by name (TLS_BACKEND_OS is another library on
// macOS and Windows).
#define OSSL (TLS_BACKEND_IMPL_BASE + TLS_IMPL_OPENSSL)

static int g_have_openssl = -1;   // the openssl command

static int have_openssl_cli() {
    if (g_have_openssl < 0) {
        unsafe { g_have_openssl = sys_system("openssl version >/dev/null 2>&1") == 0; }
        if (!g_have_openssl) { printf("    (openssl not found: skipping the s_server/s_client interop cases)\n"); }
    }
    return g_have_openssl;
}

extern int getsockname(int fd, unsigned char* addr, unsigned int* len);

// The port a listening socket got (sockaddr_in: the port at byte 2).
static int bound_port(int fd) {
    unsigned char sa[16];
    unsigned int len = 16U;
    unsafe {
        if (getsockname(fd, (unsigned char*)&sa[0], &len) != 0) { return -1; }
    }
    return ((int)sa[2] << 8) | (int)sa[3];
}

// A currently free port (bind to 0, read it back, close).
static int next_port() {
    int fd = tcp_listen_nb((unsigned short)0);
    if (fd < 0) { return -1; }
    int p = bound_port(fd);
    close(fd);
    return p;
}

static int hexv(char c) {
    if (c >= '0' && c <= '9') return (int)(c - '0');
    return (int)(c - 'a') + 10;
}
static unsigned long hb(const &char h, &unsigned char out) {
    unsafe {
        unsigned long i = 0UL;
        while (h[2UL * i] != (char)0) {
            out[i] = (unsigned char)(hexv(h[2UL * i]) * 16 + hexv(h[2UL * i + 1UL]));
            i = i + 1UL;
        }
        return i;
    }
}
static void hx(const &unsigned char d, unsigned long n, &char out) {
    unsafe {
        const &char digits = "0123456789abcdef";
        unsigned long i = 0UL;
        while (i < n) { out[i * 2UL] = digits[d[i] >> 4]; out[i * 2UL + 1UL] = digits[d[i] & 15]; i = i + 1UL; }
        out[n * 2UL] = (char)0;
    }
}

static struct String read_fx(const &char name) {
    struct String path = String::from(FX);
    path.push(name);
    struct String s = String::new();
    tls_read_file_(path.as_ptr(), &s);
    path.free();
    return s;
}

static int connect_port(int port) {
    int fd = tcp_connect_timeout(sched_ipv4((unsigned char)127, (unsigned char)0, (unsigned char)0, (unsigned char)1),
                                 (unsigned short)port, 5000);
    if (fd < 0) return -1;
    fd_set_blocking(fd);
    sock_set_io_timeout(fd, 10000);
    return fd;
}

// ── RFC 8448 §3 (simple 1-RTT handshake) key schedule ────────────────────────

void t_rfc8448() {
    unsafe {
        unsigned char cpriv[32]; unsigned char spub[32]; unsigned char shared[32]; unsigned char pub[32];
        char h[200];
        hb("49af42ba7f7994852d713ef2784bcbcaa7911de26adc5642cb634540e7ea5005", cpriv);
        hb("c9828876112095fe66762bdbf7c672e156d6cc253b833df1dd69b1b04e751f0f", spub);
        x25519_public(pub[..], cpriv[..]);
        hx(pub, 32UL, h);
        ASSERT_STR_EQ(h, "99381de560e4bd43d23d8e435a7dbafeb3c06e51c13cae4d5413691e529aaf2c");
        ASSERT_EQ(x25519(shared[..], cpriv[..], spub[..]), 1);
        hx(shared, 32UL, h);
        ASSERT_STR_EQ(h, "8bd4054fb55b9d63fdfbacf9f04b9f0d35e6d63f537563efd46272900f89492d");
        unsigned char zeros[32]; unsigned char early[32]; unsigned char empty[32]; unsigned char derived[32];
        unsigned char hs[32]; unsigned char th[32]; unsigned char t[32];
        int i = 0; while (i < 32) { zeros[i] = (unsigned char)0; i = i + 1; }
        hkdf_extract(Sha256Ctx::new(), str_bytes(""), ((const unsigned char*)zeros)[0..32UL], early[..]);
        hx(early, 32UL, h);
        ASSERT_STR_EQ(h, "33ad0a1c607ec03b09e6cd9893680ce210adf300aa1f2660e1b22e10f170f92a");
        sha256(str_bytes(""), empty[..]);
        tls13_expand_label_(DIGEST_SHA256, early, 32UL, "derived", empty, 32UL, derived, 32UL);
        hx(derived, 32UL, h);
        ASSERT_STR_EQ(h, "6f2615a108c702c5678f54fc9dbab69716c076189c48250cebeac3576c3611ba");
        hkdf_extract(Sha256Ctx::new(), ((const unsigned char*)derived)[0..32UL], ((const unsigned char*)shared)[0..32UL], hs[..]);
        hx(hs, 32UL, h);
        ASSERT_STR_EQ(h, "1dc826e93606aa6fdc0aadc12f741b01046aa6b99f691ed221a9f0ca043fbeac");
        // Transcript-Hash(ClientHello, ServerHello) from the trace.
        hb("860c06edc07858ee8e78f0e7428c58edd6b43f2ca3e6e95f02ed063cf0e1cad8", th);
        tls13_expand_label_(DIGEST_SHA256, hs, 32UL, "c hs traffic", th, 32UL, t, 32UL);
        hx(t, 32UL, h);
        ASSERT_STR_EQ(h, "b3eddb126e067f35a780b3abf45e2d8f3b1a950738f52e9600746a0e27a55a21");
        tls13_expand_label_(DIGEST_SHA256, hs, 32UL, "s hs traffic", th, 32UL, t, 32UL);
        hx(t, 32UL, h);
        ASSERT_STR_EQ(h, "b67b7d690cc16c4e75e54213cb2d37b4e9c912bcded9105d42befd59d391ad38");
        tls13_expand_label_(DIGEST_SHA256, hs, 32UL, "derived", empty, 32UL, derived, 32UL);
        unsigned char master[32];
        hkdf_extract(Sha256Ctx::new(), ((const unsigned char*)derived)[0..32UL], ((const unsigned char*)zeros)[0..32UL], master[..]);
        hx(master, 32UL, h);
        ASSERT_STR_EQ(h, "18df06843d13a08bf2a449844c5f8a478001bc4d4c627984d5a41da8d0402919");
        // The server handshake traffic key and IV (AES-128-GCM).
        unsigned char key[16]; unsigned char iv[12];
        tls13_expand_label_(DIGEST_SHA256, t, 32UL, "key", null, 0UL, key, 16UL);
        tls13_expand_label_(DIGEST_SHA256, t, 32UL, "iv", null, 0UL, iv, 12UL);
        hx(key, 16UL, h);
        ASSERT_STR_EQ(h, "3fce516009c21727d0f2e4e86ee403bc");
        hx(iv, 12UL, h);
        ASSERT_STR_EQ(h, "5d313eb2671276ee13000b30");
    }
}

// ── X.509 ─────────────────────────────────────────────────────────────────────

static int parse_fx(const &char name, &Vec<struct Vec<unsigned char>> ders) {
    struct String pem = read_fx(name);
    unsigned long n = tls_pem_decode_(pem.as_ptr(), "CERTIFICATE", ders);
    pem.free();
    return (int)n;
}

void t_x509() {
    unsafe {
        struct Vec<struct Vec<unsigned char>> c = Vec::new();
        ASSERT_EQ(parse_fx("chain-ec.pem", &c), 2);
        struct X509Parsed leaf;
        struct X509Parsed ica;
        &Vec<unsigned char> l = c.at(0UL);
        &Vec<unsigned char> i = c.at(1UL);
        []const unsigned char ld = ((const unsigned char*)l.data)[0..l.len];
        []const unsigned char id = ((const unsigned char*)i.data)[0..i.len];
        ASSERT_EQ(x509_parse(ld[0..l.len], &leaf), 0);
        ASSERT_EQ(x509_parse(id[0..i.len], &ica), 0);
        ASSERT_EQ(leaf.key_type, X509_KEY_EC_P256);
        ASSERT_EQ(leaf.sig_type, X509_SIG_ECDSA);
        ASSERT_EQ(leaf.sig_hash, DIGEST_SHA384);
        ASSERT_EQ(ica.key_type, X509_KEY_EC_P384);
        ASSERT_EQ(ica.sig_type, X509_SIG_RSA_PKCS1);
        ASSERT_TRUE(ica.is_ca);
        ASSERT_FALSE(leaf.is_ca);
        ASSERT_TRUE(leaf.has_eku && leaf.eku_server && !leaf.eku_client);
        ASSERT_TRUE((leaf.key_usage & X509_KU_DIGITAL_SIGNATURE) != 0);
        ASSERT_TRUE(leaf.not_after > 4000000000ULL);   // ~2122: GeneralizedTime
        ASSERT_EQ(x509_issued_by(ld[..], &leaf, id[..], &ica), 1);
        ASSERT_EQ(x509_verify_signed_by(ld[..], &leaf, id[..], &ica), 1);
        ASSERT_EQ(x509_verify_signed_by(id[..], &ica, ld[..], &leaf), 0);
        // host names: SAN DNS (exact, wildcard), IPv4, IPv6; no CN fallback with SANs
        ASSERT_EQ(x509_match_host(ld[..], &leaf, "localhost"), 1);
        ASSERT_EQ(x509_match_host(ld[..], &leaf, "LocalHost."), 1);
        ASSERT_EQ(x509_match_host(ld[..], &leaf, "a.test.example"), 1);
        ASSERT_EQ(x509_match_host(ld[..], &leaf, "a.b.test.example"), 0);
        ASSERT_EQ(x509_match_host(ld[..], &leaf, "test.example"), 0);
        ASSERT_EQ(x509_match_host(ld[..], &leaf, "127.0.0.1"), 1);
        ASSERT_EQ(x509_match_host(ld[..], &leaf, "127.0.0.2"), 0);
        ASSERT_EQ(x509_match_host(ld[..], &leaf, "::1"), 1);
        ASSERT_EQ(x509_match_host(ld[..], &leaf, "[0:0::1]"), 1);
        ASSERT_EQ(x509_match_host(ld[..], &leaf, "::2"), 0);
        ASSERT_EQ(x509_match_host(ld[..], &leaf, "example.com"), 0);
        // the X509Cert summary works on the same bytes
        struct X509Cert sum;
        ASSERT_EQ(sum.init(ld[0..l.len]), 0);
        ASSERT_STR_EQ(sum.subject.cn, "localhost");
        ASSERT_STR_EQ(sum.issuer.cn, "SafeC Test Intermediate");
        // RSA-PSS signed leaf
        struct Vec<struct Vec<unsigned char>> r = Vec::new();
        ASSERT_EQ(parse_fx("leaf-rsa.pem", &r), 1);
        struct Vec<struct Vec<unsigned char>> root = Vec::new();
        ASSERT_EQ(parse_fx("root.pem", &root), 1);
        &Vec<unsigned char> rl = r.at(0UL);
        &Vec<unsigned char> rr = root.at(0UL);
        struct X509Parsed rp;
        struct X509Parsed rootp;
        []const unsigned char rld = ((const unsigned char*)rl.data)[0..rl.len];
        []const unsigned char rrd = ((const unsigned char*)rr.data)[0..rr.len];
        ASSERT_EQ(x509_parse(rld[0..rl.len], &rp), 0);
        ASSERT_EQ(x509_parse(rrd[0..rr.len], &rootp), 0);
        ASSERT_EQ(rp.sig_type, X509_SIG_RSA_PSS);
        ASSERT_EQ(rp.sig_hash, DIGEST_SHA256);
        ASSERT_EQ(rp.pss_salt, 32);
        ASSERT_EQ(rp.key_type, X509_KEY_RSA);
        ASSERT_EQ(x509_verify_signed_by(rld[..], &rp, rrd[..], &rootp), 1);
        ASSERT_EQ(x509_verify_signed_by(id[..], &ica, rrd[..], &rootp), 1);
        // corrupt the signature
        unsigned char* p = (unsigned char*)l.data;
        p[leaf.sig.off + 10UL] = p[leaf.sig.off + 10UL] ^ (unsigned char)1;
        ASSERT_EQ(x509_verify_signed_by(ld[..], &leaf, id[..], &ica), 0);
        // truncated DER doesn't parse
        ASSERT_TRUE(x509_parse(ld[0..l.len - 5UL], &leaf) != 0);
        c.drop();
        r.drop();
        root.drop();
    }
}

static int chain_code(const &char chainFile, const &char rootFile, const ?&char host, int server) {
    struct Vec<struct Vec<unsigned char>> chain = Vec::new();
    parse_fx(chainFile, &chain);
    struct TlsTrust_ t = tls_trust_new_();
    struct String pem = read_fx(rootFile);
    tls_trust_add_pem_(&t, pem.as_ptr());
    pem.free();
    struct TlsError e = TlsError::none();
    int code = TLS_OK;
    unsafe { code = tls_verify_chain_(&chain, (?&TlsTrust_)&t, null, host, server, tls_now_(), &e); }
    if (code != TLS_OK) { printf("      %s + %s: %s\n", chainFile, rootFile, e.message); }
    t.drop();
    chain.drop();
    return code;
}

void t_chain() {
    ASSERT_EQ(chain_code("chain-ec.pem", "root.pem", "localhost", 1), TLS_OK);
    ASSERT_EQ(chain_code("chain-ec.pem", "root.pem", "127.0.0.1", 1), TLS_OK);
    ASSERT_EQ(chain_code("chain-ec.pem", "root.pem", null, 1), TLS_OK);
    ASSERT_EQ(chain_code("leaf-rsa.pem", "root.pem", "localhost", 1), TLS_OK);
    ASSERT_EQ(chain_code("chain-ec.pem", "root.pem", "evil.example", 1), TLS_ERR_CERT_HOSTNAME);
    ASSERT_EQ(chain_code("chain-ec.pem", "other-root.pem", "localhost", 1), TLS_ERR_CERT_UNTRUSTED);
    ASSERT_EQ(chain_code("leaf-ec.pem", "root.pem", "localhost", 1), TLS_ERR_CERT_UNTRUSTED);   // no intermediate
    ASSERT_EQ(chain_code("chain-ec.pem", "int.pem", "localhost", 1), TLS_OK);                    // intermediate as anchor
    ASSERT_EQ(chain_code("chain-expired.pem", "root.pem", "localhost", 1), TLS_ERR_CERT_EXPIRED);
    ASSERT_EQ(chain_code("self.pem", "root.pem", "localhost", 1), TLS_ERR_CERT_UNTRUSTED);
    ASSERT_EQ(chain_code("self.pem", "self.pem", "localhost", 1), TLS_OK);                       // pinned
    ASSERT_EQ(chain_code("other-leaf.pem", "other-root.pem", "localhost", 1), TLS_OK);
    // EKU: a client certificate is no server certificate, and vice versa
    ASSERT_EQ(chain_code("chain-client.pem", "root.pem", null, 0), TLS_OK);
    ASSERT_EQ(chain_code("chain-client.pem", "root.pem", null, 1), TLS_ERR_CERT_INVALID);
    ASSERT_EQ(chain_code("chain-ec.pem", "root.pem", null, 0), TLS_ERR_CERT_INVALID);
    // the leaf can't vouch for itself as a CA
    ASSERT_EQ(chain_code("chain-ec.pem", "leaf-ec.pem", "localhost", 1), TLS_OK);                // pinned leaf
    struct Vec<struct Vec<unsigned char>> empty = Vec::new();
    struct TlsError e = TlsError::none();
    ASSERT_EQ(tls_verify_chain_(&empty, null, null, null, 1, tls_now_(), &e), TLS_ERR_NO_CERT);
    empty.drop();
}

void t_config() {
    unsafe {
        struct TlsConfig c = TlsConfig::new();
        ASSERT_EQ(c.verify_peer, 1);
        ASSERT_EQ(c.verify_hostname, 1);
        ASSERT_EQ(c.use_system_roots, 1);
        ASSERT_EQ(c.min_version, TLS_VERSION_1_2);
        ASSERT_EQ(c.max_version, TLS_VERSION_1_3);
        c.set_alpn("h2, http/1.1");
        struct TlsConfig d = c.clone();
        c.insecure_skip_verify();
        ASSERT_EQ(c.verify_peer, 0);
        ASSERT_EQ(d.verify_peer, 1);
        struct Vec<unsigned char> w = tls_alpn_wire_(d.alpn.as_ptr());
        ASSERT_EQ((long long)w.len, 12LL);
        ASSERT_EQ(*w.at(0UL), 2);
        ASSERT_EQ(*w.at(3UL), 8);
        w.free();
        c.free();
        d.free();
        ASSERT_STR_EQ(tls_impl_name(TLS_IMPL_SAFEC), "safec");
        ASSERT_STR_EQ(tls_error_name(TLS_ERR_CERT_HOSTNAME), "certificate name mismatch");
        ASSERT_EQ(tls_backend_available(TLS_IMPL_SAFEC), 1);
        ASSERT_EQ(tls_backend_available(TLS_IMPL_SCHANNEL), 0);
        ASSERT_TRUE(tls_name_is_ip_("10.0.0.1"));
        ASSERT_TRUE(tls_name_is_ip_("fe80::1"));
        ASSERT_FALSE(tls_name_is_ip_("example.com"));
        // SAFEC_TLS picks the backend for TLS_BACKEND_AUTO
        int os = tls_backend_available(TLS_IMPL_OPENSSL);
        printf("    libssl %s; default backend: %s\n", os ? "loads" : "not found", tls_impl_name(tls_default_impl()));
        setenv("SAFEC_TLS", "safec", 1);
        ASSERT_EQ(tls_default_impl(), TLS_IMPL_SAFEC);
#if !defined(__APPLE__)
        setenv("SAFEC_TLS", "os", 1);
        ASSERT_EQ(tls_default_impl(), TLS_IMPL_OPENSSL);
        unsetenv("SAFEC_TLS");
        ASSERT_EQ(tls_default_impl(), os ? TLS_IMPL_OPENSSL : TLS_IMPL_SAFEC);
#else
        unsetenv("SAFEC_TLS");
        ASSERT_EQ(tls_default_impl(), TLS_IMPL_SECURE_TRANSPORT);
#endif
        // Server config without a certificate
        struct TlsServerConfig sc = TlsServerConfig::new();
        struct Result<TlsAcceptor, TlsError> ar = TlsAcceptor::new(&sc);
        ASSERT_FALSE(ar.ok());
        ASSERT_EQ(ar.error.code, TLS_ERR_CONFIG);
        sc.key_pem.push("-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----\n");
        struct String cert = read_fx("leaf-ec.pem");
        sc.cert_pem.push_str(&cert);
        cert.free();
        sc.backend = TLS_BACKEND_SAFEC;
        ar = TlsAcceptor::new(&sc);
        ASSERT_FALSE(ar.ok());
        ASSERT_EQ(ar.error.code, TLS_ERR_CONFIG);
        sc.free();
    }
}

// ── a TLS echo-ish server on a thread ─────────────────────────────────────────

static struct TlsAcceptor g_acc;
static int g_srv_port = 0;
static int g_srv_conns = 0;
static int g_srv_ready = 0;
static int g_srv_failures = 0;

static ?&int run_tls_server(?&int arg) {
    unsafe {
        int lfd = tcp_listen_nb((unsigned short)0);
        if (lfd < 0) { g_srv_ready = -1; return null; }
        g_srv_port = bound_port(lfd);
        fd_set_blocking(lfd);
        g_srv_ready = 1;
        int n = 0;
        while (n < g_srv_conns) {
            int fd = tcp_accept_nb(lfd);
            if (fd < 0) { thread_sleep_ms(2UL); continue; }
            n = n + 1;
            fd_set_blocking(fd);
            sock_set_io_timeout(fd, 10000);
            struct Result<TlsStream, TlsError> r = g_acc.accept(fd);
            if (!r.ok()) {
                printf("      server: %s\n", r.error.message);
                g_srv_failures = g_srv_failures + 1;
                r.drop();
                close(fd);
                continue;
            }
            struct TlsStream t = r.unwrap();
            unsigned char buf[2048];
            long long k = t.read(buf[0..2000UL]);
            char out[600];
            struct X509Cert pc;
            int hasPeer = t.peer_certificate(&pc);
            sprintf(out, "HTTP/1.0 200 OK\r\n\r\nhello via %s %s alpn=%s peer=%s req=%lld\n", t.backend_name(), t.cipher(),
                    t.alpn(), hasPeer ? (const char*)&pc.subject.cn[0] : (const char*)"-", k);
            t.write(out[0..str_len((const char*)&out[0])]);
            t.close();
        }
        close(lfd);
    }
    return null;
}

// Starts a server for 'conns' connections with cert/key files and backend.
static int start_server(const &char cert, const &char key, int backend, int verify_client, int conns) {
    struct TlsServerConfig sc = TlsServerConfig::new();
    struct String cp = String::from(FX); cp.push(cert);
    struct String kp = String::from(FX); kp.push(key);
    sc.load_files(cp.as_ptr(), kp.as_ptr());
    cp.free(); kp.free();
    sc.backend = backend;
    sc.set_alpn("h2,http/1.1");
    sc.verify_client = verify_client;
    if (verify_client) {
        struct String r = read_fx("root.pem");
        sc.client_ca_pem.push_str(&r);
        r.free();
    }
    struct Result<TlsAcceptor, TlsError> ar = TlsAcceptor::new(&sc);
    sc.free();
    if (!ar.ok()) { printf("      acceptor: %s\n", ar.error.message); ar.drop(); return 0; }
    g_acc = ar.unwrap();
    g_srv_port = 0;
    g_srv_conns = conns;
    g_srv_ready = 0;
    g_srv_failures = 0;
    unsafe {
        unsigned long long tid = 0ULL;
        thread_create(&tid, run_tls_server, null);
        thread_detach(tid);
        while (g_srv_ready == 0) { thread_sleep_ms(2UL); }
    }
    return g_srv_ready == 1;
}

static void stop_server() {
    // the server thread exits after its connections; give it a moment
    thread_sleep_ms(50UL);
    g_acc.free();
}

// A client request against port: the code of the failure (0 on success);
// alpn_want, if non-null, must be what was negotiated.
static int client_req(int port, int backend, const &char host, const &char root, int client_cert,
                      const ?&char alpn, const ?&char alpn_want, ?&String reply) {
    struct TlsConfig cfg = TlsConfig::new();
    cfg.backend = backend;
    cfg.use_system_roots = 0;
    struct String ca = String::from(FX); ca.push(root);
    cfg.set_ca_file(ca.as_ptr());
    ca.free();
    match (alpn) { case some(a): { cfg.set_alpn(a); } case null: { } }
    if (client_cert) {
        struct String c = read_fx("chain-client.pem");
        struct String k = read_fx("client.key");
        cfg.set_client_cert(c.as_ptr(), k.as_ptr());
        c.free(); k.free();
    }
    int fd = connect_port(port);
    if (fd < 0) { printf("      client: cannot connect to port %d\n", port); cfg.free(); return TLS_ERR_IO; }
    struct Result<TlsStream, TlsError> r = TlsStream::connect(fd, host, &cfg);
    cfg.free();
    if (!r.ok()) {
        printf("      client(%d) %s: %s\n", backend, host, r.error.message);
        close(fd);
        int failed = r.error.code;
        r.drop();
        return failed;
    }
    struct TlsStream t = r.unwrap();
    int code = TLS_OK;
    match (alpn_want) {
        case some(w): { if (!str_eq(t.alpn(), w)) { printf("      alpn '%s' != '%s'\n", t.alpn(), w); code = TLS_ERR_INTERNAL; } }
        case null: { }
    }
    unsafe {
        const &char req = "GET / HTTP/1.0\r\n\r\n";
        t.write(str_bytes(req));
        unsigned char buf[16384];
        long long total = 0LL;
        while (total < 16000LL) {
            long long n = t.read(buf[(unsigned long)total..16000UL]);
            if (n <= 0LL) { if (n < 0LL && total == 0LL) { code = t.last_error().code; if (code == 0) code = TLS_ERR_IO; } break; }
            total = total + n;
        }
        buf[total] = (unsigned char)0;
        if (total == 0LL && code == TLS_OK) { code = TLS_ERR_CLOSED; }
        if (code == TLS_OK && str_ncmp((const char*)&buf[0], "HTTP/1.", 7UL) != 0) { code = TLS_ERR_PROTOCOL; }
        match (reply) {
            case some(rp): { rp.push((const char*)&buf[0]); }
            case null: { }
        }
    }
    t.close();
    return code;
}

void t_loopback_safec() {
    ASSERT_TRUE(start_server("chain-ec.pem", "leaf-ec.key", TLS_BACKEND_SAFEC, 0, 5));
    struct String reply = String::new();
    ASSERT_EQ(client_req(g_srv_port, TLS_BACKEND_SAFEC, "localhost", "root.pem", 0, "http/1.1", "http/1.1", &reply), TLS_OK);
    ASSERT_TRUE(reply.contains("hello via safec TLS_CHACHA20_POLY1305_SHA256 alpn=http/1.1"));
    ASSERT_EQ(client_req(g_srv_port, TLS_BACKEND_SAFEC, "localhost", "root.pem", 0, "spdy/3,h2", "h2", null), TLS_OK);
    ASSERT_EQ(client_req(g_srv_port, TLS_BACKEND_SAFEC, "::1", "root.pem", 0, null, "", null), TLS_OK);
    ASSERT_EQ(client_req(g_srv_port, TLS_BACKEND_SAFEC, "bad.example", "root.pem", 0, null, null, null), TLS_ERR_CERT_HOSTNAME);
    ASSERT_EQ(client_req(g_srv_port, TLS_BACKEND_SAFEC, "localhost", "other-root.pem", 0, null, null, null), TLS_ERR_CERT_UNTRUSTED);
    stop_server();
    reply.free();
    // RSA key (rsa_pss_rsae_sha256 CertificateVerify)
    ASSERT_TRUE(start_server("leaf-rsa.pem", "leaf-rsa.key", TLS_BACKEND_SAFEC, 0, 1));
    ASSERT_EQ(client_req(g_srv_port, TLS_BACKEND_SAFEC, "127.0.0.1", "root.pem", 0, null, "", null), TLS_OK);
    stop_server();
    // client certificates required
    ASSERT_TRUE(start_server("chain-ec.pem", "leaf-ec.key", TLS_BACKEND_SAFEC, 2, 2));
    struct String r2 = String::new();
    ASSERT_EQ(client_req(g_srv_port, TLS_BACKEND_SAFEC, "localhost", "root.pem", 1, null, null, &r2), TLS_OK);
    ASSERT_TRUE(r2.contains("peer=SafeC Test Client"));
    ASSERT_TRUE(client_req(g_srv_port, TLS_BACKEND_SAFEC, "localhost", "root.pem", 0, null, null, null) != TLS_OK);
    stop_server();
    r2.free();
    ASSERT_EQ(g_srv_failures, 1);
}

void t_nonblocking() {
    unsafe {
        ASSERT_TRUE(start_server("chain-ec.pem", "leaf-ec.key", TLS_BACKEND_SAFEC, 0, 1));
        struct TlsConfig cfg = TlsConfig::new();
        cfg.backend = TLS_BACKEND_SAFEC;
        cfg.use_system_roots = 0;
        cfg.set_ca_file(FX "root.pem");
        int fd = connect_port(g_srv_port);
        ASSERT_TRUE(fd >= 0);
        fd_set_nonblocking(fd);
        struct Result<TlsStream, TlsError> r = TlsStream::client(fd, "localhost", &cfg);
        ASSERT_TRUE(r.ok());
        struct TlsStream t = r.unwrap();
        int wants = 0;
        int h = TLS_WANT_READ;
        int spins = 0;
        while (spins < 5000) {
            h = t.handshake();
            if (h != TLS_WANT_READ && h != TLS_WANT_WRITE) break;
            wants = wants + 1;
            thread_sleep_ms(1UL);
            spins = spins + 1;
        }
        ASSERT_EQ(h, 0);
        ASSERT_TRUE(wants > 0);   // the socket did make us wait
        ASSERT_EQ(t.version(), TLS_VERSION_1_3);
        const &char req = "GET / HTTP/1.0\r\n\r\n";
        long long w = TLS_WANT_WRITE;
        while (w == (long long)TLS_WANT_WRITE || w == (long long)TLS_WANT_READ) { w = t.write(str_bytes(req)); }
        ASSERT_EQ(w, (long long)str_len(req));
        unsigned char buf[2048];
        long long n = TLS_WANT_READ;
        spins = 0;
        while ((n == (long long)TLS_WANT_READ || n == (long long)TLS_WANT_WRITE) && spins < 5000) {
            n = t.read(buf[0..2000UL]);
            thread_sleep_ms(1UL);
            spins = spins + 1;
        }
        ASSERT_TRUE(n > 0LL);
        t.close();
        cfg.free();
        stop_server();
    }
}

void t_loopback_openssl() {
    if (!tls_backend_available(TLS_IMPL_OPENSSL)) { printf("    SKIP: libssl not found\n"); return; }
    // OpenSSL server, SafeC + OpenSSL clients
    ASSERT_TRUE(start_server("chain-ec.pem", "leaf-ec.key", OSSL, 2, 4));
    struct String reply = String::new();
    ASSERT_EQ(client_req(g_srv_port, TLS_BACKEND_SAFEC, "localhost", "root.pem", 1, "http/1.1", "http/1.1", &reply), TLS_OK);
    ASSERT_TRUE(reply.contains("hello via openssl"));
    ASSERT_TRUE(reply.contains("peer=SafeC Test Client"));
    ASSERT_EQ(client_req(g_srv_port, OSSL, "localhost", "root.pem", 1, "h2", "h2", null), TLS_OK);
    ASSERT_TRUE(client_req(g_srv_port, TLS_BACKEND_SAFEC, "localhost", "root.pem", 0, null, null, null) != TLS_OK);
    ASSERT_TRUE(client_req(g_srv_port, OSSL, "localhost", "root.pem", 0, null, null, null) != TLS_OK);
    stop_server();
    reply.free();
    // SafeC server, OpenSSL client: verification outcomes
    ASSERT_TRUE(start_server("chain-ec.pem", "leaf-ec.key", TLS_BACKEND_SAFEC, 0, 4));
    ASSERT_EQ(client_req(g_srv_port, OSSL, "localhost", "root.pem", 0, "http/1.1", "http/1.1", null), TLS_OK);
    ASSERT_EQ(client_req(g_srv_port, OSSL, "127.0.0.1", "root.pem", 0, null, null, null), TLS_OK);
    ASSERT_EQ(client_req(g_srv_port, OSSL, "bad.example", "root.pem", 0, null, null, null), TLS_ERR_CERT_HOSTNAME);
    ASSERT_EQ(client_req(g_srv_port, OSSL, "localhost", "other-root.pem", 0, null, null, null), TLS_ERR_CERT_UNTRUSTED);
    stop_server();
}

// ── the openssl command ───────────────────────────────────────────────────────

static int g_ss_port = 0;

static void s_server_start(const &char args) {
    g_ss_port = next_port();
    unsafe {
        char cmd[1024];
        sprintf(cmd, "cd " FX " && (openssl s_server -accept %d -www -quiet %s >/dev/null 2>&1 & echo $! > /tmp/safec_tls_sserver_%d.pid)",
                g_ss_port, args, g_ss_port);
        sys_system(cmd);
        // wait until it listens
        int i = 0;
        while (i < 200) {
            int fd = tcp_connect_timeout(sched_ipv4((unsigned char)127, (unsigned char)0, (unsigned char)0, (unsigned char)1),
                                         (unsigned short)g_ss_port, 200);
            if (fd >= 0) { close(fd); break; }
            thread_sleep_ms(10UL);
            i = i + 1;
        }
    }
}

static void s_server_stop() {
    unsafe {
        char cmd[256];
        sprintf(cmd, "kill $(cat /tmp/safec_tls_sserver_%d.pid) 2>/dev/null; rm -f /tmp/safec_tls_sserver_%d.pid", g_ss_port, g_ss_port);
        sys_system(cmd);
    }
}

static int both_(const &char host, const &char root, int want) {
    int a = client_req(g_ss_port, TLS_BACKEND_SAFEC, host, root, 0, null, null, null);
    int ok = a == want;
    if (tls_backend_available(TLS_IMPL_OPENSSL)) {
        int b = client_req(g_ss_port, OSSL, host, root, 0, null, null, null);
        ok = ok && b == want;
    }
    return ok;
}

void t_openssl_s_server() {
    if (!have_openssl_cli()) { printf("    SKIP: no openssl command\n"); return; }
    s_server_start("-cert leaf-ec.pem -key leaf-ec.key -cert_chain int.pem -alpn h2,http/1.1");
    ASSERT_EQ(client_req(g_ss_port, TLS_BACKEND_SAFEC, "localhost", "root.pem", 0, "http/1.1", "http/1.1", null), TLS_OK);
    ASSERT_TRUE(both_("localhost", "root.pem", TLS_OK));
    ASSERT_TRUE(both_("127.0.0.1", "root.pem", TLS_OK));
    ASSERT_TRUE(both_("wrong.example", "root.pem", TLS_ERR_CERT_HOSTNAME));
    ASSERT_TRUE(both_("localhost", "other-root.pem", TLS_ERR_CERT_UNTRUSTED));
    s_server_stop();
    // each TLS 1.3 cipher suite
    const char* suites[3];
    suites[0] = "TLS_AES_128_GCM_SHA256"; suites[1] = "TLS_AES_256_GCM_SHA384"; suites[2] = "TLS_CHACHA20_POLY1305_SHA256";
    int i = 0;
    while (i < 3) {
        char a[300];
        unsafe { sprintf(a, "-cert leaf-ec.pem -key leaf-ec.key -cert_chain int.pem -ciphersuites %s", suites[i]); }
        s_server_start(&a[0]);
        ASSERT_EQ(client_req(g_ss_port, TLS_BACKEND_SAFEC, "localhost", "root.pem", 0, null, null, null), TLS_OK);
        s_server_stop();
        i = i + 1;
    }
    // groups other than x25519: HelloRetryRequest
    s_server_start("-cert leaf-ec.pem -key leaf-ec.key -cert_chain int.pem -groups P-256");
    ASSERT_EQ(client_req(g_ss_port, TLS_BACKEND_SAFEC, "localhost", "root.pem", 0, null, null, null), TLS_OK);
    s_server_stop();
    s_server_start("-cert leaf-ec.pem -key leaf-ec.key -cert_chain int.pem -groups P-384");
    ASSERT_EQ(client_req(g_ss_port, TLS_BACKEND_SAFEC, "localhost", "root.pem", 0, null, null, null), TLS_OK);
    s_server_stop();
    // RSA leaf with a PSS signature from the root
    s_server_start("-cert leaf-rsa.pem -key leaf-rsa.key");
    ASSERT_TRUE(both_("localhost", "root.pem", TLS_OK));
    s_server_stop();
    // expired, self-signed, missing intermediate
    s_server_start("-cert expired.pem -key expired.key -cert_chain int.pem");
    ASSERT_TRUE(both_("localhost", "root.pem", TLS_ERR_CERT_EXPIRED));
    s_server_stop();
    s_server_start("-cert self.pem -key self.key");
    ASSERT_TRUE(both_("localhost", "root.pem", TLS_ERR_CERT_UNTRUSTED));
    ASSERT_TRUE(both_("localhost", "self.pem", TLS_OK));
    s_server_stop();
    s_server_start("-cert leaf-ec.pem -key leaf-ec.key");
    ASSERT_TRUE(both_("localhost", "root.pem", TLS_ERR_CERT_UNTRUSTED));
    s_server_stop();
    // TLS 1.2 only: the SafeC backend (1.3 only) fails cleanly, OpenSSL connects
    s_server_start("-cert leaf-ec.pem -key leaf-ec.key -cert_chain int.pem -tls1_2");
    ASSERT_EQ(client_req(g_ss_port, TLS_BACKEND_SAFEC, "localhost", "root.pem", 0, null, null, null), TLS_ERR_HANDSHAKE);
    if (tls_backend_available(TLS_IMPL_OPENSSL)) {
        ASSERT_EQ(client_req(g_ss_port, OSSL, "localhost", "root.pem", 0, null, null, null), TLS_OK);
    }
    s_server_stop();
    // a client certificate the server requires
    s_server_start("-cert leaf-ec.pem -key leaf-ec.key -cert_chain int.pem -Verify 3 -verify_return_error -CAfile root.pem");
    ASSERT_EQ(client_req(g_ss_port, TLS_BACKEND_SAFEC, "localhost", "root.pem", 1, null, null, null), TLS_OK);
    ASSERT_TRUE(client_req(g_ss_port, TLS_BACKEND_SAFEC, "localhost", "root.pem", 0, null, null, null) != TLS_OK);
    s_server_stop();
}

static int s_client(const &char extra, const &char expect) {
    unsafe {
        char cmd[1200];
        sprintf(cmd, "cd " FX " && printf 'GET / HTTP/1.0\\r\\n\\r\\n' | timeout 10 openssl s_client -connect 127.0.0.1:%d "
                     "-servername localhost -CAfile root.pem -verify_return_error -verify_hostname localhost -quiet %s "
                     "2>/dev/null | grep -q '%s'", g_srv_port, extra, expect);
        return sys_system(cmd) == 0;
    }
}

void t_openssl_s_client() {
    if (!have_openssl_cli()) { printf("    SKIP: no openssl command\n"); return; }
    ASSERT_TRUE(start_server("chain-ec.pem", "leaf-ec.key", TLS_BACKEND_SAFEC, 0, 6));
    ASSERT_TRUE(s_client("-tls1_3", "hello via safec"));
    ASSERT_TRUE(s_client("-tls1_3 -ciphersuites TLS_AES_128_GCM_SHA256", "TLS_AES_128_GCM_SHA256"));
    ASSERT_TRUE(s_client("-tls1_3 -ciphersuites TLS_AES_256_GCM_SHA384 -groups P-384", "TLS_AES_256_GCM_SHA384"));
    ASSERT_TRUE(s_client("-tls1_3 -groups P-256 -alpn http/1.1", "alpn=http/1.1"));
    ASSERT_TRUE(s_client("-tls1_3 -sigalgs ecdsa_secp256r1_sha256", "hello"));
    ASSERT_FALSE(s_client("-tls1_2", "hello"));   // 1.3 only
    stop_server();
    ASSERT_TRUE(start_server("leaf-rsa.pem", "leaf-rsa.key", TLS_BACKEND_SAFEC, 0, 1));
    ASSERT_TRUE(s_client("-tls1_3 -sigalgs rsa_pss_rsae_sha256", "hello via safec"));
    stop_server();
    ASSERT_TRUE(start_server("chain-ec.pem", "leaf-ec.key", TLS_BACKEND_SAFEC, 2, 2));
    ASSERT_TRUE(s_client("-tls1_3 -cert client.pem -key client.key -cert_chain int.pem", "peer=SafeC Test Client"));
    ASSERT_FALSE(s_client("-tls1_3", "hello"));
    stop_server();
}

// ── HTTPS ─────────────────────────────────────────────────────────────────────

static struct HttpResponse https_handler(&HttpRequest req) {
    if (req.path.eq_cstr("/hello")) { return HttpResponse::text(200, "hello over tls"); }
    if (req.path.eq_cstr("/redirect")) { return HttpResponse::redirect(302, "/hello"); }
    if (req.path.eq_cstr("/echo")) { return HttpResponse::text(200, req.body.as_ptr()); }
    return HttpResponse::error(404);
}

static struct HttpResponse route_handler(&HttpRequest req, &HttpCtx ctx) {
    if (ws_is_upgrade(req)) {
        int ok = 0;
        struct WsConn c = WsConn::accept_upgrade(req, ctx.conn, &ok);
        if (ok) {
            struct WsMessage m;
            if (c.recv_message(&m, 0UL)) {
                match (m.data) {
                    case some(d): { unsafe { c.send_binary(((const unsigned char*)d)[0..m.len]); } }
                    case null: { }
                }
                m.free();
            } else {
                unsafe { forget(m); }   // nothing was stored in m
            }
            c.close();
        } else {
            c.drop();
        }
        return HttpResponse::new(200);
    }
    if (req.path.eq_cstr("/events")) {
        struct SseStream s = SseStream::start(ctx.conn, null);
        s.send_event("tick", "one", "1");
        s.send("two");
        s.close();
        return HttpResponse::new(200);
    }
    return HttpResponse::text(200, "router over tls");
}

void t_https() {
    unsafe {
        struct TlsServerConfig sc = TlsServerConfig::new();
        sc.load_files(FX "chain-ec.pem", FX "leaf-ec.key");
        struct HttpServerConfig cfg = HttpServerConfig::defaults((unsigned short)0);
        cfg.threads = 2;
        cfg.tls = &sc;
        int ok = 0;
        struct HttpServer srv = HttpServer::bind(&cfg, &ok);
        ASSERT_TRUE(ok);
        srv.start(https_handler);
        char url[128];
        int b = 0;
        while (b < 2) {
            if (b == 1 && !tls_backend_available(TLS_IMPL_OPENSSL)) { break; }
            struct TlsConfig tc = TlsConfig::new();
            tc.backend = b == 0 ? TLS_BACKEND_SAFEC : OSSL;
            tc.use_system_roots = 0;
            tc.set_ca_file(FX "root.pem");
            struct HttpClientOptions o = HttpClientOptions::defaults();
            o.tls = &tc;
            struct TlsError te = TlsError::none();
            o.tls_error = &te;
            sprintf(url, "https://localhost:%d/redirect", (int)srv.port);
            int fok = 0;
            struct String fin = String::new();
            struct HttpResponse r = http_fetch("GET", url, null, str_bytes(""), &o, &fin, &fok);
            ASSERT_TRUE(fok);
            ASSERT_EQ(r.status, 200);
            ASSERT_STR_EQ(r.body.as_ptr(), "hello over tls");
            ASSERT_TRUE(fin.ends_with("/hello"));
            r.free();
            fin.free();
            sprintf(url, "https://127.0.0.1:%d/echo", (int)srv.port);
            r = http_fetch("POST", url, "Content-Type: text/plain\r\n", str_bytes("posted body"), &o, null, &fok);
            ASSERT_TRUE(fok);
            ASSERT_STR_EQ(r.body.as_ptr(), "posted body");
            r.free();
            // a name the certificate doesn't have
            sprintf(url, "https://127.0.0.2:%d/hello", (int)srv.port);
            r = http_fetch("GET", url, null, str_bytes(""), &o, null, &fok);
            ASSERT_FALSE(fok);
            ASSERT_EQ(te.code, TLS_ERR_CERT_HOSTNAME);
            r.free();
            tc.free();
            b = b + 1;
        }
        // the system roots don't know the test root
        sprintf(url, "https://localhost:%d/hello", (int)srv.port);
        int fok = 1;
        struct HttpResponse r = http_get_url(url, &fok);
        ASSERT_FALSE(fok);
        r.free();
        // plain http to the https port fails, it doesn't hang
        sprintf(url, "http://localhost:%d/hello", (int)srv.port);
        struct HttpClientOptions o2 = HttpClientOptions::defaults();
        o2.io_timeout_ms = 3000;
        r = http_fetch("GET", url, null, str_bytes(""), &o2, null, &fok);
        ASSERT_FALSE(fok && r.status == 200);
        r.free();
        srv.stop();
        sc.free();
    }
}

void t_wss_sse() {
    unsafe {
        struct TlsServerConfig sc = TlsServerConfig::new();
        sc.load_files(FX "leaf-rsa.pem", FX "leaf-rsa.key");
        sc.backend = TLS_BACKEND_SAFEC;
        struct HttpServerConfig cfg = HttpServerConfig::defaults((unsigned short)0);
        cfg.threads = 2;
        cfg.tls = &sc;
        struct Router rt = Router::new();
        rt.get("/*", route_handler);
        int ok = 0;
        struct HttpServer srv = HttpServer::bind(&cfg, &ok);
        ASSERT_TRUE(ok);
        rt.start(&srv);
        struct TlsConfig tc = TlsConfig::new();
        tc.use_system_roots = 0;
        tc.set_ca_file(FX "root.pem");
        char url[128];
        sprintf(url, "wss://localhost:%d/ws", (int)srv.port);
        int wok = 0;
        struct WsConn c = WsConn::connect_url_tls(url, &tc, 5000, &wok);
        ASSERT_TRUE(wok);
        ASSERT_FALSE(c.tls.is_null());
        if (wok) {
            ASSERT_TRUE(c.send_text("ping over wss"));
            struct WsMessage m;
            ASSERT_TRUE(c.recv_message(&m, 0UL));
            ASSERT_EQ((long long)m.len, 13LL);
            m.free();
            c.close();
        }
        // wss:// with the default (system) roots: refused
        struct WsConn c2 = WsConn::connect_url(url, 5000, &wok);
        ASSERT_FALSE(wok);
        sprintf(url, "https://localhost:%d/events", (int)srv.port);
        int sok = 0;
        struct SseClient s = SseClient::connect_tls(url, null, &tc, 5000, &sok);
        ASSERT_TRUE(sok);
        if (sok) {
            struct SseEvent e;
            ASSERT_TRUE(s.next(&e));
            ASSERT_STR_EQ(e.event.as_ptr(), "tick");
            ASSERT_STR_EQ(e.data.as_ptr(), "one");
            e.free();
            ASSERT_TRUE(s.next(&e));
            ASSERT_STR_EQ(e.data.as_ptr(), "two");
            e.free();
            s.close();
        }
        tc.free();
        srv.stop();
        rt.free();
        sc.free();
    }
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("RFC 8448 key schedule", t_rfc8448);
        t.add("x509 parse, signatures, host names", t_x509);
        t.add("certificate path validation", t_chain);
        t.add("config, backends, SAFEC_TLS", t_config);
        t.add("safec server <-> safec client", t_loopback_safec);
        t.add("non-blocking handshake and I/O", t_nonblocking);
        t.add("openssl backend <-> safec", t_loopback_openssl);
        t.add("interop: openssl s_server", t_openssl_s_server);
        t.add("interop: openssl s_client", t_openssl_s_client);
        t.add("https fetch (redirect, POST, name check)", t_https);
        t.add("wss:// and https SSE", t_wss_sse);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
