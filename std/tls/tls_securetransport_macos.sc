// SafeC Standard Library — TLS: the macOS backend (TLS_IMPL_SECURE_TRANSPORT),
// Security.framework's SecureTransport with SecTrust verification.
//
// Why SecureTransport and not Network.framework: TlsStream runs TLS over a
// socket the caller connected (an fd it can poll, a std/async IoDriver
// registration, an accepted server connection). Network.framework's
// nw_connection opens and owns its own socket and delivers data through
// dispatch-queue blocks, so it can't sit on an existing fd; SecureTransport
// is a plain C API that runs over caller-supplied read/write functions —
// the same shape as every other backend here. Its limits: deprecated since
// macOS 10.15 (still shipped and working) and TLS 1.2 at most. A config that
// asks for TLS 1.3 (min_version) gets TLS_ERR_UNAVAILABLE here, which
// TLS_BACKEND_AUTO answers with the SafeC backend (TLS 1.3).
//
// Certificates are checked by SecTrust (the system keychains' roots, SSL
// policy with the host name, OCSP/CRL as the system does them): the
// handshake breaks at server authentication (kSSLSessionOptionBreakOnServer
// Auth) and SecTrustEvaluateWithError decides; TlsConfig::ca_pem / ca_file
// become extra anchors (SecTrustSetAnchorCertificates; with
// use_system_roots off, the only ones).
//
// Not covered (TLS_ERR_UNAVAILABLE, so AUTO uses the SafeC backend):
// servers and client certificates — SecureTransport takes those as a
// SecIdentity from the keychain, not as PEM in memory.
// safeguard: link macos -framework Security -framework CoreFoundation
#pragma once
#include <std/stddef.h>
#include <std/tls/tls_backend.h>
#include <std/mem.h>
#include <std/str.h>

#define ST_NOERR_              0
#define ST_WOULD_BLOCK_        (-9803)
#define ST_CLOSED_GRACEFUL_    (-9805)
#define ST_CLOSED_ABORT_       (-9806)
#define ST_CLOSED_NO_NOTIFY_   (-9816)
#define ST_PEER_AUTH_DONE_     (-9841)
#define ST_IO_ERR_             (-36)
#define ST_CLIENT_SIDE_        1
#define ST_STREAM_TYPE_        0
#define ST_PROTO_TLS12_        8
#define ST_BREAK_ON_SERVER_AUTH_ 0
#define ST_UTF8_               0x08000100U

namespace std {

// CoreFoundation / Security (opaque refs are pointers).
struct StArrayCallBacks_ {
    long  version;
    unsigned char* retain;
    unsigned char* release;
    unsigned char* copyDescription;
    unsigned char* equal;
};
extern struct StArrayCallBacks_ kCFTypeArrayCallBacks;
extern unsigned char* CFDataCreate(unsigned char* alloc, const unsigned char* bytes, long len);
extern const unsigned char* CFDataGetBytePtr(unsigned char* data);
extern long CFDataGetLength(unsigned char* data);
extern unsigned char* CFArrayCreateMutable(unsigned char* alloc, long cap, const struct StArrayCallBacks_* cb);
extern void  CFArrayAppendValue(unsigned char* array, const unsigned char* value);
extern long  CFArrayGetCount(unsigned char* array);
extern const unsigned char* CFArrayGetValueAtIndex(unsigned char* array, long i);
extern unsigned char* CFStringCreateWithCString(unsigned char* alloc, const char* s, unsigned int encoding);
extern int   CFStringGetCString(unsigned char* s, char* buf, long size, unsigned int encoding);
extern long  CFErrorGetCode(unsigned char* err);
extern void  CFRelease(const unsigned char* ref);

extern unsigned char* SSLCreateContext(unsigned char* alloc, int side, int type);
extern int   SSLSetIOFuncs(unsigned char* ctx, unsigned char* readFn, unsigned char* writeFn);
extern int   SSLSetConnection(unsigned char* ctx, const unsigned char* conn);
extern int   SSLSetPeerDomainName(unsigned char* ctx, const char* name, size_t len);
extern int   SSLSetProtocolVersionMin(unsigned char* ctx, int proto);
extern int   SSLSetProtocolVersionMax(unsigned char* ctx, int proto);
extern int   SSLSetSessionOption(unsigned char* ctx, int option, unsigned char value);
extern int   SSLSetALPNProtocols(unsigned char* ctx, unsigned char* protocols);
extern int   SSLCopyALPNProtocols(unsigned char* ctx, unsigned char** protocols);
extern int   SSLHandshake(unsigned char* ctx);
extern int   SSLRead(unsigned char* ctx, unsigned char* data, size_t len, size_t* processed);
extern int   SSLWrite(unsigned char* ctx, const unsigned char* data, size_t len, size_t* processed);
extern int   SSLClose(unsigned char* ctx);
extern int   SSLGetNegotiatedProtocolVersion(unsigned char* ctx, int* proto);
extern int   SSLGetNegotiatedCipher(unsigned char* ctx, unsigned int* suite);
extern int   SSLCopyPeerTrust(unsigned char* ctx, unsigned char** trust);
extern unsigned char* SecPolicyCreateSSL(unsigned char server, unsigned char* hostname);
extern int   SecTrustSetPolicies(unsigned char* trust, const unsigned char* policies);
extern int   SecTrustSetAnchorCertificates(unsigned char* trust, unsigned char* anchors);
extern int   SecTrustSetAnchorCertificatesOnly(unsigned char* trust, unsigned char only);
extern unsigned char SecTrustEvaluateWithError(unsigned char* trust, unsigned char** error);
extern long  SecTrustGetCertificateCount(unsigned char* trust);
extern unsigned char* SecTrustGetCertificateAtIndex(unsigned char* trust, long i);
extern unsigned char* SecCertificateCreateWithData(unsigned char* alloc, unsigned char* data);
extern unsigned char* SecCertificateCopyData(unsigned char* cert);

struct StConn_ {
    unsigned char* ctx;           // SSLContextRef
    int   fd;
    int   want;          // the last I/O callback would block: TLS_WANT_READ / _WRITE, else 0
    int   eof;
    int   verify;
    int   verify_host;
    int   use_system;
    struct String name;
    unsigned char* anchors;       // CFMutableArray of SecCertificateRef, or null
    int   sent_close;
};

int tls_st_available_() { return 1; }

static struct StConn_* st_of_(&TlsStream s) {
    unsafe { return (struct StConn_*)s.impl_; }
}

// SSLReadFunc: fills *len bytes or reports how many before blocking.
static int st_read_cb_(const unsigned char* conn, unsigned char* data, size_t* len) {
    unsafe {
        struct StConn_* c = (struct StConn_*)conn;
        size_t want = *len;
        size_t got = (size_t)0;
        while (got < want) {
            long long n = tls_sock_recv_(c->fd, (&unsigned char)((unsigned char*)data + got), (unsigned long)(want - got));
            if (n == (long long)TLS_WANT_READ) { *len = got; c->want = TLS_WANT_READ; return ST_WOULD_BLOCK_; }
            if (n == 0LL) { *len = got; c->eof = 1; return ST_CLOSED_GRACEFUL_; }
            if (n < 0LL) { *len = got; return ST_IO_ERR_; }
            got = got + (size_t)n;
        }
        *len = got;
    }
    return ST_NOERR_;
}

static int st_write_cb_(const unsigned char* conn, const unsigned char* data, size_t* len) {
    unsafe {
        struct StConn_* c = (struct StConn_*)conn;
        size_t want = *len;
        size_t put = (size_t)0;
        while (put < want) {
            long long n = tls_sock_send_(c->fd, (const &unsigned char)((const unsigned char*)data + put), (unsigned long)(want - put));
            if (n == (long long)TLS_WANT_WRITE) { *len = put; c->want = TLS_WANT_WRITE; return ST_WOULD_BLOCK_; }
            if (n <= 0LL) { *len = put; return ST_IO_ERR_; }
            put = put + (size_t)n;
        }
        *len = put;
    }
    return ST_NOERR_;
}

static int st_fail_(&TlsStream s, int code, const &char what, long long status) {
    struct String m = String::from(what);
    m.push(" (OSStatus ");
    m.push_int(status);
    m.push(")");
    tls_fail_(&s.err, code, m.as_ptr());
    m.free();
    s.err.detail = status;
    return -1;
}

static unsigned char* st_cfstring_(const &char text) {
    unsafe { return CFStringCreateWithCString((unsigned char*)0, (const char*)text, ST_UTF8_); }
}

int tls_st_client_(&TlsStream s, const &char name, const &TlsConfig cfg) {
    if (cfg.min_version > TLS_VERSION_1_2) {
        tls_fail_(&s.err, TLS_ERR_UNAVAILABLE, "SecureTransport supports TLS 1.2 at most");
        return TLS_ERR_UNAVAILABLE;
    }
    if (!cfg.cert_pem.is_empty()) {
        tls_fail_(&s.err, TLS_ERR_UNAVAILABLE, "client certificates from PEM aren't supported by the SecureTransport backend");
        return TLS_ERR_UNAVAILABLE;
    }
    unsigned char* mem = alloc_zeroed_raw_(sizeof(struct StConn_));
    if (mem == (unsigned char*)0) { tls_fail_(&s.err, TLS_ERR_NO_MEMORY, "out of memory"); return TLS_ERR_NO_MEMORY; }
    unsafe { s.impl_ = (?&heap TlsImpl_)mem; }
    struct StConn_* c = st_of_(s);
    unsafe {
        c->fd = s.fd;
        c->name = String::from(name);
        c->verify = cfg.verify_peer;
        c->verify_host = cfg.verify_hostname;
        c->use_system = cfg.use_system_roots;
        // extra anchors
        struct String pem = String::new();
        pem.push(cfg.ca_pem.as_ptr());
        if (!cfg.ca_file.is_empty() && !tls_read_file_(cfg.ca_file.as_ptr(), &pem)) {
            pem.free();
            tls_fail2_(&s.err, TLS_ERR_CONFIG, "cannot read ca_file", cfg.ca_file.as_ptr());
            return TLS_ERR_CONFIG;
        }
        if (!pem.is_empty()) {
            struct Vec<struct Vec<unsigned char>> ders = Vec::new();
            tls_pem_decode_(pem.as_ptr(), "CERTIFICATE", &ders);
            c->anchors = CFArrayCreateMutable((unsigned char*)0, 0L, &kCFTypeArrayCallBacks);
            unsigned long i = 0UL;
            while (i < ders.len) {
                &Vec<unsigned char> d = ders.at(i);
                unsigned char* data = CFDataCreate((unsigned char*)0, (const unsigned char*)d.data, (long)d.len);
                unsigned char* cert = SecCertificateCreateWithData((unsigned char*)0, data);
                if (cert != (unsigned char*)0) { CFArrayAppendValue(c->anchors, (const unsigned char*)cert); CFRelease((const unsigned char*)cert); }
                CFRelease((const unsigned char*)data);
                i = i + 1UL;
            }
            tls_der_list_free_(&ders);
        }
        pem.free();
        unsigned char* ctx = SSLCreateContext((unsigned char*)0, ST_CLIENT_SIDE_, ST_STREAM_TYPE_);
        if (ctx == (unsigned char*)0) { tls_fail_(&s.err, TLS_ERR_INTERNAL, "SSLCreateContext failed"); return TLS_ERR_INTERNAL; }
        c->ctx = ctx;
        SSLSetIOFuncs(ctx, (unsigned char*)st_read_cb_, (unsigned char*)st_write_cb_);
        SSLSetConnection(ctx, (const unsigned char*)c);
        SSLSetProtocolVersionMin(ctx, ST_PROTO_TLS12_);
        SSLSetProtocolVersionMax(ctx, ST_PROTO_TLS12_);
        if (!tls_name_is_ip_(name)) { SSLSetPeerDomainName(ctx, (const char*)name, (size_t)str_len(name)); }
        // Verification is ours (SecTrust below), with the right policy and anchors.
        SSLSetSessionOption(ctx, ST_BREAK_ON_SERVER_AUTH_, (unsigned char)1);
        if (!cfg.alpn.is_empty()) {
            unsigned char* list = CFArrayCreateMutable((unsigned char*)0, 0L, &kCFTypeArrayCallBacks);
            struct String a = cfg.alpn.clone();
            struct String parts[16];
            unsigned long np = a.split(",", parts[0..16]);
            unsigned long k = 0UL;
            while (k < np) {
                struct String t = parts[k].trim();
                if (!t.is_empty()) {
                    unsigned char* str = st_cfstring_(t.as_ptr());
                    CFArrayAppendValue(list, (const unsigned char*)str);
                    CFRelease((const unsigned char*)str);
                }
                t.free();
                parts[k].free();
                k = k + 1UL;
            }
            a.free();
            SSLSetALPNProtocols(ctx, list);
            CFRelease((const unsigned char*)list);
        }
    }
    return TLS_OK;
}

// SecTrust evaluation of the server's chain.
static int st_check_trust_(&TlsStream s, struct StConn_* c) {
    unsigned char* trust = (unsigned char*)0;
    int code = TLS_OK;
    unsafe {
        int st = SSLCopyPeerTrust(c->ctx, &trust);
        if (st != ST_NOERR_ || trust == (unsigned char*)0) { return st_fail_(s, TLS_ERR_NO_CERT, "no server certificate", (long long)st); }
        // the leaf for TlsStream::peer_certificate
        if (SecTrustGetCertificateCount(trust) > 0L) {
            unsigned char* leaf = SecTrustGetCertificateAtIndex(trust, 0L);
            unsigned char* der = SecCertificateCopyData(leaf);
            if (der != (unsigned char*)0) {
                s.peer_der_.free();
                s.peer_der_ = Vec::from_slice(((const unsigned char*)CFDataGetBytePtr(der))[0..(unsigned long)CFDataGetLength(der)]);
                CFRelease((const unsigned char*)der);
            }
        }
        if (c->verify) {
            unsigned char* host = (unsigned char*)0;
            if (c->verify_host) { host = st_cfstring_(c->name.as_ptr()); }
            unsigned char* policy = SecPolicyCreateSSL((unsigned char)1, host);
            SecTrustSetPolicies(trust, (const unsigned char*)policy);
            CFRelease((const unsigned char*)policy);
            if (host != (unsigned char*)0) { CFRelease((const unsigned char*)host); }
            if (c->anchors != (unsigned char*)0) {
                SecTrustSetAnchorCertificates(trust, c->anchors);
                SecTrustSetAnchorCertificatesOnly(trust, (unsigned char)(c->use_system ? 0 : 1));
            } else if (!c->use_system) {
                unsigned char* none = CFArrayCreateMutable((unsigned char*)0, 0L, &kCFTypeArrayCallBacks);
                SecTrustSetAnchorCertificates(trust, none);
                SecTrustSetAnchorCertificatesOnly(trust, (unsigned char)1);
                CFRelease((const unsigned char*)none);
            }
            unsigned char* err = (unsigned char*)0;
            if (!SecTrustEvaluateWithError(trust, &err)) {
                long e = err != (unsigned char*)0 ? CFErrorGetCode(err) : 0L;
                const &char why = "certificate verification failed";
                code = TLS_ERR_CERT_INVALID;
                if (e == -67602L) { code = TLS_ERR_CERT_HOSTNAME; why = "the certificate doesn't match the host name"; }
                else if (e == -67818L || e == -67819L) { code = TLS_ERR_CERT_EXPIRED; why = "the certificate has expired or is not yet valid"; }
                else if (e == -67843L || e == -67808L || e == -67609L) { code = TLS_ERR_CERT_UNTRUSTED; why = "the certificate chain doesn't lead to a trusted root"; }
                else if (e == -67820L) { code = TLS_ERR_CERT_REVOKED; why = "the certificate has been revoked"; }
                st_fail_(s, code, why, (long long)e);
                if (err != (unsigned char*)0) { CFRelease((const unsigned char*)err); }
            }
        }
        CFRelease((const unsigned char*)trust);
    }
    return code == TLS_OK ? 0 : -1;
}

static void st_fill_info_(&TlsStream s, struct StConn_* c) {
    unsafe {
        int proto = 0;
        SSLGetNegotiatedProtocolVersion(c->ctx, &proto);
        if (proto == ST_PROTO_TLS12_) { s.version_ = TLS_VERSION_1_2; }
        else if (proto == 10) { s.version_ = TLS_VERSION_1_3; }
        unsigned int suite = 0U;
        SSLGetNegotiatedCipher(c->ctx, &suite);
        tls_copy_cstr_(&s.cipher_[0], 64UL, tls_suite_name_((int)suite));
        unsigned char* alpn = (unsigned char*)0;
        if (SSLCopyALPNProtocols(c->ctx, &alpn) == ST_NOERR_ && alpn != (unsigned char*)0) {
            if (CFArrayGetCount(alpn) > 0L) {
                CFStringGetCString((unsigned char*)CFArrayGetValueAtIndex(alpn, 0L), (char*)&s.alpn_[0], 64L, ST_UTF8_);
            }
            CFRelease((const unsigned char*)alpn);
        }
    }
}

int tls_st_handshake_(&TlsStream s) {
    struct StConn_* c = st_of_(s);
    unsafe {
        while (1) {
            c->want = 0;
            int st = SSLHandshake(c->ctx);
            if (st == ST_NOERR_) {
                st_fill_info_(s, c);
                return 0;
            }
            if (st == ST_PEER_AUTH_DONE_) {
                if (st_check_trust_(s, c) != 0) { return -1; }
                continue;
            }
            if (st == ST_WOULD_BLOCK_) { return c->want != 0 ? c->want : TLS_WANT_READ; }
            if (c->eof || st == ST_CLOSED_GRACEFUL_ || st == ST_CLOSED_ABORT_) {
                return tls_fail_(&s.err, TLS_ERR_CLOSED, "the peer closed the connection during the TLS handshake");
            }
            int code = TLS_ERR_HANDSHAKE;
            if (st == ST_IO_ERR_) { code = TLS_ERR_IO; }
            return st_fail_(s, code, "SecureTransport handshake failed", (long long)st);
        }
    }
    return -1;
}

long long tls_st_read_(&TlsStream s, &unsigned char buf, unsigned long n) {
    struct StConn_* c = st_of_(s);
    unsafe {
        c->want = 0;
        size_t got = (size_t)0;
        int st = SSLRead(c->ctx, (unsigned char*)buf, (size_t)n, &got);
        if (got > (size_t)0) { return (long long)got; }
        if (st == ST_WOULD_BLOCK_) { return (long long)(c->want != 0 ? c->want : TLS_WANT_READ); }
        if (st == ST_CLOSED_GRACEFUL_ || st == ST_CLOSED_NO_NOTIFY_ || st == ST_CLOSED_ABORT_ || c->eof) { return 0LL; }
        st_fail_(s, TLS_ERR_PROTOCOL, "SSLRead failed", (long long)st);
    }
    return -1LL;
}

long long tls_st_write_(&TlsStream s, const &unsigned char buf, unsigned long n) {
    struct StConn_* c = st_of_(s);
    unsafe {
        c->want = 0;
        size_t put = (size_t)0;
        int st = SSLWrite(c->ctx, (const unsigned char*)buf, (size_t)n, &put);
        if (put > (size_t)0) { return (long long)put; }
        if (st == ST_WOULD_BLOCK_) { return (long long)(c->want != 0 ? c->want : TLS_WANT_WRITE); }
        st_fail_(s, TLS_ERR_IO, "SSLWrite failed", (long long)st);
    }
    return -1LL;
}

int tls_st_shutdown_(&TlsStream s) {
    struct StConn_* c = st_of_(s);
    unsafe {
        if (c == (struct StConn_*)0 || c->sent_close) { return 0; }
        c->sent_close = 1;
        return SSLClose(c->ctx) == ST_NOERR_ ? 0 : -1;
    }
}

void tls_st_free_(&TlsStream s) {
    struct StConn_* c = st_of_(s);
    if (c == (struct StConn_*)0) { return; }
    unsafe {
        if (c->ctx != (unsigned char*)0) { CFRelease((const unsigned char*)c->ctx); }
        if (c->anchors != (unsigned char*)0) { CFRelease((const unsigned char*)c->anchors); }
        c->name.free();
        dealloc(c);
    }
    s.impl_ = null;
}

// Servers need a keychain SecIdentity; TLS_BACKEND_AUTO serves with the
// SafeC backend instead.
int tls_st_acceptor_(&TlsAcceptor a, const &TlsServerConfig cfg, &TlsError err) {
    tls_fail_(err, TLS_ERR_UNAVAILABLE, "the SecureTransport backend doesn't serve TLS (PEM keys); use the SafeC backend");
    return TLS_ERR_UNAVAILABLE;
}
int tls_st_accept_(&TlsStream s, const &TlsAcceptor a) {
    tls_fail_(&s.err, TLS_ERR_UNAVAILABLE, "the SecureTransport backend doesn't serve TLS");
    return TLS_ERR_UNAVAILABLE;
}
void tls_st_acceptor_free_(&TlsAcceptor a) { }

} // namespace std
