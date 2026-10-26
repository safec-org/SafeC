// SafeC Standard Library — TLS: the OpenSSL backend (TLS_IMPL_OPENSSL),
// libssl + libcrypto 3.x or 1.1.1 loaded with dlopen() at run time.
//
// Nothing is linked and no OpenSSL headers are needed: the functions below
// are looked up by name (their C signatures follow <openssl/ssl.h>), so a
// program runs on machines without OpenSSL and simply gets the SafeC
// backend there. OpenSSL does the protocol and certificate verification
// (SSL_VERIFY_PEER with SSL_set1_host / X509_VERIFY_PARAM_set1_ip_asc and the
// default CA paths — OPENSSLDIR's cert.pem/certs, SSL_CERT_FILE/SSL_CERT_DIR —
// plus the distribution bundles when those come up empty). The socket I/O
// stays here: OpenSSL reads and writes memory BIOs that this file moves to
// and from the socket with tls_sock_recv_/send_ (send never raises SIGPIPE,
// and a non-blocking socket gives TLS_WANT_READ/WRITE).
//
// Library names tried: libssl.so.3, libssl.so.1.1, libssl.so (and the
// FreeBSD/OpenBSD and Homebrew names), with the matching libcrypto.
// safeguard: link linux -ldl
#pragma once
#include <std/stddef.h>
#include <std/tls/tls_backend.h>
#include <std/sync/spinlock.h>
#include <std/mem.h>
#include <std/str.h>

#define OSSL_RTLD_NOW_     2
#define OSSL_VERIFY_NONE_  0
#define OSSL_VERIFY_PEER_  1
#define OSSL_VERIFY_FAIL_NO_CERT_ 2
#define OSSL_CTRL_SET_TLSEXT_HOSTNAME_ 55
#define OSSL_CTRL_EXTRA_CHAIN_CERT_    14
#define OSSL_CTRL_CHAIN_CERT_          89
#define OSSL_CTRL_MODE_                33
#define OSSL_CTRL_SET_MIN_PROTO_       123
#define OSSL_CTRL_SET_MAX_PROTO_       124
#define OSSL_C_SET_BUF_MEM_EOF_RETURN_ 130
#define OSSL_ERROR_SSL_         1
#define OSSL_ERROR_WANT_READ_   2
#define OSSL_ERROR_WANT_WRITE_  3
#define OSSL_ERROR_SYSCALL_     5
#define OSSL_ERROR_ZERO_RETURN_ 6
#define OSSL_X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS_ 4U

namespace std {

// What OpenSSL hands out, opaque: the library handles, SSL_METHOD,
// SSL_CTX, SSL, BIO_METHOD, BIO, X509, EVP_PKEY, X509_STORE,
// X509_VERIFY_PARAM, SSL_CIPHER and the (never passed) init settings.
struct OsslDl_ { char opaque_; };
struct OsslSslMethod_ { char opaque_; };
struct OsslCtx_ { char opaque_; };
struct OsslSsl_ { char opaque_; };
struct OsslBioMethod_ { char opaque_; };
struct OsslBio_ { char opaque_; };
struct OsslX509_ { char opaque_; };
struct OsslPkey_ { char opaque_; };
struct OsslStore_ { char opaque_; };
struct OsslStoreCtx_ { char opaque_; };
struct OsslParam_ { char opaque_; };
struct OsslCipher_ { char opaque_; };
struct OsslInitSettings_ { char opaque_; };

// The acceptor's state, which the ALPN selection callback gets as its arg.
struct OsslAcc_ {
    OsslCtx_* ctx;
    struct Vec<unsigned char> alpn;   // wire list for the selection callback
};

extern OsslDl_* dlopen(const char* name, int flags);
extern ?fn void() dlsym(OsslDl_* lib, const char* name);   // a function, cast to its type

typedef fn int(OsslSsl_*, unsigned char**, unsigned char*, const unsigned char*, unsigned int, OsslAcc_*) OsslAlpnCb_;

typedef fn int(unsigned long long, const OsslInitSettings_*) OsslInitFn_;
typedef fn const OsslSslMethod_*() OsslMethodFn_;
typedef fn OsslCtx_*(const OsslSslMethod_*) OsslCtxNew_;
typedef fn void(OsslCtx_*) OsslCtxFree_;
typedef fn int(OsslCtx_*) OsslCtxInt_;                            // set_default_verify_paths, check_private_key
typedef fn int(OsslCtx_*, const char*, const char*) OsslLoadLoc_;
typedef fn OsslStore_*(OsslCtx_*) OsslCtxStore_;
typedef fn void(OsslCtx_*, int, ?fn int(int, OsslStoreCtx_*)) OsslCtxSetVerify_;
typedef fn long(OsslCtx_*, int, long, OsslX509_*) OsslCtxCtrl_;    // the X509 argument, or null
typedef fn void(OsslCtx_*, OsslAlpnCb_, OsslAcc_*) OsslSetAlpnCb_;
typedef fn int(OsslCtx_*, OsslX509_*) OsslCtxUseCert_;
typedef fn int(OsslCtx_*, OsslPkey_*) OsslCtxUseKey_;
typedef fn int(unsigned char**, unsigned char*, const unsigned char*, unsigned int, const unsigned char*, unsigned int) OsslSelectProto_;
typedef fn OsslSsl_*(OsslCtx_*) OsslSslNew_;
typedef fn void(OsslSsl_*) OsslSslVoid_;                           // SSL_free, SSL_set_connect_state, SSL_set_accept_state
typedef fn int(OsslSsl_*) OsslSslInt_;                             // SSL_do_handshake, SSL_shutdown, SSL_version
typedef fn long(OsslSsl_*, int, long, OsslX509_*) OsslSslCtrl_;    // the X509 argument, or null
typedef fn long(OsslSsl_*, int, long, const char*) OsslSslCtrlName_;   // the host name argument
typedef fn int(OsslSsl_*, const unsigned char*, unsigned int) OsslSetAlpn_;
typedef fn void(OsslSsl_*, int, ?fn int(int, OsslStoreCtx_*)) OsslSslSetVerify_;
typedef fn int(OsslSsl_*, OsslX509_*) OsslSslUseCert_;
typedef fn int(OsslSsl_*, OsslPkey_*) OsslSslUseKey_;
typedef fn int(OsslSsl_*, const char*) OsslSet1Host_;
typedef fn OsslParam_*(OsslSsl_*) OsslGet0Param_;
typedef fn int(OsslSsl_*, unsigned char*, int) OsslSslRead_;
typedef fn int(OsslSsl_*, const unsigned char*, int) OsslSslWrite_;
typedef fn int(const OsslSsl_*, int) OsslGetError_;
typedef fn const OsslCipher_*(const OsslSsl_*) OsslGetCipher_;
typedef fn const char*(const OsslCipher_*) OsslCipherName_;
typedef fn void(const OsslSsl_*, const unsigned char**, unsigned int*) OsslAlpnSelected_;
typedef fn long(const OsslSsl_*) OsslVerifyResult_;
typedef fn OsslX509_*(const OsslSsl_*) OsslPeerCert_;
typedef fn void(OsslSsl_*, OsslBio_*, OsslBio_*) OsslSetBio_;
typedef fn unsigned long() OsslErrGet_;
typedef fn void(unsigned long, char*, size_t) OsslErrString_;
typedef fn void() OsslErrClear_;
typedef fn const char*(long) OsslVerifyString_;
typedef fn const OsslBioMethod_*() OsslBioMethodFn_;
typedef fn OsslBio_*(const OsslBioMethod_*) OsslBioNew_;
typedef fn int(OsslBio_*) OsslBioFree_;
typedef fn int(OsslBio_*, unsigned char*, int) OsslBioRead_;
typedef fn int(OsslBio_*, const unsigned char*, int) OsslBioWrite_;
typedef fn size_t(OsslBio_*) OsslPending_;
typedef fn long(OsslBio_*, int, long, unsigned char*) OsslBioCtrl_;   // parg: null here
typedef fn OsslBio_*(const char*, int) OsslBioMemBuf_;
typedef fn OsslX509_*(OsslBio_*, OsslX509_**, ?fn int(char*, int, int, ?&Erased), ?&Erased) OsslPemReadX509_;
typedef fn OsslPkey_*(OsslBio_*, OsslPkey_**, ?fn int(char*, int, int, ?&Erased), ?&Erased) OsslPemReadKey_;
typedef fn void(OsslX509_*) OsslX509Free_;
typedef fn void(OsslPkey_*) OsslPkeyFree_;
typedef fn int(OsslStore_*, OsslX509_*) OsslStoreAdd_;
typedef fn int(const OsslX509_*, unsigned char**) OsslI2d_;
typedef fn int(OsslParam_*, const char*) OsslParamSetIp_;
typedef fn void(OsslParam_*, unsigned int) OsslHostFlags_;

struct OsslLib_ {
    int loaded;     // 0 not tried, 1 ok, -1 unavailable
    int v3;
    int has_std_name;   // SSL_CIPHER_standard_name exists (1.1.1+)
    OsslInitFn_ init_ssl;
    OsslMethodFn_ client_method;
    OsslMethodFn_ server_method;
    OsslCtxNew_ ctx_new;
    OsslCtxFree_ ctx_free;
    OsslCtxInt_ ctx_default_paths;
    OsslLoadLoc_ ctx_load_locations;
    OsslCtxStore_ ctx_get_store;
    OsslCtxSetVerify_ ctx_set_verify;
    OsslCtxCtrl_ ctx_ctrl;
    OsslSetAlpnCb_ ctx_set_alpn_cb;
    OsslCtxUseCert_ ctx_use_cert;
    OsslCtxUseKey_ ctx_use_key;
    OsslCtxInt_ ctx_check_key;
    OsslSelectProto_ select_next_proto;
    OsslSslNew_ ssl_new;
    OsslSslVoid_ ssl_free;
    OsslSslCtrl_ ssl_ctrl;
    OsslSslCtrlName_ ssl_ctrl_name;
    OsslSetAlpn_ ssl_set_alpn;
    OsslSslSetVerify_ ssl_set_verify;
    OsslSslUseCert_ ssl_use_cert;
    OsslSslUseKey_ ssl_use_key;
    OsslSet1Host_ ssl_set1_host;
    OsslGet0Param_ ssl_get0_param;
    OsslSslVoid_ ssl_set_connect;
    OsslSslVoid_ ssl_set_accept;
    OsslSslInt_ ssl_do_handshake;
    OsslSslRead_ ssl_read;
    OsslSslWrite_ ssl_write;
    OsslGetError_ ssl_get_error;
    OsslSslInt_ ssl_shutdown;
    OsslSslInt_ ssl_version;
    OsslGetCipher_ ssl_get_cipher;
    OsslCipherName_ cipher_std_name;
    OsslCipherName_ cipher_name;
    OsslAlpnSelected_ ssl_alpn_selected;
    OsslVerifyResult_ ssl_verify_result;
    OsslPeerCert_ ssl_peer_cert;
    OsslSetBio_ ssl_set_bio;
    // libcrypto
    OsslErrGet_ err_get;
    OsslErrString_ err_string;
    OsslErrClear_ err_clear;
    OsslVerifyString_ verify_string;
    OsslBioMethodFn_ bio_s_mem;
    OsslBioNew_ bio_new;
    OsslBioFree_ bio_free;
    OsslBioRead_ bio_read;
    OsslBioWrite_ bio_write;
    OsslPending_ bio_pending;
    OsslBioCtrl_ bio_ctrl;
    OsslBioMemBuf_ bio_mem_buf;
    OsslPemReadX509_ pem_read_x509;
    OsslPemReadKey_ pem_read_key;
    OsslX509Free_ x509_free;
    OsslPkeyFree_ pkey_free;
    OsslStoreAdd_ store_add_cert;
    OsslI2d_ i2d_x509;
    OsslParamSetIp_ param_set_ip;
    OsslHostFlags_ param_set_hostflags;
};

static struct OsslLib_ g_ossl_;
static struct Spinlock g_ossl_lock_;
static OsslCtx_* g_ossl_client_ctx_ = (OsslCtx_*)0;   // the shared system-roots client context

static void ossl_sym_missing_() { }

// The function 'name' (cast by the caller to its type); a missing one
// clears ok and reads as a no-op.
static fn void() ossl_sym_(OsslDl_* lib, const &char name, &int ok) {
    ?fn void() p = null;
    unsafe { p = dlsym(lib, (const char*)name); }
    match (p) {
        case some(f): { return f; }
        case null: {}
    }
    *ok = 0;
    return ossl_sym_missing_;
}

static OsslDl_* ossl_open_(const &char name) {
    unsafe { return dlopen((const char*)name, OSSL_RTLD_NOW_); }
}

static int ossl_load_libs_(&OsslDl_* ssl, &OsslDl_* crypto) {
    unsafe {
        // pairs: libssl, libcrypto
        if ((*ssl = ossl_open_("libssl.so.3")) != (OsslDl_*)0) { *crypto = ossl_open_("libcrypto.so.3"); }
        else if ((*ssl = ossl_open_("libssl.so.1.1")) != (OsslDl_*)0) { *crypto = ossl_open_("libcrypto.so.1.1"); }
        else if ((*ssl = ossl_open_("libssl.so.30")) != (OsslDl_*)0) { *crypto = ossl_open_("libcrypto.so.30"); }
        else if ((*ssl = ossl_open_("libssl.so.111")) != (OsslDl_*)0) { *crypto = ossl_open_("libcrypto.so.111"); }
#if defined(__APPLE__)
        else if ((*ssl = ossl_open_("/opt/homebrew/opt/openssl@3/lib/libssl.3.dylib")) != (OsslDl_*)0) { *crypto = ossl_open_("/opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib"); }
        else if ((*ssl = ossl_open_("/usr/local/opt/openssl@3/lib/libssl.3.dylib")) != (OsslDl_*)0) { *crypto = ossl_open_("/usr/local/opt/openssl@3/lib/libcrypto.3.dylib"); }
        else if ((*ssl = ossl_open_("libssl.3.dylib")) != (OsslDl_*)0) { *crypto = ossl_open_("libcrypto.3.dylib"); }
#endif
        else if ((*ssl = ossl_open_("libssl.so")) != (OsslDl_*)0) { *crypto = ossl_open_("libcrypto.so"); }
        return *ssl != (OsslDl_*)0 && *crypto != (OsslDl_*)0;
    }
}

static int ossl_load_() {
    g_ossl_lock_.lock();
    if (g_ossl_.loaded != 0) {
        int r = g_ossl_.loaded > 0;
        g_ossl_lock_.unlock();
        return r;
    }
    g_ossl_.loaded = -1;
    OsslDl_* ls = (OsslDl_*)0;
    OsslDl_* lc = (OsslDl_*)0;
    if (!ossl_load_libs_(&ls, &lc)) { g_ossl_lock_.unlock(); return 0; }
    int ok = 1;
    unsafe {
        g_ossl_.init_ssl = (OsslInitFn_)ossl_sym_(ls, "OPENSSL_init_ssl", &ok);
        g_ossl_.client_method = (OsslMethodFn_)ossl_sym_(ls, "TLS_client_method", &ok);
        g_ossl_.server_method = (OsslMethodFn_)ossl_sym_(ls, "TLS_server_method", &ok);
        g_ossl_.ctx_new = (OsslCtxNew_)ossl_sym_(ls, "SSL_CTX_new", &ok);
        g_ossl_.ctx_free = (OsslCtxFree_)ossl_sym_(ls, "SSL_CTX_free", &ok);
        g_ossl_.ctx_default_paths = (OsslCtxInt_)ossl_sym_(ls, "SSL_CTX_set_default_verify_paths", &ok);
        g_ossl_.ctx_load_locations = (OsslLoadLoc_)ossl_sym_(ls, "SSL_CTX_load_verify_locations", &ok);
        g_ossl_.ctx_get_store = (OsslCtxStore_)ossl_sym_(ls, "SSL_CTX_get_cert_store", &ok);
        g_ossl_.ctx_set_verify = (OsslCtxSetVerify_)ossl_sym_(ls, "SSL_CTX_set_verify", &ok);
        g_ossl_.ctx_ctrl = (OsslCtxCtrl_)ossl_sym_(ls, "SSL_CTX_ctrl", &ok);
        g_ossl_.ctx_set_alpn_cb = (OsslSetAlpnCb_)ossl_sym_(ls, "SSL_CTX_set_alpn_select_cb", &ok);
        g_ossl_.ctx_use_cert = (OsslCtxUseCert_)ossl_sym_(ls, "SSL_CTX_use_certificate", &ok);
        g_ossl_.ctx_use_key = (OsslCtxUseKey_)ossl_sym_(ls, "SSL_CTX_use_PrivateKey", &ok);
        g_ossl_.ctx_check_key = (OsslCtxInt_)ossl_sym_(ls, "SSL_CTX_check_private_key", &ok);
        g_ossl_.select_next_proto = (OsslSelectProto_)ossl_sym_(ls, "SSL_select_next_proto", &ok);
        g_ossl_.ssl_new = (OsslSslNew_)ossl_sym_(ls, "SSL_new", &ok);
        g_ossl_.ssl_free = (OsslSslVoid_)ossl_sym_(ls, "SSL_free", &ok);
        g_ossl_.ssl_ctrl = (OsslSslCtrl_)ossl_sym_(ls, "SSL_ctrl", &ok);
        g_ossl_.ssl_ctrl_name = (OsslSslCtrlName_)ossl_sym_(ls, "SSL_ctrl", &ok);
        g_ossl_.ssl_set_alpn = (OsslSetAlpn_)ossl_sym_(ls, "SSL_set_alpn_protos", &ok);
        g_ossl_.ssl_set_verify = (OsslSslSetVerify_)ossl_sym_(ls, "SSL_set_verify", &ok);
        g_ossl_.ssl_use_cert = (OsslSslUseCert_)ossl_sym_(ls, "SSL_use_certificate", &ok);
        g_ossl_.ssl_use_key = (OsslSslUseKey_)ossl_sym_(ls, "SSL_use_PrivateKey", &ok);
        g_ossl_.ssl_set1_host = (OsslSet1Host_)ossl_sym_(ls, "SSL_set1_host", &ok);
        g_ossl_.ssl_get0_param = (OsslGet0Param_)ossl_sym_(ls, "SSL_get0_param", &ok);
        g_ossl_.ssl_set_connect = (OsslSslVoid_)ossl_sym_(ls, "SSL_set_connect_state", &ok);
        g_ossl_.ssl_set_accept = (OsslSslVoid_)ossl_sym_(ls, "SSL_set_accept_state", &ok);
        g_ossl_.ssl_do_handshake = (OsslSslInt_)ossl_sym_(ls, "SSL_do_handshake", &ok);
        g_ossl_.ssl_read = (OsslSslRead_)ossl_sym_(ls, "SSL_read", &ok);
        g_ossl_.ssl_write = (OsslSslWrite_)ossl_sym_(ls, "SSL_write", &ok);
        g_ossl_.ssl_get_error = (OsslGetError_)ossl_sym_(ls, "SSL_get_error", &ok);
        g_ossl_.ssl_shutdown = (OsslSslInt_)ossl_sym_(ls, "SSL_shutdown", &ok);
        g_ossl_.ssl_version = (OsslSslInt_)ossl_sym_(ls, "SSL_version", &ok);
        g_ossl_.ssl_get_cipher = (OsslGetCipher_)ossl_sym_(ls, "SSL_get_current_cipher", &ok);
        g_ossl_.cipher_name = (OsslCipherName_)ossl_sym_(ls, "SSL_CIPHER_get_name", &ok);
        int stdNameOk = 1;
        g_ossl_.cipher_std_name = (OsslCipherName_)ossl_sym_(ls, "SSL_CIPHER_standard_name", &stdNameOk);
        g_ossl_.has_std_name = stdNameOk;
        g_ossl_.ssl_alpn_selected = (OsslAlpnSelected_)ossl_sym_(ls, "SSL_get0_alpn_selected", &ok);
        g_ossl_.ssl_verify_result = (OsslVerifyResult_)ossl_sym_(ls, "SSL_get_verify_result", &ok);
        g_ossl_.ssl_set_bio = (OsslSetBio_)ossl_sym_(ls, "SSL_set_bio", &ok);
        int v3 = 1;
        g_ossl_.ssl_peer_cert = (OsslPeerCert_)ossl_sym_(ls, "SSL_get1_peer_certificate", &v3);     // 3.x
        g_ossl_.v3 = v3;
        if (!v3) { g_ossl_.ssl_peer_cert = (OsslPeerCert_)ossl_sym_(ls, "SSL_get_peer_certificate", &ok); }   // 1.1
        g_ossl_.err_get = (OsslErrGet_)ossl_sym_(lc, "ERR_get_error", &ok);
        g_ossl_.err_string = (OsslErrString_)ossl_sym_(lc, "ERR_error_string_n", &ok);
        g_ossl_.err_clear = (OsslErrClear_)ossl_sym_(lc, "ERR_clear_error", &ok);
        g_ossl_.verify_string = (OsslVerifyString_)ossl_sym_(lc, "X509_verify_cert_error_string", &ok);
        g_ossl_.bio_s_mem = (OsslBioMethodFn_)ossl_sym_(lc, "BIO_s_mem", &ok);
        g_ossl_.bio_new = (OsslBioNew_)ossl_sym_(lc, "BIO_new", &ok);
        g_ossl_.bio_free = (OsslBioFree_)ossl_sym_(lc, "BIO_free", &ok);
        g_ossl_.bio_read = (OsslBioRead_)ossl_sym_(lc, "BIO_read", &ok);
        g_ossl_.bio_write = (OsslBioWrite_)ossl_sym_(lc, "BIO_write", &ok);
        g_ossl_.bio_pending = (OsslPending_)ossl_sym_(lc, "BIO_ctrl_pending", &ok);
        g_ossl_.bio_ctrl = (OsslBioCtrl_)ossl_sym_(lc, "BIO_ctrl", &ok);
        g_ossl_.bio_mem_buf = (OsslBioMemBuf_)ossl_sym_(lc, "BIO_new_mem_buf", &ok);
        g_ossl_.pem_read_x509 = (OsslPemReadX509_)ossl_sym_(lc, "PEM_read_bio_X509", &ok);
        g_ossl_.pem_read_key = (OsslPemReadKey_)ossl_sym_(lc, "PEM_read_bio_PrivateKey", &ok);
        g_ossl_.x509_free = (OsslX509Free_)ossl_sym_(lc, "X509_free", &ok);
        g_ossl_.pkey_free = (OsslPkeyFree_)ossl_sym_(lc, "EVP_PKEY_free", &ok);
        g_ossl_.store_add_cert = (OsslStoreAdd_)ossl_sym_(lc, "X509_STORE_add_cert", &ok);
        g_ossl_.i2d_x509 = (OsslI2d_)ossl_sym_(lc, "i2d_X509", &ok);
        g_ossl_.param_set_ip = (OsslParamSetIp_)ossl_sym_(lc, "X509_VERIFY_PARAM_set1_ip_asc", &ok);
        g_ossl_.param_set_hostflags = (OsslHostFlags_)ossl_sym_(lc, "X509_VERIFY_PARAM_set_hostflags", &ok);
        if (ok) {
            if (g_ossl_.init_ssl(0ULL, (const OsslInitSettings_*)0) != 1) { ok = 0; }
        }
    }
    if (ok) { g_ossl_.loaded = 1; }
    g_ossl_lock_.unlock();
    return ok;
}

int tls_openssl_available_() { return ossl_load_(); }

// ── errors ────────────────────────────────────────────────────────────────────

// The OpenSSL error queue's first entry (or 'fallback') into e.
static int ossl_fail_(&TlsError e, int code, const &char what) {
    unsigned long err = 0UL;
    char buf[200];
    buf[0] = (char)0;
    unsafe {
        err = g_ossl_.err_get();
        if (err != 0UL) { g_ossl_.err_string(err, (char*)&buf[0], (size_t)200); }
        g_ossl_.err_clear();
    }
    e.detail = (long long)err;
    if (err != 0UL) {
        // Reasons 1000 + n are TLS alerts received from the peer.
        unsigned long reason = g_ossl_.v3 ? (err & 0x7FFFFFUL) : (err & 0xFFFUL);
        if (reason >= 1000UL && reason < 1256UL) {
            e.alert = (int)(reason - 1000UL);
            code = TLS_ERR_ALERT;
        }
        return tls_fail2_(e, code, what, &buf[0]);
    }
    return tls_fail_(e, code, what);
}

static int ossl_verify_code_(long v) {
    if (v == 9L || v == 10L) { return TLS_ERR_CERT_EXPIRED; }
    if (v == 2L || v == 18L || v == 19L || v == 20L || v == 21L) { return TLS_ERR_CERT_UNTRUSTED; }
    if (v == 62L || v == 63L || v == 64L) { return TLS_ERR_CERT_HOSTNAME; }
    if (v == 23L) { return TLS_ERR_CERT_REVOKED; }
    return TLS_ERR_CERT_INVALID;
}

// ── per-stream state ──────────────────────────────────────────────────────────

struct OsslConn_ {
    OsslSsl_* ssl;
    OsslCtx_* ctx;          // owned context (null: the shared one)
    OsslBio_* rbio;         // network -> OpenSSL (owned by ssl)
    OsslBio_* wbio;         // OpenSSL -> network
    int   eof;
    unsigned char out[16384 + 512];
    unsigned long out_pos;
    unsigned long out_len;
    unsigned char in[16384 + 512];
};

static struct OsslConn_* ossl_of_(&TlsStream s) {
    unsafe { return (struct OsslConn_*)s.impl_; }
}

// Sends what OpenSSL wrote to wbio: 0 when all is out, TLS_WANT_WRITE, -1.
static int ossl_flush_(&TlsStream s, struct OsslConn_* c) {
    unsafe {
        while (1) {
            if (c->out_pos < c->out_len) {
                long long n = tls_sock_send_(s.fd, (const &unsigned char)((unsigned char*)&c->out[0] + c->out_pos),
                                             c->out_len - c->out_pos);
                if (n == (long long)TLS_WANT_WRITE) { return TLS_WANT_WRITE; }
                if (n <= 0LL) { return -1; }
                c->out_pos = c->out_pos + (unsigned long)n;
                continue;
            }
            if (g_ossl_.bio_pending(c->wbio) == (size_t)0) { return 0; }
            int k = g_ossl_.bio_read(c->wbio, (unsigned char*)&c->out[0], (int)sizeof(c->out));
            if (k <= 0) { return 0; }
            c->out_pos = 0UL;
            c->out_len = (unsigned long)k;
        }
    }
    return 0;
}

// Feeds socket bytes to rbio: 1 fed, 0 EOF (rbio then reports EOF),
// TLS_WANT_READ, -1.
static int ossl_fill_(&TlsStream s, struct OsslConn_* c) {
    unsafe {
        long long n = tls_sock_recv_(s.fd, (&unsigned char)&c->in[0], (unsigned long)sizeof(c->in));
        if (n == (long long)TLS_WANT_READ) { return TLS_WANT_READ; }
        if (n < 0LL) { return -1; }
        if (n == 0LL) {
            c->eof = 1;
            g_ossl_.bio_ctrl(c->rbio, OSSL_C_SET_BUF_MEM_EOF_RETURN_, 0L, (unsigned char*)0);
            return 0;
        }
        g_ossl_.bio_write(c->rbio, (unsigned char*)&c->in[0], (int)n);
    }
    return 1;
}

// op: 0 handshake, 1 read, 2 write, 3 shutdown.
static long long ossl_call_(&TlsStream s, int op, ?&unsigned char rbuf, const ?&unsigned char wbuf, unsigned long n) {
    struct OsslConn_* c = ossl_of_(s);
    unsafe {
        while (1) {
            int f = ossl_flush_(s, c);
            if (f == TLS_WANT_WRITE) { return (long long)TLS_WANT_WRITE; }
            if (f < 0) { return (long long)tls_fail_(&s.err, TLS_ERR_IO, "TLS: socket write failed"); }
            g_ossl_.err_clear();
            int r = 0;
            int ask = n > 0x40000000UL ? 0x40000000 : (int)n;
            if (op == 0) { r = g_ossl_.ssl_do_handshake(c->ssl); }
            else if (op == 1) { r = g_ossl_.ssl_read(c->ssl, (unsigned char*)rbuf, ask); }
            else if (op == 2) { r = g_ossl_.ssl_write(c->ssl, (const unsigned char*)wbuf, ask); }
            else { r = g_ossl_.ssl_shutdown(c->ssl); }
            int f2 = ossl_flush_(s, c);
            if (f2 < 0 && f2 != TLS_WANT_WRITE) { return (long long)tls_fail_(&s.err, TLS_ERR_IO, "TLS: socket write failed"); }
            if (r > 0) { return op == 0 || op == 3 ? 0LL : (long long)r; }
            if (op == 3 && r == 0) { return 0LL; }   // close_notify sent, peer's not awaited
            int e = g_ossl_.ssl_get_error(c->ssl, r);
            if (e == OSSL_ERROR_WANT_READ_) {
                if (f2 == TLS_WANT_WRITE) { return (long long)TLS_WANT_WRITE; }
                if (c->eof) {
                    if (op == 1) { return 0LL; }
                    return (long long)tls_fail_(&s.err, TLS_ERR_CLOSED, "the peer closed the connection during the TLS handshake");
                }
                int fr = ossl_fill_(s, c);
                if (fr == TLS_WANT_READ) { return (long long)TLS_WANT_READ; }
                if (fr < 0) { return (long long)tls_fail_(&s.err, TLS_ERR_IO, "TLS: socket read failed"); }
                continue;
            }
            if (e == OSSL_ERROR_WANT_WRITE_) { continue; }
            if (e == OSSL_ERROR_ZERO_RETURN_) {
                if (op == 1) { return 0LL; }
                return (long long)tls_fail_(&s.err, TLS_ERR_CLOSED, "the peer closed the TLS connection");
            }
            if (op == 1 && c->eof && e == OSSL_ERROR_SYSCALL_) { return 0LL; }   // EOF without close_notify
            if (op == 0) {
                long v = g_ossl_.ssl_verify_result(c->ssl);
                if (v != 0L) {
                    g_ossl_.err_clear();
                    const char* vs = g_ossl_.verify_string(v);
                    tls_fail2_(&s.err, ossl_verify_code_(v), "certificate verification failed", (const &char)vs);
                    s.err.detail = (long long)v;
                    return -1LL;
                }
                if (c->eof && e == OSSL_ERROR_SYSCALL_) {
                    return (long long)tls_fail_(&s.err, TLS_ERR_CLOSED, "the peer closed the connection during the TLS handshake");
                }
                return (long long)ossl_fail_(&s.err, TLS_ERR_HANDSHAKE, "TLS handshake failed");
            }
            return (long long)ossl_fail_(&s.err, TLS_ERR_PROTOCOL, "TLS error");
        }
    }
    return -1LL;
}

// ── contexts ──────────────────────────────────────────────────────────────────

static void ossl_set_versions_(OsslCtx_* ctx, int min_version, int max_version) {
    unsafe {
        g_ossl_.ctx_ctrl(ctx, OSSL_CTRL_SET_MIN_PROTO_, (long)min_version, (OsslX509_*)0);
        g_ossl_.ctx_ctrl(ctx, OSSL_CTRL_SET_MAX_PROTO_, (long)max_version, (OsslX509_*)0);
        // SSL_MODE_ENABLE_PARTIAL_WRITE | SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER | SSL_MODE_AUTO_RETRY
        g_ossl_.ctx_ctrl(ctx, OSSL_CTRL_MODE_, 7L, (OsslX509_*)0);
    }
}

// Adds every certificate of a PEM text to the context's trust store.
static int ossl_add_ca_pem_(OsslCtx_* ctx, const &char pem) {
    int n = 0;
    unsafe {
        OsslStore_* store = g_ossl_.ctx_get_store(ctx);
        OsslBio_* bio = g_ossl_.bio_mem_buf((const char*)pem, -1);
        if (bio == (OsslBio_*)0) { return 0; }
        while (1) {
            OsslX509_* x = g_ossl_.pem_read_x509(bio, (OsslX509_**)0, null, null);
            if (x == (OsslX509_*)0) { break; }
            if (g_ossl_.store_add_cert(store, x) == 1) { n = n + 1; }
            g_ossl_.x509_free(x);
        }
        g_ossl_.bio_free(bio);
        g_ossl_.err_clear();
    }
    return n;
}

// The system roots: OpenSSL's defaults, then the distribution bundles
// (OpenSSL builds that don't look there, e.g. a statically configured
// OPENSSLDIR) when the defaults hold nothing usable.
static void ossl_system_roots_(OsslCtx_* ctx) {
    unsafe {
        g_ossl_.ctx_default_paths(ctx);
        const char* paths[6];
        paths[0] = "/etc/ssl/certs/ca-certificates.crt";
        paths[1] = "/etc/pki/tls/certs/ca-bundle.crt";
        paths[2] = "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem";
        paths[3] = "/etc/ssl/ca-bundle.pem";
        paths[4] = "/etc/ssl/cert.pem";
        paths[5] = "/usr/local/share/certs/ca-root-nss.crt";
        ?&char env = tls_getenv_("SSL_CERT_FILE");
        if (env.is_null()) {
            int i = 0;
            while (i < 6) {
                if (g_ossl_.ctx_load_locations(ctx, paths[i], (const char*)0) == 1) { break; }
                i = i + 1;
            }
        }
        g_ossl_.err_clear();
    }
}

// The shared client context with the system roots (created once).
static OsslCtx_* ossl_shared_client_ctx_() {
    g_ossl_lock_.lock();
    if (g_ossl_client_ctx_ == (OsslCtx_*)0) {
        unsafe {
            OsslCtx_* ctx = g_ossl_.ctx_new(g_ossl_.client_method());
            if (ctx != (OsslCtx_*)0) {
                ossl_set_versions_(ctx, TLS_VERSION_1_2, TLS_VERSION_1_3);
                ossl_system_roots_(ctx);
                g_ossl_client_ctx_ = ctx;
            }
        }
    }
    OsslCtx_* r = g_ossl_client_ctx_;
    g_ossl_lock_.unlock();
    return r;
}

// Loads a PEM certificate chain + key into an SSL (client) or SSL_CTX (server).
// Into an SSL_CTX (server: ctx set) or an SSL (client: ssl set).
static int ossl_use_cert_key_(OsslCtx_* ctx, OsslSsl_* ssl, const &char cert_pem, const &char key_pem, &TlsError e) {
    int is_ctx = ctx != (OsslCtx_*)0;
    unsafe {
        OsslBio_* bio = g_ossl_.bio_mem_buf((const char*)cert_pem, -1);
        if (bio == (OsslBio_*)0) { return tls_fail_(e, TLS_ERR_NO_MEMORY, "out of memory"); }
        OsslX509_* leaf = g_ossl_.pem_read_x509(bio, (OsslX509_**)0, null, null);
        if (leaf == (OsslX509_*)0) {
            g_ossl_.bio_free(bio);
            return ossl_fail_(e, TLS_ERR_CONFIG, "cannot read the certificate PEM");
        }
        int ok = is_ctx ? g_ossl_.ctx_use_cert(ctx, leaf) : g_ossl_.ssl_use_cert(ssl, leaf);
        g_ossl_.x509_free(leaf);
        if (ok != 1) { g_ossl_.bio_free(bio); return ossl_fail_(e, TLS_ERR_CONFIG, "certificate rejected"); }
        while (1) {
            OsslX509_* x = g_ossl_.pem_read_x509(bio, (OsslX509_**)0, null, null);
            if (x == (OsslX509_*)0) { break; }
            if (is_ctx) {
                // the context owns x from here
                if (g_ossl_.ctx_ctrl(ctx, OSSL_CTRL_EXTRA_CHAIN_CERT_, 0L, x) != 1L) { g_ossl_.x509_free(x); }
            } else {
                // SSL_add1_chain_cert: the SSL takes its own reference
                g_ossl_.ssl_ctrl(ssl, OSSL_CTRL_CHAIN_CERT_, 1L, x);
                g_ossl_.x509_free(x);
            }
        }
        g_ossl_.bio_free(bio);
        g_ossl_.err_clear();
        OsslBio_* kb = g_ossl_.bio_mem_buf((const char*)key_pem, -1);
        if (kb == (OsslBio_*)0) { return tls_fail_(e, TLS_ERR_NO_MEMORY, "out of memory"); }
        OsslPkey_* key = g_ossl_.pem_read_key(kb, (OsslPkey_**)0, null, null);
        g_ossl_.bio_free(kb);
        if (key == (OsslPkey_*)0) { return ossl_fail_(e, TLS_ERR_CONFIG, "cannot read the private key PEM"); }
        ok = is_ctx ? g_ossl_.ctx_use_key(ctx, key) : g_ossl_.ssl_use_key(ssl, key);
        g_ossl_.pkey_free(key);
        if (ok != 1) { return ossl_fail_(e, TLS_ERR_CONFIG, "private key rejected (does it match the certificate?)"); }
        if (is_ctx && g_ossl_.ctx_check_key(ctx) != 1) {
            return ossl_fail_(e, TLS_ERR_CONFIG, "the private key doesn't match the certificate");
        }
    }
    return 0;
}

static int ossl_new_conn_(&TlsStream s, OsslCtx_* ctx, OsslCtx_* owned) {
    unsigned char* mem = alloc_zeroed_raw_(sizeof(struct OsslConn_));
    if (mem == (unsigned char*)0) { return tls_fail_(&s.err, TLS_ERR_NO_MEMORY, "out of memory"); }
    unsafe {
        struct OsslConn_* c = (struct OsslConn_*)mem;
        c->ctx = owned;
        c->ssl = g_ossl_.ssl_new(ctx);
        if (c->ssl == (OsslSsl_*)0) {
            dealloc_raw_(mem);
            return ossl_fail_(&s.err, TLS_ERR_INTERNAL, "SSL_new failed");
        }
        c->rbio = g_ossl_.bio_new(g_ossl_.bio_s_mem());
        c->wbio = g_ossl_.bio_new(g_ossl_.bio_s_mem());
        g_ossl_.ssl_set_bio(c->ssl, c->rbio, c->wbio);
    }
    unsafe { s.impl_ = (?&heap TlsImpl_)mem; }
    return 0;
}

int tls_openssl_client_(&TlsStream s, const &char name, const &TlsConfig cfg) {
    if (!ossl_load_()) { tls_fail_(&s.err, TLS_ERR_UNAVAILABLE, "OpenSSL (libssl) not found"); return TLS_ERR_UNAVAILABLE; }
    OsslCtx_* ctx = (OsslCtx_*)0;
    OsslCtx_* owned = (OsslCtx_*)0;
    int custom = !cfg.ca_pem.is_empty() || !cfg.ca_file.is_empty() || !cfg.use_system_roots;
    unsafe {
        if (custom) {
            owned = g_ossl_.ctx_new(g_ossl_.client_method());
            if (owned == (OsslCtx_*)0) { ossl_fail_(&s.err, TLS_ERR_INTERNAL, "SSL_CTX_new failed"); return TLS_ERR_INTERNAL; }
            ossl_set_versions_(owned, TLS_VERSION_1_2, TLS_VERSION_1_3);
            if (cfg.use_system_roots) { ossl_system_roots_(owned); }
            if (!cfg.ca_pem.is_empty() && ossl_add_ca_pem_(owned, cfg.ca_pem.as_ptr()) == 0) {
                g_ossl_.ctx_free(owned);
                tls_fail_(&s.err, TLS_ERR_CONFIG, "ca_pem holds no certificate");
                return TLS_ERR_CONFIG;
            }
            if (!cfg.ca_file.is_empty() && g_ossl_.ctx_load_locations(owned, (const char*)cfg.ca_file.as_ptr(), (const char*)0) != 1) {
                g_ossl_.ctx_free(owned);
                ossl_fail_(&s.err, TLS_ERR_CONFIG, "cannot load ca_file");
                return TLS_ERR_CONFIG;
            }
            ctx = owned;
        } else {
            ctx = ossl_shared_client_ctx_();
            if (ctx == (OsslCtx_*)0) { ossl_fail_(&s.err, TLS_ERR_INTERNAL, "SSL_CTX_new failed"); return TLS_ERR_INTERNAL; }
        }
    }
    if (ossl_new_conn_(s, ctx, owned) != 0) {
        unsafe { if (owned != (OsslCtx_*)0) { g_ossl_.ctx_free(owned); } }
        return s.err.code;
    }
    struct OsslConn_* c = ossl_of_(s);
    unsafe {
        OsslSsl_* ssl = c->ssl;
        g_ossl_.ssl_ctrl(ssl, OSSL_CTRL_SET_MIN_PROTO_, (long)cfg.min_version, (OsslX509_*)0);
        g_ossl_.ssl_ctrl(ssl, OSSL_CTRL_SET_MAX_PROTO_, (long)cfg.max_version, (OsslX509_*)0);
        int ip = tls_name_is_ip_(name);
        if (!ip && str_len(name) > 0UL) {
            g_ossl_.ssl_ctrl_name(ssl, OSSL_CTRL_SET_TLSEXT_HOSTNAME_, 0L, (const char*)name);
        }
        if (cfg.verify_peer) {
            g_ossl_.ssl_set_verify(ssl, OSSL_VERIFY_PEER_, null);
            if (cfg.verify_hostname) {
                if (ip) {
                    g_ossl_.param_set_ip(g_ossl_.ssl_get0_param(ssl), (const char*)name);
                } else {
                    g_ossl_.param_set_hostflags(g_ossl_.ssl_get0_param(ssl), OSSL_X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS_);
                    g_ossl_.ssl_set1_host(ssl, (const char*)name);
                }
            }
        } else {
            g_ossl_.ssl_set_verify(ssl, OSSL_VERIFY_NONE_, null);
        }
        if (!cfg.alpn.is_empty()) {
            struct Vec<unsigned char> w = tls_alpn_wire_(cfg.alpn.as_ptr());
            // SSL_set_alpn_protos returns 0 on success
            if (w.len > 0UL) { g_ossl_.ssl_set_alpn(ssl, (const unsigned char*)w.data, (unsigned int)w.len); }
            w.free();
        }
        if (!cfg.cert_pem.is_empty()) {
            if (ossl_use_cert_key_((OsslCtx_*)0, ssl, cfg.cert_pem.as_ptr(), cfg.key_pem.as_ptr(), &s.err) != 0) {
                int code = s.err.code;
                tls_openssl_free_(s);
                return code;
            }
        }
        g_ossl_.ssl_set_connect(ssl);
    }
    return TLS_OK;
}

static void ossl_fill_info_(&TlsStream s, struct OsslConn_* c) {
    unsafe {
        int v = g_ossl_.ssl_version(c->ssl);
        s.version_ = v;
        const OsslCipher_* ci = g_ossl_.ssl_get_cipher(c->ssl);
        if (ci != (const OsslCipher_*)0) {
            const char* nm = (const char*)0;
            if (g_ossl_.has_std_name) { nm = g_ossl_.cipher_std_name(ci); }
            if (nm == (const char*)0) { nm = g_ossl_.cipher_name(ci); }
            tls_copy_cstr_(&s.cipher_[0], 64UL, (const ?&char)nm);
        }
        const unsigned char* ap = (const unsigned char*)0;
        unsigned int al = 0U;
        g_ossl_.ssl_alpn_selected(c->ssl, &ap, &al);
        unsigned int i = 0U;
        while (i < al && i < 63U) { s.alpn_[i] = (char)ap[i]; i = i + 1U; }
        s.alpn_[i] = (char)0;
        OsslX509_* x = g_ossl_.ssl_peer_cert(c->ssl);
        if (x != (OsslX509_*)0) {
            int len = g_ossl_.i2d_x509(x, (unsigned char**)0);
            if (len > 0) {
                s.peer_der_.free();
                s.peer_der_ = Vec::with_cap((unsigned long)len);
                unsigned char* p = (unsigned char*)s.peer_der_.data;
                if (p != (unsigned char*)0) {
                    unsigned char* q = p;
                    g_ossl_.i2d_x509(x, &q);
                    s.peer_der_.len = (unsigned long)len;
                }
            }
            g_ossl_.x509_free(x);   // get1 (3.x) and get (1.1) both return a reference to drop
        }
    }
}

int tls_openssl_handshake_(&TlsStream s) {
    long long r = ossl_call_(s, 0, null, null, 0UL);
    if (r == 0LL) {
        ossl_fill_info_(s, ossl_of_(s));
        return 0;
    }
    return (int)r;
}

long long tls_openssl_read_(&TlsStream s, &unsigned char buf, unsigned long n) {
    return ossl_call_(s, 1, buf, null, n);
}

long long tls_openssl_write_(&TlsStream s, const &unsigned char buf, unsigned long n) {
    return ossl_call_(s, 2, null, buf, n);
}

int tls_openssl_shutdown_(&TlsStream s) {
    return (int)ossl_call_(s, 3, null, null, 0UL);
}

void tls_openssl_free_(&TlsStream s) {
    struct OsslConn_* c = ossl_of_(s);
    if (c == (struct OsslConn_*)0) { return; }
    unsafe {
        if (c->ssl != (OsslSsl_*)0) { g_ossl_.ssl_free(c->ssl); }   // frees the BIOs too
        if (c->ctx != (OsslCtx_*)0) { g_ossl_.ctx_free(c->ctx); }
        dealloc(c);
    }
    s.impl_ = null;
}

// ── server ────────────────────────────────────────────────────────────────────

// SSL_CTX_set_alpn_select_cb's callback: the server's preference among the
// client's offers (SSL_select_next_proto), no ALPN when nothing matches.
static int ossl_alpn_select_(OsslSsl_* ssl, unsigned char** out, unsigned char* outlen,
                             const unsigned char* in, unsigned int inlen, OsslAcc_* a) {
    unsafe {
        if (a->alpn.len == 0UL) { return 3; }
        int r = g_ossl_.select_next_proto(out, outlen, (const unsigned char*)a->alpn.data, (unsigned int)a->alpn.len, in, inlen);
        return r == 1 ? 0 : 3;   // SSL_TLSEXT_ERR_OK / _NOACK
    }
}

int tls_openssl_acceptor_(&TlsAcceptor a, const &TlsServerConfig cfg, &TlsError e) {
    if (!ossl_load_()) { tls_fail_(e, TLS_ERR_UNAVAILABLE, "OpenSSL (libssl) not found"); return TLS_ERR_UNAVAILABLE; }
    unsigned char* mem = alloc_zeroed_raw_(sizeof(struct OsslAcc_));
    if (mem == (unsigned char*)0) { tls_fail_(e, TLS_ERR_NO_MEMORY, "out of memory"); return TLS_ERR_NO_MEMORY; }
    unsafe {
        struct OsslAcc_* acc = (struct OsslAcc_*)mem;
        acc->alpn = Vec::new();
        OsslCtx_* ctx = g_ossl_.ctx_new(g_ossl_.server_method());
        if (ctx == (OsslCtx_*)0) { dealloc_raw_(mem); ossl_fail_(e, TLS_ERR_INTERNAL, "SSL_CTX_new failed"); return TLS_ERR_INTERNAL; }
        acc->ctx = ctx;
        ossl_set_versions_(ctx, cfg.min_version, cfg.max_version);
        if (ossl_use_cert_key_(ctx, (OsslSsl_*)0, cfg.cert_pem.as_ptr(), cfg.key_pem.as_ptr(), e) != 0) {
            g_ossl_.ctx_free(ctx);
            dealloc_raw_(mem);
            return e.code;
        }
        if (!cfg.alpn.is_empty()) {
            acc->alpn.free();
            acc->alpn = tls_alpn_wire_(cfg.alpn.as_ptr());
            g_ossl_.ctx_set_alpn_cb(ctx, ossl_alpn_select_, acc);
        }
        if (cfg.verify_client != 0) {
            if (ossl_add_ca_pem_(ctx, cfg.client_ca_pem.as_ptr()) == 0) {
                g_ossl_.ctx_free(ctx);
                acc->alpn.free();
                dealloc_raw_(mem);
                tls_fail_(e, TLS_ERR_CONFIG, "verify_client needs client_ca_pem roots");
                return TLS_ERR_CONFIG;
            }
            int mode = OSSL_VERIFY_PEER_;
            if (cfg.verify_client == 2) { mode = mode | OSSL_VERIFY_FAIL_NO_CERT_; }
            g_ossl_.ctx_set_verify(ctx, mode, null);
        }
    }
    unsafe { a.impl_ = (?&heap TlsImpl_)mem; }
    return TLS_OK;
}

int tls_openssl_accept_(&TlsStream s, const &TlsAcceptor a) {
    unsafe {
        struct OsslAcc_* acc = (struct OsslAcc_*)a.impl_;
        if (ossl_new_conn_(s, acc->ctx, (OsslCtx_*)0) != 0) { return s.err.code; }
        g_ossl_.ssl_set_accept(ossl_of_(s)->ssl);
    }
    return TLS_OK;
}

void tls_openssl_acceptor_free_(&TlsAcceptor a) {
    unsafe {
        struct OsslAcc_* acc = (struct OsslAcc_*)a.impl_;
        if (acc == (struct OsslAcc_*)0) { return; }
        g_ossl_.ctx_free(acc->ctx);   // reference-counted: live SSLs keep it
        acc->alpn.free();
        dealloc(acc);
    }
}

} // namespace std
