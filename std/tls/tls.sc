// SafeC Standard Library — TLS: configuration, backend selection and the
// TlsStream / TlsAcceptor front end (see tls.h; backends in tls_*.sc).
#pragma once
#include <std/stddef.h>
#include <std/tls/tls.h>
#include <std/tls/tls_backend.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>
#include <std/encoding/base64.h>
#include <std/sched/io_nb.h>
#include <std/errno.h>
#include <std/mem.h>
#include <std/str.h>

namespace std {

extern char* getenv(const char* name);
extern void* fopen(const char* path, const char* mode);
extern unsigned long fread(void* buf, unsigned long size, unsigned long n, void* f);
extern int fclose(void* f);

#ifdef _WIN32
extern int recv(unsigned long long fd, void* buf, int count, int flags);
extern int send(unsigned long long fd, const void* buf, int count, int flags);
extern int closesocket(unsigned long long fd);
static int tls_sock_close_(int fd) { unsafe { return closesocket((unsigned long long)fd); } }
#else
extern ssize_t recv(int fd, void* buf, size_t count, int flags);
extern ssize_t send(int fd, const void* buf, size_t count, int flags);
extern int close(int fd);
static int tls_sock_close_(int fd) { unsafe { return close(fd); } }
#endif
#ifdef __linux__
#define TLS_MSG_NOSIGNAL_ 0x4000
#else
#define TLS_MSG_NOSIGNAL_ 0
#endif
#define TLS_EINTR_ 4

// ── errors ────────────────────────────────────────────────────────────────────

struct TlsError TlsError::none() {
    struct TlsError e;
    e.code = TLS_OK;
    e.alert = -1;
    e.detail = 0LL;
    e.message[0] = (char)0;
    return e;
}

struct TlsError TlsError::make(int code, const &char message) {
    struct TlsError e = TlsError::none();
    e.code = code;
    tls_copy_cstr_(&e.message[0], (unsigned long)TLS_ERROR_MESSAGE_MAX, message);
    return e;
}

int TlsError::ok() const { return self.code == TLS_OK; }

const &char tls_error_name(int code) {
    if (code == TLS_OK) { return "ok"; }
    if (code == TLS_ERR_UNAVAILABLE) { return "backend unavailable"; }
    if (code == TLS_ERR_CONFIG) { return "bad configuration"; }
    if (code == TLS_ERR_IO) { return "socket error"; }
    if (code == TLS_ERR_CLOSED) { return "connection closed"; }
    if (code == TLS_ERR_TIMEOUT) { return "timed out"; }
    if (code == TLS_ERR_PROTOCOL) { return "protocol error"; }
    if (code == TLS_ERR_ALERT) { return "alert from peer"; }
    if (code == TLS_ERR_HANDSHAKE) { return "handshake failure"; }
    if (code == TLS_ERR_CERT_UNTRUSTED) { return "untrusted certificate"; }
    if (code == TLS_ERR_CERT_HOSTNAME) { return "certificate name mismatch"; }
    if (code == TLS_ERR_CERT_EXPIRED) { return "certificate expired or not yet valid"; }
    if (code == TLS_ERR_CERT_INVALID) { return "invalid certificate"; }
    if (code == TLS_ERR_CERT_REVOKED) { return "certificate revoked"; }
    if (code == TLS_ERR_NO_CERT) { return "no peer certificate"; }
    if (code == TLS_ERR_NO_MEMORY) { return "out of memory"; }
    if (code == TLS_ERR_STATE) { return "invalid state"; }
    return "internal error";
}

void tls_copy_cstr_(&char dst, unsigned long cap, const ?&char src) {
    unsafe {
        char* d = (char*)dst;
        unsigned long i = 0UL;
        const char* s = (const char*)src;
        if (s != (const char*)0) {
            while (i + 1UL < cap && s[i] != (char)0) { d[i] = s[i]; i = i + 1UL; }
        }
        d[i] = (char)0;
    }
}

int tls_fail_(&TlsError e, int code, const &char message) {
    e.code = code;
    tls_copy_cstr_(&e.message[0], (unsigned long)TLS_ERROR_MESSAGE_MAX, message);
    return -1;
}

int tls_fail2_(&TlsError e, int code, const &char prefix, const &char detail) {
    struct String m = String::from(prefix);
    m.push(": ");
    m.push(detail);
    tls_fail_(e, code, m.as_ptr());
    m.free();
    return -1;
}

// ── small helpers ─────────────────────────────────────────────────────────────

long long tls_sock_recv_(int fd, &unsigned char buf, unsigned long n) {
    while (1) {
        long long r = 0LL;
        unsafe {
#ifdef _WIN32
            r = (long long)recv((unsigned long long)fd, (void*)buf, (int)n, 0);
#else
            r = (long long)recv(fd, (void*)buf, n, 0);
#endif
        }
        if (r >= 0LL) { return r; }
        if (sock_would_block()) { return (long long)TLS_WANT_READ; }
#ifndef _WIN32
        if (errno_get() == TLS_EINTR_) { continue; }
#endif
        return -1LL;
    }
    return -1LL;
}

long long tls_sock_send_(int fd, const &unsigned char buf, unsigned long n) {
    while (1) {
        long long r = 0LL;
        unsafe {
#ifdef _WIN32
            r = (long long)send((unsigned long long)fd, (const void*)buf, (int)n, 0);
#else
            r = (long long)send(fd, (const void*)buf, n, TLS_MSG_NOSIGNAL_);
#endif
        }
        if (r >= 0LL) { return r; }
        if (sock_would_block()) { return (long long)TLS_WANT_WRITE; }
#ifndef _WIN32
        if (errno_get() == TLS_EINTR_) { continue; }
#endif
        return -1LL;
    }
    return -1LL;
}

int tls_name_is_ip_(const &char name) {
    unsafe {
        const char* s = (const char*)name;
        int colon = 0;
        int nondigit = 0;
        unsigned long i = 0UL;
        while (s[i] != (char)0) {
            char c = s[i];
            if (c == ':') { colon = 1; }
            else if (!((c >= '0' && c <= '9') || c == '.')) { nondigit = 1; }
            i = i + 1UL;
        }
        if (i == 0UL) { return 0; }
        return colon || !nondigit;
    }
}

?&char tls_getenv_(const &char name) {
    unsafe { return (?&char)getenv((const char*)name); }
}

int tls_read_file_(const &char path, &String out) {
    unsafe {
        void* f = fopen((const char*)path, "rb");
        if (f == (void*)0) { return 0; }
        unsigned char chunk[4096];
        while (1) {
            unsigned long n = fread((void*)&chunk[0], 1UL, 4096UL, f);
            if (n == 0UL) { break; }
            out.push_n((const &char)(const char*)&chunk[0], n);
        }
        fclose(f);
    }
    return 1;
}

static int tls_starts_(const char* s, const char* prefix) {
    unsafe {
        unsigned long i = 0UL;
        while (prefix[i] != (char)0) { if (s[i] != prefix[i]) { return 0; } i = i + 1UL; }
    }
    return 1;
}

// Scans text from *pos for the next "-----BEGIN <label>-----": the label
// into lab, the base64 body (up to "-----END") into body. 0 when no block.
static int tls_pem_next_(const char* text, &unsigned long pos, &String lab, &String body) {
    unsafe {
        unsigned long i = *pos;
        while (text[i] != (char)0) {
            if (tls_starts_(text + i, "-----BEGIN ")) {
                unsigned long j = i + 11UL;
                unsigned long ls = j;
                while (text[j] != (char)0 && text[j] != '-') { j = j + 1UL; }
                if (!tls_starts_(text + j, "-----")) { i = i + 1UL; continue; }
                lab.clear();
                lab.push_n((const &char)(text + ls), j - ls);
                j = j + 5UL;
                unsigned long bs = j;
                while (text[j] != (char)0 && !tls_starts_(text + j, "-----END")) { j = j + 1UL; }
                if (text[j] == (char)0) { *pos = j; return 0; }
                body.clear();
                body.push_n((const &char)(text + bs), j - bs);
                // past the END line
                while (text[j] != (char)0 && text[j] != '\n') { j = j + 1UL; }
                *pos = j;
                return 1;
            }
            i = i + 1UL;
        }
        *pos = i;
    }
    return 0;
}

unsigned long tls_pem_decode_(const &char text, const ?&char label, &Vec<struct Vec<unsigned char>> out) {
    unsigned long pos = 0UL;
    unsigned long count = 0UL;
    struct String lab = String::new();
    struct String body = String::new();
    unsafe {
        while (tls_pem_next_((const char*)text, &pos, &lab, &body)) {
            int want = 1;
            match (label) {
                case some(l): { want = lab.eq_cstr(l); }
                case null: { }
            }
            if (!want) { continue; }
            int ok = 0;
            struct Vec<unsigned char> der = base64_decode(body.as_ptr(), &ok);
            if (!ok || der.len == 0UL) { der.free(); continue; }
            out.push(der);
            count = count + 1UL;
        }
    }
    lab.free();
    body.free();
    return count;
}

int tls_pem_private_key_(const &char text, &Vec<unsigned char> der, &int kind) {
    unsigned long pos = 0UL;
    struct String lab = String::new();
    struct String body = String::new();
    int found = 0;
    unsafe {
        while (!found && tls_pem_next_((const char*)text, &pos, &lab, &body)) {
            int k = 0;
            if (lab.eq_cstr("PRIVATE KEY")) { k = 1; }
            else if (lab.eq_cstr("RSA PRIVATE KEY")) { k = 2; }
            else if (lab.eq_cstr("EC PRIVATE KEY")) { k = 3; }
            if (k == 0) { continue; }
            int ok = 0;
            struct Vec<unsigned char> d = base64_decode(body.as_ptr(), &ok);
            if (!ok) { d.free(); continue; }
            der.free();
            *der = d;
            *kind = k;
            found = 1;
        }
    }
    mem_zero_secure(body.as_ptr(), body.length());
    lab.free();
    body.free();
    return found;
}

void tls_der_list_free_(&Vec<struct Vec<unsigned char>> list) {
    unsigned long i = 0UL;
    while (i < list.len) {
        match (list.get(i)) {
            case some(v): { v.free(); }
            case null: { }
        }
        i = i + 1UL;
    }
    list.free();
}

const &char tls_suite_name_(int id) {
    if (id == 0x1301) { return "TLS_AES_128_GCM_SHA256"; }
    if (id == 0x1302) { return "TLS_AES_256_GCM_SHA384"; }
    if (id == 0x1303) { return "TLS_CHACHA20_POLY1305_SHA256"; }
    if (id == 0xC02B) { return "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256"; }
    if (id == 0xC02C) { return "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384"; }
    if (id == 0xC02F) { return "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"; }
    if (id == 0xC030) { return "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384"; }
    if (id == 0xCCA8) { return "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256"; }
    if (id == 0xCCA9) { return "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256"; }
    return "";
}

struct Vec<unsigned char> tls_alpn_wire_(const &char list) {
    struct Vec<unsigned char> out = Vec::new();
    unsafe {
        const char* s = (const char*)list;
        unsigned long i = 0UL;
        while (1) {
            unsigned long st = i;
            while (s[i] != (char)0 && s[i] != ',') { i = i + 1UL; }
            unsigned long a = st;
            unsigned long b = i;
            while (a < b && s[a] == ' ') { a = a + 1UL; }
            while (b > a && s[b - 1UL] == ' ') { b = b - 1UL; }
            if (b > a && b - a < 256UL) {
                out.push((unsigned char)(b - a));
                out.extend(((const unsigned char*)(s + a))[0..b - a]);
            }
            if (s[i] == (char)0) { break; }
            i = i + 1UL;
        }
    }
    return out;
}

// ── TlsConfig / TlsServerConfig ───────────────────────────────────────────────

struct TlsConfig TlsConfig::new() {
    struct TlsConfig c;
    c.verify_peer = 1;
    c.verify_hostname = 1;
    c.use_system_roots = 1;
    c.ca_pem = String::new();
    c.ca_file = String::new();
    c.server_name = String::new();
    c.alpn = String::new();
    c.min_version = TLS_VERSION_1_2;
    c.max_version = TLS_VERSION_1_3;
    c.backend = TLS_BACKEND_AUTO;
    c.cert_pem = String::new();
    c.key_pem = String::new();
    return c;
}

void TlsConfig::set_ca_pem(const &char pem) { self.ca_pem.clear(); self.ca_pem.push(pem); }
void TlsConfig::set_ca_file(const &char path) { self.ca_file.clear(); self.ca_file.push(path); }
void TlsConfig::set_server_name(const &char name) { self.server_name.clear(); self.server_name.push(name); }
void TlsConfig::set_alpn(const &char protocols) { self.alpn.clear(); self.alpn.push(protocols); }
void TlsConfig::set_client_cert(const &char cert_pem, const &char key_pem) {
    self.cert_pem.clear(); self.cert_pem.push(cert_pem);
    self.key_pem.clear(); self.key_pem.push(key_pem);
}
void TlsConfig::insecure_skip_verify() { self.verify_peer = 0; self.verify_hostname = 0; }

struct TlsConfig TlsConfig::clone() const {
    struct TlsConfig c = *self;
    c.ca_pem = self.ca_pem.clone();
    c.ca_file = self.ca_file.clone();
    c.server_name = self.server_name.clone();
    c.alpn = self.alpn.clone();
    c.cert_pem = self.cert_pem.clone();
    c.key_pem = self.key_pem.clone();
    return c;
}

void TlsConfig::free() {
    self.ca_pem.free();
    self.ca_file.free();
    self.server_name.free();
    self.alpn.free();
    self.cert_pem.free();
    mem_zero_secure(self.key_pem.as_ptr(), self.key_pem.length());
    self.key_pem.free();
}

struct TlsServerConfig TlsServerConfig::new() {
    struct TlsServerConfig c;
    c.cert_pem = String::new();
    c.key_pem = String::new();
    c.alpn = String::new();
    c.min_version = TLS_VERSION_1_2;
    c.max_version = TLS_VERSION_1_3;
    c.backend = TLS_BACKEND_AUTO;
    c.verify_client = 0;
    c.client_ca_pem = String::new();
    return c;
}

int TlsServerConfig::load_files(const &char cert_path, const &char key_path) {
    self.cert_pem.clear();
    self.key_pem.clear();
    if (!tls_read_file_(cert_path, &self.cert_pem)) { return 0; }
    if (!tls_read_file_(key_path, &self.key_pem)) { return 0; }
    return 1;
}

void TlsServerConfig::set_alpn(const &char protocols) { self.alpn.clear(); self.alpn.push(protocols); }

void TlsServerConfig::free() {
    self.cert_pem.free();
    mem_zero_secure(self.key_pem.as_ptr(), self.key_pem.length());
    self.key_pem.free();
    self.alpn.free();
    self.client_ca_pem.free();
}

// ── backend selection ─────────────────────────────────────────────────────────

const &char tls_impl_name(int impl) {
    if (impl == TLS_IMPL_OPENSSL) { return "openssl"; }
    if (impl == TLS_IMPL_SCHANNEL) { return "schannel"; }
    if (impl == TLS_IMPL_SECURE_TRANSPORT) { return "securetransport"; }
    if (impl == TLS_IMPL_SAFEC) { return "safec"; }
    return "none";
}

static int tls_os_impl_() {
#if defined(_WIN32)
    return TLS_IMPL_SCHANNEL;
#elif defined(__APPLE__)
    return TLS_IMPL_SECURE_TRANSPORT;
#else
    return TLS_IMPL_OPENSSL;
#endif
}

int tls_backend_available(int impl) {
    if (impl == TLS_IMPL_SAFEC) { return 1; }
#if defined(_WIN32)
    if (impl == TLS_IMPL_SCHANNEL) { return tls_schannel_available_(); }
#else
    if (impl == TLS_IMPL_OPENSSL) { return tls_openssl_available_(); }
#endif
#if defined(__APPLE__)
    if (impl == TLS_IMPL_SECURE_TRANSPORT) { return tls_st_available_(); }
#endif
    return 0;
}

// The configured backend, SAFEC_TLS applied: TLS_BACKEND_OS / _SAFEC /
// a forced TLS_BACKEND_IMPL_BASE + impl, or TLS_BACKEND_AUTO.
static int tls_requested_(int backend) {
    if (backend != TLS_BACKEND_AUTO) { return backend; }
    ?&char env = tls_getenv_("SAFEC_TLS");
    match (env) {
        case some(e): {
            unsafe {
                if (str_eq(e, "os")) { return TLS_BACKEND_OS; }
                if (str_eq(e, "safec")) { return TLS_BACKEND_SAFEC; }
                if (str_eq(e, "openssl")) { return TLS_BACKEND_IMPL_BASE + TLS_IMPL_OPENSSL; }
                if (str_eq(e, "schannel")) { return TLS_BACKEND_IMPL_BASE + TLS_IMPL_SCHANNEL; }
                if (str_eq(e, "securetransport")) { return TLS_BACKEND_IMPL_BASE + TLS_IMPL_SECURE_TRANSPORT; }
            }
        }
        case null: { }
    }
    return TLS_BACKEND_AUTO;
}

// Candidates in order: first[0], then (when fallback is set) SafeC.
static int tls_choose_(int backend, &int fallback) {
    int req = tls_requested_(backend);
    *fallback = 0;
    if (req == TLS_BACKEND_SAFEC) { return TLS_IMPL_SAFEC; }
    if (req == TLS_BACKEND_OS) { return tls_os_impl_(); }
    if (req > TLS_BACKEND_IMPL_BASE) { return req - TLS_BACKEND_IMPL_BASE; }
    int os = tls_os_impl_();
    if (tls_backend_available(os)) {
        *fallback = 1;
        return os;
    }
    return TLS_IMPL_SAFEC;
}

int tls_default_impl() {
    int fb = 0;
    return tls_choose_(TLS_BACKEND_AUTO, &fb);
}

// ── TlsStream ─────────────────────────────────────────────────────────────────

static struct TlsStream tls_stream_new_(int fd, int impl, int server) {
    struct TlsStream s;
    s.fd = fd;
    s.impl = impl;
    s.state = 0;
    s.is_server = server;
    s.impl_ = null;
    s.err = TlsError::none();
    s.version_ = 0;
    s.cipher_[0] = (char)0;
    s.alpn_[0] = (char)0;
    s.peer_der_ = Vec::new();
    return s;
}

static int tls_client_start_(&TlsStream s, int impl, const &char name, const &TlsConfig cfg) {
    s.impl = impl;
    s.err = TlsError::none();
    if (!tls_backend_available(impl)) {
        struct String m = String::from(tls_impl_name(impl));
        m.push(" is not available on this system");
        tls_fail_(&s.err, TLS_ERR_UNAVAILABLE, m.as_ptr());
        m.free();
        return TLS_ERR_UNAVAILABLE;
    }
#if defined(_WIN32)
    if (impl == TLS_IMPL_SCHANNEL) { return tls_schannel_client_(s, name, cfg); }
#else
    if (impl == TLS_IMPL_OPENSSL) { return tls_openssl_client_(s, name, cfg); }
#endif
#if defined(__APPLE__)
    if (impl == TLS_IMPL_SECURE_TRANSPORT) { return tls_st_client_(s, name, cfg); }
#endif
    if (impl == TLS_IMPL_SAFEC) { return tls_safec_client_(s, name, cfg); }
    tls_fail_(&s.err, TLS_ERR_UNAVAILABLE, "unknown TLS backend");
    return TLS_ERR_UNAVAILABLE;
}

struct Result<TlsStream, TlsError> TlsStream::client(int fd, const &char host, const ?&TlsConfig cfgIn) {
    struct TlsConfig defaults = TlsConfig::new();
    struct TlsStream s = tls_stream_new_(fd, TLS_IMPL_NONE, 0);
    struct Result<TlsStream, TlsError> r;
    const &TlsConfig cfg = &defaults;
    unsafe {
        match (cfgIn) { case some(c): { cfg = c; } case null: { } }
    }
    const &char name = host;
    if (!cfg.server_name.is_empty()) { name = cfg.server_name.as_ptr(); }
    int fallback = 0;
    int impl = tls_choose_(cfg.backend, &fallback);
    int rc = tls_client_start_(&s, impl, name, cfg);
    if (rc == TLS_ERR_UNAVAILABLE && fallback) {
        rc = tls_client_start_(&s, TLS_IMPL_SAFEC, name, cfg);
    }
    defaults.free();
    if (rc != TLS_OK) {
        r = Result::new_err(s.err);
        s.peer_der_.free();
        return r;
    }
    r = Result::new_ok(s);
    return r;
}

// Runs handshake() to completion on a blocking socket.
static int tls_finish_blocking_(&TlsStream s) {
    while (1) {
        int h = s.handshake();
        if (h == 0) { return 0; }
        if (h == TLS_WANT_READ || h == TLS_WANT_WRITE) {
            tls_fail_(&s.err, TLS_ERR_TIMEOUT, "TLS handshake: the socket timed out (or is non-blocking: use client()/start() and handshake())");
            s.state = 3;
            return -1;
        }
        return -1;
    }
    return -1;
}

struct Result<TlsStream, TlsError> TlsStream::connect(int fd, const &char host, const ?&TlsConfig cfg) {
    struct Result<TlsStream, TlsError> r = TlsStream::client(fd, host, cfg);
    if (!r.ok()) { return r; }
    struct TlsStream s = r.value;
    if (tls_finish_blocking_(&s) != 0) {
        struct TlsError e = s.err;
        s.release();
        return Result::new_err(e);
    }
    return Result::new_ok(s);
}

int TlsStream::handshake() {
    if (self.state == 1) { return 0; }
    if (self.state != 0) {
        if (self.err.code == TLS_OK) { tls_fail_(&self.err, TLS_ERR_STATE, "handshake on a closed or failed stream"); }
        return -1;
    }
    int r = -1;
#if defined(_WIN32)
    if (self.impl == TLS_IMPL_SCHANNEL) { r = tls_schannel_handshake_(self); }
#else
    if (self.impl == TLS_IMPL_OPENSSL) { r = tls_openssl_handshake_(self); }
#endif
#if defined(__APPLE__)
    if (self.impl == TLS_IMPL_SECURE_TRANSPORT) { r = tls_st_handshake_(self); }
#endif
    if (self.impl == TLS_IMPL_SAFEC) { r = tls_safec_handshake_(self); }
    if (r == 0) { self.state = 1; }
    else if (r != TLS_WANT_READ && r != TLS_WANT_WRITE) {
        if (self.err.code == TLS_OK) { tls_fail_(&self.err, TLS_ERR_INTERNAL, "handshake failed"); }
        self.state = 3;
    }
    return r;
}

long long TlsStream::read([]unsigned char buf_s) {
    unsigned long n = buf_s.len;
    &unsigned char buf;
    unsafe { buf = (&unsigned char)buf_s.ptr; }
    if (self.state == 0) {
        int h = self.handshake();
        if (h != 0) { return (long long)h; }
    }
    if (self.state == 2) { return 0LL; }
    if (self.state != 1) { return -1LL; }
    if (n == 0UL) { return 0LL; }
    long long r = -1LL;
#if defined(_WIN32)
    if (self.impl == TLS_IMPL_SCHANNEL) { r = tls_schannel_read_(self, buf, n); }
#else
    if (self.impl == TLS_IMPL_OPENSSL) { r = tls_openssl_read_(self, buf, n); }
#endif
#if defined(__APPLE__)
    if (self.impl == TLS_IMPL_SECURE_TRANSPORT) { r = tls_st_read_(self, buf, n); }
#endif
    if (self.impl == TLS_IMPL_SAFEC) { r = tls_safec_read_(self, buf, n); }
    return r;
}

long long TlsStream::write([]const unsigned char buf_s) {
    unsigned long n = buf_s.len;
    const &unsigned char buf;
    unsafe { buf = (const &unsigned char)buf_s.ptr; }
    if (self.state == 0) {
        int h = self.handshake();
        if (h != 0) { return (long long)h; }
    }
    if (self.state != 1) {
        if (self.err.code == TLS_OK) { tls_fail_(&self.err, TLS_ERR_STATE, "write on a closed stream"); }
        return -1LL;
    }
    if (n == 0UL) { return 0LL; }
    long long r = -1LL;
#if defined(_WIN32)
    if (self.impl == TLS_IMPL_SCHANNEL) { r = tls_schannel_write_(self, buf, n); }
#else
    if (self.impl == TLS_IMPL_OPENSSL) { r = tls_openssl_write_(self, buf, n); }
#endif
#if defined(__APPLE__)
    if (self.impl == TLS_IMPL_SECURE_TRANSPORT) { r = tls_st_write_(self, buf, n); }
#endif
    if (self.impl == TLS_IMPL_SAFEC) { r = tls_safec_write_(self, buf, n); }
    return r;
}

int TlsStream::shutdown() {
    if (self.state != 1 && self.state != 2) { return -1; }
    int r = -1;
#if defined(_WIN32)
    if (self.impl == TLS_IMPL_SCHANNEL) { r = tls_schannel_shutdown_(self); }
#else
    if (self.impl == TLS_IMPL_OPENSSL) { r = tls_openssl_shutdown_(self); }
#endif
#if defined(__APPLE__)
    if (self.impl == TLS_IMPL_SECURE_TRANSPORT) { r = tls_st_shutdown_(self); }
#endif
    if (self.impl == TLS_IMPL_SAFEC) { r = tls_safec_shutdown_(self); }
    self.state = 2;
    return r;
}

void TlsStream::release() {
    if (!self.impl_.is_null()) {
#if defined(_WIN32)
        if (self.impl == TLS_IMPL_SCHANNEL) { tls_schannel_free_(self); }
#else
        if (self.impl == TLS_IMPL_OPENSSL) { tls_openssl_free_(self); }
#endif
#if defined(__APPLE__)
        if (self.impl == TLS_IMPL_SECURE_TRANSPORT) { tls_st_free_(self); }
#endif
        if (self.impl == TLS_IMPL_SAFEC) { tls_safec_free_(self); }
    }
    self.impl_ = null;
    self.peer_der_.free();
    if (self.state != 3) { self.state = 2; }
}

int TlsStream::close() {
    if (self.state == 1) { self.shutdown(); }
    self.release();
    int r = 0;
    if (self.fd >= 0) { r = tls_sock_close_(self.fd); }
    self.fd = -1;
    return r;
}

int TlsStream::is_open() const { return self.state == 1; }
int TlsStream::version() const { return self.version_; }
const &char TlsStream::cipher() const { return &self.cipher_[0]; }
const &char TlsStream::alpn() const { return &self.alpn_[0]; }
const &char TlsStream::backend_name() const { return tls_impl_name(self.impl); }
const &TlsError TlsStream::last_error() const { return &self.err; }

int TlsStream::peer_certificate(&X509Cert out) const {
    match (self.peer_der_.data) {
        case some(d): {
            if (self.peer_der_.len == 0UL) { return 0; }
            return out.init(d, self.peer_der_.len) == 0;
        }
        case null: { }
    }
    return 0;
}

// ── TlsAcceptor ───────────────────────────────────────────────────────────────

static int tls_acceptor_start_(&TlsAcceptor a, int impl, const &TlsServerConfig cfg, &TlsError e) {
    a.impl = impl;
    if (!tls_backend_available(impl)) {
        tls_fail_(e, TLS_ERR_UNAVAILABLE, "TLS backend not available");
        return TLS_ERR_UNAVAILABLE;
    }
#if defined(_WIN32)
    if (impl == TLS_IMPL_SCHANNEL) { return tls_schannel_acceptor_(a, cfg, e); }
#else
    if (impl == TLS_IMPL_OPENSSL) { return tls_openssl_acceptor_(a, cfg, e); }
#endif
#if defined(__APPLE__)
    if (impl == TLS_IMPL_SECURE_TRANSPORT) { return tls_st_acceptor_(a, cfg, e); }
#endif
    if (impl == TLS_IMPL_SAFEC) { return tls_safec_acceptor_(a, cfg, e); }
    tls_fail_(e, TLS_ERR_UNAVAILABLE, "unknown TLS backend");
    return TLS_ERR_UNAVAILABLE;
}

struct Result<TlsAcceptor, TlsError> TlsAcceptor::new(const &TlsServerConfig cfg) {
    struct TlsAcceptor a;
    a.impl = TLS_IMPL_NONE;
    a.impl_ = null;
    a.verify_client = cfg.verify_client;
    struct TlsError e = TlsError::none();
    if (cfg.cert_pem.is_empty() || cfg.key_pem.is_empty()) {
        tls_fail_(&e, TLS_ERR_CONFIG, "TlsServerConfig needs cert_pem and key_pem");
        return Result::new_err(e);
    }
    int fallback = 0;
    int impl = tls_choose_(cfg.backend, &fallback);
    int rc = tls_acceptor_start_(&a, impl, cfg, &e);
    if (rc == TLS_ERR_UNAVAILABLE && fallback) {
        e = TlsError::none();
        rc = tls_acceptor_start_(&a, TLS_IMPL_SAFEC, cfg, &e);
    }
    if (rc != TLS_OK) { return Result::new_err(e); }
    return Result::new_ok(a);
}

struct Result<TlsStream, TlsError> TlsAcceptor::start(int fd) {
    struct TlsStream s = tls_stream_new_(fd, self.impl, 1);
    int rc = TLS_ERR_UNAVAILABLE;
#if defined(_WIN32)
    if (self.impl == TLS_IMPL_SCHANNEL) { rc = tls_schannel_accept_(&s, self); }
#else
    if (self.impl == TLS_IMPL_OPENSSL) { rc = tls_openssl_accept_(&s, self); }
#endif
#if defined(__APPLE__)
    if (self.impl == TLS_IMPL_SECURE_TRANSPORT) { rc = tls_st_accept_(&s, self); }
#endif
    if (self.impl == TLS_IMPL_SAFEC) { rc = tls_safec_accept_(&s, self); }
    if (rc != TLS_OK) {
        if (s.err.code == TLS_OK) { tls_fail_(&s.err, rc, "TLS accept failed"); }
        struct TlsError e = s.err;
        s.release();
        return Result::new_err(e);
    }
    return Result::new_ok(s);
}

struct Result<TlsStream, TlsError> TlsAcceptor::accept(int fd) {
    struct Result<TlsStream, TlsError> r = self.start(fd);
    if (!r.ok()) { return r; }
    struct TlsStream s = r.value;
    if (tls_finish_blocking_(&s) != 0) {
        struct TlsError e = s.err;
        s.release();
        return Result::new_err(e);
    }
    return Result::new_ok(s);
}

void TlsAcceptor::free() {
    if (self.impl_.is_null()) { return; }
#if defined(_WIN32)
    if (self.impl == TLS_IMPL_SCHANNEL) { tls_schannel_acceptor_free_(self); }
#else
    if (self.impl == TLS_IMPL_OPENSSL) { tls_openssl_acceptor_free_(self); }
#endif
#if defined(__APPLE__)
    if (self.impl == TLS_IMPL_SECURE_TRANSPORT) { tls_st_acceptor_free_(self); }
#endif
    if (self.impl == TLS_IMPL_SAFEC) { tls_safec_acceptor_free_(self); }
    self.impl_ = null;
}

} // namespace std
