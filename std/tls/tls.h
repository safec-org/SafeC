// SafeC Standard Library — TLS: encrypted, authenticated streams over a
// connected socket (client and server), on the operating system's TLS
// implementation when there is one and SafeC's own otherwise.
//
//   struct TlsConfig cfg = std::TlsConfig::new();          // verify against the system roots
//   cfg.set_alpn("http/1.1");
//   struct Result<TlsStream, TlsError> r = std::TlsStream::connect(fd, "example.com", &cfg);
//   match (r.get_err()) { case some(e): { printf("%s\n", e.message); } case null: {} }
//   struct TlsStream t = r.value;
//   t.write(...); t.read(...);                             // io_traits.h Stream: BufReader<TlsStream> works
//   t.close();                                             // close_notify, then closes fd
//   cfg.free();
//
// Backends (the 'impl' of a stream; tls_impl_name() names them):
//   TLS_IMPL_OPENSSL         Linux/BSD: libssl/libcrypto 3.x or 1.1, loaded
//                            with dlopen() at run time — nothing to link,
//                            no headers; the system CA store through
//                            OpenSSL's default paths plus the usual bundles.
//   TLS_IMPL_SCHANNEL        Windows: SChannel through SSPI; peers verified
//                            against the Windows certificate store.
//   TLS_IMPL_SECURE_TRANSPORT macOS: Security.framework (SecureTransport,
//                            TLS 1.2 at most; verified with SecTrust against
//                            the system keychains).
//   TLS_IMPL_SAFEC           everywhere: std's own TLS 1.3 (std/crypto's
//                            X25519/P-256/P-384, AES-GCM, ChaCha20-Poly1305,
//                            RSA/ECDSA), verifying against the system's PEM
//                            bundle on disk (or the Windows store).
// Selection (TlsConfig::backend / TlsServerConfig::backend): TLS_BACKEND_AUTO
// takes the SAFEC_TLS environment variable ("os", "safec", "openssl",
// "schannel", "securetransport") when it is set, else the OS backend when it
// is available (and can meet min_version: SecureTransport can't do 1.3),
// else SafeC's. An OS backend that is present but fails is an error, never
// a silent fallback. TLS_BACKEND_OS / _SAFEC / a TLS_IMPL_* force one.
//
// Blocking and non-blocking use: connect()/TlsAcceptor::accept() run the
// whole handshake on a blocking socket. On a non-blocking socket, start with
// TlsStream::client()/TlsAcceptor::start() and call handshake() until it
// returns 0; it, read() and write() return TLS_WANT_READ / TLS_WANT_WRITE
// when the socket would block (wait for that readiness — std/async's
// IoReady, std::Reactor — and call again).
//
// Security notes: peers are verified by default (chain to a trusted root,
// validity dates, key usage, the host name); turn verification off only for
// tests. The SafeC backend is young code: its primitives are checked against
// RFC/NIST test vectors and its handshake against OpenSSL, but it hasn't had
// an external audit; AES there is table-based (ChaCha20-Poly1305, constant
// time, is preferred). See stdlib/tls.md.
#pragma once
#include <std/collections/string.h>
#include <std/collections/vec.h>
#include <std/crypto/x509.h>
#include <std/result.h>

#define TLS_VERSION_1_2 0x0303
#define TLS_VERSION_1_3 0x0304

#define TLS_BACKEND_AUTO   0
#define TLS_BACKEND_OS     1
#define TLS_BACKEND_SAFEC  2

#define TLS_IMPL_NONE              0
#define TLS_IMPL_OPENSSL           1
#define TLS_IMPL_SCHANNEL          2
#define TLS_IMPL_SECURE_TRANSPORT  3
#define TLS_IMPL_SAFEC             4
// TlsConfig::backend also accepts a TLS_IMPL_* (+ this offset) to force one.
#define TLS_BACKEND_IMPL_BASE      16

// read / write / handshake results besides counts and 0.
#define TLS_WANT_READ   -2
#define TLS_WANT_WRITE  -3

// TlsError codes.
#define TLS_OK                    0
#define TLS_ERR_UNAVAILABLE       1   // the backend isn't on this system (or not built for this target)
#define TLS_ERR_CONFIG            2   // bad configuration: unreadable PEM, unknown setting
#define TLS_ERR_IO                3   // the socket failed
#define TLS_ERR_CLOSED            4   // the peer closed the connection mid-handshake
#define TLS_ERR_TIMEOUT           5   // a blocking socket's timeout expired
#define TLS_ERR_PROTOCOL          6   // malformed or unexpected message
#define TLS_ERR_ALERT             7   // the peer sent a fatal alert (TlsError::alert)
#define TLS_ERR_HANDSHAKE         8   // no common version / cipher suite / group / signature scheme
#define TLS_ERR_CERT_UNTRUSTED    9   // no chain to a trusted root (self-signed, unknown CA)
#define TLS_ERR_CERT_HOSTNAME    10   // the certificate isn't for this host
#define TLS_ERR_CERT_EXPIRED     11   // outside its validity period
#define TLS_ERR_CERT_INVALID     12   // bad signature, malformed, wrong key usage, ...
#define TLS_ERR_CERT_REVOKED     13   // revoked (OS backends that check)
#define TLS_ERR_NO_CERT          14   // the peer sent no certificate where one is required
#define TLS_ERR_NO_MEMORY        15
#define TLS_ERR_STATE            16   // wrong call for the stream's state
#define TLS_ERR_INTERNAL         17

#define TLS_ERROR_MESSAGE_MAX 224

namespace std {

// A backend's private state (opaque; each backend keeps its own struct).
struct TlsImpl_ { unsigned char opaque_; };

struct TlsError {
    int       code;      // TLS_OK or TLS_ERR_*
    int       alert;     // the TLS alert (RFC 8446 §6) received or sent, -1 if none
    long long detail;    // the backend's own code (OpenSSL error, SECURITY_STATUS, OSStatus), 0 if none
    char      message[TLS_ERROR_MESSAGE_MAX];

    static struct TlsError none();
    static struct TlsError make(int code, const &char message);
    int ok() const;
};

// The fixed English name of a TLS_ERR_* code.
const &char tls_error_name(int code);

// Client settings. Owns its strings: free() when done (a stream doesn't
// keep a reference to its config).
struct TlsConfig {
    int           verify_peer;       // 1: check the chain and dates (default)
    int           verify_hostname;   // 1: check the name (default)
    int           use_system_roots;  // 1: trust the OS's roots (default)
    struct String ca_pem;            // more trusted roots, PEM (empty: none)
    struct String ca_file;           // a PEM bundle file of more trusted roots
    struct String server_name;       // SNI + the name to verify; empty: connect()'s host
    struct String alpn;              // protocols to offer, comma-separated ("h2,http/1.1")
    int           min_version;       // TLS_VERSION_1_2 (default) or _1_3
    int           max_version;       // TLS_VERSION_1_3 (default)
    int           backend;           // TLS_BACKEND_AUTO (default)
    struct String cert_pem;          // client certificate chain (PEM), if the server asks
    struct String key_pem;           // its private key (PEM: PKCS#8, RSA or EC)

    static struct TlsConfig new();
    void set_ca_pem(const &char pem);
    void set_ca_file(const &char path);
    void set_server_name(const &char name);
    void set_alpn(const &char protocols);
    void set_client_cert(const &char cert_pem, const &char key_pem);
    // Verification off (tests against self-signed servers only).
    void insecure_skip_verify();
    struct TlsConfig clone() const;
    void free();
};

// Server settings: the certificate chain and key are required.
struct TlsServerConfig {
    struct String cert_pem;          // leaf first, then intermediates (PEM)
    struct String key_pem;           // private key (PEM: PKCS#8, RSA or EC)
    struct String alpn;              // protocols the server speaks, in preference order
    int           min_version;       // TLS_VERSION_1_2 (default)
    int           max_version;       // TLS_VERSION_1_3 (default)
    int           backend;           // TLS_BACKEND_AUTO (default)
    int           verify_client;     // 0 none (default), 1 request a certificate, 2 require one
    struct String client_ca_pem;     // roots for client certificates (verify_client != 0)

    static struct TlsServerConfig new();
    // Reads cert_pem / key_pem from files. 0 if either can't be read.
    int  load_files(const &char cert_path, const &char key_path);
    void set_alpn(const &char protocols);
    void free();
};

// A TLS connection over a socket fd it owns once created.
struct TlsStream {
    int           fd;
    int           impl;           // TLS_IMPL_*
    int           state;          // internal: 0 handshaking, 1 open, 2 closed, 3 failed
    int           is_server;
    ?&heap TlsImpl_ impl_;        // internal: the backend's state
    struct TlsError err;          // the last failure
    // Negotiated, once the handshake is done:
    int           version_;       // TLS_VERSION_*
    char          cipher_[64];    // IANA suite name, e.g. "TLS_AES_128_GCM_SHA256"
    char          alpn_[64];      // the protocol agreed, "" if none
    struct Vec<unsigned char> peer_der_;   // the peer's leaf certificate (DER), empty if none

    // Client: handshake on a blocking, connected socket. 'host' names the
    // server (SNI and verification) unless cfg.server_name is set; null cfg
    // = TlsConfig::new()'s defaults. On failure the fd is left open (the
    // caller's to close).
    static struct Result<TlsStream, TlsError> connect(int fd, const &char host, const ?&TlsConfig cfg);
    // Client, non-blocking: set up only; drive with handshake().
    static struct Result<TlsStream, TlsError> client(int fd, const &char host, const ?&TlsConfig cfg);

    // Advances the handshake: 0 when complete, TLS_WANT_READ/WRITE, -1 on
    // failure (see last_error()).
    int handshake();
    // Up to n bytes of application data: the count, 0 once the peer closed
    // (close_notify or EOF), -1 on error, TLS_WANT_READ/WRITE.
    long long read(&unsigned char buf, unsigned long n);
    // Up to n bytes (at most one record's worth may be accepted at a
    // time): the count, -1 on error, TLS_WANT_WRITE/READ.
    long long write(const &unsigned char buf, unsigned long n);
    // Sends close_notify (once). 0 on success.
    int shutdown();
    // shutdown() if open, frees the TLS state and closes the socket.
    int close();
    // Frees the TLS state but leaves the fd open (after a failed handshake,
    // or to hand the socket on).
    void release();

    int is_open() const;
    int version() const;            // TLS_VERSION_*, 0 before the handshake
    const &char cipher() const;     // "" before the handshake
    const &char alpn() const;
    const &char backend_name() const;
    // The peer's leaf certificate summary (subject, issuer, validity,
    // SANs); 0 if there is none.
    int peer_certificate(&X509Cert out) const;
    const &TlsError last_error() const;
};

// A server context: the parsed certificate and key, reused for every
// connection.
struct TlsAcceptor {
    int    impl;     // TLS_IMPL_*
    ?&heap TlsImpl_ impl_;   // internal
    int    verify_client;

    static struct Result<TlsAcceptor, TlsError> new(const &TlsServerConfig cfg);
    // Handshake on a blocking accepted socket; the fd stays open on failure.
    struct Result<TlsStream, TlsError> accept(int fd);
    // Non-blocking: set up only; drive with handshake().
    struct Result<TlsStream, TlsError> start(int fd);
    void free();
};

// Is the backend usable here (its library loads, the target has it)?
int tls_backend_available(int impl);
// What TLS_BACKEND_AUTO picks for a client (SAFEC_TLS included).
int tls_default_impl();
// "openssl", "schannel", "securetransport", "safec", "none".
const &char tls_impl_name(int impl);

} // namespace std
