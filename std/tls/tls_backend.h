// SafeC Standard Library — TLS: the interface between tls.sc and its
// backends (std-internal).
//
// Each backend X (openssl: tls_openssl_posix.sc, schannel:
// tls_schannel_win32.sc, st: tls_securetransport_macos.sc, safec:
// tls_safec*.sc) implements the functions below; tls.sc picks one per
// stream and dispatches. A backend fills TlsStream's version_/cipher_/
// alpn_/peer_der_ when its handshake completes, and reports failures in
// TlsStream::err (or the TlsError passed in) with the TLS_ERR_* code it
// returns. Return conventions follow tls.h (0 / TLS_WANT_* / -1).
#pragma once
#include <std/tls/tls.h>

namespace std {

// ── shared helpers (tls.sc) ──────────────────────────────────────────────────

// Sets e (code + message) and returns -1.
int  tls_fail_(&TlsError e, int code, const &char message);
// "<prefix>: <detail>".
int  tls_fail2_(&TlsError e, int code, const &char prefix, const &char detail);
// One recv()/send() on a socket: the count, 0 at EOF (recv), -1 on error,
// TLS_WANT_READ / TLS_WANT_WRITE when it would block (or a blocking
// socket's SO_RCVTIMEO/SO_SNDTIMEO expired). EINTR is retried.
long long tls_sock_recv_(int fd, &unsigned char buf, unsigned long n);
long long tls_sock_send_(int fd, const &unsigned char buf, unsigned long n);
// 1 if name is an IPv4 or IPv6 literal (no SNI for those).
int  tls_name_is_ip_(const &char name);
// getenv(); null when unset.
?&char tls_getenv_(const &char name);
// The whole file into out (appended). 0 if it can't be read.
int  tls_read_file_(const &char path, &String out);
// Every PEM block labelled 'label' ("CERTIFICATE"; null: any) in text,
// base64-decoded, appended to out. Returns how many.
unsigned long tls_pem_decode_(const &char text, const ?&char label, &Vec<struct Vec<unsigned char>> out);
// The first private-key PEM block (PRIVATE KEY, RSA PRIVATE KEY, EC
// PRIVATE KEY): its DER into der, the label's kind into *kind (1 PKCS#8,
// 2 RSA, 3 EC). 0 if none.
int  tls_pem_private_key_(const &char text, &Vec<unsigned char> der, &int kind);
void tls_der_list_free_(&Vec<struct Vec<unsigned char>> list);
// Copies a C string into a fixed buffer (always NUL-terminated).
void tls_copy_cstr_(&char dst, unsigned long cap, const ?&char src);
// IANA name of a cipher suite id ("TLS_AES_128_GCM_SHA256"), "" if unknown.
const &char tls_suite_name_(int id);
// The ALPN wire list (length-prefixed names) for a comma-separated list.
struct Vec<unsigned char> tls_alpn_wire_(const &char list);

// ── per-backend entry points ─────────────────────────────────────────────────

int       tls_openssl_available_();
int       tls_openssl_client_(&TlsStream s, const &char name, const &TlsConfig cfg);
int       tls_openssl_acceptor_(&TlsAcceptor a, const &TlsServerConfig cfg, &TlsError err);
int       tls_openssl_accept_(&TlsStream s, const &TlsAcceptor a);
int       tls_openssl_handshake_(&TlsStream s);
long long tls_openssl_read_(&TlsStream s, &unsigned char buf, unsigned long n);
long long tls_openssl_write_(&TlsStream s, const &unsigned char buf, unsigned long n);
int       tls_openssl_shutdown_(&TlsStream s);
void      tls_openssl_free_(&TlsStream s);
void      tls_openssl_acceptor_free_(&TlsAcceptor a);

int       tls_schannel_available_();
int       tls_schannel_client_(&TlsStream s, const &char name, const &TlsConfig cfg);
int       tls_schannel_acceptor_(&TlsAcceptor a, const &TlsServerConfig cfg, &TlsError err);
int       tls_schannel_accept_(&TlsStream s, const &TlsAcceptor a);
int       tls_schannel_handshake_(&TlsStream s);
long long tls_schannel_read_(&TlsStream s, &unsigned char buf, unsigned long n);
long long tls_schannel_write_(&TlsStream s, const &unsigned char buf, unsigned long n);
int       tls_schannel_shutdown_(&TlsStream s);
void      tls_schannel_free_(&TlsStream s);
void      tls_schannel_acceptor_free_(&TlsAcceptor a);

int       tls_st_available_();
int       tls_st_client_(&TlsStream s, const &char name, const &TlsConfig cfg);
int       tls_st_acceptor_(&TlsAcceptor a, const &TlsServerConfig cfg, &TlsError err);
int       tls_st_accept_(&TlsStream s, const &TlsAcceptor a);
int       tls_st_handshake_(&TlsStream s);
long long tls_st_read_(&TlsStream s, &unsigned char buf, unsigned long n);
long long tls_st_write_(&TlsStream s, const &unsigned char buf, unsigned long n);
int       tls_st_shutdown_(&TlsStream s);
void      tls_st_free_(&TlsStream s);
void      tls_st_acceptor_free_(&TlsAcceptor a);

int       tls_safec_client_(&TlsStream s, const &char name, const &TlsConfig cfg);
int       tls_safec_acceptor_(&TlsAcceptor a, const &TlsServerConfig cfg, &TlsError err);
int       tls_safec_accept_(&TlsStream s, const &TlsAcceptor a);
int       tls_safec_handshake_(&TlsStream s);
long long tls_safec_read_(&TlsStream s, &unsigned char buf, unsigned long n);
long long tls_safec_write_(&TlsStream s, const &unsigned char buf, unsigned long n);
int       tls_safec_shutdown_(&TlsStream s);
void      tls_safec_free_(&TlsStream s);
void      tls_safec_acceptor_free_(&TlsAcceptor a);

} // namespace std
