// SafeC Standard Library — TLS: SafeC's own TLS 1.3 implementation
// (std-internal: the engine behind TLS_IMPL_SAFEC, and the certificate
// path validation it shares with nothing else).
//
//   tls_trust.sc        trust stores, PEM roots, path building + validation
//   tls_roots_posix.sc  the system roots on Unix: the PEM bundle on disk
//   tls_roots_win32.sc  the system roots on Windows: the ROOT store (crypt32)
//   tls_safec.sc        records, key schedule, client and server handshakes
#pragma once
#include <std/tls/tls.h>
#include <std/collections/vec.h>
#include <std/crypto/x509.h>
#include <std/crypto/aes.h>
#include <std/crypto/rsa.h>
#include <std/crypto/ec.h>

namespace std {

// ── trust (tls_trust.sc) ─────────────────────────────────────────────────────

struct TlsCert_ {
    struct Vec<unsigned char> der;
    struct X509Parsed         p;
    void drop();
};

struct TlsTrust_ {
    struct Vec<struct TlsCert_> certs;
    void drop();
};

struct TlsTrust_ tls_trust_new_();
// Empties t (drops its certificates); t stays usable.
void tls_trust_free_(&TlsTrust_ t);
// Adds one DER certificate (parsed; malformed ones are skipped): 1 if added.
int  tls_trust_add_der_(&TlsTrust_ t, const &unsigned char der, unsigned long len);
// Adds every CERTIFICATE block of a PEM text; returns how many.
unsigned long tls_trust_add_pem_(&TlsTrust_ t, const &char pem);
// The system's roots, loaded once per process (tls_roots_*.sc), or null if
// none could be found.
?&TlsTrust_ tls_trust_system_();
// Appends the OS's root certificates (DER) to out (tls_roots_*.sc).
void tls_system_roots_(&TlsTrust_ out);

// Validates a peer chain (leaf first, as sent) against the anchors in
// 'a' and 'b' (either may be null): signatures, validity at 'now', CA
// constraints and key usage, the leaf's extended key usage for 'server'
// (serverAuth) or client (clientAuth), and — when host is non-null — the
// leaf's name. TLS_OK or a TLS_ERR_CERT_* code with e set.
int tls_verify_chain_(const &Vec<struct Vec<unsigned char>> chain, const ?&TlsTrust_ a, const ?&TlsTrust_ b,
                      const ?&char host, int server, unsigned long long now, &TlsError e);
// Seconds since the epoch.
unsigned long long tls_now_();

// HKDF-Expand-Label (RFC 8446 §7.1) with hash DIGEST_SHA256 / _SHA384:
// the key schedule's one primitive, exposed for the RFC 8448 checks.
void tls13_expand_label_(int hash, const &unsigned char secret, unsigned long sl, const &char label,
                         const ?&unsigned char ctx, unsigned long cl, &unsigned char out, unsigned long ol);

// ── private keys (tls_safec.sc) ──────────────────────────────────────────────

#define TLS_KEY_NONE_ 0
#define TLS_KEY_EC_   1
#define TLS_KEY_RSA_  2

struct TlsKey_ {
    int                  type;
    struct EcPrivateKey  ec;
    struct RsaPrivateKey rsa;
};

// From a PEM private key. TLS_OK or TLS_ERR_CONFIG with e set.
int  tls_key_from_pem_(&TlsKey_ k, const &char pem, &TlsError e);
void tls_key_destroy_(&TlsKey_ k);

} // namespace std
