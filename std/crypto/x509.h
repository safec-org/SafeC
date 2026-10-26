// SafeC Standard Library — X.509 Certificate Parsing (DER format, read-only)
// Parses the minimum fields needed for TLS: subject, issuer, validity, public key, SAN.
// Freestanding-safe; no dynamic allocation.
#pragma once
#include <std/crypto/buf.h>

#define X509_MAX_NAME_LEN   128
#define X509_MAX_SAN        8
#define X509_MAX_SAN_LEN    64
#define X509_KEY_LEN        256   // max RSA/EC public key bytes

// Validity period (Unix timestamps, 64-bit).
namespace std {

struct X509Validity {
    unsigned long long not_before;   // seconds since epoch (dates before 1970 clamp to 0)
    unsigned long long not_after;
};

// Distinguished Name (simplified: CN + O + C as strings).
struct X509Name {
    char cn[X509_MAX_NAME_LEN];   // Common Name
    char o[X509_MAX_NAME_LEN];    // Organization
    char c[4];                    // Country (2 chars + null)
};

// Subject Alternative Name (DNS names only).
struct X509San {
    char names[X509_MAX_SAN][X509_MAX_SAN_LEN];
    int  count;
};

// Parsed certificate (all fields, no dynamic allocation).
struct X509Cert {
    struct X509Name    subject;
    struct X509Name    issuer;
    struct X509Validity validity;
    struct X509San     san;
    unsigned char      pubkey[X509_KEY_LEN];
    unsigned long      pubkey_len;
    unsigned char      serial[20];
    unsigned long      serial_len;
    int                is_ca;          // 1 if BasicConstraints CA=TRUE
    int                key_usage;      // raw KeyUsage bits

    // Is the certificate currently valid for the given Unix timestamp?
    int  is_valid_at(unsigned long long ts) const;

    // Does `hostname` match a DNS Subject Alternative Name (or, only when the
    // certificate has none, the subject CN)? Case-insensitive; '*.' wildcards
    // match one leftmost label.
    int  matches_hostname(const &char hostname) const;

    // Is this a CA certificate?
    int  is_ca_cert() const;

    // Parse a DER-encoded certificate from `der` (len bytes) into this
    // struct in place (zeroed first).  Returns 0 on success, negative on
    // error.
    int  init([]const unsigned char der_ref);
    // internal: the (reference, length) core of init()
    int  init_n_(const &unsigned char der, unsigned long len);

    // Does this certificate's issuer name match `issuer_cert`'s subject?
    // A structural check only: X509Cert keeps no signature. Verify
    // signatures with X509Parsed and x509_verify_signed_by() below (std/tls
    // builds and verifies whole chains with them).
    int  verify_chain(const &X509Cert issuer_cert) const;
};

// ── Full parsing and signature verification ───────────────────────────────────
//
// X509Parsed locates every field certificate-path validation needs inside
// a DER certificate the caller keeps (spans are byte ranges of it), with
// no fixed limits on names or SAN entries. x509_verify_signed_by checks a
// certificate's signature with its issuer's key (RSA PKCS#1 v1.5 and PSS,
// ECDSA P-256/P-384 — std/crypto/rsa.h and ec.h); x509_match_host is RFC
// 6125 / RFC 9525 host matching. Path building and trust anchors are
// std/tls's (tls.h's TlsTrustStore).

#define X509_KEY_UNKNOWN  0
#define X509_KEY_RSA      1
#define X509_KEY_EC_P256  2
#define X509_KEY_EC_P384  3
#define X509_KEY_ED25519  4   // recognized, not verifiable here

#define X509_SIG_UNKNOWN     0
#define X509_SIG_RSA_PKCS1   1
#define X509_SIG_RSA_PSS     2
#define X509_SIG_ECDSA       3
#define X509_SIG_ED25519     4

// KeyUsage bits (as X509Parsed::key_usage holds them).
#define X509_KU_DIGITAL_SIGNATURE 0x80
#define X509_KU_KEY_ENCIPHERMENT  0x20
#define X509_KU_KEY_AGREEMENT     0x08
#define X509_KU_KEY_CERT_SIGN     0x04

// Error codes of x509_parse (negative).
#define X509_ERR_MALFORMED   -1
#define X509_ERR_UNSUPPORTED -2

struct X509Span {
    unsigned long off;   // byte offset into the certificate's DER
    unsigned long len;   // 0 when absent
};

struct X509Parsed {
    unsigned long      der_len;
    struct X509Span    tbs;          // the signed TBSCertificate (with its header)
    struct X509Span    serial;       // INTEGER content
    struct X509Span    issuer;       // the Name, whole TLV (compare bytes)
    struct X509Span    subject;
    struct X509Span    spki;         // SubjectPublicKeyInfo, whole TLV
    struct X509Span    key;          // RSA: RSAPublicKey DER; EC: the SEC1 point
    struct X509Span    sig;          // signature bits (BIT STRING content after the unused-bits byte)
    struct X509Span    san;          // SubjectAltName GeneralNames content (absent: len 0)
    struct X509Span    subject_cn;   // first commonName value of the subject
    int                version;      // 1..3
    int                key_type;     // X509_KEY_*
    int                sig_type;     // X509_SIG_*
    int                sig_hash;     // DIGEST_* (std/crypto/digest.h)
    int                pss_salt;     // RSA-PSS salt length
    unsigned long long not_before;   // Unix seconds
    unsigned long long not_after;
    int                has_basic_constraints;
    int                is_ca;
    int                path_len;     // -1 when unlimited / absent
    int                has_key_usage;
    int                key_usage;    // X509_KU_* bits of the first byte (| second byte << 8)
    int                has_eku;
    int                eku_server;   // id-kp-serverAuth present
    int                eku_client;   // id-kp-clientAuth present
    int                eku_any;      // anyExtendedKeyUsage present
    int                has_name_constraints;
    int                unknown_critical;  // a critical extension this parser doesn't process
};

// Parses a DER certificate. 0 on success, X509_ERR_* otherwise.
int x509_parse([]const unsigned char der, &X509Parsed out);
// internal: the (reference, length) core of x509_parse()
int x509_parse_n_(const &unsigned char der, unsigned long len, &X509Parsed out);

// 1 if the signature on cert (der/c) verifies with issuer's public key
// (issuer_der/issuer); 0 otherwise (bad signature, unsupported algorithm,
// SHA-1, or an RSA key under 1024 bits).
int x509_verify_signed_by([]const unsigned char der, const &X509Parsed c, []const unsigned char issuer_der, const &X509Parsed issuer);
// internal: the (reference, length) core of x509_verify_signed_by()
int x509_verify_signed_by_n_(const &unsigned char der, const &X509Parsed c,
                          const &unsigned char issuer_der, const &X509Parsed issuer);

// 1 if cert's issuer name is byte-for-byte issuer's subject name.
int x509_issued_by([]const unsigned char der, const &X509Parsed c, []const unsigned char issuer_der, const &X509Parsed issuer);
// internal: the (reference, length) core of x509_issued_by()
int x509_issued_by_n_(const &unsigned char der, const &X509Parsed c,
                   const &unsigned char issuer_der, const &X509Parsed issuer);

// 1 if 'host' (a DNS name, or an IPv4/IPv6 literal) matches the
// certificate: DNS names against dNSName SANs (case-insensitive, a '*'
// leftmost label matching exactly one label, never in IP literals or a
// public-suffix-like two-label pattern), IP literals against iPAddress
// SANs; the subject CN is consulted only when the certificate has no DNS
// or IP SAN at all.
int x509_match_host([]const unsigned char der, const &X509Parsed c, const &char host);
// internal: the (reference, length) core of x509_match_host()
int x509_match_host_n_(const &unsigned char der, const &X509Parsed c, const &char host);

} // namespace std
