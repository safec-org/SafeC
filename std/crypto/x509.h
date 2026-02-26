// SafeC Standard Library — X.509 Certificate Parsing (DER format, read-only)
// Parses the minimum fields needed for TLS: subject, issuer, validity, public key, SAN.
// Freestanding-safe; no dynamic allocation.
#pragma once

#define X509_MAX_NAME_LEN   128
#define X509_MAX_SAN        8
#define X509_MAX_SAN_LEN    64
#define X509_KEY_LEN        256   // max RSA/EC public key bytes

// Validity period (Unix timestamps, 64-bit).
struct X509Validity {
    unsigned long long not_before;   // seconds since epoch
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

    // Does a Subject Alternative Name match `hostname`?
    int  matches_hostname(const char* hostname) const;

    // Is this a CA certificate?
    int  is_ca_cert() const;
};

// Parse a DER-encoded certificate from `der` (len bytes).
// Fills `cert_out` on success.  Returns 0 on success, negative on error.
int x509_parse_der(const unsigned char* der, unsigned long len,
                   &stack X509Cert cert_out);

// Verify that `cert` was signed by `issuer_cert` (structural chain check only).
// Returns 1 if issuer name matches, 0 otherwise.
// NOTE: Full crypto verification requires std/crypto/rsa.h (not yet implemented);
//       this function checks only the structural chain (issuer name match).
int x509_verify_chain(const &stack X509Cert cert,
                       const &stack X509Cert issuer_cert);
