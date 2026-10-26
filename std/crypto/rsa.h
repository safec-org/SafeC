// SafeC Standard Library — RSA signatures (RFC 8017).
//
// Verification with PKCS#1 v1.5 (RSASSA-PKCS1-v1_5) and PSS (RSASSA-PSS,
// MGF1 with the message hash) — what X.509 certificate chains and TLS
// CertificateVerify messages use — for moduli of 1024..4096 bits (see
// bignum.h's BN_LIMBS_MAX), and signing with a private key for TLS servers.
// Signing exponentiates in constant time (bignum.h's bn_mod_exp_secret,
// no CRT) and checks the signature with the public key before returning it.
//
//   struct RsaPublicKey k;
//   if (k.init_der(&spkiKey[0], keyLen) &&
//       std::rsa_verify_pss(&k, DIGEST_SHA256, &hash[0], 32UL, &sig[0], sigLen)) { ... }
//
// Hash ids are digest.h's DIGEST_*. Freestanding-safe.
#pragma once
#include <std/crypto/buf.h>
#include <std/crypto/bignum.h>
#include <std/crypto/digest.h>
#include <std/crypto/rng.h>

#define RSA_MAX_BYTES  512   // 4096-bit moduli
#define RSA_MIN_BITS   1024

namespace std {

struct RsaPublicKey {
    struct BnMont n;           // the modulus
    unsigned char e[8];        // public exponent, big-endian
    unsigned long e_len;
    unsigned long bits;        // modulus length in bits

    // From a DER RSAPublicKey (SEQUENCE { modulus, publicExponent }: the
    // BIT STRING content of an rsaEncryption SubjectPublicKeyInfo). 0 if
    // malformed, under RSA_MIN_BITS or over RSA_MAX_BYTES, or e is even,
    // 1 or wider than 64 bits.
    int init_der([]const unsigned char der);
    // internal: the (reference, length) core of init_der()
    int init_der_n_(const &unsigned char der, unsigned long len);
};

// RSASSA-PKCS1-v1_5 verification of 'hash' (the digest_size(alg)-byte
// digest of the message) against sig[0..sig_len). 1 if valid.
int rsa_verify_pkcs1(const &RsaPublicKey key, int alg, []const unsigned char hash, []const unsigned char sig);
// internal: the (reference, length) core of rsa_verify_pkcs1()
int rsa_verify_pkcs1_n_(const &RsaPublicKey key, int alg, const &unsigned char hash, unsigned long hash_len,
                     const &unsigned char sig, unsigned long sig_len);

// RSASSA-PSS verification, MGF1 with 'alg', salt length 'salt_len' (TLS
// and most certificates: the hash length). 1 if valid.
int rsa_verify_pss(const &RsaPublicKey key, int alg, []const unsigned char hash, []const unsigned char sig, unsigned long salt_len);
// internal: the (reference, length) core of rsa_verify_pss()
int rsa_verify_pss_n_(const &RsaPublicKey key, int alg, const &unsigned char hash, unsigned long hash_len,
                   const &unsigned char sig, unsigned long sig_len, unsigned long salt_len);

struct RsaPrivateKey {
    struct RsaPublicKey pub;
    unsigned char d[RSA_MAX_BYTES];   // private exponent, big-endian, pub.n.bytes long
    unsigned long d_len;

    // From a DER RSAPrivateKey (PKCS#1) or a PKCS#8 PrivateKeyInfo holding
    // one. 0 if malformed or unsupported.
    int  init_der([]const unsigned char der);
    // internal: the (reference, length) core of init_der()
    int  init_der_n_(const &unsigned char der, unsigned long len);
    // Wipes the private exponent.
    void destroy();
};

// Signs 'hash' (digest_size(alg) bytes) into sig (key.pub.n.bytes bytes;
// *sig_len set). PSS uses a random salt of the hash's length from 'rng'.
// 1 on success.
int rsa_sign_pss(const &RsaPrivateKey key, int alg, []const unsigned char hash, &RngCtx rng, []unsigned char sig, &unsigned long sig_len);
// internal: the (reference, length) core of rsa_sign_pss()
int rsa_sign_pss_n_(const &RsaPrivateKey key, int alg, const &unsigned char hash, unsigned long hash_len,
                 &RngCtx rng, &unsigned char sig, unsigned long sig_cap, &unsigned long sig_len);
int rsa_sign_pkcs1(const &RsaPrivateKey key, int alg, []const unsigned char hash, []unsigned char sig, &unsigned long sig_len);
// internal: the (reference, length) core of rsa_sign_pkcs1()
int rsa_sign_pkcs1_n_(const &RsaPrivateKey key, int alg, const &unsigned char hash, unsigned long hash_len,
                   &unsigned char sig, unsigned long sig_cap, &unsigned long sig_len);

} // namespace std
