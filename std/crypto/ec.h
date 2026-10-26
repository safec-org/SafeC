// SafeC Standard Library — elliptic curves P-256 and P-384 (FIPS 186-4):
// ECDH key agreement and ECDSA signatures.
//
// Points use projective coordinates with the complete addition formulas
// of Renes, Costello and Batina (2016, a = -3), so no input — the point at
// infinity, doubling, an attacker's point — takes a different code path.
// Scalar multiplication is a fixed-length double-and-add-always ladder
// with branch-free selection, and field/scalar arithmetic is bignum.h's
// branch-free Montgomery code: key generation, ECDH and signing run in
// constant time. Peer public keys are checked to lie on the curve.
// ECDSA signing derives its nonce deterministically (RFC 6979).
//
// Curves are named by their TLS NamedGroup ids. Keys and coordinates are
// big-endian; public keys are uncompressed SEC1 points (0x04 || X || Y).
//
//   unsigned char priv[EC_MAX_SCALAR], pub[EC_MAX_POINT], shared[EC_MAX_SCALAR];
//   std::ec_keygen(EC_P256, &rng, &priv[0], &pub[0]);         // pub: 65 bytes
//   if (!std::ecdh_shared(EC_P256, &priv[0], &peer[0], 65UL, &shared[0])) { /* bad point */ }
//   int ok = std::ecdsa_verify_der(EC_P256, &pub[0], 65UL, &hash[0], 32UL, &sig[0], sigLen);
//
// Freestanding-safe.
#pragma once
#include <std/crypto/rng.h>

#define EC_P256 23            // secp256r1 / prime256v1
#define EC_P384 24            // secp384r1
#define EC_MAX_SCALAR 48      // bytes of a P-384 scalar or coordinate
#define EC_MAX_POINT  97      // an uncompressed P-384 point
#define EC_MAX_SIG    104     // a DER ECDSA P-384 signature

namespace std {

// Bytes per coordinate/scalar: 32 (P-256), 48 (P-384), 0 for another id.
unsigned long ec_scalar_size(int curve);
// Bytes of an uncompressed point: 65 / 97 (0 for another id).
unsigned long ec_point_size(int curve);

// A random private key (in [1, n-1]) and its public point. 1 on success.
int ec_keygen(int curve, &RngCtx rng, &unsigned char priv, &unsigned char pub);
// pub = priv * G. 0 if priv is 0 or >= n.
int ec_public_key(int curve, const &unsigned char priv, &unsigned char pub);
// 1 if pub[0..len) is an uncompressed point on the curve (not infinity).
int ec_point_valid(int curve, const &unsigned char pub, unsigned long len);

// ECDH: out (ec_scalar_size bytes) = the x coordinate of priv * peer.
// 0 if the peer point is invalid or the result is the point at infinity.
int ecdh_shared(int curve, const &unsigned char priv, const &unsigned char peer, unsigned long peer_len,
                &unsigned char out);

// ECDSA verification of 'hash' (truncated to the curve's size as FIPS
// 186-4 specifies) against (r, s) — each ec_scalar_size bytes — or a DER
// Ecdsa-Sig-Value. 1 if valid.
int ecdsa_verify(int curve, const &unsigned char pub, unsigned long pub_len,
                 const &unsigned char hash, unsigned long hash_len,
                 const &unsigned char r, const &unsigned char s);
int ecdsa_verify_der(int curve, const &unsigned char pub, unsigned long pub_len,
                     const &unsigned char hash, unsigned long hash_len,
                     const &unsigned char sig, unsigned long sig_len);

// ECDSA signature of 'hash' as DER (at most EC_MAX_SIG bytes) with an
// RFC 6979 nonce (HMAC-SHA-256 for P-256, HMAC-SHA-384 for P-384). 1 on
// success.
int ecdsa_sign_der(int curve, const &unsigned char priv, const &unsigned char hash, unsigned long hash_len,
                   &unsigned char sig, unsigned long sig_cap, &unsigned long sig_len);

// A private key with its curve and public point, from DER: SEC1
// ECPrivateKey (RFC 5915) or PKCS#8 PrivateKeyInfo (RFC 5208) holding one.
struct EcPrivateKey {
    int           curve;
    unsigned char d[EC_MAX_SCALAR];
    unsigned char pub[EC_MAX_POINT];
    unsigned long pub_len;

    int  init_der(const &unsigned char der, unsigned long len);
    void destroy();
};

} // namespace std
