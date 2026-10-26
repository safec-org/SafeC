// SafeC Standard Library — X25519 Diffie-Hellman (RFC 7748).
//
// Curve25519's Montgomery-ladder scalar multiplication on 32-byte
// little-endian u-coordinates, in constant time: a fixed 255-step ladder
// with branch-free conditional swaps, and field arithmetic (mod 2^255 - 19,
// sixteen 16-bit limbs) with no data-dependent branches or table lookups.
// The key exchange TLS 1.3's x25519 group uses (std/tls).
//
//   unsigned char priv[32], pub[32], shared[32];
//   rng.fill(&priv[0], 32UL);
//   std::x25519_public(pub[..], priv[..]);           // 32-byte slices
//   if (!std::x25519(shared[..], priv[..], peer_pub[..])) { /* low-order peer key */ }
//
// Freestanding-safe.
#pragma once
#include <std/crypto/buf.h>

#define X25519_KEY_SIZE 32

namespace std {

// out = scalar * point (both 32 bytes; the scalar is clamped as RFC 7748
// §5 specifies). 0 when the result is all zeros — the peer sent a
// low-order point and the shared secret must be rejected (RFC 8446
// §7.4.2) — 1 otherwise.
int  x25519([]unsigned char out, []const unsigned char scalar, []const unsigned char point);
// internal: the (reference, length) core of x25519()
int  x25519_n_(&unsigned char out, const &unsigned char scalar, const &unsigned char point);

// out = scalar * 9 (the base point): the public key for 'scalar'.
void x25519_public([]unsigned char out, []const unsigned char scalar);
// internal: the (reference, length) core of x25519_public()
void x25519_public_n_(&unsigned char out, const &unsigned char scalar);

} // namespace std
