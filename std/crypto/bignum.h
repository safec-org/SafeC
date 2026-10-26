// SafeC Standard Library — fixed-width modular arithmetic (Montgomery form).
//
// The multi-precision core std's public-key code shares: RSA (rsa.h) and
// the NIST curves (ec.h). Numbers are little-endian arrays of 32-bit limbs
// of a fixed width n (the modulus' width), so a number's length — and the
// work done on it — never depends on its value. Montgomery multiplication
// (CIOS) ends with a branch-free conditional subtraction, and the modular
// add/sub/select helpers are branch-free too; only bn_cmp_ and the public-
// exponent bn_mod_exp_public look at values, and they are for public data.
//
// BnMont is a modulus with its Montgomery constants, for moduli of up to
// BN_LIMBS_MAX * 32 bits (RSA-4096 and a little headroom):
//
//   struct BnMont m;
//   if (!m.init(&modulus[0], modLen)) { /* even, zero or too big */ }
//   bn_mod_exp_public(&m, &sig[0], sigLen, &e[0], eLen, &out[0]);   // out: modLen bytes
//
// The '_'-suffixed raw-limb functions are std-internal (unsafe pointers).
// Odd moduli only (every RSA modulus and curve prime/order is odd).
// Freestanding-safe.
#pragma once

#define BN_LIMBS_MAX 130

namespace std {

struct BnMont {
    unsigned int  m[BN_LIMBS_MAX];    // the modulus
    unsigned int  rr[BN_LIMBS_MAX];   // R^2 mod m, R = 2^(32 n)
    unsigned int  n0inv;              // -m^-1 mod 2^32
    int           n;                  // limbs in use
    unsigned long bytes;              // the modulus' length in bytes (no leading zeros)

    // From a big-endian modulus (leading zero bytes allowed). 0 if it is
    // even, zero or wider than BN_LIMBS_MAX limbs.
    int init(const &unsigned char mod_be, unsigned long len);
};

// out (m.bytes bytes, big-endian) = base^exp mod m, for a public exponent
// (its bits are looked at: not for private keys). 'base' (big-endian,
// base_len <= m.bytes) must be < m; 0 if it isn't, 1 otherwise.
int bn_mod_exp_public(const &BnMont m, const &unsigned char base, unsigned long base_len,
                      const &unsigned char exp, unsigned long exp_len, &unsigned char out);

// out = base^exp mod m in constant time for a secret exponent (fixed
// 4-bit windows over every exponent bit, table entries read by masked
// scans). exp_len bytes of exponent are processed regardless of value.
int bn_mod_exp_secret(const &BnMont m, const &unsigned char base, unsigned long base_len,
                      const &unsigned char exp, unsigned long exp_len, &unsigned char out);

// ── raw limbs (std-internal) ─────────────────────────────────────────────────
// -m0^-1 mod 2^32 for odd m0.
unsigned int bn_n0inv_(unsigned int m0);
// r = a * b * R^-1 mod m (a, b < m; r may alias a or b).
void bn_mont_mul_(unsigned int* r, const unsigned int* a, const unsigned int* b,
                  const unsigned int* m, unsigned int n0inv, int n);
// R^2 mod m into rr.
void bn_rr_(unsigned int* rr, const unsigned int* m, int n);
// Big-endian bytes <-> limbs. from_be: 0 if the value needs more than n
// limbs. to_be writes exactly 'len' bytes (high limbs truncated/zero-padded).
int  bn_from_be_(unsigned int* r, int n, const unsigned char* be, unsigned long len);
void bn_to_be_(unsigned char* out, unsigned long len, const unsigned int* a, int n);
// r = a + b / a - b over n limbs; returns the carry / borrow (0 or 1).
unsigned int bn_add_(unsigned int* r, const unsigned int* a, const unsigned int* b, int n);
unsigned int bn_sub_(unsigned int* r, const unsigned int* a, const unsigned int* b, int n);
// r = a + b mod m / a - b mod m (a, b < m), branch-free.
void bn_mod_add_(unsigned int* r, const unsigned int* a, const unsigned int* b, const unsigned int* m, int n);
void bn_mod_sub_(unsigned int* r, const unsigned int* a, const unsigned int* b, const unsigned int* m, int n);
// If a >= m, a -= m (once), branch-free; for values below 2m.
void bn_reduce_once_(unsigned int* a, const unsigned int* m, int n);
// r = bit ? a : r (bit 0 or 1), branch-free.
void bn_select_(unsigned int* r, const unsigned int* a, unsigned int bit, int n);
// -1 / 0 / 1 (not constant time: public values only).
int  bn_cmp_(const unsigned int* a, const unsigned int* b, int n);
// 1 if a == 0, branch-free.
unsigned int bn_is_zero_(const unsigned int* a, int n);
void bn_zero_(unsigned int* a, int n);
void bn_copy_(unsigned int* r, const unsigned int* a, int n);

} // namespace std
