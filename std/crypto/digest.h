// SafeC Standard Library — hash algorithms chosen at run time.
//
// Signature schemes name their hash in data (an X.509 AlgorithmIdentifier,
// a TLS SignatureScheme), so the verifiers in rsa.h / ec.h / x509.h take a
// DIGEST_* id rather than a generic hash type. digest() hashes with the
// algorithm an id names; DigestCtx streams. For code that knows its hash
// at compile time, hash.h's Hasher generics remain the direct route.
#pragma once
#include <std/crypto/buf.h>
#include <std/crypto/sha1.h>
#include <std/crypto/sha256.h>
#include <std/crypto/sha512.h>

#define DIGEST_NONE   0
#define DIGEST_SHA1   1
#define DIGEST_SHA224 2
#define DIGEST_SHA256 3
#define DIGEST_SHA384 4
#define DIGEST_SHA512 5
#define DIGEST_MAX_SIZE 64

namespace std {

// The digest length of 'alg' in bytes (0 for an unknown id).
unsigned long digest_size(int alg);

// Hash data[0..len) with 'alg' into 'out' (digest_size(alg) bytes).
// Returns the digest length, 0 for an unknown id.
unsigned long digest(int alg, []const unsigned char data, []unsigned char out);
// internal: the (reference, length) core of digest()
unsigned long digest_n_(int alg, const ?&unsigned char data, unsigned long len, &unsigned char out);

// A streaming hash of any DIGEST_* algorithm.
struct DigestCtx {
    int              alg;
    struct Sha1Ctx   s1;
    struct Sha256Ctx s256;
    struct Sha512Ctx s512;

    // 'alg' must be a known id (else alg is DIGEST_NONE and finish writes 0 bytes).
    static struct DigestCtx new(int alg);
    void update([]const unsigned char data);
    // internal: the (reference, length) core of update()
    void update_n_(const &unsigned char data, unsigned long len);
    // Writes the digest; returns its length.
    unsigned long finish([]unsigned char out);
    // internal: the (reference, length) core of finish()
    unsigned long finish_n_(&unsigned char out);
};

// MGF1 (RFC 8017 §B.2.1) with 'alg': XORs the mask generated from
// seed[0..seed_len) into data[0..len).
void mgf1_xor(int alg, []const unsigned char seed, []unsigned char data);
// internal: the (reference, length) core of mgf1_xor()
void mgf1_xor_n_(int alg, const &unsigned char seed, unsigned long seed_len,
              &unsigned char data, unsigned long len);

} // namespace std
