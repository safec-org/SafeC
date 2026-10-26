// SafeC Standard Library — SHA-1 Hash (FIPS 180-4)
// Freestanding-safe. No dynamic allocation.
//
// SHA-1 is cryptographically broken (practical collision attacks exist
// since 2017) — don't use it for anything security-sensitive. It's
// provided because some still-current wire protocols require it
// specifically, not as a general-purpose hash recommendation: notably the
// WebSocket handshake (RFC 6455 section 1.3's Sec-WebSocket-Accept
// computation is defined in terms of SHA-1, with no alternative). Prefer
// std::sha256/std::sha512 (see sha256.h/sha512.h) for anything else.
#pragma once
#include <std/crypto/buf.h>
// Sha1Ctx conforms to hash.h's Hasher and BlockHash (generic hashing, Hmac).

#define SHA1_DIGEST_SIZE  20   // bytes
#define SHA1_BLOCK_SIZE   64   // bytes

namespace std {

struct Sha1Ctx {
    unsigned int  h[5];               // current hash state
    unsigned char buf[SHA1_BLOCK_SIZE]; // partial block buffer
    unsigned long total_bytes;         // total bytes processed

    // A fresh SHA-1 context.
    static struct Sha1Ctx new();

    // Feed `len` bytes of data into the hash.
    void update([]const unsigned char data);
    // internal: the (reference, length) core of update()
    void update_n_(const &unsigned char data, unsigned long len);

    // Finalise and write the 20-byte digest.
    void finish([]unsigned char digest);
    // internal: the (reference, length) core of finish()
    void finish_n_(&unsigned char digest);

    // SHA1_DIGEST_SIZE / SHA1_BLOCK_SIZE (hash.h's BlockHash).
    unsigned long digest_size() const;
    unsigned long block_size() const;
};

// Convenience: hash `len` bytes and write the 20-byte digest.
void sha1([]const unsigned char data, []unsigned char digest);
// internal: the (reference, length) core of sha1()
void sha1_n_(const &unsigned char data, unsigned long len,
          &unsigned char digest);

} // namespace std
