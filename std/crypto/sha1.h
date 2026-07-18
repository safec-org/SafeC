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

#define SHA1_DIGEST_SIZE  20   // bytes
#define SHA1_BLOCK_SIZE   64   // bytes

namespace std {

struct Sha1Ctx {
    unsigned int  h[5];               // current hash state
    unsigned char buf[SHA1_BLOCK_SIZE]; // partial block buffer
    unsigned long total_bytes;         // total bytes processed

    // Feed `len` bytes of data into the hash.
    void update(const &stack unsigned char data, unsigned long len);

    // Finalise and write the 20-byte digest.
    void finish(&stack unsigned char digest);
};

// Initialise a SHA-1 context.
struct Sha1Ctx sha1_init();

// Convenience: hash `len` bytes and write the 20-byte digest.
void sha1(const &stack unsigned char data, unsigned long len,
          &stack unsigned char digest);

} // namespace std
