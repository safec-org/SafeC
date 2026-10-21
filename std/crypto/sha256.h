// SafeC Standard Library — SHA-256 / SHA-224 Hash
// Freestanding-safe.  No dynamic allocation.
#pragma once

#define SHA256_DIGEST_SIZE  32   // bytes
#define SHA224_DIGEST_SIZE  28   // bytes
#define SHA256_BLOCK_SIZE   64   // bytes

namespace std {

struct Sha256Ctx {
    unsigned int  h[8];          // current hash state
    unsigned char buf[SHA256_BLOCK_SIZE]; // partial block buffer
    unsigned long total_bytes;   // total bytes processed
    int           is224;         // 1 = SHA-224 variant

    // A fresh SHA-256 context.
    static struct Sha256Ctx new();

    // A fresh SHA-224 context (same struct, is224 = 1).
    static struct Sha256Ctx new_224();

    // Feed `len` bytes of data into the hash.
    void update(const &unsigned char data, unsigned long len);

    // Finalise and write the digest.
    // For SHA-256: digest must be at least 32 bytes.
    // For SHA-224: digest must be at least 28 bytes.
    void finish(&unsigned char digest);
};

// Convenience: hash `len` bytes and write 32-byte digest.
void sha256(const &unsigned char data, unsigned long len,
            &unsigned char digest);

} // namespace std
