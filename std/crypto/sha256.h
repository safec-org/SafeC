// SafeC Standard Library — SHA-256 / SHA-224 Hash
// Freestanding-safe.  No dynamic allocation.
#pragma once

#define SHA256_DIGEST_SIZE  32   // bytes
#define SHA224_DIGEST_SIZE  28   // bytes
#define SHA256_BLOCK_SIZE   64   // bytes

struct Sha256Ctx {
    unsigned int  h[8];          // current hash state
    unsigned char buf[SHA256_BLOCK_SIZE]; // partial block buffer
    unsigned long total_bytes;   // total bytes processed
    int           is224;         // 1 = SHA-224 variant

    // Feed `len` bytes of data into the hash.
    void update(const &stack unsigned char data, unsigned long len);

    // Finalise and write the digest.
    // For SHA-256: digest must be at least 32 bytes.
    // For SHA-224: digest must be at least 28 bytes.
    void finish(&stack unsigned char digest);
};

// Initialise a SHA-256 context.
struct Sha256Ctx sha256_init();

// Initialise a SHA-224 context.
struct Sha256Ctx sha224_init();

// Convenience: hash `len` bytes and write 32-byte digest.
void sha256(const &stack unsigned char data, unsigned long len,
            &stack unsigned char digest);
