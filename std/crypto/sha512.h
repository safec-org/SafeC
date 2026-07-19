// SafeC Standard Library — SHA-512 / SHA-384 Hash (FIPS 180-4)
// Freestanding-safe. No dynamic allocation.
#pragma once

#define SHA512_DIGEST_SIZE  64   // bytes
#define SHA384_DIGEST_SIZE  48   // bytes
#define SHA512_BLOCK_SIZE   128  // bytes

namespace std {

struct Sha512Ctx {
    unsigned long long h[8];               // current hash state
    unsigned char       buf[SHA512_BLOCK_SIZE]; // partial block buffer
    // Total bytes processed. SHA-512's length field is 128 bits, but a
    // single 64-bit counter is tracked here (matching every other hash
    // module in this library) since no realistic input reaches 2^64 bytes
    // — the length field's high 8 bytes are simply written as zero.
    unsigned long long  total_bytes;
    int                 is384;             // 1 = SHA-384 variant

    // Feed `len` bytes of data into the hash.
    void update(const &stack unsigned char data, unsigned long len);

    // Finalise and write the digest.
    // For SHA-512: digest must be at least 64 bytes.
    // For SHA-384: digest must be at least 48 bytes.
    void finish(&stack unsigned char digest);
};

// Initialise a SHA-512 context.
struct Sha512Ctx sha512_init();

// Initialise a SHA-384 context.
struct Sha512Ctx sha384_init();

// Convenience: hash `len` bytes and write the 64-byte digest.
void sha512(const &stack unsigned char data, unsigned long len,
            &stack unsigned char digest);

// Convenience: hash `len` bytes and write the 48-byte digest.
void sha384(const &stack unsigned char data, unsigned long len,
            &stack unsigned char digest);

} // namespace std
