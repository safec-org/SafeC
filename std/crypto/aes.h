// SafeC Standard Library — AES Block Cipher
// AES-128 and AES-256 in ECB, CBC and GCM modes.  Pure software; no SIMD.
// Freestanding-safe.
#pragma once

#define AES_BLOCK_SIZE  16   // bytes
#define AES128_KEY_SIZE 16   // bytes
#define AES256_KEY_SIZE 32   // bytes
#define AES128_ROUNDS   10
#define AES256_ROUNDS   14

// Expanded key schedule: max 15 round keys × 4 words × 4 bytes = 240 bytes.
#define AES_KS_WORDS   60

namespace std {

struct AesCtx {
    unsigned int  ks[AES_KS_WORDS]; // expanded key schedule
    int           rounds;            // 10 (AES-128) or 14 (AES-256)
    unsigned char iv[AES_BLOCK_SIZE]; // CBC initialisation vector

    // An AES-128 context from a 16-byte key (IV zeroed).
    static struct AesCtx new_128(const &unsigned char key);

    // An AES-256 context from a 32-byte key (IV zeroed).
    static struct AesCtx new_256(const &unsigned char key);

    // Encrypt one 16-byte block in-place (ECB).
    void encrypt_block(&unsigned char block);

    // Decrypt one 16-byte block in-place (ECB).
    void decrypt_block(&unsigned char block);

    // Encrypt `len` bytes (must be multiple of 16) in CBC mode.
    // Uses and updates self.iv.
    void cbc_encrypt(&unsigned char data, unsigned long len);

    // Decrypt `len` bytes in CBC mode.  Uses and updates self.iv.
    void cbc_decrypt(&unsigned char data, unsigned long len);

    // Set the CBC initialisation vector.
    void set_iv(const &unsigned char iv);

    // AES-GCM (NIST SP 800-38D) with a 12-byte IV and a 16-byte tag:
    // encrypts data[0..len) in place and writes the tag over the
    // additional data aad[0..aad_len) (aad may be null when aad_len is 0)
    // and the ciphertext. Never reuse an IV with the same key.
    void gcm_encrypt(const &unsigned char iv, const ?&unsigned char aad, unsigned long aad_len,
                     &unsigned char data, unsigned long len, &unsigned char tag);

    // Checks the tag (in constant time) and, only if it matches, decrypts
    // data[0..len) in place. Returns 1 if authentic, 0 if not (data is then
    // left as the unauthenticated ciphertext).
    int  gcm_decrypt(const &unsigned char iv, const ?&unsigned char aad, unsigned long aad_len,
                     &unsigned char data, unsigned long len, const &unsigned char tag);

    // Securely wipe the expanded key schedule and IV (via
    // mem_zero_secure — a volatile write-through that survives
    // dead-store elimination, unlike a plain assignment the optimizer
    // could see is never read again and drop). Call this once a context
    // is done being used, before it goes out of scope, so the key
    // material doesn't linger readable in memory (e.g. in a stack frame
    // reused by a later, unrelated function, or in a crash dump).
    void drop();
    void destroy() consume;   // = drop()
};

} // namespace std
