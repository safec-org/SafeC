// SafeC Standard Library — AES Block Cipher
// AES-128 and AES-256 in ECB and CBC modes.  Pure software; no SIMD.
// Freestanding-safe.
#pragma once

#define AES_BLOCK_SIZE  16   // bytes
#define AES128_KEY_SIZE 16   // bytes
#define AES256_KEY_SIZE 32   // bytes
#define AES128_ROUNDS   10
#define AES256_ROUNDS   14

// Expanded key schedule: max 15 round keys × 4 words × 4 bytes = 240 bytes.
#define AES_KS_WORDS   60

struct AesCtx {
    unsigned int  ks[AES_KS_WORDS]; // expanded key schedule
    int           rounds;            // 10 (AES-128) or 14 (AES-256)
    unsigned char iv[AES_BLOCK_SIZE]; // CBC initialisation vector

    // Encrypt one 16-byte block in-place (ECB).
    void encrypt_block(&stack unsigned char block);

    // Decrypt one 16-byte block in-place (ECB).
    void decrypt_block(&stack unsigned char block);

    // Encrypt `len` bytes (must be multiple of 16) in CBC mode.
    // Uses and updates self.iv.
    void cbc_encrypt(&stack unsigned char data, unsigned long len);

    // Decrypt `len` bytes in CBC mode.  Uses and updates self.iv.
    void cbc_decrypt(&stack unsigned char data, unsigned long len);

    // Set the CBC initialisation vector.
    void set_iv(const &stack unsigned char iv);
};

// Initialise AES-128 context from a 16-byte key.
struct AesCtx aes128_init(const &stack unsigned char key);

// Initialise AES-256 context from a 32-byte key.
struct AesCtx aes256_init(const &stack unsigned char key);
