// SafeC Standard Library — ChaCha20, Poly1305 and the ChaCha20-Poly1305
// AEAD (RFC 8439).
//
// ChaCha20 is the 20-round stream cipher with a 32-byte key, a 12-byte
// nonce and a 32-bit block counter (RFC 8439 §2.4); Poly1305 the one-time
// authenticator (§2.5); the AEAD composes them with the Poly1305 key taken
// from block 0 of the keystream (§2.8). All three run in constant time:
// no table lookups and no branches on key, nonce, message or tag bytes,
// and the tag comparison of chacha20_poly1305_open looks at every byte.
// It is the record protection of TLS_CHACHA20_POLY1305_SHA256 (std/tls).
//
//   unsigned char tag[16];
//   std::chacha20_poly1305_seal(key, nonce, aad, aadLen, msg, msgLen, tag);  // msg encrypted in place
//   if (!std::chacha20_poly1305_open(key, nonce, aad, aadLen, msg, msgLen, tag)) { /* forged */ }
//
// Never reuse a nonce with the same key. Freestanding-safe.
#pragma once

#define CHACHA20_KEY_SIZE    32
#define CHACHA20_NONCE_SIZE  12
#define POLY1305_KEY_SIZE    32
#define POLY1305_TAG_SIZE    16

namespace std {

// XORs the ChaCha20 keystream for (key, nonce), starting at block
// 'counter', into data[0..len) — encryption and decryption are the same.
void chacha20_xor(const &unsigned char key, const &unsigned char nonce, unsigned int counter,
                  &unsigned char data, unsigned long len);

// Writes the 64-byte ChaCha20 block 'counter' for (key, nonce).
void chacha20_block(const &unsigned char key, const &unsigned char nonce, unsigned int counter,
                    &unsigned char out);

// Streaming Poly1305 with a one-time 32-byte key (r || s).
struct Poly1305 {
    unsigned int  r[5];     // clamped r, 26-bit limbs
    unsigned int  h[5];     // accumulator
    unsigned int  pad[4];   // s
    unsigned char buf[16];  // a partial block
    unsigned long buf_len;

    static struct Poly1305 new(const &unsigned char key);
    void update(const &unsigned char data, unsigned long len);
    // Pads the input so far to a multiple of 16 bytes with zeros (the
    // AEAD's pad16 between the additional data and the ciphertext).
    void pad16();
    // Writes the 16-byte tag and wipes the state.
    void finish(&unsigned char tag);
};

// One-shot Poly1305: the 16-byte tag of msg[0..len) under 'key'.
void poly1305(const &unsigned char key, const ?&unsigned char msg, unsigned long len, &unsigned char tag);

// AEAD_CHACHA20_POLY1305 (RFC 8439 §2.8): encrypts data[0..len) in place
// and writes the 16-byte tag over aad[0..aad_len) (aad may be null when
// aad_len is 0) and the ciphertext.
void chacha20_poly1305_seal(const &unsigned char key, const &unsigned char nonce,
                            const ?&unsigned char aad, unsigned long aad_len,
                            &unsigned char data, unsigned long len, &unsigned char tag);

// Checks the tag in constant time and, only if it is authentic, decrypts
// data[0..len) in place. 1 if authentic, 0 if not (data is then left as
// the unauthenticated ciphertext).
int  chacha20_poly1305_open(const &unsigned char key, const &unsigned char nonce,
                            const ?&unsigned char aad, unsigned long aad_len,
                            &unsigned char data, unsigned long len, const &unsigned char tag);

} // namespace std
