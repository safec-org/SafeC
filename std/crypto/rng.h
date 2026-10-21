// SafeC Standard Library — Cryptographic RNG
// A ChaCha20 keystream generator, seeded from rdrand (x86-64) or
// /dev/urandom (hosted), or from a caller-supplied 32-byte seed.
// Freestanding-safe with RngCtx::init_seed() or rdrand.
// Usage: struct RngCtx rng; rng.init(); rng.fill(buf, n);
#pragma once

#define RNG_BLOCK_SIZE 64   // ChaCha20 output block (bytes)

namespace std {

struct RngCtx {
    unsigned int  state[16];  // ChaCha20 key + nonce + counter
    unsigned char buf[RNG_BLOCK_SIZE];
    unsigned long buf_pos;    // next byte to consume from buf

    // Initialise in place with automatic seeding (rdrand on x86-64, else
    // /dev/urandom). Returns 1; aborts with a diagnostic if no entropy
    // source is available (use init_seed() with your own entropy source on
    // such targets).
    int init();

    // Initialise in place with a user-provided 32-byte seed (deterministic;
    // useful for tests).
    void init_seed(const unsigned char seed[32]);

    // Fill `out[0..len-1]` with cryptographically strong random bytes.
    void fill(&unsigned char out, unsigned long len);

    // Return a random 32-bit unsigned integer.
    unsigned int  rand32();

    // Return a random 64-bit unsigned integer.
    unsigned long long rand64();

    // Return a uniform random integer in [0, bound).
    unsigned long rand_range(unsigned long bound);
};

} // namespace std
