// SafeC Standard Library — Cryptographic RNG
// Hardware-backed (rdrand on x86-64, or /dev/urandom on hosted).
// Falls back to a ChaCha20-based CSPRNG seeded from a timer or user seed.
// Freestanding-safe when hardware entropy source is available.
#pragma once

#define RNG_BLOCK_SIZE 64   // ChaCha20 output block (bytes)

struct RngCtx {
    unsigned int  state[16];  // ChaCha20 key + nonce + counter
    unsigned char buf[RNG_BLOCK_SIZE];
    unsigned long buf_pos;    // next byte to consume from buf

    // Fill `out[0..len-1]` with cryptographically strong random bytes.
    void fill(unsigned char* out, unsigned long len);

    // Return a random 32-bit unsigned integer.
    unsigned int  rand32();

    // Return a random 64-bit unsigned integer.
    unsigned long rand64();

    // Return a uniform random integer in [0, bound).
    unsigned long rand_range(unsigned long bound);
};

// Initialise with automatic seeding (hardware entropy or /dev/urandom).
// Returns 1 on success, 0 if no entropy source available.
int rng_init(&stack RngCtx ctx);

// Initialise with a user-provided 32-byte seed (deterministic; useful for tests).
void rng_init_seed(&stack RngCtx ctx, const unsigned char seed[32]);
