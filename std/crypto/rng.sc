// SafeC Standard Library — Cryptographic RNG (ChaCha20-based)
#pragma once
#include "rng.h"

extern void* memcpy(void* d, const void* s, unsigned long n);

// ── ChaCha20 core ─────────────────────────────────────────────────────────────

static unsigned int chacha_rotr_(unsigned int v, int n) {
    return (v << n) | (v >> (32 - n));
}

static void chacha_qr_(unsigned int* a, unsigned int* b,
                        unsigned int* c, unsigned int* d) {
    unsafe {
        *a = *a + *b; *d = chacha_rotr_(*d ^ *a, 16);
        *c = *c + *d; *b = chacha_rotr_(*b ^ *c, 12);
        *a = *a + *b; *d = chacha_rotr_(*d ^ *a,  8);
        *c = *c + *d; *b = chacha_rotr_(*b ^ *c,  7);
    }
}

// Produce 64 bytes of keystream into `out` from state `s`.
static void chacha20_block_(const unsigned int* s, unsigned char* out) {
    unsigned int x[16];
    int i = 0;
    while (i < 16) { x[i] = s[i]; i = i + 1; }

    // 10 double-rounds
    int r = 0;
    while (r < 10) {
        chacha_qr_(&x[ 0],&x[ 4],&x[ 8],&x[12]);
        chacha_qr_(&x[ 1],&x[ 5],&x[ 9],&x[13]);
        chacha_qr_(&x[ 2],&x[ 6],&x[10],&x[14]);
        chacha_qr_(&x[ 3],&x[ 7],&x[11],&x[15]);
        chacha_qr_(&x[ 0],&x[ 5],&x[10],&x[15]);
        chacha_qr_(&x[ 1],&x[ 6],&x[11],&x[12]);
        chacha_qr_(&x[ 2],&x[ 7],&x[ 8],&x[13]);
        chacha_qr_(&x[ 3],&x[ 4],&x[ 9],&x[14]);
        r = r + 1;
    }

    i = 0;
    while (i < 16) { x[i] = x[i] + s[i]; i = i + 1; }

    // Serialise little-endian
    unsafe {
        i = 0;
        while (i < 16) {
            out[i*4+0] = (unsigned char)( x[i]       & 0xFF);
            out[i*4+1] = (unsigned char)((x[i] >>  8) & 0xFF);
            out[i*4+2] = (unsigned char)((x[i] >> 16) & 0xFF);
            out[i*4+3] = (unsigned char)((x[i] >> 24) & 0xFF);
            i = i + 1;
        }
    }
}

// ── Initialisation ─────────────────────────────────────────────────────────────

// ChaCha20 "expa nd 32-byte k" constant
static unsigned int chacha_const_[4] = {
    0x61707865, 0x3320646e, 0x79622d32, 0x6b206574
};

static void rng_setup_(unsigned int* state, const unsigned char* key32,
                        unsigned int ctr, unsigned int nonce) {
    int i = 0;
    while (i < 4) { state[i] = chacha_const_[i]; i = i + 1; }
    unsafe {
        i = 0;
        while (i < 8) {
            state[4+i] = ((unsigned int)key32[i*4+0])
                       | ((unsigned int)key32[i*4+1] << 8)
                       | ((unsigned int)key32[i*4+2] << 16)
                       | ((unsigned int)key32[i*4+3] << 24);
            i = i + 1;
        }
    }
    state[12] = ctr;
    state[13] = (unsigned int)0;
    state[14] = nonce;
    state[15] = (unsigned int)0;
}

void rng_init_seed(&stack RngCtx ctx, const unsigned char seed[32]) {
    rng_setup_(ctx.state, seed, (unsigned int)0, (unsigned int)1);
    chacha20_block_(ctx.state, (unsigned char*)ctx.buf);
    ctx.state[12] = ctx.state[12] + (unsigned int)1;
    ctx.buf_pos = (unsigned long)0;
}

int rng_init(&stack RngCtx ctx) {
#ifdef __x86_64__
    // Try rdrand for seeding
    unsigned int seed_words[8];
    int ok = 1;
    unsafe {
        int i = 0;
        while (i < 8) {
            unsigned int val = (unsigned int)0;
            int success = 0;
            int tries = 0;
            while (tries < 10 && success == 0) {
                asm volatile (
                    "xorl %0, %0\n\t"
                    "rdrand %1\n\t"
                    "setc %b0"
                    : "=r"(success), "=r"(val) : : "cc"
                );
                tries = tries + 1;
            }
            if (success == 0) { ok = 0; }
            seed_words[i] = val;
            i = i + 1;
        }
    }
    if (ok != 0) {
        rng_setup_(ctx.state, (const unsigned char*)seed_words, (unsigned int)0, (unsigned int)1);
        chacha20_block_(ctx.state, (unsigned char*)ctx.buf);
        ctx.state[12] = ctx.state[12] + (unsigned int)1;
        ctx.buf_pos = (unsigned long)0;
        return 1;
    }
#endif

#ifndef __SAFEC_FREESTANDING__
    // Hosted: seed from /dev/urandom
    extern void* fopen(const char* path, const char* mode);
    extern unsigned long fread(void* buf, unsigned long size, unsigned long n, void* f);
    extern int fclose(void* f);
    unsafe {
        void* f = fopen("/dev/urandom", "rb");
        if (f != (void*)0) {
            unsigned char seed[32];
            fread((void*)seed, (unsigned long)1, (unsigned long)32, f);
            fclose(f);
            rng_init_seed(ctx, seed);
            return 1;
        }
    }
#endif

    // Fallback: deterministic seed (not cryptographically safe)
    unsigned char fallback[32];
    int k = 0;
    while (k < 32) {
        unsafe { fallback[k] = (unsigned char)(k * 37 + 1); }
        k = k + 1;
    }
    rng_init_seed(ctx, (const unsigned char*)fallback);
    return 0;
}

// ── RngCtx methods ─────────────────────────────────────────────────────────────

void RngCtx::fill(unsigned char* out, unsigned long len) {
    unsigned long i = (unsigned long)0;
    while (i < len) {
        if (self.buf_pos >= (unsigned long)RNG_BLOCK_SIZE) {
            chacha20_block_(self.state, (unsigned char*)self.buf);
            self.state[12] = self.state[12] + (unsigned int)1;
            self.buf_pos = (unsigned long)0;
        }
        unsafe { out[i] = self.buf[self.buf_pos]; }
        self.buf_pos = self.buf_pos + (unsigned long)1;
        i = i + (unsigned long)1;
    }
}

unsigned int RngCtx::rand32() {
    unsigned char b[4];
    self.fill((unsigned char*)b, (unsigned long)4);
    unsafe {
        return ((unsigned int)b[0])
             | ((unsigned int)b[1] << 8)
             | ((unsigned int)b[2] << 16)
             | ((unsigned int)b[3] << 24);
    }
}

unsigned long RngCtx::rand64() {
    unsigned long lo = (unsigned long)self.rand32();
    unsigned long hi = (unsigned long)self.rand32();
    return (hi << (unsigned long)32) | lo;
}

unsigned long RngCtx::rand_range(unsigned long bound) {
    if (bound <= (unsigned long)1) { return (unsigned long)0; }
    // Rejection sampling to eliminate bias
    unsigned long threshold = (~bound + (unsigned long)1) % bound;
    unsafe {
        while (1) {
            unsigned long r = self.rand64();
            if (r >= threshold) { return r % bound; }
        }
    }
    return (unsigned long)0;
}
