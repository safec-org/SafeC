// SafeC Standard Library — SHA-256
#pragma once
#include "sha256.h"

extern void* memcpy(void* d, const void* s, unsigned long n);
extern void* memset(void* p, int v, unsigned long n);

// ── SHA-256 constants ─────────────────────────────────────────────────────────

static unsigned int sha256_K_[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
    0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
    0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
    0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
    0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
    0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static unsigned int sha256_H0_[8] = {
    0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
};

static unsigned int sha224_H0_[8] = {
    0xc1059ed8,0x367cd507,0x3070dd17,0xf70e5939,
    0xffc00b31,0x68581511,0x64f98fa7,0xbefa4fa4
};

// ── Bit-rotation helpers ──────────────────────────────────────────────────────

static unsigned int rotr32_(unsigned int x, int n) {
    return (x >> n) | (x << (32 - n));
}

// ── Process one 64-byte block ─────────────────────────────────────────────────

static void sha256_compress_(unsigned int* h, const unsigned char* block) {
    unsigned int W[64];
    int t = 0;
    unsafe {
        while (t < 16) {
            W[t] = ((unsigned int)block[t*4+0] << 24)
                 | ((unsigned int)block[t*4+1] << 16)
                 | ((unsigned int)block[t*4+2] <<  8)
                 | ((unsigned int)block[t*4+3]);
            t = t + 1;
        }
        while (t < 64) {
            unsigned int s0 = rotr32_(W[t-15],7)^rotr32_(W[t-15],18)^(W[t-15]>>3);
            unsigned int s1 = rotr32_(W[t- 2],17)^rotr32_(W[t- 2],19)^(W[t- 2]>>10);
            W[t] = W[t-16] + s0 + W[t-7] + s1;
            t = t + 1;
        }
    }

    unsigned int a=h[0],b=h[1],c=h[2],d=h[3];
    unsigned int e=h[4],f=h[5],g=h[6],hh=h[7];

    t = 0;
    while (t < 64) {
        unsigned int S1  = rotr32_(e,6)^rotr32_(e,11)^rotr32_(e,25);
        unsigned int ch  = (e & f) ^ (~e & g);
        unsigned int T1  = hh + S1 + ch + sha256_K_[t] + W[t];
        unsigned int S0  = rotr32_(a,2)^rotr32_(a,13)^rotr32_(a,22);
        unsigned int maj = (a & b) ^ (a & c) ^ (b & c);
        unsigned int T2  = S0 + maj;
        hh=g; g=f; f=e; e=d+T1;
        d=c; c=b; b=a; a=T1+T2;
        t = t + 1;
    }

    h[0]=h[0]+a; h[1]=h[1]+b; h[2]=h[2]+c; h[3]=h[3]+d;
    h[4]=h[4]+e; h[5]=h[5]+f; h[6]=h[6]+g; h[7]=h[7]+hh;
}

// ── Public API ────────────────────────────────────────────────────────────────

struct Sha256Ctx sha256_init() {
    struct Sha256Ctx ctx;
    int i = 0;
    while (i < 8) { ctx.h[i] = sha256_H0_[i]; i = i + 1; }
    ctx.total_bytes = (unsigned long)0;
    ctx.is224       = 0;
    unsafe { memset((void*)ctx.buf, 0, (unsigned long)SHA256_BLOCK_SIZE); }
    return ctx;
}

struct Sha256Ctx sha224_init() {
    struct Sha256Ctx ctx = sha256_init();
    int i = 0;
    while (i < 8) { ctx.h[i] = sha224_H0_[i]; i = i + 1; }
    ctx.is224 = 1;
    return ctx;
}

void Sha256Ctx::update(const &stack unsigned char data, unsigned long len) {
    unsigned long buf_used = self.total_bytes % (unsigned long)SHA256_BLOCK_SIZE;
    self.total_bytes = self.total_bytes + len;

    unsigned long i = (unsigned long)0;
    while (i < len) {
        unsafe {
            self.buf[buf_used] = data[i];
            buf_used = buf_used + (unsigned long)1;
            if (buf_used == (unsigned long)SHA256_BLOCK_SIZE) {
                sha256_compress_(self.h, (const unsigned char*)self.buf);
                buf_used = (unsigned long)0;
            }
        }
        i = i + (unsigned long)1;
    }
}

void Sha256Ctx::finish(&stack unsigned char digest) {
    unsigned long buf_used = self.total_bytes % (unsigned long)SHA256_BLOCK_SIZE;
    unsigned long bit_len  = self.total_bytes * (unsigned long)8;

    // Append 0x80 and zero-pad
    unsafe {
        self.buf[buf_used] = (unsigned char)0x80;
        buf_used = buf_used + (unsigned long)1;

        if (buf_used > (unsigned long)56) {
            while (buf_used < (unsigned long)64) {
                self.buf[buf_used] = (unsigned char)0;
                buf_used = buf_used + (unsigned long)1;
            }
            sha256_compress_(self.h, (const unsigned char*)self.buf);
            buf_used = (unsigned long)0;
        }
        while (buf_used < (unsigned long)56) {
            self.buf[buf_used] = (unsigned char)0;
            buf_used = buf_used + (unsigned long)1;
        }
        // Big-endian 64-bit bit length
        int k = 7;
        while (k >= 0) {
            self.buf[56 + k] = (unsigned char)(bit_len & (unsigned long)0xFF);
            bit_len = bit_len >> (unsigned long)8;
            k = k - 1;
        }
        sha256_compress_(self.h, (const unsigned char*)self.buf);

        // Write digest (big-endian words)
        int out_words = self.is224 != 0 ? 7 : 8;
        int j = 0;
        while (j < out_words) {
            digest[j*4+0] = (unsigned char)((self.h[j] >> 24) & 0xFF);
            digest[j*4+1] = (unsigned char)((self.h[j] >> 16) & 0xFF);
            digest[j*4+2] = (unsigned char)((self.h[j] >>  8) & 0xFF);
            digest[j*4+3] = (unsigned char)((self.h[j]      ) & 0xFF);
            j = j + 1;
        }
    }
}

void sha256(const &stack unsigned char data, unsigned long len,
            &stack unsigned char digest) {
    struct Sha256Ctx ctx = sha256_init();
    ctx.update(data, len);
    ctx.finish(digest);
}
