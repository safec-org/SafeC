// SafeC Standard Library — SHA-512 / SHA-384 (see sha512.h)
#pragma once
#include <std/crypto/sha512.h>

namespace std {

extern void* memset(void* p, int v, unsigned long n);

// ── SHA-512 constants ─────────────────────────────────────────────────────────

static unsigned long long sha512_K_[80] = {
    0x428a2f98d728ae22ULL,0x7137449123ef65cdULL,0xb5c0fbcfec4d3b2fULL,0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL,0x59f111f1b605d019ULL,0x923f82a4af194f9bULL,0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL,0x12835b0145706fbeULL,0x243185be4ee4b28cULL,0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL,0x80deb1fe3b1696b1ULL,0x9bdc06a725c71235ULL,0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL,0xefbe4786384f25e3ULL,0x0fc19dc68b8cd5b5ULL,0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL,0x4a7484aa6ea6e483ULL,0x5cb0a9dcbd41fbd4ULL,0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL,0xa831c66d2db43210ULL,0xb00327c898fb213fULL,0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL,0xd5a79147930aa725ULL,0x06ca6351e003826fULL,0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL,0x2e1b21385c26c926ULL,0x4d2c6dfc5ac42aedULL,0x53380d139d95b3dfULL,
    0x650a73548baf63deULL,0x766a0abb3c77b2a8ULL,0x81c2c92e47edaee6ULL,0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL,0xa81a664bbc423001ULL,0xc24b8b70d0f89791ULL,0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL,0xd69906245565a910ULL,0xf40e35855771202aULL,0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL,0x1e376c085141ab53ULL,0x2748774cdf8eeb99ULL,0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL,0x4ed8aa4ae3418acbULL,0x5b9cca4f7763e373ULL,0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL,0x78a5636f43172f60ULL,0x84c87814a1f0ab72ULL,0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL,0xa4506cebde82bde9ULL,0xbef9a3f7b2c67915ULL,0xc67178f2e372532bULL,
    0xca273eceea26619cULL,0xd186b8c721c0c207ULL,0xeada7dd6cde0eb1eULL,0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL,0x0a637dc5a2c898a6ULL,0x113f9804bef90daeULL,0x1b710b35131c471bULL,
    0x28db77f523047d84ULL,0x32caab7b40c72493ULL,0x3c9ebe0a15c9bebcULL,0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL,0x597f299cfc657e2aULL,0x5fcb6fab3ad6faecULL,0x6c44198c4a475817ULL
};

static unsigned long long sha512_H0_[8] = {
    0x6a09e667f3bcc908ULL,0xbb67ae8584caa73bULL,0x3c6ef372fe94f82bULL,0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL,0x9b05688c2b3e6c1fULL,0x1f83d9abfb41bd6bULL,0x5be0cd19137e2179ULL
};

static unsigned long long sha384_H0_[8] = {
    0xcbbb9d5dc1059ed8ULL,0x629a292a367cd507ULL,0x9159015a3070dd17ULL,0x152fecd8f70e5939ULL,
    0x67332667ffc00b31ULL,0x8eb44a8768581511ULL,0xdb0c2e0d64f98fa7ULL,0x47b5481dbefa4fa4ULL
};

static unsigned long long rotr64_(unsigned long long x, int n) {
    return (x >> n) | (x << (64 - n));
}

// ── Process one 128-byte block ────────────────────────────────────────────────

static void sha512_compress_(unsigned long long* h, const unsigned char* block) {
    unsigned long long W[80];
    int t = 0;
    unsafe {
        while (t < 16) {
            unsigned long long w = 0ULL;
            int k = 0;
            while (k < 8) {
                w = (w << 8) | (unsigned long long)block[t*8+k];
                k = k + 1;
            }
            W[t] = w;
            t = t + 1;
        }
        while (t < 80) {
            unsigned long long s0 = rotr64_(W[t-15],1) ^ rotr64_(W[t-15],8) ^ (W[t-15] >> 7);
            unsigned long long s1 = rotr64_(W[t- 2],19) ^ rotr64_(W[t- 2],61) ^ (W[t- 2] >> 6);
            W[t] = W[t-16] + s0 + W[t-7] + s1;
            t = t + 1;
        }
    }

    unsigned long long a; unsigned long long b; unsigned long long c; unsigned long long d;
    unsigned long long e; unsigned long long f; unsigned long long g; unsigned long long hh;
    unsafe {
        a=h[0]; b=h[1]; c=h[2]; d=h[3];
        e=h[4]; f=h[5]; g=h[6]; hh=h[7];
    }

    t = 0;
    while (t < 80) {
        unsigned long long S1  = rotr64_(e,14)^rotr64_(e,18)^rotr64_(e,41);
        unsigned long long ch  = (e & f) ^ (~e & g);
        unsigned long long T1;
        unsafe { T1 = hh + S1 + ch + sha512_K_[t] + W[t]; }
        unsigned long long S0  = rotr64_(a,28)^rotr64_(a,34)^rotr64_(a,39);
        unsigned long long maj = (a & b) ^ (a & c) ^ (b & c);
        unsigned long long T2  = S0 + maj;
        hh=g; g=f; f=e; e=d+T1;
        d=c; c=b; b=a; a=T1+T2;
        t = t + 1;
    }

    unsafe {
        h[0]=h[0]+a; h[1]=h[1]+b; h[2]=h[2]+c; h[3]=h[3]+d;
        h[4]=h[4]+e; h[5]=h[5]+f; h[6]=h[6]+g; h[7]=h[7]+hh;
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

inline struct Sha512Ctx sha512_init() {
    struct Sha512Ctx ctx;
    int i = 0;
    while (i < 8) { ctx.h[i] = sha512_H0_[i]; i = i + 1; }
    ctx.total_bytes = (unsigned long long)0;
    ctx.is384       = 0;
    unsafe { memset((void*)ctx.buf, 0, (unsigned long)SHA512_BLOCK_SIZE); }
    return ctx;
}

inline struct Sha512Ctx sha384_init() {
    struct Sha512Ctx ctx = sha512_init();
    int i = 0;
    while (i < 8) { ctx.h[i] = sha384_H0_[i]; i = i + 1; }
    ctx.is384 = 1;
    return ctx;
}

inline void Sha512Ctx::update(const &stack unsigned char data, unsigned long len) {
    unsigned long buf_used = (unsigned long)(self.total_bytes % (unsigned long long)SHA512_BLOCK_SIZE);
    self.total_bytes = self.total_bytes + (unsigned long long)len;

    unsigned long i = (unsigned long)0;
    while (i < len) {
        unsafe {
            self.buf[buf_used] = data[i];
            buf_used = buf_used + (unsigned long)1;
            if (buf_used == (unsigned long)SHA512_BLOCK_SIZE) {
                sha512_compress_(self.h, (const unsigned char*)self.buf);
                buf_used = (unsigned long)0;
            }
        }
        i = i + (unsigned long)1;
    }
}

void Sha512Ctx::finish(&stack unsigned char digest) {
    unsigned long buf_used = (unsigned long)(self.total_bytes % (unsigned long long)SHA512_BLOCK_SIZE);
    unsigned long long bit_len = self.total_bytes * (unsigned long long)8;

    unsafe {
        self.buf[buf_used] = (unsigned char)0x80;
        buf_used = buf_used + (unsigned long)1;

        if (buf_used > (unsigned long)112) {
            while (buf_used < (unsigned long)128) {
                self.buf[buf_used] = (unsigned char)0;
                buf_used = buf_used + (unsigned long)1;
            }
            sha512_compress_(self.h, (const unsigned char*)self.buf);
            buf_used = (unsigned long)0;
        }
        while (buf_used < (unsigned long)112) {
            self.buf[buf_used] = (unsigned char)0;
            buf_used = buf_used + (unsigned long)1;
        }
        // 128-bit big-endian bit length: high 8 bytes are always zero here
        // (see Sha512Ctx::total_bytes's doc comment on the 64-bit-counter
        // simplification), low 8 bytes are the real length.
        int k = 0;
        while (k < 8) {
            self.buf[112 + k] = (unsigned char)0;
            k = k + 1;
        }
        k = 7;
        while (k >= 0) {
            self.buf[120 + k] = (unsigned char)(bit_len & (unsigned long long)0xFF);
            bit_len = bit_len >> (unsigned long long)8;
            k = k - 1;
        }
        sha512_compress_(self.h, (const unsigned char*)self.buf);

        int out_words = self.is384 != 0 ? 6 : 8;
        int j = 0;
        while (j < out_words) {
            unsigned long long word = self.h[j];
            int b = 0;
            while (b < 8) {
                digest[j*8+b] = (unsigned char)((word >> (56 - b*8)) & 0xFF);
                b = b + 1;
            }
            j = j + 1;
        }
    }
}

void sha512(const &stack unsigned char data, unsigned long len,
            &stack unsigned char digest) {
    struct Sha512Ctx ctx = sha512_init();
    ctx.update(data, len);
    ctx.finish(digest);
}

void sha384(const &stack unsigned char data, unsigned long len,
            &stack unsigned char digest) {
    struct Sha512Ctx ctx = sha384_init();
    ctx.update(data, len);
    ctx.finish(digest);
}

} // namespace std
