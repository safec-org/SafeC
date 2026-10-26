// SafeC Standard Library — SHA-1 (see sha1.h)
#pragma once
#include <std/stddef.h>
#include <std/crypto/sha1.h>
#include <std/crypto/hash.h>

namespace std {

extern unsigned char* memset(unsigned char* p, int v, size_t n);

static unsigned int sha1_H0_[5] = {
    0x67452301,0xEFCDAB89,0x98BADCFE,0x10325476,0xC3D2E1F0
};

static unsigned int rotl32_(unsigned int x, int n) {
    return (x << n) | (x >> (32 - n));
}

// ── Process one 64-byte block ─────────────────────────────────────────────────

static void sha1_compress_(unsigned int* h, const unsigned char* block) {
    unsigned int W[80];
    int t = 0;
    unsafe {
        while (t < 16) {
            W[t] = ((unsigned int)block[t*4+0] << 24)
                 | ((unsigned int)block[t*4+1] << 16)
                 | ((unsigned int)block[t*4+2] <<  8)
                 | ((unsigned int)block[t*4+3]);
            t = t + 1;
        }
        while (t < 80) {
            W[t] = rotl32_(W[t-3] ^ W[t-8] ^ W[t-14] ^ W[t-16], 1);
            t = t + 1;
        }
    }

    unsigned int a; unsigned int b; unsigned int c; unsigned int d; unsigned int e;
    unsafe { a=h[0]; b=h[1]; c=h[2]; d=h[3]; e=h[4]; }

    t = 0;
    while (t < 80) {
        unsigned int f;
        unsigned int k;
        if (t < 20) {
            f = (b & c) | ((~b) & d);
            k = 0x5A827999U;
        } else if (t < 40) {
            f = b ^ c ^ d;
            k = 0x6ED9EBA1U;
        } else if (t < 60) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8F1BBCDCU;
        } else {
            f = b ^ c ^ d;
            k = 0xCA62C1D6U;
        }
        unsafe {
            unsigned int temp = rotl32_(a, 5) + f + e + k + W[t];
            e = d; d = c; c = rotl32_(b, 30); b = a; a = temp;
        }
        t = t + 1;
    }

    unsafe {
        h[0]=h[0]+a; h[1]=h[1]+b; h[2]=h[2]+c; h[3]=h[3]+d; h[4]=h[4]+e;
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

inline struct Sha1Ctx Sha1Ctx::new() {
    struct Sha1Ctx ctx;
    int i = 0;
    while (i < 5) { ctx.h[i] = sha1_H0_[i]; i = i + 1; }
    ctx.total_bytes = (unsigned long)0;
    unsafe { memset((unsigned char*)ctx.buf, 0, (unsigned long)SHA1_BLOCK_SIZE); }
    return ctx;
}

inline void Sha1Ctx::update_n_(const &unsigned char data, unsigned long len) {
    unsigned long buf_used = self.total_bytes % (unsigned long)SHA1_BLOCK_SIZE;
    self.total_bytes = self.total_bytes + len;

    unsigned long i = (unsigned long)0;
    while (i < len) {
        unsafe {
            self.buf[buf_used] = data[i];
            buf_used = buf_used + (unsigned long)1;
            if (buf_used == (unsigned long)SHA1_BLOCK_SIZE) {
                sha1_compress_(self.h, (const unsigned char*)self.buf);
                buf_used = (unsigned long)0;
            }
        }
        i = i + (unsigned long)1;
    }
}

inline void Sha1Ctx::update([]const unsigned char data_s) {
    unsigned long len = data_s.len;
    const &unsigned char data = crypto_in_(data_s, 0UL);
    self.update_n_(data, len);
}

void Sha1Ctx::finish_n_(&unsigned char digest) {
    unsigned long buf_used = self.total_bytes % (unsigned long)SHA1_BLOCK_SIZE;
    unsigned long bit_len  = self.total_bytes * (unsigned long)8;

    unsafe {
        self.buf[buf_used] = (unsigned char)0x80;
        buf_used = buf_used + (unsigned long)1;

        if (buf_used > (unsigned long)56) {
            while (buf_used < (unsigned long)64) {
                self.buf[buf_used] = (unsigned char)0;
                buf_used = buf_used + (unsigned long)1;
            }
            sha1_compress_(self.h, (const unsigned char*)self.buf);
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
        sha1_compress_(self.h, (const unsigned char*)self.buf);

        int j = 0;
        while (j < 5) {
            digest[j*4+0] = (unsigned char)((self.h[j] >> 24) & 0xFF);
            digest[j*4+1] = (unsigned char)((self.h[j] >> 16) & 0xFF);
            digest[j*4+2] = (unsigned char)((self.h[j] >>  8) & 0xFF);
            digest[j*4+3] = (unsigned char)((self.h[j]      ) & 0xFF);
            j = j + 1;
        }
    }
}

void Sha1Ctx::finish([]unsigned char digest_s) {
    &unsigned char digest = crypto_out_(digest_s, self.digest_size());
    self.finish_n_(digest);
}

inline unsigned long Sha1Ctx::digest_size() const { return (unsigned long)SHA1_DIGEST_SIZE; }

inline unsigned long Sha1Ctx::block_size() const { return (unsigned long)SHA1_BLOCK_SIZE; }

void sha1_n_(const &unsigned char data, unsigned long len,
          &unsigned char digest) {
    struct Sha1Ctx c = Sha1Ctx::new();
    c.update_n_(data, len);
    c.finish_n_(digest);
}

void sha1([]const unsigned char data_s, []unsigned char digest_s) {
    unsigned long len = data_s.len;
    const &unsigned char data = crypto_in_(data_s, 0UL);
    &unsigned char digest = crypto_out_(digest_s, 20UL);
    sha1_n_(data, len, digest);
}

} // namespace std
