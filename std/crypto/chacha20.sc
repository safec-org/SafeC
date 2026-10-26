// SafeC Standard Library — ChaCha20, Poly1305, ChaCha20-Poly1305 (RFC 8439).
#pragma once
#include <std/crypto/chacha20.h>
#include <std/crypto/ct.h>
#include <std/mem.h>

namespace std {

// ── ChaCha20 (RFC 8439 §2.3) ──────────────────────────────────────────────────

static unsigned int cc_load32_(const unsigned char* p) {
    unsafe {
        return (unsigned int)p[0] | ((unsigned int)p[1] << 8) |
               ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
    }
}

static void cc_store32_(unsigned char* p, unsigned int v) {
    unsafe {
        p[0] = (unsigned char)(v & 0xFFU);
        p[1] = (unsigned char)((v >> 8) & 0xFFU);
        p[2] = (unsigned char)((v >> 16) & 0xFFU);
        p[3] = (unsigned char)((v >> 24) & 0xFFU);
    }
}

static unsigned int cc_rotl_(unsigned int v, unsigned int n) {
    return (v << n) | (v >> (32U - n));
}

// 64 bytes of keystream for the state built from key/nonce/counter.
static void cc_block_(const unsigned char* key, const unsigned char* nonce, unsigned int counter,
                      unsigned char* out) {
    unsigned int s[16];
    unsigned int x[16];
    s[0] = 0x61707865U; s[1] = 0x3320646eU; s[2] = 0x79622d32U; s[3] = 0x6b206574U;
    int i = 0;
    unsafe {
        while (i < 8) { s[4 + i] = cc_load32_(key + i * 4); i = i + 1; }
        s[12] = counter;
        s[13] = cc_load32_(nonce);
        s[14] = cc_load32_(nonce + 4);
        s[15] = cc_load32_(nonce + 8);
    }
    i = 0;
    while (i < 16) { x[i] = s[i]; i = i + 1; }
    int r = 0;
    while (r < 10) {
        // Column round, then diagonal round: QR(a, b, c, d) on index quadruples.
        int q = 0;
        while (q < 8) {
            int a; int b; int c; int d;
            if (q < 4) { a = q; b = q + 4; c = q + 8; d = q + 12; }
            else {
                a = q - 4;
                b = 4 + ((q - 3) & 3);
                c = 8 + ((q - 2) & 3);
                d = 12 + ((q - 1) & 3);
            }
            x[a] = x[a] + x[b]; x[d] = cc_rotl_(x[d] ^ x[a], 16U);
            x[c] = x[c] + x[d]; x[b] = cc_rotl_(x[b] ^ x[c], 12U);
            x[a] = x[a] + x[b]; x[d] = cc_rotl_(x[d] ^ x[a], 8U);
            x[c] = x[c] + x[d]; x[b] = cc_rotl_(x[b] ^ x[c], 7U);
            q = q + 1;
        }
        r = r + 1;
    }
    i = 0;
    unsafe {
        while (i < 16) { cc_store32_(out + i * 4, x[i] + s[i]); i = i + 1; }
    }
    mem_zero_secure(s[0..16]);
    mem_zero_secure(x[0..16]);
}

void chacha20_block_n_(const &unsigned char key, const &unsigned char nonce, unsigned int counter,
                    &unsigned char out) {
    unsafe { cc_block_((const unsigned char*)key, (const unsigned char*)nonce, counter, (unsigned char*)out); }
}

void chacha20_block([]const unsigned char key_s, []const unsigned char nonce_s, unsigned int counter, []unsigned char out_s) {
    const &unsigned char key = crypto_in_(key_s, 32UL);
    const &unsigned char nonce = crypto_in_(nonce_s, 12UL);
    &unsigned char out = crypto_out_(out_s, 64UL);
    chacha20_block_n_(key, nonce, counter, out);
}

void chacha20_xor_n_(const &unsigned char key, const &unsigned char nonce, unsigned int counter,
                  &unsigned char data, unsigned long len) {
    unsigned char ks[64];
    unsigned long off = 0UL;
    unsafe {
        unsigned char* d = (unsigned char*)data;
        while (off < len) {
            cc_block_((const unsigned char*)key, (const unsigned char*)nonce, counter, &ks[0]);
            unsigned long n = len - off < 64UL ? len - off : 64UL;
            unsigned long i = 0UL;
            while (i < n) { d[off + i] = d[off + i] ^ ks[i]; i = i + 1UL; }
            off = off + n;
            counter = counter + 1U;
        }
    }
    mem_zero_secure(ks[0..64UL]);
}

void chacha20_xor([]const unsigned char key_s, []const unsigned char nonce_s, unsigned int counter, []unsigned char data_s) {
    const &unsigned char key = crypto_in_(key_s, 32UL);
    const &unsigned char nonce = crypto_in_(nonce_s, 12UL);
    unsigned long len = data_s.len;
    &unsigned char data = crypto_out_(data_s, 0UL);
    chacha20_xor_n_(key, nonce, counter, data, len);
}

// ── Poly1305 (RFC 8439 §2.5; 26-bit limbs, poly1305-donna's 32-bit form) ─────

struct Poly1305 Poly1305::new_n_(const &unsigned char key) {
    struct Poly1305 p;
    unsafe {
        const unsigned char* k = (const unsigned char*)key;
        p.r[0] = cc_load32_(k) & 0x3ffffffU;
        p.r[1] = (cc_load32_(k + 3) >> 2) & 0x3ffff03U;
        p.r[2] = (cc_load32_(k + 6) >> 4) & 0x3ffc0ffU;
        p.r[3] = (cc_load32_(k + 9) >> 6) & 0x3f03fffU;
        p.r[4] = (cc_load32_(k + 12) >> 8) & 0x00fffffU;
        p.pad[0] = cc_load32_(k + 16);
        p.pad[1] = cc_load32_(k + 20);
        p.pad[2] = cc_load32_(k + 24);
        p.pad[3] = cc_load32_(k + 28);
    }
    int i = 0;
    while (i < 5) { p.h[i] = 0U; i = i + 1; }
    i = 0;
    while (i < 16) { p.buf[i] = (unsigned char)0; i = i + 1; }
    p.buf_len = 0UL;
    return p;
}

struct Poly1305 Poly1305::new([]const unsigned char key_s) {
    const &unsigned char key = crypto_in_(key_s, 32UL);
    return Poly1305::new_n_(key);
}

// Absorbs one 16-byte block; hibit is 1<<24 for full blocks, 0 for the
// padded final one.
static void poly_block_(&Poly1305 p, const unsigned char* m, unsigned int hibit) {
    unsigned long long r0 = (unsigned long long)p.r[0];
    unsigned long long r1 = (unsigned long long)p.r[1];
    unsigned long long r2 = (unsigned long long)p.r[2];
    unsigned long long r3 = (unsigned long long)p.r[3];
    unsigned long long r4 = (unsigned long long)p.r[4];
    unsigned long long s1 = r1 * 5ULL;
    unsigned long long s2 = r2 * 5ULL;
    unsigned long long s3 = r3 * 5ULL;
    unsigned long long s4 = r4 * 5ULL;
    unsigned int h0 = p.h[0];
    unsigned int h1 = p.h[1];
    unsigned int h2 = p.h[2];
    unsigned int h3 = p.h[3];
    unsigned int h4 = p.h[4];
    unsafe {
        h0 = h0 + (cc_load32_(m) & 0x3ffffffU);
        h1 = h1 + ((cc_load32_(m + 3) >> 2) & 0x3ffffffU);
        h2 = h2 + ((cc_load32_(m + 6) >> 4) & 0x3ffffffU);
        h3 = h3 + ((cc_load32_(m + 9) >> 6) & 0x3ffffffU);
        h4 = h4 + ((cc_load32_(m + 12) >> 8) | hibit);
    }
    unsigned long long d0 = (unsigned long long)h0 * r0 + (unsigned long long)h1 * s4 +
                            (unsigned long long)h2 * s3 + (unsigned long long)h3 * s2 +
                            (unsigned long long)h4 * s1;
    unsigned long long d1 = (unsigned long long)h0 * r1 + (unsigned long long)h1 * r0 +
                            (unsigned long long)h2 * s4 + (unsigned long long)h3 * s3 +
                            (unsigned long long)h4 * s2;
    unsigned long long d2 = (unsigned long long)h0 * r2 + (unsigned long long)h1 * r1 +
                            (unsigned long long)h2 * r0 + (unsigned long long)h3 * s4 +
                            (unsigned long long)h4 * s3;
    unsigned long long d3 = (unsigned long long)h0 * r3 + (unsigned long long)h1 * r2 +
                            (unsigned long long)h2 * r1 + (unsigned long long)h3 * r0 +
                            (unsigned long long)h4 * s4;
    unsigned long long d4 = (unsigned long long)h0 * r4 + (unsigned long long)h1 * r3 +
                            (unsigned long long)h2 * r2 + (unsigned long long)h3 * r1 +
                            (unsigned long long)h4 * r0;
    unsigned long long c = d0 >> 26ULL;
    h0 = (unsigned int)(d0 & 0x3ffffffULL);
    d1 = d1 + c; c = d1 >> 26ULL; h1 = (unsigned int)(d1 & 0x3ffffffULL);
    d2 = d2 + c; c = d2 >> 26ULL; h2 = (unsigned int)(d2 & 0x3ffffffULL);
    d3 = d3 + c; c = d3 >> 26ULL; h3 = (unsigned int)(d3 & 0x3ffffffULL);
    d4 = d4 + c; c = d4 >> 26ULL; h4 = (unsigned int)(d4 & 0x3ffffffULL);
    h0 = h0 + (unsigned int)(c * 5ULL);
    unsigned int cc = h0 >> 26;
    h0 = h0 & 0x3ffffffU;
    h1 = h1 + cc;
    p.h[0] = h0; p.h[1] = h1; p.h[2] = h2; p.h[3] = h3; p.h[4] = h4;
}

void Poly1305::update_n_(const &unsigned char data, unsigned long len) {
    unsafe {
        const unsigned char* m = (const unsigned char*)data;
        unsigned long off = 0UL;
        if (self.buf_len > 0UL) {
            while (off < len && self.buf_len < 16UL) {
                self.buf[self.buf_len] = m[off];
                self.buf_len = self.buf_len + 1UL;
                off = off + 1UL;
            }
            if (self.buf_len < 16UL) { return; }
            poly_block_(self, (const unsigned char*)&self.buf[0], 1U << 24);
            self.buf_len = 0UL;
        }
        while (len - off >= 16UL) {
            poly_block_(self, m + off, 1U << 24);
            off = off + 16UL;
        }
        while (off < len) {
            self.buf[self.buf_len] = m[off];
            self.buf_len = self.buf_len + 1UL;
            off = off + 1UL;
        }
    }
}

void Poly1305::update([]const unsigned char data_s) {
    unsigned long len = data_s.len;
    const &unsigned char data = crypto_in_(data_s, 0UL);
    self.update_n_(data, len);
}

void Poly1305::pad16() {
    if (self.buf_len == 0UL) { return; }
    while (self.buf_len < 16UL) {
        self.buf[self.buf_len] = (unsigned char)0;
        self.buf_len = self.buf_len + 1UL;
    }
    unsafe { poly_block_(self, (const unsigned char*)&self.buf[0], 1U << 24); }
    self.buf_len = 0UL;
}

void Poly1305::finish_n_(&unsigned char tag) {
    if (self.buf_len > 0UL) {
        unsigned long i = self.buf_len;
        self.buf[i] = (unsigned char)1;
        i = i + 1UL;
        while (i < 16UL) { self.buf[i] = (unsigned char)0; i = i + 1UL; }
        unsafe { poly_block_(self, (const unsigned char*)&self.buf[0], 0U); }
    }
    unsigned int h0 = self.h[0];
    unsigned int h1 = self.h[1];
    unsigned int h2 = self.h[2];
    unsigned int h3 = self.h[3];
    unsigned int h4 = self.h[4];
    unsigned int c = h1 >> 26; h1 = h1 & 0x3ffffffU;
    h2 = h2 + c; c = h2 >> 26; h2 = h2 & 0x3ffffffU;
    h3 = h3 + c; c = h3 >> 26; h3 = h3 & 0x3ffffffU;
    h4 = h4 + c; c = h4 >> 26; h4 = h4 & 0x3ffffffU;
    h0 = h0 + c * 5U; c = h0 >> 26; h0 = h0 & 0x3ffffffU;
    h1 = h1 + c;
    // g = h + 5 - 2^130; take g when it doesn't underflow (h >= p).
    unsigned int g0 = h0 + 5U; c = g0 >> 26; g0 = g0 & 0x3ffffffU;
    unsigned int g1 = h1 + c; c = g1 >> 26; g1 = g1 & 0x3ffffffU;
    unsigned int g2 = h2 + c; c = g2 >> 26; g2 = g2 & 0x3ffffffU;
    unsigned int g3 = h3 + c; c = g3 >> 26; g3 = g3 & 0x3ffffffU;
    unsigned int g4 = h4 + c - (1U << 26);
    unsigned int mask = (g4 >> 31) - 1U;
    g0 = g0 & mask; g1 = g1 & mask; g2 = g2 & mask; g3 = g3 & mask; g4 = g4 & mask;
    mask = ~mask;
    h0 = (h0 & mask) | g0;
    h1 = (h1 & mask) | g1;
    h2 = (h2 & mask) | g2;
    h3 = (h3 & mask) | g3;
    h4 = (h4 & mask) | g4;
    // h % 2^128, then + s.
    unsigned int w0 = h0 | (h1 << 26);
    unsigned int w1 = (h1 >> 6) | (h2 << 20);
    unsigned int w2 = (h2 >> 12) | (h3 << 14);
    unsigned int w3 = (h3 >> 18) | (h4 << 8);
    unsigned long long f = (unsigned long long)w0 + (unsigned long long)self.pad[0];
    w0 = (unsigned int)f;
    f = (unsigned long long)w1 + (unsigned long long)self.pad[1] + (f >> 32ULL);
    w1 = (unsigned int)f;
    f = (unsigned long long)w2 + (unsigned long long)self.pad[2] + (f >> 32ULL);
    w2 = (unsigned int)f;
    f = (unsigned long long)w3 + (unsigned long long)self.pad[3] + (f >> 32ULL);
    w3 = (unsigned int)f;
    unsafe {
        unsigned char* t = (unsigned char*)tag;
        cc_store32_(t, w0);
        cc_store32_(t + 4, w1);
        cc_store32_(t + 8, w2);
        cc_store32_(t + 12, w3);
    }
    mem_zero_secure_obj(self);
}

void Poly1305::finish([]unsigned char tag_s) {
    &unsigned char tag = crypto_out_(tag_s, 16UL);
    self.finish_n_(tag);
}

void poly1305_n_(const &unsigned char key, const ?&unsigned char msg, unsigned long len, &unsigned char tag) {
    struct Poly1305 p = Poly1305::new_n_(key);
    match (msg) {
        case some(m): { p.update_n_(m, len); }
        case null: { }
    }
    p.finish_n_(tag);
}

void poly1305([]const unsigned char key_s, []const unsigned char msg_s, []unsigned char tag_s) {
    const &unsigned char key = crypto_in_(key_s, 32UL);
    unsigned long len = msg_s.len;
    const ?&unsigned char msg = crypto_opt_(msg_s);
    &unsigned char tag = crypto_out_(tag_s, 16UL);
    poly1305_n_(key, msg, len, tag);
}

// ── AEAD_CHACHA20_POLY1305 (RFC 8439 §2.8) ────────────────────────────────────

static void aead_tag_(const unsigned char* key, const unsigned char* nonce,
                      const ?&unsigned char aad, unsigned long aad_len,
                      const unsigned char* ct, unsigned long len, unsigned char* tag) {
    unsigned char block0[64];
    unsafe { cc_block_(key, nonce, 0U, &block0[0]); }
    struct Poly1305 p = Poly1305::new_n_(&block0[0]);
    mem_zero_secure(block0[0..64UL]);
    match (aad) {
        case some(a): { if (aad_len > 0UL) { p.update_n_(a, aad_len); } }
        case null: { }
    }
    p.pad16();
    unsafe { if (len > 0UL) { p.update_n_(ct, len); } }
    p.pad16();
    unsigned char lens[16];
    unsigned long long al = (unsigned long long)aad_len;
    unsigned long long cl = (unsigned long long)len;
    int i = 0;
    while (i < 8) {
        lens[i] = (unsigned char)((al >> (unsigned long long)(8 * i)) & 0xFFULL);
        lens[8 + i] = (unsigned char)((cl >> (unsigned long long)(8 * i)) & 0xFFULL);
        i = i + 1;
    }
    p.update_n_(&lens[0], 16UL);
    unsafe { p.finish_n_(tag); }
}

void chacha20_poly1305_seal_n_(const &unsigned char key, const &unsigned char nonce,
                            const ?&unsigned char aad, unsigned long aad_len,
                            &unsigned char data, unsigned long len, &unsigned char tag) {
    chacha20_xor_n_(key, nonce, 1U, data, len);
    unsafe {
        aead_tag_((const unsigned char*)key, (const unsigned char*)nonce, aad, aad_len,
                  (const unsigned char*)data, len, (unsigned char*)tag);
    }
}

void chacha20_poly1305_seal([]const unsigned char key_s, []const unsigned char nonce_s, []const unsigned char aad_s, []unsigned char data_s, []unsigned char tag_s) {
    const &unsigned char key = crypto_in_(key_s, 32UL);
    const &unsigned char nonce = crypto_in_(nonce_s, 12UL);
    unsigned long aad_len = aad_s.len;
    const ?&unsigned char aad = crypto_opt_(aad_s);
    unsigned long len = data_s.len;
    &unsigned char data = crypto_out_(data_s, 0UL);
    &unsigned char tag = crypto_out_(tag_s, 16UL);
    chacha20_poly1305_seal_n_(key, nonce, aad, aad_len, data, len, tag);
}

int chacha20_poly1305_open_n_(const &unsigned char key, const &unsigned char nonce,
                           const ?&unsigned char aad, unsigned long aad_len,
                           &unsigned char data, unsigned long len, const &unsigned char tag) {
    unsigned char want[16];
    unsafe {
        aead_tag_((const unsigned char*)key, (const unsigned char*)nonce, aad, aad_len,
                  (const unsigned char*)data, len, &want[0]);
    }
    int good = ct_eq_n_(&want[0], tag, 16UL);
    mem_zero_secure(want[0..16UL]);
    if (!good) { return 0; }
    chacha20_xor_n_(key, nonce, 1U, data, len);
    return 1;
}

int chacha20_poly1305_open([]const unsigned char key_s, []const unsigned char nonce_s, []const unsigned char aad_s, []unsigned char data_s, []const unsigned char tag_s) {
    const &unsigned char key = crypto_in_(key_s, 32UL);
    const &unsigned char nonce = crypto_in_(nonce_s, 12UL);
    unsigned long aad_len = aad_s.len;
    const ?&unsigned char aad = crypto_opt_(aad_s);
    unsigned long len = data_s.len;
    &unsigned char data = crypto_out_(data_s, 0UL);
    const &unsigned char tag = crypto_in_(tag_s, 16UL);
    return chacha20_poly1305_open_n_(key, nonce, aad, aad_len, data, len, tag);
}

} // namespace std
