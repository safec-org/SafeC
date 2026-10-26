// SafeC Standard Library — run-time hash selection (see digest.h).
#pragma once
#include <std/crypto/digest.h>

namespace std {

unsigned long digest_size(int alg) {
    if (alg == DIGEST_SHA1) { return 20UL; }
    if (alg == DIGEST_SHA224) { return 28UL; }
    if (alg == DIGEST_SHA256) { return 32UL; }
    if (alg == DIGEST_SHA384) { return 48UL; }
    if (alg == DIGEST_SHA512) { return 64UL; }
    return 0UL;
}

struct DigestCtx DigestCtx::new(int alg) {
    struct DigestCtx c;
    c.alg = alg;
    c.s1 = Sha1Ctx::new();
    c.s256 = Sha256Ctx::new();
    c.s512 = Sha512Ctx::new();
    if (alg == DIGEST_SHA224) { c.s256 = Sha256Ctx::new_224(); }
    else if (alg == DIGEST_SHA384) { c.s512 = Sha512Ctx::new_384(); }
    else if (alg != DIGEST_SHA1 && alg != DIGEST_SHA256 && alg != DIGEST_SHA512) { c.alg = DIGEST_NONE; }
    return c;
}

void DigestCtx::update_n_(const &unsigned char data, unsigned long len) {
    if (self.alg == DIGEST_SHA1) { self.s1.update_n_(data, len); }
    else if (self.alg == DIGEST_SHA224 || self.alg == DIGEST_SHA256) { self.s256.update_n_(data, len); }
    else if (self.alg == DIGEST_SHA384 || self.alg == DIGEST_SHA512) { self.s512.update_n_(data, len); }
}

void DigestCtx::update([]const unsigned char data_s) {
    unsigned long len = data_s.len;
    const &unsigned char data = crypto_in_(data_s, 0UL);
    self.update_n_(data, len);
}

unsigned long DigestCtx::finish_n_(&unsigned char out) {
    if (self.alg == DIGEST_SHA1) { self.s1.finish_n_(out); }
    else if (self.alg == DIGEST_SHA224 || self.alg == DIGEST_SHA256) { self.s256.finish_n_(out); }
    else if (self.alg == DIGEST_SHA384 || self.alg == DIGEST_SHA512) { self.s512.finish_n_(out); }
    return digest_size(self.alg);
}

unsigned long DigestCtx::finish([]unsigned char out_s) {
    &unsigned char out = crypto_out_(out_s, digest_size(self.alg));
    return self.finish_n_(out);
}

unsigned long digest_n_(int alg, const ?&unsigned char data, unsigned long len, &unsigned char out) {
    struct DigestCtx c = DigestCtx::new(alg);
    if (c.alg == DIGEST_NONE) { return 0UL; }
    match (data) {
        case some(d): { if (len > 0UL) { c.update_n_(d, len); } }
        case null: { }
    }
    return c.finish_n_(out);
}

unsigned long digest(int alg, []const unsigned char data_s, []unsigned char out_s) {
    unsigned long len = data_s.len;
    const ?&unsigned char data = crypto_opt_(data_s);
    &unsigned char out = crypto_out_(out_s, digest_size(alg));
    return digest_n_(alg, data, len, out);
}

void mgf1_xor_n_(int alg, const &unsigned char seed, unsigned long seed_len,
              &unsigned char data, unsigned long len) {
    unsigned long hl = digest_size(alg);
    if (hl == 0UL) { return; }
    unsigned char h[DIGEST_MAX_SIZE];
    unsigned int counter = 0U;
    unsigned long off = 0UL;
    while (off < len) {
        struct DigestCtx c = DigestCtx::new(alg);
        c.update_n_(seed, seed_len);
        unsigned char ctr[4];
        ctr[0] = (unsigned char)(counter >> 24);
        ctr[1] = (unsigned char)((counter >> 16) & 0xFFU);
        ctr[2] = (unsigned char)((counter >> 8) & 0xFFU);
        ctr[3] = (unsigned char)(counter & 0xFFU);
        c.update_n_(&ctr[0], 4UL);
        c.finish_n_(&h[0]);
        unsafe {
            unsigned char* d = (unsigned char*)data;
            unsigned long i = 0UL;
            while (i < hl && off + i < len) { d[off + i] = d[off + i] ^ h[i]; i = i + 1UL; }
        }
        off = off + hl;
        counter = counter + 1U;
    }
}

void mgf1_xor(int alg, []const unsigned char seed_s, []unsigned char data_s) {
    unsigned long seed_len = seed_s.len;
    const &unsigned char seed = crypto_in_(seed_s, 0UL);
    unsigned long len = data_s.len;
    &unsigned char data = crypto_out_(data_s, 0UL);
    mgf1_xor_n_(alg, seed, seed_len, data, len);
}

} // namespace std
