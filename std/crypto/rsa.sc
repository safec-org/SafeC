// SafeC Standard Library — RSA signatures (see rsa.h).
#pragma once
#include <std/crypto/rsa.h>
#include <std/crypto/asn1.h>
#include <std/crypto/ct.h>
#include <std/mem.h>

namespace std {

// DigestInfo DER prefixes (RFC 8017 §9.2 note 1).
static unsigned char rsa_di_sha1_[15] = { 0x30,0x21,0x30,0x09,0x06,0x05,0x2b,0x0e,0x03,0x02,0x1a,0x05,0x00,0x04,0x14 };
static unsigned char rsa_di_sha224_[19] = { 0x30,0x2d,0x30,0x0d,0x06,0x09,0x60,0x86,0x48,0x01,0x65,0x03,0x04,0x02,0x04,0x05,0x00,0x04,0x1c };
static unsigned char rsa_di_sha256_[19] = { 0x30,0x31,0x30,0x0d,0x06,0x09,0x60,0x86,0x48,0x01,0x65,0x03,0x04,0x02,0x01,0x05,0x00,0x04,0x20 };
static unsigned char rsa_di_sha384_[19] = { 0x30,0x41,0x30,0x0d,0x06,0x09,0x60,0x86,0x48,0x01,0x65,0x03,0x04,0x02,0x02,0x05,0x00,0x04,0x30 };
static unsigned char rsa_di_sha512_[19] = { 0x30,0x51,0x30,0x0d,0x06,0x09,0x60,0x86,0x48,0x01,0x65,0x03,0x04,0x02,0x03,0x05,0x00,0x04,0x40 };
// rsaEncryption 1.2.840.113549.1.1.1
static unsigned char rsa_oid_enc_[9] = { 0x2a,0x86,0x48,0x86,0xf7,0x0d,0x01,0x01,0x01 };

static unsigned long rsa_bits_(const &BnMont m) {
    unsigned long bits = (unsigned long)m.n * 32UL;
    unsigned int top = m.m[m.n - 1];
    unsigned int b = 0x80000000U;
    while (b != 0U && (top & b) == 0U) { bits = bits - 1UL; b = b >> 1; }
    return bits;
}

int RsaPublicKey::init_der(const &unsigned char der, unsigned long len) {
    struct Asn1_ in;
    unsafe { in = asn1_view_((const unsigned char*)der, len); }
    struct Asn1_ seq;
    if (!asn1_get_(&in, (unsigned char)ASN1_TAG_SEQUENCE, &seq)) { return 0; }
    struct Asn1_ nn;
    struct Asn1_ ee;
    if (!asn1_uint_(&seq, &nn) || !asn1_uint_(&seq, &ee)) { return 0; }
    if (nn.len > (unsigned long)RSA_MAX_BYTES || ee.len > 8UL || ee.len == 0UL) { return 0; }
    unsafe {
        if ((ee.p[ee.len - 1UL] & (unsigned char)1) == (unsigned char)0) { return 0; }
        if (ee.len == 1UL && ee.p[0] == (unsigned char)1) { return 0; }
        if (!self.n.init((const &unsigned char)nn.p, nn.len)) { return 0; }
        unsigned long i = 0UL;
        while (i < ee.len) { self.e[i] = ee.p[i]; i = i + 1UL; }
    }
    self.e_len = ee.len;
    self.bits = rsa_bits_(&self.n);
    if (self.bits < (unsigned long)RSA_MIN_BITS) { return 0; }
    return 1;
}

// s^e mod n into em (k bytes). 0 if sig is the wrong length or >= n.
static int rsa_public_op_(const &RsaPublicKey key, const &unsigned char sig, unsigned long sig_len,
                          &unsigned char em) {
    if (sig_len != key.n.bytes) { return 0; }
    return bn_mod_exp_public(&key.n, sig, sig_len, &key.e[0], key.e_len, em);
}

static int rsa_digest_info_(int alg, &unsigned char* prefix, &unsigned long plen) {
    unsafe {
        if (alg == DIGEST_SHA1) { *prefix = &rsa_di_sha1_[0]; *plen = 15UL; return 1; }
        if (alg == DIGEST_SHA224) { *prefix = &rsa_di_sha224_[0]; *plen = 19UL; return 1; }
        if (alg == DIGEST_SHA256) { *prefix = &rsa_di_sha256_[0]; *plen = 19UL; return 1; }
        if (alg == DIGEST_SHA384) { *prefix = &rsa_di_sha384_[0]; *plen = 19UL; return 1; }
        if (alg == DIGEST_SHA512) { *prefix = &rsa_di_sha512_[0]; *plen = 19UL; return 1; }
    }
    return 0;
}

// EMSA-PKCS1-v1_5 encoding of 'hash' into em[0..k). 0 if it doesn't fit.
static int rsa_pkcs1_encode_(int alg, const &unsigned char hash, unsigned long hash_len,
                             &unsigned char em, unsigned long k) {
    unsigned char* prefix;
    unsigned long plen = 0UL;
    if (!rsa_digest_info_(alg, &prefix, &plen)) { return 0; }
    if (hash_len != digest_size(alg)) { return 0; }
    unsigned long tlen = plen + hash_len;
    if (k < tlen + 11UL) { return 0; }
    unsafe {
        unsigned char* e = (unsigned char*)em;
        const unsigned char* h = (const unsigned char*)hash;
        e[0] = (unsigned char)0;
        e[1] = (unsigned char)1;
        unsigned long i = 2UL;
        while (i < k - tlen - 1UL) { e[i] = (unsigned char)0xFF; i = i + 1UL; }
        e[i] = (unsigned char)0;
        i = i + 1UL;
        unsigned long j = 0UL;
        while (j < plen) { e[i + j] = prefix[j]; j = j + 1UL; }
        j = 0UL;
        while (j < hash_len) { e[i + plen + j] = h[j]; j = j + 1UL; }
    }
    return 1;
}

int rsa_verify_pkcs1(const &RsaPublicKey key, int alg, const &unsigned char hash, unsigned long hash_len,
                     const &unsigned char sig, unsigned long sig_len) {
    unsigned char em[RSA_MAX_BYTES];
    unsigned char want[RSA_MAX_BYTES];
    unsigned long k = key.n.bytes;
    if (!rsa_public_op_(key, sig, sig_len, &em[0])) { return 0; }
    if (!rsa_pkcs1_encode_(alg, hash, hash_len, &want[0], k)) { return 0; }
    return ct_eq(&em[0], &want[0], k);
}

int rsa_verify_pss(const &RsaPublicKey key, int alg, const &unsigned char hash, unsigned long hash_len,
                   const &unsigned char sig, unsigned long sig_len, unsigned long salt_len) {
    unsigned char out[RSA_MAX_BYTES];
    unsigned long hl = digest_size(alg);
    if (hl == 0UL || hash_len != hl) { return 0; }
    if (!rsa_public_op_(key, sig, sig_len, &out[0])) { return 0; }
    unsigned long k = key.n.bytes;
    unsigned long em_bits = key.bits - 1UL;
    unsigned long em_len = (em_bits + 7UL) / 8UL;
    unsigned long off = k - em_len;         // 0 or 1 leading byte
    if (off == 1UL && out[0] != (unsigned char)0) { return 0; }
    if (em_len < hl + salt_len + 2UL) { return 0; }
    unsigned char* em;
    unsafe { em = (unsigned char*)&out[0] + off; }
    unsafe {
        if (em[em_len - 1UL] != (unsigned char)0xbc) { return 0; }
        unsigned long db_len = em_len - hl - 1UL;
        unsigned char* h = em + db_len;
        unsigned long top = 8UL * em_len - em_bits;   // 0..7 bits that must be zero
        unsigned char top_mask = (unsigned char)(0xFFU >> (unsigned int)top);
        if ((em[0] & (unsigned char)~top_mask) != (unsigned char)0) { return 0; }
        mgf1_xor(alg, (const &unsigned char)h, hl, (&unsigned char)em, db_len);
        em[0] = em[0] & top_mask;
        unsigned long ps = db_len - salt_len - 1UL;
        unsigned long i = 0UL;
        while (i < ps) { if (em[i] != (unsigned char)0) { return 0; } i = i + 1UL; }
        if (em[ps] != (unsigned char)1) { return 0; }
        // H' = Hash(0x00 * 8 || mHash || salt)
        unsigned char zeros[8];
        i = 0UL;
        while (i < 8UL) { zeros[i] = (unsigned char)0; i = i + 1UL; }
        struct DigestCtx c = DigestCtx::new(alg);
        c.update(&zeros[0], 8UL);
        c.update(hash, hl);
        if (salt_len > 0UL) { c.update((const &unsigned char)(em + ps + 1UL), salt_len); }
        unsigned char h2[DIGEST_MAX_SIZE];
        c.finish(&h2[0]);
        return ct_eq((const &unsigned char)h, &h2[0], hl);
    }
}

// ── Private keys ──────────────────────────────────────────────────────────────

int RsaPrivateKey::init_der(const &unsigned char der, unsigned long len) {
    struct Asn1_ in;
    unsafe { in = asn1_view_((const unsigned char*)der, len); }
    struct Asn1_ seq;
    if (!asn1_get_(&in, (unsigned char)ASN1_TAG_SEQUENCE, &seq)) { return 0; }
    int version = -1;
    if (!asn1_small_int_(&seq, &version) || version != 0) { return 0; }
    // PKCS#8: AlgorithmIdentifier then an OCTET STRING with the PKCS#1 key.
    if (asn1_peek_(&seq) == ASN1_TAG_SEQUENCE) {
        struct Asn1_ algId;
        struct Asn1_ oid;
        struct Asn1_ inner;
        if (!asn1_get_(&seq, (unsigned char)ASN1_TAG_SEQUENCE, &algId)) { return 0; }
        if (!asn1_get_(&algId, (unsigned char)ASN1_TAG_OID, &oid)) { return 0; }
        unsafe { if (!asn1_oid_is_(&oid, &rsa_oid_enc_[0], 9UL)) { return 0; } }
        if (!asn1_get_(&seq, (unsigned char)ASN1_TAG_OCTET_STRING, &inner)) { return 0; }
        unsafe { return self.init_der((const &unsigned char)inner.p, inner.len); }
    }
    struct Asn1_ nn;
    struct Asn1_ ee;
    struct Asn1_ dd;
    if (!asn1_uint_(&seq, &nn) || !asn1_uint_(&seq, &ee) || !asn1_uint_(&seq, &dd)) { return 0; }
    if (nn.len > (unsigned long)RSA_MAX_BYTES || ee.len > 8UL || ee.len == 0UL || dd.len > nn.len) { return 0; }
    unsafe {
        if (!self.pub.n.init((const &unsigned char)nn.p, nn.len)) { return 0; }
        unsigned long i = 0UL;
        while (i < ee.len) { self.pub.e[i] = ee.p[i]; i = i + 1UL; }
        self.pub.e_len = ee.len;
        // d left-padded to the modulus length: the exponentiation's work
        // then depends only on the key size.
        unsigned long k = self.pub.n.bytes;
        i = 0UL;
        while (i < k) { self.d[i] = (unsigned char)0; i = i + 1UL; }
        i = 0UL;
        while (i < dd.len) { self.d[k - dd.len + i] = dd.p[i]; i = i + 1UL; }
        self.d_len = k;
    }
    self.pub.bits = rsa_bits_(&self.pub.n);
    if (self.pub.bits < (unsigned long)RSA_MIN_BITS) { return 0; }
    return 1;
}

void RsaPrivateKey::destroy() {
    mem_zero_secure(self.d[0..(unsigned long)RSA_MAX_BYTES]);
    self.d_len = 0UL;
}

// sig = em^d mod n, then checked with e (a faulty computation must not
// leave the process: RSA's CRT-less private op can't leak factors this
// way, but checking is cheap).
static int rsa_private_op_(const &RsaPrivateKey key, const &unsigned char em, &unsigned char sig) {
    unsigned long k = key.pub.n.bytes;
    if (!bn_mod_exp_secret(&key.pub.n, em, k, &key.d[0], key.d_len, sig)) { return 0; }
    unsigned char back[RSA_MAX_BYTES];
    if (!bn_mod_exp_public(&key.pub.n, sig, k, &key.pub.e[0], key.pub.e_len, &back[0])) { return 0; }
    return ct_eq(&back[0], em, k);
}

int rsa_sign_pkcs1(const &RsaPrivateKey key, int alg, const &unsigned char hash, unsigned long hash_len,
                   &unsigned char sig, unsigned long sig_cap, &unsigned long sig_len) {
    unsigned long k = key.pub.n.bytes;
    if (sig_cap < k) { return 0; }
    unsigned char em[RSA_MAX_BYTES];
    if (!rsa_pkcs1_encode_(alg, hash, hash_len, &em[0], k)) { return 0; }
    if (!rsa_private_op_(key, &em[0], sig)) { return 0; }
    *sig_len = k;
    return 1;
}

int rsa_sign_pss(const &RsaPrivateKey key, int alg, const &unsigned char hash, unsigned long hash_len,
                 &RngCtx rng, &unsigned char sig, unsigned long sig_cap, &unsigned long sig_len) {
    unsigned long hl = digest_size(alg);
    unsigned long k = key.pub.n.bytes;
    if (hl == 0UL || hash_len != hl || sig_cap < k) { return 0; }
    unsigned long em_bits = key.pub.bits - 1UL;
    unsigned long em_len = (em_bits + 7UL) / 8UL;
    unsigned long salt_len = hl;
    if (em_len < hl + salt_len + 2UL) { return 0; }
    unsigned char out[RSA_MAX_BYTES];
    unsigned long off = k - em_len;
    unsigned long i = 0UL;
    while (i < k) { out[i] = (unsigned char)0; i = i + 1UL; }
    unsafe {
        unsigned char* em = (unsigned char*)&out[0] + off;
        unsigned long db_len = em_len - hl - 1UL;
        unsigned char* salt = em + db_len - salt_len;
        rng.fill((&unsigned char)salt, salt_len);
        // H = Hash(0x00 * 8 || mHash || salt)
        unsigned char zeros[8];
        i = 0UL;
        while (i < 8UL) { zeros[i] = (unsigned char)0; i = i + 1UL; }
        struct DigestCtx c = DigestCtx::new(alg);
        c.update(&zeros[0], 8UL);
        c.update(hash, hl);
        c.update((const &unsigned char)salt, salt_len);
        unsigned char* h = em + db_len;
        c.finish((&unsigned char)h);
        em[db_len - salt_len - 1UL] = (unsigned char)1;   // DB = PS || 0x01 || salt
        mgf1_xor(alg, (const &unsigned char)h, hl, (&unsigned char)em, db_len);
        unsigned long top = 8UL * em_len - em_bits;
        em[0] = em[0] & (unsigned char)(0xFFU >> (unsigned int)top);
        em[em_len - 1UL] = (unsigned char)0xbc;
    }
    if (!rsa_private_op_(key, &out[0], sig)) { return 0; }
    *sig_len = k;
    return 1;
}

} // namespace std
