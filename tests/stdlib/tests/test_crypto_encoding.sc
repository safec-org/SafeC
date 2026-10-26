// std::crypto (sha1/sha256/sha512/hmac/aes/rng/x509/tls/secure_alloc) +
// std::encoding (base64/url/utf8/ascii)
// Known-answer vectors: FIPS 180-4 "abc", RFC 4231 case 2, FIPS-197 Appendix C.
#include <std/test/test.h>
#include <std/mem.h>
#include <std/str.h>
#include <std/crypto/sha1.h>
#include <std/crypto/sha256.h>
#include <std/crypto/sha512.h>
#include <std/crypto/hash.h>
#include <std/crypto/hmac.h>
#include <std/crypto/aes.h>
#include <std/crypto/rng.h>
#include <std/crypto/x509.h>
#include <std/crypto/tls.h>
#include <std/crypto/secure_alloc.h>
#include <std/encoding/base64.h>
#include <std/encoding/url.h>
#include <std/encoding/utf8.h>
#include <std/encoding/ascii.h>

static void to_hex(const &unsigned char d, unsigned long n, &char out) {
    unsafe {
        const &char digits = "0123456789abcdef";
        unsigned long i = 0UL;
        while (i < n) {
            out[i * 2UL] = digits[d[i] >> 4];
            out[i * 2UL + 1UL] = digits[d[i] & 15];
            i = i + 1UL;
        }
        out[n * 2UL] = (char)0;
    }
}

static int hexv(char c) {
    if (c >= '0' && c <= '9') return (int)(c - '0');
    return (int)(c - 'a') + 10;
}
static void hex_to_bytes(const &char h, &unsigned char out) {
    unsafe {
        unsigned long i = 0UL;
        while (h[2UL * i] != (char)0) {
            out[i] = (unsigned char)(hexv(h[2UL * i]) * 16 + hexv(h[2UL * i + 1UL]));
            i = i + 1UL;
        }
    }
}

void t_sha1() {
    unsafe {
        unsigned char d[20];
        char hex[64];
        sha1(str_bytes("abc"), d[..]);
        to_hex(d, 20UL, hex);
        ASSERT_STR_EQ(hex, "a9993e364706816aba3e25717850c26c9cd0d89d");
    }
}

void t_sha256() {
    unsafe {
        unsigned char d[32];
        char hex[80];
        sha256(str_bytes("abc"), d[..]);
        to_hex(d, 32UL, hex);
        ASSERT_STR_EQ(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        // Streaming in odd-size chunks must equal one-shot (448-bit message, FIPS 180-4 B.2).
        const &char msg = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        struct Sha256Ctx c = Sha256Ctx::new();
        c.update(str_bytes(msg)[0..5]);
        c.update(str_bytes(msg)[5..56]);
        c.finish(d[..]);
        to_hex(d, 32UL, hex);
        ASSERT_STR_EQ(hex, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
        // Empty input
        sha256(str_bytes(""), d[..]);
        to_hex(d, 32UL, hex);
        ASSERT_STR_EQ(hex, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }
}

void t_sha512() {
    unsafe {
        unsigned char d[64];
        char hex[160];
        sha512(str_bytes("abc"), d[..]);
        to_hex(d, 64UL, hex);
        ASSERT_STR_EQ(hex, "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f");
        sha384(str_bytes("abc"), d[..]);
        to_hex(d, 48UL, hex);
        ASSERT_STR_EQ(hex, "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7");
    }
}

void t_hmac() {
    unsafe {
        unsigned char d[32];
        char hex[80];
        hmac_sha256(str_bytes("Jefe"), str_bytes("what do ya want for nothing?"), d[..]);
        to_hex(d, 32UL, hex);
        ASSERT_STR_EQ(hex, "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    }
}

// ── hash.h's Hasher / BlockHash ─────────────────────────────────────────────
// One generic check run over every hash context and over Hmac<H> for each:
// hash_bytes one-shot, and streaming in uneven pieces across block
// boundaries, both give the expected digest of digest_size() bytes.
generic<H: Hasher>
int hasher_kat(H fresh, const &unsigned char msg, unsigned long len, const &char expected) {
    unsigned char d[64];
    char hex[160];
    int ok = 1;
    unsafe {
        if (fresh.digest_size() * 2UL != str_len(expected)) ok = 0;
        hash_bytes(fresh, ((const unsigned char*)msg)[0..len], d[..]);
        to_hex(&d[0], fresh.digest_size(), &hex[0]);
        if (!str_eq(&hex[0], expected)) ok = 0;
        H h = fresh;
        unsigned long cuts[4] = { 1UL, 63UL, 64UL, 129UL };
        unsigned long at = 0UL;
        unsigned long c = 0UL;
        while (c < 4UL) {
            unsigned long n = cuts[c] < len - at ? cuts[c] : len - at;
            h.update(((const unsigned char*)(((const unsigned char*)msg + at)))[0..n]);
            at = at + n;
            c = c + 1UL;
        }
        h.update(((const unsigned char*)(((const unsigned char*)msg + at)))[0..len - at]);
        h.finish(d[..]);
        to_hex(&d[0], fresh.digest_size(), &hex[0]);
        if (!str_eq(&hex[0], expected)) ok = 0;
    }
    return ok;
}

void t_hasher_generic() {
    unsafe {
        unsigned char m[1000];
        unsigned long i = 0UL;
        while (i < 1000UL) { m[i] = (unsigned char)(i * 7UL + 3UL); i = i + 1UL; }
        const &unsigned char msg = &m[0];
        ASSERT_EQ(hasher_kat(Sha1Ctx::new(), msg, 1000UL, "4231a8a50a10fa9758db8ec71fdef855b751048a"), 1);
        ASSERT_EQ(hasher_kat(Sha256Ctx::new_224(), msg, 1000UL, "23729dacbd480285c5c68439e2fe22ca5611a63cd6c14d2ec5ae2c85"), 1);
        ASSERT_EQ(hasher_kat(Sha256Ctx::new(), msg, 1000UL, "1e9bc38cbf860b9ec31918b065f9b52476c549a782e0e7990bed8ce3868d2371"), 1);
        ASSERT_EQ(hasher_kat(Sha512Ctx::new_384(), msg, 1000UL, "94c38db521ca733b8904c2d14b6e82d33dcfcc26e1318c579dbee1fc2f472019034e792263b9d46e90e700de2f6e7e91"), 1);
        ASSERT_EQ(hasher_kat(Sha512Ctx::new(), msg, 1000UL, "00e36fccf193e59697a92b5ab24666ce6326d7fa16bf10832d0991ddc591112e9dfa6a636950ed9c4d67344a760654c2ff7785e1d60094d651038735b5dccabd"), 1);

        // Hmac<H> over every BlockHash: RFC 2202 / RFC 4231 case 2 ("Jefe")
        // and case 6 (a 131-byte key, longer than every block: hashed first).
        const &unsigned char jefe = (const &unsigned char)"Jefe";
        const &unsigned char want = (const &unsigned char)"what do ya want for nothing?";
        ASSERT_EQ(hasher_kat(Hmac::new(Sha1Ctx::new(), ((const unsigned char*)jefe)[0..4UL]), want, 28UL, "effcdf6ae5eb2fa2d27416d5f184df9c259a7c79"), 1);
        ASSERT_EQ(hasher_kat(Hmac::new(Sha256Ctx::new_224(), ((const unsigned char*)jefe)[0..4UL]), want, 28UL, "a30e01098bc6dbbf45690f3a7e9e6d0f8bbea2a39e6148008fd05e44"), 1);
        ASSERT_EQ(hasher_kat(Hmac::new(Sha256Ctx::new(), ((const unsigned char*)jefe)[0..4UL]), want, 28UL, "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"), 1);
        ASSERT_EQ(hasher_kat(Hmac::new(Sha512Ctx::new_384(), ((const unsigned char*)jefe)[0..4UL]), want, 28UL, "af45d2e376484031617f78d2b58a6b1b9c7ef464f5a01b47e42ec3736322445e8e2240ca5e69e2c78b3239ecfab21649"), 1);
        ASSERT_EQ(hasher_kat(Hmac::new(Sha512Ctx::new(), ((const unsigned char*)jefe)[0..4UL]), want, 28UL, "164b7a7bfcf819e2e395fbe73b56e0a387bd64222e831fd610270cd7ea2505549758bf75c05a994a6d034f65f8f0e6fdcaeab1a34d4a6b4b636e070a38bce737"), 1);
        unsigned char longKey[131];
        i = 0UL;
        while (i < 131UL) { longKey[i] = (unsigned char)0xaa; i = i + 1UL; }
        const &unsigned char lk = &longKey[0];
        const &unsigned char big = (const &unsigned char)"Test Using Larger Than Block-Size Key - Hash Key First";
        ASSERT_EQ(hasher_kat(Hmac::new(Sha1Ctx::new(), ((const unsigned char*)lk)[0..131UL]), big, 54UL, "90d0dace1c1bdc957339307803160335bde6df2b"), 1);
        ASSERT_EQ(hasher_kat(Hmac::new(Sha256Ctx::new(), ((const unsigned char*)lk)[0..131UL]), big, 54UL, "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"), 1);
        ASSERT_EQ(hasher_kat(Hmac::new(Sha512Ctx::new_384(), ((const unsigned char*)lk)[0..131UL]), big, 54UL, "4ece084485813e9088d2c63a041bc5b44f9ef1012a2b588f3cd11f05033ac4c60c2ef6ab4030fe8296248df163f44952"), 1);
        ASSERT_EQ(hasher_kat(Hmac::new(Sha512Ctx::new(), ((const unsigned char*)lk)[0..131UL]), big, 54UL, "80b24263c7c1a3ebb71493c1dd7be8b49b46d1f41b4aeec1121b013783f8f3526b56d037e05f2598bd0fd2215d6a1e5295e64f73f63f0aec8b915a985d786598"), 1);
    }
}

void t_aes() {
    unsafe {
        unsigned char key[32];
        unsigned char blk[16];
        char hex[40];
        int i = 0;
        while (i < 32) { key[i] = (unsigned char)i; i = i + 1; }
        i = 0;
        while (i < 16) { blk[i] = (unsigned char)(i * 17); i = i + 1; }  // 00 11 22 .. ff
        struct AesCtx c = AesCtx::new_128(key[..]);
        c.encrypt_block(blk[..]);
        to_hex(blk, 16UL, hex);
        ASSERT_STR_EQ(hex, "69c4e0d86a7b0430d8cdb78070b4c55a");
        c.decrypt_block(blk[..]);
        ASSERT_EQ(blk[1], 0x11);
        ASSERT_EQ(blk[15], 0xff);
        struct AesCtx c2 = AesCtx::new_256(key[..]);
        c2.encrypt_block(blk[..]);
        to_hex(blk, 16UL, hex);
        ASSERT_STR_EQ(hex, "8ea2b7ca516745bfeafc49904b496089");
        // CBC round-trip
        unsigned char iv[16];
        unsigned char buf[64];
        i = 0;
        while (i < 16) { iv[i] = (unsigned char)(255 - i); i = i + 1; }
        i = 0;
        while (i < 64) { buf[i] = (unsigned char)(i * 3); i = i + 1; }
        c2.set_iv(iv[..]);
        c2.cbc_encrypt(((const unsigned char*)buf)[0..64UL]);
        ASSERT_NE(buf[0], 0);
        c2.set_iv(iv[..]);
        c2.cbc_decrypt(((const unsigned char*)buf)[0..64UL]);
        int same = 1;
        i = 0;
        while (i < 64) { if (buf[i] != (unsigned char)(i * 3)) same = 0; i = i + 1; }
        ASSERT_TRUE(same);
        c.destroy();
        c2.destroy();
    }
}

void t_rng() {
    unsafe {
        unsigned char seed[32];
        int i = 0;
        while (i < 32) { seed[i] = (unsigned char)i; i = i + 1; }
        struct RngCtx a;
        struct RngCtx b;
        a.init_seed(seed);
        b.init_seed(seed);
        ASSERT_EQ(a.rand32(), b.rand32());
        i = 0;
        int inrange = 1;
        while (i < 1000) { if (a.rand_range(10UL) >= 10UL) inrange = 0; i = i + 1; }
        ASSERT_TRUE(inrange);
        struct RngCtx sys;
        ASSERT_TRUE(sys.init() == 0 || sys.init() == 1);
        // Seeded from the OS: two contexts differ, and a stream isn't constant.
        struct RngCtx s1;
        struct RngCtx s2;
        ASSERT_EQ(s1.init(), 1);
        ASSERT_EQ(s2.init(), 1);
        ASSERT_TRUE(s1.rand64() != s2.rand64());
        unsigned int first = s1.rand32();
        int constant = 1;
        i = 0;
        while (i < 16) { if (s1.rand32() != first) constant = 0; i = i + 1; }
        ASSERT_TRUE(constant == 0);
    }
}

void t_base64() {
    unsafe {
        struct String e = base64_encode(str_bytes("foobar"));
        ASSERT_STR_EQ(e.as_ptr(), "Zm9vYmFy");
        e.free();
        struct String e2 = base64_encode(str_bytes("fo"));
        ASSERT_STR_EQ(e2.as_ptr(), "Zm8=");
        e2.free();
        int ok = 0;
        struct Vec<unsigned char> d = base64_decode("Zm9vYg==", &ok);
        ASSERT_TRUE(ok);
        ASSERT_EQ(d.length(), 4);
        ASSERT_EQ(d.get(3UL).default((unsigned char)0), 'b');
        d.free();
        struct Vec<unsigned char> bad = base64_decode("Zm9v*", &ok);
        ASSERT_FALSE(ok);
        bad.free();
    }
}

void t_url() {
    unsafe {
        struct String e = url_encode("a b&c=d/é");
        ASSERT_STR_EQ(e.as_ptr(), "a%20b%26c%3Dd%2F%C3%A9");
        int ok = 0;
        struct String d = url_decode(e.as_ptr(), &ok);
        ASSERT_TRUE(ok);
        ASSERT_STR_EQ(d.as_ptr(), "a b&c=d/é");
        e.free(); d.free();
    }
}

void t_utf8() {
    unsafe {
        char buf[8];
        ASSERT_EQ(utf8_encode(0x20ACu, (char*)buf), 3);   // €
        ASSERT_EQ((unsigned char)buf[0], 0xE2);
        unsigned int cp = 0u;
        ASSERT_EQ(utf8_decode((const char*)buf, 0UL, &cp), 3);
        ASSERT_EQ(cp, 0x20AC);
        ASSERT_EQ(utf8_len("h€llo😀"), 6);
        ASSERT_TRUE(utf8_is_valid("h€llo"));
        ASSERT_FALSE(utf8_is_valid("\xC3\x28"));
        ASSERT_EQ(utf8_encoded_len(0x1F600u), 4);
        ASSERT_TRUE(ascii_is_valid("plain"));
        ASSERT_FALSE(ascii_is_valid("é"));
        ASSERT_EQ(ascii_to_upper('q'), 'Q');
    }
}

// The generator is ChaCha20 (RFC 8439 block function): seed = key, counter
// from 0, nonce word 14 = 1. Reference values from an independent ChaCha20.
void t_rng_kat() {
    unsafe {
        unsigned char seed[32];
        int i = 0;
        while (i < 32) { seed[i] = (unsigned char)i; i = i + 1; }
        struct RngCtx r;
        r.init_seed(seed);
        test_assert_eq_u(r.rand32(), 0x02F1A42FULL, "chacha20 word 0", __FILE__, __LINE__);
        test_assert_eq_u(r.rand32(), 0x898E8050ULL, "chacha20 word 1", __FILE__, __LINE__);
        test_assert_eq_u(r.rand64(), 0xE06EDF0FE53152A2ULL, "chacha20 words 2-3", __FILE__, __LINE__);
        unsigned char buf[100];
        r.fill(buf[0..100UL]);            // crosses into the second block
        struct RngCtx q;
        q.init_seed(seed);
        unsigned char all[116];
        q.fill(all[0..116UL]);
        ASSERT_EQ(safe_memcmp(buf[..], all[16..116]), 0);
        ASSERT_EQ(r.rand_range(1UL), 0);
    }
}

// A self-signed P-256 certificate: C=KR, O=SafeC Test, CN=www.Example.com,
// SAN DNS:example.com + DNS:*.Example.org, CA:TRUE, serial 0x1234,
// valid 2026-10-21 12:29:05 .. 2036-10-18 12:29:05 UTC.
static const unsigned char kCertDer[485] = {
    0x30U, 0x82U, 0x01U, 0xE1U, 0x30U, 0x82U, 0x01U, 0x88U, 0xA0U, 0x03U, 0x02U, 0x01U, 0x02U, 0x02U, 0x02U, 0x12U,
    0x34U, 0x30U, 0x0AU, 0x06U, 0x08U, 0x2AU, 0x86U, 0x48U, 0xCEU, 0x3DU, 0x04U, 0x03U, 0x02U, 0x30U, 0x3CU, 0x31U,
    0x0BU, 0x30U, 0x09U, 0x06U, 0x03U, 0x55U, 0x04U, 0x06U, 0x13U, 0x02U, 0x4BU, 0x52U, 0x31U, 0x13U, 0x30U, 0x11U,
    0x06U, 0x03U, 0x55U, 0x04U, 0x0AU, 0x0CU, 0x0AU, 0x53U, 0x61U, 0x66U, 0x65U, 0x43U, 0x20U, 0x54U, 0x65U, 0x73U,
    0x74U, 0x31U, 0x18U, 0x30U, 0x16U, 0x06U, 0x03U, 0x55U, 0x04U, 0x03U, 0x0CU, 0x0FU, 0x77U, 0x77U, 0x77U, 0x2EU,
    0x45U, 0x78U, 0x61U, 0x6DU, 0x70U, 0x6CU, 0x65U, 0x2EU, 0x63U, 0x6FU, 0x6DU, 0x30U, 0x1EU, 0x17U, 0x0DU, 0x32U,
    0x36U, 0x31U, 0x30U, 0x32U, 0x31U, 0x31U, 0x32U, 0x32U, 0x39U, 0x30U, 0x35U, 0x5AU, 0x17U, 0x0DU, 0x33U, 0x36U,
    0x31U, 0x30U, 0x31U, 0x38U, 0x31U, 0x32U, 0x32U, 0x39U, 0x30U, 0x35U, 0x5AU, 0x30U, 0x3CU, 0x31U, 0x0BU, 0x30U,
    0x09U, 0x06U, 0x03U, 0x55U, 0x04U, 0x06U, 0x13U, 0x02U, 0x4BU, 0x52U, 0x31U, 0x13U, 0x30U, 0x11U, 0x06U, 0x03U,
    0x55U, 0x04U, 0x0AU, 0x0CU, 0x0AU, 0x53U, 0x61U, 0x66U, 0x65U, 0x43U, 0x20U, 0x54U, 0x65U, 0x73U, 0x74U, 0x31U,
    0x18U, 0x30U, 0x16U, 0x06U, 0x03U, 0x55U, 0x04U, 0x03U, 0x0CU, 0x0FU, 0x77U, 0x77U, 0x77U, 0x2EU, 0x45U, 0x78U,
    0x61U, 0x6DU, 0x70U, 0x6CU, 0x65U, 0x2EU, 0x63U, 0x6FU, 0x6DU, 0x30U, 0x59U, 0x30U, 0x13U, 0x06U, 0x07U, 0x2AU,
    0x86U, 0x48U, 0xCEU, 0x3DU, 0x02U, 0x01U, 0x06U, 0x08U, 0x2AU, 0x86U, 0x48U, 0xCEU, 0x3DU, 0x03U, 0x01U, 0x07U,
    0x03U, 0x42U, 0x00U, 0x04U, 0x75U, 0xA5U, 0xD0U, 0x54U, 0x98U, 0x93U, 0x28U, 0xF8U, 0xC0U, 0x02U, 0x6FU, 0xB0U,
    0x78U, 0x13U, 0x38U, 0x9BU, 0xEEU, 0xE6U, 0xF5U, 0x08U, 0xE4U, 0x0AU, 0x9CU, 0xFBU, 0x54U, 0xB0U, 0x20U, 0x26U,
    0x31U, 0xD0U, 0x44U, 0x6EU, 0xA1U, 0x62U, 0xE2U, 0x1BU, 0xFDU, 0x96U, 0xBDU, 0x84U, 0x2FU, 0xF4U, 0xA6U, 0xD3U,
    0xB0U, 0x9BU, 0x2FU, 0xC1U, 0x8FU, 0xFEU, 0xBEU, 0x44U, 0x5CU, 0x29U, 0x86U, 0x6FU, 0xD2U, 0x0DU, 0xC9U, 0x66U,
    0x55U, 0xE4U, 0x19U, 0x86U, 0xA3U, 0x7AU, 0x30U, 0x78U, 0x30U, 0x1DU, 0x06U, 0x03U, 0x55U, 0x1DU, 0x0EU, 0x04U,
    0x16U, 0x04U, 0x14U, 0xFDU, 0x9DU, 0x77U, 0x5CU, 0xC6U, 0x81U, 0x3AU, 0x2CU, 0xB6U, 0x9BU, 0x96U, 0xE1U, 0xC5U,
    0x64U, 0xC5U, 0x97U, 0xFBU, 0x5DU, 0xF9U, 0xCFU, 0x30U, 0x1FU, 0x06U, 0x03U, 0x55U, 0x1DU, 0x23U, 0x04U, 0x18U,
    0x30U, 0x16U, 0x80U, 0x14U, 0xFDU, 0x9DU, 0x77U, 0x5CU, 0xC6U, 0x81U, 0x3AU, 0x2CU, 0xB6U, 0x9BU, 0x96U, 0xE1U,
    0xC5U, 0x64U, 0xC5U, 0x97U, 0xFBU, 0x5DU, 0xF9U, 0xCFU, 0x30U, 0x25U, 0x06U, 0x03U, 0x55U, 0x1DU, 0x11U, 0x04U,
    0x1EU, 0x30U, 0x1CU, 0x82U, 0x0BU, 0x65U, 0x78U, 0x61U, 0x6DU, 0x70U, 0x6CU, 0x65U, 0x2EU, 0x63U, 0x6FU, 0x6DU,
    0x82U, 0x0DU, 0x2AU, 0x2EU, 0x45U, 0x78U, 0x61U, 0x6DU, 0x70U, 0x6CU, 0x65U, 0x2EU, 0x6FU, 0x72U, 0x67U, 0x30U,
    0x0FU, 0x06U, 0x03U, 0x55U, 0x1DU, 0x13U, 0x01U, 0x01U, 0xFFU, 0x04U, 0x05U, 0x30U, 0x03U, 0x01U, 0x01U, 0xFFU,
    0x30U, 0x0AU, 0x06U, 0x08U, 0x2AU, 0x86U, 0x48U, 0xCEU, 0x3DU, 0x04U, 0x03U, 0x02U, 0x03U, 0x47U, 0x00U, 0x30U,
    0x44U, 0x02U, 0x20U, 0x6DU, 0x53U, 0xF9U, 0xF8U, 0x85U, 0x5EU, 0xF3U, 0x84U, 0xADU, 0x63U, 0x5BU, 0xF1U, 0x0AU,
    0x56U, 0xD1U, 0x16U, 0xC0U, 0x56U, 0x17U, 0xDFU, 0xD7U, 0x84U, 0x72U, 0xB5U, 0x41U, 0x7AU, 0x5EU, 0xC6U, 0xB5U,
    0x8BU, 0x80U, 0xB6U, 0x02U, 0x20U, 0x6BU, 0x98U, 0x05U, 0x76U, 0x77U, 0xFEU, 0x9FU, 0x75U, 0x88U, 0x4CU, 0x98U,
    0xFAU, 0x0CU, 0xC1U, 0xF9U, 0xDDU, 0xA7U, 0x6FU, 0x73U, 0x92U, 0x6FU, 0xC4U, 0xA5U, 0xD0U, 0xF6U, 0xE9U, 0x79U,
    0xBDU, 0xB3U, 0x5AU, 0x1AU, 0xE8U
};

void t_x509() {
    unsafe {
        struct X509Cert c;
        ASSERT_EQ(c.init(kCertDer[0..485UL]), 0);
        ASSERT_STR_EQ((const char*)c.subject.cn, "www.Example.com");
        ASSERT_STR_EQ((const char*)c.subject.o, "SafeC Test");
        ASSERT_STR_EQ((const char*)c.subject.c, "KR");
        ASSERT_STR_EQ((const char*)c.issuer.cn, "www.Example.com");
        ASSERT_EQ(c.serial_len, 2);
        ASSERT_EQ(c.serial[0], 0x12);
        ASSERT_EQ(c.serial[1], 0x34);
        test_assert_eq_u(c.validity.not_before, 1792585745ULL, "not_before", __FILE__, __LINE__);
        test_assert_eq_u(c.validity.not_after, 2107945745ULL, "not_after", __FILE__, __LINE__);
        ASSERT_TRUE(c.is_valid_at(1900000000ULL));
        ASSERT_FALSE(c.is_valid_at(1700000000ULL));
        ASSERT_TRUE(c.is_ca_cert());
        ASSERT_EQ(c.san.count, 2);
        ASSERT_TRUE(c.pubkey_len > 60UL);
        // DNS names match case-insensitively; '*.' covers one label; the CN
        // is ignored because the certificate has DNS SANs (RFC 6125).
        ASSERT_TRUE(c.matches_hostname("example.com"));
        ASSERT_TRUE(c.matches_hostname("EXAMPLE.com"));
        ASSERT_TRUE(c.matches_hostname("api.example.org"));
        ASSERT_FALSE(c.matches_hostname("a.b.example.org"));
        ASSERT_FALSE(c.matches_hostname("www.example.com"));
        ASSERT_TRUE(c.verify_chain(&c));
        ASSERT_TRUE(c.init(kCertDer[0..40UL]) != 0);   // truncated
    }
}

// AES-256-GCM, McGrew & Viega test case 16 (60-byte message, 20-byte AAD).
void t_aes_gcm() {
    unsafe {
        unsigned char key[32];
        unsigned char iv[12];
        unsigned char aad[20];
        unsigned char msg[60];
        unsigned char tag[16];
        char hex[160];
        hex_to_bytes("feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308", &key[0]);
        hex_to_bytes("cafebabefacedbaddecaf888", &iv[0]);
        hex_to_bytes("feedfacedeadbeeffeedfacedeadbeefabaddad2", &aad[0]);
        hex_to_bytes("d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39", &msg[0]);
        struct AesCtx aes = AesCtx::new_256(key[..]);
        aes.gcm_encrypt(iv[..], ((const unsigned char*)((const ?&unsigned char)&aad[0]))[0..20UL], msg[0..60UL], tag[..]);
        to_hex(msg, 60UL, hex);
        ASSERT_STR_EQ(hex, "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662");
        to_hex(tag, 16UL, hex);
        ASSERT_STR_EQ(hex, "76fc6ece0f4e1768cddf8853bb2d551b");
        ASSERT_TRUE(aes.gcm_decrypt(iv[..], ((const unsigned char*)((const ?&unsigned char)&aad[0]))[0..20UL], msg[0..60UL], tag[..]));
        ASSERT_EQ(msg[0], 0xd9);
        aes.gcm_encrypt(iv[..], ((const unsigned char*)((const ?&unsigned char)&aad[0]))[0..20UL], msg[0..60UL], tag[..]);
        aad[3] = aad[3] ^ (unsigned char)1;                 // tampered AAD
        ASSERT_FALSE(aes.gcm_decrypt(iv[..], ((const unsigned char*)((const ?&unsigned char)&aad[0]))[0..20UL], msg[0..60UL], tag[..]));
        ASSERT_EQ(msg[0], 0x52);                            // left as ciphertext
        aes.destroy();
        // Empty message and AAD, zero key and IV (test case 13).
        struct AesCtx z;
        unsigned char zk[32];
        int i = 0;
        while (i < 32) { zk[i] = (unsigned char)0; i = i + 1; }
        z = AesCtx::new_256(zk[..]);
        z.gcm_encrypt(zk[..], str_bytes(""), msg[0..0UL], tag[..]);
        to_hex(tag, 16UL, hex);
        ASSERT_STR_EQ(hex, "530f8afbc74536b9a963b4f1c4cb738b");
    }
}

// "secret data" as application data, sequence 0, sealed by an independent
// AES-GCM implementation (Python cryptography) with the keys used below.
static const unsigned char kTlsRecord[33] = {
    0x17U, 0x03U, 0x03U, 0x00U, 0x1CU, 0x05U, 0xB9U, 0xDAU, 0xA5U, 0x8BU, 0x79U, 0x30U, 0xAAU, 0x57U, 0xB8U, 0x74U,
    0x3FU, 0x7EU, 0x02U, 0xC2U, 0x33U, 0xA9U, 0x8DU, 0xA8U, 0x22U, 0x90U, 0x7DU, 0x06U, 0xEAU, 0xD0U, 0x28U, 0x95U,
    0xA2U
};

void t_tls_record() {
    unsafe {
        struct TlsSession a = TlsSession::new();
        struct TlsSession b = TlsSession::new();
        unsigned char out[256];
        unsigned char got[256];
        unsigned char ct = (unsigned char)0;
        // Plaintext records before the handshake.
        unsigned long n = a.encode_record((unsigned char)TLS_HANDSHAKE, str_bytes("hello"), out[0..256UL]);
        ASSERT_EQ(n, 10);
        ASSERT_EQ(out[0], TLS_HANDSHAKE);
        ASSERT_EQ(out[4], 5);
        ASSERT_EQ(b.decode_record(out[0..n], &ct, got[0..256UL]), 5);
        ASSERT_EQ(ct, TLS_HANDSHAKE);
        ASSERT_FALSE(a.is_established());

        unsigned char k1[32];
        unsigned char k2[32];
        unsigned char iv1[12];
        unsigned char iv2[12];
        int i = 0;
        while (i < 32) { k1[i] = (unsigned char)i; k2[i] = (unsigned char)(100 + i); i = i + 1; }
        i = 0;
        while (i < 12) { iv1[i] = (unsigned char)(7 * i); iv2[i] = (unsigned char)(3 + i); i = i + 1; }
        a.install_keys(k1, iv1, k2, iv2);
        b.install_keys(k2, iv2, k1, iv1);
        ASSERT_TRUE(a.is_established());          // installing keys turns protection on

        // A record that doesn't fit is refused without consuming a sequence number.
        ASSERT_EQ(a.encode_record((unsigned char)TLS_APPLICATION_DATA, str_bytes("secret data"), out[0..8UL]), 0);
        ASSERT_EQ(a.write_seq, 0);
        int round = 0;
        while (round < 3) {
            n = a.encode_record((unsigned char)TLS_APPLICATION_DATA, str_bytes("secret data"), out[0..256UL]);
            ASSERT_EQ(n, 5UL + 11UL + 1UL + 16UL);
            if (round == 0) { ASSERT_EQ(safe_memcmp(out[0..33], kTlsRecord[0..33]), 0); }   // RFC 8446 wire format
            ASSERT_EQ(out[0], TLS_APPLICATION_DATA);
            ASSERT_EQ(b.decode_record(out[0..n], &ct, got[0..256UL]), 11);
            ASSERT_EQ(ct, TLS_APPLICATION_DATA);
            []const unsigned char want = str_bytes("secret data");
            ASSERT_EQ(safe_memcmp(got[0..11], want), 0);
            round = round + 1;
        }
        // A tampered record fails authentication and doesn't advance read_seq.
        n = a.encode_record((unsigned char)TLS_HANDSHAKE, str_bytes("hs"), out[0..256UL]);
        out[7] = out[7] ^ (unsigned char)0x40;
        ASSERT_EQ(b.decode_record(out[0..n], &ct, got[0..256UL]), 0);
        ASSERT_EQ(b.read_seq, 3);
        out[7] = out[7] ^ (unsigned char)0x40;
        ASSERT_EQ(b.decode_record(out[0..n], &ct, got[0..256UL]), 2);
        ASSERT_EQ(ct, TLS_HANDSHAKE);                     // inner type, outer is application_data
        ASSERT_EQ(out[0], TLS_APPLICATION_DATA);
        n = a.build_alert((unsigned char)TLS_ALERT_FATAL, (unsigned char)TLS_ALERT_CLOSE_NOTIFY, out[0..256UL]);
        ASSERT_EQ(b.decode_record(out[0..n], &ct, got[0..256UL]), 2);
        ASSERT_EQ(ct, TLS_ALERT);
        ASSERT_EQ(got[0], TLS_ALERT_FATAL);
        ASSERT_EQ(b.decode_record(out[0..4UL], &ct, got[0..256UL]), 0);   // short header
        unsigned char nonce[12];
        a.compute_nonce(nonce, 1);
        ASSERT_EQ(nonce[11], iv1[11] ^ 5);   // write_seq is 5 now

        struct TlsTranscript tr;
        tr.hash = Sha256Ctx::new();
        tr.update(str_bytes("abc"));
        unsigned char d[32];
        char hex[80];
        tr.finish(d[..]);
        to_hex(d, 32UL, hex);
        ASSERT_STR_EQ(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
}

void t_crypto_misc() {
    unsafe {
        unsigned char d[64];
        char hex[160];
        struct Sha1Ctx s1 = Sha1Ctx::new();
        s1.update(str_bytes("a"));
        s1.update(str_bytes("bc"));
        s1.finish(d[..]);
        to_hex(d, 20UL, hex);
        ASSERT_STR_EQ(hex, "a9993e364706816aba3e25717850c26c9cd0d89d");
        struct Sha512Ctx s5 = Sha512Ctx::new();
        s5.update(str_bytes("ab"));
        s5.update(str_bytes("c"));
        s5.finish(d[..]);
        to_hex(d, 64UL, hex);
        ASSERT_STR_EQ(hex, "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f");

        struct SecureAllocator sa = SecureAllocator::new(32UL, 4UL);
        ASSERT_EQ(sa.available(), 4);
        ?&heap unsigned char p = sa.alloc(32UL, 8UL);
        ASSERT_FALSE(p.is_null());
        match (p) { case null: { } case some(r): { unsafe { safe_memset(((unsigned char*)r)[0..32], 0x5A); } } }
        ASSERT_EQ(sa.available(), 3);
        sa.dealloc(p);
        ASSERT_EQ(sa.available(), 4);
        ASSERT_TRUE(sa.alloc(33UL, 1UL).is_null());     // bigger than a slot
        ASSERT_TRUE(sa.alloc(8UL, 16UL).is_null());     // alignment above 8
        // Allocator conformance: the typed helpers work over it, and a slot
        // comes back zeroed after dealloc_t.
        ?&heap unsigned long long w = alloc_t(&sa);
        ASSERT_FALSE(w.is_null());
        match (w) {
            case null: { }
            case some(x): {
                *x = 0xDEADBEEFULL;
                unsigned long long* raw = (unsigned long long*)x;
                raw[1] = 0x77ULL; raw[3] = 0x77ULL;     // the rest of the 32-byte slot
                dealloc_t(&sa, x);
                ASSERT_EQ(raw[1], 0);                   // zeroed past the free-list link
                ASSERT_EQ(raw[3], 0);
            }
        }
        ASSERT_EQ(sa.available(), 4);
        sa.destroy();

        ASSERT_TRUE(ascii_is_alpha('g'));
        ASSERT_FALSE(ascii_is_alpha('1'));
        ASSERT_TRUE(ascii_is_alnum('1'));
        ASSERT_TRUE(ascii_is_upper('G'));
        ASSERT_FALSE(ascii_is_upper('g'));
        ASSERT_TRUE(ascii_is_print('~'));
        ASSERT_FALSE(ascii_is_print((char)9));
        ASSERT_EQ(ascii_to_lower('Q'), 'q');
        ASSERT_TRUE(ascii_is_valid_bytes(str_bytes("ab\xC3")[0..2]));
        ASSERT_FALSE(ascii_is_valid_bytes(str_bytes("ab\xC3")));
    }
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("sha1", t_sha1);
        t.add("sha256", t_sha256);
        t.add("sha512/384", t_sha512);
        t.add("hmac-sha256", t_hmac);
        t.add("Hasher: every hash and Hmac<H> through one generic check", t_hasher_generic);
        t.add("aes128/256", t_aes);
        t.add("rng", t_rng);
        t.add("base64", t_base64);
        t.add("url", t_url);
        t.add("utf8/ascii", t_utf8);
        t.add("rng chacha20 vectors", t_rng_kat);
        t.add("x509", t_x509);
        t.add("aes-256-gcm vectors", t_aes_gcm);
        t.add("tls record layer", t_tls_record);
        t.add("sha streaming/secure_alloc/ascii", t_crypto_misc);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
