// std::crypto (sha1/sha256/sha512/hmac/aes/rng) + std::encoding (base64/url/utf8/ascii)
// Known-answer vectors: FIPS 180-4 "abc", RFC 4231 case 2, FIPS-197 Appendix C.
#include <std/test/test.h>
#include <std/mem.h>
#include <std/str.h>
#include <std/crypto/sha1.h>
#include <std/crypto/sha256.h>
#include <std/crypto/sha512.h>
#include <std/crypto/hmac.h>
#include <std/crypto/aes.h>
#include <std/crypto/rng.h>
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

void t_sha1() {
    unsafe {
        unsigned char d[20];
        char hex[64];
        sha1((const unsigned char*)"abc", 3UL, d);
        to_hex(d, 20UL, hex);
        ASSERT_STR_EQ(hex, "a9993e364706816aba3e25717850c26c9cd0d89d");
    }
}

void t_sha256() {
    unsafe {
        unsigned char d[32];
        char hex[80];
        sha256((const unsigned char*)"abc", 3UL, d);
        to_hex(d, 32UL, hex);
        ASSERT_STR_EQ(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        // Streaming in odd-size chunks must equal one-shot (448-bit message, FIPS 180-4 B.2).
        const &char msg = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        struct Sha256Ctx c = sha256_init();
        c.update((const unsigned char*)&msg[0], 5UL);
        c.update((const unsigned char*)&msg[5], 51UL);
        c.finish(d);
        to_hex(d, 32UL, hex);
        ASSERT_STR_EQ(hex, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
        // Empty input
        sha256((const unsigned char*)"", 0UL, d);
        to_hex(d, 32UL, hex);
        ASSERT_STR_EQ(hex, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }
}

void t_sha512() {
    unsafe {
        unsigned char d[64];
        char hex[160];
        sha512((const unsigned char*)"abc", 3UL, d);
        to_hex(d, 64UL, hex);
        ASSERT_STR_EQ(hex, "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f");
        sha384((const unsigned char*)"abc", 3UL, d);
        to_hex(d, 48UL, hex);
        ASSERT_STR_EQ(hex, "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7");
    }
}

void t_hmac() {
    unsafe {
        unsigned char d[32];
        char hex[80];
        hmac_sha256((const unsigned char*)"Jefe", 4UL,
                    (const unsigned char*)"what do ya want for nothing?", 28UL, (unsigned char*)d);
        to_hex(d, 32UL, hex);
        ASSERT_STR_EQ(hex, "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
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
        struct AesCtx c = aes128_init(key);
        c.encrypt_block(blk);
        to_hex(blk, 16UL, hex);
        ASSERT_STR_EQ(hex, "69c4e0d86a7b0430d8cdb78070b4c55a");
        c.decrypt_block(blk);
        ASSERT_EQ(blk[1], 0x11);
        ASSERT_EQ(blk[15], 0xff);
        struct AesCtx c2 = aes256_init(key);
        c2.encrypt_block(blk);
        to_hex(blk, 16UL, hex);
        ASSERT_STR_EQ(hex, "8ea2b7ca516745bfeafc49904b496089");
        // CBC round-trip
        unsigned char iv[16];
        unsigned char buf[64];
        i = 0;
        while (i < 16) { iv[i] = (unsigned char)(255 - i); i = i + 1; }
        i = 0;
        while (i < 64) { buf[i] = (unsigned char)(i * 3); i = i + 1; }
        c2.set_iv(iv);
        c2.cbc_encrypt(buf, 64UL);
        ASSERT_NE(buf[0], 0);
        c2.set_iv(iv);
        c2.cbc_decrypt(buf, 64UL);
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
        rng_init_seed(&a, seed);
        rng_init_seed(&b, seed);
        ASSERT_EQ(a.rand32(), b.rand32());
        i = 0;
        int inrange = 1;
        while (i < 1000) { if (a.rand_range(10UL) >= 10UL) inrange = 0; i = i + 1; }
        ASSERT_TRUE(inrange);
        struct RngCtx sys;
        ASSERT_TRUE(rng_init(&sys) == 0 || rng_init(&sys) == 1);
    }
}

void t_base64() {
    unsafe {
        struct String e = base64_encode((const unsigned char*)"foobar", 6UL);
        ASSERT_STR_EQ(e.as_ptr(), "Zm9vYmFy");
        e.free();
        struct String e2 = base64_encode((const unsigned char*)"fo", 2UL);
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

int main() {
    struct TestSuite t = test_suite_init();
    unsafe {
        t.add("sha1", t_sha1);
        t.add("sha256", t_sha256);
        t.add("sha512/384", t_sha512);
        t.add("hmac-sha256", t_hmac);
        t.add("aes128/256", t_aes);
        t.add("rng", t_rng);
        t.add("base64", t_base64);
        t.add("url", t_url);
        t.add("utf8/ascii", t_utf8);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
