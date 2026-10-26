// std::crypto public-key and AEAD primitives: ChaCha20-Poly1305 (RFC 8439
// §2.3.2, §2.5.2, §2.8.2), X25519 (RFC 7748 §5.2, §6.1, the 1000-iteration
// vector, low-order rejection), HKDF-SHA256 (RFC 5869 cases 1 and 3),
// Montgomery modular exponentiation (against Python, 1032/2048/4093-bit
// moduli, public and constant-time paths), RSA PKCS#1 v1.5 / PSS (OpenSSL
// signatures, sign + verify), P-256/P-384 ECDSA + ECDH (pyca/cryptography
// vectors, SEC1/PKCS#8 keys, invalid points, RFC 6979 A.2.5).
#include <std/test/test.h>
#include <std/str.h>
#include <std/crypto/chacha20.h>
#include <std/crypto/ct.h>
#include <std/crypto/ec.h>
#include <std/crypto/rsa.h>
#include <std/crypto/rng.h>
#include <std/crypto/digest.h>
#include <std/crypto/bignum.h>
#include <std/crypto/hkdf.h>
#include <std/crypto/sha256.h>
#include <std/crypto/x25519.h>
extern int printf(const char* f, ...);

static int hexv(char c) {
    if (c >= '0' && c <= '9') return (int)(c - '0');
    if (c >= 'A' && c <= 'F') return (int)(c - 'A') + 10;
    return (int)(c - 'a') + 10;
}
static unsigned long hb(const &char h, &unsigned char out) {
    unsafe {
        unsigned long i = 0UL;
        while (h[2UL * i] != (char)0) {
            out[i] = (unsigned char)(hexv(h[2UL * i]) * 16 + hexv(h[2UL * i + 1UL]));
            i = i + 1UL;
        }
        return i;
    }
}
static void hx(const &unsigned char d, unsigned long n, &char out) {
    unsafe {
        const &char digits = "0123456789abcdef";
        unsigned long i = 0UL;
        while (i < n) { out[i*2UL] = digits[d[i] >> 4]; out[i*2UL+1UL] = digits[d[i] & 15]; i = i + 1UL; }
        out[n*2UL] = (char)0;
    }
}

void t_chacha() {
    unsafe {
        unsigned char key[32]; unsigned char nonce[12]; unsigned char blk[64]; char h[300];
        hb("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", key);
        hb("000000090000004a00000000", nonce);
        chacha20_block(key[..], nonce[..], 1U, blk[..]);
        hx(blk, 64UL, h);
        ASSERT_STR_EQ(h, "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4ed2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e");
        // poly1305 2.5.2
        unsigned char pk[32]; unsigned char tag[16];
        hb("85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b", pk);
        const &char msg = "Cryptographic Forum Research Group";
        poly1305(pk[..], ((const unsigned char*)msg)[0..34UL], tag[..]);
        hx(tag, 16UL, h);
        ASSERT_STR_EQ(h, "a8061dc1305136c6c22b8baf0c0127a9");
        // AEAD 2.8.2
        unsigned char k2[32]; unsigned char n2[12]; unsigned char aad[12]; unsigned char pt[200];
        hb("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f", k2);
        hb("070000004041424344454647", n2);
        hb("50515253c0c1c2c3c4c5c6c7", aad);
        const &char p = "Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
        unsigned long pl = str_len(p);
        unsigned long i = 0UL;
        while (i < pl) { pt[i] = (unsigned char)p[i]; i = i + 1UL; }
        chacha20_poly1305_seal(k2[..], n2[..], ((const unsigned char*)aad)[0..12UL], ((const unsigned char*)pt)[0..pl], tag[..]);
        hx(tag, 16UL, h);
        ASSERT_STR_EQ(h, "1ae10b594f09e26a7e902ecbd0600691");
        hx(pt, 16UL, h);
        ASSERT_STR_EQ(h, "d31a8d34648e60db7b86afbc53ef7ec2");
        ASSERT_EQ(chacha20_poly1305_open(k2[..], n2[..], ((const unsigned char*)aad)[0..12UL], ((const unsigned char*)pt)[0..pl], tag[..]), 1);
        ASSERT_EQ(pt[0], 'L');
        chacha20_poly1305_seal(k2[..], n2[..], ((const unsigned char*)aad)[0..12UL], ((const unsigned char*)pt)[0..pl], tag[..]);
        tag[3] = tag[3] ^ (unsigned char)1;
        ASSERT_EQ(chacha20_poly1305_open(k2[..], n2[..], ((const unsigned char*)aad)[0..12UL], ((const unsigned char*)pt)[0..pl], tag[..]), 0);
    }
}

void t_x25519() {
    unsafe {
        unsigned char k[32]; unsigned char u[32]; unsigned char o[32]; char h[80];
        hb("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4", k);
        hb("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c", u);
        ASSERT_EQ(x25519(o[..], k[..], u[..]), 1);
        hx(o, 32UL, h);
        ASSERT_STR_EQ(h, "c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552");
        unsigned char a[32]; unsigned char ap[32]; unsigned char b[32]; unsigned char bp[32];
        hb("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a", a);
        hb("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb", b);
        x25519_public(ap[..], a[..]); hx(ap, 32UL, h);
        ASSERT_STR_EQ(h, "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a");
        x25519_public(bp[..], b[..]);
        x25519(o[..], a[..], bp[..]); hx(o, 32UL, h);
        ASSERT_STR_EQ(h, "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742");
        // 1000 iterations
        unsigned char kk[32]; unsigned char uu[32]; unsigned char t[32];
        int i = 0;
        while (i < 32) { kk[i] = (unsigned char)0; uu[i] = (unsigned char)0; i = i + 1; }
        kk[0] = (unsigned char)9; uu[0] = (unsigned char)9;
        i = 0;
        while (i < 1000) {
            x25519(t[..], kk[..], uu[..]);
            int j = 0; while (j < 32) { uu[j] = kk[j]; kk[j] = t[j]; j = j + 1; }
            i = i + 1;
        }
        hx(kk, 32UL, h);
        ASSERT_STR_EQ(h, "684cf59ba83309552800ef566f2f4d3c1c3887c49360e3875f2eb94d99532c51");
        // low-order point -> 0
        int z = 0; while (z < 32) { u[z] = (unsigned char)0; z = z + 1; }
        ASSERT_EQ(x25519(o[..], k[..], u[..]), 0);
    }
}

void t_hkdf() {
    unsafe {
        unsigned char ikm[22]; unsigned char salt[13]; unsigned char info[10]; unsigned char prk[32]; unsigned char okm[42]; char h[100];
        hb("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b", ikm);
        hb("000102030405060708090a0b0c", salt);
        hb("f0f1f2f3f4f5f6f7f8f9", info);
        hkdf_extract(Sha256Ctx::new(), ((const unsigned char*)salt)[0..13UL], ((const unsigned char*)ikm)[0..22UL], prk[..]);
        hx(prk, 32UL, h);
        ASSERT_STR_EQ(h, "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5");
        ASSERT_EQ(hkdf_expand(Sha256Ctx::new(), ((const unsigned char*)prk)[0..32UL], ((const unsigned char*)info)[0..10UL], ((const unsigned char*)okm)[0..42UL]), 1);
        hx(okm, 42UL, h);
        ASSERT_STR_EQ(h, "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865");
        // case 3: no salt/info
        hkdf_extract(Sha256Ctx::new(), str_bytes(""), ((const unsigned char*)ikm)[0..22UL], prk[..]);
        hkdf_expand(Sha256Ctx::new(), ((const unsigned char*)prk)[0..32UL], str_bytes(""), ((const unsigned char*)okm)[0..42UL]);
        hx(okm, 42UL, h);
        ASSERT_STR_EQ(h, "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8");
    }
}

void t_bn() {
    unsafe {
        unsigned char mod[600]; unsigned char base[600]; unsigned char ex[40]; unsigned char out[600]; char h[1200];
        unsigned char e65537[3]; e65537[0] = (unsigned char)1; e65537[1] = (unsigned char)0; e65537[2] = (unsigned char)1;
        struct BnMont mm; unsigned long L = 0UL;

        L = hb("833fd4235992edcf451a1afe878b33e968617959ce3f1f65a8de5271007814e8a25f2dd97f1cfb10f62827688de6a16a3b0d464138a62332553fc1ea36f17fd374c6a5387777330bdbd7210dff076ce2ef87b0b125ec1d7da0a6eb8c9ebd69fe29d76d4330f1446beab0c11fdecb91ce375bc8fbbcbde5c0994164d8399f767c45", mod); hb("504a0fe75d2a9eba0cdf561d802a759159fb7ff337f5cae3bf3729c619c60a3cab359eeefb015c33b2df1461aaf8eb18b90074513021da8978206f5c6671e0c07e9e115e4b9e30691c238642ea126a1e48cc11d357c30d8b7628dbd25e63b229f1c4069545de11cc9dea959c212e9c82b1478c281d687c966c377b9aa2bb2edb20", base); hb("2e81d66d346c6e2ba02fdaa1ad864c44e049548e8a0a8c9632ea6928f6236bf2", ex);
        ASSERT_EQ(mm.init(((const unsigned char*)mod)[0..L]), 1);
        ASSERT_EQ(bn_mod_exp_public(&mm, ((const unsigned char*)base)[0..L], ((const unsigned char*)ex)[0..32UL], out[..]), 1);
        hx(out, L, h); ASSERT_STR_EQ(h, "50306ca739f53f0dd31e6dce4a01f54248e4c91cd35df41a2fe4c5de81becb5763e6c6afffa3d5d0b947e5b8ecb5030c388c479f47b4140cd4295f68b317c1b7cb84e432622b9ebdd1a1f43f91125d7cf85c4c99f14dbaa28beeaea1ecf56e11003e3933c78457e2a880aada1d6e32263055bc8998681cb8bc2f069bf2ddec5335");
        ASSERT_EQ(bn_mod_exp_secret(&mm, ((const unsigned char*)base)[0..L], ((const unsigned char*)ex)[0..32UL], out[..]), 1);
        hx(out, L, h); ASSERT_STR_EQ(h, "50306ca739f53f0dd31e6dce4a01f54248e4c91cd35df41a2fe4c5de81becb5763e6c6afffa3d5d0b947e5b8ecb5030c388c479f47b4140cd4295f68b317c1b7cb84e432622b9ebdd1a1f43f91125d7cf85c4c99f14dbaa28beeaea1ecf56e11003e3933c78457e2a880aada1d6e32263055bc8998681cb8bc2f069bf2ddec5335");
        ASSERT_EQ(bn_mod_exp_public(&mm, ((const unsigned char*)base)[0..L], ((const unsigned char*)e65537)[0..3UL], out[..]), 1);
        hx(out, L, h); ASSERT_STR_EQ(h, "039acf4431a5512e2d9b1c02f2f99219e529c67a5e744f10a5b173fa3ab87505514e81a0d1ab84e818f736356dce604230c4c41fd1292159f4246810826b00d63de0a426834b155c2e43439bfdbc7f10765d7ed595687bcadeedf243fafb3c743a02b1d1561532a2c2f71121ede51624780a669a33595489ed4f9541ed51075d35");

        L = hb("9faba8272e50bd4eb52fa53c0bf64ef773ec28d00252f615d75b1e249419cf4d60597bdc5dbe44096b384309c9a937a68c8f95ef04a012e8677fd139d84a1d3a5b8e8fb2bff29101f3001cee05da8467f06313fff9a01fe8419521fe0e979cf32d1634b4b465325278f845f57b3120df2f4d4c8650d7d13fb24891917b121dc54e5a3a26d18a669a5af84e6b4f59672710e6d8e6568068b9b52a43abad8d194a9892139600ddb74d960d5a8f9a656aafd14125844d25deb354f46a6910acff0043892dfc254cb864ef901b932a7c18806a3753915c76f18a0585a01c4c7d6df0621aef57e4cc4132f7108e96f770c2263266aa3bb0cde917f7f35634f0e3cd97", mod); hb("99c4258bfc98500014b9adb50c6dea3db85a5cd22959fea37bd9e8a1ff297d0e4f2e84fcb06dbee0b89c4e56261c374ba07657d61404ab1ecc7c6d812d6f2efc4e613a365119cdccaf9b74f84ffcbf4217921e6c8b8e8f4eb3de08f9ec9837044692ba035707967025f02628eb07c30d5cd5061c9c5f319e834c1b69573ac59a355f2af41757905efa749692f21ff5eb6ed78f5d0960afe94bbdbb01dc14ed575e0730b3cc170c31c7eec61bbf9703c096fabb7b1ba95a54767d5274c68deb5d403a960a8652dbd0e488b6c85ad3ba328332f05a5829681876531737f129c8c6d1ca4dc4edfde4163eff8b3fc177f1131e782196324f3e81f453324ef486ab73", base); hb("58056ed0980dc6ffbd953dc23cc217790825c7cc67e84707ec717f158895787f", ex);
        ASSERT_EQ(mm.init(((const unsigned char*)mod)[0..L]), 1);
        ASSERT_EQ(bn_mod_exp_public(&mm, ((const unsigned char*)base)[0..L], ((const unsigned char*)ex)[0..32UL], out[..]), 1);
        hx(out, L, h); ASSERT_STR_EQ(h, "7ecfb75a5182d304cd3b5fe294f5c820a8cc5fa9c0d173d6dafc9c17e919ff63dc1618373ddf145d8a7b59101ad43ad73015f3c3312e72d8f58a66c24ff6ac7e3724064f2ccdc781473bd0dfa9f4553685879fb04897e61268c266e89f276d5964c843a4b17fa8472537b7f8f45dff9134bf17ddbaf1ca0bd88da99f5ec9722c2a2894e572a41ea462dfb22af240849d1c60b2c7a10e72dda4164e083e1c9b5ff928539cef7382f2777eba9436672e6c7c30abad1fdd4452be6ff566e98db950d76b6e08cc6a80351fe78ad00a19ddd63f342b4d40dd28cb41d7fe51ccab1075c4114150f5edf91763352026cc6130d8b3c2b1fc4b677a16029dc113b7bf3c41");
        ASSERT_EQ(bn_mod_exp_secret(&mm, ((const unsigned char*)base)[0..L], ((const unsigned char*)ex)[0..32UL], out[..]), 1);
        hx(out, L, h); ASSERT_STR_EQ(h, "7ecfb75a5182d304cd3b5fe294f5c820a8cc5fa9c0d173d6dafc9c17e919ff63dc1618373ddf145d8a7b59101ad43ad73015f3c3312e72d8f58a66c24ff6ac7e3724064f2ccdc781473bd0dfa9f4553685879fb04897e61268c266e89f276d5964c843a4b17fa8472537b7f8f45dff9134bf17ddbaf1ca0bd88da99f5ec9722c2a2894e572a41ea462dfb22af240849d1c60b2c7a10e72dda4164e083e1c9b5ff928539cef7382f2777eba9436672e6c7c30abad1fdd4452be6ff566e98db950d76b6e08cc6a80351fe78ad00a19ddd63f342b4d40dd28cb41d7fe51ccab1075c4114150f5edf91763352026cc6130d8b3c2b1fc4b677a16029dc113b7bf3c41");
        ASSERT_EQ(bn_mod_exp_public(&mm, ((const unsigned char*)base)[0..L], ((const unsigned char*)e65537)[0..3UL], out[..]), 1);
        hx(out, L, h); ASSERT_STR_EQ(h, "70de9b9dda7cfdbd9aef991cd1a48935eda6f045c1531c085e3e46ff1bcaa1b4ca6d9f079de73511d784ea0b12f7d9836dff625835aa3972e29d7ad45143b22fb226af824238b78f27a6e3ae04651cf16142b309b8f1db9d3d90797d7203a2ed82257cf5e3f2ccc518a24ff39a3119d5bf0c0e3029cb79e5ddb58ea96ff96355f9de96ffd3d1200c2567ceaa527d91647e4ee741a459b059b33bc4127997e3307b629ce410e2148be842fbeb07ee41b3d0fba4a973ff3d27ebf43b3b6b893fd66031d325d90afdd98de2ff77e349db7da950df50e45142a1670409e5b6f9c989af4523e6b394b9e6b40e60092cd2f4714007a93bbe98a3a51c8e0edd209e7b5d");

        L = hb("18c396dc590ef5945b48368ecc3136d8ac738c8ca5be6d6b8016e14dd111eb3749f088d476832b6246103a2babf0ca69536e816b402d5ebbc61251908c07985b796bfa0002f6f33dda1101aa94a6300c2dbbf762a6523995570f0cbef20f678df4f6717c4b39e336b74acffbb0ddb48bfb1ff91ac50e2b9943cdc3567c55dde1d5d1cbd047e5eb8dab163406b4581e378f2f39495d09dd294e424b92c277af3208c9196da9fd1ce996655f23b1d117b75b0f13b0c65fb8a2cb8f1137c938c68d1027e5d35c7fe05880b91b2ede0d0fc58c5050eac832652e83e2c32845b61df27deaab6416d1b5ac2e1c5d3a56a68b41de28123c7fd39898efe8b3b5a080c6a52dece95af5b67e6e8820e3bdfd06f9f6d272324860831ef21cd0c151258170f9e6e053f7675ebe3b785737974a807546462731417aa286ac5154ef5fbac0d757b057c1627cf7fcf6f5cb2afc741b324d85ae8770f9dba1db2b56955dda2d3b3d9ed2aa0cffd21f09ec08693c7401f5ce2425d75a4b78dc3d6baf71d7d8407b1a0f0b97522634f16f5f7cc5d86f3f02402b37d8171b4c24c269f0441ec9bafe62e580c35ea161d90921fdac3af325ffbe90656c8fbb4e5c11fb368220216d27a23501e088d6a34d3e558d2adb7e5a3930cd39e15808606af8a36939efea83854afaa30fac0e42d43c2547f19c6bf84914a6a5bc9974a677c6400db00dd3881a51", mod); hb("042213a1d18aa8665a03a612f5532e9c4b0e9e7f94b59db0f37b7afb56e12b56833ce1378b93983c4b978eaf18463b8a4d27667d9d1289458cc8678b093b5dd5f6e0ff055fb693a68c181ea8d4d4ab0bb058d2ee1d7584adddcc78de68f8886fdb194b90474d9d8ef3bd423e67c6b46259bbb436002b4b4c5fbf2e16e736f4fcb7cfd8913ce3808e345158db971503bd126d15699c9e8752a99b2c28615b43f359e1adc4771fe0ac6a5d89bd0920da718c5ebf8731800c217b8fba722e79e0d7f7cd9231513de7811c5f4755222ed171e44da6a60ddc50fb2a6e4bd639eb08e4fad18c6a78b9d3d6fc8931500ab772dcb7ed25daa18061082e376b8baf9ae4aac24da3363812e2992e45282aa66e12779fc32bb570e9dc059621787d28f745873212c44f07123c8dfefaf558b86d152d454cfe2ba309b1a6ddfc1c08c6530cc38d9f07119fad3bf44d7b86616bb906c9efb9f2c6aa61e4888436682694b24700efac6de6c7c66f539de552dd2bfcd6866aa87c09baf87131abbd5e94b83d5f7b142ea4cdb101446a4869650e7a2c54a0da963a45eff83f045caded41d62cbf2532daef9f2a8a49f0faeeb40c8782d4d52450d1c384a4c93f559741e7f4de438f5d411b0df9e324de73250c60b036d9ddddf1d79fdeb280e62c273105d7c6f4e2ec1e07d7599c8f1c68916d08d4042c36f000feb0bce240c658780d0aa4e15158", base); hb("77ac435f5e93cc5789d3a4b7a46d8bd9905d11ddd0201a4168c422a76b4a4399", ex);
        ASSERT_EQ(mm.init(((const unsigned char*)mod)[0..L]), 1);
        ASSERT_EQ(bn_mod_exp_public(&mm, ((const unsigned char*)base)[0..L], ((const unsigned char*)ex)[0..32UL], out[..]), 1);
        hx(out, L, h); ASSERT_STR_EQ(h, "0f0002a859c375b1baaf623e2569726f164f3a95b667da66cb2c7c69568e836df74904fb99155401ece0db4e841b99418c779c2f3e92c7261c96ac7321cd18af71f328ba495debca66a812f4eb831733eba6d816078504d5efa62d8ba7779760018b8dc5b60011543774d6c1a54a1075edebe3438fdbf5fa849cbe23aa6317076b6fabb46774be4d4c022e5d8eac3da6404e0cc6f5cf3487308db915e3fdfec43dcd03feb2c172cdf23bfe8f7dafca4bae41dac2d12ace0d56ca2b99ee9521135bcabaf7c81bb49c000105c0aae1e17dd1a15581fb8eb9390f487b2080dddae2929bc81d1c92fccca836b7a1a8390fe9f8f8fee6bd692cee07786990fdcc4a01a5c2e6095640d5bb61fe184f6fb99f695afe3160fb345e811c6d58effd402454d51e66851f5ebd15123e6c06015a9f9123b5f6a490b0b0058b1a0e228448ec5d83322fe9e80c907896c4d4d6aa1211e6cc10a542ebf828420d2f5df2d0b5e93f5470d5e6d267d4a81010ea376f4d36cd17355f11d19c83ec6bab9a465c0ee9c765d93a36782b96d1bd76b5f71590b9352247069527c19dfb74b7b59fa15357c61fc910b885b924dd34ab05074f01aed8bbf5b32b61d42035248433bc1378b9cb78b3a56af5c6c2d6595b582e5c752a63a25d1cfd5636a3e84b73d426653d41aef9a66934e5825cf3417aad229c90cda016fc15dfeb5251b634a4ced747a39961");
        ASSERT_EQ(bn_mod_exp_secret(&mm, ((const unsigned char*)base)[0..L], ((const unsigned char*)ex)[0..32UL], out[..]), 1);
        hx(out, L, h); ASSERT_STR_EQ(h, "0f0002a859c375b1baaf623e2569726f164f3a95b667da66cb2c7c69568e836df74904fb99155401ece0db4e841b99418c779c2f3e92c7261c96ac7321cd18af71f328ba495debca66a812f4eb831733eba6d816078504d5efa62d8ba7779760018b8dc5b60011543774d6c1a54a1075edebe3438fdbf5fa849cbe23aa6317076b6fabb46774be4d4c022e5d8eac3da6404e0cc6f5cf3487308db915e3fdfec43dcd03feb2c172cdf23bfe8f7dafca4bae41dac2d12ace0d56ca2b99ee9521135bcabaf7c81bb49c000105c0aae1e17dd1a15581fb8eb9390f487b2080dddae2929bc81d1c92fccca836b7a1a8390fe9f8f8fee6bd692cee07786990fdcc4a01a5c2e6095640d5bb61fe184f6fb99f695afe3160fb345e811c6d58effd402454d51e66851f5ebd15123e6c06015a9f9123b5f6a490b0b0058b1a0e228448ec5d83322fe9e80c907896c4d4d6aa1211e6cc10a542ebf828420d2f5df2d0b5e93f5470d5e6d267d4a81010ea376f4d36cd17355f11d19c83ec6bab9a465c0ee9c765d93a36782b96d1bd76b5f71590b9352247069527c19dfb74b7b59fa15357c61fc910b885b924dd34ab05074f01aed8bbf5b32b61d42035248433bc1378b9cb78b3a56af5c6c2d6595b582e5c752a63a25d1cfd5636a3e84b73d426653d41aef9a66934e5825cf3417aad229c90cda016fc15dfeb5251b634a4ced747a39961");
        ASSERT_EQ(bn_mod_exp_public(&mm, ((const unsigned char*)base)[0..L], ((const unsigned char*)e65537)[0..3UL], out[..]), 1);
        hx(out, L, h); ASSERT_STR_EQ(h, "0094d2e271b09ccb4a5b78c5534ac19a183b5e9aa9a08f9f24bc3e759baa9786a0bc2dd2dc738d88e0c3397550c1a303d42290be273f76a99515c3e23f62d1c20dbf467cee00c44a57f9f23915ab68c8d27c19b794a7d9841fab3405af8304ab4c1f68a50accc0d406d0a7b634e4d00f21cfe453a7ba81e251a20fb8cd1ec53de340c96e66df656bb3ebc1d2b5b8fcdb5a248bdb049e3f01094d8de3c0a1a725e97ea0197c6d4ea77190e24f839a47744989a789ae16d75a6a72a4c3d34f42af256f01510d27c644d4bc5393ded0ddc0f1492e7715f7cd6a9a44c35da391bf36a4b58f2a12ac2af01a94eb6f02256e72fb4ba658ee4cadf92845db06cd64fc5d2cb1800631a1898fd12f707c568f4fd3147041bbf675c2889b27a0252057baa68fdb9a337da16492fe350f787cdc035215f1fda82de1007e3ba0e9fce6af688b22df275a445017c0e03333bbcad0ac8380c0ad4c400f741bdf7fb05a0789c0bf35bab2dce90993d1bac149966d9bb64c2dd9ba88a61cb32518b410308d9500a2844d31c1bc64bff15ea6ba8b1d032d6d59714d9564cf2b2a47d6157e617fb344f50df2ebe30eec01975a6e413684f002dc5f609e16b3350b009afe120b896d7c0fdab5f10dcfb8f17bcce71857e95d1faf02cfb80d362f5c0db552e8683b56701a0d7761a039c111cf3cd59fbe9e67af37078d4241b3371ea8dd174f836c0e9c");

        ASSERT_EQ(bn_mod_exp_public(&mm, ((const unsigned char*)mod)[0..L], ((const unsigned char*)e65537)[0..3UL], out[..]), 0);
    }
}

void t_rsa() {
    unsafe {
        // An RSA-2048 key (PKCS#8) and signatures over "hello rsa" made by
        // OpenSSL: PKCS#1 v1.5 / SHA-256 and PSS / SHA-384 (salt 48).
        unsigned char pub[300]; unsigned char k8[1300]; unsigned char s1[256]; unsigned char s2[256]; unsigned char sig[600];
        unsigned long pl = hb("3082010a0282010100aeda6ad15dfe3166981147fd9fa77c6384e58f22f8e266c73f6fe455b3f74f854bdb7b0b8ff4978020b8342c734960f7e66457f3b254292e9fb03abbe4dc4262db3cd03e3d29e48a97d62b4c310da4783d88e4e39f016986aefc50c3c6ff9ecc27d8d7d2b28710959c8d44be6f5806d35198ac47ef19bcfb37c38c3a45a7a5f464a25e4585c974ec8060f85131827d398f36add485802b506efd4210b4e9ae7e3e623e5d80e4d454616e7177cae39d3d1b9ec8b0f158ee8215f4a8c5fd2d201c0a8912acd2afe15f1a523ece3cf8544b4c593d428bac7958202c5f0821fb5d2cbe63677e85741a955332cff2de3f71d1e5d5fa1105dae82149dac70ee49a7ae90203010001", pub);
        unsigned long kl = hb("308204be020100300d06092a864886f70d0101010500048204a8308204a40201000282010100aeda6ad15dfe3166981147fd9fa77c6384e58f22f8e266c73f6fe455b3f74f854bdb7b0b8ff4978020b8342c734960f7e66457f3b254292e9fb03abbe4dc4262db3cd03e3d29e48a97d62b4c310da4783d88e4e39f016986aefc50c3c6ff9ecc27d8d7d2b28710959c8d44be6f5806d35198ac47ef19bcfb37c38c3a45a7a5f464a25e4585c974ec8060f85131827d398f36add485802b506efd4210b4e9ae7e3e623e5d80e4d454616e7177cae39d3d1b9ec8b0f158ee8215f4a8c5fd2d201c0a8912acd2afe15f1a523ece3cf8544b4c593d428bac7958202c5f0821fb5d2cbe63677e85741a955332cff2de3f71d1e5d5fa1105dae82149dac70ee49a7ae90203010001028201003259c318b73053996f80ca72465f3c7a8bf26edd6b3e06caa981d706b6e4288a01e619f0a26b9494d8d7326efa63cf0093dfa15413ae188dcc7e3aeb9063bcda2ab7c05fabcd3c7a8c6ea7d930e6f4cd8b2ab87eb372fca8440b74566c1f7fe2da369f2f95293fa55587789c3e03e33346f3e7e86ef9c269b50d227c4542998ee3c8c3a98800e79fe2c1c9f4c3c6509c90400424e13a211ba0338421fccc0f344470390f1fac4423af36ac577d63d7eb72a5560e8cb638b64d528c1601fe1c07fd07d096046454abf452f7c63759421cf1524a0f3194250e95adcdc18f35d35144702615164d81c974ea83cdbb4ef10cc1c511d10c595f3b5318beca1b32546102818100f2e00637c368ae024f1797a702bd19644243bc2f8969d97ddf5a1937dcafd0a22dcd8473b0ad4f3fd0c878dde911d88bf363cf46e6d2ab76e0124c174d25c014ec4df1e679e4d4e41330fb129b425e2ee6d87e191f10077524c5d04c3ee9a72285dfa1baa235c44250051d7b9aece508d440a58aee6cd2c0c2f8157b4c64983f02818100b84d5dc631cbacc029aadfa7488ea2d8380d09ab6c4f96388d7a839aa655449fc9549fe094447d97d4554ac6ad779fcdfd2317189cf66608d9858ecfb4348be55a92319ff8b87a30e511b0c7b07a09c70d80cf33e6c28dd911ce0a216515ccf337bc62d4f9747cdcf5ab07535acedafa840c38557fcb410c10fa1bc78c89e2d7028181009b4fc10a25cea9ffd5fe1b3f7493e249e7ca12ec3f2e4e239d11abb0ba523fd38600b7a397a6ac57fdb644fc7b4fab2f8bbc16cf9b49bad630e1e280c017d4e64c5a67c75fc9828e3aeb8692d670317442b0ccaa59dfd1350b07e874e982571bcf8a3b8cf2aff94a028bd2d7b931a620e84252da2e2faef20f7bbf98f2381dd7028180326c283db31df0135ad2ff017b5e5abb210aa01ca0335dfbf6e90fdc5da9726eecdc14299326f1dd7a2023a71b8a60d4306f4b4056a12ae53cedc756980cd0415d7825cf8872a02b846044a73d670609c96df64252dde38501d406fd7ff468f019cd969481f316be3510279eec772f7dde66a17af2cd9eec3830d52cd10c605d02818100be44114fdac1daf626b98ba022bd0500babb5294dacb76482c3ae79196e2ee86451614584d708780e1589df661918a3ca6b937b2da2803df7cff1711f9e647abb92d1b47e4a6335fde9975e1bc2b7f888868517524324ba0104f85f05becc37059d3546237e30bf340f551dffabd0bac7d509a25519096ba0861daef899df24d", k8);
        hb("1e767bd4c32634ed69f5076d315c8160b373cbe492b0ead27857693adc3dda4849f1488b8d7094a20ed756dfde8e8817faf80fca817bac497896deb520e1e25441dfdba5493952db1b1681eb2e4fbc3ce444d42ae95ffaa077cdc7e2c7f63883bcdd69dd15b76889b1f44cb3d9741a49260d8f9a6cee5805ad3128b09ed7e0dd2aac643cad07dac61c3b45836725c796a957222dfd2d97fe3c20522401c2d1d912bab885fb407c6e742db2df6a9e532df5aaf5979fab5e4e91359644224d2067571558ec3ad0fcc2a2996444e21e307191690e9f3589961840168be2090aabf2a61b9fb9058de36b8978da076f9ccbed6bbadaafb9946f48cc360f42096bafc3", s1);
        hb("8667a644fb6af6800221838c9f3b629a968835a90b719b07c2c4c47912fd901b9703e1c8b84d7cbcd93bc7cdddd8d4a437b59f4a57bef2e86241d8d04760737b6a35d201073352d6908fe62f0f0b2341a5e0b8b85d5b1c57c0c0ba84341ab0676a8cc82ecbca98b7c20ace20b1d11176ff5a07804a6af21dda5d10ce767b7c0c3af5ea76dccb12ec88f39c44e325985511913df254e7c5e973e1c8f6227760b5666c902f726917fea2843d811042b68a163f3f9ad47aee2ed04bb72ccab1a464873bc35bf10be0554ed26c883acde74c4bcefda6dec5381f2b649a73f6200a4ecb8444bfbd8709c0c98679aab38d6fe3edc4f740963ec7c60f1867842f291aa9", s2);
        unsigned char h256[32]; unsigned char h384[48];
        digest(DIGEST_SHA256, str_bytes("hello rsa"), h256[..]);
        digest(DIGEST_SHA384, str_bytes("hello rsa"), h384[..]);
        struct RsaPublicKey k;
        ASSERT_EQ(k.init_der(((const unsigned char*)pub)[0..pl]), 1);
        ASSERT_EQ((int)k.bits, 2048);
        ASSERT_EQ(rsa_verify_pkcs1(&k, DIGEST_SHA256, ((const unsigned char*)h256)[0..32UL], ((const unsigned char*)s1)[0..256UL]), 1);
        ASSERT_EQ(rsa_verify_pss(&k, DIGEST_SHA384, ((const unsigned char*)h384)[0..48UL], ((const unsigned char*)s2)[0..256UL], 48UL), 1);
        ASSERT_EQ(rsa_verify_pss(&k, DIGEST_SHA384, ((const unsigned char*)h384)[0..48UL], ((const unsigned char*)s1)[0..256UL], 48UL), 0);
        ASSERT_EQ(rsa_verify_pss(&k, DIGEST_SHA384, ((const unsigned char*)h384)[0..48UL], ((const unsigned char*)s2)[0..256UL], 32UL), 0);
        h256[0] = h256[0] ^ (unsigned char)1;
        ASSERT_EQ(rsa_verify_pkcs1(&k, DIGEST_SHA256, ((const unsigned char*)h256)[0..32UL], ((const unsigned char*)s1)[0..256UL]), 0);
        h256[0] = h256[0] ^ (unsigned char)1;
        s1[100] = s1[100] ^ (unsigned char)4;
        ASSERT_EQ(rsa_verify_pkcs1(&k, DIGEST_SHA256, ((const unsigned char*)h256)[0..32UL], ((const unsigned char*)s1)[0..256UL]), 0);
        s1[100] = s1[100] ^ (unsigned char)4;
        struct RsaPrivateKey pk;
        ASSERT_EQ(pk.init_der(((const unsigned char*)k8)[0..kl]), 1);
        struct RngCtx rng; rng.init();
        unsigned long sl = 0UL;
        ASSERT_EQ(rsa_sign_pss(&pk, DIGEST_SHA256, ((const unsigned char*)h256)[0..32UL], &rng, ((const unsigned char*)sig)[0..600UL], &sl), 1);
        ASSERT_EQ((long long)sl, 256LL);
        ASSERT_EQ(rsa_verify_pss(&k, DIGEST_SHA256, ((const unsigned char*)h256)[0..32UL], ((const unsigned char*)sig)[0..sl], 32UL), 1);
        // PKCS#1 v1.5 is deterministic: our signature is OpenSSL's.
        ASSERT_EQ(rsa_sign_pkcs1(&pk, DIGEST_SHA256, ((const unsigned char*)h256)[0..32UL], ((const unsigned char*)sig)[0..600UL], &sl), 1);
        ASSERT_EQ(ct_eq(((const unsigned char*)sig)[0..256UL], ((const unsigned char*)s1)[0..256UL]), 1);
        pk.destroy();
    }
}

void t_ec() {
    unsafe {
        unsigned char pub[100]; unsigned char bpub[100]; unsigned char sig[120]; unsigned char sig2[120]; unsigned char dig[64]; unsigned char d[48]; unsigned char out[100]; unsigned char key[300];
        char h[400]; unsigned long pl = 0UL; unsigned long bl = 0UL; unsigned long sl = 0UL; unsigned long hl = 0UL; unsigned long kl = 0UL; unsigned long ml = 0UL;
        struct EcPrivateKey ek;

        pl = hb("040169af86624005dfdc690d9f1e3caf34b9042906e034a6d60c1bd45b339dca4e2c1724a2be1dbef807a67a021f11d8e69fffcba37c507b4f1fc56199f166941b", pub); bl = hb("04d006ba4df4dab3c2f445493441b8d94adb292acaa556d2c5feee84433ae32255e9dbba2e3f39e911b0d611ccc88a4c87aa3f6d16bb4d59ec2ed0976bbd07eafa", bpub); sl = hb("3046022100e77df3266ff71de55b9956d7534b648321fe28f83eb37d337c1887b191854628022100c4b88a09c28472d2083f832b2bc191f5c2794cb9ef9a70099bcc6fe0fbc369b3", sig); hl = hb("cbeb18647f37820f6540c1cc7234e0c1b61d5a014ddc9937fce98663cdccd7c7", dig); hb("cef293864e1e8371605e67d8ace6e4055ee43c2b261da0ca806efd44be36520b", d);
        ASSERT_EQ(ec_point_valid(EC_P256, ((const unsigned char*)pub)[0..pl]), 1);
        ASSERT_EQ(ecdsa_verify_der(EC_P256, ((const unsigned char*)pub)[0..pl], ((const unsigned char*)dig)[0..hl], ((const unsigned char*)sig)[0..sl]), 1);
        dig[0] = dig[0] ^ (unsigned char)1;
        ASSERT_EQ(ecdsa_verify_der(EC_P256, ((const unsigned char*)pub)[0..pl], ((const unsigned char*)dig)[0..hl], ((const unsigned char*)sig)[0..sl]), 0);
        dig[0] = dig[0] ^ (unsigned char)1;
        ASSERT_EQ(ec_public_key(EC_P256, d[..], out[..]), 1);
        hx(out, pl, h); ASSERT_STR_EQ(h, "040169af86624005dfdc690d9f1e3caf34b9042906e034a6d60c1bd45b339dca4e2c1724a2be1dbef807a67a021f11d8e69fffcba37c507b4f1fc56199f166941b");
        ASSERT_EQ(ecdh_shared(EC_P256, d[..], ((const unsigned char*)bpub)[0..bl], out[..]), 1);
        hx(out, ec_scalar_size(EC_P256), h); ASSERT_STR_EQ(h, "4d1e7516639d4191984503f25f15131d48f314d86d4fa7bd23a6bae22cd044c5");
        ml = 0UL;
        ASSERT_EQ(ecdsa_sign_der(EC_P256, d[..], ((const unsigned char*)dig)[0..hl], ((const unsigned char*)sig2)[0..120UL], &ml), 1);
        ASSERT_EQ(ecdsa_verify_der(EC_P256, ((const unsigned char*)pub)[0..pl], ((const unsigned char*)dig)[0..hl], ((const unsigned char*)sig2)[0..ml]), 1);
        kl = hb("308187020100301306072a8648ce3d020106082a8648ce3d030107046d306b0201010420cef293864e1e8371605e67d8ace6e4055ee43c2b261da0ca806efd44be36520ba144034200040169af86624005dfdc690d9f1e3caf34b9042906e034a6d60c1bd45b339dca4e2c1724a2be1dbef807a67a021f11d8e69fffcba37c507b4f1fc56199f166941b", key); ASSERT_EQ(ek.init_der(((const unsigned char*)key)[0..kl]), 1); ASSERT_EQ(ek.curve, EC_P256);
        hx(ek.pub, ek.pub_len, h); ASSERT_STR_EQ(h, "040169af86624005dfdc690d9f1e3caf34b9042906e034a6d60c1bd45b339dca4e2c1724a2be1dbef807a67a021f11d8e69fffcba37c507b4f1fc56199f166941b");
        kl = hb("30770201010420cef293864e1e8371605e67d8ace6e4055ee43c2b261da0ca806efd44be36520ba00a06082a8648ce3d030107a144034200040169af86624005dfdc690d9f1e3caf34b9042906e034a6d60c1bd45b339dca4e2c1724a2be1dbef807a67a021f11d8e69fffcba37c507b4f1fc56199f166941b", key); ASSERT_EQ(ek.init_der(((const unsigned char*)key)[0..kl]), 1);
        hx(ek.pub, ek.pub_len, h); ASSERT_STR_EQ(h, "040169af86624005dfdc690d9f1e3caf34b9042906e034a6d60c1bd45b339dca4e2c1724a2be1dbef807a67a021f11d8e69fffcba37c507b4f1fc56199f166941b");
        pub[5] = pub[5] ^ (unsigned char)1;
        ASSERT_EQ(ec_point_valid(EC_P256, ((const unsigned char*)pub)[0..pl]), 0);
        ASSERT_EQ(ecdh_shared(EC_P256, d[..], ((const unsigned char*)pub)[0..pl], out[..]), 0);

        pl = hb("043c4a51e0fdb9fd3a6548ff2161c0c8078b39e328dd216938214819647ab90209ef3045aa7fd3f6c3bf0c1e8ac8ab49db929e1df91d832e9ac3b3c35242788c5d5b064eea199c9e6d78f4c6f9f9648615280cad78d309fdab56b4df99cb1e578c", pub); bl = hb("04473c066a96b33b614ad9b6b3c21c86764929b367ba84bc32b83d5cb8d5e95a2e70847f95fb1b86d791e701441fcca5ee9ab55797e4c367fa12b38f6dfd297e41930f2fd1e098727e3613585cf0d875ef89e13decd648831f6535ed392cd92e6b", bpub); sl = hb("3066023100e98ec77fef2eb9c679de22eaea02c8e99e988517b81d8f28b129a5893dd23496f52bcbfa4330df7e645f53d5dd471ef7023100de5fc47194b7c3a16678faea8eda18574fb36b4192844947e1dd94a3e47a995c77dc134e80d636d1b8bb8337e186ff80", sig); hl = hb("e814a9950733ffb6ef027e5b3f2170af89aa911462204ab0b6358f03a18e85918754893c2f419650862d7659c0cd5719", dig); hb("e6f9daaa7decd399492bb2d67a7a93701651e05779919af59fb189dc1728b4acf73afdae0e40cf21ffec5cf5fa0684d2", d);
        ASSERT_EQ(ec_point_valid(EC_P384, ((const unsigned char*)pub)[0..pl]), 1);
        ASSERT_EQ(ecdsa_verify_der(EC_P384, ((const unsigned char*)pub)[0..pl], ((const unsigned char*)dig)[0..hl], ((const unsigned char*)sig)[0..sl]), 1);
        dig[0] = dig[0] ^ (unsigned char)1;
        ASSERT_EQ(ecdsa_verify_der(EC_P384, ((const unsigned char*)pub)[0..pl], ((const unsigned char*)dig)[0..hl], ((const unsigned char*)sig)[0..sl]), 0);
        dig[0] = dig[0] ^ (unsigned char)1;
        ASSERT_EQ(ec_public_key(EC_P384, d[..], out[..]), 1);
        hx(out, pl, h); ASSERT_STR_EQ(h, "043c4a51e0fdb9fd3a6548ff2161c0c8078b39e328dd216938214819647ab90209ef3045aa7fd3f6c3bf0c1e8ac8ab49db929e1df91d832e9ac3b3c35242788c5d5b064eea199c9e6d78f4c6f9f9648615280cad78d309fdab56b4df99cb1e578c");
        ASSERT_EQ(ecdh_shared(EC_P384, d[..], ((const unsigned char*)bpub)[0..bl], out[..]), 1);
        hx(out, ec_scalar_size(EC_P384), h); ASSERT_STR_EQ(h, "aebb36350c0cb4500eef1ff2297ad8d424cdedc6b93846db2da7a8d6e812c0f0287efe3138042ac2b7e27871c007891b");
        ml = 0UL;
        ASSERT_EQ(ecdsa_sign_der(EC_P384, d[..], ((const unsigned char*)dig)[0..hl], ((const unsigned char*)sig2)[0..120UL], &ml), 1);
        ASSERT_EQ(ecdsa_verify_der(EC_P384, ((const unsigned char*)pub)[0..pl], ((const unsigned char*)dig)[0..hl], ((const unsigned char*)sig2)[0..ml]), 1);
        kl = hb("3081b6020100301006072a8648ce3d020106052b8104002204819e30819b0201010430e6f9daaa7decd399492bb2d67a7a93701651e05779919af59fb189dc1728b4acf73afdae0e40cf21ffec5cf5fa0684d2a164036200043c4a51e0fdb9fd3a6548ff2161c0c8078b39e328dd216938214819647ab90209ef3045aa7fd3f6c3bf0c1e8ac8ab49db929e1df91d832e9ac3b3c35242788c5d5b064eea199c9e6d78f4c6f9f9648615280cad78d309fdab56b4df99cb1e578c", key); ASSERT_EQ(ek.init_der(((const unsigned char*)key)[0..kl]), 1); ASSERT_EQ(ek.curve, EC_P384);
        hx(ek.pub, ek.pub_len, h); ASSERT_STR_EQ(h, "043c4a51e0fdb9fd3a6548ff2161c0c8078b39e328dd216938214819647ab90209ef3045aa7fd3f6c3bf0c1e8ac8ab49db929e1df91d832e9ac3b3c35242788c5d5b064eea199c9e6d78f4c6f9f9648615280cad78d309fdab56b4df99cb1e578c");
        kl = hb("3081a40201010430e6f9daaa7decd399492bb2d67a7a93701651e05779919af59fb189dc1728b4acf73afdae0e40cf21ffec5cf5fa0684d2a00706052b81040022a164036200043c4a51e0fdb9fd3a6548ff2161c0c8078b39e328dd216938214819647ab90209ef3045aa7fd3f6c3bf0c1e8ac8ab49db929e1df91d832e9ac3b3c35242788c5d5b064eea199c9e6d78f4c6f9f9648615280cad78d309fdab56b4df99cb1e578c", key); ASSERT_EQ(ek.init_der(((const unsigned char*)key)[0..kl]), 1);
        hx(ek.pub, ek.pub_len, h); ASSERT_STR_EQ(h, "043c4a51e0fdb9fd3a6548ff2161c0c8078b39e328dd216938214819647ab90209ef3045aa7fd3f6c3bf0c1e8ac8ab49db929e1df91d832e9ac3b3c35242788c5d5b064eea199c9e6d78f4c6f9f9648615280cad78d309fdab56b4df99cb1e578c");
        pub[5] = pub[5] ^ (unsigned char)1;
        ASSERT_EQ(ec_point_valid(EC_P384, ((const unsigned char*)pub)[0..pl]), 0);
        ASSERT_EQ(ecdh_shared(EC_P384, d[..], ((const unsigned char*)pub)[0..pl], out[..]), 0);

        // RFC 6979 A.2.5, P-256 / SHA-256, "sample"
        hb("c9afa9d845ba75166b5c215767b1d6934e50c3db36e89b127b8a622b120f6721", d);
        sha256(str_bytes("sample"), dig[..]);
        ml = 0UL;
        ASSERT_EQ(ecdsa_sign_der(EC_P256, d[..], ((const unsigned char*)dig)[0..32UL], ((const unsigned char*)sig2)[0..120UL], &ml), 1);
        hx(sig2, ml, h);
        ASSERT_STR_EQ(h, "3046022100efd48b2aacb6a8fd1140dd9cd45e81d69d2c877b56aaf991c34d0ea84eaf3716022100f7cb1c942d657c41d436c7a1b6e29f65f3e900dbb9aff4064dc4ab2f843acda8");
    }
}

void t_edge() {
    unsafe {
        unsigned char d[32]; unsigned char pub[65]; unsigned char h[32]; unsigned char r[32]; unsigned char sv[32];
        int i = 0;
        while (i < 32) { d[i] = (unsigned char)(i + 1); h[i] = (unsigned char)(7 * i); r[i] = (unsigned char)0; sv[i] = (unsigned char)0; i = i + 1; }
        ASSERT_EQ(ec_public_key(EC_P256, d[..], pub[..]), 1);
        sv[31] = (unsigned char)1;
        ASSERT_EQ(ecdsa_verify(EC_P256, ((const unsigned char*)pub)[0..65UL], ((const unsigned char*)h)[0..32UL], r[..], sv[..]), 0);      // r = 0
        r[31] = (unsigned char)1;
        ASSERT_EQ(ecdsa_verify(EC_P256, ((const unsigned char*)pub)[0..65UL], ((const unsigned char*)h)[0..32UL], r[..], sv[..]), 0);      // forged (1, 1)
        // s = n is out of range
        hb("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551", sv);
        ASSERT_EQ(ecdsa_verify(EC_P256, ((const unsigned char*)pub)[0..65UL], ((const unsigned char*)h)[0..32UL], r[..], sv[..]), 0);
        // private key 0 and >= n are rejected
        unsigned char z[32]; i = 0; while (i < 32) { z[i] = (unsigned char)0; i = i + 1; }
        ASSERT_EQ(ec_public_key(EC_P256, z[..], pub[..]), 0);
        ASSERT_EQ(ec_public_key(EC_P256, sv[..], pub[..]), 0);
        // the point at infinity / wrong encodings are not public keys
        unsigned char inf[65]; i = 0; while (i < 65) { inf[i] = (unsigned char)0; i = i + 1; }
        inf[0] = (unsigned char)4;
        ASSERT_EQ(ec_point_valid(EC_P256, ((const unsigned char*)inf)[0..65UL]), 0);
        ASSERT_EQ(ec_point_valid(EC_P256, ((const unsigned char*)pub)[0..64UL]), 0);
        ASSERT_EQ(ec_scalar_size(EC_P384), 48);
        ASSERT_EQ(ec_scalar_size(99), 0);
        // ct helpers
        unsigned char a[4]; unsigned char b[4];
        a[0] = (unsigned char)1; a[1] = (unsigned char)2; a[2] = (unsigned char)3; a[3] = (unsigned char)4;
        b[0] = (unsigned char)1; b[1] = (unsigned char)2; b[2] = (unsigned char)3; b[3] = (unsigned char)5;
        ASSERT_EQ(ct_eq(((const unsigned char*)a)[0..3UL], ((const unsigned char*)b)[0..3UL]), 1);
        ASSERT_EQ(ct_eq(((const unsigned char*)a)[0..4UL], ((const unsigned char*)b)[0..4UL]), 0);
        ASSERT_EQ(ct_is_zero(((const unsigned char*)z)[0..32UL]), 1);
        ct_copy_if(0U, ((const unsigned char*)a)[0..4UL], ((const unsigned char*)b)[0..4UL]);
        ASSERT_EQ(a[3], 4);
        ct_copy_if(1U, ((const unsigned char*)a)[0..4UL], ((const unsigned char*)b)[0..4UL]);
        ASSERT_EQ(a[3], 5);
        ASSERT_EQ(ct_eq_u32(5U, 5U), (long long)0xFFFFFFFFU);
        ASSERT_EQ(ct_eq_u32(5U, 6U), 0);
        // digest ids
        ASSERT_EQ(digest_size(DIGEST_SHA384), 48);
        unsigned char out[64];
        ASSERT_EQ(digest(DIGEST_SHA512, str_bytes("abc"), out[..]), 64);
        ASSERT_EQ(out[0], 0xdd);
        ASSERT_EQ(digest(42, str_bytes("abc"), out[..]), 0);
    }
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("chacha20 / poly1305 / aead (RFC 8439)", t_chacha);
        t.add("x25519 (RFC 7748)", t_x25519);
        t.add("hkdf-sha256 (RFC 5869)", t_hkdf);
        t.add("montgomery modexp, public + secret", t_bn);
        t.add("rsa pkcs1 v1.5 / pss sign + verify", t_rsa);
        t.add("p-256 / p-384 ecdsa + ecdh", t_ec);
        t.add("ecdsa edge cases, constant-time helpers", t_edge);
    }
    t.run(); t.print_summary();
    return t.all_passed() ? 0 : 1;
}
