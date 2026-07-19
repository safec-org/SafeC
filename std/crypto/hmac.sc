// SafeC Standard Library — HMAC-SHA256 implementation (see hmac.h).
#pragma once
#include <std/crypto/hmac.h>
#include <std/crypto/sha256.h>
#include <std/crypto/sha256.sc>
#include <std/mem.sc>

namespace std {

void hmac_sha256(const unsigned char* key, unsigned long keyLen,
                  const unsigned char* data, unsigned long dataLen,
                  unsigned char* out) {
    unsafe {
        unsigned char keyBlock[SHA256_BLOCK_SIZE];
        unsigned long i = 0UL;

        if (keyLen > (unsigned long)SHA256_BLOCK_SIZE) {
            unsigned char keyDigest[SHA256_DIGEST_SIZE];
            sha256((const &stack unsigned char)key, keyLen, (&stack unsigned char)keyDigest);
            i = 0UL;
            while (i < (unsigned long)SHA256_DIGEST_SIZE) { keyBlock[i] = keyDigest[i]; i = i + 1UL; }
            while (i < (unsigned long)SHA256_BLOCK_SIZE) { keyBlock[i] = (unsigned char)0; i = i + 1UL; }
        } else {
            i = 0UL;
            while (i < keyLen) { keyBlock[i] = key[i]; i = i + 1UL; }
            while (i < (unsigned long)SHA256_BLOCK_SIZE) { keyBlock[i] = (unsigned char)0; i = i + 1UL; }
        }

        unsigned char ipad[SHA256_BLOCK_SIZE];
        unsigned char opad[SHA256_BLOCK_SIZE];
        i = 0UL;
        while (i < (unsigned long)SHA256_BLOCK_SIZE) {
            ipad[i] = keyBlock[i] ^ (unsigned char)0x36;
            opad[i] = keyBlock[i] ^ (unsigned char)0x5c;
            i = i + 1UL;
        }

        // inner = SHA256(ipad || data)
        struct Sha256Ctx innerCtx = sha256_init();
        innerCtx.update((const &stack unsigned char)ipad, (unsigned long)SHA256_BLOCK_SIZE);
        innerCtx.update((const &stack unsigned char)data, dataLen);
        unsigned char innerDigest[SHA256_DIGEST_SIZE];
        innerCtx.finish((&stack unsigned char)innerDigest);

        // out = SHA256(opad || inner)
        struct Sha256Ctx outerCtx = sha256_init();
        outerCtx.update((const &stack unsigned char)opad, (unsigned long)SHA256_BLOCK_SIZE);
        outerCtx.update((const &stack unsigned char)innerDigest, (unsigned long)SHA256_DIGEST_SIZE);
        outerCtx.finish((&stack unsigned char)out);
    }
}

} // namespace std
