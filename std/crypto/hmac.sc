// SafeC Standard Library — HMAC-SHA256 (see hmac.h; Hmac<H> is header-only).
#pragma once
#include <std/crypto/hmac.h>
#include <std/crypto/sha256.h>

namespace std {

void hmac_sha256_n_(const &unsigned char key, unsigned long keyLen,
                  const &unsigned char data, unsigned long dataLen,
                  &unsigned char out) {
    unsafe {
        hash_bytes(Hmac::new(Sha256Ctx::new(), ((const unsigned char*)key)[0..keyLen]),
                   ((const unsigned char*)data)[0..dataLen], ((unsigned char*)out)[0..32]);
    }
}

void hmac_sha256([]const unsigned char key_s, []const unsigned char data_s, []unsigned char out_s) {
    unsigned long keyLen = key_s.len;
    const &unsigned char key = crypto_in_(key_s, 0UL);
    unsigned long dataLen = data_s.len;
    const &unsigned char data = crypto_in_(data_s, 0UL);
    &unsigned char out = crypto_out_(out_s, 32UL);
    hmac_sha256_n_(key, keyLen, data, dataLen, out);
}

} // namespace std
