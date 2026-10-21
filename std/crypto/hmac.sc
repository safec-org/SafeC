// SafeC Standard Library — HMAC-SHA256 (see hmac.h; Hmac<H> is header-only).
#pragma once
#include <std/crypto/hmac.h>
#include <std/crypto/sha256.h>

namespace std {

void hmac_sha256(const &unsigned char key, unsigned long keyLen,
                  const &unsigned char data, unsigned long dataLen,
                  &unsigned char out) {
    hash_bytes(Hmac::new(Sha256Ctx::new(), key, keyLen), data, dataLen, out);
}

} // namespace std
