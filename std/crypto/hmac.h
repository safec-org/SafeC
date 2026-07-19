#pragma once
// SafeC Standard Library — HMAC-SHA256 (RFC 2104 construction over
// std::sha256).
#define HMAC_SHA256_SIZE 32 // bytes

namespace std {

// Writes the 32-byte HMAC-SHA256 of 'data' (length 'dataLen') keyed by
// 'key' (length 'keyLen') into 'out' (must be >= 32 bytes).
void hmac_sha256(const unsigned char* key, unsigned long keyLen,
                  const unsigned char* data, unsigned long dataLen,
                  unsigned char* out);

} // namespace std
