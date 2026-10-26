#pragma once
#include <std/crypto/buf.h>
// SafeC Standard Library — HMAC (RFC 2104) over any hash.
//
// Hmac<H> is the HMAC construction written once over hash.h's BlockHash, so
// HMAC-SHA1 / -SHA224 / -SHA256 / -SHA384 / -SHA512 are the same code with a
// different hash context. It streams like a hash, and is a Hasher itself
// (update / finish / digest_size): hash_bytes and any generic<H: Hasher>
// code take it.
//
//   struct Hmac<Sha512Ctx> mac = Hmac::new(Sha512Ctx::new(), key, keyLen);
//   mac.update(msg, msgLen);
//   mac.finish(&tag[0]);                       // 64 bytes: mac.digest_size()
//
// hmac_sha256 is the one-shot form for HMAC-SHA256.
#include <std/crypto/hash.h>

#define HMAC_SHA256_SIZE 32 // bytes
#define HMAC_MAX_BLOCK_  128 // largest block_size() among std's hashes (SHA-512)
#define HMAC_MAX_DIGEST_ 64  // largest digest_size() (SHA-512)

namespace std {

generic<H: BlockHash>
struct Hmac {
    H inner;   // H((key ^ ipad) || message ...)
    H outer;   // H((key ^ opad) || ...), fed the inner digest by finish()

    // An HMAC keyed by 'key' over 'fresh', a new context of the hash to use
    // (Sha256Ctx::new(), Sha512Ctx::new_384(), ...). A key longer than the
    // hash's block is hashed first, as RFC 2104 specifies.
    static struct Hmac<H> new(H fresh, []const unsigned char key);
    void          update([]const unsigned char data);
    // Writes the digest_size()-byte MAC into mac (a shorter slice panics).
    // The context is spent after.
    void          finish([]unsigned char mac);
    unsigned long digest_size() const;
};

generic<H: BlockHash>
struct Hmac<H> Hmac::new(H fresh, []const unsigned char key) {
    unsigned long block = fresh.block_size();
    unsigned char keyBlock[HMAC_MAX_BLOCK_];
    unsigned long i = 0UL;
    if (key.len > block) {
        hash_bytes(fresh, key, keyBlock[..]);
        i = fresh.digest_size();
    } else {
        while (i < key.len) { keyBlock[i] = key[i]; i = i + 1UL; }
    }
    while (i < block) { keyBlock[i] = (unsigned char)0; i = i + 1UL; }

    unsigned char ipad[HMAC_MAX_BLOCK_];
    unsigned char opad[HMAC_MAX_BLOCK_];
    i = 0UL;
    while (i < block) {
        ipad[i] = keyBlock[i] ^ (unsigned char)0x36;
        opad[i] = keyBlock[i] ^ (unsigned char)0x5c;
        i = i + 1UL;
    }
    struct Hmac<H> m;
    m.inner = fresh;
    m.inner.update(ipad[0..block]);
    m.outer = fresh;
    m.outer.update(opad[0..block]);
    return m;
}

generic<H: BlockHash>
void Hmac::update([]const unsigned char data) { self.inner.update(data); }

generic<H: BlockHash>
void Hmac::finish([]unsigned char mac) {
    unsigned char innerDigest[HMAC_MAX_DIGEST_];
    self.inner.finish(innerDigest[..]);
    self.outer.update(innerDigest[0..self.inner.digest_size()]);
    self.outer.finish(mac);
}

generic<H: BlockHash>
unsigned long Hmac::digest_size() const { return self.outer.digest_size(); }

// Writes the 32-byte HMAC-SHA256 of 'data' (length 'dataLen') keyed by
// 'key' (length 'keyLen') into 'out' (must be >= 32 bytes).
void hmac_sha256([]const unsigned char key, []const unsigned char data, []unsigned char out);
// internal: the (reference, length) core of hmac_sha256()
void hmac_sha256_n_(const &unsigned char key, unsigned long keyLen,
                  const &unsigned char data, unsigned long dataLen,
                  &unsigned char out);

} // namespace std
