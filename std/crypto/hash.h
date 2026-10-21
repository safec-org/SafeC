#pragma once
// SafeC Standard Library — hash function traits.
//
// std's hash contexts share one streaming shape: feed bytes with update(),
// then finish() writes the digest. The traits below name it, and code that
// works over any hash (HMAC, one-shot hashing) is written once against them;
// each call is resolved at compile time (no dispatch at run time).
//
//   Hasher     update + finish + digest_size: Sha1Ctx (sha1.h), Sha256Ctx
//              (sha256.h; SHA-256 or SHA-224), Sha512Ctx (sha512.h; SHA-512
//              or SHA-384), Hmac<H> (hmac.h)
//   BlockHash  Hasher + block_size: a hash over fixed-size blocks, as HMAC
//              requires — Sha1Ctx, Sha256Ctx, Sha512Ctx
//
//   unsigned char d[SHA384_DIGEST_SIZE];
//   hash_bytes(Sha512Ctx::new_384(), msg, len, &d[0]);
//   struct Hmac<Sha512Ctx> mac = Hmac::new(Sha512Ctx::new(), key, keyLen);
//   mac.update(part1, n1); mac.update(part2, n2); mac.finish(&tag[0]);

namespace std {

trait Hasher {
    // Feeds 'len' bytes of the message.
    void update(const &unsigned char data, unsigned long len);
    // Writes the digest (digest_size() bytes). The context is spent after.
    void finish(&unsigned char digest);
    // Bytes finish() writes.
    unsigned long digest_size() const;
}

trait BlockHash : Hasher {
    // Bytes per compression-function block (HMAC pads its key to this).
    unsigned long block_size() const;
}

// The digest of 'len' bytes with a fresh context 'h' (taken by value, so
// the caller's copy stays fresh): hash_bytes(Sha256Ctx::new(), ...).
generic<H: Hasher>
void hash_bytes(H h, const &unsigned char data, unsigned long len, &unsigned char digest) {
    h.update(data, len);
    h.finish(digest);
}

} // namespace std
