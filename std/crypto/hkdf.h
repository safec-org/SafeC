#pragma once
// SafeC Standard Library — HKDF (RFC 5869) over any hash.
//
// Written once over hash.h's BlockHash through Hmac<H>, so HKDF-SHA256 and
// HKDF-SHA384 (TLS 1.3's key schedule, std/tls) are the same code:
//
//   unsigned char prk[SHA256_DIGEST_SIZE];
//   std::hkdf_extract(Sha256Ctx::new(), salt, saltLen, ikm, ikmLen, &prk[0]);
//   std::hkdf_expand(Sha256Ctx::new(), &prk[0], 32UL, info, infoLen, &okm[0], 42UL);
//
// Freestanding-safe (no allocation).
#include <std/crypto/hash.h>
#include <std/crypto/hmac.h>
#include <std/mem.h>

namespace std {

// PRK = HMAC-Hash(salt, IKM), digest_size() bytes into 'prk'. An
// empty salt is the all-zero string of digest_size() bytes (RFC 5869 §2.2).
generic<H: BlockHash>
void hkdf_extract(H fresh, []const unsigned char salt, []const unsigned char ikm, []unsigned char prk) {
    unsigned char zeros[HMAC_MAX_DIGEST_];
    unsigned long i = 0UL;
    while (i < (unsigned long)HMAC_MAX_DIGEST_) { zeros[i] = (unsigned char)0; i = i + 1UL; }
    struct Hmac<H> m;
    unsigned long hl = fresh.digest_size();
    if (salt.len > 0UL) { m = Hmac::new(fresh, salt); }
    else { m = Hmac::new(fresh, zeros[0..hl]); }
    m.update(ikm);
    m.finish(prk);
    mem_zero_secure_obj(&m);
}

// OKM = T(1) || T(2) || ... filling out (out.len bytes), T(i) =
// HMAC-Hash(PRK, T(i-1) || info || i); info may be empty. 0 (nothing
// written) if out.len > 255 * digest_size(), 1 otherwise.
generic<H: BlockHash>
int hkdf_expand(H fresh, []const unsigned char prk, []const unsigned char info, []unsigned char out) {
    unsigned long out_len = out.len;
    unsigned long hl = fresh.digest_size();
    if (out_len > 255UL * hl) { return 0; }
    unsigned char t[HMAC_MAX_DIGEST_];
    unsigned long tlen = 0UL;
    unsigned long done = 0UL;
    unsigned char counter = (unsigned char)1;
    while (done < out_len) {
        struct Hmac<H> m = Hmac::new(fresh, prk);
        if (tlen > 0UL) { m.update(t[0..tlen]); }
        if (info.len > 0UL) { m.update(info); }
        unsigned char ctr[1];
        ctr[0] = counter;
        m.update(ctr[..]);
        m.finish(t[..]);
        mem_zero_secure_obj(&m);
        tlen = hl;
        unsigned long n = out_len - done < hl ? out_len - done : hl;
        unsigned long i = 0UL;
        while (i < n) { out[done + i] = t[i]; i = i + 1UL; }
        done = done + n;
        counter = (unsigned char)(counter + (unsigned char)1);
    }
    mem_zero_secure(t[0..(unsigned long)HMAC_MAX_DIGEST_]);
    return 1;
}

} // namespace std
