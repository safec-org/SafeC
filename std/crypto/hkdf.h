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

// PRK = HMAC-Hash(salt, IKM), digest_size() bytes into 'prk'. A null or
// empty salt is the all-zero string of digest_size() bytes (RFC 5869 §2.2).
generic<H: BlockHash>
void hkdf_extract(H fresh, const ?&unsigned char salt, unsigned long salt_len,
                  const &unsigned char ikm, unsigned long ikm_len, &unsigned char prk) {
    unsigned char zeros[HMAC_MAX_DIGEST_];
    unsigned long i = 0UL;
    while (i < (unsigned long)HMAC_MAX_DIGEST_) { zeros[i] = (unsigned char)0; i = i + 1UL; }
    struct Hmac<H> m;
    unsigned long hl = fresh.digest_size();
    match (salt) {
        case some(s): {
            if (salt_len > 0UL) { m = Hmac::new(fresh, s, salt_len); }
            else { m = Hmac::new(fresh, &zeros[0], hl); }
        }
        case null: { m = Hmac::new(fresh, &zeros[0], hl); }
    }
    m.update(ikm, ikm_len);
    m.finish(prk);
    mem_zero_secure(&m, sizeof(struct Hmac<H>));
}

// OKM = T(1) || T(2) || ... truncated to out_len bytes, T(i) =
// HMAC-Hash(PRK, T(i-1) || info || i). 'info' may be null when info_len
// is 0. 0 (nothing written) if out_len > 255 * digest_size(), 1 otherwise.
generic<H: BlockHash>
int hkdf_expand(H fresh, const &unsigned char prk, unsigned long prk_len,
                const ?&unsigned char info, unsigned long info_len,
                &unsigned char out, unsigned long out_len) {
    unsigned long hl = fresh.digest_size();
    if (out_len > 255UL * hl) { return 0; }
    unsigned char t[HMAC_MAX_DIGEST_];
    unsigned long tlen = 0UL;
    unsigned long done = 0UL;
    unsigned char counter = (unsigned char)1;
    while (done < out_len) {
        struct Hmac<H> m = Hmac::new(fresh, prk, prk_len);
        if (tlen > 0UL) { m.update(&t[0], tlen); }
        match (info) {
            case some(inf): { if (info_len > 0UL) { m.update(inf, info_len); } }
            case null: { }
        }
        m.update(&counter, 1UL);
        m.finish(&t[0]);
        mem_zero_secure(&m, sizeof(struct Hmac<H>));
        tlen = hl;
        unsigned long n = out_len - done < hl ? out_len - done : hl;
        unsafe {
            unsigned char* o = (unsigned char*)out;
            unsigned long i = 0UL;
            while (i < n) { o[done + i] = t[i]; i = i + 1UL; }
        }
        done = done + n;
        counter = (unsigned char)(counter + (unsigned char)1);
    }
    mem_zero_secure(&t[0], (unsigned long)HMAC_MAX_DIGEST_);
    return 1;
}

} // namespace std
