// SafeC Standard Library — X.509: full parsing, signature verification and
// host matching (X509Parsed & co. in x509.h).
#pragma once
#include <std/crypto/x509.h>
#include <std/crypto/asn1.h>
#include <std/crypto/digest.h>
#include <std/crypto/rsa.h>
#include <std/crypto/ec.h>

namespace std {

// ── OIDs (DER content bytes) ──────────────────────────────────────────────────
static unsigned char x5_oid_rsa_[9]       = { 0x2a,0x86,0x48,0x86,0xf7,0x0d,0x01,0x01,0x01 };
static unsigned char x5_oid_rsa_pss_[9]   = { 0x2a,0x86,0x48,0x86,0xf7,0x0d,0x01,0x01,0x0a };
static unsigned char x5_oid_sha1rsa_[9]   = { 0x2a,0x86,0x48,0x86,0xf7,0x0d,0x01,0x01,0x05 };
static unsigned char x5_oid_sha256rsa_[9] = { 0x2a,0x86,0x48,0x86,0xf7,0x0d,0x01,0x01,0x0b };
static unsigned char x5_oid_sha384rsa_[9] = { 0x2a,0x86,0x48,0x86,0xf7,0x0d,0x01,0x01,0x0c };
static unsigned char x5_oid_sha512rsa_[9] = { 0x2a,0x86,0x48,0x86,0xf7,0x0d,0x01,0x01,0x0d };
static unsigned char x5_oid_ecpk_[7]      = { 0x2a,0x86,0x48,0xce,0x3d,0x02,0x01 };
static unsigned char x5_oid_p256_[8]      = { 0x2a,0x86,0x48,0xce,0x3d,0x03,0x01,0x07 };
static unsigned char x5_oid_p384_[5]      = { 0x2b,0x81,0x04,0x00,0x22 };
static unsigned char x5_oid_ecdsa1_[7]    = { 0x2a,0x86,0x48,0xce,0x3d,0x04,0x01 };
static unsigned char x5_oid_ecdsa256_[8]  = { 0x2a,0x86,0x48,0xce,0x3d,0x04,0x03,0x02 };
static unsigned char x5_oid_ecdsa384_[8]  = { 0x2a,0x86,0x48,0xce,0x3d,0x04,0x03,0x03 };
static unsigned char x5_oid_ecdsa512_[8]  = { 0x2a,0x86,0x48,0xce,0x3d,0x04,0x03,0x04 };
static unsigned char x5_oid_ed25519_[3]   = { 0x2b,0x65,0x70 };
static unsigned char x5_oid_sha256_[9]    = { 0x60,0x86,0x48,0x01,0x65,0x03,0x04,0x02,0x01 };
static unsigned char x5_oid_sha384_[9]    = { 0x60,0x86,0x48,0x01,0x65,0x03,0x04,0x02,0x02 };
static unsigned char x5_oid_sha512_[9]    = { 0x60,0x86,0x48,0x01,0x65,0x03,0x04,0x02,0x03 };
static unsigned char x5_oid_cn_[3]        = { 0x55,0x04,0x03 };
static unsigned char x5_oid_eku_server_[8] = { 0x2b,0x06,0x01,0x05,0x05,0x07,0x03,0x01 };
static unsigned char x5_oid_eku_client_[8] = { 0x2b,0x06,0x01,0x05,0x05,0x07,0x03,0x02 };
static unsigned char x5_oid_eku_any_[4]   = { 0x55,0x1d,0x25,0x00 };
static unsigned char x5_oid_aia_[8]       = { 0x2b,0x06,0x01,0x05,0x05,0x07,0x01,0x01 };

static struct X509Span x5_span_(const unsigned char* base, const &Asn1_ v) {
    struct X509Span s;
    unsafe { s.off = (unsigned long)v.p - (unsigned long)base; }
    s.len = v.len;
    return s;
}

static struct X509Span x5_none_() {
    struct X509Span s;
    s.off = 0UL;
    s.len = 0UL;
    return s;
}

// ── Time ──────────────────────────────────────────────────────────────────────

static int x5_digits_(const unsigned char* s, unsigned long n, &int out) {
    int v = 0;
    unsafe {
        unsigned long i = 0UL;
        while (i < n) {
            if (s[i] < (unsigned char)'0' || s[i] > (unsigned char)'9') { return 0; }
            v = v * 10 + (int)(s[i] - (unsigned char)'0');
            i = i + 1UL;
        }
    }
    *out = v;
    return 1;
}

// Days since 1970-01-01 of a proleptic Gregorian date (Hinnant's algorithm).
static long long x5_days_(int y, int m, int d) {
    if (m <= 2) { y = y - 1; }
    long long era = (long long)(y >= 0 ? y : y - 399) / 400LL;
    long long yoe = (long long)y - era * 400LL;
    long long mp = (long long)(m > 2 ? m - 3 : m + 9);
    long long doy = (153LL * mp + 2LL) / 5LL + (long long)d - 1LL;
    long long doe = yoe * 365LL + yoe / 4LL - yoe / 100LL + doy;
    return era * 146097LL + doe - 719468LL;
}

static int x5_time_(&Asn1_ in, &unsigned long long out) {
    unsigned char tag = (unsigned char)0;
    struct Asn1_ t;
    if (!asn1_any_(in, &tag, &t, null)) { return 0; }
    int y = 0; int mo = 0; int d = 0; int h = 0; int mi = 0; int s = 0;
    unsafe {
        const unsigned char* p = t.p;
        if (tag == (unsigned char)ASN1_TAG_UTCTIME) {
            if (t.len != 13UL || p[12] != (unsigned char)'Z') { return 0; }
            if (!x5_digits_(p, 2UL, &y)) { return 0; }
            y = y < 50 ? 2000 + y : 1900 + y;
            p = p + 2;
        } else if (tag == (unsigned char)ASN1_TAG_GENTIME) {
            if (t.len != 15UL || p[14] != (unsigned char)'Z') { return 0; }
            if (!x5_digits_(p, 4UL, &y)) { return 0; }
            p = p + 4;
        } else {
            return 0;
        }
        if (!x5_digits_(p, 2UL, &mo) || !x5_digits_(p + 2, 2UL, &d) || !x5_digits_(p + 4, 2UL, &h) ||
            !x5_digits_(p + 6, 2UL, &mi) || !x5_digits_(p + 8, 2UL, &s)) { return 0; }
    }
    if (mo < 1 || mo > 12 || d < 1 || d > 31 || h > 23 || mi > 59 || s > 60) { return 0; }
    long long days = x5_days_(y, mo, d);
    long long secs = days * 86400LL + (long long)h * 3600LL + (long long)mi * 60LL + (long long)s;
    *out = secs < 0LL ? 0ULL : (unsigned long long)secs;
    return 1;
}

// ── Algorithm identifiers ─────────────────────────────────────────────────────

static int x5_hash_of_oid_(const &Asn1_ oid) {
    unsafe {
        if (asn1_oid_is_(oid, &x5_oid_sha256_[0], 9UL)) { return DIGEST_SHA256; }
        if (asn1_oid_is_(oid, &x5_oid_sha384_[0], 9UL)) { return DIGEST_SHA384; }
        if (asn1_oid_is_(oid, &x5_oid_sha512_[0], 9UL)) { return DIGEST_SHA512; }
    }
    return DIGEST_NONE;
}

// RSASSA-PSS-params: [0] hashAlgorithm, [1] maskGenAlgorithm (MGF1 with
// the same hash is all that's accepted), [2] saltLength, [3] trailerField.
static int x5_pss_params_(struct Asn1_ params, &X509Parsed out) {
    struct Asn1_ seq;
    if (!asn1_get_(&params, (unsigned char)ASN1_TAG_SEQUENCE, &seq)) { return 0; }
    int hash = DIGEST_SHA1;
    int salt = 20;
    struct Asn1_ f;
    if (asn1_opt_(&seq, (unsigned char)0xA0, &f) == 1) {
        struct Asn1_ alg;
        struct Asn1_ oid;
        if (!asn1_get_(&f, (unsigned char)ASN1_TAG_SEQUENCE, &alg) || !asn1_get_(&alg, (unsigned char)ASN1_TAG_OID, &oid)) { return 0; }
        hash = x5_hash_of_oid_(&oid);
    }
    if (asn1_opt_(&seq, (unsigned char)0xA1, &f) == 1) {
        struct Asn1_ alg;
        struct Asn1_ oid;
        struct Asn1_ inner;
        struct Asn1_ ioid;
        if (!asn1_get_(&f, (unsigned char)ASN1_TAG_SEQUENCE, &alg) || !asn1_get_(&alg, (unsigned char)ASN1_TAG_OID, &oid)) { return 0; }
        if (!asn1_get_(&alg, (unsigned char)ASN1_TAG_SEQUENCE, &inner) || !asn1_get_(&inner, (unsigned char)ASN1_TAG_OID, &ioid)) { return 0; }
        if (x5_hash_of_oid_(&ioid) != hash) { return 0; }
    }
    if (asn1_opt_(&seq, (unsigned char)0xA2, &f) == 1) {
        if (!asn1_small_int_(&f, &salt)) { return 0; }
    }
    out.sig_hash = hash;
    out.pss_salt = salt;
    return 1;
}

static int x5_sig_alg_(struct Asn1_ alg, &X509Parsed out) {
    struct Asn1_ oid;
    if (!asn1_get_(&alg, (unsigned char)ASN1_TAG_OID, &oid)) { return 0; }
    out.sig_type = X509_SIG_UNKNOWN;
    out.sig_hash = DIGEST_NONE;
    unsafe {
        if (asn1_oid_is_(&oid, &x5_oid_sha256rsa_[0], 9UL)) { out.sig_type = X509_SIG_RSA_PKCS1; out.sig_hash = DIGEST_SHA256; }
        else if (asn1_oid_is_(&oid, &x5_oid_sha384rsa_[0], 9UL)) { out.sig_type = X509_SIG_RSA_PKCS1; out.sig_hash = DIGEST_SHA384; }
        else if (asn1_oid_is_(&oid, &x5_oid_sha512rsa_[0], 9UL)) { out.sig_type = X509_SIG_RSA_PKCS1; out.sig_hash = DIGEST_SHA512; }
        else if (asn1_oid_is_(&oid, &x5_oid_sha1rsa_[0], 9UL)) { out.sig_type = X509_SIG_RSA_PKCS1; out.sig_hash = DIGEST_SHA1; }
        else if (asn1_oid_is_(&oid, &x5_oid_ecdsa256_[0], 8UL)) { out.sig_type = X509_SIG_ECDSA; out.sig_hash = DIGEST_SHA256; }
        else if (asn1_oid_is_(&oid, &x5_oid_ecdsa384_[0], 8UL)) { out.sig_type = X509_SIG_ECDSA; out.sig_hash = DIGEST_SHA384; }
        else if (asn1_oid_is_(&oid, &x5_oid_ecdsa512_[0], 8UL)) { out.sig_type = X509_SIG_ECDSA; out.sig_hash = DIGEST_SHA512; }
        else if (asn1_oid_is_(&oid, &x5_oid_ecdsa1_[0], 7UL)) { out.sig_type = X509_SIG_ECDSA; out.sig_hash = DIGEST_SHA1; }
        else if (asn1_oid_is_(&oid, &x5_oid_ed25519_[0], 3UL)) { out.sig_type = X509_SIG_ED25519; }
        else if (asn1_oid_is_(&oid, &x5_oid_rsa_pss_[0], 9UL)) {
            out.sig_type = X509_SIG_RSA_PSS;
            if (!x5_pss_params_(alg, out)) { return 0; }
        }
    }
    return 1;
}

static int x5_spki_(const unsigned char* base, struct Asn1_ spki, &X509Parsed out) {
    struct Asn1_ alg;
    struct Asn1_ oid;
    struct Asn1_ bits;
    if (!asn1_get_(&spki, (unsigned char)ASN1_TAG_SEQUENCE, &alg)) { return 0; }
    if (!asn1_get_(&alg, (unsigned char)ASN1_TAG_OID, &oid)) { return 0; }
    if (!asn1_get_(&spki, (unsigned char)ASN1_TAG_BIT_STRING, &bits) || bits.len < 1UL) { return 0; }
    unsafe {
        if (bits.p[0] != (unsigned char)0) { return 0; }
        bits.p = bits.p + 1;
        bits.len = bits.len - 1UL;
        out.key = x5_span_(base, &bits);
        out.key_type = X509_KEY_UNKNOWN;
        if (asn1_oid_is_(&oid, &x5_oid_rsa_[0], 9UL) || asn1_oid_is_(&oid, &x5_oid_rsa_pss_[0], 9UL)) {
            out.key_type = X509_KEY_RSA;
        } else if (asn1_oid_is_(&oid, &x5_oid_ecpk_[0], 7UL)) {
            struct Asn1_ curve;
            if (asn1_get_(&alg, (unsigned char)ASN1_TAG_OID, &curve)) {
                if (asn1_oid_is_(&curve, &x5_oid_p256_[0], 8UL)) { out.key_type = X509_KEY_EC_P256; }
                else if (asn1_oid_is_(&curve, &x5_oid_p384_[0], 5UL)) { out.key_type = X509_KEY_EC_P384; }
            }
        } else if (asn1_oid_is_(&oid, &x5_oid_ed25519_[0], 3UL)) {
            out.key_type = X509_KEY_ED25519;
        }
    }
    return 1;
}

static int x5_first_cn_(const unsigned char* base, struct Asn1_ name, &X509Parsed out) {
    struct Asn1_ seq;
    if (!asn1_get_(&name, (unsigned char)ASN1_TAG_SEQUENCE, &seq)) { return 0; }
    while (seq.len > 0UL) {
        struct Asn1_ set;
        if (!asn1_get_(&seq, (unsigned char)ASN1_TAG_SET, &set)) { return 0; }
        while (set.len > 0UL) {
            struct Asn1_ atv;
            struct Asn1_ oid;
            struct Asn1_ val;
            unsigned char tag = (unsigned char)0;
            if (!asn1_get_(&set, (unsigned char)ASN1_TAG_SEQUENCE, &atv)) { return 0; }
            if (!asn1_get_(&atv, (unsigned char)ASN1_TAG_OID, &oid)) { return 0; }
            if (!asn1_any_(&atv, &tag, &val, null)) { return 0; }
            unsafe {
                if (out.subject_cn.len == 0UL && asn1_oid_is_(&oid, &x5_oid_cn_[0], 3UL) &&
                    (tag == (unsigned char)ASN1_TAG_UTF8 || tag == (unsigned char)ASN1_TAG_PRINTABLE ||
                     tag == (unsigned char)ASN1_TAG_IA5 || tag == (unsigned char)ASN1_TAG_T61)) {
                    out.subject_cn = x5_span_(base, &val);
                }
            }
        }
    }
    return 1;
}

static int x5_extensions_(const unsigned char* base, struct Asn1_ exts, &X509Parsed out) {
    struct Asn1_ seq;
    if (!asn1_get_(&exts, (unsigned char)ASN1_TAG_SEQUENCE, &seq)) { return 0; }
    while (seq.len > 0UL) {
        struct Asn1_ ext;
        struct Asn1_ oid;
        struct Asn1_ val;
        struct Asn1_ crit;
        int critical = 0;
        if (!asn1_get_(&seq, (unsigned char)ASN1_TAG_SEQUENCE, &ext)) { return 0; }
        if (!asn1_get_(&ext, (unsigned char)ASN1_TAG_OID, &oid)) { return 0; }
        int r = asn1_opt_(&ext, (unsigned char)ASN1_TAG_BOOLEAN, &crit);
        if (r < 0) { return 0; }
        unsafe { if (r == 1 && crit.len == 1UL && crit.p[0] != (unsigned char)0) { critical = 1; } }
        if (!asn1_get_(&ext, (unsigned char)ASN1_TAG_OCTET_STRING, &val)) { return 0; }
        int known = 0;
        unsafe {
            if (oid.len == 3UL && oid.p[0] == (unsigned char)0x55 && oid.p[1] == (unsigned char)0x1d) {
                unsigned char id = oid.p[2];
                if (id == (unsigned char)0x11) {             // subjectAltName
                    struct Asn1_ names;
                    if (!asn1_get_(&val, (unsigned char)ASN1_TAG_SEQUENCE, &names)) { return 0; }
                    out.san = x5_span_(base, &names);
                    known = 1;
                } else if (id == (unsigned char)0x13) {      // basicConstraints
                    struct Asn1_ bc;
                    if (!asn1_get_(&val, (unsigned char)ASN1_TAG_SEQUENCE, &bc)) { return 0; }
                    out.has_basic_constraints = 1;
                    struct Asn1_ ca;
                    int rr = asn1_opt_(&bc, (unsigned char)ASN1_TAG_BOOLEAN, &ca);
                    if (rr < 0) { return 0; }
                    if (rr == 1 && ca.len == 1UL && ca.p[0] != (unsigned char)0) { out.is_ca = 1; }
                    if (bc.len > 0UL) {
                        int pl = 0;
                        if (!asn1_small_int_(&bc, &pl)) { return 0; }
                        out.path_len = pl;
                    }
                    known = 1;
                } else if (id == (unsigned char)0x0f) {      // keyUsage
                    struct Asn1_ ku;
                    if (!asn1_get_(&val, (unsigned char)ASN1_TAG_BIT_STRING, &ku) || ku.len < 2UL) { return 0; }
                    out.has_key_usage = 1;
                    out.key_usage = (int)ku.p[1];
                    if (ku.len > 2UL) { out.key_usage = out.key_usage | ((int)ku.p[2] << 8); }
                    known = 1;
                } else if (id == (unsigned char)0x25) {      // extKeyUsage
                    struct Asn1_ list;
                    if (!asn1_get_(&val, (unsigned char)ASN1_TAG_SEQUENCE, &list)) { return 0; }
                    out.has_eku = 1;
                    while (list.len > 0UL) {
                        struct Asn1_ k;
                        if (!asn1_get_(&list, (unsigned char)ASN1_TAG_OID, &k)) { return 0; }
                        if (asn1_oid_is_(&k, &x5_oid_eku_server_[0], 8UL)) { out.eku_server = 1; }
                        if (asn1_oid_is_(&k, &x5_oid_eku_client_[0], 8UL)) { out.eku_client = 1; }
                        if (asn1_oid_is_(&k, &x5_oid_eku_any_[0], 4UL)) { out.eku_any = 1; }
                    }
                    known = 1;
                } else if (id == (unsigned char)0x1e) {      // nameConstraints: noted, not processed
                    out.has_name_constraints = 1;
                } else if (id == (unsigned char)0x20 || id == (unsigned char)0x24 || id == (unsigned char)0x36 ||
                           id == (unsigned char)0x23 || id == (unsigned char)0x0e || id == (unsigned char)0x1f) {
                    // certificatePolicies, policyConstraints, inhibitAnyPolicy,
                    // authority/subject key ids, CRL distribution points: no
                    // effect on a path validated without policy processing.
                    known = 1;
                }
            } else if (asn1_oid_is_(&oid, &x5_oid_aia_[0], 8UL)) {
                known = 1;
            }
        }
        if (critical && !known) { out.unknown_critical = 1; }
    }
    return 1;
}

int x509_parse(const &unsigned char der, unsigned long len, &X509Parsed out) {
    unsafe {
        const unsigned char* base = (const unsigned char*)der;
        out.der_len = len;
        out.tbs = x5_none_(); out.serial = x5_none_(); out.issuer = x5_none_(); out.subject = x5_none_();
        out.spki = x5_none_(); out.key = x5_none_(); out.sig = x5_none_(); out.san = x5_none_();
        out.subject_cn = x5_none_();
        out.version = 1; out.key_type = X509_KEY_UNKNOWN; out.sig_type = X509_SIG_UNKNOWN; out.sig_hash = DIGEST_NONE;
        out.pss_salt = 0; out.not_before = 0ULL; out.not_after = 0ULL;
        out.has_basic_constraints = 0; out.is_ca = 0; out.path_len = -1; out.has_key_usage = 0; out.key_usage = 0;
        out.has_eku = 0; out.eku_server = 0; out.eku_client = 0; out.eku_any = 0;
        out.has_name_constraints = 0; out.unknown_critical = 0;

        struct Asn1_ in = asn1_view_(base, len);
        struct Asn1_ cert;
        if (!asn1_get_(&in, (unsigned char)ASN1_TAG_SEQUENCE, &cert) || in.len != 0UL) { return X509_ERR_MALFORMED; }
        struct Asn1_ tbsWhole;
        struct Asn1_ tbs;
        unsigned char tag = (unsigned char)0;
        if (!asn1_any_(&cert, &tag, &tbs, &tbsWhole) || tag != (unsigned char)ASN1_TAG_SEQUENCE) { return X509_ERR_MALFORMED; }
        out.tbs = x5_span_(base, &tbsWhole);
        struct Asn1_ outerAlg;
        struct Asn1_ outerAlgWhole;
        if (!asn1_any_(&cert, &tag, &outerAlg, &outerAlgWhole) || tag != (unsigned char)ASN1_TAG_SEQUENCE) { return X509_ERR_MALFORMED; }
        struct Asn1_ sigBits;
        if (!asn1_get_(&cert, (unsigned char)ASN1_TAG_BIT_STRING, &sigBits) || sigBits.len < 1UL ||
            sigBits.p[0] != (unsigned char)0 || cert.len != 0UL) { return X509_ERR_MALFORMED; }
        sigBits.p = sigBits.p + 1;
        sigBits.len = sigBits.len - 1UL;
        out.sig = x5_span_(base, &sigBits);
        if (!x5_sig_alg_(outerAlg, out)) { return X509_ERR_MALFORMED; }

        // TBSCertificate
        struct Asn1_ v;
        int r = asn1_opt_(&tbs, (unsigned char)0xA0, &v);
        if (r < 0) { return X509_ERR_MALFORMED; }
        if (r == 1) {
            int ver = 0;
            if (!asn1_small_int_(&v, &ver) || ver > 2) { return X509_ERR_MALFORMED; }
            out.version = ver + 1;
        }
        struct Asn1_ serial;
        if (!asn1_get_(&tbs, (unsigned char)ASN1_TAG_INTEGER, &serial)) { return X509_ERR_MALFORMED; }
        out.serial = x5_span_(base, &serial);
        struct Asn1_ innerAlg;
        struct Asn1_ innerAlgWhole;
        if (!asn1_any_(&tbs, &tag, &innerAlg, &innerAlgWhole) || tag != (unsigned char)ASN1_TAG_SEQUENCE) { return X509_ERR_MALFORMED; }
        if (!asn1_eq_(&innerAlgWhole, &outerAlgWhole)) { return X509_ERR_MALFORMED; }
        struct Asn1_ issuer;
        struct Asn1_ issuerWhole;
        if (!asn1_any_(&tbs, &tag, &issuer, &issuerWhole) || tag != (unsigned char)ASN1_TAG_SEQUENCE) { return X509_ERR_MALFORMED; }
        out.issuer = x5_span_(base, &issuerWhole);
        struct Asn1_ validity;
        if (!asn1_get_(&tbs, (unsigned char)ASN1_TAG_SEQUENCE, &validity)) { return X509_ERR_MALFORMED; }
        if (!x5_time_(&validity, &out.not_before) || !x5_time_(&validity, &out.not_after)) { return X509_ERR_MALFORMED; }
        struct Asn1_ subject;
        struct Asn1_ subjectWhole;
        if (!asn1_any_(&tbs, &tag, &subject, &subjectWhole) || tag != (unsigned char)ASN1_TAG_SEQUENCE) { return X509_ERR_MALFORMED; }
        out.subject = x5_span_(base, &subjectWhole);
        if (!x5_first_cn_(base, subjectWhole, out)) { return X509_ERR_MALFORMED; }
        struct Asn1_ spki;
        struct Asn1_ spkiWhole;
        if (!asn1_any_(&tbs, &tag, &spki, &spkiWhole) || tag != (unsigned char)ASN1_TAG_SEQUENCE) { return X509_ERR_MALFORMED; }
        out.spki = x5_span_(base, &spkiWhole);
        if (!x5_spki_(base, spki, out)) { return X509_ERR_MALFORMED; }
        // issuerUniqueID [1], subjectUniqueID [2] (skipped), extensions [3]
        while (tbs.len > 0UL) {
            struct Asn1_ f;
            if (!asn1_any_(&tbs, &tag, &f, null)) { return X509_ERR_MALFORMED; }
            if (tag == (unsigned char)0xA3) {
                if (!x5_extensions_(base, f, out)) { return X509_ERR_MALFORMED; }
            } else if (tag != (unsigned char)0x81 && tag != (unsigned char)0x82) {
                return X509_ERR_MALFORMED;
            }
        }
    }
    return 0;
}

static struct Asn1_ x5_view_(const &unsigned char der, const &X509Span s) {
    unsafe { return asn1_view_((const unsigned char*)der + s.off, s.len); }
}

int x509_issued_by(const &unsigned char der, const &X509Parsed c,
                   const &unsigned char issuer_der, const &X509Parsed issuer) {
    struct Asn1_ a = x5_view_(der, &c.issuer);
    struct Asn1_ b = x5_view_(issuer_der, &issuer.subject);
    return asn1_eq_(&a, &b);
}

int x509_verify_signed_by(const &unsigned char der, const &X509Parsed c,
                          const &unsigned char issuer_der, const &X509Parsed issuer) {
    if (c.sig_hash == DIGEST_NONE || c.sig_hash == DIGEST_SHA1) { return 0; }
    unsigned char h[DIGEST_MAX_SIZE];
    unsigned long hl = 0UL;
    unsafe { hl = digest(c.sig_hash, (const unsigned char*)der + c.tbs.off, c.tbs.len, &h[0]); }
    if (hl == 0UL) { return 0; }
    unsafe {
        const unsigned char* sig = (const unsigned char*)der + c.sig.off;
        const unsigned char* key = (const unsigned char*)issuer_der + issuer.key.off;
        if (c.sig_type == X509_SIG_RSA_PKCS1 || c.sig_type == X509_SIG_RSA_PSS) {
            if (issuer.key_type != X509_KEY_RSA) { return 0; }
            struct RsaPublicKey k;
            if (!k.init_der((const &unsigned char)key, issuer.key.len)) { return 0; }
            if (c.sig_type == X509_SIG_RSA_PKCS1) {
                return rsa_verify_pkcs1(&k, c.sig_hash, &h[0], hl, (const &unsigned char)sig, c.sig.len);
            }
            if (c.pss_salt < 0) { return 0; }
            return rsa_verify_pss(&k, c.sig_hash, &h[0], hl, (const &unsigned char)sig, c.sig.len, (unsigned long)c.pss_salt);
        }
        if (c.sig_type == X509_SIG_ECDSA) {
            int curve = 0;
            if (issuer.key_type == X509_KEY_EC_P256) { curve = EC_P256; }
            else if (issuer.key_type == X509_KEY_EC_P384) { curve = EC_P384; }
            else { return 0; }
            return ecdsa_verify_der(curve, (const &unsigned char)key, issuer.key.len, &h[0], hl,
                                    (const &unsigned char)sig, c.sig.len);
        }
    }
    return 0;
}

// ── Host matching ─────────────────────────────────────────────────────────────

static unsigned char x5_lower_(unsigned char c) {
    if (c >= (unsigned char)'A' && c <= (unsigned char)'Z') { return (unsigned char)(c + (unsigned char)32); }
    return c;
}

// Parses an IPv4 dotted quad into 4 bytes.
static int x5_ipv4_(const char* s, unsigned long n, unsigned char* out) {
    int part = 0;
    int val = -1;
    unsafe {
        unsigned long i = 0UL;
        while (i <= n) {
            char ch = i < n ? s[i] : '.';
            if (ch == '.') {
                if (val < 0 || part > 3) { return 0; }
                out[part] = (unsigned char)val;
                part = part + 1;
                val = -1;
            } else if (ch >= '0' && ch <= '9') {
                val = (val < 0 ? 0 : val * 10) + (int)(ch - '0');
                if (val > 255) { return 0; }
            } else {
                return 0;
            }
            i = i + 1UL;
        }
    }
    return part == 4;
}

static int x5_hexv_(char c) {
    if (c >= '0' && c <= '9') { return (int)(c - '0'); }
    if (c >= 'a' && c <= 'f') { return (int)(c - 'a') + 10; }
    if (c >= 'A' && c <= 'F') { return (int)(c - 'A') + 10; }
    return -1;
}

// Parses an IPv6 literal (optionally in brackets, '::' compression, no
// embedded IPv4 or zone) into 16 bytes.
static int x5_ipv6_(const char* s, unsigned long n, unsigned char* out) {
    unsigned int groups[8];
    int count = 0;
    int gap = -1;
    unsafe {
        if (n >= 2UL && s[0] == '[' && s[n - 1UL] == ']') { s = s + 1; n = n - 2UL; }
        unsigned long i = 0UL;
        if (n >= 2UL && s[0] == ':' && s[1] == ':') { gap = 0; i = 2UL; }
        while (i < n) {
            unsigned int v = 0U;
            int digits = 0;
            while (i < n && x5_hexv_(s[i]) >= 0) {
                v = (v << 4) | (unsigned int)x5_hexv_(s[i]);
                digits = digits + 1;
                i = i + 1UL;
            }
            if (digits == 0 || digits > 4 || count >= 8) { return 0; }
            groups[count] = v;
            count = count + 1;
            if (i == n) { break; }
            if (s[i] != ':') { return 0; }
            i = i + 1UL;
            if (i < n && s[i] == ':') {
                if (gap >= 0) { return 0; }
                gap = count;
                i = i + 1UL;
            } else if (i == n) {
                return 0;
            }
        }
    }
    if (gap < 0 && count != 8) { return 0; }
    if (gap >= 0 && count > 7) { return 0; }
    unsafe {
        int k = 0;
        while (k < 16) { out[k] = (unsigned char)0; k = k + 1; }
        int g = 0;
        int pos = 0;
        while (g < count) {
            if (g == gap) { pos = pos + (8 - count); }
            out[pos * 2] = (unsigned char)(groups[g] >> 8);
            out[pos * 2 + 1] = (unsigned char)(groups[g] & 0xFFU);
            pos = pos + 1;
            g = g + 1;
        }
    }
    return 1;
}

// DNS name (pattern p, len pn) against host h (len hn, lowercased, no
// trailing dot). A leading "*." matches exactly one non-empty label, and
// only with at least two more labels after it.
static int x5_dns_match_(const unsigned char* p, unsigned long pn, const unsigned char* h, unsigned long hn) {
    unsafe {
        if (pn > 0UL && p[pn - 1UL] == (unsigned char)'.') { pn = pn - 1UL; }
        if (pn == 0UL) { return 0; }
        if (pn >= 2UL && p[0] == (unsigned char)'*' && p[1] == (unsigned char)'.') {
            const unsigned char* rest = p + 1;      // ".example.com"
            unsigned long rn = pn - 1UL;
            unsigned long dots = 0UL;
            unsigned long i = 1UL;
            while (i < rn) {
                if (rest[i] == (unsigned char)'*') { return 0; }
                if (rest[i] == (unsigned char)'.') { dots = dots + 1UL; }
                i = i + 1UL;
            }
            if (dots < 1UL) { return 0; }
            // the host's first label, then the same suffix
            unsigned long first = 0UL;
            while (first < hn && h[first] != (unsigned char)'.') { first = first + 1UL; }
            if (first == 0UL || first == hn) { return 0; }
            if (hn - first != rn) { return 0; }
            i = 0UL;
            while (i < rn) {
                if (x5_lower_(rest[i]) != h[first + i]) { return 0; }
                i = i + 1UL;
            }
            return 1;
        }
        if (pn != hn) { return 0; }
        unsigned long i = 0UL;
        while (i < pn) {
            if (p[i] == (unsigned char)'*') { return 0; }
            if (x5_lower_(p[i]) != h[i]) { return 0; }
            i = i + 1UL;
        }
    }
    return 1;
}

int x509_match_host(const &unsigned char der, const &X509Parsed c, const &char host) {
    unsigned char h[256];
    unsigned long hn = 0UL;
    unsafe {
        const char* s = (const char*)host;
        while (s[hn] != (char)0) {
            if (hn >= 255UL) { return 0; }
            h[hn] = x5_lower_((unsigned char)s[hn]);
            hn = hn + 1UL;
        }
    }
    if (hn == 0UL) { return 0; }
    if (h[hn - 1UL] == (unsigned char)'.') { hn = hn - 1UL; }
    unsigned char ip[16];
    unsigned long ipLen = 0UL;
    unsafe {
        if (x5_ipv4_((const char*)&h[0], hn, &ip[0])) { ipLen = 4UL; }
        else if (x5_ipv6_((const char*)&h[0], hn, &ip[0])) { ipLen = 16UL; }
    }
    int sawName = 0;
    if (c.san.len > 0UL) {
        struct Asn1_ names = x5_view_(der, &c.san);
        while (names.len > 0UL) {
            unsigned char tag = (unsigned char)0;
            struct Asn1_ gn;
            if (!asn1_any_(&names, &tag, &gn, null)) { return 0; }
            unsafe {
                if (tag == (unsigned char)0x82) {
                    sawName = 1;
                    if (ipLen == 0UL && x5_dns_match_(gn.p, gn.len, (const unsigned char*)&h[0], hn)) { return 1; }
                } else if (tag == (unsigned char)0x87) {
                    sawName = 1;
                    if (ipLen != 0UL && gn.len == ipLen) {
                        unsigned long i = 0UL;
                        int same = 1;
                        while (i < ipLen) { if (gn.p[i] != ip[i]) { same = 0; } i = i + 1UL; }
                        if (same) { return 1; }
                    }
                }
            }
        }
    }
    if (sawName || ipLen != 0UL || c.subject_cn.len == 0UL) { return 0; }
    unsafe {
        return x5_dns_match_((const unsigned char*)der + c.subject_cn.off, c.subject_cn.len, (const unsigned char*)&h[0], hn);
    }
}

} // namespace std
