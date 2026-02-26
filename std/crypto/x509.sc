// SafeC Standard Library — X.509 Certificate Parsing (DER / ASN.1)
#pragma once
#include "x509.h"

extern void* memcpy(void* d, const void* s, unsigned long n);
extern void* memset(void* p, int v, unsigned long n);

// ── ASN.1 tag constants ────────────────────────────────────────────────────────

#define ASN1_BOOLEAN          0x01
#define ASN1_INTEGER          0x02
#define ASN1_BIT_STRING       0x03
#define ASN1_OCTET_STRING     0x04
#define ASN1_OID              0x06
#define ASN1_UTF8STRING       0x0C
#define ASN1_PRINTABLESTRING  0x13
#define ASN1_IA5STRING        0x16
#define ASN1_UTCTIME          0x17
#define ASN1_GENERALIZEDTIME  0x18
#define ASN1_BMPSTRING        0x1E
#define ASN1_SEQUENCE         0x30
#define ASN1_SET              0x31
#define ASN1_CONTEXT_0        0xA0
#define ASN1_CONTEXT_1        0xA1
#define ASN1_CONTEXT_2        0xA2
#define ASN1_CONTEXT_3        0xA3

// OID bytes for id-ce-subjectAltName (2.5.29.17)
static unsigned char oid_san_[3] = { 0x55, 0x1d, 0x11 };

// OID bytes for id-ce-basicConstraints (2.5.29.19)
static unsigned char oid_bc_[3]  = { 0x55, 0x1d, 0x13 };

// OID bytes for id-ce-keyUsage (2.5.29.15)
static unsigned char oid_ku_[3]  = { 0x55, 0x1d, 0x0f };

// ── Low-level string helpers ───────────────────────────────────────────────────

static int str_len_(const char* s) {
    int n = 0;
    unsafe { while (s[n] != (char)0) { n = n + 1; } }
    return n;
}

static int str_eq_(const char* a, const char* b) {
    int i = 0;
    unsafe {
        while (a[i] != (char)0 && b[i] != (char)0) {
            if (a[i] != b[i]) { return 0; }
            i = i + 1;
        }
        return (a[i] == (char)0 && b[i] == (char)0) ? 1 : 0;
    }
}

// Copy at most `max-1` bytes from src (not null-terminated) into dst, add null.
static void copy_str_(char* dst, const unsigned char* src, unsigned long srclen,
                      unsigned long max) {
    unsigned long n = srclen;
    if (n >= max) { n = max - (unsigned long)1; }
    unsafe {
        unsigned long i = (unsigned long)0;
        while (i < n) { dst[i] = (char)src[i]; i = i + (unsigned long)1; }
        dst[n] = (char)0;
    }
}

// ── ASN.1 TLV decoder ─────────────────────────────────────────────────────────

// Reads the tag-length header at buf[*offset].
// Sets *content_start to the first byte of content and *content_len to the
// number of content bytes.  Returns 0 on success, -1 on truncation.
// tag_out receives the raw tag byte.
static int asn1_tag_len_(const unsigned char* buf, unsigned long buf_len,
                          unsigned long* offset,
                          unsigned char* tag_out,
                          unsigned long* content_start,
                          unsigned long* content_len) {
    unsafe {
        if (*offset >= buf_len) { return -1; }
        *tag_out = buf[*offset];
        *offset  = *offset + (unsigned long)1;

        if (*offset >= buf_len) { return -1; }
        unsigned char lb = buf[*offset];
        *offset = *offset + (unsigned long)1;

        if ((lb & (unsigned char)0x80) == (unsigned char)0) {
            // Short form: low 7 bits = length
            *content_len   = (unsigned long)lb;
        } else {
            // Long form: low 7 bits = number of subsequent length bytes
            unsigned long num_bytes = (unsigned long)(lb & (unsigned char)0x7F);
            if (num_bytes == (unsigned long)0 || num_bytes > (unsigned long)4) {
                return -1;  // indefinite or too large
            }
            *content_len = (unsigned long)0;
            unsigned long k = (unsigned long)0;
            while (k < num_bytes) {
                if (*offset >= buf_len) { return -1; }
                *content_len = (*content_len << (unsigned long)8)
                             | (unsigned long)buf[*offset];
                *offset = *offset + (unsigned long)1;
                k = k + (unsigned long)1;
            }
        }
        *content_start = *offset;
        if (*content_start + *content_len > buf_len) { return -1; }
        return 0;
    }
}

// Skip past a TLV field (advance *offset to byte after it).
static int asn1_skip_(const unsigned char* buf, unsigned long buf_len,
                       unsigned long* offset) {
    unsigned char tag_out;
    unsigned long cstart;
    unsigned long clen;
    int rc = asn1_tag_len_(buf, buf_len, offset, &tag_out, &cstart, &clen);
    if (rc != 0) { return rc; }
    unsafe { *offset = cstart + clen; }
    return 0;
}

// ── OID comparison ─────────────────────────────────────────────────────────────

static int oid_eq_(const unsigned char* oid_content, unsigned long oid_len,
                   const unsigned char* expected, unsigned long expected_len) {
    if (oid_len != expected_len) { return 0; }
    unsigned long i = (unsigned long)0;
    unsafe {
        while (i < oid_len) {
            if (oid_content[i] != expected[i]) { return 0; }
            i = i + (unsigned long)1;
        }
    }
    return 1;
}

// ── UTCTime / GeneralizedTime → Unix timestamp ────────────────────────────────

// Parses a 2-digit decimal number from s[0..1].
static int parse2_(const unsigned char* s) {
    unsafe {
        return ((int)(s[0] - (unsigned char)0x30)) * 10
             +  (int)(s[1] - (unsigned char)0x30);
    }
}

// Days in month (non-leap-year).
static int days_in_month_(int m) {
    int dm[12] = { 31,28,31,30,31,30,31,31,30,31,30,31 };
    if (m < 1 || m > 12) { return 30; }
    return dm[m - 1];
}

static int is_leap_(int y) {
    return ((y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)) ? 1 : 0;
}

// Very simplified epoch conversion (ignores sub-second, assumes UTC Z suffix).
static unsigned long long date_to_epoch_(int year, int month, int day,
                                          int hour, int min, int sec) {
    // Days from 1970-01-01 to year-01-01
    int y = 1970;
    unsigned long long days = (unsigned long long)0;
    while (y < year) {
        days = days + (unsigned long long)(is_leap_(y) != 0 ? 366 : 365);
        y = y + 1;
    }
    int m = 1;
    while (m < month) {
        int d = days_in_month_(m);
        if (m == 2 && is_leap_(year) != 0) { d = d + 1; }
        days = days + (unsigned long long)d;
        m = m + 1;
    }
    days = days + (unsigned long long)(day - 1);
    return days * (unsigned long long)86400
         + (unsigned long long)hour * (unsigned long long)3600
         + (unsigned long long)min  * (unsigned long long)60
         + (unsigned long long)sec;
}

// Parse UTCTime "YYMMDDHHMMSSZ" (13 bytes) to Unix timestamp.
static unsigned long long utctime_to_epoch_(const unsigned char* s,
                                             unsigned long len) {
    if (len < (unsigned long)13) { return (unsigned long long)0; }
    int yy    = parse2_(s);
    int year  = (yy >= 50) ? (1900 + yy) : (2000 + yy);
    int month = parse2_(s + 2);
    int day   = parse2_(s + 4);
    int hour  = parse2_(s + 6);
    int min   = parse2_(s + 8);
    int sec   = parse2_(s + 10);
    return date_to_epoch_(year, month, day, hour, min, sec);
}

// Parse GeneralizedTime "YYYYMMDDHHMMSSZ" (15 bytes) to Unix timestamp.
static unsigned long long gentime_to_epoch_(const unsigned char* s,
                                             unsigned long len) {
    if (len < (unsigned long)15) { return (unsigned long long)0; }
    unsafe {
        int year = ((int)(s[0]-0x30))*1000 + ((int)(s[1]-0x30))*100
                 + ((int)(s[2]-0x30))*10   +  (int)(s[3]-0x30);
        int month = parse2_(s + 4);
        int day   = parse2_(s + 6);
        int hour  = parse2_(s + 8);
        int min   = parse2_(s + 10);
        int sec   = parse2_(s + 12);
        return date_to_epoch_(year, month, day, hour, min, sec);
    }
}

// ── Name parsing ──────────────────────────────────────────────────────────────

// OID for CN  = 2.5.4.3  → {0x55, 0x04, 0x03}
// OID for O   = 2.5.4.10 → {0x55, 0x04, 0x0A}
// OID for C   = 2.5.4.6  → {0x55, 0x04, 0x06}
static unsigned char oid_cn_[3] = { 0x55, 0x04, 0x03 };
static unsigned char oid_o_[3]  = { 0x55, 0x04, 0x0A };
static unsigned char oid_c_[3]  = { 0x55, 0x04, 0x06 };

// Parse a SEQUENCE OF SET OF { OID, string } name structure into X509Name.
static int parse_name_(const unsigned char* buf, unsigned long buf_len,
                        unsigned long name_start, unsigned long name_len,
                        &stack X509Name name) {
    unsigned long pos = name_start;
    unsigned long end = name_start + name_len;

    while (pos < end) {
        // Each RDN is a SET
        unsigned char set_tag;
        unsigned long set_start;
        unsigned long set_len;
        if (asn1_tag_len_(buf, buf_len, &pos, &set_tag, &set_start, &set_len) != 0) {
            return -1;
        }
        unsigned long set_end = set_start + set_len;
        pos = set_start;

        while (pos < set_end) {
            // AttributeTypeAndValue SEQUENCE
            unsigned char atv_tag;
            unsigned long atv_start;
            unsigned long atv_len;
            if (asn1_tag_len_(buf, buf_len, &pos, &atv_tag, &atv_start, &atv_len) != 0) {
                return -1;
            }
            pos = atv_start;

            // OID
            unsigned char oid_tag;
            unsigned long oid_start;
            unsigned long oid_len;
            if (asn1_tag_len_(buf, buf_len, &pos, &oid_tag, &oid_start, &oid_len) != 0) {
                return -1;
            }
            pos = oid_start + oid_len;

            // String value (any supported string type)
            unsigned char val_tag;
            unsigned long val_start;
            unsigned long val_len;
            if (asn1_tag_len_(buf, buf_len, &pos, &val_tag, &val_start, &val_len) != 0) {
                return -1;
            }
            pos = val_start + val_len;

            // Match OID and copy string
            unsafe {
                const unsigned char* oid_bytes = buf + oid_start;
                const unsigned char* val_bytes = buf + val_start;
                if (oid_eq_(oid_bytes, oid_len, (const unsigned char*)oid_cn_, (unsigned long)3) != 0) {
                    copy_str_((char*)name.cn, val_bytes, val_len,
                              (unsigned long)X509_MAX_NAME_LEN);
                } else if (oid_eq_(oid_bytes, oid_len, (const unsigned char*)oid_o_, (unsigned long)3) != 0) {
                    copy_str_((char*)name.o, val_bytes, val_len,
                              (unsigned long)X509_MAX_NAME_LEN);
                } else if (oid_eq_(oid_bytes, oid_len, (const unsigned char*)oid_c_, (unsigned long)3) != 0) {
                    unsigned long n = val_len;
                    if (n > (unsigned long)3) { n = (unsigned long)3; }
                    unsigned long k = (unsigned long)0;
                    while (k < n) {
                        name.c[k] = (char)val_bytes[k];
                        k = k + (unsigned long)1;
                    }
                    name.c[n] = (char)0;
                }
            }

            // Advance to end of this ATV
            if (pos < atv_start + atv_len) { pos = atv_start + atv_len; }
        }

        if (pos < set_end) { pos = set_end; }
    }
    return 0;
}

// ── SAN extension parsing ─────────────────────────────────────────────────────

static int parse_san_(const unsigned char* buf, unsigned long buf_len,
                       unsigned long ext_val_start, unsigned long ext_val_len,
                       &stack X509San san) {
    // ext_val is OCTET STRING wrapping a SEQUENCE OF GeneralName
    unsigned long pos = ext_val_start;
    unsigned long end = ext_val_start + ext_val_len;

    // Skip outer OCTET STRING wrapper if present (we are already inside it from
    // the caller which advanced past the OCTET STRING header)
    // Parse SEQUENCE OF GeneralName
    unsigned char seq_tag;
    unsigned long seq_start;
    unsigned long seq_len;
    if (asn1_tag_len_(buf, buf_len, &pos, &seq_tag, &seq_start, &seq_len) != 0) {
        return -1;
    }
    pos = seq_start;
    unsigned long seq_end = seq_start + seq_len;

    while (pos < seq_end && san.count < X509_MAX_SAN) {
        unsigned char gn_tag;
        unsigned long gn_start;
        unsigned long gn_len;
        if (asn1_tag_len_(buf, buf_len, &pos, &gn_tag, &gn_start, &gn_len) != 0) {
            return -1;
        }
        // dNSName [2] IMPLICIT IA5String
        if ((gn_tag & (unsigned char)0x1F) == (unsigned char)2) {
            unsafe {
                copy_str_((char*)san.names[san.count], buf + gn_start, gn_len,
                           (unsigned long)X509_MAX_SAN_LEN);
            }
            san.count = san.count + 1;
        }
        pos = gn_start + gn_len;
    }
    return 0;
}

// ── Main DER parser ───────────────────────────────────────────────────────────

int x509_parse_der(const unsigned char* der, unsigned long len,
                   &stack X509Cert cert_out) {
    unsafe {
        memset((void*)&cert_out, 0, (unsigned long)sizeof(struct X509Cert));
    }

    unsigned long pos = (unsigned long)0;

    // Outer SEQUENCE (Certificate)
    unsigned char tag;
    unsigned long cstart;
    unsigned long clen;
    if (asn1_tag_len_(der, len, &pos, &tag, &cstart, &clen) != 0) { return -1; }
    if (tag != (unsigned char)ASN1_SEQUENCE) { return -1; }
    pos = cstart;

    // TBSCertificate SEQUENCE
    unsigned char tbs_tag;
    unsigned long tbs_start;
    unsigned long tbs_len;
    if (asn1_tag_len_(der, len, &pos, &tbs_tag, &tbs_start, &tbs_len) != 0) { return -1; }
    if (tbs_tag != (unsigned char)ASN1_SEQUENCE) { return -1; }

    unsigned long tbs_end = tbs_start + tbs_len;
    pos = tbs_start;

    // Optional version [0] EXPLICIT INTEGER
    unsafe {
        if (pos < tbs_end && der[pos] == (unsigned char)ASN1_CONTEXT_0) {
            if (asn1_skip_(der, len, &pos) != 0) { return -1; }
        }
    }

    // serialNumber INTEGER
    unsigned char sn_tag;
    unsigned long sn_start;
    unsigned long sn_len;
    if (asn1_tag_len_(der, len, &pos, &sn_tag, &sn_start, &sn_len) != 0) { return -1; }
    unsafe {
        unsigned long serial_copy = sn_len;
        if (serial_copy > (unsigned long)20) { serial_copy = (unsigned long)20; }
        memcpy((void*)cert_out.serial, (const void*)(der + sn_start), serial_copy);
        cert_out.serial_len = serial_copy;
    }
    pos = sn_start + sn_len;

    // signature AlgorithmIdentifier SEQUENCE (skip)
    if (asn1_skip_(der, len, &pos) != 0) { return -1; }

    // issuer Name SEQUENCE
    unsigned char iss_tag;
    unsigned long iss_start;
    unsigned long iss_len;
    if (asn1_tag_len_(der, len, &pos, &iss_tag, &iss_start, &iss_len) != 0) { return -1; }
    if (parse_name_(der, len, iss_start, iss_len, cert_out.issuer) != 0) { return -1; }
    pos = iss_start + iss_len;

    // validity SEQUENCE
    unsigned char val_tag;
    unsigned long val_start;
    unsigned long val_len;
    if (asn1_tag_len_(der, len, &pos, &val_tag, &val_start, &val_len) != 0) { return -1; }
    {
        unsigned long vpos = val_start;

        // notBefore
        unsigned char nb_tag;
        unsigned long nb_start;
        unsigned long nb_len;
        if (asn1_tag_len_(der, len, &vpos, &nb_tag, &nb_start, &nb_len) != 0) { return -1; }
        unsafe {
            if (nb_tag == (unsigned char)ASN1_UTCTIME) {
                cert_out.validity.not_before = utctime_to_epoch_(der + nb_start, nb_len);
            } else {
                cert_out.validity.not_before = gentime_to_epoch_(der + nb_start, nb_len);
            }
        }
        vpos = nb_start + nb_len;

        // notAfter
        unsigned char na_tag;
        unsigned long na_start;
        unsigned long na_len;
        if (asn1_tag_len_(der, len, &vpos, &na_tag, &na_start, &na_len) != 0) { return -1; }
        unsafe {
            if (na_tag == (unsigned char)ASN1_UTCTIME) {
                cert_out.validity.not_after = utctime_to_epoch_(der + na_start, na_len);
            } else {
                cert_out.validity.not_after = gentime_to_epoch_(der + na_start, na_len);
            }
        }
    }
    pos = val_start + val_len;

    // subject Name SEQUENCE
    unsigned char subj_tag;
    unsigned long subj_start;
    unsigned long subj_len;
    if (asn1_tag_len_(der, len, &pos, &subj_tag, &subj_start, &subj_len) != 0) { return -1; }
    if (parse_name_(der, len, subj_start, subj_len, cert_out.subject) != 0) { return -1; }
    pos = subj_start + subj_len;

    // subjectPublicKeyInfo SEQUENCE
    unsigned char spki_tag;
    unsigned long spki_start;
    unsigned long spki_len;
    if (asn1_tag_len_(der, len, &pos, &spki_tag, &spki_start, &spki_len) != 0) { return -1; }
    unsafe {
        unsigned long pk_copy = spki_len;
        if (pk_copy > (unsigned long)X509_KEY_LEN) { pk_copy = (unsigned long)X509_KEY_LEN; }
        memcpy((void*)cert_out.pubkey, (const void*)(der + spki_start), pk_copy);
        cert_out.pubkey_len = pk_copy;
    }
    pos = spki_start + spki_len;

    // Skip optional issuerUniqueID [1] and subjectUniqueID [2]
    unsafe {
        while (pos < tbs_end &&
               (der[pos] == (unsigned char)ASN1_CONTEXT_1 ||
                der[pos] == (unsigned char)ASN1_CONTEXT_2)) {
            if (asn1_skip_(der, len, &pos) != 0) { return -1; }
        }
    }

    // Extensions [3] EXPLICIT SEQUENCE OF Extension (optional)
    unsafe {
        if (pos < tbs_end && der[pos] == (unsigned char)ASN1_CONTEXT_3) {
            unsigned char ext_ctx_tag;
            unsigned long ext_ctx_start;
            unsigned long ext_ctx_len;
            if (asn1_tag_len_(der, len, &pos, &ext_ctx_tag,
                               &ext_ctx_start, &ext_ctx_len) != 0) { return -1; }

            // Inner SEQUENCE OF Extension
            unsigned long epos = ext_ctx_start;
            unsigned char exts_tag;
            unsigned long exts_start;
            unsigned long exts_len;
            if (asn1_tag_len_(der, len, &epos, &exts_tag,
                               &exts_start, &exts_len) != 0) { return -1; }

            unsigned long exts_end = exts_start + exts_len;
            epos = exts_start;

            while (epos < exts_end) {
                // Extension SEQUENCE
                unsigned char ext_tag;
                unsigned long ext_start;
                unsigned long ext_len;
                if (asn1_tag_len_(der, len, &epos, &ext_tag,
                                   &ext_start, &ext_len) != 0) { return -1; }

                unsigned long ext_end = ext_start + ext_len;
                unsigned long ipos    = ext_start;

                // extnID OID
                unsigned char oid_tag2;
                unsigned long oid_start2;
                unsigned long oid_len2;
                if (asn1_tag_len_(der, len, &ipos, &oid_tag2,
                                   &oid_start2, &oid_len2) != 0) { return -1; }
                ipos = oid_start2 + oid_len2;

                // Optional critical BOOLEAN
                if (ipos < ext_end && der[ipos] == (unsigned char)ASN1_BOOLEAN) {
                    if (asn1_skip_(der, len, &ipos) != 0) { return -1; }
                }

                // extnValue OCTET STRING
                unsigned char ev_tag;
                unsigned long ev_start;
                unsigned long ev_len;
                if (asn1_tag_len_(der, len, &ipos, &ev_tag,
                                   &ev_start, &ev_len) != 0) { return -1; }

                const unsigned char* oid_bytes = der + oid_start2;

                if (oid_eq_(oid_bytes, oid_len2,
                             (const unsigned char*)oid_san_, (unsigned long)3) != 0) {
                    parse_san_(der, len, ev_start, ev_len, cert_out.san);
                } else if (oid_eq_(oid_bytes, oid_len2,
                                    (const unsigned char*)oid_bc_, (unsigned long)3) != 0) {
                    // BasicConstraints: SEQUENCE { cA BOOLEAN OPTIONAL, ... }
                    unsigned long bpos = ev_start;
                    unsigned char bc_seq_tag;
                    unsigned long bc_seq_start;
                    unsigned long bc_seq_len;
                    if (asn1_tag_len_(der, len, &bpos, &bc_seq_tag,
                                       &bc_seq_start, &bc_seq_len) == 0) {
                        unsigned long bcpos = bc_seq_start;
                        if (bcpos < bc_seq_start + bc_seq_len &&
                            der[bcpos] == (unsigned char)ASN1_BOOLEAN) {
                            unsigned char bt;
                            unsigned long bs;
                            unsigned long bl;
                            if (asn1_tag_len_(der, len, &bcpos, &bt, &bs, &bl) == 0
                                && bl >= (unsigned long)1) {
                                cert_out.is_ca = (der[bs] != (unsigned char)0) ? 1 : 0;
                            }
                        }
                    }
                } else if (oid_eq_(oid_bytes, oid_len2,
                                    (const unsigned char*)oid_ku_, (unsigned long)3) != 0) {
                    // KeyUsage: BIT STRING inside OCTET STRING
                    unsigned long kpos = ev_start;
                    unsigned char ks_tag;
                    unsigned long ks_start;
                    unsigned long ks_len;
                    if (asn1_tag_len_(der, len, &kpos, &ks_tag,
                                       &ks_start, &ks_len) == 0
                        && ks_len >= (unsigned long)2) {
                        // First byte = unused bits count; second byte = usage bits
                        cert_out.key_usage = (int)der[ks_start + 1];
                    }
                }

                epos = ext_end;
            }
        }
    }

    return 0;
}

// ── X509Cert methods ──────────────────────────────────────────────────────────

int X509Cert::is_valid_at(unsigned long long ts) const {
    return (ts >= self.validity.not_before && ts <= self.validity.not_after)
           ? 1 : 0;
}

int X509Cert::is_ca_cert() const {
    return self.is_ca;
}

// Wildcard match: *.example.com matches sub.example.com but not a.b.example.com
static int wildcard_match_(const char* pattern, const char* hostname) {
    // pattern must start with "*."
    unsafe {
        if (pattern[0] != (char)'*' || pattern[1] != (char)'.') { return 0; }
        // Find first '.' in hostname
        int i = 0;
        while (hostname[i] != (char)'.' && hostname[i] != (char)0) { i = i + 1; }
        if (hostname[i] == (char)0) { return 0; }
        // suffix of hostname after the first '.' must match pattern+1
        const char* hs = hostname + i;      // ".example.com"
        const char* ps = pattern  + 1;      // ".example.com"
        return str_eq_(hs, ps);
    }
}

int X509Cert::matches_hostname(const char* hostname) const {
    int i = 0;
    while (i < self.san.count) {
        unsafe {
            const char* name = (const char*)self.san.names[i];
            if (name[0] == (char)'*') {
                if (wildcard_match_(name, hostname) != 0) { return 1; }
            } else {
                if (str_eq_(name, hostname) != 0) { return 1; }
            }
        }
        i = i + 1;
    }
    // Fallback: check CN
    unsafe {
        const char* cn = (const char*)self.subject.cn;
        if (cn[0] == (char)'*') {
            if (wildcard_match_(cn, hostname) != 0) { return 1; }
        } else {
            if (str_eq_(cn, hostname) != 0) { return 1; }
        }
    }
    return 0;
}

// ── Chain verification (structural only) ──────────────────────────────────────

int x509_verify_chain(const &stack X509Cert cert,
                       const &stack X509Cert issuer_cert) {
    // Check that cert.issuer CN/O/C match issuer_cert.subject CN/O/C
    unsafe {
        if (str_eq_((const char*)cert.issuer.cn,
                    (const char*)issuer_cert.subject.cn) == 0) { return 0; }
        if (str_eq_((const char*)cert.issuer.o,
                    (const char*)issuer_cert.subject.o) == 0) { return 0; }
        if (str_eq_((const char*)cert.issuer.c,
                    (const char*)issuer_cert.subject.c) == 0) { return 0; }
    }
    return 1;
}
