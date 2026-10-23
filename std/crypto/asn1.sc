// SafeC Standard Library — minimal DER reader (see asn1.h).
#pragma once
#include <std/crypto/asn1.h>

namespace std {

struct Asn1_ asn1_view_(const unsigned char* p, unsigned long len) {
    struct Asn1_ v;
    v.p = p;
    v.len = len;
    return v;
}

int asn1_peek_(const &Asn1_ in) {
    if (in.len == 0UL) { return -1; }
    unsafe { return (int)in.p[0]; }
}

int asn1_any_(&Asn1_ in, &unsigned char tag, &Asn1_ content, ?&Asn1_ whole) {
    if (in.len < 2UL) { return 0; }
    unsafe {
        const unsigned char* p = in.p;
        *tag = p[0];
        if ((p[0] & (unsigned char)0x1F) == (unsigned char)0x1F) { return 0; }  // high tag numbers: unused here
        unsigned long hdr = 2UL;
        unsigned long clen = 0UL;
        unsigned char lb = p[1];
        if ((lb & (unsigned char)0x80) == (unsigned char)0) {
            clen = (unsigned long)lb;
        } else {
            unsigned long nb = (unsigned long)(lb & (unsigned char)0x7F);
            if (nb == 0UL || nb > 4UL || in.len < 2UL + nb) { return 0; }
            unsigned long k = 0UL;
            while (k < nb) { clen = (clen << 8UL) | (unsigned long)p[2UL + k]; k = k + 1UL; }
            hdr = 2UL + nb;
        }
        if (clen > in.len - hdr) { return 0; }
        content.p = p + hdr;
        content.len = clen;
        match (whole) {
            case some(w): { w.p = p; w.len = hdr + clen; }
            case null: { }
        }
        in.p = p + hdr + clen;
        in.len = in.len - hdr - clen;
    }
    return 1;
}

int asn1_get_(&Asn1_ in, unsigned char want, &Asn1_ content) {
    unsigned char tag = (unsigned char)0;
    struct Asn1_ save = *in;
    if (!asn1_any_(in, &tag, content, null)) { return 0; }
    if (tag != want) { *in = save; return 0; }
    return 1;
}

int asn1_opt_(&Asn1_ in, unsigned char want, &Asn1_ content) {
    if (asn1_peek_(in) != (int)want) { return 0; }
    return asn1_get_(in, want, content) ? 1 : -1;
}

int asn1_uint_(&Asn1_ in, &Asn1_ magnitude) {
    struct Asn1_ c;
    if (!asn1_get_(in, (unsigned char)ASN1_TAG_INTEGER, &c)) { return 0; }
    if (c.len == 0UL) { return 0; }
    unsafe {
        if ((c.p[0] & (unsigned char)0x80) != (unsigned char)0) { return 0; }
        while (c.len > 1UL && c.p[0] == (unsigned char)0) { c.p = c.p + 1; c.len = c.len - 1UL; }
    }
    *magnitude = c;
    return 1;
}

int asn1_small_int_(&Asn1_ in, &int v) {
    struct Asn1_ m;
    if (!asn1_uint_(in, &m)) { return 0; }
    if (m.len > 4UL) { return 0; }
    unsigned int x = 0U;
    unsafe {
        unsigned long i = 0UL;
        while (i < m.len) { x = (x << 8) | (unsigned int)m.p[i]; i = i + 1UL; }
    }
    if (x > 0x7FFFFFFFU) { return 0; }
    *v = (int)x;
    return 1;
}

int asn1_oid_is_(const &Asn1_ oid, const unsigned char* bytes, unsigned long n) {
    if (oid.len != n) { return 0; }
    unsafe {
        unsigned long i = 0UL;
        while (i < n) { if (oid.p[i] != bytes[i]) { return 0; } i = i + 1UL; }
    }
    return 1;
}

int asn1_eq_(const &Asn1_ a, const &Asn1_ b) {
    if (a.len != b.len) { return 0; }
    unsafe {
        unsigned long i = 0UL;
        while (i < a.len) { if (a.p[i] != b.p[i]) { return 0; } i = i + 1UL; }
    }
    return 1;
}

} // namespace std
