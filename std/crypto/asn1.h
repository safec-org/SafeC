// SafeC Standard Library — a minimal DER (ASN.1) reader for std's crypto.
//
// std-internal ('_' names): the TLV walking x509.h's certificate parser,
// rsa.h's key parsing and ec.h's signature parsing share. An Asn1_ is a
// view of bytes owned elsewhere; reading a TLV advances it. Definite
// lengths only (DER), at most four length bytes.
#pragma once

#define ASN1_TAG_BOOLEAN      0x01
#define ASN1_TAG_INTEGER      0x02
#define ASN1_TAG_BIT_STRING   0x03
#define ASN1_TAG_OCTET_STRING 0x04
#define ASN1_TAG_NULL         0x05
#define ASN1_TAG_OID          0x06
#define ASN1_TAG_UTF8         0x0C
#define ASN1_TAG_PRINTABLE    0x13
#define ASN1_TAG_T61          0x14
#define ASN1_TAG_IA5          0x16
#define ASN1_TAG_UTCTIME      0x17
#define ASN1_TAG_GENTIME      0x18
#define ASN1_TAG_BMP          0x1E
#define ASN1_TAG_SEQUENCE     0x30
#define ASN1_TAG_SET          0x31

namespace std {

struct Asn1_ {
    const unsigned char* p;
    unsigned long        len;
};

struct Asn1_ asn1_view_(const unsigned char* p, unsigned long len);
// The tag of the next TLV, or -1 when the view is empty.
int  asn1_peek_(const &Asn1_ in);
// Reads the next TLV: its tag, its content and (if non-null) the whole
// encoding (header included). 0 on truncation / a bad length.
int  asn1_any_(&Asn1_ in, &unsigned char tag, &Asn1_ content, ?&Asn1_ whole);
// The next TLV, which must have tag 'want'. 0 otherwise.
int  asn1_get_(&Asn1_ in, unsigned char want, &Asn1_ content);
// Like asn1_get_, but only if the next tag is 'want': 1 read, 0 absent
// (nothing consumed), -1 malformed.
int  asn1_opt_(&Asn1_ in, unsigned char want, &Asn1_ content);
// A non-negative INTEGER's magnitude, leading zero bytes stripped. 0 on a
// negative or empty integer.
int  asn1_uint_(&Asn1_ in, &Asn1_ magnitude);
// A small non-negative INTEGER (up to 31 bits) into *v.
int  asn1_small_int_(&Asn1_ in, &int v);
// 1 if the OID content equals bytes[0..n).
int  asn1_oid_is_(const &Asn1_ oid, const unsigned char* bytes, unsigned long n);
// 1 if the two views hold the same bytes.
int  asn1_eq_(const &Asn1_ a, const &Asn1_ b);

} // namespace std
