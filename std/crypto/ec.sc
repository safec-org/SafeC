// SafeC Standard Library — P-256 / P-384 ECDH and ECDSA (see ec.h).
#pragma once
#include <std/crypto/ec.h>
#include <std/crypto/bignum.h>
#include <std/crypto/asn1.h>
#include <std/crypto/ct.h>
#include <std/crypto/hmac.h>
#include <std/crypto/sha256.h>
#include <std/crypto/sha512.h>
#include <std/mem.h>

#define EC_LIMBS_ 12

namespace std {

// Curve constants (FIPS 186-4 D.1.2.3 / D.1.2.4), big-endian.
static unsigned char ec_p256_p_[32] = { 0xff,0xff,0xff,0xff,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff };
static unsigned char ec_p256_n_[32] = { 0xff,0xff,0xff,0xff,0x00,0x00,0x00,0x00,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xbc,0xe6,0xfa,0xad,0xa7,0x17,0x9e,0x84,0xf3,0xb9,0xca,0xc2,0xfc,0x63,0x25,0x51 };
static unsigned char ec_p256_b_[32] = { 0x5a,0xc6,0x35,0xd8,0xaa,0x3a,0x93,0xe7,0xb3,0xeb,0xbd,0x55,0x76,0x98,0x86,0xbc,0x65,0x1d,0x06,0xb0,0xcc,0x53,0xb0,0xf6,0x3b,0xce,0x3c,0x3e,0x27,0xd2,0x60,0x4b };
static unsigned char ec_p256_gx_[32] = { 0x6b,0x17,0xd1,0xf2,0xe1,0x2c,0x42,0x47,0xf8,0xbc,0xe6,0xe5,0x63,0xa4,0x40,0xf2,0x77,0x03,0x7d,0x81,0x2d,0xeb,0x33,0xa0,0xf4,0xa1,0x39,0x45,0xd8,0x98,0xc2,0x96 };
static unsigned char ec_p256_gy_[32] = { 0x4f,0xe3,0x42,0xe2,0xfe,0x1a,0x7f,0x9b,0x8e,0xe7,0xeb,0x4a,0x7c,0x0f,0x9e,0x16,0x2b,0xce,0x33,0x57,0x6b,0x31,0x5e,0xce,0xcb,0xb6,0x40,0x68,0x37,0xbf,0x51,0xf5 };
static unsigned char ec_p384_p_[48] = { 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xfe,0xff,0xff,0xff,0xff,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xff,0xff,0xff,0xff };
static unsigned char ec_p384_n_[48] = { 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xc7,0x63,0x4d,0x81,0xf4,0x37,0x2d,0xdf,0x58,0x1a,0x0d,0xb2,0x48,0xb0,0xa7,0x7a,0xec,0xec,0x19,0x6a,0xcc,0xc5,0x29,0x73 };
static unsigned char ec_p384_b_[48] = { 0xb3,0x31,0x2f,0xa7,0xe2,0x3e,0xe7,0xe4,0x98,0x8e,0x05,0x6b,0xe3,0xf8,0x2d,0x19,0x18,0x1d,0x9c,0x6e,0xfe,0x81,0x41,0x12,0x03,0x14,0x08,0x8f,0x50,0x13,0x87,0x5a,0xc6,0x56,0x39,0x8d,0x8a,0x2e,0xd1,0x9d,0x2a,0x85,0xc8,0xed,0xd3,0xec,0x2a,0xef };
static unsigned char ec_p384_gx_[48] = { 0xaa,0x87,0xca,0x22,0xbe,0x8b,0x05,0x37,0x8e,0xb1,0xc7,0x1e,0xf3,0x20,0xad,0x74,0x6e,0x1d,0x3b,0x62,0x8b,0xa7,0x9b,0x98,0x59,0xf7,0x41,0xe0,0x82,0x54,0x2a,0x38,0x55,0x02,0xf2,0x5d,0xbf,0x55,0x29,0x6c,0x3a,0x54,0x5e,0x38,0x72,0x76,0x0a,0xb7 };
static unsigned char ec_p384_gy_[48] = { 0x36,0x17,0xde,0x4a,0x96,0x26,0x2c,0x6f,0x5d,0x9e,0x98,0xbf,0x92,0x92,0xdc,0x29,0xf8,0xf4,0x1d,0xbd,0x28,0x9a,0x14,0x7c,0xe9,0xda,0x31,0x13,0xb5,0xf0,0xb8,0xc0,0x0a,0x60,0xb1,0xce,0x1d,0x7e,0x81,0x9d,0x7a,0x43,0x1d,0x7c,0x90,0xea,0x0e,0x5f };

// oids: id-ecPublicKey, prime256v1, secp384r1
static unsigned char ec_oid_pk_[7] = { 0x2a,0x86,0x48,0xce,0x3d,0x02,0x01 };
static unsigned char ec_oid_p256_[8] = { 0x2a,0x86,0x48,0xce,0x3d,0x03,0x01,0x07 };
static unsigned char ec_oid_p384_[5] = { 0x2b,0x81,0x04,0x00,0x22 };

struct EcFe_ { unsigned int v[EC_LIMBS_]; };
struct EcPt_ { struct EcFe_ x; struct EcFe_ y; struct EcFe_ z; };

// A curve with its Montgomery constants (built per call: cheap next to
// any scalar multiplication, and nothing global to initialize).
struct EcCurve_ {
    int           id;
    int           n;           // limbs
    unsigned long sz;          // bytes per coordinate
    unsigned int  p[EC_LIMBS_];
    unsigned int  p_inv;
    unsigned int  p_rr[EC_LIMBS_];
    unsigned int  q[EC_LIMBS_];        // the group order
    unsigned int  q_inv;
    unsigned int  q_rr[EC_LIMBS_];
    struct EcFe_  b;           // Montgomery form
    struct EcFe_  one;         // R mod p
    struct EcFe_  gx;
    struct EcFe_  gy;
    unsigned char pm2[EC_MAX_SCALAR];  // p - 2, big-endian
    unsigned char qm2[EC_MAX_SCALAR];  // q - 2
};

unsigned long ec_scalar_size(int curve) {
    if (curve == EC_P256) { return 32UL; }
    if (curve == EC_P384) { return 48UL; }
    return 0UL;
}

unsigned long ec_point_size(int curve) {
    unsigned long sz = ec_scalar_size(curve);
    return sz == 0UL ? 0UL : 1UL + 2UL * sz;
}

static void ec_fe_mul_(const &EcCurve_ c, &EcFe_ r, const &EcFe_ a, const &EcFe_ b) {
    unsafe {
        bn_mont_mul_((unsigned int*)&r.v[0], (const unsigned int*)&a.v[0], (const unsigned int*)&b.v[0],
                     (const unsigned int*)&c.p[0], c.p_inv, c.n);
    }
}
static void ec_fe_add_(const &EcCurve_ c, &EcFe_ r, const &EcFe_ a, const &EcFe_ b) {
    unsafe {
        bn_mod_add_((unsigned int*)&r.v[0], (const unsigned int*)&a.v[0], (const unsigned int*)&b.v[0],
                    (const unsigned int*)&c.p[0], c.n);
    }
}
static void ec_fe_sub_(const &EcCurve_ c, &EcFe_ r, const &EcFe_ a, const &EcFe_ b) {
    unsafe {
        bn_mod_sub_((unsigned int*)&r.v[0], (const unsigned int*)&a.v[0], (const unsigned int*)&b.v[0],
                    (const unsigned int*)&c.p[0], c.n);
    }
}
static void ec_fe_zero_(&EcFe_ r) {
    int i = 0;
    while (i < EC_LIMBS_) { r.v[i] = 0U; i = i + 1; }
}

// (a mod m) - 2 for odd prime m as big-endian bytes.
static void ec_minus2_(const unsigned int* m, int n, unsigned long sz, unsigned char* out) {
    unsigned int two[EC_LIMBS_];
    unsigned int r[EC_LIMBS_];
    unsafe {
        bn_zero_(&two[0], n);
        two[0] = 2U;
        bn_sub_(&r[0], m, &two[0], n);
        bn_to_be_(out, sz, &r[0], n);
    }
}

// Big-endian bytes (< p) into Montgomery form. 0 if >= p.
static int ec_fe_from_be_(const &EcCurve_ c, &EcFe_ r, const unsigned char* be) {
    struct EcFe_ t;
    ec_fe_zero_(&t);
    unsafe {
        bn_from_be_((unsigned int*)&t.v[0], c.n, be, c.sz);
        if (bn_cmp_((const unsigned int*)&t.v[0], (const unsigned int*)&c.p[0], c.n) >= 0) { return 0; }
        bn_mont_mul_((unsigned int*)&r.v[0], (const unsigned int*)&t.v[0], (const unsigned int*)&c.p_rr[0],
                     (const unsigned int*)&c.p[0], c.p_inv, c.n);
    }
    return 1;
}

static void ec_fe_to_be_(const &EcCurve_ c, const &EcFe_ a, unsigned char* out) {
    struct EcFe_ one;
    struct EcFe_ t;
    ec_fe_zero_(&one);
    one.v[0] = 1U;
    ec_fe_mul_(c, &t, a, &one);
    unsafe { bn_to_be_(out, c.sz, (const unsigned int*)&t.v[0], c.n); }
}

static int ec_curve_(int id, &EcCurve_ c) {
    const unsigned char* p;
    const unsigned char* q;
    const unsigned char* b;
    const unsigned char* gx;
    const unsigned char* gy;
    unsafe {
        if (id == EC_P256) {
            p = &ec_p256_p_[0]; q = &ec_p256_n_[0]; b = &ec_p256_b_[0]; gx = &ec_p256_gx_[0]; gy = &ec_p256_gy_[0];
            c.n = 8; c.sz = 32UL;
        } else if (id == EC_P384) {
            p = &ec_p384_p_[0]; q = &ec_p384_n_[0]; b = &ec_p384_b_[0]; gx = &ec_p384_gx_[0]; gy = &ec_p384_gy_[0];
            c.n = 12; c.sz = 48UL;
        } else {
            return 0;
        }
        c.id = id;
        int i = 0;
        while (i < EC_LIMBS_) { c.p[i] = 0U; c.q[i] = 0U; c.p_rr[i] = 0U; c.q_rr[i] = 0U; i = i + 1; }
        bn_from_be_((unsigned int*)&c.p[0], c.n, p, c.sz);
        bn_from_be_((unsigned int*)&c.q[0], c.n, q, c.sz);
        c.p_inv = bn_n0inv_(c.p[0]);
        c.q_inv = bn_n0inv_(c.q[0]);
        bn_rr_((unsigned int*)&c.p_rr[0], (const unsigned int*)&c.p[0], c.n);
        bn_rr_((unsigned int*)&c.q_rr[0], (const unsigned int*)&c.q[0], c.n);
        ec_fe_from_be_(c, &c.b, b);
        ec_fe_from_be_(c, &c.gx, gx);
        ec_fe_from_be_(c, &c.gy, gy);
        struct EcFe_ one;
        ec_fe_zero_(&one);
        one.v[0] = 1U;
        ec_fe_zero_(&c.one);
        bn_mont_mul_((unsigned int*)&c.one.v[0], (const unsigned int*)&one.v[0], (const unsigned int*)&c.p_rr[0],
                     (const unsigned int*)&c.p[0], c.p_inv, c.n);
        ec_minus2_((const unsigned int*)&c.p[0], c.n, c.sz, (unsigned char*)&c.pm2[0]);
        ec_minus2_((const unsigned int*)&c.q[0], c.n, c.sz, (unsigned char*)&c.qm2[0]);
    }
    return 1;
}

// r = a^e (Montgomery form in and out) for a public exponent e (sz bytes)
// modulo m: the sequence of operations depends on e only.
static void ec_pow_(const unsigned int* m, unsigned int minv, int n, unsigned int* r, const unsigned int* a,
                    const unsigned int* one_m, const unsigned char* e, unsigned long sz) {
    unsigned int acc[EC_LIMBS_];
    unsafe {
        bn_copy_(&acc[0], one_m, n);
        unsigned long i = 0UL;
        while (i < sz) {
            int bit = 7;
            while (bit >= 0) {
                bn_mont_mul_(&acc[0], &acc[0], &acc[0], m, minv, n);
                if (((e[i] >> (unsigned char)bit) & (unsigned char)1) != (unsigned char)0) {
                    bn_mont_mul_(&acc[0], &acc[0], a, m, minv, n);
                }
                bit = bit - 1;
            }
            i = i + 1UL;
        }
        bn_copy_(r, &acc[0], n);
    }
}

// ── Points: complete projective formulas, a = -3 (RCB 2016, Algorithm 4) ──────

static void ec_pt_inf_(const &EcCurve_ c, &EcPt_ r) {
    ec_fe_zero_(&r.x);
    r.y = c.one;
    ec_fe_zero_(&r.z);
}

static void ec_pt_add_(const &EcCurve_ c, &EcPt_ out, const &EcPt_ P, const &EcPt_ Q) {
    struct EcFe_ t0; struct EcFe_ t1; struct EcFe_ t2; struct EcFe_ t3; struct EcFe_ t4;
    struct EcFe_ X3; struct EcFe_ Y3; struct EcFe_ Z3; struct EcFe_ u;
    ec_fe_mul_(c, &t0, &P.x, &Q.x);
    ec_fe_mul_(c, &t1, &P.y, &Q.y);
    ec_fe_mul_(c, &t2, &P.z, &Q.z);
    ec_fe_add_(c, &t3, &P.x, &P.y);
    ec_fe_add_(c, &t4, &Q.x, &Q.y);
    u = t3; ec_fe_mul_(c, &t3, &u, &t4);
    ec_fe_add_(c, &t4, &t0, &t1);
    u = t3; ec_fe_sub_(c, &t3, &u, &t4);
    ec_fe_add_(c, &t4, &P.y, &P.z);
    ec_fe_add_(c, &X3, &Q.y, &Q.z);
    u = t4; ec_fe_mul_(c, &t4, &u, &X3);
    ec_fe_add_(c, &X3, &t1, &t2);
    u = t4; ec_fe_sub_(c, &t4, &u, &X3);
    ec_fe_add_(c, &X3, &P.x, &P.z);
    ec_fe_add_(c, &Y3, &Q.x, &Q.z);
    u = X3; ec_fe_mul_(c, &X3, &u, &Y3);
    ec_fe_add_(c, &Y3, &t0, &t2);
    u = Y3; ec_fe_sub_(c, &Y3, &X3, &u);
    ec_fe_mul_(c, &Z3, &c.b, &t2);
    ec_fe_sub_(c, &X3, &Y3, &Z3);
    ec_fe_add_(c, &Z3, &X3, &X3);
    u = X3; ec_fe_add_(c, &X3, &u, &Z3);
    ec_fe_sub_(c, &Z3, &t1, &X3);
    u = X3; ec_fe_add_(c, &X3, &t1, &u);
    u = Y3; ec_fe_mul_(c, &Y3, &c.b, &u);
    ec_fe_add_(c, &t1, &t2, &t2);
    u = t2; ec_fe_add_(c, &t2, &t1, &u);
    u = Y3; ec_fe_sub_(c, &Y3, &u, &t2);
    u = Y3; ec_fe_sub_(c, &Y3, &u, &t0);
    ec_fe_add_(c, &t1, &Y3, &Y3);
    u = Y3; ec_fe_add_(c, &Y3, &t1, &u);
    ec_fe_add_(c, &t1, &t0, &t0);
    u = t0; ec_fe_add_(c, &t0, &t1, &u);
    u = t0; ec_fe_sub_(c, &t0, &u, &t2);
    ec_fe_mul_(c, &t1, &t4, &Y3);
    ec_fe_mul_(c, &t2, &t0, &Y3);
    ec_fe_mul_(c, &Y3, &X3, &Z3);
    u = Y3; ec_fe_add_(c, &Y3, &u, &t2);
    u = X3; ec_fe_mul_(c, &X3, &t3, &u);
    u = X3; ec_fe_sub_(c, &X3, &u, &t1);
    u = Z3; ec_fe_mul_(c, &Z3, &t4, &u);
    ec_fe_mul_(c, &t1, &t3, &t0);
    u = Z3; ec_fe_add_(c, &Z3, &u, &t1);
    out.x = X3;
    out.y = Y3;
    out.z = Z3;
}

static void ec_pt_select_(const &EcCurve_ c, &EcPt_ r, const &EcPt_ a, unsigned int bit) {
    unsafe {
        bn_select_((unsigned int*)&r.x.v[0], (const unsigned int*)&a.x.v[0], bit, c.n);
        bn_select_((unsigned int*)&r.y.v[0], (const unsigned int*)&a.y.v[0], bit, c.n);
        bn_select_((unsigned int*)&r.z.v[0], (const unsigned int*)&a.z.v[0], bit, c.n);
    }
}

// r = k * P for a big-endian scalar k of c.sz bytes: double-and-add-always
// over every bit, branch-free.
static void ec_pt_mul_(const &EcCurve_ c, &EcPt_ r, const &EcPt_ P, const unsigned char* k) {
    struct EcPt_ acc;
    struct EcPt_ t;
    struct EcPt_ d;
    ec_pt_inf_(c, &acc);
    unsafe {
        unsigned long i = 0UL;
        while (i < c.sz) {
            int bit = 7;
            while (bit >= 0) {
                ec_pt_add_(c, &d, &acc, &acc);
                ec_pt_add_(c, &t, &d, P);
                acc = d;
                unsigned int b = (unsigned int)((k[i] >> (unsigned char)bit) & (unsigned char)1);
                ec_pt_select_(c, &acc, &t, b);
                bit = bit - 1;
            }
            i = i + 1UL;
        }
    }
    *r = acc;
    mem_zero_secure_obj(&acc);
    mem_zero_secure_obj(&t);
    mem_zero_secure_obj(&d);
}

static void ec_pt_gen_(const &EcCurve_ c, &EcPt_ g) {
    g.x = c.gx;
    g.y = c.gy;
    g.z = c.one;
}

// Affine big-endian coordinates of P; 0 for the point at infinity.
static int ec_pt_affine_(const &EcCurve_ c, const &EcPt_ P, unsigned char* x, unsigned char* y) {
    unsafe {
        if (bn_is_zero_((const unsigned int*)&P.z.v[0], c.n)) { return 0; }
        struct EcFe_ zi;
        ec_fe_zero_(&zi);
        ec_pow_((const unsigned int*)&c.p[0], c.p_inv, c.n, (unsigned int*)&zi.v[0], (const unsigned int*)&P.z.v[0],
                (const unsigned int*)&c.one.v[0], (const unsigned char*)&c.pm2[0], c.sz);
        struct EcFe_ ax;
        struct EcFe_ ay;
        ec_fe_mul_(c, &ax, &P.x, &zi);
        ec_fe_mul_(c, &ay, &P.y, &zi);
        ec_fe_to_be_(c, &ax, x);
        if (y != (unsigned char*)0) { ec_fe_to_be_(c, &ay, y); }
    }
    return 1;
}

// Decodes and validates an uncompressed point.
static int ec_pt_decode_(const &EcCurve_ c, &EcPt_ P, const unsigned char* pub, unsigned long len) {
    unsafe {
        if (len != 1UL + 2UL * c.sz || pub[0] != (unsigned char)4) { return 0; }
        if (!ec_fe_from_be_(c, &P.x, pub + 1)) { return 0; }
        if (!ec_fe_from_be_(c, &P.y, pub + 1 + c.sz)) { return 0; }
    }
    P.z = c.one;
    // y^2 == x^3 - 3x + b
    struct EcFe_ lhs; struct EcFe_ rhs; struct EcFe_ t;
    ec_fe_mul_(c, &lhs, &P.y, &P.y);
    ec_fe_mul_(c, &t, &P.x, &P.x);
    ec_fe_mul_(c, &rhs, &t, &P.x);
    ec_fe_add_(c, &t, &P.x, &P.x);
    struct EcFe_ u = t;
    ec_fe_add_(c, &t, &u, &P.x);
    u = rhs; ec_fe_sub_(c, &rhs, &u, &t);
    u = rhs; ec_fe_add_(c, &rhs, &u, &c.b);
    unsafe { return bn_cmp_((const unsigned int*)&lhs.v[0], (const unsigned int*)&rhs.v[0], c.n) == 0; }
}

// 1 if the scalar (c.sz bytes) is in [1, q-1].
static int ec_scalar_ok_(const &EcCurve_ c, const unsigned char* k) {
    unsigned int t[EC_LIMBS_];
    unsafe {
        bn_from_be_(&t[0], c.n, k, c.sz);
        if (bn_is_zero_(&t[0], c.n)) { return 0; }
        return bn_cmp_(&t[0], (const unsigned int*)&c.q[0], c.n) < 0;
    }
}

static void ec_encode_pub_(const &EcCurve_ c, const unsigned char* x, const unsigned char* y, unsigned char* out) {
    unsafe {
        out[0] = (unsigned char)4;
        unsigned long i = 0UL;
        while (i < c.sz) { out[1UL + i] = x[i]; out[1UL + c.sz + i] = y[i]; i = i + 1UL; }
    }
}

int ec_public_key(int curve, const &unsigned char priv, &unsigned char pub) {
    struct EcCurve_ c;
    if (!ec_curve_(curve, &c)) { return 0; }
    unsigned char x[EC_MAX_SCALAR];
    unsigned char y[EC_MAX_SCALAR];
    unsafe {
        if (!ec_scalar_ok_(&c, (const unsigned char*)priv)) { return 0; }
        struct EcPt_ g;
        struct EcPt_ r;
        ec_pt_gen_(&c, &g);
        ec_pt_mul_(&c, &r, &g, (const unsigned char*)priv);
        if (!ec_pt_affine_(&c, &r, &x[0], &y[0])) { return 0; }
        ec_encode_pub_(&c, &x[0], &y[0], (unsigned char*)pub);
    }
    return 1;
}

int ec_keygen(int curve, &RngCtx rng, &unsigned char priv, &unsigned char pub) {
    struct EcCurve_ c;
    if (!ec_curve_(curve, &c)) { return 0; }
    int tries = 0;
    while (tries < 64) {
        rng.fill(priv, c.sz);
        unsafe { if (ec_scalar_ok_(&c, (const unsigned char*)priv)) { return ec_public_key(curve, priv, pub); } }
        tries = tries + 1;
    }
    return 0;
}

int ec_point_valid(int curve, const &unsigned char pub, unsigned long len) {
    struct EcCurve_ c;
    if (!ec_curve_(curve, &c)) { return 0; }
    struct EcPt_ P;
    unsafe { return ec_pt_decode_(&c, &P, (const unsigned char*)pub, len); }
}

int ecdh_shared(int curve, const &unsigned char priv, const &unsigned char peer, unsigned long peer_len,
                &unsigned char out) {
    struct EcCurve_ c;
    if (!ec_curve_(curve, &c)) { return 0; }
    struct EcPt_ P;
    struct EcPt_ r;
    unsafe {
        if (!ec_pt_decode_(&c, &P, (const unsigned char*)peer, peer_len)) { return 0; }
        if (!ec_scalar_ok_(&c, (const unsigned char*)priv)) { return 0; }
        ec_pt_mul_(&c, &r, &P, (const unsigned char*)priv);
        return ec_pt_affine_(&c, &r, (unsigned char*)out, (unsigned char*)0);
    }
}

// ── ECDSA ─────────────────────────────────────────────────────────────────────

// The hash as an integer mod q (leftmost sz bytes, then one subtraction).
static void ec_hash_int_(const &EcCurve_ c, unsigned int* e, const unsigned char* hash, unsigned long hash_len) {
    unsigned char buf[EC_MAX_SCALAR];
    unsafe {
        unsigned long i = 0UL;
        while (i < c.sz) { buf[i] = (unsigned char)0; i = i + 1UL; }
        if (hash_len >= c.sz) {
            i = 0UL;
            while (i < c.sz) { buf[i] = hash[i]; i = i + 1UL; }
        } else {
            i = 0UL;
            while (i < hash_len) { buf[c.sz - hash_len + i] = hash[i]; i = i + 1UL; }
        }
        bn_from_be_(e, c.n, &buf[0], c.sz);
        bn_reduce_once_(e, (const unsigned int*)&c.q[0], c.n);
    }
}

// a * b mod q for plain (non-Montgomery) a, b < q.
static void ec_q_mul_(const &EcCurve_ c, unsigned int* r, const unsigned int* a, const unsigned int* b) {
    unsigned int am[EC_LIMBS_];
    unsafe {
        bn_mont_mul_(&am[0], a, (const unsigned int*)&c.q_rr[0], (const unsigned int*)&c.q[0], c.q_inv, c.n);
        bn_mont_mul_(r, &am[0], b, (const unsigned int*)&c.q[0], c.q_inv, c.n);
    }
}

// a^-1 mod q (Fermat; constant time) for plain a.
static void ec_q_inv_(const &EcCurve_ c, unsigned int* r, const unsigned int* a) {
    unsigned int am[EC_LIMBS_];
    unsigned int onem[EC_LIMBS_];
    unsigned int one[EC_LIMBS_];
    unsigned int im[EC_LIMBS_];
    unsafe {
        bn_zero_(&one[0], c.n);
        one[0] = 1U;
        const unsigned int* q = (const unsigned int*)&c.q[0];
        bn_mont_mul_(&am[0], a, (const unsigned int*)&c.q_rr[0], q, c.q_inv, c.n);
        bn_mont_mul_(&onem[0], &one[0], (const unsigned int*)&c.q_rr[0], q, c.q_inv, c.n);
        ec_pow_(q, c.q_inv, c.n, &im[0], &am[0], &onem[0], (const unsigned char*)&c.qm2[0], c.sz);
        bn_mont_mul_(r, &im[0], &one[0], q, c.q_inv, c.n);
    }
}

int ecdsa_verify(int curve, const &unsigned char pub, unsigned long pub_len,
                 const &unsigned char hash, unsigned long hash_len,
                 const &unsigned char r, const &unsigned char s) {
    struct EcCurve_ c;
    if (!ec_curve_(curve, &c)) { return 0; }
    struct EcPt_ Q;
    unsigned int rr[EC_LIMBS_];
    unsigned int ss[EC_LIMBS_];
    unsigned int e[EC_LIMBS_];
    unsigned int w[EC_LIMBS_];
    unsigned int u1[EC_LIMBS_];
    unsigned int u2[EC_LIMBS_];
    unsigned char k1[EC_MAX_SCALAR];
    unsigned char k2[EC_MAX_SCALAR];
    unsafe {
        if (!ec_pt_decode_(&c, &Q, (const unsigned char*)pub, pub_len)) { return 0; }
        if (!ec_scalar_ok_(&c, (const unsigned char*)r) || !ec_scalar_ok_(&c, (const unsigned char*)s)) { return 0; }
        bn_from_be_(&rr[0], c.n, (const unsigned char*)r, c.sz);
        bn_from_be_(&ss[0], c.n, (const unsigned char*)s, c.sz);
        ec_hash_int_(&c, &e[0], (const unsigned char*)hash, hash_len);
        ec_q_inv_(&c, &w[0], &ss[0]);
        ec_q_mul_(&c, &u1[0], &e[0], &w[0]);
        ec_q_mul_(&c, &u2[0], &rr[0], &w[0]);
        bn_to_be_(&k1[0], c.sz, &u1[0], c.n);
        bn_to_be_(&k2[0], c.sz, &u2[0], c.n);
        struct EcPt_ g;
        struct EcPt_ a;
        struct EcPt_ b;
        struct EcPt_ sum;
        ec_pt_gen_(&c, &g);
        ec_pt_mul_(&c, &a, &g, &k1[0]);
        ec_pt_mul_(&c, &b, &Q, &k2[0]);
        ec_pt_add_(&c, &sum, &a, &b);
        unsigned char x[EC_MAX_SCALAR];
        if (!ec_pt_affine_(&c, &sum, &x[0], (unsigned char*)0)) { return 0; }
        unsigned int v[EC_LIMBS_];
        bn_from_be_(&v[0], c.n, &x[0], c.sz);
        bn_reduce_once_(&v[0], (const unsigned int*)&c.q[0], c.n);
        return bn_cmp_(&v[0], &rr[0], c.n) == 0;
    }
}

// INTEGER magnitude into a fixed sz-byte big-endian buffer.
static int ec_der_int_(&Asn1_ in, unsigned char* out, unsigned long sz) {
    struct Asn1_ m;
    if (!asn1_uint_(in, &m)) { return 0; }
    if (m.len > sz) { return 0; }
    unsafe {
        unsigned long i = 0UL;
        while (i < sz) { out[i] = (unsigned char)0; i = i + 1UL; }
        i = 0UL;
        while (i < m.len) { out[sz - m.len + i] = m.p[i]; i = i + 1UL; }
    }
    return 1;
}

int ecdsa_verify_der(int curve, const &unsigned char pub, unsigned long pub_len,
                     const &unsigned char hash, unsigned long hash_len,
                     const &unsigned char sig, unsigned long sig_len) {
    unsigned long sz = ec_scalar_size(curve);
    if (sz == 0UL) { return 0; }
    struct Asn1_ in;
    struct Asn1_ seq;
    unsigned char r[EC_MAX_SCALAR];
    unsigned char s[EC_MAX_SCALAR];
    unsafe {
        in = asn1_view_((const unsigned char*)sig, sig_len);
        if (!asn1_get_(&in, (unsigned char)ASN1_TAG_SEQUENCE, &seq) || in.len != 0UL) { return 0; }
        if (!ec_der_int_(&seq, &r[0], sz) || !ec_der_int_(&seq, &s[0], sz) || seq.len != 0UL) { return 0; }
    }
    return ecdsa_verify(curve, pub, pub_len, hash, hash_len, &r[0], &s[0]);
}

// RFC 6979 §3.2 nonce generation with HMAC over H, for the private key x
// and the hash already reduced to bits2octets form (both sz bytes).
generic<H: BlockHash>
static int ec_rfc6979_(H fresh, const &EcCurve_ c, const unsigned char* x, const unsigned char* h1, unsigned char* k) {
    unsigned long hl = fresh.digest_size();
    unsigned char V[64];
    unsigned char K[64];
    unsigned long i = 0UL;
    while (i < hl) { V[i] = (unsigned char)1; K[i] = (unsigned char)0; i = i + 1UL; }
    unsigned char sep = (unsigned char)0;
    int round = 0;
    unsafe {
        while (round < 2) {
            struct Hmac<H> m = Hmac::new(fresh, &K[0], hl);
            m.update(&V[0], hl);
            sep = (unsigned char)round;
            m.update(&sep, 1UL);
            m.update((const &unsigned char)x, c.sz);
            m.update((const &unsigned char)h1, c.sz);
            m.finish(&K[0]);
            m = Hmac::new(fresh, &K[0], hl);
            m.update(&V[0], hl);
            m.finish(&V[0]);
            round = round + 1;
        }
        int tries = 0;
        while (tries < 64) {
            unsigned long tlen = 0UL;
            while (tlen < c.sz) {
                struct Hmac<H> m = Hmac::new(fresh, &K[0], hl);
                m.update(&V[0], hl);
                m.finish(&V[0]);
                unsigned long j = 0UL;
                while (j < hl && tlen < c.sz) { k[tlen] = V[j]; tlen = tlen + 1UL; j = j + 1UL; }
            }
            if (ec_scalar_ok_(c, (const unsigned char*)k)) {
                mem_zero_secure(K[0..64UL]);
                return 1;
            }
            struct Hmac<H> m2 = Hmac::new(fresh, &K[0], hl);
            m2.update(&V[0], hl);
            sep = (unsigned char)0;
            m2.update(&sep, 1UL);
            m2.finish(&K[0]);
            m2 = Hmac::new(fresh, &K[0], hl);
            m2.update(&V[0], hl);
            m2.finish(&V[0]);
            tries = tries + 1;
        }
    }
    return 0;
}

// Appends INTEGER (minimal, non-negative) for the sz-byte big-endian value.
static unsigned long ec_put_int_(unsigned char* out, const unsigned char* v, unsigned long sz) {
    unsafe {
        unsigned long start = 0UL;
        while (start + 1UL < sz && v[start] == (unsigned char)0) { start = start + 1UL; }
        unsigned long n = sz - start;
        unsigned long pad = (v[start] & (unsigned char)0x80) != (unsigned char)0 ? 1UL : 0UL;
        out[0] = (unsigned char)ASN1_TAG_INTEGER;
        out[1] = (unsigned char)(n + pad);
        unsigned long o = 2UL;
        if (pad == 1UL) { out[o] = (unsigned char)0; o = o + 1UL; }
        unsigned long i = 0UL;
        while (i < n) { out[o + i] = v[start + i]; i = i + 1UL; }
        return o + n;
    }
}

int ecdsa_sign_der(int curve, const &unsigned char priv, const &unsigned char hash, unsigned long hash_len,
                   &unsigned char sig, unsigned long sig_cap, &unsigned long sig_len) {
    struct EcCurve_ c;
    if (!ec_curve_(curve, &c)) { return 0; }
    unsigned int e[EC_LIMBS_];
    unsigned int d[EC_LIMBS_];
    unsigned int kk[EC_LIMBS_];
    unsigned int ki[EC_LIMBS_];
    unsigned int r[EC_LIMBS_];
    unsigned int s[EC_LIMBS_];
    unsigned int t[EC_LIMBS_];
    unsigned char h1[EC_MAX_SCALAR];
    unsigned char k[EC_MAX_SCALAR];
    unsigned char rb[EC_MAX_SCALAR];
    unsigned char sb[EC_MAX_SCALAR];
    int good = 0;
    unsafe {
        if (!ec_scalar_ok_(&c, (const unsigned char*)priv)) { return 0; }
        ec_hash_int_(&c, &e[0], (const unsigned char*)hash, hash_len);
        bn_to_be_(&h1[0], c.sz, &e[0], c.n);
        int ok = 0;
        if (curve == EC_P256) { ok = ec_rfc6979_(Sha256Ctx::new(), &c, (const unsigned char*)priv, &h1[0], &k[0]); }
        else { ok = ec_rfc6979_(Sha512Ctx::new_384(), &c, (const unsigned char*)priv, &h1[0], &k[0]); }
        if (ok) {
            struct EcPt_ g;
            struct EcPt_ R;
            ec_pt_gen_(&c, &g);
            ec_pt_mul_(&c, &R, &g, &k[0]);
            unsigned char x[EC_MAX_SCALAR];
            if (ec_pt_affine_(&c, &R, &x[0], (unsigned char*)0)) {
                bn_from_be_(&r[0], c.n, &x[0], c.sz);
                bn_reduce_once_(&r[0], (const unsigned int*)&c.q[0], c.n);
                bn_from_be_(&d[0], c.n, (const unsigned char*)priv, c.sz);
                bn_from_be_(&kk[0], c.n, &k[0], c.sz);
                ec_q_inv_(&c, &ki[0], &kk[0]);
                ec_q_mul_(&c, &t[0], &r[0], &d[0]);
                bn_mod_add_(&t[0], &t[0], &e[0], (const unsigned int*)&c.q[0], c.n);
                ec_q_mul_(&c, &s[0], &ki[0], &t[0]);
                if (!bn_is_zero_(&r[0], c.n) && !bn_is_zero_(&s[0], c.n)) {
                    bn_to_be_(&rb[0], c.sz, &r[0], c.n);
                    bn_to_be_(&sb[0], c.sz, &s[0], c.n);
                    unsigned char body[EC_MAX_SIG];
                    unsigned long n1 = ec_put_int_(&body[0], &rb[0], c.sz);
                    unsigned long n2 = ec_put_int_(&body[n1], &sb[0], c.sz);
                    unsigned long blen = n1 + n2;
                    unsigned long hdr = blen < 128UL ? 2UL : 3UL;
                    if (hdr + blen <= sig_cap) {
                        unsigned char* o = (unsigned char*)sig;
                        o[0] = (unsigned char)ASN1_TAG_SEQUENCE;
                        if (hdr == 2UL) { o[1] = (unsigned char)blen; }
                        else { o[1] = (unsigned char)0x81; o[2] = (unsigned char)blen; }
                        unsigned long i = 0UL;
                        while (i < blen) { o[hdr + i] = body[i]; i = i + 1UL; }
                        *sig_len = hdr + blen;
                        good = 1;
                    }
                }
            }
        }
    }
    mem_zero_secure(k[0..(unsigned long)EC_MAX_SCALAR]);
    mem_zero_secure(kk[0..(unsigned long)EC_LIMBS_]);
    mem_zero_secure(ki[0..(unsigned long)EC_LIMBS_]);
    mem_zero_secure(d[0..(unsigned long)EC_LIMBS_]);
    return good;
}

// ── Private keys (RFC 5915 / PKCS#8) ──────────────────────────────────────────

static int ec_curve_of_oid_(const &Asn1_ oid) {
    unsafe {
        if (asn1_oid_is_(oid, &ec_oid_p256_[0], 8UL)) { return EC_P256; }
        if (asn1_oid_is_(oid, &ec_oid_p384_[0], 5UL)) { return EC_P384; }
    }
    return 0;
}

static int ec_parse_sec1_(&EcPrivateKey k, struct Asn1_ in, int curve) {
    struct Asn1_ seq;
    if (!asn1_get_(&in, (unsigned char)ASN1_TAG_SEQUENCE, &seq)) { return 0; }
    int version = 0;
    if (!asn1_small_int_(&seq, &version) || version != 1) { return 0; }
    struct Asn1_ d;
    if (!asn1_get_(&seq, (unsigned char)ASN1_TAG_OCTET_STRING, &d)) { return 0; }
    struct Asn1_ params;
    if (asn1_opt_(&seq, (unsigned char)0xA0, &params) == 1) {
        struct Asn1_ oid;
        if (!asn1_get_(&params, (unsigned char)ASN1_TAG_OID, &oid)) { return 0; }
        int c2 = ec_curve_of_oid_(&oid);
        if (c2 == 0 || (curve != 0 && c2 != curve)) { return 0; }
        curve = c2;
    }
    unsigned long sz = ec_scalar_size(curve);
    if (sz == 0UL || d.len > sz) { return 0; }
    k.curve = curve;
    unsafe {
        unsigned long i = 0UL;
        while (i < sz) { k.d[i] = (unsigned char)0; i = i + 1UL; }
        i = 0UL;
        while (i < d.len) { k.d[sz - d.len + i] = d.p[i]; i = i + 1UL; }
    }
    if (!ec_public_key(curve, &k.d[0], &k.pub[0])) { return 0; }
    k.pub_len = ec_point_size(curve);
    return 1;
}

int EcPrivateKey::init_der(const &unsigned char der, unsigned long len) {
    struct Asn1_ in;
    unsafe { in = asn1_view_((const unsigned char*)der, len); }
    struct Asn1_ probe = in;
    struct Asn1_ seq;
    if (!asn1_get_(&probe, (unsigned char)ASN1_TAG_SEQUENCE, &seq)) { return 0; }
    int version = -1;
    if (!asn1_small_int_(&seq, &version)) { return 0; }
    if (version == 1) { return ec_parse_sec1_(self, in, 0); }
    if (version != 0) { return 0; }
    struct Asn1_ alg;
    struct Asn1_ oid;
    struct Asn1_ curveOid;
    struct Asn1_ inner;
    if (!asn1_get_(&seq, (unsigned char)ASN1_TAG_SEQUENCE, &alg)) { return 0; }
    if (!asn1_get_(&alg, (unsigned char)ASN1_TAG_OID, &oid)) { return 0; }
    unsafe { if (!asn1_oid_is_(&oid, &ec_oid_pk_[0], 7UL)) { return 0; } }
    if (!asn1_get_(&alg, (unsigned char)ASN1_TAG_OID, &curveOid)) { return 0; }
    int curve = ec_curve_of_oid_(&curveOid);
    if (curve == 0) { return 0; }
    if (!asn1_get_(&seq, (unsigned char)ASN1_TAG_OCTET_STRING, &inner)) { return 0; }
    return ec_parse_sec1_(self, inner, curve);
}

void EcPrivateKey::destroy() {
    mem_zero_secure(self.d[0..(unsigned long)EC_MAX_SCALAR]);
}

} // namespace std
