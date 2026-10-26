// SafeC Standard Library — X25519 (RFC 7748), see x25519.h.
//
// Field elements are sixteen signed 64-bit limbs of 16 bits each
// (TweetNaCl's representation): carries are propagated with arithmetic
// shifts, never branches, so every operation runs in constant time.
#pragma once
#include <std/crypto/x25519.h>
#include <std/crypto/ct.h>
#include <std/mem.h>

namespace std {

struct Fe25519_ { long long v[16]; };

static void fe_car_(&Fe25519_ o) {
    int i = 0;
    while (i < 16) {
        o.v[i] = o.v[i] + (1LL << 16);
        long long c = o.v[i] >> 16;
        if (i < 15) { o.v[i + 1] = o.v[i + 1] + c - 1LL; }
        else { o.v[0] = o.v[0] + 38LL * (c - 1LL); }
        o.v[i] = o.v[i] - (c << 16);
        i = i + 1;
    }
}

// Swap p and q when b == 1 (b is 0 or 1), without branching on b.
static void fe_sel_(&Fe25519_ p, &Fe25519_ q, long long b) {
    long long c = ~(b - 1LL);
    int i = 0;
    while (i < 16) {
        long long t = c & (p.v[i] ^ q.v[i]);
        p.v[i] = p.v[i] ^ t;
        q.v[i] = q.v[i] ^ t;
        i = i + 1;
    }
}

static void fe_pack_(&unsigned char out, const &Fe25519_ n) {
    struct Fe25519_ t = *n;
    struct Fe25519_ m;
    fe_car_(&t); fe_car_(&t); fe_car_(&t);
    int j = 0;
    while (j < 2) {
        m.v[0] = t.v[0] - 0xffedLL;
        int i = 1;
        while (i < 15) {
            m.v[i] = t.v[i] - 0xffffLL - ((m.v[i - 1] >> 16) & 1LL);
            m.v[i - 1] = m.v[i - 1] & 0xffffLL;
            i = i + 1;
        }
        m.v[15] = t.v[15] - 0x7fffLL - ((m.v[14] >> 16) & 1LL);
        long long b = (m.v[15] >> 16) & 1LL;
        m.v[14] = m.v[14] & 0xffffLL;
        fe_sel_(&t, &m, 1LL - b);
        j = j + 1;
    }
    unsafe {
        unsigned char* o = (unsigned char*)out;
        int i = 0;
        while (i < 16) {
            o[2 * i] = (unsigned char)(t.v[i] & 0xffLL);
            o[2 * i + 1] = (unsigned char)((t.v[i] >> 8) & 0xffLL);
            i = i + 1;
        }
    }
    mem_zero_secure_obj(&t);
    mem_zero_secure_obj(&m);
}

static void fe_unpack_(&Fe25519_ o, const &unsigned char n) {
    unsafe {
        const unsigned char* p = (const unsigned char*)n;
        int i = 0;
        while (i < 16) {
            o.v[i] = (long long)p[2 * i] + ((long long)p[2 * i + 1] << 8);
            i = i + 1;
        }
    }
    o.v[15] = o.v[15] & 0x7fffLL;   // RFC 7748: the top bit of u is ignored
}

static void fe_add_(&Fe25519_ o, const &Fe25519_ a, const &Fe25519_ b) {
    int i = 0;
    while (i < 16) { o.v[i] = a.v[i] + b.v[i]; i = i + 1; }
}

static void fe_sub_(&Fe25519_ o, const &Fe25519_ a, const &Fe25519_ b) {
    int i = 0;
    while (i < 16) { o.v[i] = a.v[i] - b.v[i]; i = i + 1; }
}

static void fe_mul_(&Fe25519_ o, const &Fe25519_ a, const &Fe25519_ b) {
    long long t[31];
    int i = 0;
    while (i < 31) { t[i] = 0LL; i = i + 1; }
    i = 0;
    while (i < 16) {
        int j = 0;
        while (j < 16) { t[i + j] = t[i + j] + a.v[i] * b.v[j]; j = j + 1; }
        i = i + 1;
    }
    i = 0;
    while (i < 15) { t[i] = t[i] + 38LL * t[i + 16]; i = i + 1; }
    i = 0;
    while (i < 16) { o.v[i] = t[i]; i = i + 1; }
    fe_car_(o);
    fe_car_(o);
}

static void fe_sq_(&Fe25519_ o, const &Fe25519_ a) { fe_mul_(o, a, a); }

static void fe_inv_(&Fe25519_ o, const &Fe25519_ i) {
    struct Fe25519_ c = *i;
    int a = 253;
    while (a >= 0) {
        struct Fe25519_ t = c;
        fe_sq_(&c, &t);
        if (a != 2 && a != 4) {
            t = c;
            fe_mul_(&c, &t, i);
        }
        a = a - 1;
    }
    *o = c;
}

static void x25519_raw_(&unsigned char q, const &unsigned char n, const &unsigned char p) {
    unsigned char z[32];
    unsafe {
        const unsigned char* np = (const unsigned char*)n;
        int i = 0;
        while (i < 32) { z[i] = np[i]; i = i + 1; }
    }
    z[31] = (unsigned char)((z[31] & (unsigned char)127) | (unsigned char)64);
    z[0] = (unsigned char)(z[0] & (unsigned char)248);
    struct Fe25519_ x;
    fe_unpack_(&x, p);
    struct Fe25519_ a;
    struct Fe25519_ b = x;
    struct Fe25519_ c;
    struct Fe25519_ d;
    struct Fe25519_ e;
    struct Fe25519_ f;
    struct Fe25519_ k121665;
    int i = 0;
    while (i < 16) {
        a.v[i] = 0LL; c.v[i] = 0LL; d.v[i] = 0LL; k121665.v[i] = 0LL;
        i = i + 1;
    }
    a.v[0] = 1LL;
    d.v[0] = 1LL;
    k121665.v[0] = 0xDB41LL;
    k121665.v[1] = 1LL;
    i = 254;
    while (i >= 0) {
        long long r = (long long)((z[i >> 3] >> (unsigned char)(i & 7)) & (unsigned char)1);
        fe_sel_(&a, &b, r);
        fe_sel_(&c, &d, r);
        struct Fe25519_ t;
        fe_add_(&e, &a, &c);
        t = a; fe_sub_(&a, &t, &c);
        fe_add_(&c, &b, &d);
        t = b; fe_sub_(&b, &t, &d);
        fe_sq_(&d, &e);
        fe_sq_(&f, &a);
        t = a; fe_mul_(&a, &c, &t);
        fe_mul_(&c, &b, &e);
        fe_add_(&e, &a, &c);
        t = a; fe_sub_(&a, &t, &c);
        fe_sq_(&b, &a);
        fe_sub_(&c, &d, &f);
        fe_mul_(&a, &c, &k121665);
        t = a; fe_add_(&a, &t, &d);
        t = c; fe_mul_(&c, &t, &a);
        fe_mul_(&a, &d, &f);
        fe_mul_(&d, &b, &x);
        fe_sq_(&b, &e);
        fe_sel_(&a, &b, r);
        fe_sel_(&c, &d, r);
        i = i - 1;
    }
    struct Fe25519_ ci;
    fe_inv_(&ci, &c);
    struct Fe25519_ res;
    fe_mul_(&res, &a, &ci);
    fe_pack_(q, &res);
    mem_zero_secure(z[0..32UL]);
    mem_zero_secure_obj(&a);
    mem_zero_secure_obj(&b);
    mem_zero_secure_obj(&c);
    mem_zero_secure_obj(&d);
    mem_zero_secure_obj(&e);
    mem_zero_secure_obj(&f);
}

int x25519_n_(&unsigned char out, const &unsigned char scalar, const &unsigned char point) {
    x25519_raw_(out, scalar, point);
    return !ct_is_zero_n_(out, 32UL);
}

int x25519([]unsigned char out_s, []const unsigned char scalar_s, []const unsigned char point_s) {
    &unsigned char out = crypto_out_(out_s, 32UL);
    const &unsigned char scalar = crypto_in_(scalar_s, 32UL);
    const &unsigned char point = crypto_in_(point_s, 32UL);
    return x25519_n_(out, scalar, point);
}

void x25519_public_n_(&unsigned char out, const &unsigned char scalar) {
    unsigned char nine[32];
    int i = 0;
    while (i < 32) { nine[i] = (unsigned char)0; i = i + 1; }
    nine[0] = (unsigned char)9;
    x25519_raw_(out, scalar, &nine[0]);
}

void x25519_public([]unsigned char out_s, []const unsigned char scalar_s) {
    &unsigned char out = crypto_out_(out_s, 32UL);
    const &unsigned char scalar = crypto_in_(scalar_s, 32UL);
    x25519_public_n_(out, scalar);
}

} // namespace std
