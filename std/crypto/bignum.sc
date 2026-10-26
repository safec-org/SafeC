// SafeC Standard Library — fixed-width Montgomery arithmetic (see bignum.h).
#pragma once
#include <std/crypto/bignum.h>
#include <std/crypto/ct.h>
#include <std/mem.h>

namespace std {

unsigned int bn_n0inv_(unsigned int m0) {
    // Newton's iteration for m0^-1 mod 2^32: each step doubles the
    // correct low bits (x = m0 is right mod 8 for odd m0).
    unsigned int x = m0;
    int i = 0;
    while (i < 5) { x = x * (2U - m0 * x); i = i + 1; }
    return 0U - x;
}

void bn_zero_(unsigned int* a, int n) {
    unsafe { int i = 0; while (i < n) { a[i] = 0U; i = i + 1; } }
}

void bn_copy_(unsigned int* r, const unsigned int* a, int n) {
    unsafe { int i = 0; while (i < n) { r[i] = a[i]; i = i + 1; } }
}

unsigned int bn_add_(unsigned int* r, const unsigned int* a, const unsigned int* b, int n) {
    unsigned long long c = 0ULL;
    unsafe {
        int i = 0;
        while (i < n) {
            c = c + (unsigned long long)a[i] + (unsigned long long)b[i];
            r[i] = (unsigned int)c;
            c = c >> 32ULL;
            i = i + 1;
        }
    }
    return (unsigned int)c;
}

unsigned int bn_sub_(unsigned int* r, const unsigned int* a, const unsigned int* b, int n) {
    unsigned long long borrow = 0ULL;
    unsafe {
        int i = 0;
        while (i < n) {
            unsigned long long d = (unsigned long long)a[i] - (unsigned long long)b[i] - borrow;
            r[i] = (unsigned int)d;
            borrow = (d >> 63ULL) & 1ULL;
            i = i + 1;
        }
    }
    return (unsigned int)borrow;
}

void bn_select_(unsigned int* r, const unsigned int* a, unsigned int bit, int n) {
    unsigned int mask = 0U - (bit & 1U);
    unsafe {
        int i = 0;
        while (i < n) { r[i] = r[i] ^ (mask & (r[i] ^ a[i])); i = i + 1; }
    }
}

unsigned int bn_is_zero_(const unsigned int* a, int n) {
    unsigned int acc = 0U;
    unsafe { int i = 0; while (i < n) { acc = acc | a[i]; i = i + 1; } }
    return ((acc | (0U - acc)) >> 31) ^ 1U;
}

int bn_cmp_(const unsigned int* a, const unsigned int* b, int n) {
    unsafe {
        int i = n - 1;
        while (i >= 0) {
            if (a[i] < b[i]) { return -1; }
            if (a[i] > b[i]) { return 1; }
            i = i - 1;
        }
    }
    return 0;
}

void bn_reduce_once_(unsigned int* a, const unsigned int* m, int n) {
    unsigned int t[BN_LIMBS_MAX];
    unsafe {
        unsigned int borrow = bn_sub_(&t[0], a, m, n);
        bn_select_(a, &t[0], borrow ^ 1U, n);
    }
}

void bn_mod_add_(unsigned int* r, const unsigned int* a, const unsigned int* b, const unsigned int* m, int n) {
    unsigned int t[BN_LIMBS_MAX];
    unsafe {
        unsigned int carry = bn_add_(r, a, b, n);
        unsigned int borrow = bn_sub_(&t[0], r, m, n);
        // Take r - m when the sum overflowed or didn't borrow.
        bn_select_(r, &t[0], carry | (borrow ^ 1U), n);
    }
}

void bn_mod_sub_(unsigned int* r, const unsigned int* a, const unsigned int* b, const unsigned int* m, int n) {
    unsigned int t[BN_LIMBS_MAX];
    unsafe {
        unsigned int borrow = bn_sub_(r, a, b, n);
        bn_add_(&t[0], r, m, n);
        bn_select_(r, &t[0], borrow, n);
    }
}

void bn_mont_mul_(unsigned int* r, const unsigned int* a, const unsigned int* b,
                  const unsigned int* m, unsigned int n0inv, int n) {
    unsigned int t[BN_LIMBS_MAX + 2];
    int i = 0;
    while (i < n + 2) { t[i] = 0U; i = i + 1; }
    unsafe {
        i = 0;
        while (i < n) {
            unsigned long long bi = (unsigned long long)b[i];
            unsigned long long c = 0ULL;
            int j = 0;
            while (j < n) {
                c = (unsigned long long)t[j] + (unsigned long long)a[j] * bi + c;
                t[j] = (unsigned int)c;
                c = c >> 32ULL;
                j = j + 1;
            }
            c = (unsigned long long)t[n] + c;
            t[n] = (unsigned int)c;
            t[n + 1] = (unsigned int)(c >> 32ULL);
            unsigned long long q = (unsigned long long)(t[0] * n0inv);
            c = ((unsigned long long)t[0] + q * (unsigned long long)m[0]) >> 32ULL;
            j = 1;
            while (j < n) {
                c = (unsigned long long)t[j] + q * (unsigned long long)m[j] + c;
                t[j - 1] = (unsigned int)c;
                c = c >> 32ULL;
                j = j + 1;
            }
            c = (unsigned long long)t[n] + c;
            t[n - 1] = (unsigned int)c;
            t[n] = t[n + 1] + (unsigned int)(c >> 32ULL);
            i = i + 1;
        }
        // t < 2m: subtract m once when t[n] is set or t[0..n) >= m.
        unsigned int u[BN_LIMBS_MAX];
        unsigned int borrow = bn_sub_(&u[0], &t[0], m, n);
        unsigned int take = (t[n] & 1U) | (borrow ^ 1U);
        bn_copy_(r, &t[0], n);
        bn_select_(r, &u[0], take, n);
    }
}

void bn_rr_(unsigned int* rr, const unsigned int* m, int n) {
    // 2^(64 n) mod m by doubling 1 that many times (x stays below m).
    unsigned int x[BN_LIMBS_MAX];
    unsigned int t[BN_LIMBS_MAX];
    unsafe {
        bn_zero_(&x[0], n);
        x[0] = 1U;
        int k = 0;
        while (k < 64 * n) {
            unsigned int carry = bn_add_(&x[0], &x[0], &x[0], n);
            unsigned int borrow = bn_sub_(&t[0], &x[0], m, n);
            bn_select_(&x[0], &t[0], carry | (borrow ^ 1U), n);
            k = k + 1;
        }
        bn_copy_(rr, &x[0], n);
    }
}

int bn_from_be_(unsigned int* r, int n, const unsigned char* be, unsigned long len) {
    unsafe {
        bn_zero_(r, n);
        unsigned long i = 0UL;
        while (i < len) {
            unsigned char byte = be[len - 1UL - i];
            unsigned long limb = i / 4UL;
            if (limb >= (unsigned long)n) {
                if (byte != (unsigned char)0) { return 0; }
            } else {
                r[limb] = r[limb] | ((unsigned int)byte << (unsigned int)(8UL * (i % 4UL)));
            }
            i = i + 1UL;
        }
    }
    return 1;
}

void bn_to_be_(unsigned char* out, unsigned long len, const unsigned int* a, int n) {
    unsafe {
        unsigned long i = 0UL;
        while (i < len) {
            unsigned long limb = i / 4UL;
            unsigned char byte = (unsigned char)0;
            if (limb < (unsigned long)n) {
                byte = (unsigned char)((a[limb] >> (unsigned int)(8UL * (i % 4UL))) & 0xFFU);
            }
            out[len - 1UL - i] = byte;
            i = i + 1UL;
        }
    }
}

int BnMont::init_n_(const &unsigned char mod_be, unsigned long len) {
    unsafe {
        const unsigned char* p = (const unsigned char*)mod_be;
        while (len > 0UL && p[0] == (unsigned char)0) { p = p + 1; len = len - 1UL; }
        if (len == 0UL || len > (unsigned long)(BN_LIMBS_MAX * 4)) { return 0; }
        if ((p[len - 1UL] & (unsigned char)1) == (unsigned char)0) { return 0; }
        self.n = (int)((len + 3UL) / 4UL);
        self.bytes = len;
        bn_from_be_(&self.m[0], self.n, p, len);
        self.n0inv = bn_n0inv_(self.m[0]);
        bn_rr_(&self.rr[0], &self.m[0], self.n);
    }
    return 1;
}

int BnMont::init([]const unsigned char mod_be_s) {
    unsigned long len = mod_be_s.len;
    const &unsigned char mod_be = crypto_in_(mod_be_s, 0UL);
    return self.init_n_(mod_be, len);
}

// x (big-endian, < m) into Montgomery form: xm = x * R mod m.
static int bn_to_mont_(const &BnMont m, unsigned int* xm, const &unsigned char base, unsigned long base_len) {
    unsigned int x[BN_LIMBS_MAX];
    unsafe {
        if (!bn_from_be_(&x[0], m.n, (const unsigned char*)base, base_len)) { return 0; }
        if (bn_cmp_(&x[0], &m.m[0], m.n) >= 0) { return 0; }
        bn_mont_mul_(xm, &x[0], &m.rr[0], &m.m[0], m.n0inv, m.n);
    }
    return 1;
}

// out = am * R^-1 (leaves Montgomery form) as m.bytes big-endian bytes.
static void bn_from_mont_out_(const &BnMont m, const unsigned int* am, &unsigned char out) {
    unsigned int one[BN_LIMBS_MAX];
    unsigned int r[BN_LIMBS_MAX];
    unsafe {
        bn_zero_(&one[0], m.n);
        one[0] = 1U;
        bn_mont_mul_(&r[0], am, &one[0], &m.m[0], m.n0inv, m.n);
        bn_to_be_((unsigned char*)out, m.bytes, &r[0], m.n);
    }
}

int bn_mod_exp_public_n_(const &BnMont m, const &unsigned char base, unsigned long base_len,
                      const &unsigned char exp, unsigned long exp_len, &unsigned char out) {
    unsigned int xm[BN_LIMBS_MAX];
    unsigned int acc[BN_LIMBS_MAX];
    if (!bn_to_mont_(m, &xm[0], base, base_len)) { return 0; }
    unsafe {
        // acc = R mod m (1 in Montgomery form).
        unsigned int one[BN_LIMBS_MAX];
        bn_zero_(&one[0], m.n);
        one[0] = 1U;
        bn_mont_mul_(&acc[0], &one[0], &m.rr[0], &m.m[0], m.n0inv, m.n);
        const unsigned char* e = (const unsigned char*)exp;
        unsigned long i = 0UL;
        while (i < exp_len) {
            int bit = 7;
            while (bit >= 0) {
                bn_mont_mul_(&acc[0], &acc[0], &acc[0], &m.m[0], m.n0inv, m.n);
                if (((e[i] >> (unsigned char)bit) & (unsigned char)1) != (unsigned char)0) {
                    bn_mont_mul_(&acc[0], &acc[0], &xm[0], &m.m[0], m.n0inv, m.n);
                }
                bit = bit - 1;
            }
            i = i + 1UL;
        }
    }
    bn_from_mont_out_(m, &acc[0], out);
    return 1;
}

int bn_mod_exp_public(const &BnMont m, []const unsigned char base_s, []const unsigned char exp_s, []unsigned char out_s) {
    unsigned long base_len = base_s.len;
    const &unsigned char base = crypto_in_(base_s, 0UL);
    unsigned long exp_len = exp_s.len;
    const &unsigned char exp = crypto_in_(exp_s, 0UL);
    &unsigned char out = crypto_out_(out_s, m.bytes);
    return bn_mod_exp_public_n_(m, base, base_len, exp, exp_len, out);
}

int bn_mod_exp_secret_n_(const &BnMont m, const &unsigned char base, unsigned long base_len,
                      const &unsigned char exp, unsigned long exp_len, &unsigned char out) {
    unsigned int table[16 * BN_LIMBS_MAX];
    unsigned int acc[BN_LIMBS_MAX];
    unsigned int sel[BN_LIMBS_MAX];
    int n = m.n;
    unsafe {
        unsigned int* tb = (unsigned int*)&table[0];
        unsigned int one[BN_LIMBS_MAX];
        bn_zero_(&one[0], n);
        one[0] = 1U;
        // table[k] = base^k in Montgomery form.
        bn_mont_mul_(tb, &one[0], &m.rr[0], &m.m[0], m.n0inv, n);
        if (!bn_to_mont_(m, tb + n, base, base_len)) { return 0; }
        int k = 2;
        while (k < 16) {
            bn_mont_mul_(tb + k * n, tb + (k - 1) * n, tb + n, &m.m[0], m.n0inv, n);
            k = k + 1;
        }
        bn_copy_(&acc[0], tb, n);
        const unsigned char* e = (const unsigned char*)exp;
        unsigned long i = 0UL;
        while (i < exp_len * 2UL) {
            unsigned int w = (unsigned int)e[i / 2UL];
            w = (i % 2UL == 0UL) ? (w >> 4) : (w & 15U);
            int s = 0;
            while (s < 4) {
                bn_mont_mul_(&acc[0], &acc[0], &acc[0], &m.m[0], m.n0inv, n);
                s = s + 1;
            }
            // sel = table[w], reading every entry.
            bn_zero_(&sel[0], n);
            k = 0;
            while (k < 16) {
                bn_select_(&sel[0], tb + k * n, ct_eq_u32((unsigned int)k, w) & 1U, n);
                k = k + 1;
            }
            bn_mont_mul_(&acc[0], &acc[0], &sel[0], &m.m[0], m.n0inv, n);
            i = i + 1UL;
        }
    }
    bn_from_mont_out_(m, &acc[0], out);
    mem_zero_secure(table[0..(unsigned long)(16 * BN_LIMBS_MAX)]);
    mem_zero_secure(acc[0..(unsigned long)BN_LIMBS_MAX]);
    mem_zero_secure(sel[0..(unsigned long)BN_LIMBS_MAX]);
    return 1;
}

int bn_mod_exp_secret(const &BnMont m, []const unsigned char base_s, []const unsigned char exp_s, []unsigned char out_s) {
    unsigned long base_len = base_s.len;
    const &unsigned char base = crypto_in_(base_s, 0UL);
    unsigned long exp_len = exp_s.len;
    const &unsigned char exp = crypto_in_(exp_s, 0UL);
    &unsigned char out = crypto_out_(out_s, m.bytes);
    return bn_mod_exp_secret_n_(m, base, base_len, exp, exp_len, out);
}

} // namespace std
