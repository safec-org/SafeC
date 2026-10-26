// SafeC Standard Library — TLS: trust stores and certificate path
// validation for the SafeC backend (see tls_safec.h).
//
// Path building: from the leaf, each step looks for an issuer — a trust
// anchor first, then a not-yet-used certificate the peer sent — whose
// subject is byte-for-byte the certificate's issuer and whose key verifies
// its signature, and backtracks over alternatives (cross-signed chains), up
// to TLS_MAX_CHAIN_ certificates. Intermediates must be CAs (basicConstraints
// cA, keyCertSign when keyUsage is present) within their pathLenConstraint;
// every certificate on the path must be within its validity period and carry
// no critical extension this code doesn't process (name constraints among
// them: such chains are rejected rather than half-checked). Trust anchors
// are taken as configured — like OpenSSL, their own signature isn't checked,
// their dates are.
#pragma once
#include <std/tls/tls_safec.h>
#include <std/tls/tls_backend.h>
#include <std/sync/spinlock.h>
#include <std/time.h>

#define TLS_MAX_CHAIN_ 8

namespace std {

struct TlsTrust_ tls_trust_new_() {
    struct TlsTrust_ t;
    t.certs = Vec::new();
    return t;
}

void TlsCert_::drop() { self.der.drop(); }
void TlsTrust_::drop() { self.certs.drop(); }

void tls_trust_free_(&TlsTrust_ t) {
    struct Vec<struct TlsCert_> none = Vec::new();
    std::replace(&t.certs, none).drop();
}

int tls_trust_add_der_(&TlsTrust_ t, const &unsigned char der, unsigned long len) {
    struct TlsCert_ c;
    c.der = Vec::from_arr(der, len);
    match (c.der.data) {
        case some(d): {
            if (x509_parse(d, len, &c.p) != 0) { c.drop(); return 0; }
        }
        case null: { c.drop(); return 0; }
    }
    t.certs.push(c);
    return 1;
}

unsigned long tls_trust_add_pem_(&TlsTrust_ t, const &char pem) {
    struct Vec<struct Vec<unsigned char>> ders = Vec::new();
    tls_pem_decode_(pem, "CERTIFICATE", &ders);
    unsigned long n = 0UL;
    unsigned long i = 0UL;
    while (i < ders.len) {
        match (ders.get(i)) {
            case some(d): {
                match (d.data) {
                    case some(p): { n = n + (unsigned long)tls_trust_add_der_(t, p, d.len); }
                    case null: { }
                }
            }
            case null: { }
        }
        i = i + 1UL;
    }
    ders.drop();
    return n;
}

static struct TlsTrust_ g_tls_system_trust_;
static int g_tls_system_state_ = 0;     // 0 not loaded, 1 loaded
static struct Spinlock g_tls_system_lock_;

?&TlsTrust_ tls_trust_system_() {
    g_tls_system_lock_.lock();
    if (g_tls_system_state_ == 0) {
        g_tls_system_trust_ = tls_trust_new_();
        tls_system_roots_(&g_tls_system_trust_);
        g_tls_system_state_ = 1;
    }
    g_tls_system_lock_.unlock();
    if (g_tls_system_trust_.certs.len == 0UL) { return null; }
    unsafe { return (?&TlsTrust_)&g_tls_system_trust_; }
}

unsigned long long tls_now_() {
    long long ns = time_wall_ns();
    if (ns < 0LL) { return 0ULL; }
    return (unsigned long long)(ns / 1000000000LL);
}

// ── path validation ───────────────────────────────────────────────────────────

static const unsigned char* tv_ptr_(const &TlsCert_ c) {
    unsafe { return (const unsigned char*)c.der.data; }
}

static int tv_same_der_(const &TlsCert_ a, const &TlsCert_ b) {
    if (a.der.len != b.der.len) { return 0; }
    unsafe {
        const unsigned char* x = tv_ptr_(a);
        const unsigned char* y = tv_ptr_(b);
        unsigned long i = 0UL;
        while (i < a.der.len) { if (x[i] != y[i]) { return 0; } i = i + 1UL; }
    }
    return 1;
}

static int tv_issued_by_(const &TlsCert_ c, const &TlsCert_ issuer) {
    unsafe {
        return x509_issued_by((const &unsigned char)tv_ptr_(c), &c.p,
                              (const &unsigned char)tv_ptr_(issuer), &issuer.p);
    }
}

static int tv_signed_by_(const &TlsCert_ c, const &TlsCert_ issuer) {
    unsafe {
        return x509_verify_signed_by((const &unsigned char)tv_ptr_(c), &c.p,
                                     (const &unsigned char)tv_ptr_(issuer), &issuer.p);
    }
}

static int tv_in_dates_(const &TlsCert_ c, unsigned long long now) {
    return now >= c.p.not_before && now <= c.p.not_after;
}

// The worse of two outcomes (a more specific reason wins over "untrusted").
static int tv_worse_(int cur, int next) {
    if (cur == TLS_ERR_CERT_UNTRUSTED || cur == TLS_OK) { return next; }
    return cur;
}

struct TvCtx_ {
    &Vec<struct TlsCert_> pres;
    ?&TlsTrust_           a;
    ?&TlsTrust_           b;
    unsigned long long    now;
    int                   why;   // best error so far
};

// Does an anchor in t issue c (or is c itself an anchor)?
static int tv_anchor_(&TvCtx_ x, const ?&TlsTrust_ tq, const &TlsCert_ c) {
    match (tq) {
        case some(t): {
            unsigned long i = 0UL;
            while (i < t.certs.len) {
                &TlsCert_ a = t.certs.at(i);
                if (tv_same_der_(a, c)) {
                    if (!tv_in_dates_(a, x.now)) { x.why = tv_worse_(x.why, TLS_ERR_CERT_EXPIRED); }
                    else { return 1; }
                } else if (tv_issued_by_(c, a)) {
                    if (a.p.has_key_usage && (a.p.key_usage & X509_KU_KEY_CERT_SIGN) == 0) {
                        x.why = tv_worse_(x.why, TLS_ERR_CERT_INVALID);
                    } else if (!tv_signed_by_(c, a)) {
                        x.why = tv_worse_(x.why, TLS_ERR_CERT_INVALID);
                    } else if (!tv_in_dates_(a, x.now)) {
                        x.why = tv_worse_(x.why, TLS_ERR_CERT_EXPIRED);
                    } else {
                        return 1;
                    }
                }
                i = i + 1UL;
            }
        }
        case null: { }
    }
    return 0;
}

// Builds a path upward from pres[cur]; 'used' marks certificates on the
// path, 'below' counts the intermediates under the next issuer.
static int tv_build_(&TvCtx_ x, unsigned long cur, unsigned int used, int depth, int below) {
    &TlsCert_ c = x.pres.at(cur);
    if (tv_anchor_(x, x.a, c) || tv_anchor_(x, x.b, c)) { return 1; }
    if (depth + 1 >= TLS_MAX_CHAIN_) { return 0; }
    unsigned long i = 1UL;
    while (i < x.pres.len) {
        if ((used & (1U << (unsigned int)i)) == 0U) {
            &TlsCert_ ic = x.pres.at(i);
            if (tv_issued_by_(c, ic)) {
                int ok = 1;
                if (!ic.p.has_basic_constraints || !ic.p.is_ca) { ok = 0; x.why = tv_worse_(x.why, TLS_ERR_CERT_INVALID); }
                else if (ic.p.has_key_usage && (ic.p.key_usage & X509_KU_KEY_CERT_SIGN) == 0) { ok = 0; x.why = tv_worse_(x.why, TLS_ERR_CERT_INVALID); }
                else if (ic.p.unknown_critical) { ok = 0; x.why = tv_worse_(x.why, TLS_ERR_CERT_INVALID); }
                else if (ic.p.path_len >= 0 && below > ic.p.path_len) { ok = 0; x.why = tv_worse_(x.why, TLS_ERR_CERT_INVALID); }
                else if (!tv_in_dates_(ic, x.now)) { ok = 0; x.why = tv_worse_(x.why, TLS_ERR_CERT_EXPIRED); }
                else if (!tv_signed_by_(c, ic)) { ok = 0; x.why = tv_worse_(x.why, TLS_ERR_CERT_INVALID); }
                if (ok) {
                    // A self-issued intermediate doesn't count toward pathLen.
                    int selfIssued = tv_issued_by_(ic, ic);
                    if (tv_build_(x, i, used | (1U << (unsigned int)i), depth + 1, below + (selfIssued ? 0 : 1))) { return 1; }
                }
            }
        }
        i = i + 1UL;
    }
    return 0;
}

int tls_verify_chain_(const &Vec<struct Vec<unsigned char>> chain, const ?&TlsTrust_ a, const ?&TlsTrust_ b,
                      const ?&char host, int server, unsigned long long now, &TlsError e) {
    if (chain.len == 0UL) {
        tls_fail_(e, TLS_ERR_NO_CERT, "the peer sent no certificate");
        return TLS_ERR_NO_CERT;
    }
    struct Vec<struct TlsCert_> pres = Vec::new();
    unsigned long i = 0UL;
    while (i < chain.len && i < 31UL) {
        &Vec<unsigned char> d = chain.at(i);
        struct TlsCert_ c;
        c.der = d.clone();
        int bad = 1;
        match (c.der.data) {
            case some(p): { bad = x509_parse(p, c.der.len, &c.p) != 0; }
            case null: { }
        }
        if (bad) {
            c.drop();
            pres.drop();
            if (i == 0UL) { tls_fail_(e, TLS_ERR_CERT_INVALID, "the peer's certificate doesn't parse"); }
            else { tls_fail_(e, TLS_ERR_CERT_INVALID, "a certificate in the peer's chain doesn't parse"); }
            return TLS_ERR_CERT_INVALID;
        }
        pres.push(c);
        i = i + 1UL;
    }
    &TlsCert_ leaf = pres.at(0UL);
    int code = TLS_OK;
    const &char why = "";
    if (leaf.p.unknown_critical) { code = TLS_ERR_CERT_INVALID; why = "the certificate has an unsupported critical extension"; }
    else if (now < leaf.p.not_before) { code = TLS_ERR_CERT_EXPIRED; why = "the certificate is not yet valid"; }
    else if (now > leaf.p.not_after) { code = TLS_ERR_CERT_EXPIRED; why = "the certificate has expired"; }
    else if (leaf.p.has_eku && !leaf.p.eku_any && !(server ? leaf.p.eku_server : leaf.p.eku_client)) {
        code = TLS_ERR_CERT_INVALID;
        why = server ? "the certificate is not for TLS servers (extendedKeyUsage)" : "the certificate is not for TLS clients (extendedKeyUsage)";
    } else if (leaf.p.has_key_usage && (leaf.p.key_usage & X509_KU_DIGITAL_SIGNATURE) == 0) {
        code = TLS_ERR_CERT_INVALID; why = "the certificate's keyUsage forbids signatures";
    }
    if (code == TLS_OK) {
        struct TvCtx_ x;
        x.pres = &pres;
        unsafe {
            x.a = (?&TlsTrust_)a;
            x.b = (?&TlsTrust_)b;
        }
        x.now = now;
        x.why = TLS_ERR_CERT_UNTRUSTED;
        if (!tv_build_(&x, 0UL, 1U, 0, 0)) {
            code = x.why;
            if (code == TLS_ERR_CERT_EXPIRED) { why = "a certificate in the chain has expired"; }
            else if (code == TLS_ERR_CERT_INVALID) { why = "the certificate chain is invalid (signature, CA constraints or key usage)"; }
            else { code = TLS_ERR_CERT_UNTRUSTED; why = "the certificate chain doesn't lead to a trusted root"; }
        }
    }
    if (code == TLS_OK) {
        match (host) {
            case some(h): {
                unsafe {
                    if (!x509_match_host((const &unsigned char)tv_ptr_(leaf), &leaf.p, h)) {
                        code = TLS_ERR_CERT_HOSTNAME;
                        why = "the certificate doesn't match the host name";
                    }
                }
            }
            case null: { }
        }
    }
    pres.drop();
    if (code != TLS_OK) {
        if (code == TLS_ERR_CERT_HOSTNAME) {
            match (host) {
                case some(h): { tls_fail2_(e, code, why, h); }
                case null: { tls_fail_(e, code, why); }
            }
        } else {
            tls_fail_(e, code, why);
        }
    }
    return code;
}

} // namespace std
