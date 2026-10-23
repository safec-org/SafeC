// SafeC Standard Library — TLS: SafeC's own TLS 1.3 (RFC 8446), client and
// server (TLS_IMPL_SAFEC; see tls.h and tls_safec.h).
//
// Scope: full (EC)DHE handshakes — no PSK/resumption, no 0-RTT, no TLS 1.2.
//   cipher suites     TLS_CHACHA20_POLY1305_SHA256 (preferred: constant time
//                     in software), TLS_AES_128_GCM_SHA256, TLS_AES_256_GCM_SHA384
//   groups            x25519 (offered first), secp256r1, secp384r1 (on
//                     HelloRetryRequest)
//   signatures        ecdsa_secp256r1_sha256, ecdsa_secp384r1_sha384,
//                     rsa_pss_rsae_sha256/384/512, rsa_pss_pss_*
//   certificates      verified by tls_trust.sc (chain, dates, usage, name)
//   extensions        server_name, supported_groups, signature_algorithms,
//                     supported_versions, key_share, cookie, ALPN; client
//                     certificates both ways; KeyUpdate; middlebox-compatible
//                     ChangeCipherSpec. NewSessionTickets are read and dropped.
// I/O goes straight to the socket (tls_sock_recv_/send_) with internal
// buffers, so a non-blocking socket gets TLS_WANT_READ/WRITE and the call
// is simply repeated.
#pragma once
#include <std/stddef.h>
#include <std/tls/tls_safec.h>
#include <std/tls/tls_backend.h>
#include <std/crypto/aes.h>
#include <std/crypto/chacha20.h>
#include <std/crypto/x25519.h>
#include <std/crypto/ec.h>
#include <std/crypto/rsa.h>
#include <std/crypto/digest.h>
#include <std/crypto/hkdf.h>
#include <std/crypto/hmac.h>
#include <std/crypto/sha256.h>
#include <std/crypto/sha512.h>
#include <std/crypto/rng.h>
#include <std/crypto/ct.h>
#include <std/atomic.h>
#include <std/mem.h>
#include <std/str.h>

#define T13_MAX_PLAIN_  16384
#define T13_MAX_CIPHER_ (16384 + 256)
#define T13_RBUF_       (5 + 16384 + 256)
#define T13_MAX_HS_     262144

// content types
#define T13_CT_CCS_   20
#define T13_CT_ALERT_ 21
#define T13_CT_HS_    22
#define T13_CT_APP_   23

// handshake types
#define T13_CLIENT_HELLO_   1
#define T13_SERVER_HELLO_   2
#define T13_NEW_TICKET_     4
#define T13_ENC_EXT_        8
#define T13_CERTIFICATE_    11
#define T13_CERT_REQUEST_   13
#define T13_CERT_VERIFY_    15
#define T13_FINISHED_       20
#define T13_KEY_UPDATE_     24
#define T13_MESSAGE_HASH_   254

// alerts
#define T13_A_CLOSE_NOTIFY_      0
#define T13_A_UNEXPECTED_        10
#define T13_A_BAD_MAC_           20
#define T13_A_OVERFLOW_          22
#define T13_A_HS_FAILURE_        40
#define T13_A_BAD_CERT_          42
#define T13_A_UNSUPPORTED_CERT_  43
#define T13_A_CERT_EXPIRED_      45
#define T13_A_ILLEGAL_PARAM_     47
#define T13_A_UNKNOWN_CA_        48
#define T13_A_DECODE_            50
#define T13_A_DECRYPT_           51
#define T13_A_PROTOCOL_VERSION_  70
#define T13_A_INTERNAL_          80
#define T13_A_MISSING_EXT_       109
#define T13_A_UNRECOGNIZED_NAME_ 112
#define T13_A_CERT_REQUIRED_     116
#define T13_A_NO_ALPN_           120

// groups
#define T13_X25519_    0x001d
#define T13_P256_      0x0017
#define T13_P384_      0x0018

// AEADs
#define T13_AES128_  1
#define T13_AES256_  2
#define T13_CHACHA_  3

// states
#define T13_C_START_     1
#define T13_C_WAIT_SH_   2
#define T13_C_WAIT_EE_   3
#define T13_C_WAIT_CERT_CR_ 4
#define T13_C_WAIT_CERT_ 5
#define T13_C_WAIT_CV_   6
#define T13_C_WAIT_FIN_  7
#define T13_S_WAIT_CH_   10
#define T13_S_WAIT_CERT_ 11
#define T13_S_WAIT_CV_   12
#define T13_S_WAIT_FIN_  13
#define T13_DONE_        20
#define T13_FAILED_      21

namespace std {

extern void* memmove(void* d, const void* s, size_t n);

// SHA-256("HelloRetryRequest"), the ServerHello.random of a HelloRetryRequest.
static unsigned char t13_hrr_random_[32] = {
    0xcf,0x21,0xad,0x74,0xe5,0x9a,0x61,0x11,0xbe,0x1d,0x8c,0x02,0x1e,0x65,0xb8,0x91,
    0xc2,0xa2,0x11,0x16,0x7a,0xbb,0x8c,0x5e,0x07,0x9e,0x09,0xe2,0xc8,0xa8,0x33,0x9c };

// signature_algorithms we accept (and offer), in preference order.
static unsigned int t13_sigalgs_[9] = { 0x0403, 0x0503, 0x0804, 0x0805, 0x0806, 0x0809, 0x080a, 0x080b, 0x0603 };
#define T13_NSIGALGS_ 8

struct T13Aead_ {
    int                alg;
    unsigned char      key[32];
    unsigned char      iv[12];
    unsigned long long seq;
    struct AesCtx      aes;
};

// A server context, shared by the streams it starts (reference-counted).
struct T13Acc_ {
    int                                   refs;
    struct Vec<struct Vec<unsigned char>> chain;
    struct TlsKey_                        key;
    struct Vec<unsigned char>             alpn;       // wire list, preference order
    struct TlsTrust_                      client_ca;
    int                                   verify_client;
};

struct T13_ {
    int  server;
    int  st;
    int  hash;                 // DIGEST_SHA256 / DIGEST_SHA384, 0 before the suite is known
    unsigned long hlen;
    int  suite;
    int  group;
    int  hrr;
    int  verify;
    int  verify_host;
    int  use_system;
    struct String name;        // client: the name to verify (and SNI); server: the SNI received
    struct Vec<unsigned char> alpn;   // client: offered wire list
    struct TlsTrust_ extra;    // client: configured roots
    struct T13Acc_* acc;       // server: its context
    // our certificate (client certificates; servers use acc's)
    struct Vec<struct Vec<unsigned char>> own_chain;
    struct TlsKey_ key;
    int  has_key;
    // transport
    unsigned char rbuf[T13_RBUF_];
    unsigned long rlen;
    struct Vec<unsigned char> tx;
    unsigned long tx_pos;
    struct Vec<unsigned char> hs;
    unsigned long hs_pos;
    unsigned char app[T13_MAX_PLAIN_];
    unsigned long app_pos;
    unsigned long app_len;
    int  peer_closed;
    int  sent_close;
    // handshake
    struct Vec<unsigned char> tr;     // the transcript (handshake messages)
    unsigned char random[32];
    unsigned char sid[32];
    unsigned long sid_len;
    unsigned char x_priv[32];
    unsigned char ec_priv[EC_MAX_SCALAR];
    int  ec_kx_group;          // the EC group ec_priv belongs to, 0 if none
    struct Vec<unsigned char> cookie;
    struct Vec<struct Vec<unsigned char>> peer_chain;
    unsigned int peer_sigs[32];
    int  npeer_sigs;
    int  cert_requested;
    struct Vec<unsigned char> cr_context;
    int  peer_sent_cert;
    // secrets
    unsigned char c_hs[48];
    unsigned char s_hs[48];
    unsigned char master[48];
    unsigned char c_ap[48];
    unsigned char s_ap[48];
    unsigned char rd_secret[48];
    unsigned char wr_secret[48];
    struct T13Aead_ rd;
    struct T13Aead_ wr;
    int  rd_on;
    int  wr_on;
    struct RngCtx rng;
};

// ── byte building / reading ───────────────────────────────────────────────────

static void b8_(&Vec<unsigned char> v, unsigned int x) { v.push((unsigned char)(x & 0xFFU)); }
static void b16_(&Vec<unsigned char> v, unsigned int x) { b8_(v, x >> 8); b8_(v, x); }
static void b24_(&Vec<unsigned char> v, unsigned int x) { b8_(v, x >> 16); b8_(v, x >> 8); b8_(v, x); }
static void bput_(&Vec<unsigned char> v, const unsigned char* p, unsigned long n) {
    if (n == 0UL) { return; }
    unsafe { v.extend((const &unsigned char)p, n); }
}
// Opens a length field of 'width' bytes; bclose_ fills it in.
static unsigned long bopen_(&Vec<unsigned char> v, int width) {
    unsigned long pos = v.len;
    int i = 0;
    while (i < width) { v.push((unsigned char)0); i = i + 1; }
    return pos;
}
static void bclose_(&Vec<unsigned char> v, unsigned long pos, int width) {
    unsigned long n = v.len - pos - (unsigned long)width;
    unsafe {
        unsigned char* d = (unsigned char*)v.data;
        int i = 0;
        while (i < width) {
            d[pos + (unsigned long)i] = (unsigned char)((n >> (unsigned long)(8 * (width - 1 - i))) & 0xFFUL);
            i = i + 1;
        }
    }
}
static unsigned char* vptr_(const &Vec<unsigned char> v) {
    unsafe { return (unsigned char*)v.data; }
}

struct T13Rd_ {
    const unsigned char* p;
    unsigned long        n;
    unsigned long        i;
    int                  bad;
};

static struct T13Rd_ rd_new_(const unsigned char* p, unsigned long n) {
    struct T13Rd_ r;
    r.p = p;
    r.n = n;
    r.i = 0UL;
    r.bad = 0;
    return r;
}
static unsigned long rleft_(const &T13Rd_ r) { return r.n - r.i; }
static unsigned int r8_(&T13Rd_ r) {
    if (r.bad || r.i + 1UL > r.n) { r.bad = 1; return 0U; }
    unsigned int v = 0U;
    unsafe { v = (unsigned int)r.p[r.i]; }
    r.i = r.i + 1UL;
    return v;
}
static unsigned int r16_(&T13Rd_ r) { unsigned int a = r8_(r); return (a << 8) | r8_(r); }
static unsigned int r24_(&T13Rd_ r) { unsigned int a = r8_(r); unsigned int b = r8_(r); return (a << 16) | (b << 8) | r8_(r); }
static const unsigned char* rbytes_(&T13Rd_ r, unsigned long n) {
    if (r.bad || n > r.n - r.i) { r.bad = 1; return (const unsigned char*)0; }
    const unsigned char* p;
    unsafe { p = r.p + r.i; }
    r.i = r.i + n;
    return p;
}
// A length-prefixed sub-vector (width 1, 2 or 3).
static struct T13Rd_ rvec_(&T13Rd_ r, int width) {
    unsigned long n = 0UL;
    if (width == 1) { n = (unsigned long)r8_(r); }
    else if (width == 2) { n = (unsigned long)r16_(r); }
    else { n = (unsigned long)r24_(r); }
    const unsigned char* p = rbytes_(r, n);
    struct T13Rd_ s = rd_new_(p, r.bad ? 0UL : n);
    s.bad = r.bad;
    return s;
}

// ── key schedule (RFC 8446 §7.1) ──────────────────────────────────────────────

static void t13_extract_(int h, const unsigned char* salt, unsigned long sl, const unsigned char* ikm, unsigned long il,
                         unsigned char* out) {
    unsafe {
        if (h == DIGEST_SHA384) {
            hkdf_extract(Sha512Ctx::new_384(), (const ?&unsigned char)salt, sl, (const &unsigned char)ikm, il, (&unsigned char)out);
        } else {
            hkdf_extract(Sha256Ctx::new(), (const ?&unsigned char)salt, sl, (const &unsigned char)ikm, il, (&unsigned char)out);
        }
    }
}

static void t13_expand_label_(int h, const unsigned char* secret, unsigned long sl, const &char label,
                              const unsigned char* ctx, unsigned long cl, unsigned char* out, unsigned long ol) {
    unsigned char info[2 + 1 + 255 + 1 + 64];
    unsigned long n = 0UL;
    info[0] = (unsigned char)(ol >> 8);
    info[1] = (unsigned char)(ol & 0xFFUL);
    unsigned long ll = str_len(label);
    info[2] = (unsigned char)(6UL + ll);
    const &char pre = "tls13 ";
    n = 3UL;
    unsafe {
        unsigned long i = 0UL;
        while (i < 6UL) { info[n] = (unsigned char)pre[i]; n = n + 1UL; i = i + 1UL; }
        i = 0UL;
        while (i < ll) { info[n] = (unsigned char)label[i]; n = n + 1UL; i = i + 1UL; }
        info[n] = (unsigned char)cl;
        n = n + 1UL;
        i = 0UL;
        while (i < cl) { info[n] = ctx[i]; n = n + 1UL; i = i + 1UL; }
        if (h == DIGEST_SHA384) {
            hkdf_expand(Sha512Ctx::new_384(), (const &unsigned char)secret, sl, &info[0], n, (&unsigned char)out, ol);
        } else {
            hkdf_expand(Sha256Ctx::new(), (const &unsigned char)secret, sl, &info[0], n, (&unsigned char)out, ol);
        }
    }
}

static void t13_hmac_(int h, const unsigned char* key, unsigned long kl, const unsigned char* data, unsigned long dl,
                      unsigned char* out) {
    unsafe {
        if (h == DIGEST_SHA384) {
            struct Hmac<Sha512Ctx> m = Hmac::new(Sha512Ctx::new_384(), (const &unsigned char)key, kl);
            m.update((const &unsigned char)data, dl);
            m.finish((&unsigned char)out);
        } else {
            struct Hmac<Sha256Ctx> m = Hmac::new(Sha256Ctx::new(), (const &unsigned char)key, kl);
            m.update((const &unsigned char)data, dl);
            m.finish((&unsigned char)out);
        }
    }
}

// Hash of the transcript so far.
static void t13_tr_hash_(&T13_ t, unsigned char* out) {
    unsafe { digest(t.hash, (const ?&unsigned char)t.tr.data, t.tr.len, (&unsigned char)out); }
}

static void t13_set_aead_(&T13_ t, &T13Aead_ a, const unsigned char* secret) {
    unsigned long kl = t.suite == 0x1301 ? 16UL : 32UL;
    a.alg = t.suite == 0x1301 ? T13_AES128_ : (t.suite == 0x1302 ? T13_AES256_ : T13_CHACHA_);
    unsafe {
        t13_expand_label_(t.hash, secret, t.hlen, "key", (const unsigned char*)0, 0UL, (unsigned char*)&a.key[0], kl);
        t13_expand_label_(t.hash, secret, t.hlen, "iv", (const unsigned char*)0, 0UL, (unsigned char*)&a.iv[0], 12UL);
    }
    a.seq = 0ULL;
    if (a.alg == T13_AES128_) { a.aes = AesCtx::new_128(&a.key[0]); }
    else if (a.alg == T13_AES256_) { a.aes = AesCtx::new_256(&a.key[0]); }
}

static void t13_set_read_(&T13_ t, const unsigned char* secret) {
    unsafe {
        unsigned long i = 0UL;
        while (i < t.hlen) { t.rd_secret[i] = secret[i]; i = i + 1UL; }
    }
    t13_set_aead_(t, &t.rd, secret);
    t.rd_on = 1;
}

static void t13_set_write_(&T13_ t, const unsigned char* secret) {
    unsafe {
        unsigned long i = 0UL;
        while (i < t.hlen) { t.wr_secret[i] = secret[i]; i = i + 1UL; }
    }
    t13_set_aead_(t, &t.wr, secret);
    t.wr_on = 1;
}

static void t13_nonce_(const &T13Aead_ a, unsigned char* nonce) {
    unsafe {
        int i = 0;
        while (i < 12) { nonce[i] = a.iv[i]; i = i + 1; }
        i = 0;
        while (i < 8) {
            nonce[4 + i] = nonce[4 + i] ^ (unsigned char)((a.seq >> (unsigned long long)(8 * (7 - i))) & 0xFFULL);
            i = i + 1;
        }
    }
}

// Seals data[0..len) (len includes the inner content type) in place,
// writing the tag after it; hdr is the 5-byte record header (the AAD).
static void t13_seal_(&T13Aead_ a, const unsigned char* hdr, unsigned char* data, unsigned long len) {
    unsigned char nonce[12];
    t13_nonce_(a, &nonce[0]);
    unsafe {
        if (a.alg == T13_CHACHA_) {
            chacha20_poly1305_seal(&a.key[0], &nonce[0], (const ?&unsigned char)hdr, 5UL, (&unsigned char)data, len,
                                   (&unsigned char)(data + len));
        } else {
            a.aes.gcm_encrypt(&nonce[0], (const ?&unsigned char)hdr, 5UL, (&unsigned char)data, len, (&unsigned char)(data + len));
        }
    }
    a.seq = a.seq + 1ULL;
}

static int t13_open_(&T13Aead_ a, const unsigned char* hdr, unsigned char* data, unsigned long len) {
    if (len < 16UL) { return 0; }
    unsigned char nonce[12];
    t13_nonce_(a, &nonce[0]);
    int ok = 0;
    unsafe {
        if (a.alg == T13_CHACHA_) {
            ok = chacha20_poly1305_open(&a.key[0], &nonce[0], (const ?&unsigned char)hdr, 5UL, (&unsigned char)data,
                                        len - 16UL, (const &unsigned char)(data + len - 16UL));
        } else {
            ok = a.aes.gcm_decrypt(&nonce[0], (const ?&unsigned char)hdr, 5UL, (&unsigned char)data, len - 16UL,
                                   (const &unsigned char)(data + len - 16UL));
        }
    }
    if (ok) { a.seq = a.seq + 1ULL; }
    return ok;
}

// ── records out ───────────────────────────────────────────────────────────────

// Queues one record (len <= 2^14) of content 'type'; protected when write
// keys are on unless 'plain'.
static void t13_queue_record_(&T13_ t, unsigned int type, const unsigned char* data, unsigned long len, int plain) {
    unsigned long start = t.tx.len;
    if (!t.wr_on || plain) {
        b8_(&t.tx, type);
        b16_(&t.tx, (type == (unsigned int)T13_CT_HS_ && t.st == T13_C_START_) ? 0x0301U : 0x0303U);
        b16_(&t.tx, (unsigned int)len);
        bput_(&t.tx, data, len);
        return;
    }
    unsigned long clen = len + 1UL + 16UL;
    b8_(&t.tx, (unsigned int)T13_CT_APP_);
    b16_(&t.tx, 0x0303U);
    b16_(&t.tx, (unsigned int)clen);
    bput_(&t.tx, data, len);
    b8_(&t.tx, type);
    unsigned long i = 0UL;
    while (i < 16UL) { t.tx.push((unsigned char)0); i = i + 1UL; }
    unsafe {
        unsigned char* base = vptr_(&t.tx) + start;
        t13_seal_(&t.wr, base, base + 5, len + 1UL);
    }
}

// A handshake message: added to the transcript, then queued (split
// across records as needed).
static void t13_queue_hs_(&T13_ t, const &Vec<unsigned char> msg) {
    unsafe {
        const unsigned char* p = vptr_(msg);
        bput_(&t.tr, p, msg.len);
        unsigned long off = 0UL;
        while (off < msg.len) {
            unsigned long n = msg.len - off;
            if (n > (unsigned long)T13_MAX_PLAIN_) { n = (unsigned long)T13_MAX_PLAIN_; }
            t13_queue_record_(t, (unsigned int)T13_CT_HS_, p + off, n, 0);
            off = off + n;
        }
    }
}

static void t13_queue_alert_(&T13_ t, int level, int desc) {
    unsigned char a[2];
    a[0] = (unsigned char)level;
    a[1] = (unsigned char)desc;
    t13_queue_record_(t, (unsigned int)T13_CT_ALERT_, &a[0], 2UL, 0);
}

// 0 when everything queued is sent, TLS_WANT_WRITE, or -1.
static int t13_flush_(&T13_ t, int fd) {
    while (t.tx_pos < t.tx.len) {
        long long n = 0LL;
        unsafe { n = tls_sock_send_(fd, (const &unsigned char)(vptr_(&t.tx) + t.tx_pos), t.tx.len - t.tx_pos); }
        if (n == (long long)TLS_WANT_WRITE) { return TLS_WANT_WRITE; }
        if (n <= 0LL) { return -1; }
        t.tx_pos = t.tx_pos + (unsigned long)n;
    }
    t.tx.clear();
    t.tx_pos = 0UL;
    return 0;
}

static struct T13_* t13_of_(&TlsStream s) {
    unsafe { return (struct T13_*)s.impl_; }
}

// Records an error, sends an alert (best effort, never blocking for it)
// and fails the stream.
static int t13_fail_(&TlsStream s, int code, int alert, const &char msg) {
    unsafe {
        struct T13_* t = t13_of_(s);
        if (t != (struct T13_*)0) {
            if (alert >= 0 && !t->sent_close) {
                t13_queue_alert_(t, 2, alert);
                t->sent_close = 1;
                t13_flush_(t, s.fd);
            }
            t->st = T13_FAILED_;
        }
    }
    tls_fail_(&s.err, code, msg);
    if (alert >= 0) { s.err.alert = alert; }
    return -1;
}

// ── records in ────────────────────────────────────────────────────────────────

// Reads until rbuf holds one whole record; 1 then (type, payload — the
// plaintext, decrypted in place — and the record's total size), 0 at EOF,
// TLS_WANT_READ, or -1 (s.err set).
static int t13_next_record_(&TlsStream s, &T13_ t, &unsigned int type, &unsigned long poff, &unsigned long plen,
                            &unsigned long total) {
    while (1) {
        if (t.rlen >= 5UL) {
            unsigned long len = ((unsigned long)t.rbuf[3] << 8) | (unsigned long)t.rbuf[4];
            if (len > (unsigned long)T13_MAX_CIPHER_) {
                return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_OVERFLOW_, "TLS record too large");
            }
            if (t.rlen >= 5UL + len) {
                unsigned int ct = (unsigned int)t.rbuf[0];
                *total = 5UL + len;
                if (t.rd_on && ct == (unsigned int)T13_CT_APP_) {
                    int ok = 0;
                    unsafe { ok = t13_open_(&t.rd, (const unsigned char*)&t.rbuf[0], (unsigned char*)&t.rbuf[5], len); }
                    if (!ok) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_BAD_MAC_, "TLS record failed authentication"); }
                    // strip padding: the last non-zero byte is the real type
                    unsigned long n = len - 16UL;
                    while (n > 0UL && t.rbuf[5UL + n - 1UL] == (unsigned char)0) { n = n - 1UL; }
                    if (n == 0UL) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "TLS record without a content type"); }
                    *type = (unsigned int)t.rbuf[5UL + n - 1UL];
                    *poff = 5UL;
                    *plen = n - 1UL;
                    if (*plen > (unsigned long)T13_MAX_PLAIN_) {
                        return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_OVERFLOW_, "TLS plaintext too large");
                    }
                    return 1;
                }
                if (ct == (unsigned int)T13_CT_CCS_) {
                    *type = ct; *poff = 5UL; *plen = len;
                    return 1;
                }
                if (t.rd_on && ct != (unsigned int)T13_CT_ALERT_) {
                    return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "unprotected TLS record after the handshake keys");
                }
                if (len > (unsigned long)T13_MAX_PLAIN_) {
                    return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_OVERFLOW_, "TLS record too large");
                }
                *type = ct; *poff = 5UL; *plen = len;
                return 1;
            }
        }
        long long n = 0LL;
        unsafe {
            n = tls_sock_recv_(s.fd, (&unsigned char)((unsigned char*)&t.rbuf[0] + t.rlen),
                               (unsigned long)T13_RBUF_ - t.rlen);
        }
        if (n == (long long)TLS_WANT_READ) { return TLS_WANT_READ; }
        if (n == 0LL) { return 0; }
        if (n < 0LL) {
            tls_fail_(&s.err, TLS_ERR_IO, "TLS: socket read failed");
            t.st = T13_FAILED_;
            return -1;
        }
        t.rlen = t.rlen + (unsigned long)n;
    }
    return -1;
}

static void t13_consume_(&T13_ t, unsigned long total) {
    unsafe {
        if (t.rlen > total) {
            memmove((void*)&t.rbuf[0], (const void*)((unsigned char*)&t.rbuf[0] + total), t.rlen - total);
        }
    }
    t.rlen = t.rlen - total;
}

// An alert record: close_notify -> 0 (peer closed), anything else fails.
static int t13_on_alert_(&TlsStream s, &T13_ t, unsigned long poff, unsigned long plen) {
    if (plen != 2UL) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "malformed TLS alert"); }
    int desc = (int)t.rbuf[poff + 1UL];
    if (desc == T13_A_CLOSE_NOTIFY_) {
        t.peer_closed = 1;
        return 0;
    }
    if (desc == 90) { return 1; }   // user_canceled: a warning, close_notify follows
    struct String m = String::from("TLS alert from peer: ");
    m.push_int((long long)desc);
    if (desc == T13_A_HS_FAILURE_) { m.push(" (handshake_failure)"); }
    else if (desc == T13_A_PROTOCOL_VERSION_) { m.push(" (protocol_version)"); }
    else if (desc == T13_A_BAD_CERT_) { m.push(" (bad_certificate)"); }
    else if (desc == T13_A_UNKNOWN_CA_) { m.push(" (unknown_ca)"); }
    else if (desc == T13_A_DECRYPT_) { m.push(" (decrypt_error)"); }
    else if (desc == T13_A_CERT_REQUIRED_) { m.push(" (certificate_required)"); }
    else if (desc == T13_A_NO_ALPN_) { m.push(" (no_application_protocol)"); }
    else if (desc == T13_A_UNRECOGNIZED_NAME_) { m.push(" (unrecognized_name)"); }
    else if (desc == T13_A_ILLEGAL_PARAM_) { m.push(" (illegal_parameter)"); }
    tls_fail_(&s.err, TLS_ERR_ALERT, m.as_ptr());
    m.free();
    s.err.alert = desc;
    t.st = T13_FAILED_;
    t.sent_close = 1;
    return -1;
}

// ── keys and certificates ─────────────────────────────────────────────────────

int tls_key_from_pem_(&TlsKey_ k, const &char pem, &TlsError e) {
    k.type = TLS_KEY_NONE_;
    struct Vec<unsigned char> der = Vec::new();
    int kind = 0;
    if (!tls_pem_private_key_(pem, &der, &kind)) {
        der.free();
        tls_fail_(e, TLS_ERR_CONFIG, "no PEM private key (PRIVATE KEY, RSA PRIVATE KEY or EC PRIVATE KEY)");
        return TLS_ERR_CONFIG;
    }
    int ok = 0;
    match (der.data) {
        case some(d): {
            if (kind != 2 && k.ec.init_der(d, der.len)) { k.type = TLS_KEY_EC_; ok = 1; }
            else if (kind != 3 && k.rsa.init_der(d, der.len)) { k.type = TLS_KEY_RSA_; ok = 1; }
        }
        case null: { }
    }
    unsafe { if (der.len > 0UL) { mem_zero_secure_raw_((void*)vptr_(&der), der.len); } }
    der.free();
    if (!ok) {
        tls_fail_(e, TLS_ERR_CONFIG, "unsupported private key (RSA 1024-4096 bits, EC P-256 or P-384)");
        return TLS_ERR_CONFIG;
    }
    return TLS_OK;
}

void tls_key_destroy_(&TlsKey_ k) {
    k.ec.destroy();
    k.rsa.destroy();
    k.type = TLS_KEY_NONE_;
}

// 1 if 'scheme' is one we can verify.
static int t13_sig_supported_(unsigned int scheme) {
    int i = 0;
    while (i < T13_NSIGALGS_) { if (t13_sigalgs_[i] == scheme) { return 1; } i = i + 1; }
    return 0;
}

static int t13_scheme_hash_(unsigned int scheme) {
    if (scheme == 0x0403U || scheme == 0x0804U || scheme == 0x0809U) { return DIGEST_SHA256; }
    if (scheme == 0x0503U || scheme == 0x0805U || scheme == 0x080aU) { return DIGEST_SHA384; }
    if (scheme == 0x0806U || scheme == 0x080bU || scheme == 0x0603U) { return DIGEST_SHA512; }
    return DIGEST_NONE;
}

// The 64 spaces || context string || 0 || transcript hash, hashed for 'scheme'.
static unsigned long t13_cv_digest_(&T13_ t, int server_side, unsigned int scheme, unsigned char* out) {
    unsigned char th[64];
    t13_tr_hash_(t, &th[0]);
    struct DigestCtx c = DigestCtx::new(t13_scheme_hash_(scheme));
    unsigned char sp[64];
    int i = 0;
    while (i < 64) { sp[i] = (unsigned char)0x20; i = i + 1; }
    c.update(&sp[0], 64UL);
    const &char ctx = server_side ? "TLS 1.3, server CertificateVerify" : "TLS 1.3, client CertificateVerify";
    unsafe { c.update((const &unsigned char)(const unsigned char*)ctx, str_len(ctx) + 1UL); }
    c.update(&th[0], t.hlen);
    unsafe { return c.finish((&unsigned char)out); }
}

// Verifies a CertificateVerify from the peer (whose leaf is peer_chain[0]).
static int t13_check_cv_(&TlsStream s, &T13_ t, unsigned int scheme, const unsigned char* sig, unsigned long slen) {
    if (!t13_sig_supported_(scheme) || scheme == 0x0603U) {
        return t13_fail_(s, TLS_ERR_HANDSHAKE, T13_A_ILLEGAL_PARAM_, "peer used a signature scheme we didn't offer");
    }
    unsigned char h[64];
    unsigned long hl = t13_cv_digest_(t, !t.server, scheme, &h[0]);
    &Vec<unsigned char> leaf = t.peer_chain.at(0UL);
    struct X509Parsed p;
    unsigned char* der = vptr_(leaf);
    int ok = 0;
    unsafe {
        if (x509_parse((const &unsigned char)der, leaf.len, &p) != 0) {
            return t13_fail_(s, TLS_ERR_CERT_INVALID, T13_A_BAD_CERT_, "peer certificate doesn't parse");
        }
        const unsigned char* key = der + p.key.off;
        if (scheme == 0x0403U || scheme == 0x0503U) {
            int curve = scheme == 0x0403U ? EC_P256 : EC_P384;
            int kt = scheme == 0x0403U ? X509_KEY_EC_P256 : X509_KEY_EC_P384;
            if (p.key_type == kt) {
                ok = ecdsa_verify_der(curve, (const &unsigned char)key, p.key.len, &h[0], hl, (const &unsigned char)sig, slen);
            }
        } else if (p.key_type == X509_KEY_RSA) {
            struct RsaPublicKey k;
            if (k.init_der((const &unsigned char)key, p.key.len)) {
                ok = rsa_verify_pss(&k, t13_scheme_hash_(scheme), &h[0], hl, (const &unsigned char)sig, slen, hl);
            }
        }
    }
    if (!ok) { return t13_fail_(s, TLS_ERR_HANDSHAKE, T13_A_DECRYPT_, "peer's CertificateVerify signature is invalid"); }
    return 0;
}

// Signs our CertificateVerify; the scheme is picked from what the peer accepts.
static int t13_make_cv_(&TlsStream s, &T13_ t, const &TlsKey_ key, &Vec<unsigned char> msg) {
    unsigned int scheme = 0U;
    int i = 0;
    unsigned int want[3];
    int nwant = 0;
    if (key.type == TLS_KEY_EC_) {
        want[0] = key.ec.curve == EC_P256 ? 0x0403U : 0x0503U;
        nwant = 1;
    } else {
        want[0] = 0x0804U; want[1] = 0x0805U; want[2] = 0x0806U;
        nwant = 3;
    }
    int w = 0;
    while (w < nwant && scheme == 0U) {
        i = 0;
        while (i < t.npeer_sigs) { if (t.peer_sigs[i] == want[w]) { scheme = want[w]; } i = i + 1; }
        w = w + 1;
    }
    if (scheme == 0U) {
        return t13_fail_(s, TLS_ERR_HANDSHAKE, T13_A_HS_FAILURE_, "the peer accepts no signature scheme for our key");
    }
    unsigned char h[64];
    unsigned long hl = t13_cv_digest_(t, t.server, scheme, &h[0]);
    unsigned char sig[RSA_MAX_BYTES];
    unsigned long slen = 0UL;
    int ok = 0;
    if (key.type == TLS_KEY_EC_) {
        ok = ecdsa_sign_der(key.ec.curve, &key.ec.d[0], &h[0], hl, &sig[0], (unsigned long)RSA_MAX_BYTES, &slen);
    } else {
        ok = rsa_sign_pss(&key.rsa, t13_scheme_hash_(scheme), &h[0], hl, &t.rng, &sig[0], (unsigned long)RSA_MAX_BYTES, &slen);
    }
    if (!ok) { return t13_fail_(s, TLS_ERR_INTERNAL, T13_A_INTERNAL_, "signing CertificateVerify failed"); }
    b8_(msg, (unsigned int)T13_CERT_VERIFY_);
    unsigned long l = bopen_(msg, 3);
    b16_(msg, scheme);
    b16_(msg, (unsigned int)slen);
    bput_(msg, &sig[0], slen);
    bclose_(msg, l, 3);
    return 0;
}

static void t13_put_certificate_(&Vec<unsigned char> msg, const &Vec<unsigned char> ctx,
                                 const &Vec<struct Vec<unsigned char>> chain) {
    b8_(msg, (unsigned int)T13_CERTIFICATE_);
    unsigned long l = bopen_(msg, 3);
    b8_(msg, (unsigned int)ctx.len);
    bput_(msg, vptr_(ctx), ctx.len);
    unsigned long ll = bopen_(msg, 3);
    unsigned long i = 0UL;
    while (i < chain.len) {
        &Vec<unsigned char> c = chain.at(i);
        b24_(msg, (unsigned int)c.len);
        bput_(msg, vptr_(c), c.len);
        b16_(msg, 0U);
        i = i + 1UL;
    }
    bclose_(msg, ll, 3);
    bclose_(msg, l, 3);
}

// Parses a Certificate message into t.peer_chain.
static int t13_read_certificate_(&TlsStream s, &T13_ t, &T13Rd_ r) {
    struct T13Rd_ ctx = rvec_(r, 1);
    struct T13Rd_ list = rvec_(r, 3);
    if (r.bad || ctx.bad || list.bad || rleft_(r) != 0UL) {
        return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "malformed Certificate message");
    }
    tls_der_list_free_(&t.peer_chain);
    t.peer_chain = Vec::new();
    while (rleft_(&list) > 0UL) {
        struct T13Rd_ cert = rvec_(&list, 3);
        struct T13Rd_ ext = rvec_(&list, 2);
        if (list.bad || cert.n == 0UL) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "malformed Certificate entry"); }
        struct Vec<unsigned char> d = Vec::new();
        bput_(&d, cert.p, cert.n);
        t.peer_chain.push(d);
        if (t.peer_chain.len > 16UL) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_BAD_CERT_, "certificate chain too long"); }
    }
    return 0;
}

static int t13_alert_for_cert_(int code) {
    if (code == TLS_ERR_CERT_UNTRUSTED) { return T13_A_UNKNOWN_CA_; }
    if (code == TLS_ERR_CERT_EXPIRED) { return T13_A_CERT_EXPIRED_; }
    if (code == TLS_ERR_NO_CERT) { return T13_A_CERT_REQUIRED_; }
    return T13_A_BAD_CERT_;
}

// Verifies t.peer_chain (the server's, on a client).
static int t13_verify_peer_chain_(&TlsStream s, &T13_ t, int peer_is_server) {
    struct TlsError e = TlsError::none();
    int code = TLS_OK;
    if (peer_is_server) {
        ?&TlsTrust_ sys = null;
        if (t.use_system) { sys = tls_trust_system_(); }
        ?&TlsTrust_ extra = null;
        unsafe { if (t.extra.certs.len > 0UL) { extra = (?&TlsTrust_)&t.extra; } }
        if (sys.is_null() && extra.is_null()) {
            return t13_fail_(s, TLS_ERR_CERT_UNTRUSTED, T13_A_UNKNOWN_CA_,
                             "no trusted roots: no system CA bundle found (set SSL_CERT_FILE or TlsConfig::ca_pem/ca_file)");
        }
        const ?&char host = null;
        if (t.verify_host && !t.name.is_empty()) { host = t.name.as_ptr(); }
        code = tls_verify_chain_(&t.peer_chain, sys, extra, host, 1, tls_now_(), &e);
    } else {
        ?&TlsTrust_ ca = null;
        unsafe { ca = (?&TlsTrust_)&t.acc->client_ca; }
        code = tls_verify_chain_(&t.peer_chain, ca, null, null, 0, tls_now_(), &e);
    }
    if (code != TLS_OK) {
        int r = t13_fail_(s, code, t13_alert_for_cert_(code), &e.message[0]);
        return r;
    }
    return 0;
}

// ── shared key exchange ───────────────────────────────────────────────────────

static int t13_group_ok_(unsigned int g) {
    return g == (unsigned int)T13_X25519_ || g == (unsigned int)T13_P256_ || g == (unsigned int)T13_P384_;
}

static int t13_ec_curve_(unsigned int g) { return g == (unsigned int)T13_P384_ ? EC_P384 : EC_P256; }

// Our key share for group g into msg (the key bytes only).
static int t13_make_share_(&T13_ t, unsigned int g, &Vec<unsigned char> out) {
    if (g == (unsigned int)T13_X25519_) {
        t.rng.fill(&t.x_priv[0], 32UL);
        unsigned char pub[32];
        x25519_public(&pub[0], &t.x_priv[0]);
        bput_(out, &pub[0], 32UL);
        return 1;
    }
    int curve = t13_ec_curve_(g);
    unsigned char pub[EC_MAX_POINT];
    if (!ec_keygen(curve, &t.rng, &t.ec_priv[0], &pub[0])) { return 0; }
    t.ec_kx_group = (int)g;
    bput_(out, &pub[0], ec_point_size(curve));
    return 1;
}

// The shared secret with the peer's share for group g into out; its length or 0.
static unsigned long t13_shared_(&T13_ t, unsigned int g, const unsigned char* peer, unsigned long plen, unsigned char* out) {
    unsafe {
        if (g == (unsigned int)T13_X25519_) {
            if (plen != 32UL) { return 0UL; }
            if (!x25519((&unsigned char)out, &t.x_priv[0], (const &unsigned char)peer)) { return 0UL; }
            return 32UL;
        }
        int curve = t13_ec_curve_(g);
        if (t.ec_kx_group != (int)g) { return 0UL; }
        if (!ecdh_shared(curve, &t.ec_priv[0], (const &unsigned char)peer, plen, (&unsigned char)out)) { return 0UL; }
        return ec_scalar_size(curve);
    }
}

// handshake secret -> c_hs/s_hs (transcript through ServerHello), master.
static void t13_handshake_secrets_(&T13_ t, const unsigned char* shared, unsigned long sl) {
    unsigned char zeros[48];
    unsigned char early[48];
    unsigned char derived[48];
    unsigned char hs[48];
    unsigned char empty_hash[64];
    unsigned char th[64];
    int i = 0;
    while (i < 48) { zeros[i] = (unsigned char)0; i = i + 1; }
    unsafe {
        digest(t.hash, null, 0UL, &empty_hash[0]);
        t13_extract_(t.hash, (const unsigned char*)0, 0UL, &zeros[0], t.hlen, &early[0]);
        t13_expand_label_(t.hash, &early[0], t.hlen, "derived", &empty_hash[0], t.hlen, &derived[0], t.hlen);
        t13_extract_(t.hash, &derived[0], t.hlen, shared, sl, &hs[0]);
        t13_tr_hash_(t, &th[0]);
        t13_expand_label_(t.hash, &hs[0], t.hlen, "c hs traffic", &th[0], t.hlen, (unsigned char*)&t.c_hs[0], t.hlen);
        t13_expand_label_(t.hash, &hs[0], t.hlen, "s hs traffic", &th[0], t.hlen, (unsigned char*)&t.s_hs[0], t.hlen);
        t13_expand_label_(t.hash, &hs[0], t.hlen, "derived", &empty_hash[0], t.hlen, &derived[0], t.hlen);
        t13_extract_(t.hash, &derived[0], t.hlen, &zeros[0], t.hlen, (unsigned char*)&t.master[0]);
    }
    mem_zero_secure(&hs[0], 48UL);
    mem_zero_secure(&early[0], 48UL);
}

// c_ap / s_ap from the transcript through the server Finished.
static void t13_app_secrets_(&T13_ t) {
    unsigned char th[64];
    t13_tr_hash_(t, &th[0]);
    unsafe {
        t13_expand_label_(t.hash, (const unsigned char*)&t.master[0], t.hlen, "c ap traffic", &th[0], t.hlen, (unsigned char*)&t.c_ap[0], t.hlen);
        t13_expand_label_(t.hash, (const unsigned char*)&t.master[0], t.hlen, "s ap traffic", &th[0], t.hlen, (unsigned char*)&t.s_ap[0], t.hlen);
    }
}

// verify_data for the Finished keyed by base_secret over the transcript so far.
static void t13_finished_(&T13_ t, const unsigned char* base, unsigned char* out) {
    unsigned char fk[48];
    unsigned char th[64];
    unsafe {
        t13_expand_label_(t.hash, base, t.hlen, "finished", (const unsigned char*)0, 0UL, &fk[0], t.hlen);
    }
    t13_tr_hash_(t, &th[0]);
    t13_hmac_(t.hash, &fk[0], t.hlen, &th[0], t.hlen, out);
    mem_zero_secure(&fk[0], 48UL);
}

static void t13_put_finished_(&T13_ t, const unsigned char* base, &Vec<unsigned char> msg) {
    unsigned char vd[48];
    t13_finished_(t, base, &vd[0]);
    b8_(msg, (unsigned int)T13_FINISHED_);
    b24_(msg, (unsigned int)t.hlen);
    bput_(msg, &vd[0], t.hlen);
}

static int t13_check_finished_(&TlsStream s, &T13_ t, const unsigned char* base, &T13Rd_ r) {
    unsigned char vd[48];
    t13_finished_(t, base, &vd[0]);
    const unsigned char* got = rbytes_(r, t.hlen);
    int ok = 0;
    unsafe { ok = !r.bad && rleft_(r) == 0UL && ct_eq((const &unsigned char)got, &vd[0], t.hlen); }
    if (!ok) { return t13_fail_(s, TLS_ERR_HANDSHAKE, T13_A_DECRYPT_, "the peer's Finished doesn't verify"); }
    return 0;
}

static void t13_set_suite_(&T13_ t, int suite) {
    t.suite = suite;
    t.hash = suite == 0x1302 ? DIGEST_SHA384 : DIGEST_SHA256;
    t.hlen = digest_size(t.hash);
}

// Replaces the transcript's first ClientHello by message_hash (after a
// HelloRetryRequest; RFC 8446 §4.4.1).
static void t13_tr_rehash_ch1_(&T13_ t) {
    unsigned char h[64];
    t13_tr_hash_(t, &h[0]);
    t.tr.clear();
    b8_(&t.tr, (unsigned int)T13_MESSAGE_HASH_);
    b24_(&t.tr, (unsigned int)t.hlen);
    bput_(&t.tr, &h[0], t.hlen);
}

static void t13_put_sigalgs_ext_(&Vec<unsigned char> m) {
    b16_(m, 0x000dU);
    unsigned long l = bopen_(m, 2);
    unsigned long l2 = bopen_(m, 2);
    int i = 0;
    while (i < T13_NSIGALGS_) { b16_(m, t13_sigalgs_[i]); i = i + 1; }
    bclose_(m, l2, 2);
    bclose_(m, l, 2);
}

static void t13_read_sigalgs_(&T13_ t, &T13Rd_ e) {
    struct T13Rd_ list = rvec_(e, 2);
    t.npeer_sigs = 0;
    while (!list.bad && rleft_(&list) >= 2UL) {
        unsigned int sch = r16_(&list);
        if (t.npeer_sigs < 32) { t.peer_sigs[t.npeer_sigs] = sch; t.npeer_sigs = t.npeer_sigs + 1; }
    }
}

// ── client ────────────────────────────────────────────────────────────────────

static int t13_send_client_hello_(&TlsStream s, &T13_ t, unsigned int share_group) {
    struct Vec<unsigned char> m = Vec::new();
    b8_(&m, (unsigned int)T13_CLIENT_HELLO_);
    unsigned long l = bopen_(&m, 3);
    b16_(&m, 0x0303U);
    bput_(&m, &t.random[0], 32UL);
    b8_(&m, (unsigned int)t.sid_len);
    bput_(&m, &t.sid[0], t.sid_len);
    b16_(&m, 6U);
    b16_(&m, 0x1303U); b16_(&m, 0x1301U); b16_(&m, 0x1302U);
    b8_(&m, 1U); b8_(&m, 0U);
    unsigned long el = bopen_(&m, 2);
    // server_name
    if (!t.name.is_empty() && !tls_name_is_ip_(t.name.as_ptr())) {
        unsigned long nl = t.name.length();
        b16_(&m, 0x0000U);
        b16_(&m, (unsigned int)(nl + 5UL));
        b16_(&m, (unsigned int)(nl + 3UL));
        b8_(&m, 0U);
        b16_(&m, (unsigned int)nl);
        unsafe { bput_(&m, (const unsigned char*)t.name.as_ptr(), nl); }
    }
    // supported_groups
    b16_(&m, 0x000aU); b16_(&m, 8U); b16_(&m, 6U);
    b16_(&m, (unsigned int)T13_X25519_); b16_(&m, (unsigned int)T13_P256_); b16_(&m, (unsigned int)T13_P384_);
    t13_put_sigalgs_ext_(&m);
    // supported_versions: TLS 1.3 only
    b16_(&m, 0x002bU); b16_(&m, 3U); b8_(&m, 2U); b16_(&m, 0x0304U);
    // key_share
    b16_(&m, 0x0033U);
    unsigned long kl = bopen_(&m, 2);
    unsigned long kl2 = bopen_(&m, 2);
    b16_(&m, share_group);
    unsigned long kl3 = bopen_(&m, 2);
    if (!t13_make_share_(t, share_group, &m)) { m.free(); return t13_fail_(s, TLS_ERR_INTERNAL, -1, "key generation failed"); }
    bclose_(&m, kl3, 2);
    bclose_(&m, kl2, 2);
    bclose_(&m, kl, 2);
    // ALPN
    if (t.alpn.len > 0UL) {
        b16_(&m, 0x0010U);
        b16_(&m, (unsigned int)(t.alpn.len + 2UL));
        b16_(&m, (unsigned int)t.alpn.len);
        bput_(&m, vptr_(&t.alpn), t.alpn.len);
    }
    // cookie (echoed after a HelloRetryRequest)
    if (t.cookie.len > 0UL) {
        b16_(&m, 0x002cU);
        b16_(&m, (unsigned int)(t.cookie.len + 2UL));
        b16_(&m, (unsigned int)t.cookie.len);
        bput_(&m, vptr_(&t.cookie), t.cookie.len);
    }
    bclose_(&m, el, 2);
    bclose_(&m, l, 3);
    t13_queue_hs_(t, &m);
    m.free();
    t.st = T13_C_WAIT_SH_;
    return 0;
}

static int t13_on_server_hello_(&TlsStream s, &T13_ t, &T13Rd_ r, const unsigned char* msg, unsigned long mlen) {
    unsigned int legacy = r16_(r);
    const unsigned char* random = rbytes_(r, 32UL);
    struct T13Rd_ sid = rvec_(r, 1);
    unsigned int suite = r16_(r);
    unsigned int comp = r8_(r);
    struct T13Rd_ exts = rvec_(r, 2);
    if (r.bad || sid.bad || exts.bad) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "malformed ServerHello"); }
    int isHrr = 1;
    unsafe {
        int i = 0;
        while (i < 32) { if (random[i] != t13_hrr_random_[i]) { isHrr = 0; } i = i + 1; }
    }
    unsigned int version = 0U;
    unsigned int group = 0U;
    const unsigned char* key = (const unsigned char*)0;
    unsigned long keyLen = 0UL;
    struct T13Rd_ cookie = rd_new_((const unsigned char*)0, 0UL);
    int hasCookie = 0;
    while (rleft_(&exts) > 0UL) {
        unsigned int et = r16_(&exts);
        struct T13Rd_ e = rvec_(&exts, 2);
        if (exts.bad) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "malformed ServerHello extensions"); }
        if (et == 0x002bU) { version = r16_(&e); }
        else if (et == 0x0033U) {
            group = r16_(&e);
            if (!isHrr) {
                struct T13Rd_ k = rvec_(&e, 2);
                key = k.p;
                keyLen = k.n;
                if (k.bad) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "malformed key_share"); }
            }
        } else if (et == 0x002cU) {
            cookie = rvec_(&e, 2);
            hasCookie = 1;
        }
        if (e.bad) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "malformed ServerHello extension"); }
    }
    if (version != 0x0304U) {
        // A TLS 1.2 server (or older): this backend speaks 1.3 only.
        return t13_fail_(s, TLS_ERR_HANDSHAKE, T13_A_PROTOCOL_VERSION_,
                         "the server doesn't support TLS 1.3 (the SafeC TLS backend implements TLS 1.3 only)");
    }
    if (legacy != 0x0303U || comp != 0U || sid.n != t.sid_len) {
        return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_ILLEGAL_PARAM_, "invalid ServerHello");
    }
    unsafe {
        if (sid.n > 0UL && !ct_eq((const &unsigned char)sid.p, &t.sid[0], sid.n)) {
            return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_ILLEGAL_PARAM_, "ServerHello session id mismatch");
        }
    }
    if (suite != 0x1301U && suite != 0x1302U && suite != 0x1303U) {
        return t13_fail_(s, TLS_ERR_HANDSHAKE, T13_A_ILLEGAL_PARAM_, "server chose a cipher suite we didn't offer");
    }
    if (t.hrr && t.suite != (int)suite) {
        return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_ILLEGAL_PARAM_, "cipher suite changed after HelloRetryRequest");
    }
    t13_set_suite_(t, (int)suite);
    if (isHrr) {
        if (t.hrr) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "second HelloRetryRequest"); }
        if (!t13_group_ok_(group) || group == (unsigned int)t.group) {
            return t13_fail_(s, TLS_ERR_HANDSHAKE, T13_A_ILLEGAL_PARAM_, "HelloRetryRequest asks for an unsupported group");
        }
        t.hrr = 1;
        t.group = (int)group;
        // transcript: message_hash(CH1) || HRR
        t13_tr_rehash_ch1_(t);
        bput_(&t.tr, msg, mlen);
        t.cookie.clear();
        if (hasCookie && cookie.n > 0UL) { bput_(&t.cookie, cookie.p, cookie.n); }
        // RFC 8446 D.4: a ChangeCipherSpec before the second ClientHello.
        unsigned char ccs = (unsigned char)1;
        t13_queue_record_(t, (unsigned int)T13_CT_CCS_, &ccs, 1UL, 1);
        return t13_send_client_hello_(s, t, group);
    }
    if (group != (unsigned int)t.group || key == (const unsigned char*)0) {
        return t13_fail_(s, TLS_ERR_HANDSHAKE, T13_A_ILLEGAL_PARAM_, "server key share for a group we didn't send");
    }
    bput_(&t.tr, msg, mlen);
    unsigned char shared[64];
    unsigned long sl = t13_shared_(t, group, key, keyLen, &shared[0]);
    if (sl == 0UL) { return t13_fail_(s, TLS_ERR_HANDSHAKE, T13_A_ILLEGAL_PARAM_, "invalid server key share"); }
    t13_handshake_secrets_(t, &shared[0], sl);
    mem_zero_secure(&shared[0], 64UL);
    mem_zero_secure(&t.x_priv[0], 32UL);
    t13_set_read_(t, &t.s_hs[0]);
    t.st = T13_C_WAIT_EE_;
    return 0;
}

static int t13_on_encrypted_extensions_(&TlsStream s, &T13_ t, &T13Rd_ r) {
    struct T13Rd_ exts = rvec_(r, 2);
    if (r.bad || rleft_(r) != 0UL) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "malformed EncryptedExtensions"); }
    while (rleft_(&exts) > 0UL) {
        unsigned int et = r16_(&exts);
        struct T13Rd_ e = rvec_(&exts, 2);
        if (exts.bad) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "malformed EncryptedExtensions"); }
        if (et == 0x0010U) {
            struct T13Rd_ list = rvec_(&e, 2);
            struct T13Rd_ name = rvec_(&list, 1);
            if (e.bad || list.bad || rleft_(&list) != 0UL || name.n == 0UL) {
                return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "malformed ALPN response");
            }
            // must be one we offered
            struct T13Rd_ ours = rd_new_(vptr_(&t.alpn), t.alpn.len);
            int found = 0;
            while (rleft_(&ours) > 0UL) {
                struct T13Rd_ o = rvec_(&ours, 1);
                unsafe { if (o.n == name.n && ct_eq((const &unsigned char)o.p, (const &unsigned char)name.p, o.n)) { found = 1; } }
            }
            if (!found || name.n >= 64UL) {
                return t13_fail_(s, TLS_ERR_HANDSHAKE, T13_A_ILLEGAL_PARAM_, "server selected an ALPN protocol we didn't offer");
            }
            unsafe {
                unsigned long i = 0UL;
                while (i < name.n) { s.alpn_[i] = (char)name.p[i]; i = i + 1UL; }
                s.alpn_[name.n] = (char)0;
            }
        }
    }
    t.st = T13_C_WAIT_CERT_CR_;
    return 0;
}

static int t13_on_cert_request_(&TlsStream s, &T13_ t, &T13Rd_ r) {
    struct T13Rd_ ctx = rvec_(r, 1);
    struct T13Rd_ exts = rvec_(r, 2);
    if (r.bad || rleft_(r) != 0UL) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "malformed CertificateRequest"); }
    t.cr_context.clear();
    bput_(&t.cr_context, ctx.p, ctx.n);
    while (rleft_(&exts) > 0UL) {
        unsigned int et = r16_(&exts);
        struct T13Rd_ e = rvec_(&exts, 2);
        if (exts.bad) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "malformed CertificateRequest"); }
        if (et == 0x000dU) { t13_read_sigalgs_(t, &e); }
    }
    t.cert_requested = 1;
    t.st = T13_C_WAIT_CERT_;
    return 0;
}

static int t13_client_finish_(&TlsStream s, &T13_ t) {
    // app secrets from the transcript through the server Finished
    t13_app_secrets_(t);
    t13_set_read_(t, &t.s_ap[0]);
    unsigned char ccs = (unsigned char)1;
    t13_queue_record_(t, (unsigned int)T13_CT_CCS_, &ccs, 1UL, 1);
    t13_set_write_(t, &t.c_hs[0]);
    if (t.cert_requested) {
        struct Vec<unsigned char> m = Vec::new();
        struct Vec<struct Vec<unsigned char>> none = Vec::new();
        int send = t.has_key && t.own_chain.len > 0UL;
        if (send) { t13_put_certificate_(&m, &t.cr_context, &t.own_chain); }
        else { t13_put_certificate_(&m, &t.cr_context, &none); }
        t13_queue_hs_(t, &m);
        m.clear();
        if (send) {
            if (t13_make_cv_(s, t, &t.key, &m) != 0) { m.free(); return -1; }
            t13_queue_hs_(t, &m);
        }
        m.free();
        none.free();
    }
    struct Vec<unsigned char> fin = Vec::new();
    t13_put_finished_(t, &t.c_hs[0], &fin);
    t13_queue_hs_(t, &fin);
    fin.free();
    t13_set_write_(t, &t.c_ap[0]);
    t.st = T13_DONE_;
    return 0;
}

static int t13_client_msg_(&TlsStream s, &T13_ t, unsigned int type, &T13Rd_ r, const unsigned char* msg, unsigned long mlen) {
    if (t.st == T13_C_WAIT_SH_) {
        if (type != (unsigned int)T13_SERVER_HELLO_) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "expected ServerHello"); }
        return t13_on_server_hello_(s, t, r, msg, mlen);
    }
    if (t.st == T13_C_WAIT_EE_) {
        if (type != (unsigned int)T13_ENC_EXT_) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "expected EncryptedExtensions"); }
        int rc = t13_on_encrypted_extensions_(s, t, r);
        bput_(&t.tr, msg, mlen);
        return rc;
    }
    if (t.st == T13_C_WAIT_CERT_CR_ && type == (unsigned int)T13_CERT_REQUEST_) {
        int rc = t13_on_cert_request_(s, t, r);
        bput_(&t.tr, msg, mlen);
        return rc;
    }
    if (t.st == T13_C_WAIT_CERT_CR_ || t.st == T13_C_WAIT_CERT_) {
        if (type != (unsigned int)T13_CERTIFICATE_) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "expected Certificate"); }
        if (t13_read_certificate_(s, t, r) != 0) { return -1; }
        if (t.peer_chain.len == 0UL) {
            return t13_fail_(s, TLS_ERR_NO_CERT, T13_A_DECODE_, "the server sent no certificate");
        }
        if (t.verify && t13_verify_peer_chain_(s, t, 1) != 0) { return -1; }
        bput_(&t.tr, msg, mlen);
        t.st = T13_C_WAIT_CV_;
        return 0;
    }
    if (t.st == T13_C_WAIT_CV_) {
        if (type != (unsigned int)T13_CERT_VERIFY_) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "expected CertificateVerify"); }
        unsigned int scheme = r16_(r);
        struct T13Rd_ sig = rvec_(r, 2);
        if (r.bad || rleft_(r) != 0UL) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "malformed CertificateVerify"); }
        if (t13_check_cv_(s, t, scheme, sig.p, sig.n) != 0) { return -1; }
        bput_(&t.tr, msg, mlen);
        t.st = T13_C_WAIT_FIN_;
        return 0;
    }
    if (t.st == T13_C_WAIT_FIN_) {
        if (type != (unsigned int)T13_FINISHED_) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "expected Finished"); }
        if (t13_check_finished_(s, t, &t.s_hs[0], r) != 0) { return -1; }
        bput_(&t.tr, msg, mlen);
        return t13_client_finish_(s, t);
    }
    return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "unexpected handshake message");
}

// ── server ────────────────────────────────────────────────────────────────────

// What the server learns from a ClientHello.
struct T13Ch_ {
    int           ok13;           // supported_versions includes 1.3
    unsigned int  suite;
    unsigned int  share_group;    // a group with a usable key share, 0 if none
    const unsigned char* share;
    unsigned long share_len;
    unsigned int  hrr_group;      // a supported group to ask for, 0 if none
    const unsigned char* sid;
    unsigned long sid_len;
    struct T13Rd_ alpn;           // the client's protocol list
    int           has_alpn;
    int           has_sigalgs;
};

static int t13_parse_ch_(&TlsStream s, &T13_ t, &T13Rd_ r, &T13Ch_ ch) {
    ch.ok13 = 0; ch.suite = 0U; ch.share_group = 0U; ch.share = (const unsigned char*)0; ch.share_len = 0UL;
    ch.hrr_group = 0U; ch.has_alpn = 0; ch.has_sigalgs = 0;
    r16_(r);
    rbytes_(r, 32UL);
    struct T13Rd_ sid = rvec_(r, 1);
    struct T13Rd_ suites = rvec_(r, 2);
    struct T13Rd_ comp = rvec_(r, 1);
    struct T13Rd_ exts = rvec_(r, 2);
    if (r.bad || sid.bad || suites.bad || comp.bad || exts.bad || sid.n > 32UL) {
        return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "malformed ClientHello");
    }
    ch.sid = sid.p;
    ch.sid_len = sid.n;
    // our preference: ChaCha20-Poly1305 (constant time here), AES-128-GCM, AES-256-GCM
    unsigned int prefs[3];
    prefs[0] = 0x1303U; prefs[1] = 0x1301U; prefs[2] = 0x1302U;
    int pi = 0;
    while (pi < 3 && ch.suite == 0U) {
        struct T13Rd_ sc = suites;
        while (rleft_(&sc) >= 2UL) { if (r16_(&sc) == prefs[pi]) { ch.suite = prefs[pi]; } }
        pi = pi + 1;
    }
    unsigned int groups[3];
    int ng = 0;
    while (rleft_(&exts) > 0UL) {
        unsigned int et = r16_(&exts);
        struct T13Rd_ e = rvec_(&exts, 2);
        if (exts.bad) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "malformed ClientHello extensions"); }
        if (et == 0x002bU) {
            struct T13Rd_ v = rvec_(&e, 1);
            while (rleft_(&v) >= 2UL) { if (r16_(&v) == 0x0304U) { ch.ok13 = 1; } }
        } else if (et == 0x000aU) {
            struct T13Rd_ g = rvec_(&e, 2);
            while (rleft_(&g) >= 2UL) {
                unsigned int gg = r16_(&g);
                if (t13_group_ok_(gg) && ng < 3) { groups[ng] = gg; ng = ng + 1; }
            }
        } else if (et == 0x000dU) {
            t13_read_sigalgs_(t, &e);
            ch.has_sigalgs = 1;
        } else if (et == 0x0033U) {
            struct T13Rd_ shares = rvec_(&e, 2);
            int bestRank = 9;
            while (rleft_(&shares) > 0UL) {
                unsigned int g = r16_(&shares);
                struct T13Rd_ k = rvec_(&shares, 2);
                if (shares.bad) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "malformed key_share"); }
                int rank = g == (unsigned int)T13_X25519_ ? 0 : (g == (unsigned int)T13_P256_ ? 1 : (g == (unsigned int)T13_P384_ ? 2 : 9));
                if (rank < bestRank) { bestRank = rank; ch.share_group = g; ch.share = k.p; ch.share_len = k.n; }
            }
        } else if (et == 0x0010U) {
            ch.alpn = rvec_(&e, 2);
            ch.has_alpn = !ch.alpn.bad;
        } else if (et == 0x0000U) {
            struct T13Rd_ list = rvec_(&e, 2);
            unsigned int nt = r8_(&list);
            struct T13Rd_ host = rvec_(&list, 2);
            if (!list.bad && nt == 0U && host.n < 256UL) {
                t.name.clear();
                unsafe { t.name.push_n((const &char)(const char*)host.p, host.n); }
            }
        }
    }
    // a group to request on HelloRetryRequest: x25519, P-256, P-384 in that order
    int k = 0;
    int bestRank = 9;
    while (k < ng) {
        int rank = groups[k] == (unsigned int)T13_X25519_ ? 0 : (groups[k] == (unsigned int)T13_P256_ ? 1 : 2);
        if (rank < bestRank) { bestRank = rank; ch.hrr_group = groups[k]; }
        k = k + 1;
    }
    return 0;
}

static void t13_put_server_hello_(&T13_ t, &Vec<unsigned char> m, int hrr, const unsigned char* sid, unsigned long sid_len,
                                  const &Vec<unsigned char> share) {
    b8_(m, (unsigned int)T13_SERVER_HELLO_);
    unsigned long l = bopen_(m, 3);
    b16_(m, 0x0303U);
    if (hrr) { bput_(m, &t13_hrr_random_[0], 32UL); } else { bput_(m, &t.random[0], 32UL); }
    b8_(m, (unsigned int)sid_len);
    bput_(m, sid, sid_len);
    b16_(m, (unsigned int)t.suite);
    b8_(m, 0U);
    unsigned long el = bopen_(m, 2);
    b16_(m, 0x002bU); b16_(m, 2U); b16_(m, 0x0304U);
    b16_(m, 0x0033U);
    unsigned long kl = bopen_(m, 2);
    b16_(m, (unsigned int)t.group);
    if (!hrr) {
        b16_(m, (unsigned int)share.len);
        bput_(m, vptr_(share), share.len);
    }
    bclose_(m, kl, 2);
    bclose_(m, el, 2);
    bclose_(m, l, 3);
}

static int t13_on_client_hello_(&TlsStream s, &T13_ t, &T13Rd_ r, const unsigned char* msg, unsigned long mlen) {
    struct T13Ch_ ch;
    if (t13_parse_ch_(s, t, r, &ch) != 0) { return -1; }
    if (!ch.ok13) {
        return t13_fail_(s, TLS_ERR_HANDSHAKE, T13_A_PROTOCOL_VERSION_,
                         "the client doesn't support TLS 1.3 (the SafeC TLS backend implements TLS 1.3 only)");
    }
    if (ch.suite == 0U) { return t13_fail_(s, TLS_ERR_HANDSHAKE, T13_A_HS_FAILURE_, "no cipher suite in common"); }
    if (!ch.has_sigalgs) { return t13_fail_(s, TLS_ERR_HANDSHAKE, T13_A_MISSING_EXT_, "ClientHello without signature_algorithms"); }
    if (t.hrr && ((int)ch.suite != t.suite || ch.share_group != (unsigned int)t.group)) {
        return t13_fail_(s, TLS_ERR_HANDSHAKE, T13_A_ILLEGAL_PARAM_, "second ClientHello doesn't follow the HelloRetryRequest");
    }
    t13_set_suite_(t, (int)ch.suite);
    if (ch.share_group == 0U) {
        if (t.hrr || ch.hrr_group == 0U) { return t13_fail_(s, TLS_ERR_HANDSHAKE, T13_A_HS_FAILURE_, "no key exchange group in common"); }
        // HelloRetryRequest for a group the client supports.
        t.hrr = 1;
        t.group = (int)ch.hrr_group;
        bput_(&t.tr, msg, mlen);
        t13_tr_rehash_ch1_(t);
        struct Vec<unsigned char> m = Vec::new();
        struct Vec<unsigned char> none = Vec::new();
        t13_put_server_hello_(t, &m, 1, ch.sid, ch.sid_len, &none);
        t13_queue_hs_(t, &m);
        if (ch.sid_len > 0UL) {
            unsigned char ccs = (unsigned char)1;
            t13_queue_record_(t, (unsigned int)T13_CT_CCS_, &ccs, 1UL, 1);
        }
        t.sid_len = ch.sid_len;
        m.free();
        none.free();
        t.st = T13_S_WAIT_CH_;
        return 0;
    }
    t.group = (int)ch.share_group;
    bput_(&t.tr, msg, mlen);
    // our share and the shared secret
    struct Vec<unsigned char> share = Vec::new();
    if (!t13_make_share_(t, ch.share_group, &share)) { share.free(); return t13_fail_(s, TLS_ERR_INTERNAL, T13_A_INTERNAL_, "key generation failed"); }
    unsigned char shared[64];
    unsigned long sl = t13_shared_(t, ch.share_group, ch.share, ch.share_len, &shared[0]);
    if (sl == 0UL) { share.free(); return t13_fail_(s, TLS_ERR_HANDSHAKE, T13_A_ILLEGAL_PARAM_, "invalid client key share"); }
    // ALPN: the first of our protocols the client offers
    struct T13Acc_* acc = t.acc;
    struct Vec<unsigned char> chosen = Vec::new();
    unsafe {
        if (ch.has_alpn && acc->alpn.len > 0UL) {
            struct T13Rd_ ours = rd_new_(vptr_(&acc->alpn), acc->alpn.len);
            while (rleft_(&ours) > 0UL && chosen.len == 0UL) {
                struct T13Rd_ o = rvec_(&ours, 1);
                struct T13Rd_ theirs = ch.alpn;
                while (rleft_(&theirs) > 0UL) {
                    struct T13Rd_ c = rvec_(&theirs, 1);
                    if (!theirs.bad && c.n == o.n && o.n > 0UL && ct_eq((const &unsigned char)c.p, (const &unsigned char)o.p, o.n)) {
                        bput_(&chosen, o.p, o.n);
                        break;
                    }
                }
            }
        }
    }
    struct Vec<unsigned char> m = Vec::new();
    t13_put_server_hello_(t, &m, 0, ch.sid, ch.sid_len, &share);
    t13_queue_hs_(t, &m);
    share.free();
    m.clear();
    if (ch.sid_len > 0UL && !t.hrr) {
        unsigned char ccs = (unsigned char)1;
        t13_queue_record_(t, (unsigned int)T13_CT_CCS_, &ccs, 1UL, 1);
    }
    t13_handshake_secrets_(t, &shared[0], sl);
    mem_zero_secure(&shared[0], 64UL);
    t13_set_write_(t, &t.s_hs[0]);
    t13_set_read_(t, &t.c_hs[0]);
    // EncryptedExtensions
    b8_(&m, (unsigned int)T13_ENC_EXT_);
    unsigned long l = bopen_(&m, 3);
    unsigned long el = bopen_(&m, 2);
    if (chosen.len > 0UL) {
        b16_(&m, 0x0010U);
        b16_(&m, (unsigned int)(chosen.len + 3UL));
        b16_(&m, (unsigned int)(chosen.len + 1UL));
        b8_(&m, (unsigned int)chosen.len);
        bput_(&m, vptr_(&chosen), chosen.len);
        unsafe {
            unsigned long i = 0UL;
            while (i < chosen.len && i < 63UL) { s.alpn_[i] = (char)vptr_(&chosen)[i]; i = i + 1UL; }
            s.alpn_[i] = (char)0;
        }
    }
    chosen.free();
    bclose_(&m, el, 2);
    bclose_(&m, l, 3);
    t13_queue_hs_(t, &m);
    m.clear();
    unsafe {
        if (acc->verify_client != 0) {
            b8_(&m, (unsigned int)T13_CERT_REQUEST_);
            unsigned long cl = bopen_(&m, 3);
            b8_(&m, 0U);
            unsigned long cel = bopen_(&m, 2);
            t13_put_sigalgs_ext_(&m);
            bclose_(&m, cel, 2);
            bclose_(&m, cl, 3);
            t13_queue_hs_(t, &m);
            m.clear();
            t.cert_requested = 1;
        }
        struct Vec<unsigned char> noctx = Vec::new();
        t13_put_certificate_(&m, &noctx, &acc->chain);
        t13_queue_hs_(t, &m);
        m.clear();
        if (t13_make_cv_(s, t, &acc->key, &m) != 0) { m.free(); return -1; }
        t13_queue_hs_(t, &m);
        m.clear();
    }
    t13_put_finished_(t, &t.s_hs[0], &m);
    t13_queue_hs_(t, &m);
    m.free();
    t13_app_secrets_(t);
    t13_set_write_(t, &t.s_ap[0]);
    t.st = t.cert_requested ? T13_S_WAIT_CERT_ : T13_S_WAIT_FIN_;
    return 0;
}

static int t13_server_msg_(&TlsStream s, &T13_ t, unsigned int type, &T13Rd_ r, const unsigned char* msg, unsigned long mlen) {
    if (t.st == T13_S_WAIT_CH_) {
        if (type != (unsigned int)T13_CLIENT_HELLO_) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "expected ClientHello"); }
        return t13_on_client_hello_(s, t, r, msg, mlen);
    }
    if (t.st == T13_S_WAIT_CERT_) {
        if (type != (unsigned int)T13_CERTIFICATE_) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "expected client Certificate"); }
        if (t13_read_certificate_(s, t, r) != 0) { return -1; }
        bput_(&t.tr, msg, mlen);
        int required = 0;
        unsafe { required = t.acc->verify_client == 2; }
        if (t.peer_chain.len == 0UL) {
            if (required) { return t13_fail_(s, TLS_ERR_NO_CERT, T13_A_CERT_REQUIRED_, "the client sent no certificate"); }
            t.st = T13_S_WAIT_FIN_;
            return 0;
        }
        if (t13_verify_peer_chain_(s, t, 0) != 0) { return -1; }
        t.peer_sent_cert = 1;
        t.st = T13_S_WAIT_CV_;
        return 0;
    }
    if (t.st == T13_S_WAIT_CV_) {
        if (type != (unsigned int)T13_CERT_VERIFY_) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "expected client CertificateVerify"); }
        unsigned int scheme = r16_(r);
        struct T13Rd_ sig = rvec_(r, 2);
        if (r.bad || rleft_(r) != 0UL) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "malformed CertificateVerify"); }
        if (t13_check_cv_(s, t, scheme, sig.p, sig.n) != 0) { return -1; }
        bput_(&t.tr, msg, mlen);
        t.st = T13_S_WAIT_FIN_;
        return 0;
    }
    if (t.st == T13_S_WAIT_FIN_) {
        if (type != (unsigned int)T13_FINISHED_) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "expected client Finished"); }
        if (t13_check_finished_(s, t, &t.c_hs[0], r) != 0) { return -1; }
        bput_(&t.tr, msg, mlen);
        t13_set_read_(t, &t.c_ap[0]);
        t.st = T13_DONE_;
        return 0;
    }
    return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "unexpected handshake message");
}

// ── post-handshake messages ───────────────────────────────────────────────────

static int t13_post_hs_(&TlsStream s, &T13_ t, unsigned int type, &T13Rd_ r) {
    if (type == (unsigned int)T13_NEW_TICKET_ && !t.server) { return 0; }
    if (type == (unsigned int)T13_KEY_UPDATE_) {
        unsigned int req = r8_(r);
        if (r.bad || rleft_(r) != 0UL || req > 1U) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "malformed KeyUpdate"); }
        unsigned char next[48];
        unsafe { t13_expand_label_(t.hash, (const unsigned char*)&t.rd_secret[0], t.hlen, "traffic upd", (const unsigned char*)0, 0UL, &next[0], t.hlen); }
        t13_set_read_(t, &next[0]);
        if (req == 1U) {
            struct Vec<unsigned char> m = Vec::new();
            b8_(&m, (unsigned int)T13_KEY_UPDATE_);
            b24_(&m, 1U);
            b8_(&m, 0U);
            unsafe { t13_queue_record_(t, (unsigned int)T13_CT_HS_, vptr_(&m), m.len, 0); }
            m.free();
            unsafe { t13_expand_label_(t.hash, (const unsigned char*)&t.wr_secret[0], t.hlen, "traffic upd", (const unsigned char*)0, 0UL, &next[0], t.hlen); }
            t13_set_write_(t, &next[0]);
        }
        mem_zero_secure(&next[0], 48UL);
        return 0;
    }
    return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "unexpected post-handshake message");
}

// Handles every complete handshake message in t.hs. During the handshake
// (post == 0) stops once the handshake is done.
static int t13_drain_hs_(&TlsStream s, &T13_ t, int post) {
    while (t.hs.len - t.hs_pos >= 4UL) {
        unsigned char* p = vptr_(&t.hs);
        unsigned int type = 0U;
        unsigned long blen = 0UL;
        unsafe {
            type = (unsigned int)p[t.hs_pos];
            blen = ((unsigned long)p[t.hs_pos + 1UL] << 16) | ((unsigned long)p[t.hs_pos + 2UL] << 8) | (unsigned long)p[t.hs_pos + 3UL];
        }
        if (blen > (unsigned long)T13_MAX_HS_) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "handshake message too large"); }
        if (t.hs.len - t.hs_pos < 4UL + blen) { break; }
        const unsigned char* msg;
        unsafe { msg = (const unsigned char*)(p + t.hs_pos); }
        struct T13Rd_ r;
        unsafe { r = rd_new_(msg + 4, blen); }
        int rc = 0;
        if (post) { rc = t13_post_hs_(s, t, type, &r); }
        else if (t.server) { rc = t13_server_msg_(s, t, type, &r, msg, 4UL + blen); }
        else { rc = t13_client_msg_(s, t, type, &r, msg, 4UL + blen); }
        t.hs_pos = t.hs_pos + 4UL + blen;
        if (rc != 0) { return -1; }
        if (!post && t.st == T13_DONE_) {
            if (t.hs_pos < t.hs.len) {
                // Anything after the last handshake message would be under
                // other keys: only NewSessionTicket & co. may follow, later.
                return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "handshake data after Finished");
            }
            break;
        }
    }
    if (t.hs_pos == t.hs.len) { t.hs.clear(); t.hs_pos = 0UL; }
    return 0;
}

// ── backend entry points ──────────────────────────────────────────────────────

static int t13_alloc_(&TlsStream s) {
    void* mem = alloc_zeroed_raw_(sizeof(struct T13_));
    if (mem == (void*)0) { return tls_fail_(&s.err, TLS_ERR_NO_MEMORY, "out of memory"); }
    s.impl_ = mem;
    unsafe {
        struct T13_* t = (struct T13_*)mem;
        t->name = String::new();
        t->alpn = Vec::new();
        t->extra = tls_trust_new_();
        t->own_chain = Vec::new();
        t->tx = Vec::new();
        t->hs = Vec::new();
        t->tr = Vec::new();
        t->cookie = Vec::new();
        t->peer_chain = Vec::new();
        t->cr_context = Vec::new();
        t->rng.init();
        t->acc = (struct T13Acc_*)0;
    }
    return 0;
}

int tls_safec_client_(&TlsStream s, const &char name, const &TlsConfig cfg) {
    if (cfg.max_version < TLS_VERSION_1_3) {
        tls_fail_(&s.err, TLS_ERR_UNAVAILABLE, "the SafeC TLS backend implements TLS 1.3 only (max_version is below it)");
        return TLS_ERR_UNAVAILABLE;
    }
    if (t13_alloc_(s) != 0) { return TLS_ERR_NO_MEMORY; }
    struct T13_* tp = t13_of_(s);
    unsafe {
        &T13_ t = (&T13_)tp;
        t.server = 0;
        t.st = T13_C_START_;
        t.verify = cfg.verify_peer;
        t.verify_host = cfg.verify_hostname;
        t.use_system = cfg.use_system_roots;
        t.name.push(name);
        if (!cfg.alpn.is_empty()) {
            struct Vec<unsigned char> w = tls_alpn_wire_(cfg.alpn.as_ptr());
            t.alpn.free();
            t.alpn = w;
        }
        if (!cfg.ca_pem.is_empty()) { tls_trust_add_pem_(&t.extra, cfg.ca_pem.as_ptr()); }
        if (!cfg.ca_file.is_empty()) {
            struct String pem = String::new();
            if (!tls_read_file_(cfg.ca_file.as_ptr(), &pem)) {
                pem.free();
                tls_fail2_(&s.err, TLS_ERR_CONFIG, "cannot read ca_file", cfg.ca_file.as_ptr());
                return TLS_ERR_CONFIG;
            }
            tls_trust_add_pem_(&t.extra, pem.as_ptr());
            pem.free();
        }
        if (!cfg.cert_pem.is_empty()) {
            tls_pem_decode_(cfg.cert_pem.as_ptr(), "CERTIFICATE", &t.own_chain);
            if (t.own_chain.len == 0UL) {
                tls_fail_(&s.err, TLS_ERR_CONFIG, "cert_pem holds no certificate");
                return TLS_ERR_CONFIG;
            }
            if (tls_key_from_pem_(&t.key, cfg.key_pem.as_ptr(), &s.err) != TLS_OK) { return TLS_ERR_CONFIG; }
            t.has_key = 1;
        }
        t.rng.fill(&t.random[0], 32UL);
        t.rng.fill(&t.sid[0], 32UL);
        t.sid_len = 32UL;
        t.group = T13_X25519_;
    }
    return TLS_OK;
}

int tls_safec_acceptor_(&TlsAcceptor a, const &TlsServerConfig cfg, &TlsError e) {
    if (cfg.max_version < TLS_VERSION_1_3) {
        tls_fail_(e, TLS_ERR_UNAVAILABLE, "the SafeC TLS backend implements TLS 1.3 only");
        return TLS_ERR_UNAVAILABLE;
    }
    void* mem = alloc_zeroed_raw_(sizeof(struct T13Acc_));
    if (mem == (void*)0) { tls_fail_(e, TLS_ERR_NO_MEMORY, "out of memory"); return TLS_ERR_NO_MEMORY; }
    unsafe {
        struct T13Acc_* acc = (struct T13Acc_*)mem;
        acc->refs = 1;
        acc->chain = Vec::new();
        acc->alpn = Vec::new();
        acc->client_ca = tls_trust_new_();
        acc->verify_client = cfg.verify_client;
        tls_pem_decode_(cfg.cert_pem.as_ptr(), "CERTIFICATE", &acc->chain);
        int bad = 0;
        if (acc->chain.len == 0UL) { tls_fail_(e, TLS_ERR_CONFIG, "cert_pem holds no certificate"); bad = 1; }
        else if (tls_key_from_pem_(&acc->key, cfg.key_pem.as_ptr(), e) != TLS_OK) { bad = 1; }
        if (bad) {
            tls_der_list_free_(&acc->chain);
            tls_trust_free_(&acc->client_ca);
            dealloc_raw_(mem);
            return TLS_ERR_CONFIG;
        }
        if (!cfg.alpn.is_empty()) {
            acc->alpn.free();
            acc->alpn = tls_alpn_wire_(cfg.alpn.as_ptr());
        }
        if (cfg.verify_client != 0) {
            if (tls_trust_add_pem_(&acc->client_ca, cfg.client_ca_pem.as_ptr()) == 0UL) {
                tls_fail_(e, TLS_ERR_CONFIG, "verify_client needs client_ca_pem roots");
                tls_der_list_free_(&acc->chain);
                tls_key_destroy_(&acc->key);
                tls_trust_free_(&acc->client_ca);
                dealloc_raw_(mem);
                return TLS_ERR_CONFIG;
            }
        }
    }
    a.impl_ = mem;
    return TLS_OK;
}

static void t13_acc_unref_(struct T13Acc_* acc) {
    unsafe {
        if (atomic_fetch_sub_int(&acc->refs, 1) != 1) { return; }
        tls_der_list_free_(&acc->chain);
        tls_key_destroy_(&acc->key);
        acc->alpn.free();
        tls_trust_free_(&acc->client_ca);
        dealloc_raw_((void*)acc);
    }
}

void tls_safec_acceptor_free_(&TlsAcceptor a) {
    unsafe { t13_acc_unref_((struct T13Acc_*)a.impl_); }
}

int tls_safec_accept_(&TlsStream s, const &TlsAcceptor a) {
    if (t13_alloc_(s) != 0) { return TLS_ERR_NO_MEMORY; }
    unsafe {
        struct T13_* t = t13_of_(s);
        t->server = 1;
        t->st = T13_S_WAIT_CH_;
        t->acc = (struct T13Acc_*)a.impl_;
        atomic_fetch_add_int(&t->acc->refs, 1);
        t->rng.fill(&t->random[0], 32UL);
    }
    return TLS_OK;
}

static void t13_fill_info_(&TlsStream s, &T13_ t) {
    s.version_ = TLS_VERSION_1_3;
    tls_copy_cstr_(&s.cipher_[0], 64UL, tls_suite_name_(t.suite));
    s.peer_der_.free();
    s.peer_der_ = Vec::new();
    if (t.peer_chain.len > 0UL) {
        &Vec<unsigned char> leaf = t.peer_chain.at(0UL);
        s.peer_der_ = leaf.clone();
    }
}

int tls_safec_handshake_(&TlsStream s) {
    struct T13_* tp = t13_of_(s);
    unsafe {
        if (tp == (struct T13_*)0) { return tls_fail_(&s.err, TLS_ERR_STATE, "no TLS state"); }
        &T13_ t = (&T13_)tp;
        while (1) {
            if (t.st == T13_FAILED_) { return -1; }
            if (t.st == T13_C_START_) {
                if (t13_send_client_hello_(s, t, (unsigned int)T13_X25519_) != 0) { return -1; }
                continue;
            }
            int f = t13_flush_(t, s.fd);
            if (f == TLS_WANT_WRITE) { return TLS_WANT_WRITE; }
            if (f != 0) { return t13_fail_(s, TLS_ERR_IO, -1, "TLS: socket write failed"); }
            if (t.st == T13_DONE_) {
                t13_fill_info_(s, t);
                return 0;
            }
            if (t13_drain_hs_(s, t, 0) != 0) { return -1; }
            if (t.st == T13_DONE_ || t.tx.len > t.tx_pos) { continue; }
            unsigned int type = 0U;
            unsigned long poff = 0UL;
            unsigned long plen = 0UL;
            unsigned long total = 0UL;
            int r = t13_next_record_(s, t, &type, &poff, &plen, &total);
            if (r == TLS_WANT_READ) { return TLS_WANT_READ; }
            if (r == 0) {
                tls_fail_(&s.err, TLS_ERR_CLOSED, "the peer closed the connection during the TLS handshake");
                t.st = T13_FAILED_;
                return -1;
            }
            if (r < 0) { return -1; }
            if (type == (unsigned int)T13_CT_HS_) {
                if (plen == 0UL) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "empty handshake record"); }
                if (t.hs.len + plen > (unsigned long)T13_MAX_HS_) { return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "handshake too large"); }
                bput_(&t.hs, (const unsigned char*)&t.rbuf[poff], plen);
            } else if (type == (unsigned int)T13_CT_CCS_) {
                if (plen != 1UL || t.rbuf[poff] != (unsigned char)1) {
                    return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "invalid ChangeCipherSpec");
                }
            } else if (type == (unsigned int)T13_CT_ALERT_) {
                int ar = t13_on_alert_(s, t, poff, plen);
                if (ar == 0) {
                    tls_fail_(&s.err, TLS_ERR_CLOSED, "the peer closed the connection during the TLS handshake");
                    t.st = T13_FAILED_;
                    return -1;
                }
                if (ar < 0) { return -1; }
            } else {
                return t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "unexpected record during the TLS handshake");
            }
            t13_consume_(t, total);
        }
    }
    return -1;
}

long long tls_safec_read_(&TlsStream s, &unsigned char buf, unsigned long n) {
    struct T13_* tp = t13_of_(s);
    unsafe {
        &T13_ t = (&T13_)tp;
        while (1) {
            if (t.app_pos < t.app_len) {
                unsigned long k = t.app_len - t.app_pos;
                if (k > n) { k = n; }
                unsigned char* d = (unsigned char*)buf;
                unsigned long i = 0UL;
                while (i < k) { d[i] = t.app[t.app_pos + i]; i = i + 1UL; }
                t.app_pos = t.app_pos + k;
                return (long long)k;
            }
            if (t.peer_closed) { return 0LL; }
            if (t.st == T13_FAILED_) { return -1LL; }
            // queued output (a KeyUpdate answer); a full socket buffer doesn't stop reading
            int f = t13_flush_(t, s.fd);
            if (f < 0 && f != TLS_WANT_WRITE) { tls_fail_(&s.err, TLS_ERR_IO, "TLS: socket write failed"); return -1LL; }
            unsigned int type = 0U;
            unsigned long poff = 0UL;
            unsigned long plen = 0UL;
            unsigned long total = 0UL;
            int r = t13_next_record_(s, t, &type, &poff, &plen, &total);
            if (r == TLS_WANT_READ) { return (long long)TLS_WANT_READ; }
            if (r == 0) { t.peer_closed = 1; return 0LL; }   // EOF without close_notify
            if (r < 0) { return -1LL; }
            if (type == (unsigned int)T13_CT_APP_) {
                unsigned long i = 0UL;
                while (i < plen) { t.app[i] = t.rbuf[poff + i]; i = i + 1UL; }
                t.app_pos = 0UL;
                t.app_len = plen;
            } else if (type == (unsigned int)T13_CT_HS_) {
                if (t.hs.len + plen > (unsigned long)T13_MAX_HS_) { t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_DECODE_, "handshake too large"); return -1LL; }
                bput_(&t.hs, (const unsigned char*)&t.rbuf[poff], plen);
                t13_consume_(t, total);
                if (t13_drain_hs_(s, t, 1) != 0) { return -1LL; }
                continue;
            } else if (type == (unsigned int)T13_CT_ALERT_) {
                int ar = t13_on_alert_(s, t, poff, plen);
                t13_consume_(t, total);
                if (ar == 0) { return 0LL; }
                if (ar < 0) { return -1LL; }
                continue;
            } else if (type == (unsigned int)T13_CT_CCS_) {
                t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "ChangeCipherSpec after the handshake");
                return -1LL;
            } else {
                t13_fail_(s, TLS_ERR_PROTOCOL, T13_A_UNEXPECTED_, "unexpected TLS record type");
                return -1LL;
            }
            t13_consume_(t, total);
        }
    }
    return -1LL;
}

long long tls_safec_write_(&TlsStream s, const &unsigned char buf, unsigned long n) {
    struct T13_* tp = t13_of_(s);
    unsafe {
        &T13_ t = (&T13_)tp;
        if (t.st == T13_FAILED_) { return -1LL; }
        // Earlier records still queued: send them first.
        int f = t13_flush_(t, s.fd);
        if (f == TLS_WANT_WRITE) { return (long long)TLS_WANT_WRITE; }
        if (f != 0) { tls_fail_(&s.err, TLS_ERR_IO, "TLS: socket write failed"); return -1LL; }
        unsigned long k = n > (unsigned long)T13_MAX_PLAIN_ ? (unsigned long)T13_MAX_PLAIN_ : n;
        t13_queue_record_(t, (unsigned int)T13_CT_APP_, (const unsigned char*)buf, k, 0);
        f = t13_flush_(t, s.fd);
        if (f < 0 && f != TLS_WANT_WRITE) { tls_fail_(&s.err, TLS_ERR_IO, "TLS: socket write failed"); return -1LL; }
        // Accepted (queued) even if the socket is full: the next call flushes.
        return (long long)k;
    }
    return -1LL;
}

int tls_safec_shutdown_(&TlsStream s) {
    struct T13_* tp = t13_of_(s);
    unsafe {
        if (tp == (struct T13_*)0) { return -1; }
        &T13_ t = (&T13_)tp;
        if (!t.sent_close && t.st == T13_DONE_) {
            t13_queue_alert_(t, 1, T13_A_CLOSE_NOTIFY_);
            t.sent_close = 1;
        }
        return t13_flush_(t, s.fd) == 0 ? 0 : -1;
    }
}

void tls_safec_free_(&TlsStream s) {
    struct T13_* tp = t13_of_(s);
    unsafe {
        if (tp == (struct T13_*)0) { return; }
        &T13_ t = (&T13_)tp;
        t.name.free();
        t.alpn.free();
        tls_trust_free_(&t.extra);
        tls_der_list_free_(&t.own_chain);
        if (t.has_key) { tls_key_destroy_(&t.key); }
        t.tx.free();
        t.hs.free();
        t.tr.free();
        t.cookie.free();
        tls_der_list_free_(&t.peer_chain);
        t.cr_context.free();
        if (t.acc != (struct T13Acc_*)0) { t13_acc_unref_(t.acc); }
        mem_zero_secure((void*)tp, sizeof(struct T13_));
        dealloc_raw_((void*)tp);
    }
    s.impl_ = (void*)0;
}

} // namespace std
