// SafeC Standard Library — TLS: the SChannel backend (TLS_IMPL_SCHANNEL),
// Windows' own TLS through SSPI (secur32: AcquireCredentialsHandle,
// InitializeSecurityContext, EncryptMessage, DecryptMessage).
//
// Credentials: SCH_CREDENTIALS (Windows 10 1809+, which can negotiate TLS
// 1.3) with SCHANNEL_CRED as the fallback on older systems. Certificates are
// validated here, not by SChannel's automatic check, so every backend
// reports the same TLS_ERR_CERT_* codes: the peer's chain through
// CertGetCertificateChain + CertVerifyCertificateChainPolicy(SSL, with the
// server name) against the Windows certificate store — or, when TlsConfig
// adds its own roots (ca_pem / ca_file) or turns use_system_roots off,
// through std's path validation (tls_trust.sc) with those roots.
//
// Not covered by this backend (TLS_ERR_UNAVAILABLE, so TLS_BACKEND_AUTO
// moves to the SafeC backend): servers and client certificates — SChannel
// needs their private key in a Windows key store, not PEM in memory.
// safeguard: link windows -lsecur32 -lcrypt32 -lws2_32
#pragma once
#include <std/stddef.h>
#include <std/tls/tls_backend.h>
#include <std/tls/tls_safec.h>
#include <std/mem.h>
#include <std/str.h>

#define SCH_UNISP_                  "Microsoft Unified Security Protocol Provider"
#define SCH_CRED_OUTBOUND_          2U
#define SCHANNEL_CRED_VERSION_      4U
#define SCH_CREDENTIALS_VERSION_    5U
#define SCH_CRED_MANUAL_CRED_VALIDATION_ 0x00000008U
#define SCH_CRED_NO_DEFAULT_CREDS_  0x00000010U
#define SCH_USE_STRONG_CRYPTO_      0x00400000U
#define SP_PROT_TLS1_0_CLIENT_      0x00000080U
#define SP_PROT_TLS1_1_CLIENT_      0x00000200U
#define SP_PROT_TLS1_2_CLIENT_      0x00000800U
#define SP_PROT_TLS1_3_CLIENT_      0x00002000U
#define SP_PROT_SSL3_CLIENT_        0x00000020U
#define ISC_FLAGS_  (0x4U | 0x8U | 0x10U | 0x100U | 0x4000U | 0x8000U | 0x80000U)
#define SEC_E_OK_                   0
#define SEC_I_CONTINUE_NEEDED_      0x00090312
#define SEC_I_CONTEXT_EXPIRED_      0x00090317
#define SEC_I_INCOMPLETE_CREDS_     0x00090320
#define SEC_I_RENEGOTIATE_          0x00090321
#define SEC_E_INCOMPLETE_MESSAGE_   ((int)0x80090318U)
#define SECBUFFER_EMPTY_            0U
#define SECBUFFER_DATA_             1U
#define SECBUFFER_TOKEN_            2U
#define SECBUFFER_EXTRA_            5U
#define SECBUFFER_STREAM_TRAILER_   6U
#define SECBUFFER_STREAM_HEADER_    7U
#define SECBUFFER_ALERT_            17U
#define SECBUFFER_APP_PROTOCOLS_    18U
#define SECPKG_ATTR_STREAM_SIZES_   4U
#define SECPKG_ATTR_APP_PROTOCOL_   35U
#define SECPKG_ATTR_REMOTE_CERT_    0x53U
#define SECPKG_ATTR_CONNECTION_INFO_ 0x5aU
#define SECPKG_ATTR_CIPHER_INFO_    0x64U
#define SCH_BUF_                    (16384 + 2048 + 512)

namespace std {

struct SchHandle_ { unsigned long long lower; unsigned long long upper; };
struct SchBuffer_ { unsigned int len; unsigned int type; void* data; };
struct SchBufferDesc_ { unsigned int version; unsigned int count; struct SchBuffer_* bufs; };

struct SchannelCred_ {
    unsigned int dwVersion;
    unsigned int cCreds;
    void*        paCred;
    void*        hRootStore;
    unsigned int cMappers;
    void*        aphMappers;
    unsigned int cSupportedAlgs;
    void*        palgSupportedAlgs;
    unsigned int grbitEnabledProtocols;
    unsigned int dwMinimumCipherStrength;
    unsigned int dwMaximumCipherStrength;
    unsigned int dwSessionLifespan;
    unsigned int dwFlags;
    unsigned int dwCredFormat;
};

struct SchTlsParameters_ {
    unsigned int cAlpnIds;
    void*        rgstrAlpnIds;
    unsigned int grbitDisabledProtocols;
    unsigned int cDisabledCrypto;
    void*        pDisabledCrypto;
    unsigned int dwFlags;
};

struct SchCredentials_ {
    unsigned int dwVersion;
    unsigned int dwCredFormat;
    unsigned int cCreds;
    void*        paCred;
    void*        hRootStore;
    unsigned int cMappers;
    void*        aphMappers;
    unsigned int dwSessionLifespan;
    unsigned int dwFlags;
    unsigned int cTlsParameters;
    struct SchTlsParameters_* pTlsParameters;
};

struct SchStreamSizes_ {
    unsigned int cbHeader;
    unsigned int cbTrailer;
    unsigned int cbMaximumMessage;
    unsigned int cBuffers;
    unsigned int cbBlockSize;
};

struct SchConnInfo_ {
    unsigned int dwProtocol;
    unsigned int aiCipher;
    unsigned int dwCipherStrength;
    unsigned int aiHash;
    unsigned int dwHashStrength;
    unsigned int aiExch;
    unsigned int dwExchStrength;
};

struct SchCipherInfo_ {
    unsigned int dwVersion;
    unsigned int dwProtocol;
    unsigned int dwCipherSuite;
    unsigned int dwBaseCipherSuite;
    unsigned short rest[64 + 64 + 64 + 64 + 64 + 16];
};

struct SchAppProtocol_ {
    unsigned int  status;      // 1: SecApplicationProtocolNegotiationStatus_Success
    unsigned int  ext;
    unsigned char size;
    unsigned char id[255];
};

// CERT_CONTEXT / chain verification structures (wincrypt.h).
struct SchCertCtx_ {
    unsigned int   encoding;
    unsigned char* data;
    unsigned int   len;
    void*          info;
    void*          store;
};
struct SchEnhKeyUsage_ { unsigned int count; char** ids; };
struct SchUsageMatch_ { unsigned int type; struct SchEnhKeyUsage_ usage; };
struct SchChainPara_ { unsigned int cbSize; struct SchUsageMatch_ usage; };
struct SchSslPolicyPara_ { unsigned int cbSize; unsigned int authType; unsigned int checks; unsigned short* serverName; };
struct SchPolicyPara_ { unsigned int cbSize; unsigned int flags; void* extra; };
struct SchPolicyStatus_ { unsigned int cbSize; unsigned int error; int chainIndex; int elementIndex; void* extra; };

extern int  AcquireCredentialsHandleA(const char* principal, const char* package, unsigned int use, void* logon,
                                      void* authData, void* getKey, void* getKeyArg, struct SchHandle_* cred, long long* expiry);
extern int  InitializeSecurityContextA(struct SchHandle_* cred, struct SchHandle_* ctx, const char* target, unsigned int req,
                                       unsigned int reserved, unsigned int rep, struct SchBufferDesc_* in, unsigned int reserved2,
                                       struct SchHandle_* newCtx, struct SchBufferDesc_* out, unsigned int* attrs, long long* expiry);
extern int  DeleteSecurityContext(struct SchHandle_* ctx);
extern int  FreeCredentialsHandle(struct SchHandle_* cred);
extern int  FreeContextBuffer(void* p);
extern int  QueryContextAttributesA(struct SchHandle_* ctx, unsigned int attr, void* buf);
extern int  EncryptMessage(struct SchHandle_* ctx, unsigned int qop, struct SchBufferDesc_* msg, unsigned int seq);
extern int  DecryptMessage(struct SchHandle_* ctx, struct SchBufferDesc_* msg, unsigned int seq, unsigned int* qop);
extern int  ApplyControlToken(struct SchHandle_* ctx, struct SchBufferDesc_* in);
extern int  CertGetCertificateChain(void* engine, void* cert, void* time, void* store, struct SchChainPara_* para,
                                    unsigned int flags, void* reserved, void** chain);
extern int  CertVerifyCertificateChainPolicy(const char* policy, void* chain, struct SchPolicyPara_* para,
                                             struct SchPolicyStatus_* status);
extern void CertFreeCertificateChain(void* chain);
extern int  CertFreeCertificateContext(void* cert);
extern void* CertEnumCertificatesInStore(void* store, void* prev);

struct SchConn_ {
    struct SchHandle_ cred;
    struct SchHandle_ ctx;
    int    have_cred;
    int    have_ctx;
    int    done;            // handshake finished (verification still to do)
    int    verified;
    int    need_read;
    int    peer_closed;
    int    sent_close;
    int    verify;
    int    verify_host;
    int    use_system;
    struct String name;
    struct TlsTrust_ extra;
    struct Vec<unsigned char> alpn;
    struct SchStreamSizes_ sizes;
    unsigned char in[SCH_BUF_];
    unsigned long in_len;
    unsigned char out[SCH_BUF_];
    unsigned long out_pos;
    unsigned long out_len;
    unsigned char plain[SCH_BUF_];
    unsigned long plain_pos;
    unsigned long plain_len;
};

int tls_schannel_available_() { return 1; }

static struct SchConn_* sch_of_(&TlsStream s) {
    unsafe { return (struct SchConn_*)s.impl_; }
}

static int sch_fail_(&TlsStream s, int code, const &char what, int status) {
    struct String m = String::from(what);
    m.push(" (SECURITY_STATUS 0x");
    unsafe {
        unsigned int v = (unsigned int)status;
        const &char hex = "0123456789abcdef";
        int i = 7;
        while (i >= 0) { m.push_char(hex[(v >> (unsigned int)(4 * i)) & 15U]); i = i - 1; }
    }
    m.push(")");
    tls_fail_(&s.err, code, m.as_ptr());
    m.free();
    s.err.detail = (long long)status;
    return -1;
}

static int sch_map_status_(int st) {
    unsigned int u = (unsigned int)st;
    if (u == 0x80090322U) { return TLS_ERR_CERT_HOSTNAME; }    // SEC_E_WRONG_PRINCIPAL
    if (u == 0x80090325U) { return TLS_ERR_CERT_UNTRUSTED; }   // SEC_E_UNTRUSTED_ROOT
    if (u == 0x80090328U) { return TLS_ERR_CERT_EXPIRED; }     // SEC_E_CERT_EXPIRED
    if (u == 0x80090327U) { return TLS_ERR_CERT_INVALID; }     // SEC_E_CERT_UNKNOWN
    if (u == 0x80090331U || u == 0x80090326U) { return TLS_ERR_HANDSHAKE; }   // ALGORITHM_MISMATCH, ILLEGAL_MESSAGE
    if (u == 0x80090308U || u == 0x80090304U) { return TLS_ERR_PROTOCOL; }    // INVALID_TOKEN, INTERNAL_ERROR
    return TLS_ERR_HANDSHAKE;
}

// Sends pending output: 0 when all is out, TLS_WANT_WRITE, -1.
static int sch_flush_(&TlsStream s, struct SchConn_* c) {
    unsafe {
        while (c->out_pos < c->out_len) {
            long long n = tls_sock_send_(s.fd, (const &unsigned char)((unsigned char*)&c->out[0] + c->out_pos),
                                         c->out_len - c->out_pos);
            if (n == (long long)TLS_WANT_WRITE) { return TLS_WANT_WRITE; }
            if (n <= 0LL) { return -1; }
            c->out_pos = c->out_pos + (unsigned long)n;
        }
        c->out_pos = 0UL;
        c->out_len = 0UL;
    }
    return 0;
}

static void sch_queue_(struct SchConn_* c, const unsigned char* p, unsigned long n) {
    unsafe {
        if (c->out_len + n > (unsigned long)SCH_BUF_) { return; }
        unsigned long i = 0UL;
        while (i < n) { c->out[c->out_len + i] = p[i]; i = i + 1UL; }
        c->out_len = c->out_len + n;
    }
}

// Reads more socket bytes into c->in: 1, 0 at EOF, TLS_WANT_READ, -1.
static int sch_fill_(&TlsStream s, struct SchConn_* c) {
    unsafe {
        if (c->in_len >= (unsigned long)SCH_BUF_) { return -1; }
        long long n = tls_sock_recv_(s.fd, (&unsigned char)((unsigned char*)&c->in[0] + c->in_len),
                                     (unsigned long)SCH_BUF_ - c->in_len);
        if (n == (long long)TLS_WANT_READ) { return TLS_WANT_READ; }
        if (n < 0LL) { return -1; }
        if (n == 0LL) { return 0; }
        c->in_len = c->in_len + (unsigned long)n;
    }
    return 1;
}

static void sch_keep_extra_(struct SchConn_* c, const &SchBuffer_ b) {
    unsafe {
        if (b.type == SECBUFFER_EXTRA_ && b.len > 0U && (unsigned long)b.len <= c->in_len) {
            unsigned long off = c->in_len - (unsigned long)b.len;
            unsigned long i = 0UL;
            while (i < (unsigned long)b.len) { c->in[i] = c->in[off + i]; i = i + 1UL; }
            c->in_len = (unsigned long)b.len;
        } else {
            c->in_len = 0UL;
        }
    }
}

static int sch_acquire_(&TlsStream s, struct SchConn_* c, int min_version, int max_version) {
    unsigned int disabled = SP_PROT_SSL3_CLIENT_ | SP_PROT_TLS1_0_CLIENT_ | SP_PROT_TLS1_1_CLIENT_;
    if (min_version > TLS_VERSION_1_2) { disabled = disabled | SP_PROT_TLS1_2_CLIENT_; }
    if (max_version < TLS_VERSION_1_3) { disabled = disabled | SP_PROT_TLS1_3_CLIENT_; }
    unsigned int flags = SCH_CRED_MANUAL_CRED_VALIDATION_ | SCH_CRED_NO_DEFAULT_CREDS_ | SCH_USE_STRONG_CRYPTO_;
    long long expiry = 0LL;
    int st = 0;
    unsafe {
        struct SchTlsParameters_ tp;
        safe_memset_raw_((void*)&tp, 0, sizeof(struct SchTlsParameters_));
        tp.grbitDisabledProtocols = disabled;
        struct SchCredentials_ sc;
        safe_memset_raw_((void*)&sc, 0, sizeof(struct SchCredentials_));
        sc.dwVersion = SCH_CREDENTIALS_VERSION_;
        sc.dwFlags = flags;
        sc.cTlsParameters = 1U;
        sc.pTlsParameters = (struct SchTlsParameters_*)&tp;
        st = AcquireCredentialsHandleA((const char*)0, SCH_UNISP_, SCH_CRED_OUTBOUND_, (void*)0, (void*)&sc,
                                       (void*)0, (void*)0, &c->cred, &expiry);
        if (st != SEC_E_OK_) {
            // Older Windows (and Wine): SCHANNEL_CRED, at most TLS 1.2.
            if (min_version > TLS_VERSION_1_2) {
                return sch_fail_(s, TLS_ERR_UNAVAILABLE, "SChannel here can't do TLS 1.3", st);
            }
            struct SchannelCred_ old;
            safe_memset_raw_((void*)&old, 0, sizeof(struct SchannelCred_));
            old.dwVersion = SCHANNEL_CRED_VERSION_;
            old.grbitEnabledProtocols = SP_PROT_TLS1_2_CLIENT_;
            old.dwFlags = flags;
            st = AcquireCredentialsHandleA((const char*)0, SCH_UNISP_, SCH_CRED_OUTBOUND_, (void*)0, (void*)&old,
                                           (void*)0, (void*)0, &c->cred, &expiry);
        }
    }
    if (st != SEC_E_OK_) { return sch_fail_(s, TLS_ERR_UNAVAILABLE, "AcquireCredentialsHandle failed", st); }
    unsafe { c->have_cred = 1; }
    return 0;
}

int tls_schannel_client_(&TlsStream s, const &char name, const &TlsConfig cfg) {
    if (!cfg.cert_pem.is_empty()) {
        tls_fail_(&s.err, TLS_ERR_UNAVAILABLE, "client certificates from PEM aren't supported by the SChannel backend");
        return TLS_ERR_UNAVAILABLE;
    }
    void* mem = alloc_zeroed_raw_(sizeof(struct SchConn_));
    if (mem == (void*)0) { tls_fail_(&s.err, TLS_ERR_NO_MEMORY, "out of memory"); return TLS_ERR_NO_MEMORY; }
    unsafe { s.impl_ = (?&heap TlsImpl_)mem; }
    struct SchConn_* c = sch_of_(s);
    unsafe {
        c->name = String::from(name);
        c->extra = tls_trust_new_();
        c->alpn = Vec::new();
        c->verify = cfg.verify_peer;
        c->verify_host = cfg.verify_hostname;
        c->use_system = cfg.use_system_roots;
        if (!cfg.alpn.is_empty()) {
            c->alpn.free();
            c->alpn = tls_alpn_wire_(cfg.alpn.as_ptr());
        }
        if (!cfg.ca_pem.is_empty()) { tls_trust_add_pem_(&c->extra, cfg.ca_pem.as_ptr()); }
        if (!cfg.ca_file.is_empty()) {
            struct String pem = String::new();
            if (!tls_read_file_(cfg.ca_file.as_ptr(), &pem)) {
                pem.free();
                tls_fail2_(&s.err, TLS_ERR_CONFIG, "cannot read ca_file", cfg.ca_file.as_ptr());
                return TLS_ERR_CONFIG;
            }
            tls_trust_add_pem_(&c->extra, pem.as_ptr());
            pem.free();
        }
    }
    if (sch_acquire_(s, c, cfg.min_version, cfg.max_version) != 0) { return s.err.code; }
    return TLS_OK;
}

// Peer certificate checks once the handshake is done.
static int sch_verify_(&TlsStream s, struct SchConn_* c) {
    void* cert = (void*)0;
    int st = 0;
    unsafe { st = QueryContextAttributesA(&c->ctx, SECPKG_ATTR_REMOTE_CERT_, (void*)&cert); }
    if (st != SEC_E_OK_ || cert == (void*)0) { return sch_fail_(s, TLS_ERR_NO_CERT, "no server certificate", st); }
    int code = TLS_OK;
    unsafe {
        struct SchCertCtx_* cc = (struct SchCertCtx_*)cert;
        s.peer_der_.free();
        s.peer_der_ = Vec::from_slice(((const unsigned char*)cc->data)[0..(unsigned long)cc->len]);
        if (c->verify) {
            if (c->extra.certs.len > 0UL || !c->use_system) {
                // std's own path validation with the configured roots
                struct Vec<struct Vec<unsigned char>> chain = Vec::new();
                chain.push(Vec::from_slice(((const unsigned char*)cc->data)[0..(unsigned long)cc->len]));
                void* it = CertEnumCertificatesInStore(cc->store, (void*)0);
                while (it != (void*)0) {
                    struct SchCertCtx_* ic = (struct SchCertCtx_*)it;
                    if (ic->len != cc->len || safe_memcmp_raw_((const void*)ic->data, (const void*)cc->data, (unsigned long)cc->len) != 0) {
                        chain.push(Vec::from_slice(((const unsigned char*)ic->data)[0..(unsigned long)ic->len]));
                    }
                    it = CertEnumCertificatesInStore(cc->store, it);
                }
                ?&TlsTrust_ sys = null;
                if (c->use_system) { sys = tls_trust_system_(); }
                const ?&char host = null;
                if (c->verify_host) { host = c->name.as_ptr(); }
                code = tls_verify_chain_(&chain, (?&TlsTrust_)&c->extra, sys, host, 1, tls_now_(), &s.err);
                tls_der_list_free_(&chain);
            } else {
                char* usage[1];
                usage[0] = "1.3.6.1.5.5.7.3.1";   // serverAuth
                struct SchChainPara_ para;
                safe_memset_raw_((void*)&para, 0, sizeof(struct SchChainPara_));
                para.cbSize = (unsigned int)sizeof(struct SchChainPara_);
                para.usage.type = 0U;   // USAGE_MATCH_TYPE_AND
                para.usage.usage.count = 1U;
                para.usage.usage.ids = (char**)&usage[0];
                void* chainCtx = (void*)0;
                if (!CertGetCertificateChain((void*)0, cert, (void*)0, cc->store, &para, 0U, (void*)0, &chainCtx)) {
                    code = TLS_ERR_CERT_INVALID;
                    tls_fail_(&s.err, code, "CertGetCertificateChain failed");
                } else {
                    unsigned short wname[256];
                    unsigned long i = 0UL;
                    const char* nm = (const char*)c->name.as_ptr();
                    while (nm[i] != (char)0 && i < 255UL) { wname[i] = (unsigned short)(unsigned char)nm[i]; i = i + 1UL; }
                    wname[i] = (unsigned short)0;
                    struct SchSslPolicyPara_ ssl;
                    ssl.cbSize = (unsigned int)sizeof(struct SchSslPolicyPara_);
                    ssl.authType = 2U;      // AUTHTYPE_SERVER
                    ssl.checks = c->verify_host ? 0U : 0x00001000U;   // SECURITY_FLAG_IGNORE_CERT_CN_INVALID
                    ssl.serverName = (unsigned short*)&wname[0];
                    struct SchPolicyPara_ pp;
                    pp.cbSize = (unsigned int)sizeof(struct SchPolicyPara_);
                    pp.flags = 0U;
                    pp.extra = (void*)&ssl;
                    struct SchPolicyStatus_ ps;
                    safe_memset_raw_((void*)&ps, 0, sizeof(struct SchPolicyStatus_));
                    ps.cbSize = (unsigned int)sizeof(struct SchPolicyStatus_);
                    if (!CertVerifyCertificateChainPolicy((const char*)4, chainCtx, &pp, &ps)) {
                        code = TLS_ERR_CERT_INVALID;
                        tls_fail_(&s.err, code, "CertVerifyCertificateChainPolicy failed");
                    } else if (ps.error != 0U) {
                        unsigned int e = ps.error;
                        const &char why = "certificate verification failed";
                        code = TLS_ERR_CERT_INVALID;
                        if (e == 0x800B010FU) { code = TLS_ERR_CERT_HOSTNAME; why = "the certificate doesn't match the host name"; }
                        else if (e == 0x800B0101U) { code = TLS_ERR_CERT_EXPIRED; why = "the certificate has expired or is not yet valid"; }
                        else if (e == 0x800B0109U || e == 0x800B010AU) { code = TLS_ERR_CERT_UNTRUSTED; why = "the certificate chain doesn't lead to a trusted root"; }
                        else if (e == 0x80092010U) { code = TLS_ERR_CERT_REVOKED; why = "the certificate has been revoked"; }
                        sch_fail_(s, code, why, (int)e);
                    }
                    CertFreeCertificateChain(chainCtx);
                }
            }
        }
        CertFreeCertificateContext(cert);
    }
    if (code != TLS_OK) { return -1; }
    return 0;
}

static void sch_fill_info_(&TlsStream s, struct SchConn_* c) {
    unsafe {
        struct SchConnInfo_ ci;
        safe_memset_raw_((void*)&ci, 0, sizeof(struct SchConnInfo_));
        if (QueryContextAttributesA(&c->ctx, SECPKG_ATTR_CONNECTION_INFO_, (void*)&ci) == SEC_E_OK_) {
            if ((ci.dwProtocol & 0x3000U) != 0U) { s.version_ = TLS_VERSION_1_3; }
            else if ((ci.dwProtocol & 0x0C00U) != 0U) { s.version_ = TLS_VERSION_1_2; }
        }
        struct SchCipherInfo_ cph;
        safe_memset_raw_((void*)&cph, 0, sizeof(struct SchCipherInfo_));
        cph.dwVersion = 1U;
        if (QueryContextAttributesA(&c->ctx, SECPKG_ATTR_CIPHER_INFO_, (void*)&cph) == SEC_E_OK_) {
            tls_copy_cstr_(&s.cipher_[0], 64UL, tls_suite_name_((int)cph.dwCipherSuite));
        }
        struct SchAppProtocol_ ap;
        safe_memset_raw_((void*)&ap, 0, sizeof(struct SchAppProtocol_));
        if (c->alpn.len > 0UL && QueryContextAttributesA(&c->ctx, SECPKG_ATTR_APP_PROTOCOL_, (void*)&ap) == SEC_E_OK_ &&
            ap.status == 1U) {
            unsigned long i = 0UL;
            while (i < (unsigned long)ap.size && i < 63UL) { s.alpn_[i] = (char)ap.id[i]; i = i + 1UL; }
            s.alpn_[i] = (char)0;
        }
        QueryContextAttributesA(&c->ctx, SECPKG_ATTR_STREAM_SIZES_, (void*)&c->sizes);
    }
}

int tls_schannel_handshake_(&TlsStream s) {
    struct SchConn_* c = sch_of_(s);
    unsafe {
        while (1) {
            int f = sch_flush_(s, c);
            if (f == TLS_WANT_WRITE) { return TLS_WANT_WRITE; }
            if (f < 0) { return tls_fail_(&s.err, TLS_ERR_IO, "TLS: socket write failed"); }
            if (c->done) {
                if (!c->verified) {
                    c->verified = 1;
                    sch_fill_info_(s, c);
                    if (sch_verify_(s, c) != 0) { return -1; }
                }
                return 0;
            }
            if (c->need_read) {
                int r = sch_fill_(s, c);
                if (r == TLS_WANT_READ) { return TLS_WANT_READ; }
                if (r == 0) { return tls_fail_(&s.err, TLS_ERR_CLOSED, "the peer closed the connection during the TLS handshake"); }
                if (r < 0) { return tls_fail_(&s.err, TLS_ERR_IO, "TLS: socket read failed"); }
                c->need_read = 0;
            }
            struct SchBuffer_ inb[2];
            struct SchBuffer_ outb[3];
            struct SchBufferDesc_ ind;
            struct SchBufferDesc_ outd;
            unsigned char alpnBuf[512];
            int first = !c->have_ctx;
            ind.version = 0U;
            ind.bufs = (struct SchBuffer_*)&inb[0];
            if (first) {
                ind.count = 0U;
                if (c->alpn.len > 0UL && c->alpn.len < 500UL) {
                    // SEC_APPLICATION_PROTOCOLS { ULONG listsSize; { ULONG ext (2: ALPN); USHORT size; bytes } }
                    unsigned long wl = c->alpn.len;
                    unsigned int lists = (unsigned int)(4UL + 2UL + wl);
                    alpnBuf[0] = (unsigned char)(lists & 0xFFU); alpnBuf[1] = (unsigned char)(lists >> 8);
                    alpnBuf[2] = (unsigned char)0; alpnBuf[3] = (unsigned char)0;
                    alpnBuf[4] = (unsigned char)2; alpnBuf[5] = (unsigned char)0; alpnBuf[6] = (unsigned char)0; alpnBuf[7] = (unsigned char)0;
                    alpnBuf[8] = (unsigned char)(wl & 0xFFUL); alpnBuf[9] = (unsigned char)(wl >> 8);
                    unsigned long i = 0UL;
                    unsigned char* w = (unsigned char*)c->alpn.data;
                    while (i < wl) { alpnBuf[10UL + i] = w[i]; i = i + 1UL; }
                    inb[0].type = SECBUFFER_APP_PROTOCOLS_;
                    inb[0].len = (unsigned int)(10UL + wl);
                    inb[0].data = (void*)&alpnBuf[0];
                    ind.count = 1U;
                }
            } else {
                inb[0].type = SECBUFFER_TOKEN_;
                inb[0].len = (unsigned int)c->in_len;
                inb[0].data = (void*)&c->in[0];
                inb[1].type = SECBUFFER_EMPTY_;
                inb[1].len = 0U;
                inb[1].data = (void*)0;
                ind.count = 2U;
            }
            outb[0].type = SECBUFFER_TOKEN_; outb[0].len = 0U; outb[0].data = (void*)0;
            outb[1].type = SECBUFFER_ALERT_; outb[1].len = 0U; outb[1].data = (void*)0;
            outb[2].type = SECBUFFER_EMPTY_; outb[2].len = 0U; outb[2].data = (void*)0;
            outd.version = 0U;
            outd.count = 3U;
            outd.bufs = (struct SchBuffer_*)&outb[0];
            unsigned int attrs = 0U;
            long long expiry = 0LL;
            const char* target = (const char*)c->name.as_ptr();
            if (tls_name_is_ip_(c->name.as_ptr())) { target = (const char*)0; }   // no SNI for IP literals
            int st = InitializeSecurityContextA(&c->cred, first ? (struct SchHandle_*)0 : (struct SchHandle_*)&c->ctx, target, ISC_FLAGS_, 0U, 0U,
                                                ind.count > 0U ? (struct SchBufferDesc_*)&ind : (struct SchBufferDesc_*)0, 0U, &c->ctx, &outd, &attrs, &expiry);
            if (first) { c->have_ctx = 1; }
            if (outb[0].len > 0U && outb[0].data != (void*)0) {
                sch_queue_(c, (const unsigned char*)outb[0].data, (unsigned long)outb[0].len);
                FreeContextBuffer(outb[0].data);
            }
            if (outb[1].data != (void*)0) { FreeContextBuffer(outb[1].data); }
            if (st == SEC_E_INCOMPLETE_MESSAGE_) { c->need_read = 1; continue; }
            if (!first) { sch_keep_extra_(c, &inb[1]); }
            if (st == SEC_E_OK_) { c->done = 1; continue; }
            if (st == SEC_I_CONTINUE_NEEDED_) {
                if (c->in_len == 0UL) { c->need_read = 1; }
                continue;
            }
            if (st == SEC_I_INCOMPLETE_CREDS_) { continue; }   // a client certificate was asked for: go on without
            sch_flush_(s, c);   // the alert, if SChannel made one
            return sch_fail_(s, sch_map_status_(st), "SChannel handshake failed", st);
        }
    }
    return -1;
}

long long tls_schannel_read_(&TlsStream s, &unsigned char buf, unsigned long n) {
    struct SchConn_* c = sch_of_(s);
    unsafe {
        while (1) {
            if (c->plain_pos < c->plain_len) {
                unsigned long k = c->plain_len - c->plain_pos;
                if (k > n) { k = n; }
                safe_memcpy_raw_((void*)buf, (const void*)((unsigned char*)&c->plain[0] + c->plain_pos), k);
                c->plain_pos = c->plain_pos + k;
                return (long long)k;
            }
            if (c->peer_closed) { return 0LL; }
            if (c->in_len > 0UL) {
                struct SchBuffer_ b[4];
                b[0].type = SECBUFFER_DATA_; b[0].len = (unsigned int)c->in_len; b[0].data = (void*)&c->in[0];
                b[1].type = SECBUFFER_EMPTY_; b[1].len = 0U; b[1].data = (void*)0;
                b[2].type = SECBUFFER_EMPTY_; b[2].len = 0U; b[2].data = (void*)0;
                b[3].type = SECBUFFER_EMPTY_; b[3].len = 0U; b[3].data = (void*)0;
                struct SchBufferDesc_ d;
                d.version = 0U; d.count = 4U; d.bufs = (struct SchBuffer_*)&b[0];
                int st = DecryptMessage(&c->ctx, &d, 0U, (unsigned int*)0);
                if (st != SEC_E_INCOMPLETE_MESSAGE_) {
                    // plaintext out, leftover kept
                    int i = 0;
                    c->plain_pos = 0UL;
                    c->plain_len = 0UL;
                    struct SchBuffer_ extra;
                    extra.type = SECBUFFER_EMPTY_; extra.len = 0U; extra.data = (void*)0;
                    while (i < 4) {
                        if (b[i].type == SECBUFFER_DATA_ && b[i].len > 0U && st == SEC_E_OK_) {
                            safe_memcpy_raw_((void*)&c->plain[0], (const void*)b[i].data, (unsigned long)b[i].len);
                            c->plain_len = (unsigned long)b[i].len;
                        }
                        if (b[i].type == SECBUFFER_EXTRA_) { extra = b[i]; }
                        i = i + 1;
                    }
                    if (st == SEC_E_OK_) { sch_keep_extra_(c, &extra); continue; }
                    if (st == SEC_I_CONTEXT_EXPIRED_) { c->peer_closed = 1; return 0LL; }
                    if (st == SEC_I_RENEGOTIATE_) {
                        // TLS 1.3 post-handshake messages: InitializeSecurityContext
                        // takes them (in the EXTRA buffer) and may answer.
                        sch_keep_extra_(c, &extra);
                        c->done = 0;
                        c->verified = 1;
                        int h = tls_schannel_handshake_(s);
                        if (h < 0 && h != TLS_WANT_READ && h != TLS_WANT_WRITE) { return -1LL; }
                        c->done = 1;
                        continue;
                    }
                    sch_fail_(s, TLS_ERR_PROTOCOL, "DecryptMessage failed", st);
                    return -1LL;
                }
            }
            int r = sch_fill_(s, c);
            if (r == TLS_WANT_READ) { return (long long)TLS_WANT_READ; }
            if (r == 0) { c->peer_closed = 1; return 0LL; }
            if (r < 0) { tls_fail_(&s.err, TLS_ERR_IO, "TLS: socket read failed"); return -1LL; }
        }
    }
    return -1LL;
}

long long tls_schannel_write_(&TlsStream s, const &unsigned char buf, unsigned long n) {
    struct SchConn_* c = sch_of_(s);
    unsafe {
        int f = sch_flush_(s, c);
        if (f == TLS_WANT_WRITE) { return (long long)TLS_WANT_WRITE; }
        if (f < 0) { tls_fail_(&s.err, TLS_ERR_IO, "TLS: socket write failed"); return -1LL; }
        unsigned long hdr = (unsigned long)c->sizes.cbHeader;
        unsigned long trl = (unsigned long)c->sizes.cbTrailer;
        unsigned long max = (unsigned long)c->sizes.cbMaximumMessage;
        if (max == 0UL || hdr + max + trl > (unsigned long)SCH_BUF_) { max = (unsigned long)SCH_BUF_ - hdr - trl; }
        unsigned long k = n < max ? n : max;
        safe_memcpy_raw_((void*)((unsigned char*)&c->out[0] + hdr), (const void*)buf, k);
        struct SchBuffer_ b[4];
        b[0].type = SECBUFFER_STREAM_HEADER_; b[0].len = (unsigned int)hdr; b[0].data = (void*)&c->out[0];
        b[1].type = SECBUFFER_DATA_; b[1].len = (unsigned int)k; b[1].data = (void*)((unsigned char*)&c->out[0] + hdr);
        b[2].type = SECBUFFER_STREAM_TRAILER_; b[2].len = (unsigned int)trl; b[2].data = (void*)((unsigned char*)&c->out[0] + hdr + k);
        b[3].type = SECBUFFER_EMPTY_; b[3].len = 0U; b[3].data = (void*)0;
        struct SchBufferDesc_ d;
        d.version = 0U; d.count = 4U; d.bufs = (struct SchBuffer_*)&b[0];
        int st = EncryptMessage(&c->ctx, 0U, &d, 0U);
        if (st != SEC_E_OK_) { sch_fail_(s, TLS_ERR_PROTOCOL, "EncryptMessage failed", st); return -1LL; }
        c->out_pos = 0UL;
        c->out_len = (unsigned long)b[0].len + (unsigned long)b[1].len + (unsigned long)b[2].len;
        f = sch_flush_(s, c);
        if (f < 0 && f != TLS_WANT_WRITE) { tls_fail_(&s.err, TLS_ERR_IO, "TLS: socket write failed"); return -1LL; }
        return (long long)k;
    }
    return -1LL;
}

int tls_schannel_shutdown_(&TlsStream s) {
    struct SchConn_* c = sch_of_(s);
    unsafe {
        if (c == (struct SchConn_*)0 || c->sent_close || !c->have_ctx) { return 0; }
        c->sent_close = 1;
        unsigned int token = 1U;   // SCHANNEL_SHUTDOWN
        struct SchBuffer_ b;
        b.type = SECBUFFER_TOKEN_; b.len = 4U; b.data = (void*)&token;
        struct SchBufferDesc_ d;
        d.version = 0U; d.count = 1U; d.bufs = (struct SchBuffer_*)&b;
        if (ApplyControlToken(&c->ctx, &d) != SEC_E_OK_) { return -1; }
        struct SchBuffer_ ob;
        ob.type = SECBUFFER_TOKEN_; ob.len = 0U; ob.data = (void*)0;
        struct SchBufferDesc_ od;
        od.version = 0U; od.count = 1U; od.bufs = (struct SchBuffer_*)&ob;
        unsigned int attrs = 0U;
        long long expiry = 0LL;
        InitializeSecurityContextA(&c->cred, &c->ctx, (const char*)0, ISC_FLAGS_, 0U, 0U, (struct SchBufferDesc_*)0, 0U,
                                   &c->ctx, &od, &attrs, &expiry);
        if (ob.len > 0U && ob.data != (void*)0) {
            sch_queue_(c, (const unsigned char*)ob.data, (unsigned long)ob.len);
            FreeContextBuffer(ob.data);
        }
        return sch_flush_(s, c) == 0 ? 0 : -1;
    }
}

void tls_schannel_free_(&TlsStream s) {
    struct SchConn_* c = sch_of_(s);
    if (c == (struct SchConn_*)0) { return; }
    unsafe {
        if (c->have_ctx) { DeleteSecurityContext(&c->ctx); }
        if (c->have_cred) { FreeCredentialsHandle(&c->cred); }
        c->name.free();
        tls_trust_free_(&c->extra);
        c->alpn.free();
        mem_zero_secure_raw_((void*)c, sizeof(struct SchConn_));
        dealloc_raw_((void*)c);
    }
    s.impl_ = null;
}

// Servers need the private key in a Windows key store; TLS_BACKEND_AUTO
// serves with the SafeC backend instead.
int tls_schannel_acceptor_(&TlsAcceptor a, const &TlsServerConfig cfg, &TlsError err) {
    tls_fail_(err, TLS_ERR_UNAVAILABLE, "the SChannel backend doesn't serve TLS (PEM keys); use the SafeC backend");
    return TLS_ERR_UNAVAILABLE;
}
int tls_schannel_accept_(&TlsStream s, const &TlsAcceptor a) {
    tls_fail_(&s.err, TLS_ERR_UNAVAILABLE, "the SChannel backend doesn't serve TLS");
    return TLS_ERR_UNAVAILABLE;
}
void tls_schannel_acceptor_free_(&TlsAcceptor a) { }

} // namespace std
