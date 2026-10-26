// SafeC Standard Library — TLS: the system's trusted roots on Windows, for
// the SafeC backend: the current user's "ROOT" certificate store (which
// includes the machine's) through crypt32.
// safeguard: link windows -lcrypt32
#pragma once
#include <std/tls/tls_safec.h>
#include <std/tls/tls_backend.h>

namespace std {

// CERT_CONTEXT: { DWORD dwCertEncodingType; BYTE* pbCertEncoded;
// DWORD cbCertEncoded; PCERT_INFO pCertInfo; HCERTSTORE hCertStore; }
struct TlsWinCertCtx_ {
    unsigned int   encoding;
    unsigned char* data;
    unsigned int   len;
    unsigned char*          info;
    unsigned char*          store;
};

extern unsigned char* CertOpenSystemStoreA(unsigned char* prov, const char* name);
extern unsigned char* CertEnumCertificatesInStore(unsigned char* store, unsigned char* prev);
extern int   CertCloseStore(unsigned char* store, unsigned int flags);

void tls_system_roots_(&TlsTrust_ out) {
    unsafe {
        unsigned char* store = CertOpenSystemStoreA((unsigned char*)0, "ROOT");
        if (store == (unsigned char*)0) { return; }
        unsigned char* ctx = CertEnumCertificatesInStore(store, (unsigned char*)0);
        while (ctx != (unsigned char*)0) {
            struct TlsWinCertCtx_* c = (struct TlsWinCertCtx_*)ctx;
            if (c->data != (unsigned char*)0 && c->len > 0U) {
                tls_trust_add_der_(out, (const &unsigned char)c->data, (unsigned long)c->len);
            }
            ctx = CertEnumCertificatesInStore(store, ctx);
        }
        CertCloseStore(store, 0U);
    }
}

} // namespace std
