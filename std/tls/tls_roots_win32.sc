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
    void*          info;
    void*          store;
};

extern void* CertOpenSystemStoreA(void* prov, const char* name);
extern void* CertEnumCertificatesInStore(void* store, void* prev);
extern int   CertCloseStore(void* store, unsigned int flags);

void tls_system_roots_(&TlsTrust_ out) {
    unsafe {
        void* store = CertOpenSystemStoreA((void*)0, "ROOT");
        if (store == (void*)0) { return; }
        void* ctx = CertEnumCertificatesInStore(store, (void*)0);
        while (ctx != (void*)0) {
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
