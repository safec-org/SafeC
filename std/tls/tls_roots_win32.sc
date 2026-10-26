// SafeC Standard Library — TLS: the system's trusted roots on Windows, for
// the SafeC backend: the current user's "ROOT" certificate store (which
// includes the machine's) through crypt32.
// safeguard: link windows -lcrypt32
#pragma once
#include <std/tls/tls_safec.h>
#include <std/tls/tls_backend.h>

namespace std {

// HCERTSTORE, CERT_INFO and the crypto provider (always null), opaque.
struct TlsWinStore_ { char opaque_; };
struct TlsWinCertInfo_ { char opaque_; };
struct TlsWinProv_ { char opaque_; };

// CERT_CONTEXT: { DWORD dwCertEncodingType; BYTE* pbCertEncoded;
// DWORD cbCertEncoded; PCERT_INFO pCertInfo; HCERTSTORE hCertStore; }
struct TlsWinCertCtx_ {
    unsigned int     encoding;
    unsigned char*   data;
    unsigned int     len;
    TlsWinCertInfo_* info;
    TlsWinStore_*    store;
};

extern TlsWinStore_* CertOpenSystemStoreA(TlsWinProv_* prov, const char* name);
extern TlsWinCertCtx_* CertEnumCertificatesInStore(TlsWinStore_* store, TlsWinCertCtx_* prev);
extern int   CertCloseStore(TlsWinStore_* store, unsigned int flags);

void tls_system_roots_(&TlsTrust_ out) {
    unsafe {
        TlsWinStore_* store = CertOpenSystemStoreA((TlsWinProv_*)0, "ROOT");
        if (store == (TlsWinStore_*)0) { return; }
        TlsWinCertCtx_* c = CertEnumCertificatesInStore(store, (TlsWinCertCtx_*)0);
        while (c != (TlsWinCertCtx_*)0) {
            if (c->data != (unsigned char*)0 && c->len > 0U) {
                tls_trust_add_der_(out, (const &unsigned char)c->data, (unsigned long)c->len);
            }
            c = CertEnumCertificatesInStore(store, c);
        }
        CertCloseStore(store, 0U);
    }
}

} // namespace std
