// SafeC Standard Library — TLS: the system's trusted roots on Unix-like
// systems, for the SafeC backend: $SSL_CERT_FILE when set, else the first
// PEM bundle found at the usual distribution paths (the same list Go's
// crypto/x509 searches). Each root is parsed once per process.
#pragma once
#include <std/tls/tls_safec.h>
#include <std/tls/tls_backend.h>

namespace std {

static int tls_roots_try_(&TlsTrust_ out, const &char path) {
    struct String pem = String::new();
    int got = 0;
    if (tls_read_file_(path, &pem)) {
        got = tls_trust_add_pem_(out, pem.as_ptr()) > 0UL;
    }
    pem.free();
    return got;
}

void tls_system_roots_(&TlsTrust_ out) {
    match (tls_getenv_("SSL_CERT_FILE")) {
        case some(f): { if (tls_roots_try_(out, f)) { return; } }
        case null: { }
    }
    if (tls_roots_try_(out, "/etc/ssl/certs/ca-certificates.crt")) { return; }          // Debian, Ubuntu, Gentoo, Arch
    if (tls_roots_try_(out, "/etc/pki/tls/certs/ca-bundle.crt")) { return; }            // Fedora, RHEL 6
    if (tls_roots_try_(out, "/etc/ssl/ca-bundle.pem")) { return; }                      // openSUSE
    if (tls_roots_try_(out, "/etc/pki/tls/cacert.pem")) { return; }                     // OpenELEC
    if (tls_roots_try_(out, "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem")) { return; } // CentOS, RHEL 7+
    if (tls_roots_try_(out, "/etc/ssl/cert.pem")) { return; }                           // Alpine, macOS, OpenBSD, FreeBSD
    if (tls_roots_try_(out, "/usr/local/etc/ssl/cert.pem")) { return; }                 // FreeBSD
    if (tls_roots_try_(out, "/usr/local/share/certs/ca-root-nss.crt")) { return; }     // FreeBSD (older)
    if (tls_roots_try_(out, "/etc/certs/ca-certificates.crt")) { return; }              // Solaris 11.2+
}

} // namespace std
