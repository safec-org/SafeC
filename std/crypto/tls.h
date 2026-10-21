// SafeC Standard Library — TLS 1.3 Record Layer
// Implements the TLS 1.3 record protocol (RFC 8446 §5).
// Crypto operations are delegated to std/crypto/aes.h and std/crypto/sha256.h.
// This module handles record framing, the inner content type and alerts.
// There is no handshake or key exchange (it needs ECDH, not in std): keys
// come from install_keys().
//
// WARNING: record protection is AES-256-CBC + PKCS#7 padding with no MAC,
// in place of TLS 1.3's AES-GCM AEAD. Records are not authenticated (an
// active attacker can alter them, and padding errors are observable), and
// the wire format does not interoperate with real TLS 1.3 peers. Use it for
// framing experiments, not to protect traffic.
// Freestanding-safe.
#pragma once
#include <std/crypto/aes.h>
#include <std/crypto/sha256.h>

// TLS content types
#define TLS_CHANGE_CIPHER_SPEC  20
#define TLS_ALERT               21
#define TLS_HANDSHAKE           22
#define TLS_APPLICATION_DATA    23

// Alert levels and descriptions
#define TLS_ALERT_WARNING        1
#define TLS_ALERT_FATAL          2
#define TLS_ALERT_CLOSE_NOTIFY   0
#define TLS_ALERT_HANDSHAKE_FAIL 40
#define TLS_ALERT_DECODE_ERROR   50

// TLS record header (5 bytes on the wire)
namespace std {

struct TlsRecord {
    unsigned char  content_type;
    unsigned char  legacy_version_major;  // 0x03
    unsigned char  legacy_version_minor;  // 0x03 for TLS 1.2 compat
    unsigned short length;                // payload length (host byte order)
};

// Maximum TLS plaintext/ciphertext fragment size (RFC 8446 §5.1)
#define TLS_MAX_RECORD_LEN  16384

// Transcript hash context (used during handshake).
struct TlsTranscript {
    struct Sha256Ctx hash;

    void  update(const &unsigned char data, unsigned long len);
    void  finish(unsigned char out[32]);
};

// Traffic key sizes (AES-256-CBC used here; real TLS 1.3 uses AES-256-GCM).
#define TLS_KEY_LEN   32   // AES-256
#define TLS_IV_LEN    12   // GCM-style IV (per RFC 8446 §5.3)

// TLS session — holds traffic keys and sequence numbers.
struct TlsSession {
    unsigned char  write_key[TLS_KEY_LEN];
    unsigned char  write_iv[TLS_IV_LEN];
    unsigned char  read_key[TLS_KEY_LEN];
    unsigned char  read_iv[TLS_IV_LEN];
    unsigned long long write_seq;
    unsigned long long read_seq;
    int            handshake_done;
    struct TlsTranscript transcript;

    // Encode a plaintext record into `out`.  Returns bytes written (0 on error).
    unsigned long encode_record(unsigned char content_type,
                                 const &unsigned char payload,
                                 unsigned long payload_len,
                                 &unsigned char out,
                                 unsigned long out_cap);

    // Decode a record from `data` (raw bytes from network).
    // Sets *content_type_out and writes decrypted payload into `out`.
    // Returns payload length, or 0 on error / insufficient data.
    unsigned long decode_record(const &unsigned char data,
                                 unsigned long data_len,
                                 &stack unsigned char content_type_out,
                                 &unsigned char out,
                                 unsigned long out_cap);

    // Build a TLS Alert record into `out`.  Returns bytes written.
    unsigned long build_alert(unsigned char level,
                               unsigned char desc,
                               &unsigned char out,
                               unsigned long out_cap);

    // XOR the IV with the sequence number per TLS 1.3 §5.3.
    // Writes TLS_IV_LEN bytes into `nonce`.
    // `is_write` != 0 selects write_iv; 0 selects read_iv.
    void  compute_nonce(unsigned char nonce[TLS_IV_LEN], int is_write) const;

    // Install traffic keys (called after key derivation in the handshake):
    // resets both sequence numbers and switches to protected records
    // (is_established() becomes 1).
    void  install_keys(const unsigned char write_key[TLS_KEY_LEN],
                       const unsigned char write_iv[TLS_IV_LEN],
                       const unsigned char read_key[TLS_KEY_LEN],
                       const unsigned char read_iv[TLS_IV_LEN]);

    // Is the session post-handshake (application data mode)?
    int   is_established() const;
};

// Initialise a TLS session (zero keys, seq=0, handshake_done=0).
struct TlsSession tls_session_init();

} // namespace std
