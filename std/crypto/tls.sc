// SafeC Standard Library — TLS 1.3 Record Layer (implementation)
#pragma once
#include <std/stddef.h>
#include <std/crypto/tls.h>

namespace std {

extern void* memcpy(void* d, const void* s, size_t n);
extern void* memset(void* p, int v, size_t n);
extern void* memmove(void* d, const void* s, size_t n);

// ── Internal helpers ───────────────────────────────────────────────────────────

// Write a 16-bit value in network (big-endian) byte order.
static void put_u16_be_(unsigned char* p, unsigned short v) {
    unsafe {
        p[0] = (unsigned char)((v >> 8) & 0xFF);
        p[1] = (unsigned char)( v       & 0xFF);
    }
}

// Read a 16-bit big-endian value.
static unsigned short get_u16_be_(const unsigned char* p) {
    unsafe {
        return (unsigned short)(((unsigned short)p[0] << 8) | (unsigned short)p[1]);
    }
}

// Write a 64-bit value in network (big-endian) byte order into 8 bytes.
static void put_u64_be_(unsigned char* p, unsigned long long v) {
    unsafe {
        int k = 7;
        while (k >= 0) {
            p[k] = (unsigned char)(v & (unsigned long long)0xFF);
            v = v >> (unsigned long long)8;
            k = k - 1;
        }
    }
}

// ── TlsTranscript ─────────────────────────────────────────────────────────────

inline void TlsTranscript::update(const &unsigned char data, unsigned long len) {
    unsafe { self.hash.update((const &unsigned char)data, len); }
}

inline void TlsTranscript::finish(unsigned char out[32]) {
    // Finish on a copy so the transcript can continue to be used.
    struct Sha256Ctx snap;
    unsafe { memcpy((void*)&snap, (const void*)&self.hash, (unsigned long)sizeof(struct Sha256Ctx)); }
    snap.finish(out);
}

// ── tls_session_init ──────────────────────────────────────────────────────────

inline struct TlsSession tls_session_init() {
    struct TlsSession s;
    unsafe { memset((void*)&s, 0, (unsigned long)sizeof(struct TlsSession)); }
    s.transcript.hash = sha256_init();
    return s;
}

// ── TlsSession methods ────────────────────────────────────────────────────────

inline int TlsSession::is_established() const {
    return self.handshake_done;
}

void TlsSession::install_keys(const unsigned char write_key[TLS_KEY_LEN],
                               const unsigned char write_iv[TLS_IV_LEN],
                               const unsigned char read_key[TLS_KEY_LEN],
                               const unsigned char read_iv[TLS_IV_LEN]) {
    unsafe {
        memcpy((void*)self.write_key, (const void*)write_key, (unsigned long)TLS_KEY_LEN);
        memcpy((void*)self.write_iv,  (const void*)write_iv,  (unsigned long)TLS_IV_LEN);
        memcpy((void*)self.read_key,  (const void*)read_key,  (unsigned long)TLS_KEY_LEN);
        memcpy((void*)self.read_iv,   (const void*)read_iv,   (unsigned long)TLS_IV_LEN);
    }
    self.write_seq = (unsigned long long)0;
    self.read_seq  = (unsigned long long)0;
    self.handshake_done = 1;   // records are protected from here on
}

inline void TlsSession::compute_nonce(unsigned char nonce[TLS_IV_LEN], int is_write) const {
    // Copy base IV into nonce, then XOR bytes [4..11] with the 8-byte big-endian
    // sequence number per RFC 8446 §5.3.
    unsigned char seq_bytes[8];
    unsafe {
        unsigned long long seq = (is_write != 0) ? self.write_seq : self.read_seq;
        put_u64_be_((unsigned char*)seq_bytes, seq);

        int k = 0;
        while (k < TLS_IV_LEN) {
            nonce[k] = (is_write != 0) ? self.write_iv[k] : self.read_iv[k];
            k = k + 1;
        }
        // XOR the last 8 bytes of the 12-byte IV with the sequence number
        k = 0;
        while (k < 8) {
            nonce[4 + k] = nonce[4 + k] ^ seq_bytes[k];
            k = k + 1;
        }
    }
}

// ── encode_record ──────────────────────────────────────────────────────────────
//
// Before keys are installed: a 5-byte header + the plaintext payload.
// After install_keys (RFC 8446 §5.2): TLSInnerPlaintext = payload ||
// content_type, sealed with AES-256-GCM under the per-record nonce; the
// outer record is TLS_APPLICATION_DATA, 0x03 0x03, length, and the 5-byte
// header is the AEAD's additional data. No zero padding is added.

static void tls_header_(unsigned char* out, unsigned char type, unsigned long len) {
    unsafe {
        out[0] = type;
        out[1] = (unsigned char)0x03;
        out[2] = (unsigned char)0x03;
        put_u16_be_(out + 3, (unsigned short)len);
    }
}

unsigned long TlsSession::encode_record(unsigned char content_type,
                                         const &unsigned char payload_ref,
                                         unsigned long payload_len,
                                         &unsigned char out_ref,
                                         unsigned long out_cap) {
    const unsigned char* payload;
    unsigned char* out;
    unsafe { payload = (const unsigned char*)payload_ref; out = (unsigned char*)out_ref; }
    if (payload_len > (unsigned long)TLS_MAX_RECORD_LEN) { return (unsigned long)0; }

    if (self.handshake_done == 0) {
        unsigned long total = (unsigned long)5 + payload_len;
        if (total > out_cap) { return (unsigned long)0; }
        unsafe {
            tls_header_(out, content_type, payload_len);
            memcpy((void*)(out + 5), (const void*)payload, payload_len);
        }
        return total;
    }

    // Check room before consuming a sequence number: a record that isn't
    // emitted must not advance write_seq (the peer's read_seq wouldn't).
    unsigned long inner_len = payload_len + 1UL;
    unsigned long rec_len = inner_len + (unsigned long)TLS_TAG_LEN;
    unsigned long total = (unsigned long)5 + rec_len;
    if (total > out_cap) { return (unsigned long)0; }

    unsigned char nonce[TLS_IV_LEN];
    self.compute_nonce((unsigned char*)nonce, 1);
    struct AesCtx aes = aes256_init(self.write_key);
    unsafe {
        tls_header_(out, (unsigned char)TLS_APPLICATION_DATA, rec_len);
        memmove((void*)(out + 5), (const void*)payload, payload_len);
        out[5UL + payload_len] = content_type;                  // TLSInnerPlaintext.type
        aes.gcm_encrypt((const &unsigned char)&nonce[0], (const ?&unsigned char)out, 5UL,
                        (&unsigned char)(out + 5), inner_len, (&unsigned char)(out + 5UL + inner_len));
    }
    aes.destroy();
    self.write_seq = self.write_seq + (unsigned long long)1;
    return total;
}

// ── decode_record ──────────────────────────────────────────────────────────────

unsigned long TlsSession::decode_record(const &unsigned char data_ref,
                                         unsigned long data_len,
                                         &unsigned char content_type_out,
                                         &unsigned char out_ref,
                                         unsigned long out_cap) {
    const unsigned char* data;
    unsigned char* out;
    unsafe { data = (const unsigned char*)data_ref; out = (unsigned char*)out_ref; }
    if (data_len < (unsigned long)5) { return (unsigned long)0; }

    unsigned char wire_ct;
    unsigned long flen;
    unsafe {
        wire_ct = data[0];
        if (data[1] != (unsigned char)0x03 || data[2] != (unsigned char)0x03) { return (unsigned long)0; }
        flen = (unsigned long)get_u16_be_(data + 3);
    }
    if (data_len < (unsigned long)5 + flen) { return (unsigned long)0; }   // incomplete

    if (self.handshake_done == 0) {
        if (flen > (unsigned long)TLS_MAX_RECORD_LEN || flen > out_cap) { return (unsigned long)0; }
        unsafe {
            memcpy((void*)out, (const void*)(data + 5), flen);
            *content_type_out = wire_ct;
        }
        return flen;
    }

    // Protected: the outer type is always application_data, and the body is
    // TLSInnerPlaintext (at least the type byte) + the 16-byte tag.
    if (wire_ct != (unsigned char)TLS_APPLICATION_DATA) { return (unsigned long)0; }
    if (flen < (unsigned long)TLS_TAG_LEN + 1UL ||
        flen > (unsigned long)TLS_MAX_RECORD_LEN + 256UL) { return (unsigned long)0; }
    unsigned long inner_len = flen - (unsigned long)TLS_TAG_LEN;

    unsigned char plain[16640];   // TLS_MAX_RECORD_LEN + 256
    unsigned char nonce[TLS_IV_LEN];
    self.compute_nonce((unsigned char*)nonce, 0);
    struct AesCtx aes = aes256_init(self.read_key);
    int ok = 0;
    unsafe {
        memcpy((void*)plain, (const void*)(data + 5), inner_len);
        ok = aes.gcm_decrypt((const &unsigned char)&nonce[0], (const ?&unsigned char)data, 5UL,
                             (&unsigned char)&plain[0], inner_len,
                             (const &unsigned char)(data + 5UL + inner_len));
    }
    aes.destroy();
    if (!ok) { return (unsigned long)0; }                    // forged or corrupted

    // The content type is the last non-zero byte; zeros after it are padding.
    unsigned long end = inner_len;
    unsafe { while (end > 0UL && plain[end - 1UL] == (unsigned char)0) { end = end - 1UL; } }
    if (end == 0UL) { return (unsigned long)0; }              // no content type
    unsigned long payload_len = end - 1UL;
    if (payload_len > out_cap || payload_len > (unsigned long)TLS_MAX_RECORD_LEN) { return (unsigned long)0; }
    unsafe {
        *content_type_out = plain[payload_len];
        memcpy((void*)out, (const void*)plain, payload_len);
    }
    self.read_seq = self.read_seq + (unsigned long long)1;
    return payload_len;
}

// ── build_alert ───────────────────────────────────────────────────────────────

unsigned long TlsSession::build_alert(unsigned char level,
                                       unsigned char desc,
                                       &unsigned char out_ref,
                                       unsigned long out_cap) {
    // Alert body: level (1 byte) + description (1 byte)
    unsigned char alert_body[2];
    unsafe {
        alert_body[0] = level;
        alert_body[1] = desc;
    }
    return self.encode_record((unsigned char)TLS_ALERT,
                               alert_body,
                               (unsigned long)2,
                               out_ref, out_cap);
}

} // namespace std
