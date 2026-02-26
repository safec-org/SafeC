// SafeC Standard Library — TLS 1.3 Record Layer (implementation)
#pragma once
#include "tls.h"

extern void* memcpy(void* d, const void* s, unsigned long n);
extern void* memset(void* p, int v, unsigned long n);

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

// Pad `data` (len bytes) in-place with PKCS#7 to a multiple of block_size.
// Returns padded length.  Caller must ensure data buffer has room.
static unsigned long pkcs7_pad_(unsigned char* data, unsigned long len,
                                 unsigned long block_size) {
    unsigned long pad = block_size - (len % block_size);
    unsigned long i = (unsigned long)0;
    unsafe {
        while (i < pad) {
            data[len + i] = (unsigned char)pad;
            i = i + (unsigned long)1;
        }
    }
    return len + pad;
}

// Verify and strip PKCS#7 padding.  Returns unpadded length or 0 on error.
static unsigned long pkcs7_unpad_(const unsigned char* data, unsigned long len,
                                   unsigned long block_size) {
    if (len == (unsigned long)0 || len % block_size != (unsigned long)0) {
        return (unsigned long)0;
    }
    unsafe {
        unsigned long pad = (unsigned long)data[len - (unsigned long)1];
        if (pad == (unsigned long)0 || pad > block_size || pad > len) {
            return (unsigned long)0;
        }
        // Verify all padding bytes
        unsigned long i = (unsigned long)0;
        while (i < pad) {
            if (data[len - (unsigned long)1 - i] != (unsigned char)pad) {
                return (unsigned long)0;
            }
            i = i + (unsigned long)1;
        }
        return len - pad;
    }
}

// ── TlsTranscript ─────────────────────────────────────────────────────────────

void TlsTranscript::update(const unsigned char* data, unsigned long len) {
    self.hash.update(data, len);
}

void TlsTranscript::finish(unsigned char out[32]) {
    // Finish on a copy so the transcript can continue to be used.
    struct Sha256Ctx snap;
    unsafe { memcpy((void*)&snap, (const void*)&self.hash, (unsigned long)sizeof(struct Sha256Ctx)); }
    snap.finish((unsigned char*)out);
}

// ── tls_session_init ──────────────────────────────────────────────────────────

struct TlsSession tls_session_init() {
    struct TlsSession s;
    unsafe { memset((void*)&s, 0, (unsigned long)sizeof(struct TlsSession)); }
    s.transcript.hash = sha256_init();
    return s;
}

// ── TlsSession methods ────────────────────────────────────────────────────────

int TlsSession::is_established() const {
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
}

void TlsSession::compute_nonce(unsigned char nonce[TLS_IV_LEN], int is_write) const {
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
// Pre-handshake: 5-byte header + plaintext payload.
// Post-handshake: 5-byte header + AES-256-CBC(payload || content_type) +
//                 record content_type = TLS_APPLICATION_DATA (23).
//
// The inner content type byte appended before encryption is the real content
// type per RFC 8446 §5.2 (TLSInnerPlaintext).
// CBC block size = 16; we use the nonce bytes 0-15 as IV (truncated from 12
// with 4 zero-prefix bytes to fill 16 bytes).

unsigned long TlsSession::encode_record(unsigned char content_type,
                                         const unsigned char* payload,
                                         unsigned long payload_len,
                                         unsigned char* out,
                                         unsigned long out_cap) {
    // Sanity checks
    if (payload_len > (unsigned long)TLS_MAX_RECORD_LEN) { return (unsigned long)0; }

    if (self.handshake_done == 0) {
        // Unencrypted record
        unsigned long total = (unsigned long)5 + payload_len;
        if (total > out_cap) { return (unsigned long)0; }
        unsafe {
            out[0] = content_type;
            out[1] = (unsigned char)0x03;
            out[2] = (unsigned char)0x03;
            put_u16_be_((unsigned char*)(out + 3), (unsigned short)payload_len);
            memcpy((void*)(out + 5), (const void*)payload, payload_len);
        }
        return total;
    }

    // Encrypted record (TLS 1.3 §5.2):
    // 1. Build TLSInnerPlaintext = payload || content_type
    // 2. PKCS#7-pad to AES block boundary
    // 3. CBC-encrypt using write_key; IV derived from compute_nonce (16-byte view)
    // 4. Outer header: TLS_APPLICATION_DATA, 0x03 0x03, ciphertext_len

    // Scratch buffer: payload + 1 (inner content type) + up to 16 pad bytes
    // Max inner plaintext = TLS_MAX_RECORD_LEN + 1 + 16
    unsigned char inner[16401];  // TLS_MAX_RECORD_LEN + 1 + 16
    if (payload_len + (unsigned long)17 > (unsigned long)16401) { return (unsigned long)0; }

    unsafe {
        memcpy((void*)inner, (const void*)payload, payload_len);
        inner[payload_len] = content_type;  // TLSInnerPlaintext.type
    }
    unsigned long inner_len = payload_len + (unsigned long)1;
    unsigned long padded_len = pkcs7_pad_((unsigned char*)inner, inner_len,
                                           (unsigned long)16);

    // Derive 16-byte CBC IV from the nonce: use first 16 bytes of
    // [0x00 0x00 0x00 0x00 || nonce[0..11]] — i.e. the nonce zero-extended.
    unsigned char nonce[TLS_IV_LEN];
    self.compute_nonce((unsigned char*)nonce, 1);

    unsigned char iv16[16];
    unsafe {
        memset((void*)iv16, 0, (unsigned long)16);
        // Copy nonce bytes into positions 4-15 of the 16-byte IV
        int k = 0;
        while (k < TLS_IV_LEN) {
            iv16[4 + k] = nonce[k];
            k = k + 1;
        }
    }

    // Encrypt in-place
    struct AesCtx aes = aes256_init((const unsigned char*)self.write_key);
    aes.set_iv((const unsigned char*)iv16);
    aes.cbc_encrypt((unsigned char*)inner, padded_len);

    self.write_seq = self.write_seq + (unsigned long long)1;

    unsigned long total = (unsigned long)5 + padded_len;
    if (total > out_cap) { return (unsigned long)0; }
    unsafe {
        out[0] = (unsigned char)TLS_APPLICATION_DATA;
        out[1] = (unsigned char)0x03;
        out[2] = (unsigned char)0x03;
        put_u16_be_((unsigned char*)(out + 3), (unsigned short)padded_len);
        memcpy((void*)(out + 5), (const void*)inner, padded_len);
    }
    return total;
}

// ── decode_record ──────────────────────────────────────────────────────────────

unsigned long TlsSession::decode_record(const unsigned char* data,
                                         unsigned long data_len,
                                         &stack unsigned char content_type_out,
                                         unsigned char* out,
                                         unsigned long out_cap) {
    // Need at least a 5-byte header
    if (data_len < (unsigned long)5) { return (unsigned long)0; }

    unsigned char wire_ct;
    unsigned char ver_major;
    unsigned char ver_minor;
    unsigned short fragment_len;
    unsafe {
        wire_ct      = data[0];
        ver_major    = data[1];
        ver_minor    = data[2];
        fragment_len = get_u16_be_((const unsigned char*)(data + 3));
    }

    // Validate version bytes (must be 0x03 0x03 for TLS 1.2 compat framing)
    if (ver_major != (unsigned char)0x03 || ver_minor != (unsigned char)0x03) {
        return (unsigned long)0;
    }

    unsigned long flen = (unsigned long)fragment_len;
    if ((unsigned long)5 + flen > data_len) { return (unsigned long)0; }
    if (flen > (unsigned long)TLS_MAX_RECORD_LEN + (unsigned long)256) {
        return (unsigned long)0;
    }

    if (self.handshake_done == 0) {
        // Unencrypted
        if (flen > out_cap) { return (unsigned long)0; }
        unsafe {
            memcpy((void*)out, (const void*)(data + 5), flen);
            content_type_out = wire_ct;
        }
        return flen;
    }

    // Encrypted: CBC-decrypt the fragment, strip PKCS#7 padding, read inner type
    if (flen == (unsigned long)0 || flen % (unsigned long)16 != (unsigned long)0) {
        return (unsigned long)0;
    }

    // Scratch decrypt buffer
    unsigned char plain[16640]; // TLS_MAX_RECORD_LEN + 256
    if (flen > (unsigned long)16640) { return (unsigned long)0; }

    unsafe {
        memcpy((void*)plain, (const void*)(data + 5), flen);
    }

    // Reconstruct IV from read_seq
    unsigned char nonce[TLS_IV_LEN];
    self.compute_nonce((unsigned char*)nonce, 0);

    unsigned char iv16[16];
    unsafe {
        memset((void*)iv16, 0, (unsigned long)16);
        int k = 0;
        while (k < TLS_IV_LEN) {
            iv16[4 + k] = nonce[k];
            k = k + 1;
        }
    }

    struct AesCtx aes = aes256_init((const unsigned char*)self.read_key);
    aes.set_iv((const unsigned char*)iv16);
    aes.cbc_decrypt((unsigned char*)plain, flen);

    unsigned long unpadded = pkcs7_unpad_((const unsigned char*)plain, flen,
                                           (unsigned long)16);
    if (unpadded == (unsigned long)0) { return (unsigned long)0; }

    self.read_seq = self.read_seq + (unsigned long long)1;

    // Last byte of TLSInnerPlaintext is the real content type; strip trailing zeros
    // per RFC 8446 §5.4 (content type byte is the rightmost non-zero byte).
    unsigned long payload_end = unpadded - (unsigned long)1;
    unsafe {
        content_type_out = plain[payload_end];
    }

    if (payload_end > out_cap) { return (unsigned long)0; }
    unsafe {
        memcpy((void*)out, (const void*)plain, payload_end);
    }
    return payload_end;
}

// ── build_alert ───────────────────────────────────────────────────────────────

unsigned long TlsSession::build_alert(unsigned char level,
                                       unsigned char desc,
                                       unsigned char* out,
                                       unsigned long out_cap) {
    // Alert body: level (1 byte) + description (1 byte)
    unsigned char alert_body[2];
    unsafe {
        alert_body[0] = level;
        alert_body[1] = desc;
    }
    return self.encode_record((unsigned char)TLS_ALERT,
                               (const unsigned char*)alert_body,
                               (unsigned long)2,
                               out, out_cap);
}
